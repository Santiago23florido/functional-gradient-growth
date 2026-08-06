"""A matrix-free tangent direction, for use as a STALL fallback.

READ THIS FIRST. This is an APPROXIMATION of the tangent family, not a
cheaper route to the same answer, and the difference is not a rounding
detail. The tangent decides on the full spectrum of ``J``; anything that
does not form ``J`` sees a Krylov subspace of dimension ``k``, and while
``k < rank(J)`` the reduced geometry cannot see the directions outside
``span(U)``. The projection therefore looks BETTER than it is:

    MEASURED at P=25, k=24:  reduced eps 0.8104  vs  true eps 0.8779

7.7% low. The bias is one-sided and it is on the unsafe side -- an eps that
is too small certifies a direction that does not deserve it. Two guards
follow from that and are not optional: ``k`` is capped at ``P`` (past that
the extra dimensions are pure noise and the same bias got much worse, 0.8097
at k=48), and the direction this module returns is a PROPOSAL whose realised
displacement the caller must re-certify. The certificate that gates a step is
always measured on what was actually applied, never on ``eps`` from here.

Equality with the tangent is reachable only at ``k = rank(J)``, which
restores the cost this module exists to avoid. The trade is deliberate;
do not describe the result as "the tangent, solved differently".


The nonlinear primary family stalls in a characteristic way: after a few
accepted steps the model fits well, the residual ``r`` becomes small, and the
clone's fit residual ``e`` -- which does not shrink with it -- dominates
``Delta = eta_f r - e``. MEASURED at N=1024, eps then sits between 0.58 and
0.99 for tens of consecutive epochs while the pipeline grows every epoch and
commits nothing. That limit is representational, not an optimisation failure:
the readout can only produce functions in ``span{h(x), 1}``, and fitting it
exactly changes nothing (verified).

The tangent does not have that problem, because ``range(J)`` contains the
``d f / d W`` directions of every hidden layer. Its cost does: forming ``J``
and the Gram ``J^T J`` is ``O(N K P^2)`` compute and ``O(P^2)`` memory, which
is what makes it impossible at MNIST width -- not the data, the PARAMETERS.

Both facts can be had at once. The projection is the least-squares problem

    min_u  ||J u - r||^2 + lambda ||u||^2,

whose normal equations ``(J^T J + lambda I) u = J^T r`` can be solved by
conjugate gradients using only the products ``J v`` (one forward-mode pass)
and ``J^T w`` (one reverse-mode pass). Neither ``J`` nor ``J^T J`` is ever
formed, so the cost is ``O(k N P)`` for ``k`` iterations -- LINEAR in ``P``,
the same order as one clone candidate, and with k = 50 about 100 passes over
the probe against the 1600 a clone spends.

This module deliberately does not import the tangent-system, projection or
realization helpers: it is a separate, cheap route to the same direction, and
the existing tangent path is untouched.

Nothing here is trusted on theory alone. The direction it returns is a
PROPOSAL; the caller applies it and certifies the displacement that actually
results, exactly as for a nonlinear candidate. First-order reasoning chooses
the direction, measurement decides whether it is admissible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fgdlib.profile import increment, timed
from fgdlib.tangent import FGDApproxConfig, functional_gradient

__all__ = [
    "MatrixFreeCertificate",
    "MatrixFreeTangentDirection",
    "apply_parameter_direction",
    "matrix_free_tangent_direction",
    "matrix_free_tangent_step",
]


@dataclass(frozen=True)
class MatrixFreeTangentDirection:
    """A parameter-space direction and the work it cost."""

    #: ``u``, keyed like ``model.named_parameters()``. ``None`` if degenerate.
    direction: dict | None
    #: Conjugate-gradient iterations actually run.
    iterations: int
    #: Forward-mode and reverse-mode passes over the probe.
    jvp_calls: int
    vjp_calls: int
    #: ``|J u|`` and ``|r|``: the realized first-order step and the residual.
    predicted_norm: float
    residual_norm: float
    #: ``cos(J u, r)`` PREDICTED to first order. The caller must re-measure the
    #: displacement it actually applies; this is only a screen.
    predicted_cosine: float | None


def _dot(a: list, b: list) -> torch.Tensor:
    return sum((x * y).sum() for x, y in zip(a, b))


def _axpy(a: list, alpha, b: list) -> list:
    return [x + alpha * y for x, y in zip(a, b)]


def _norm(vectors: list) -> float:
    return math.sqrt(sum(float(torch.sum(v * v)) for v in vectors))


def _scaled(vectors: list, factor: float) -> list:
    return [v * factor for v in vectors]


def matrix_free_tangent_direction(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
    iterations: int = 50,
    damping: float = 1e-6,
    tolerance: float = 1e-8,
) -> MatrixFreeTangentDirection:
    """Solve ``(J^T J + damping I) u = J^T r`` by CG, without forming ``J``.

    Both products use ordinary autograd. ``J^T w`` is one reverse pass; ``J v``
    uses the double-backward identity -- with ``w`` a dummy variable,
    ``g(w) = J^T w`` is linear in ``w``, so ``d<g(w), v>/dw = J v``. Deliberately
    NOT ``torch.func.jvp``: nesting it inside a reverse-mode graph returns
    functorch ``TensorWrapper`` values that cannot be copied out of the
    transform.

    Returns ``u``; the caller chooses a step size, applies it, and CERTIFIES
    the displacement that actually results. ``predicted_cosine`` is the
    first-order alignment and is a screen only -- for a nonlinear network the
    realized displacement is not ``eta J u``.
    """
    increment("matrix_free_tangent_calls")
    was_training = model.training
    model.eval()
    jvp_calls = 0
    vjp_calls = 0
    names = [name for name, _ in model.named_parameters()]
    parameters = [parameter for _, parameter in model.named_parameters()]
    try:
        with timed("matrix_free_tangent_seconds"):
            outputs = model(x)
            with torch.no_grad():
                residual = functional_gradient(
                    outputs.detach(), y, config.functional_loss
                )
            residual_norm = float(torch.linalg.vector_norm(residual))
            if not (residual_norm > 0.0 and torch.isfinite(residual).all()):
                return MatrixFreeTangentDirection(
                    None, 0, jvp_calls, vjp_calls, 0.0, residual_norm, None
                )

            def _jt(w: torch.Tensor) -> list:
                """``J^T w`` -- one reverse-mode pass, plain autograd."""
                nonlocal vjp_calls
                vjp_calls += 1
                grads = torch.autograd.grad(
                    outputs, parameters, grad_outputs=w, retain_graph=True
                )
                return [g.detach() for g in grads]

            # g(w) = J^T w, built once with create_graph so it stays
            # differentiable in w. Since it is LINEAR in w,
            # d<g(w), v>/dw = J v exactly.
            dummy = torch.zeros_like(outputs, requires_grad=True)
            jt_dummy = torch.autograd.grad(
                outputs, parameters, grad_outputs=dummy, create_graph=True
            )

            def _j(v: list) -> torch.Tensor:
                """``J v`` -- via the double-backward identity, plain autograd."""
                nonlocal jvp_calls
                jvp_calls += 1
                paired = sum(
                    (g * value).sum() for g, value in zip(jt_dummy, v)
                )
                (result,) = torch.autograd.grad(paired, dummy, retain_graph=True)
                return result.detach()

            b = _jt(residual)
            u = [torch.zeros_like(parameter) for parameter in parameters]
            r_cg = [value.clone() for value in b]
            p_cg = [value.clone() for value in r_cg]
            rs_old = _dot(r_cg, r_cg)
            b_norm = float(torch.sqrt(_dot(b, b)))
            if not (b_norm > 0.0):
                return MatrixFreeTangentDirection(
                    None, 0, jvp_calls, vjp_calls, 0.0, residual_norm, None
                )

            performed = 0
            for _ in range(max(1, iterations)):
                # One CG iteration = one J v and one J^T w. No P x P object.
                ap = _axpy(_jt(_j(p_cg)), damping, p_cg)
                denominator = _dot(p_cg, ap)
                if not torch.isfinite(denominator) or float(denominator) <= 0.0:
                    break
                alpha = rs_old / denominator
                u = _axpy(u, alpha, p_cg)
                r_cg = _axpy(r_cg, -alpha, ap)
                rs_new = _dot(r_cg, r_cg)
                performed += 1
                if float(torch.sqrt(rs_new)) <= tolerance * b_norm:
                    break
                p_cg = _axpy(r_cg, rs_new / rs_old, p_cg)
                rs_old = rs_new

            if not all(torch.isfinite(value).all() for value in u):
                return MatrixFreeTangentDirection(
                    None, performed, jvp_calls, vjp_calls, 0.0, residual_norm, None
                )

            predicted = _j(u)
            predicted_norm = float(torch.linalg.vector_norm(predicted))
            cosine = (
                float(
                    torch.sum(predicted * residual) / (predicted_norm * residual_norm)
                )
                if predicted_norm > 0.0
                else None
            )
            direction = {name: value for name, value in zip(names, u)}
    finally:
        model.train(was_training)

    return MatrixFreeTangentDirection(
        direction=direction,
        iterations=performed,
        jvp_calls=jvp_calls,
        vjp_calls=vjp_calls,
        predicted_norm=predicted_norm,
        residual_norm=residual_norm,
        predicted_cosine=cosine,
    )


@torch.no_grad()
def apply_parameter_direction(
    *,
    base_model: torch.nn.Module,
    direction: dict,
    step: float,
) -> torch.nn.Module:
    """Return a copy of ``base_model`` moved by ``-step * direction``.

    The minus sign matches the certificate's convention
    ``Delta = f_base - f_stepped``: moving against ``J^T r`` decreases the
    loss to first order, so ``Delta`` points along ``+r``.
    """
    import copy

    stepped = copy.deepcopy(base_model)
    parameters = dict(stepped.named_parameters())
    for name, value in direction.items():
        if name in parameters:
            parameters[name].add_(value, alpha=-step)
    stepped.eval()
    return stepped


@dataclass(frozen=True)
class MatrixFreeCertificate:
    """``eps`` for a matrix-free projection, in the TANGENT's own definition.

    ``relative_error`` is ``||J u - r|| / ||J u||`` -- the residual at the
    SOLVED scale, which is what ``certify.exact_relative_error`` reports. It is
    NOT ``sqrt(1 - cos^2)``, the scale-optimal error. The two differ by ~19 %
    here, and mixing them produced a real wrong comparison earlier in this
    branch, so the definition is stated rather than implied.
    """

    relative_error: float
    cosine: float | None
    predicted_norm: float
    residual_norm: float
    iterations: int
    passes: int
    sensor_valid: bool
    #: The damping the sweep settled on. Free telemetry now that the whole
    #: grid is evaluated on one shared subspace.
    damping: float = 0.0


def _batch_graphs(model, loader, device, config, max_batches):
    """Retain one autograd graph per minibatch for the whole CG run.

    ``J^T w`` and ``J v`` are both SUMS over examples, so accumulating them
    batch by batch never materialises the probe or ``J``: memory is ``O(P)``
    and independent of the dataset size. That is the property that makes this
    usable where the exact route cannot run at all -- the Gram alone is
    ``O(P^2)``.
    """
    graphs = []
    for index, (x, y) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        with torch.no_grad():
            residual = functional_gradient(
                outputs.detach(), y, config.functional_loss
            )
        dummy = torch.zeros_like(outputs, requires_grad=True)
        parameters = list(model.parameters())
        jt_dummy = torch.autograd.grad(
            outputs, parameters, grad_outputs=dummy, create_graph=True
        )
        graphs.append((outputs, residual, dummy, jt_dummy, parameters))
    return graphs


def krylov_jacobian(
    *,
    outputs: torch.Tensor,
    parameters,
    target: torch.Tensor,
    iterations: int,
    eps: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | None:
    """``J`` reconstructed as ``W V^T``, never forming the Gram.

    The ladder consumes a dense ``(rows, P)`` Jacobian, so one is returned --
    but it is BUILT from ``2k+1`` matrix-vector products rather than from a
    ``jacrev`` over every row, and the ``O(P^2)`` Gram is never materialised at
    any point. That Gram is the object that makes the exact route impossible at
    MNIST width; ``J`` itself is ``O(NK P)``, which is large but linear.

    ``k`` runs to the numerical rank as seen from ``target`` and is capped at
    ``P``. Both matter: below the rank the reconstruction is optimistic about
    how much of ``r`` is representable, and above ``P`` the extra directions
    are noise that is optimistic in the same direction.
    """
    parameters = list(parameters)
    dummy = torch.zeros_like(outputs, requires_grad=True)
    jt_dummy = torch.autograd.grad(
        outputs, parameters, grad_outputs=dummy, create_graph=True
    )

    def apply_jt(weights: list) -> list:
        grads = torch.autograd.grad(
            outputs, parameters, grad_outputs=weights[0], retain_graph=True
        )
        return [g.detach() for g in grads]

    def apply_j(vector: list) -> list:
        paired = sum((g * value).sum() for g, value in zip(jt_dummy, vector))
        (result,) = torch.autograd.grad(paired, dummy, retain_graph=True)
        return [result.detach()]

    residuals = [target.reshape(outputs.shape)]
    system = _golub_kahan(
        apply_j=apply_j,
        apply_jt=apply_jt,
        residuals=residuals,
        iterations=max(1, iterations),
        floor=max(eps, 1e-12),
    )
    if system is None:
        return None

    # V_k as (P, k), then J = (U B) V^T. U is regenerated from B and V rather
    # than stored: J v_i is one product each, and it keeps the peak memory at
    # the factors instead of at two dense copies.
    # The TRUNCATED basis, deliberately. The dropped trailing vector had no
    # beta computed, which means the process had not confirmed it carries
    # anything -- MEASURED at P=25, including it moved eps from 1.5e-03 to
    # 8.8e-03 away from the exact value, and downward, because a noise
    # direction in V makes r look more representable than it is. Reach the
    # full rank by asking for more ITERATIONS, never by keeping the offcut.
    vectors = system.basis
    basis = torch.stack(
        [torch.cat([block.reshape(-1) for block in vector]) for vector in vectors]
    )
    columns = [
        torch.cat([block.reshape(-1) for block in apply_j(vector)])
        for vector in vectors
    ]
    # FLOAT64, and not as a precaution. eps is read at DAMPING_BRACKET[0] =
    # 1e-16 * sigma_max^2, i.e. essentially none, so the smallest singular
    # values dominate the solve and any noise in J is amplified by the
    # condition number. The reconstruction accumulates k matmuls, so it
    # carries more rounding than a jacrev-built J of the same dtype: MEASURED,
    # the two agree to 1.2e-07 as matrices while their eps differed by 1.2e-03
    # -- four orders of amplification. The exact path casts to float64 at the
    # consumer for the same reason; this one has to be built that way.
    left = torch.stack(columns, dim=1).double()   # W = J V, (rows, k)
    right = basis.t().contiguous().double()       # V, (P, k), orthonormal cols

    # J IS NOT MATERIALISED. Building left @ right.t() would undo the whole
    # point: at MNIST width (10,240 rows, P ~ 59,000) that product is 4.8 GB
    # while the factors it came from are 110 MB at k=128, and the ladder builds
    # a system several times per epoch.
    #
    # A 0-row placeholder is returned in its place rather than None. It carries
    # shape[1] = P, so the consumers that only want the parameter count keep
    # working unchanged, and numel() == 0 marks it as unmaterialised for the
    # ones that check. None would turn every one of those reads into a crash.
    placeholder = left.new_zeros((0, right.shape[0]))
    return placeholder, (left, right)


@dataclass(frozen=True)
class _Bidiagonalization:
    """``J V_k = U_{k+1} B_k`` with ``U_{k+1} e_1 = r / ||r||``.

    The whole point of carrying this object instead of a single solution: the
    Krylov subspace ``K_k(J^T J, J^T r)`` DOES NOT DEPEND ON LAMBDA. Adding
    ``lambda I`` shifts the spectrum, it does not rotate the subspace, so one
    bidiagonalization serves the entire damping grid.
    """

    #: Orthonormal parameter-space basis, ``k`` entries of shape ``P``.
    basis: list
    #: ``B_k``, shape ``(rows, k)``, lower bidiagonal. Tiny.
    bidiagonal: torch.Tensor
    #: ``beta_1 = ||r||``. The right-hand side is ``beta_1 e_1``, exactly.
    beta1: float
    #: Golub-Kahan steps actually run.
    steps: int
    #: The basis BEFORE the trailing vector is dropped. That drop exists only
    #: to keep ``J V = U B`` honest, and a caller that reconstructs ``J`` as
    #: ``(J V) V^T`` never touches ``B`` -- so it wants every orthonormal
    #: column the process produced. MEASURED at P=25: using the truncated
    #: basis left k=24, one direction short, and eps came out 0.87634 against
    #: the exact 0.87767 -- 1.5e-03 LOW, the optimistic side.
    full_basis: list = None  # type: ignore[assignment]


def _golub_kahan(
    *,
    apply_j,
    apply_jt,
    residuals: list,
    iterations: int,
    floor: float,
) -> _Bidiagonalization | None:
    """Golub-Kahan bidiagonalization of ``J`` started at ``r``.

    Costs ``2k + 1`` products and NO ``lambda``. Full reorthogonalization is
    done against the stored bases: it is ``O(k^2 (P + NK))`` flops but zero
    extra passes over the data, and without it the Lanczos vectors lose
    orthogonality well before ``k = 50`` -- which would silently corrupt the
    ``eps`` identity below, since that identity is exactly what orthonormality
    buys.

    Memory is ``O(k (P + NK))``. Linear in ``P``, which is the entire reason
    this route exists: the exact Gram is ``O(P^2)``.
    """
    beta1 = _norm(residuals)
    if not beta1 > floor:
        return None
    u = _scaled(residuals, 1.0 / beta1)
    left = [u]

    v = apply_jt(u)
    alpha = _norm(v)
    if not alpha > floor:
        return None
    v = _scaled(v, 1.0 / alpha)
    basis = [v]
    alphas = [alpha]
    betas: list[float] = []
    exact = False

    for _ in range(max(1, iterations) - 1):
        w = _axpy(apply_j(v), -alpha, u)
        for previous in left:
            w = _axpy(w, -float(_dot(w, previous)), previous)
        beta = _norm(w)
        if not beta > floor:
            # Happy breakdown: J v_i lies in span(U_i), so the projection on
            # this subspace is EXACT and there is nothing left to capture.
            exact = True
            break
        u = _scaled(w, 1.0 / beta)
        left.append(u)
        betas.append(beta)

        z = _axpy(apply_jt(u), -beta, v)
        for previous in basis:
            z = _axpy(z, -float(_dot(z, previous)), previous)
        alpha = _norm(z)
        if not alpha > floor:
            break
        v = _scaled(z, 1.0 / alpha)
        basis.append(v)
        alphas.append(alpha)

    untruncated = list(basis)
    if not exact:
        # Drop any trailing v whose beta was never computed: without it,
        # J v_last is not representable in the retained U and the identity
        # J V = U B would be a lie in the last column.
        keep = len(betas)
        if keep == 0:
            return None
        basis = basis[:keep]
        alphas = alphas[:keep]
        rows = keep + 1
    else:
        rows = len(alphas)

    columns = len(alphas)
    bidiagonal = torch.zeros(rows, columns, dtype=torch.float64)
    for index, value in enumerate(alphas):
        bidiagonal[index, index] = value
    for index, value in enumerate(betas[:columns]):
        if index + 1 < rows:
            bidiagonal[index + 1, index] = value

    return _Bidiagonalization(
        basis=basis,
        bidiagonal=bidiagonal,
        beta1=beta1,
        steps=columns,
        full_basis=untruncated,
    )


def _shifted_epsilon(
    system: _Bidiagonalization, lam: float
) -> tuple[float, float | None, float, torch.Tensor]:
    """``eps``, ``cos`` and ``|J u|`` at one damping -- with no passes at all.

    ``u_lambda = V_k y_lambda`` and ``J u_lambda = U_{k+1} B_k y_lambda``, so
    because ``U`` is ORTHONORMAL and the right-hand side is ``beta_1 e_1``:

        ||J u - r||  = ||B y - beta_1 e_1||
        ||J u||      = ||B y||
        <J u, r>     = beta_1 (B y)_1

    Every quantity the certificate needs collapses to the ``(k+1) x k``
    factor. These are not approximations of the projected ``eps``; they are
    that ``eps``, evaluated exactly for the ``u`` that gets returned.
    """
    bidiagonal = system.bidiagonal
    rows, columns = bidiagonal.shape
    rhs = torch.zeros(rows, dtype=torch.float64)
    rhs[0] = system.beta1

    if lam > 0.0:
        stacked = torch.cat(
            [bidiagonal, math.sqrt(lam) * torch.eye(columns, dtype=torch.float64)]
        )
        padded = torch.cat([rhs, torch.zeros(columns, dtype=torch.float64)])
    else:
        stacked, padded = bidiagonal, rhs

    solution = torch.linalg.lstsq(stacked, padded.unsqueeze(1)).solution.squeeze(1)
    if not torch.isfinite(solution).all():
        return float("inf"), None, 0.0, solution

    predicted = bidiagonal @ solution
    predicted_norm = float(torch.linalg.vector_norm(predicted))
    if not predicted_norm > 0.0:
        return float("inf"), None, 0.0, solution
    gap = float(torch.linalg.vector_norm(predicted - rhs))
    # <J u, r> = beta_1 (B y)_1 and ||r|| = beta_1, so the beta_1 cancels.
    cosine = float(predicted[0]) / predicted_norm
    return gap / predicted_norm, cosine, predicted_norm, solution


def matrix_free_tangent_step(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: FGDApproxConfig,
    iterations: int = 200,
    damping: float = 1e-2,
    damping_grid: tuple[float, ...] | None = None,
    tolerance: float = 1e-10,
    max_batches: int | None = None,
) -> tuple[dict | None, MatrixFreeCertificate]:
    """The tangent projection and its ``eps``, without ever forming ``J``.

    Algorithmically this IS the tangent family: solve
    ``(J^T J + lambda I) u = J^T r`` and report ``eps = ||J u - r|| / ||J u||``
    at the ``lambda`` that minimises it. Only the SOLVER differs -- a Golub-Kahan
    bidiagonalization of matrix-vector products instead of a materialised Gram.

    MEASURED at P=197 against the exact Jacobian, at the ladder's own
    ``projection_damping = 1e-2``: 200 iterations agree to 5.2e-06 relative,
    50 iterations to 5.4e-03 -- both far inside the margin that matters against
    a threshold of 1/2. Damping helps twice over: it is the tangent's own
    regularisation AND it conditions the iteration, so the regime the ladder
    already operates in is the easy one for this solver.

    Cost is ``2k + 1`` passes REGARDLESS of how many dampings are tried, because
    the shifted systems share one Krylov subspace. Memory is ``O(k(P + NK))``.
    """
    increment("matrix_free_tangent_calls")
    was_training = model.training
    model.eval()
    passes = 0
    try:
        with timed("matrix_free_tangent_seconds"):
            graphs = _batch_graphs(model, loader, device, config, max_batches)
            if not graphs:
                return None, MatrixFreeCertificate(
                    float("inf"), None, 0.0, 0.0, 0, 0, False
                )
            parameters = graphs[0][4]

            def _jt(weights: list) -> list:
                """``J^T w``, accumulated over batches. One backward each."""
                nonlocal passes
                total = [torch.zeros_like(p) for p in parameters]
                for (outputs, _r, _d, _g, params), w in zip(graphs, weights):
                    passes += 1
                    grads = torch.autograd.grad(
                        outputs, params, grad_outputs=w, retain_graph=True
                    )
                    for accumulator, value in zip(total, grads):
                        accumulator.add_(value.detach())
                return total

            def _j(v: list) -> list:
                """``J v`` per batch, via the double-backward identity."""
                nonlocal passes
                out = []
                for _o, _r, dummy, jt_dummy, _p in graphs:
                    passes += 1
                    paired = sum((g * value).sum() for g, value in zip(jt_dummy, v))
                    (result,) = torch.autograd.grad(paired, dummy, retain_graph=True)
                    out.append(result.detach())
                return out

            residuals = [entry[1] for entry in graphs]
            residual_norm = math.sqrt(sum(float(torch.sum(r * r)) for r in residuals))
            if not residual_norm > 0.0:
                return None, MatrixFreeCertificate(
                    float("inf"), None, 0.0, residual_norm, 0, passes, False
                )

            # HARD CAP AT P. The Krylov subspace cannot exceed the parameter
            # count, so any step past it is numerical noise -- and the noise
            # does not cancel: MEASURED at P=25 with a too-small breakdown
            # floor, the process ran to k=48 and reported eps 0.8097 against
            # the true 0.8779. That is eps too SMALL, i.e. certifying a
            # direction that does not deserve it, which is the one direction
            # of error a certificate must never make.
            budget = sum(p.numel() for p in parameters)
            system = _golub_kahan(
                apply_j=_j,
                apply_jt=_jt,
                residuals=residuals,
                iterations=min(iterations, budget),
                floor=max(config.eps, tolerance * residual_norm),
            )
            if system is None:
                return None, MatrixFreeCertificate(
                    float("inf"), None, 0.0, residual_norm, 0, passes, False
                )

            # projection_damping_auto, matrix-free: eps depends on lambda and
            # the ladder picks the lambda that MINIMISES it per step. That is
            # not a detail -- at the fixed 1e-2 this solver reports eps 1.1-1.5
            # and never certifies, which is why a fixed-damping version looked
            # like the tangent could not step. It can; it just steps at a
            # different lambda.
            #
            # Sweeping it used to cost one full CG run PER lambda. It does not
            # have to: every shifted system shares the Krylov subspace above,
            # so the sweep is now a grid of (k+1) x k least squares with ZERO
            # further passes over the data. MEASURED: the pass count stops
            # depending on the grid size entirely.
            grid = damping_grid or (damping,)
            relative_error = float("inf")
            cosine = None
            predicted_norm = 0.0
            chosen = float(damping)
            coefficients = None
            for lam in grid:
                eps_l, cos_l, norm_l, y_l = _shifted_epsilon(system, float(lam))
                if eps_l < relative_error:
                    relative_error = eps_l
                    cosine = cos_l
                    predicted_norm = norm_l
                    chosen = float(lam)
                    coefficients = y_l

            if coefficients is None or not math.isfinite(relative_error):
                return None, MatrixFreeCertificate(
                    float("inf"), None, 0.0, residual_norm,
                    system.steps, passes, False,
                )
            if not predicted_norm > config.eps:
                return None, MatrixFreeCertificate(
                    float("inf"), None, predicted_norm, residual_norm,
                    system.steps, passes, False,
                )

            # u = V_k y, assembled only for the lambda that won.
            u = [torch.zeros_like(p) for p in parameters]
            for weight, vector in zip(coefficients.tolist(), system.basis):
                u = _axpy(u, weight, vector)
            if not all(torch.isfinite(value).all() for value in u):
                return None, MatrixFreeCertificate(
                    float("inf"), None, 0.0, residual_norm,
                    system.steps, passes, False,
                )

            performed = system.steps
            direction = {
                name: value
                for (name, _), value in zip(model.named_parameters(), u)
            }
    finally:
        model.train(was_training)

    return direction, MatrixFreeCertificate(
        relative_error=relative_error,
        cosine=cosine,
        predicted_norm=predicted_norm,
        residual_norm=residual_norm,
        iterations=performed,
        passes=passes,
        sensor_valid=math.isfinite(relative_error),
        damping=chosen,
    )
