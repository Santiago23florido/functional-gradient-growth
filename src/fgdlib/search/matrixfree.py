"""A matrix-free tangent direction, for use as a STALL fallback.

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


def matrix_free_tangent_step(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: FGDApproxConfig,
    iterations: int = 200,
    damping: float = 1e-2,
    tolerance: float = 1e-10,
    max_batches: int | None = None,
) -> tuple[dict | None, MatrixFreeCertificate]:
    """The tangent projection and its ``eps``, without ever forming ``J``.

    Algorithmically this IS the tangent family: solve
    ``(J^T J + lambda I) u = J^T r`` at the configured ``lambda`` and report
    ``eps = ||J u - r|| / ||J u||``. Only the SOLVER differs -- conjugate
    gradients on matrix-vector products instead of a materialised Gram.

    MEASURED at P=197 against the exact Jacobian, at the ladder's own
    ``projection_damping = 1e-2``: 200 iterations agree to 5.2e-06 relative,
    50 iterations to 5.4e-03 -- both far inside the margin that matters against
    a threshold of 1/2. Damping helps twice over: it is the tangent's own
    regularisation AND it conditions the CG, so the regime the ladder already
    operates in is the easy one for this solver.
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

            b = _jt(residuals)
            u = [torch.zeros_like(p) for p in parameters]
            r_cg = [value.clone() for value in b]
            p_cg = [value.clone() for value in r_cg]
            rs_old = _dot(r_cg, r_cg)
            b_norm = float(torch.sqrt(_dot(b, b)))
            if not b_norm > 0.0:
                return None, MatrixFreeCertificate(
                    float("inf"), None, 0.0, residual_norm, 0, passes, False
                )

            performed = 0
            for _ in range(max(1, iterations)):
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
                return None, MatrixFreeCertificate(
                    float("inf"), None, 0.0, residual_norm, performed, passes, False
                )

            predicted = _j(u)
            predicted_sq = sum(float(torch.sum(a * a)) for a in predicted)
            gap_sq = sum(
                float(torch.sum((a - r) ** 2)) for a, r in zip(predicted, residuals)
            )
            dot = sum(float(torch.sum(a * r)) for a, r in zip(predicted, residuals))
            predicted_norm = math.sqrt(predicted_sq)
            if not predicted_norm > config.eps:
                return None, MatrixFreeCertificate(
                    float("inf"), None, predicted_norm, residual_norm,
                    performed, passes, False,
                )
            relative_error = math.sqrt(gap_sq) / predicted_norm
            cosine = dot / (predicted_norm * residual_norm)
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
    )
