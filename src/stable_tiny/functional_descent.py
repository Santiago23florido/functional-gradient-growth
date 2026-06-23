"""Empirical functional-gradient descent steps for the stable-tiny harness.

This module deliberately lives in ``stable_tiny`` rather than in ``gromo``.  It
uses the current DAG/MLP as the representation of admissible functional changes
and certifies each batch step against the paper's relative-error condition in
the finite empirical output space of that batch.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import torch

try:
    from torch.func import functional_call, jvp
except ImportError:  # pragma: no cover - depends on installed torch.
    functional_call = None
    jvp = None


@dataclass
class FunctionalStepInfo:
    certified: bool
    loss: float
    relative_error: float
    error_bound: float
    approx_norm: float
    cg_iterations: int
    reason: str = ""
    # --- theory-grounded diagnostics (Lemma 3.5 / expressivity bottleneck) ---
    # ``in_tangent_norm`` = ||g||_B  (expressible part of the functional gradient)
    # ``residual_norm``   = ||grad L - g||_B  (the *expressivity bottleneck*: the
    #                       part of the ideal functional gradient that no parameter
    #                       move of the current network can realize).
    # ``bottleneck_fraction`` = ||r|| / ||grad L||  in [0, 1].
    # ``directional_derivative`` = <grad L, g> = DL(f; g) >= 0 (descent rate).
    # ``descent_ok`` is True when the *measured* loss decrease satisfied the
    #   sufficient-descent (Armijo) condition derived from Lemma 3.5.
    # ``lr_used`` is the step size accepted by the descent line search.
    # ``loss_after`` is the batch loss after the accepted step.
    in_tangent_norm: float = 0.0
    residual_norm: float = 0.0
    bottleneck_fraction: float = 0.0
    directional_derivative: float = 0.0
    descent_ok: bool = True
    lr_used: float = 0.0
    loss_after: float = float("nan")


@dataclass
class FunctionalEpochInfo:
    loss: float
    batches: int
    certified_batches: int
    failed: bool
    failure: FunctionalStepInfo | None
    max_relative_error: float
    mean_relative_error: float
    descent_failures: int = 0
    mean_bottleneck_fraction: float = 0.0
    max_bottleneck_fraction: float = 0.0


def _ggn_half(output: torch.Tensor) -> torch.Tensor:
    """Per-example square root ``M^{1/2}`` of the softmax-CE output Hessian.

    The paper works in a *Hilbert* space of functions; for a cross-entropy loss
    the natural inner product on the logit output space is the one induced by the
    loss Hessian (the GGN / Fisher metric), not the plain Euclidean dot product.
    Per example the Hessian of ``CE(softmax(f), y)`` w.r.t. the logits ``f`` is

        ``M = diag(p) - p p^T``  with ``p = softmax(f)``,

    a symmetric PSD matrix of rank ``C - 1`` (it annihilates the all-ones vector,
    i.e. the loss-irrelevant overall-logit shift). Measuring the expressivity
    bottleneck and projecting in the ``M`` metric is equivalent to whitening the
    output space by ``M^{1/2}``, which we return here as a ``[B, C, C]`` batch.
    """
    p = torch.softmax(output.detach(), dim=-1)
    M = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
    # As the model grows confident, p -> one-hot and M -> 0 (all eigenvalues
    # degenerate); CUDA's batched eigh fails on such ill-conditioned blocks. The
    # eigendecomposition of these tiny C x C blocks is cheap, so we do it in
    # float64 on the CPU (LAPACK syevd handles repeated/zero eigenvalues) with a
    # negligible diagonal jitter to break exact degeneracy, then clamp and return.
    M64 = M.double().cpu()
    M64.diagonal(dim1=-2, dim2=-1).add_(1e-9)
    evals, evecs = torch.linalg.eigh(M64)
    evals = evals.clamp_min(0.0)
    half = (evecs * evals.sqrt().unsqueeze(-2)) @ evecs.transpose(-1, -2)
    return half.to(device=output.device, dtype=output.dtype)


def _metric_factors(output: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Square root and inverse of the SPD metric ``A = M + tau I`` per example.

    The GGN/Fisher Hessian ``M = diag(p) - p p^T`` is only positive *semi*definite
    (it annihilates the all-ones vector), so it is not a genuine inner product and
    cannot, by itself, define a Hilbert space in the sense the base paper requires.
    Regularizing to ``A = M + tau I`` with ``tau > 0`` makes it SPD, hence a bona
    fide inner product, while keeping the loss-relevant geometry (the gradient
    ``p - y`` is orthogonal to the all-ones direction, so it lives in the range of
    ``M`` and ``A^{-1}(p-y)`` stays bounded as ``tau -> 0``). We return ``A^{1/2}``
    and ``A^{-1}`` as ``[B, C, C]`` batches, computed from one eigendecomposition.
    """
    p = torch.softmax(output.detach(), dim=-1)
    M = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
    M64 = M.double().cpu()
    M64.diagonal(dim1=-2, dim2=-1).add_(1e-9)
    evals, evecs = torch.linalg.eigh(M64)
    a = evals.clamp_min(0.0) + float(tau)  # eigenvalues of A = M + tau I
    half = (evecs * a.sqrt().unsqueeze(-2)) @ evecs.transpose(-1, -2)
    inv = (evecs * a.reciprocal().unsqueeze(-2)) @ evecs.transpose(-1, -2)
    return (
        half.to(device=output.device, dtype=output.dtype),
        inv.to(device=output.device, dtype=output.dtype),
    )


def _whiten_vector(flat: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """Apply the block-diagonal ``M^{1/2}`` to a flat ``[B*C]`` output vector."""
    B, C, _ = half.shape
    v = flat.reshape(B, C, 1)
    return torch.bmm(half, v).reshape(-1)


def _whiten_jacobian(jac: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """Apply the block-diagonal ``M^{1/2}`` to a ``[B*C, P]`` Jacobian."""
    B, C, _ = half.shape
    J = jac.reshape(B, C, -1)
    return torch.bmm(half, J).reshape(B * C, -1)


def certified_functional_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: torch.nn.Module,
    *,
    lr: float,
    damping: float,
    relative_error_tolerance: float,
    cg_min_iter: int,
    cg_max_iter: int,
    cg_tolerance: float,
    weight_decay: float = 0.0,
    grad_clip: float | None = None,
    sufficient_descent_c: float | None = None,
    lr_backtrack: float = 0.5,
    lr_min_factor: float = 1e-3,
    apply_step: bool = True,
    metric: str = "euclidean",
    metric_damping: float = 1e-3,
) -> FunctionalStepInfo:
    """Apply one certified projected functional-gradient step.

    The ideal target is ``dL / df`` on this batch.  The approximating family is
    the tangent space generated by the current model parameters.  The step is
    accepted only when ``(1 + eps) U < eps ||g||`` with ``U`` measured exactly in
    batch output space.
    """
    if functional_call is None or jvp is None:
        raise RuntimeError(
            "Certified functional descent requires torch.func.functional_call "
            "and torch.func.jvp."
        )
    if not (0.0 < relative_error_tolerance < 1.0):
        raise ValueError(
            "relative_error_tolerance must be in (0, 1), got "
            f"{relative_error_tolerance}"
        )
    if cg_min_iter < 1 or cg_max_iter < cg_min_iter:
        raise ValueError(
            f"Expected 1 <= cg_min_iter <= cg_max_iter, got "
            f"{cg_min_iter=} and {cg_max_iter=}"
        )
    if metric not in ("euclidean", "ggn", "natural"):
        raise ValueError(
            f"Unknown metric {metric!r}; expected 'euclidean', 'ggn' or 'natural'."
        )
    if metric in ("ggn", "natural"):
        # The non-Euclidean metrics require the explicit output-space Jacobian to
        # whiten; use functional_projection=exact instead of the CG approximation.
        raise NotImplementedError(
            f"metric={metric!r} is only implemented for the exact projection "
            "(functional_projection=exact), not the CG matvec path."
        )

    model.zero_grad(set_to_none=True)
    named_parameters = OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not named_parameters:
        output = model(x)
        loss = loss_fn(output, y)
        return FunctionalStepInfo(
            certified=False,
            loss=float(loss.detach()),
            relative_error=float("inf"),
            error_bound=float("inf"),
            approx_norm=0.0,
            cg_iterations=0,
            reason="model has no trainable parameters",
        )

    parameter_names = tuple(named_parameters.keys())
    parameters = tuple(named_parameters.values())
    buffers = OrderedDict(model.named_buffers())

    def call_with_parameters(parameter_values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        state = OrderedDict(zip(parameter_names, parameter_values))
        state.update(buffers)
        return functional_call(model, state, (x,))

    output = model(x)
    loss = loss_fn(output, y)
    true_gradient = torch.autograd.grad(loss, output, retain_graph=True)[0].detach()
    flat_true_gradient = true_gradient.reshape(-1)
    true_norm = torch.linalg.norm(flat_true_gradient)
    if true_norm <= torch.finfo(flat_true_gradient.dtype).eps:
        return FunctionalStepInfo(
            certified=True,
            loss=float(loss.detach()),
            relative_error=0.0,
            error_bound=0.0,
            approx_norm=0.0,
            cg_iterations=0,
            reason="zero functional gradient",
        )

    def tangent_kernel_matvec(vector: torch.Tensor) -> torch.Tensor:
        structured_vector = vector.reshape_as(output)
        vjp_tensors = torch.autograd.grad(
            output,
            parameters,
            grad_outputs=structured_vector,
            retain_graph=True,
            allow_unused=True,
        )
        tangents = tuple(
            torch.zeros_like(parameter) if tangent is None else tangent.detach()
            for parameter, tangent in zip(parameters, vjp_tensors)
        )
        _, tangent_output = jvp(
            call_with_parameters,
            (parameters,),
            (tangents,),
        )
        return tangent_output.reshape(-1).detach()

    def damped_tangent_kernel_matvec(vector: torch.Tensor) -> torch.Tensor:
        return tangent_kernel_matvec(vector) + damping * vector

    # Run the CG refinement schedule and keep the *best* approximation found
    # (lowest relative error), stopping early as soon as one certifies.  Even if
    # none certifies, g = K alpha is still a descent direction (K is PSD, so
    # <grad L, g> >= 0): we always take the step and let the relative-error
    # certificate act as the growth trigger.  This avoids the failure mode where
    # an uncertified step makes no progress at all.
    best_info: FunctionalStepInfo | None = None
    accepted_solution: torch.Tensor | None = None
    accepted_iterations = 0
    best_relative_error = float("inf")
    for max_iter in _cg_refinement_schedule(cg_min_iter, cg_max_iter):
        solution, used_iterations = _conjugate_gradient(
            damped_tangent_kernel_matvec,
            flat_true_gradient,
            max_iter=max_iter,
            tolerance=cg_tolerance,
        )
        approx_gradient = tangent_kernel_matvec(solution)
        info = _certificate(
            loss=loss,
            approx_gradient=approx_gradient,
            true_gradient=flat_true_gradient,
            eps=relative_error_tolerance,
            cg_iterations=used_iterations,
        )
        if info.relative_error < best_relative_error:
            best_relative_error = info.relative_error
            best_info = info
            accepted_solution = solution
            accepted_iterations = used_iterations
        if info.certified:
            best_info = info
            accepted_solution = solution
            accepted_iterations = used_iterations
            break

    assert best_info is not None and accepted_solution is not None
    best_info.cg_iterations = accepted_iterations
    if not best_info.certified and not best_info.reason:
        best_info.reason = "relative-error certificate failed"

    if not apply_step:
        # Certificate-only (dry-run): report the decomposition / certificate
        # without modifying parameters.  Used by the grow-until-certified loop.
        return best_info

    if not best_info.certified and sufficient_descent_c is None:
        # Legacy (v1) behaviour: never step on an uncertified batch.
        return best_info

    # The parameter step that realizes the (projected) functional gradient g in
    # output space: theta <- theta - lr * J^T alpha, whose first-order output
    # change is -lr * J J^T alpha = -lr * g.
    parameter_gradients = torch.autograd.grad(
        output,
        parameters,
        grad_outputs=accepted_solution.reshape_as(output),
        retain_graph=False,
        allow_unused=True,
    )
    step_directions = tuple(
        torch.zeros_like(parameter) if grad is None else grad.detach()
        for parameter, grad in zip(parameters, parameter_gradients)
    )
    return _take_descent_step(
        model,
        x,
        y,
        loss_fn,
        parameters,
        step_directions,
        best_info,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        sufficient_descent_c=sufficient_descent_c,
        lr_backtrack=lr_backtrack,
        lr_min_factor=lr_min_factor,
    )


def _take_descent_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: torch.nn.Module,
    parameters: tuple[torch.Tensor, ...],
    step_directions: tuple[torch.Tensor, ...],
    info: FunctionalStepInfo,
    *,
    lr: float,
    weight_decay: float,
    grad_clip: float | None,
    sufficient_descent_c: float | None,
    lr_backtrack: float,
    lr_min_factor: float,
) -> FunctionalStepInfo:
    """Apply ``theta <- theta - lr * step_directions`` with optional verified descent.

    The realized output change is ``-lr * g`` where ``g`` is the projected
    functional gradient.  When ``sufficient_descent_c`` is set, a function-space
    Armijo line search backtracks ``lr`` until the Lemma-3.5 sufficient-descent
    condition ``L(f - lr g) <= L(f) - c * lr * <grad L, g>`` holds.
    """
    if grad_clip is not None:
        total_norm = torch.linalg.norm(
            torch.stack([torch.linalg.norm(d) for d in step_directions])
        )
        if float(total_norm) > grad_clip:
            scale = grad_clip / (float(total_norm) + 1e-12)
            step_directions = tuple(d * scale for d in step_directions)

    if sufficient_descent_c is None:
        # Legacy path: a single fixed-lr step, sufficient descent not verified.
        with torch.no_grad():
            for parameter, direction in zip(parameters, step_directions):
                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(direction, alpha=-lr)
        info.lr_used = lr
        return info

    base_params = tuple(parameter.detach().clone() for parameter in parameters)
    base_loss = info.loss
    directional_derivative = info.directional_derivative

    def _apply(step_lr: float) -> None:
        with torch.no_grad():
            for parameter, base, direction in zip(parameters, base_params, step_directions):
                parameter.copy_(base)
                if weight_decay != 0.0:
                    parameter.mul_(1.0 - step_lr * weight_decay)
                parameter.add_(direction, alpha=-step_lr)

    @torch.no_grad()
    def _trial_loss() -> float:
        return float(loss_fn(model(x), y).detach())

    trial_lr = lr
    min_lr = lr * lr_min_factor
    accepted_lr = None
    accepted_loss = base_loss
    while trial_lr >= min_lr:
        _apply(trial_lr)
        trial_loss = _trial_loss()
        if trial_loss <= base_loss - sufficient_descent_c * trial_lr * directional_derivative:
            accepted_lr = trial_lr
            accepted_loss = trial_loss
            break
        trial_lr *= lr_backtrack

    if accepted_lr is None:
        # No step size produced the guaranteed decrease: the current tangent
        # space is the limiting factor.  Take the smallest tried step (still a
        # non-ascent direction) and flag the descent failure so the caller can
        # refine the representation (grow).
        accepted_lr = max(trial_lr, min_lr)
        _apply(accepted_lr)
        accepted_loss = _trial_loss()
        info.descent_ok = False
        if not info.reason:
            info.reason = "sufficient-descent line search failed"

    info.lr_used = accepted_lr
    info.loss_after = accepted_loss
    return info


def exact_functional_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: torch.nn.Module,
    *,
    lr: float,
    damping: float = 0.0,
    relative_error_tolerance: float,
    weight_decay: float = 0.0,
    grad_clip: float | None = None,
    sufficient_descent_c: float | None = None,
    lr_backtrack: float = 0.5,
    lr_min_factor: float = 1e-3,
    apply_step: bool = True,
    metric: str = "euclidean",
    metric_damping: float = 1e-3,
    **_ignored,
) -> FunctionalStepInfo:
    """Certified functional-gradient step with an *exact* tangent projection.

    For the tiny models in this harness the full output-space Jacobian
    ``J = d output / d theta`` (shape ``[B*C, P]``) fits in memory, so the
    projection of the ideal functional gradient ``grad L`` onto the tangent space
    can be computed exactly via least squares instead of conjugate gradient.

    This removes the CG conditioning pathology (CG fails to converge on trained,
    over-parameterized models), giving an exact expressivity-bottleneck residual
    ``r = grad L - g``.  The bottleneck is essentially zero once the number of
    parameters exceeds the output-space dimension and ``J`` is full rank, and is
    strictly positive when the network is too small to represent ``grad L`` --
    the genuine expressivity bottleneck that growth is meant to remove.
    """
    if functional_call is None:
        raise RuntimeError("exact functional descent requires torch.func.functional_call")
    from torch.func import jacrev

    model.zero_grad(set_to_none=True)
    named_parameters = OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    parameter_names = tuple(named_parameters.keys())
    parameters = tuple(named_parameters.values())
    buffers = OrderedDict(model.named_buffers())

    output = model(x)
    loss = loss_fn(output, y)
    true_gradient = torch.autograd.grad(loss, output)[0].detach().reshape(-1)
    output_numel = true_gradient.shape[0]
    true_norm = torch.linalg.norm(true_gradient)
    if true_norm <= torch.finfo(true_gradient.dtype).eps:
        return FunctionalStepInfo(
            certified=True, loss=float(loss.detach()), relative_error=0.0,
            error_bound=0.0, approx_norm=0.0, cg_iterations=0,
            reason="zero functional gradient",
        )

    def call_with_parameters(parameter_values):
        state = OrderedDict(zip(parameter_names, parameter_values))
        state.update(buffers)
        return functional_call(model, state, (x,)).reshape(-1)

    if metric not in ("euclidean", "ggn", "natural"):
        raise ValueError(
            f"Unknown metric {metric!r}; expected 'euclidean', 'ggn' or 'natural'."
        )

    jacobian = jacrev(call_with_parameters)(parameters)
    columns = [jac.reshape(output_numel, -1) for jac in jacobian]
    jacobian_matrix = torch.cat(columns, dim=1).detach()  # [B*C, P]

    if metric == "natural":
        # FAITHFUL natural-metric FGD. Work in the genuine Hilbert space
        # H = B = (R^m, <.,.>_A) with A = M + tau I SPD (M the softmax-CE Hessian).
        # The functional gradient is then the Riesz representer grad_A L = A^{-1} dL/df,
        # and the approximation is its A-orthogonal projection onto the tangent
        # space, which simplifies to the Gauss-Newton / natural-gradient step
        #     theta_dot = (J^T A J + lambda I)^{-1} J^T (dL/df),     g = J theta_dot.
        # All certificate norms are A-norms; <grad_A L, g>_A = ||g||_A^2 >= 0, so g
        # is a descent direction and Asm. 3.3/3.4 hold with alpha=beta=1 (B=H).
        A_half, A_inv = _metric_factors(output, metric_damping)
        whitened_jac = _whiten_jacobian(jacobian_matrix, A_half)  # (A^{1/2} J)
        rhs = jacobian_matrix.t() @ true_gradient                 # J^T dL/df (no metric!)
        gram = whitened_jac.t() @ whitened_jac                    # J^T A J
        if damping > 0.0:
            gram.diagonal().add_(damping)
        flat_solution = torch.linalg.solve(gram, rhs)             # theta_dot
        g_norm_sq = float(torch.dot(flat_solution, rhs).clamp_min(0.0))  # ||g||_A^2
        a_inv_grad = _whiten_vector(true_gradient, A_inv)         # A^{-1} dL/df
        grad_norm_sq = float(torch.dot(true_gradient, a_inv_grad).clamp_min(0.0))  # ||grad_A L||_A^2
        e_norm = (max(grad_norm_sq - g_norm_sq, 0.0)) ** 0.5      # Pythagoras in A
        g_norm = g_norm_sq ** 0.5
        true_norm_A = grad_norm_sq ** 0.5
        certified = bool((1.0 + relative_error_tolerance) * e_norm
                         < relative_error_tolerance * g_norm)
        info = FunctionalStepInfo(
            certified=certified,
            loss=float(loss.detach()),
            relative_error=(e_norm / g_norm) if g_norm > 0 else float("inf"),
            error_bound=e_norm,
            approx_norm=g_norm,
            cg_iterations=0,
            in_tangent_norm=g_norm,
            residual_norm=e_norm,
            bottleneck_fraction=(e_norm / true_norm_A) if true_norm_A > 0 else 0.0,
            # Euclidean loss slope along g equals <dL/df, g> = theta_dot^T J^T dL/df
            # = ||g||_A^2, so the Armijo model is consistent with the actual loss.
            directional_derivative=g_norm_sq,
        )
        if not certified:
            info.reason = "relative-error certificate failed"
    else:
        # The projection / certificate is computed in the chosen output-space metric.
        # For the (heuristic) GGN metric we whiten both the Jacobian and the ideal
        # functional gradient by M^{1/2}; lstsq in whitened space then projects the
        # *Euclidean* gradient under the M-norm (a Gauss-Newton preconditioner --
        # NOT FGD in the M-Hilbert space; use metric='natural' for the faithful one).
        if metric == "ggn":
            half = _ggn_half(output)
            solve_jac = _whiten_jacobian(jacobian_matrix, half)
            solve_rhs = _whiten_vector(true_gradient, half)
        else:
            solve_jac = jacobian_matrix
            solve_rhs = true_gradient

        if damping > 0.0:
            # Ridge / Levenberg-Marquardt: solve (J^T M J + damping I) theta_dot = J^T M r.
            gram = solve_jac.t() @ solve_jac
            gram.diagonal().add_(damping)
            rhs = solve_jac.t() @ solve_rhs
            flat_solution = torch.linalg.solve(gram, rhs)
        else:
            flat_solution = torch.linalg.lstsq(solve_jac, solve_rhs).solution

        # Certificate / bottleneck measured in the chosen metric (whitened space).
        approx_in_metric = solve_jac @ flat_solution
        info = _certificate(
            loss=loss,
            approx_gradient=approx_in_metric,
            true_gradient=solve_rhs,
            eps=relative_error_tolerance,
            cg_iterations=0,
        )
        if metric == "ggn":
            # The Armijo sufficient-descent model uses the *Euclidean* directional
            # derivative <grad L, g>, since the first-order loss change along the
            # output move g = J theta_dot is dL = <grad L, g> regardless of metric.
            euclid_g = jacobian_matrix @ flat_solution
            info.directional_derivative = float(torch.dot(true_gradient, euclid_g).detach())
        if not info.certified and not info.reason:
            info.reason = "relative-error certificate failed"

    if not apply_step:
        return info
    if not info.certified and sufficient_descent_c is None:
        return info

    # theta_dot reshaped back to the parameter shapes; step is theta -= lr * theta_dot.
    step_directions = []
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        step_directions.append(
            flat_solution[offset : offset + size].reshape_as(parameter).detach()
        )
        offset += size
    return _take_descent_step(
        model, x, y, loss_fn, parameters, tuple(step_directions), info,
        lr=lr, weight_decay=weight_decay, grad_clip=grad_clip,
        sufficient_descent_c=sufficient_descent_c, lr_backtrack=lr_backtrack,
        lr_min_factor=lr_min_factor,
    )


def functional_descent_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    *,
    device: torch.device,
    lr: float,
    damping: float,
    relative_error_tolerance: float,
    cg_min_iter: int,
    cg_max_iter: int,
    cg_tolerance: float,
    weight_decay: float = 0.0,
    grad_clip: float | None = None,
    batch_limit: int | None = None,
    progress_every: int | None = None,
    sufficient_descent_c: float | None = None,
    lr_backtrack: float = 0.5,
    lr_min_factor: float = 1e-3,
    stop_on_failure: bool = True,
    projection: str = "cg",
    metric: str = "euclidean",
    metric_damping: float = 1e-3,
) -> FunctionalEpochInfo:
    """Run certified functional descent over the epoch.

    When ``stop_on_failure`` is True (legacy v1) the epoch stops at the first
    uncertified batch.  When False (v2) every batch takes a verified-descent
    step regardless of certification, and certification is only recorded (the
    caller uses the certified fraction to decide whether to grow)."""
    model.train()
    losses: list[float] = []
    relative_errors: list[float] = []
    bottleneck_fractions: list[float] = []
    descent_failures = 0
    certified_batches = 0
    batches = 0
    failure = None

    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_limit is not None and batch_idx >= batch_limit:
            break
        batches += 1
        if progress_every is not None and progress_every > 0:
            if batch_idx % progress_every == 0:
                total = batch_limit if batch_limit is not None else len(train_loader)
                print(
                    f"    functional batch {batch_idx + 1}/{total} "
                    f"(cg_max={cg_max_iter})",
                    flush=True,
                )
        x, y = x.to(device), y.to(device)
        step_fn = exact_functional_step if projection == "exact" else certified_functional_step
        info = step_fn(
            model,
            x,
            y,
            loss_fn,
            lr=lr,
            damping=damping,
            relative_error_tolerance=relative_error_tolerance,
            cg_min_iter=cg_min_iter,
            cg_max_iter=cg_max_iter,
            cg_tolerance=cg_tolerance,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            sufficient_descent_c=sufficient_descent_c,
            lr_backtrack=lr_backtrack,
            lr_min_factor=lr_min_factor,
            metric=metric,
            metric_damping=metric_damping,
        )
        losses.append(info.loss)
        relative_errors.append(info.relative_error)
        bottleneck_fractions.append(info.bottleneck_fraction)
        if not info.descent_ok:
            descent_failures += 1
        if not info.certified:
            if progress_every is not None and progress_every > 0:
                print(
                    f"      failed rel_err={info.relative_error:.3g} "
                    f"cg={info.cg_iterations} reason={info.reason}",
                    flush=True,
                )
            if failure is None:
                failure = info
            if stop_on_failure:
                break
            continue
        certified_batches += 1
        if progress_every is not None and progress_every > 0:
            if batch_idx % progress_every == 0:
                print(
                    f"      certified rel_err={info.relative_error:.3g} "
                    f"cg={info.cg_iterations}",
                    flush=True,
                )

    finite_relative_errors = [
        value for value in relative_errors if value != float("inf")
    ]
    return FunctionalEpochInfo(
        loss=sum(losses) / max(1, len(losses)),
        batches=batches,
        certified_batches=certified_batches,
        failed=failure is not None,
        failure=failure,
        max_relative_error=max(relative_errors) if relative_errors else 0.0,
        mean_relative_error=(
            sum(finite_relative_errors) / len(finite_relative_errors)
            if finite_relative_errors
            else float("inf")
            if relative_errors
            else 0.0
        ),
        descent_failures=descent_failures,
        mean_bottleneck_fraction=(
            sum(bottleneck_fractions) / len(bottleneck_fractions)
            if bottleneck_fractions
            else 0.0
        ),
        max_bottleneck_fraction=(
            max(bottleneck_fractions) if bottleneck_fractions else 0.0
        ),
    )


def _certificate(
    *,
    loss: torch.Tensor,
    approx_gradient: torch.Tensor,
    true_gradient: torch.Tensor,
    eps: float,
    cg_iterations: int,
) -> FunctionalStepInfo:
    residual = true_gradient - approx_gradient
    error = torch.linalg.norm(residual)
    approx_norm = torch.linalg.norm(approx_gradient)
    true_norm = torch.linalg.norm(true_gradient)
    # DL(f; g) = <grad L, g> >= 0: the rate of functional-loss decrease.
    directional_derivative = float(torch.dot(true_gradient, approx_gradient).detach())
    bottleneck_fraction = float(
        (error / true_norm).detach()
    ) if true_norm > 0 else 0.0
    if approx_norm <= torch.finfo(approx_gradient.dtype).eps:
        return FunctionalStepInfo(
            certified=False,
            loss=float(loss.detach()),
            relative_error=float("inf"),
            error_bound=float(error.detach()),
            approx_norm=0.0,
            cg_iterations=cg_iterations,
            reason="zero approximate gradient",
            in_tangent_norm=0.0,
            residual_norm=float(error.detach()),
            bottleneck_fraction=bottleneck_fraction,
            directional_derivative=directional_derivative,
        )

    relative_error = error / approx_norm
    certified = bool((1.0 + eps) * error < eps * approx_norm)
    return FunctionalStepInfo(
        certified=certified,
        loss=float(loss.detach()),
        relative_error=float(relative_error.detach()),
        error_bound=float(error.detach()),
        approx_norm=float(approx_norm.detach()),
        cg_iterations=cg_iterations,
        in_tangent_norm=float(approx_norm.detach()),
        residual_norm=float(error.detach()),
        bottleneck_fraction=bottleneck_fraction,
        directional_derivative=directional_derivative,
    )


def _cg_refinement_schedule(min_iter: int, max_iter: int) -> list[int]:
    values = []
    current = min_iter
    while current < max_iter:
        values.append(current)
        current *= 2
    values.append(max_iter)
    return values


def _conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    max_iter: int,
    tolerance: float,
) -> tuple[torch.Tensor, int]:
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    rhs_norm = torch.linalg.norm(rhs).clamp_min(torch.finfo(rhs.dtype).eps)
    residual_inner = torch.dot(residual, residual)

    for iteration in range(1, max_iter + 1):
        matvec_direction = matvec(direction)
        curvature = torch.dot(direction, matvec_direction)
        if curvature.abs() <= torch.finfo(rhs.dtype).eps:
            return solution, iteration

        step_size = residual_inner / curvature
        solution = solution + step_size * direction
        residual = residual - step_size * matvec_direction
        if torch.linalg.norm(residual) / rhs_norm <= tolerance:
            return solution, iteration

        next_residual_inner = torch.dot(residual, residual)
        beta = next_residual_inner / residual_inner.clamp_min(
            torch.finfo(rhs.dtype).eps
        )
        direction = residual + beta * direction
        residual_inner = next_residual_inner

    return solution, max_iter
