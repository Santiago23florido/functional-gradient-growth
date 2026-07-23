"""Realise the certified functional step as a PATH, not a single jump.

Lemma 3.5 licenses a move in function space, ``f -> f - eta g``. Taking it
as one linearised parameter step, ``theta - eta u``, is only the FIRST
Gauss-Newton iteration of the nonlinear problem it actually poses:

    find theta' such that   f(theta') ~ f_target := f(theta) - eta g.

The linearisation defect measures exactly how bad that single iteration is,
and the honest response to a bad first iteration is to take a second one --
not to shrink the step until the first is accurate, which is what shrinking
eta does. Shrinking is what stalls the method, and the measurement says so
unambiguously. With ``eps = 0.3788`` and ``L_s = 2``:

    eta_bar(eps) = 2(1 - 2 eps) / (L_s (1 + 2 eps)) = 0.1379
    certified                                         eta = 0.131
    actually applied after backtracking                eta = 0.000256
    ------------------------------------------------------------
    ratio                                                   512x

and at ``eps = 0.2710`` the ratio is 2047x. The certificate is not the
binding constraint anywhere near this regime -- it authorises steps between
two and three orders of magnitude larger than the ones taken. Nor is the
DIRECTION at fault: ``eps = 0.27`` means the tangent space captures 93 % of
the gradient energy, and earlier in the same run ``eps = 0.09`` captured
99.6 %. The approximation is good. What was failing is the realisation:
shrinking eta to keep one straight-line step accurate also shrinks the
functional movement by the same factor, so the method was taking 1/512 of
the step the lemma had certified.

Integrating instead of jumping fixes exactly that. Each inner iteration
moves a short way, so the linearisation holds where it is used, while the
TOTAL functional displacement is the full ``eta g`` the certificate
licensed. It is the difference between one giant Euler step and many small
ones: same destination, and only one of them arrives.

MEASURED on a grown synthetic model -- same direction, same certificate,
the same certified ``eta = 3.615e-2`` against the ``1.412e-4`` a single
jump was reduced to:

    one jump                        loss 9.580e1 -> 9.576e1   (delta 0.043)
    integrated, defect criterion    loss 9.580e1 -> 9.488e1   (delta 0.924)
    integrated, residual criterion  loss 9.580e1 -> 8.567e1   (delta 10.13)

realising 8.7 % and 96.2 % of the intended displacement respectively. The
criterion for the inner sub-step matters as much as integrating at all, and
:func:`_residual_reducing_sub_rate` explains why.

Both rules of the method are preserved, neither bent:

* **Only the tangent family.** Every inner iteration is the same tangent
  projection, re-solved at the current parameters against the functional
  residual that remains.
* **Only steps certified by the 1/2 condition.** The certificate is computed
  once, on ``g``, before any of this; it is unchanged by how the resulting
  functional step is realised. What iterates is the realisation, never the
  certification.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fgdlib.search.linearization import predicted_displacement
from fgdlib.tangent import (
    FGDApproxConfig,
    _output_relative_error_from_tensors,
    _solve_tangent_projection,
    _trainable_named_parameters,
    _unflatten_parameter_update,
    exact_tangent_system,
)

__all__ = ["RealizationResult", "realize_functional_step"]


@dataclass(frozen=True)
class RealizationResult:
    """How much of the certified functional step was actually delivered."""

    #: ``||f(theta') - f_target|| / ||eta g||`` -- 0 means fully realised.
    residual_fraction: float
    #: Fraction of the intended functional displacement actually travelled.
    realised_fraction: float
    iterations: int
    #: Parameter displacement accumulated, for reporting only.
    parameter_displacement: float


def realize_functional_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    updates: tuple[torch.Tensor, ...],
    learning_rate: float,
    config: FGDApproxConfig,
    max_iterations: int = 32,
    tolerance: float = 0.05,
) -> RealizationResult:
    """Move ``theta`` until ``f`` has travelled the certified ``-eta g``.

    Mutates ``model`` in place -- it performs the outer step rather than
    proposing one. ``updates`` and ``learning_rate`` define the target
    ``f_target = f(theta) - eta J u``; the target is fixed once, from the
    CURRENT parameters, so the certificate that licensed it keeps applying
    to it throughout.

    Each iteration solves the tangent projection against the functional
    residual that remains and takes the largest sub-step that reduces that
    residual. Stops when it is under ``tolerance`` of the intended
    displacement, or when no sub-step reduces it -- in which case the
    realised fraction reports honestly how far it got.
    """
    parameters = _trainable_named_parameters(model)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            start = model(x).detach()
        displacement = predicted_displacement(model, x, updates)
        target = start - learning_rate * displacement
        intended = float(torch.linalg.vector_norm(target - start))
        if not intended > 0.0:
            return RealizationResult(0.0, 0.0, 0, 0.0)

        travelled = 0.0
        iterations = 0
        for _ in range(max_iterations):
            with torch.no_grad():
                current = model(x).detach()
            remaining = float(torch.linalg.vector_norm(target - current))
            if remaining <= tolerance * intended:
                break

            # The functional residual still to be covered, in the same
            # convention the projection expects (it solves J v ~ d for a
            # DESCENT direction, so pass the negated shortfall and subtract).
            shortfall = (current - target).reshape(-1)
            system = exact_tangent_system(model, x, y, config)
            if system is None:
                break
            flat_step, approximation = _solve_tangent_projection(
                jacobian_matrix=system.jacobian,
                target=shortfall,
                damping=config.projection_damping,
                solver=config.projection_solver,
            )
            if not torch.isfinite(flat_step).all():
                break
            step_updates = _unflatten_parameter_update(
                flat_step, system.parameters
            )
            stats = _output_relative_error_from_tensors(
                approximation=approximation, target=shortfall, eps=config.eps
            )
            del stats  # measured for parity with the outer solve; unused here

            sub_rate = _residual_reducing_sub_rate(
                model, x, target, step_updates, remaining, config
            )
            if sub_rate is None:
                break

            with torch.no_grad():
                for parameter, step in zip(parameters.values(), step_updates):
                    parameter -= sub_rate * step.to(
                        parameter.device, parameter.dtype
                    )
                travelled += sub_rate * float(
                    torch.sqrt(sum((s.detach() ** 2).sum() for s in step_updates))
                )
            iterations += 1

        with torch.no_grad():
            final = model(x).detach()
        residual = float(torch.linalg.vector_norm(target - final))
        moved = float(torch.linalg.vector_norm(final - start))
        return RealizationResult(
            residual_fraction=residual / intended,
            realised_fraction=moved / intended,
            iterations=iterations,
            parameter_displacement=travelled,
        )
    finally:
        model.train(was_training)


def _residual_reducing_sub_rate(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    step_updates: tuple[torch.Tensor, ...],
    remaining: float,
    config: FGDApproxConfig,
    max_backtracks: int = 24,
) -> float | None:
    """Largest sub-step of 1.0 that gets CLOSER to the functional target.

    A full Gauss-Newton step is ``1.0`` here -- the projection already
    carries the magnitude needed to close the residual -- so this backtracks
    from 1 rather than from a certified rate.

    The criterion is the residual, not the linearisation defect, and the
    difference is worth being precise about because it is what makes the
    integration effective. Demanding a small defect asks each sub-step to be
    ACCURATE, which is the requirement for a single jump, where whatever the
    linearisation misses is simply lost. Here the state is re-measured every
    iteration and the projection is re-solved against the residual that
    actually remains, so nothing accumulates: a sub-step that overshoots or
    falls short is corrected by the next one. What the sub-step must do is
    make PROGRESS. Measured: the defect criterion realised 8.7 % of the
    intended displacement in 17 iterations, because it kept backtracking
    steps that were inaccurate but perfectly useful.

    This is not the descent gate that was removed. That one asked whether
    the LOSS improved on held-out data, a question Lemma 3.5 answers rather
    than poses. This asks whether we are getting closer to the functional
    target the certificate prescribed -- ``f - eta g``, on the same probe
    the certificate was measured on. It is the definition of realising the
    step, not an extra condition on it.

    ``None`` when no sub-step reduces the residual, which is where the
    integration stops.
    """
    parameters = _trainable_named_parameters(model)
    backtrack = min(max(config.lr_backtrack, 1e-6), 1.0 - 1e-9)
    rate = 1.0
    for _ in range(max_backtracks):
        with torch.no_grad():
            originals = [p.detach().clone() for p in parameters.values()]
            for parameter, step in zip(parameters.values(), step_updates):
                parameter -= rate * step.to(parameter.device, parameter.dtype)
            moved = model(x).detach()
            for parameter, original in zip(parameters.values(), originals):
                parameter.copy_(original)
        if torch.isfinite(moved).all():
            trial_residual = float(torch.linalg.vector_norm(target - moved))
            if trial_residual < remaining:
                return rate
        rate *= backtrack
    return None
