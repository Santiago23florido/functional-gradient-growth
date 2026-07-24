"""Enforce Lemma 3.5's own hypothesis: that the step is a function-space step.

Lemma 3.5 licenses ``eta`` from the smoothness of ``L`` as a function of
``f``. Its subject is the FUNCTION-space step

    f  <-  f - eta g,        g = P_T(r),

and its conclusion -- descent for every ``eta`` in ``(0, eta_bar(eps))`` --
is about that object. What the optimiser actually performs is a
PARAMETER-space step ``theta <- theta - eta u``, whose effect on ``f`` is

    f(theta - eta u) = f(theta) - eta J u + O(eta^2 ||u||^2)
                     = f(theta) - eta g   + O(eta^2 ||u||^2)

because ``u`` is solved so that ``J u = g``. The lemma governs the first two
terms. The remainder is second order in ``eta`` and is NOT controlled by
``L_s``: it is governed by the curvature of the parameter-to-function map,
which the certificate never measures.

That gap is not academic; it is what a certified-but-unverified flow walks
straight into. MEASURED on the synthetic task with sum-MSE -- where the
theory is at its strongest, an exact function-space PL constant ``mu = 2``
with ``L* = 0``:

    step   eta        loss before -> after
    1      0.005078   2.308e3 -> 4.248e2      (a large, correct descent)
    2      0.03219    4.248e2 -> 4.378e2
    3      0.1667     4.378e2 -> 2.358e3
    4      0.5414     2.358e3 -> 2.271e4
    5      0.7926     2.271e4 -> 2.057e5
    6      0.8549     2.057e5 -> 7.493e7

while the certificate reported everything in order -- ``eps`` fell 0.50 ->
0.085, far below the 1/2 threshold, and accuracy collapsed 0.087 -> 0.004.

The mechanism is self-reinforcing, which is why it is easy to miss: as
``eps -> 0`` the bound ``eta_bar(eps) = 2(1-2 eps)/(L_s (1+2 eps))`` rises
towards ``2/L_s``, so the BETTER the certificate, the LARGER the step it
authorises and the further outside the linear regime the step lands. The
certificate and the validity of its own application move in opposite
directions.

The remedy here is not an extra condition layered on top of the lemma. It
is the lemma's own hypothesis, enforced: measure whether the parameter step
actually realises the function-space displacement the lemma is about,

    delta(eta) = || f(theta - eta u) - (f(theta) - eta g) || / (eta ||g||)

and shrink ``eta`` until it does. Note what this does NOT look at: the loss.
It is not the descent check that was deliberately removed -- it never asks
whether the step improved anything. It asks only whether the theorem is
speaking about the step being taken. A step that passes is one where
Lemma 3.5's conclusion genuinely transfers; a step that fails is one where
the lemma was being applied to an object it does not describe.

Both of the method's rules are preserved: only the tangent family, and only
steps certified by the ``1/2`` condition. This narrows ``eta`` INSIDE the
certified interval; it never enlarges it and never admits an uncertified
step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.func import functional_call, jvp

from fgdlib.tangent import (
    FGDApproxConfig,
    _clear_inaccessible_tensor_caches,
    _trainable_named_parameters,
)

__all__ = [
    "LinearizedStep",
    "linearization_defect",
    "predicted_displacement",
    "certified_linear_learning_rate",
]


@dataclass(frozen=True)
class LinearizedStep:
    """The rate that keeps the step inside the regime the lemma describes."""

    learning_rate: float | None
    defect: float
    backtracks: int
    #: The rate before any narrowing -- 0.95 * eta_bar(eps).
    certified_learning_rate: float


def predicted_displacement(
    model: torch.nn.Module,
    x: torch.Tensor,
    updates: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """``J u`` -- the function-space displacement the parameter step predicts.

    Computed by forward-mode AD, so it is the exact directional derivative
    rather than a finite difference: no step size enters, and the result is
    the ``g`` the lemma's step is written in terms of.
    """
    parameters = _trainable_named_parameters(model)
    buffers = dict(model.named_buffers())
    tangents = {
        name: update.detach().to(parameter.device, parameter.dtype)
        for (name, parameter), update in zip(parameters.items(), updates)
    }

    def call_with_parameters(state: dict[str, torch.Tensor]) -> torch.Tensor:
        return functional_call(model, {**state, **buffers}, (x,))

    # Batch-norm updates its running buffers in place, and doing so inside a
    # functorch transform leaks a wrapped tensor into the module. Same guard
    # the projection solvers use.
    was_training = model.training
    model.eval()
    try:
        primal = {name: p.detach() for name, p in parameters.items()}
        _, jacobian_vector = jvp(call_with_parameters, (primal,), (tangents,))
    finally:
        model.train(was_training)
        _clear_inaccessible_tensor_caches(model)
    return jacobian_vector.detach()


def linearization_defect(
    model: torch.nn.Module,
    x: torch.Tensor,
    updates: tuple[torch.Tensor, ...],
    learning_rate: float,
    displacement: torch.Tensor | None = None,
) -> float:
    """``||f(theta - eta u) - (f(theta) - eta g)|| / (eta ||g||)``.

    Zero when the parameter step realises the function-space step exactly;
    it grows with the second-order remainder. Returns ``inf`` when the
    measurement is degenerate (a non-finite forward pass, or ``g = 0``), so
    the caller treats "cannot tell" as "not admissible" rather than as a
    pass.
    """
    if learning_rate <= 0.0:
        return float("inf")
    if displacement is None:
        displacement = predicted_displacement(model, x, updates)

    displacement_norm = float(torch.linalg.vector_norm(displacement))
    if not displacement_norm > 0.0 or displacement_norm != displacement_norm:
        return float("inf")

    parameters = _trainable_named_parameters(model)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            base = model(x).detach()
            originals = [p.detach().clone() for p in parameters.values()]
            for parameter, update in zip(parameters.values(), updates):
                parameter -= learning_rate * update.detach().to(
                    parameter.device, parameter.dtype
                )
            moved = model(x).detach()
            for parameter, original in zip(parameters.values(), originals):
                parameter.copy_(original)
    finally:
        model.train(was_training)
        _clear_inaccessible_tensor_caches(model)

    if not torch.isfinite(moved).all():
        return float("inf")
    residual = moved - (base - learning_rate * displacement)
    defect = float(torch.linalg.vector_norm(residual)) / (
        learning_rate * displacement_norm
    )
    if defect != defect:  # NaN guard
        return float("inf")
    return defect


def certified_linear_learning_rate(
    model: torch.nn.Module,
    x: torch.Tensor,
    updates: tuple[torch.Tensor, ...],
    certified_learning_rate: float,
    config: FGDApproxConfig,
) -> LinearizedStep:
    """Narrow the certified rate until the linearisation actually holds.

    Backtracks by ``lr_backtrack`` from ``certified_learning_rate`` -- which
    is ``theory_lr_safety * eta_bar(eps)``, the rate the certificate
    licenses -- and stops at the first ``eta`` whose defect is within
    ``certify_linearization_tolerance``. The search only ever moves DOWN, so
    every rate it can return was already certified; the tolerance decides
    how much of the interval is usable, never whether the interval applies.

    Returns ``learning_rate = None`` when the floor ``theory_lr_min`` is
    reached without the defect coming into tolerance. That is a meaningful
    outcome, not a failure to report: it says no admissible rate puts this
    direction inside the regime Lemma 3.5 describes, so the structure -- not
    the step size -- is what has to change.
    """
    tolerance = config.certify_linearization_tolerance
    displacement = predicted_displacement(model, x, updates)
    backtrack = min(max(config.lr_backtrack, 1e-6), 1.0 - 1e-9)

    learning_rate = certified_learning_rate
    backtracks = 0
    defect = float("inf")
    while learning_rate > config.theory_lr_min + config.eps:
        defect = linearization_defect(
            model, x, updates, learning_rate, displacement
        )
        if defect <= tolerance:
            return LinearizedStep(
                learning_rate=learning_rate,
                defect=defect,
                backtracks=backtracks,
                certified_learning_rate=certified_learning_rate,
            )
        learning_rate *= backtrack
        backtracks += 1

    return LinearizedStep(
        learning_rate=None,
        defect=defect,
        backtracks=backtracks,
        certified_learning_rate=certified_learning_rate,
    )
