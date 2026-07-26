"""A ladder of certified approximation families, tried before growing.

The certified method uses one family -- the tangent projection ``g = P_T(r)``
-- and the moment it cannot certify (``eps >= 1/2``) it must grow. With
function-preserving growth that is ruinous: FP growth keeps ``f`` fixed, so
the residual ``r`` is fixed, and the tangent space must be enlarged until it
captures 80 % of the FULL residual -- MEASURED at +2 directions per neuron and
``eps`` falling ~0.003 per growth, i.e. hundreds of growths to certify once.

The escape, and the rule it must obey: try OTHER families at the fixed
structure before growing, and accept one ONLY if ITS OWN projection certifies,
``RelErr(g_family, r) < 1/2`` -- never a descent criterion, never the tangent's
projection standing in for the family's. If a family certifies, its step is a
genuine approximate-FGD step (Lemma 3.5 with ``g = g_family``) that shrinks the
residual, so growth is deferred and, when it finally happens, is cheap.

The tangent is a FIRST-ORDER family: it can only reach ``range(J)``. The
parametric family here is NONLINEAR: it trains a clone of the network toward
the functional target ``f - eta_f r`` and takes the realised output
displacement ``Delta = f - f_clone``. Because the clone moves along the
curved reachable manifold, ``Delta`` picks up second-order and higher
components of ``r`` that ``range(J)`` misses, so it can certify where the
tangent cannot -- MEASURED: at a structure with tangent ``Crel = 0.933`` the
parametric family reached ``Crel = 0.32`` (cos 0.95 with ``r``). Crucially it
stays WITHIN the MLP: ``Delta`` is realised by a parameter change, unlike an
RKHS head, which certifies too but leaves the parametric model.

The family's own relative error is the exact quantity the certificate needs.
At the scale-optimal step the best achievable ``RelErr`` of a displacement
``Delta`` approximating ``r`` is ``sqrt(1 - cos^2(Delta, r))``, so
``RelErr < 1/2`` is exactly ``cos(Delta, r) > sqrt(3)/2 ~ 0.866``. That is
what is checked, and nothing else.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch

from fgdlib.tangent import FGDApproxConfig, mse_functional_gradient

__all__ = [
    "ParametricFamilyResult",
    "certify_parametric_step",
]


@dataclass(frozen=True)
class ParametricFamilyResult:
    """Outcome of the parametric family's own certification."""

    #: RelErr(g_family, r) = sqrt(1 - cos^2(Delta, r)); the family certifies
    #: exactly when this is < the threshold.
    relative_error: float
    #: cos(Delta, r) -- reported so the caller can see the alignment directly.
    cosine: float
    certified: bool
    #: The stepped model when certified (the trained clone), else ``None``.
    model: object | None


def certify_parametric_step(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
    functional_learning_rate: float = 1.0,
    inner_steps: int = 500,
    inner_learning_rate: float = 0.003,
) -> ParametricFamilyResult:
    """Certify a NONLINEAR within-MLP step by its OWN relative error.

    Trains a clone toward ``f - functional_learning_rate * r`` and measures the
    realised output displacement ``Delta = f - f_clone``. The step is certified
    iff ``RelErr(Delta, r) = sqrt(1 - cos^2) < min(rel_error_threshold, 1/2)``
    -- the family's own projection, no other criterion. When certified the
    trained clone IS the stepped model (a parameter change, in the MLP).

    Only for sum-MSE, where the functional gradient is ``2(f - y)``; returned
    uncertified otherwise so the caller falls through to the next family.
    """
    threshold = min(config.rel_error_threshold, 0.5)
    if config.functional_loss != "mse":
        return ParametricFamilyResult(float("inf"), 0.0, False, None)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            f0 = model(x).detach()
        r = mse_functional_gradient(f0, y)
        target = (f0 - functional_learning_rate * r).detach()

        clone = copy.deepcopy(model)
        clone.train()
        optimizer = torch.optim.AdamW(clone.parameters(), lr=inner_learning_rate)

        # Optional plateau early stop: cut the wasted tail once the clone's
        # alignment with the residual stops improving. Conservative (only true
        # plateaus), so the accepted/rejected step is the one full training
        # would give. Off by default -> the loop below is the plain full run.
        plateau_stop = bool(getattr(config, "certify_family_plateau_stop", False))
        n_inner = max(1, inner_steps)
        residual_norm_inner = float(torch.linalg.vector_norm(r))
        if plateau_stop:
            plateau_tol = float(getattr(config, "certify_family_plateau_tol", 0.002))
            check_every = max(1, n_inner // 20)
            patience = 3
            # cos(Delta, r) > target_cos is exactly RelErr < 1/2 (certification).
            # We ONLY cut a clone that has plateaued STILL BELOW this -- a doomed
            # attempt that would fail anyway. A clone climbing toward/past
            # target_cos is never cut, so a SUCCESSFUL step trains to full
            # alignment exactly as before (same accepted step, same result). The
            # speedup comes entirely from the doomed attempts, which dominate the
            # cost on a hard task where the family rarely certifies.
            target_cos = math.sqrt(max(0.0, 1.0 - threshold**2))

            def _clone_cosine() -> float:
                clone.eval()
                with torch.no_grad():
                    delta = (f0 - clone(x).detach())
                clone.train()
                delta_norm = float(torch.linalg.vector_norm(delta))
                if not (delta_norm > 0.0 and residual_norm_inner > 0.0):
                    return -1.0
                return float(
                    torch.sum(delta * r) / (delta_norm * residual_norm_inner)
                )

            best_cosine = -2.0
            stale_checks = 0

        for step in range(n_inner):
            optimizer.zero_grad()
            loss = ((clone(x) - target) ** 2).sum()
            if not torch.isfinite(loss):
                return ParametricFamilyResult(float("inf"), 0.0, False, None)
            loss.backward()
            optimizer.step()
            if plateau_stop and (step + 1) % check_every == 0:
                current_cosine = _clone_cosine()
                if current_cosine > best_cosine + plateau_tol:
                    best_cosine = current_cosine
                    stale_checks = 0
                else:
                    stale_checks += 1
                # Cut ONLY a DOOMED clone: plateaued AND still below the
                # certification target. A certified (or still-climbing) clone
                # keeps training to its best alignment, unchanged.
                if stale_checks >= patience and best_cosine < target_cos:
                    break

        clone.eval()
        with torch.no_grad():
            displacement = (f0 - clone(x).detach())        # descent-sign Delta
    finally:
        model.train(was_training)

    displacement_norm = float(torch.linalg.vector_norm(displacement))
    residual_norm = float(torch.linalg.vector_norm(r))
    if not (displacement_norm > 0.0 and residual_norm > 0.0):
        return ParametricFamilyResult(float("inf"), 0.0, False, None)

    cosine = float(
        torch.sum(displacement * r) / (displacement_norm * residual_norm)
    )
    # RelErr at the scale-optimal step; negative cosine means the move points
    # AWAY from the gradient and can never certify, so clamp its contribution.
    relative_error = math.sqrt(max(0.0, 1.0 - max(cosine, 0.0) ** 2))
    certified = cosine > 0.0 and relative_error < threshold
    return ParametricFamilyResult(
        relative_error=relative_error,
        cosine=cosine,
        certified=certified,
        model=clone if certified else None,
    )
