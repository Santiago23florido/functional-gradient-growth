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

from fgdlib.tangent import (
    FGDApproxConfig,
    mse_functional_gradient,
    theoretical_learning_rate_upper_bound,
)

__all__ = [
    "ParametricFamilyResult",
    "certify_parametric_step",
    "certify_parametric_step_swept",
    "family_lemma35_rate",
]


def family_lemma35_rate(
    relative_error: float,
    config: FGDApproxConfig,
) -> float | None:
    """Lemma 3.5's admissible rate for the FAMILY's own relative error.

    Mirrors ``pipeline.lemma35_learning_rate`` exactly -- same interval
    ``eta_bar = 2(1 - 2 eps)/(L_s(1 + 2 eps))``, same ``theory_lr_safety``
    fraction, same ``theory_lr_min`` floor -- and lives here only because
    ``pipeline`` imports this module, so the dependency cannot run the other
    way. It is NOT a new bound: the family already measures the exact
    quantity the lemma consumes, ``RelErr = sqrt(1 - cos^2(Delta, r))``, so
    the lemma applies to its displacement verbatim.

    Why this matters: the family used to apply its displacement WHOLE. That
    is a functional step of size 1, and MEASURED against its own certified
    eps the lemma allows far less -- eps 0.292 (cos 0.9564) gives
    eta_bar = 0.263 at L_s = 2, so a full step overshoots by 3.8x, and at
    eps 0.400 by 9x. Overshooting a descent bound by that factor is exactly
    what made MNIST oscillate while its loss stayed flat: the direction was
    certified, the distance was not.

    Returns ``None`` when no rate sits strictly inside the interval, which
    is the signal that this family cannot deliver a certified step here.
    """
    upper_bound = theoretical_learning_rate_upper_bound(relative_error, config)
    if upper_bound is None:
        return None
    fraction = 0.5 if config.certify_optimal_rate else config.theory_lr_safety
    rate = fraction * upper_bound
    if rate <= config.theory_lr_min + config.eps:
        return None
    return rate


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
    #: The eta_f whose target produced this displacement. Only interesting
    #: under the sweep, where it says WHICH distance certified.
    functional_learning_rate: float | None = None


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
        return ParametricFamilyResult(
            float("inf"), 0.0, False, None, functional_learning_rate
        )

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
                return ParametricFamilyResult(
            float("inf"), 0.0, False, None, functional_learning_rate
        )
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
        return ParametricFamilyResult(
            float("inf"), 0.0, False, None, functional_learning_rate
        )

    cosine = float(
        torch.sum(displacement * r) / (displacement_norm * residual_norm)
    )
    # RelErr at the scale-optimal step; negative cosine means the move points
    # AWAY from the gradient and can never certify, so clamp its contribution.
    relative_error = math.sqrt(max(0.0, 1.0 - max(cosine, 0.0) ** 2))
    certified = cosine > 0.0 and relative_error < threshold

    # Certifying the DIRECTION does not license the DISTANCE. The clone is
    # trained toward a full functional step, so committing it whole takes
    # eta = 1 -- and Lemma 3.5, applied to the family's OWN eps, allows
    # 0.95 * 2(1-2 eps)/(L_s(1+2 eps)), which at the measured eps 0.292 is
    # 0.263. Scale the parameter displacement to that certified rate instead
    # of replacing the model outright. Alignment is scale-invariant, so the
    # certificate above is untouched: the clone still trains to its best
    # cosine and only the committed distance changes. (Shrinking the TARGET
    # instead was tried and MEASURED to be wrong -- it starves the clone's
    # alignment, N=1024 test 0.921 -> 0.841 with certifications 27 -> 3.)
    if certified and getattr(config, "certify_family_lemma35_rate", False):
        rate = family_lemma35_rate(relative_error, config)
        if rate is None:
            certified = False
        else:
            with torch.no_grad():
                for base, moved in zip(model.parameters(), clone.parameters()):
                    moved.mul_(rate).add_(base, alpha=1.0 - rate)

    return ParametricFamilyResult(
        relative_error=relative_error,
        cosine=cosine,
        certified=certified,
        model=clone if certified else None,
        functional_learning_rate=functional_learning_rate,
    )


def certify_parametric_step_swept(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
    functional_learning_rates,
    inner_steps: int = 500,
    inner_learning_rate: float = 0.003,
    progress=None,
) -> ParametricFamilyResult:
    """Try several functional step sizes; keep the FIRST that certifies.

    ``certify_parametric_step`` asks one question -- "can a clone realise the
    target ``f - eta_f r`` well enough to certify?" -- at one ``eta_f``. A "no"
    there is not a statement about the family, it is a statement about that
    distance, and the ladder was recording it as the former. ``parametric_gd``
    has always walked a list of functional rates for exactly this reason; the
    tangent ladder never did.

    More search at the SAME bar. Each ``eta_f`` is certified independently by
    ``certify_parametric_step``, which measures that candidate's own
    ``RelErr(Delta, r)`` on the same probe against the same
    ``min(rel_error_threshold, 1/2)``. Nothing is relaxed, nothing is shared
    between candidates, and a swept run only ever accepts steps a single-eta
    run would also have accepted had it happened to try that value.

    The candidates are walked in the order given -- largest first is the
    intended use, so the accepted step is the LONGEST certified one -- and the
    walk stops at the first certificate, so a run that certifies at the head
    of the list costs exactly what it costs today.

    Returns the certified result, or the LAST failure when none certifies (the
    caller only needs ``certified``; the last failure is kept so the reported
    ``relative_error`` belongs to a real attempt rather than a sentinel).
    """
    rates = tuple(float(rate) for rate in functional_learning_rates or ())
    if not rates:
        return certify_parametric_step(
            model,
            x,
            y,
            config,
            functional_learning_rate=config.certify_family_functional_lr,
            inner_steps=inner_steps,
            inner_learning_rate=inner_learning_rate,
        )

    result = None
    for rate in rates:
        result = certify_parametric_step(
            model,
            x,
            y,
            config,
            functional_learning_rate=rate,
            inner_steps=inner_steps,
            inner_learning_rate=inner_learning_rate,
        )
        if result.certified:
            if progress is not None:
                progress(
                    f"[FAMILY] eta_f={rate:g} certified "
                    f"(cos {result.cosine:.4f}, eps {result.relative_error:.4f})"
                )
            return result
        if progress is not None:
            progress(
                f"[FAMILY] eta_f={rate:g} did not certify "
                f"(cos {result.cosine:.4f}, eps {result.relative_error:.4f})"
            )
    return result
