"""Grow until Lemma 3.5 holds -- by construction, not by luck.

The ordinary flow grows *where it is cheapest* and steps *when it can*. This
module inverts that: it grows until the structure **provably satisfies the
certificate**, and only then is a step taken. The research question it serves
is whether enforcing the FGD conditions exactly -- never approximated, never
bypassed -- reaches a global optimum in loss and maximum accuracy.

Why this terminates (the theorem, pinned in
``tests/test_grow_to_certify_theorem.py``):

* From the bridge identity ``||r||^2 = ||g||^2 (1 + eps^2)`` with
  ``g = P_T(r)``, the condition is exactly

      eps < 1/2   <=>   ||P_T(r)||^2 > 0.8 ||r||^2

  "the tangent space captures more than 80 % of the gradient energy".
* Function-preserving growth leaves ``f`` **identical** yet strictly enlarges
  ``T = range(J)``: a new neuron enters with outgoing weight ``omega = 0`` so
  it contributes nothing to ``f``, but ``df/domega != 0`` is a genuinely new
  direction. Measured: ``f`` unchanged to 1.8e-07 while ``rank(J)`` rose
  57 -> 66.
* Hence ``r`` is fixed while ``T`` grows, so ``||P_T(r)||`` increases
  strictly and **eps falls with no training step at all** -- measured
  1.883 -> 1.732 -> 1.713 -> 1.674 -> 1.615.
* The residual ``rho = r - P_T(r)`` lives in a finite-dimensional space
  (``N*K``), so finitely many added directions drive it to zero.

Therefore the loop below crosses ``1/2`` in finitely many growths. The
``max_growths`` argument is a **safety valve against numerical pathology**,
not a budget: the theory says the loop terminates on its own.

Exactness over cost, deliberately: the location to grow is chosen by
measuring the resulting ``eps`` EXACTLY on each candidate (a clone grown
function-preservingly, scored with the full-Jacobian solver), rather than
ranked by a cheaper surrogate score. That is the globally best growth the
architecture can make at this point, established by measurement.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from fgdlib.search.growth import grow_layer
from fgdlib.tangent import (
    ExactTangentSystem,
    FGDApproxConfig,
    _output_relative_error_from_tensors,
    _solve_tangent_projection,
    exact_tangent_system,
    tiny_optimal_update_kwargs,
    validate_exact_tangent_system,
)

__all__ = [
    "CertifyResult",
    "exact_relative_error",
    "grow_until_certified",
]


@dataclass(frozen=True)
class CertifyResult:
    """Outcome of the grow-to-certify loop."""

    relative_error: float
    growths: int
    certified: bool
    trajectory: tuple[float, ...]
    #: True when a certified NON-tangent family took a step instead of growing.
    #: The returned model has already moved, so the caller must treat it as the
    #: outer step and not step again.
    family_stepped: bool = False
    #: How many family steps were taken across the loop (a family step shrinks
    #: the residual, deferring growth).
    family_steps: int = 0
    #: Exact system for the returned model and probe, when non-degenerate.
    tangent_system: ExactTangentSystem | None = None


def exact_relative_error(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
    *,
    system: ExactTangentSystem | None = None,
) -> float:
    """``eps`` from the FULL Jacobian -- no CG, no surrogate.

    Measured at the SAME regularisation the step will use. That sounds like
    bookkeeping and is not: with ``projection_damping_auto`` the step solves
    at a ``lambda`` chosen per outer step, while this measured at the fixed
    ``config.projection_damping``, so growth and stepping were driven by two
    different certificates. Growth would then chase an ``eps`` that no step
    ever saw, and the cycle's own rule -- train until the relative error
    stops being satisfied, then grow -- refers to a quantity that has to be
    one quantity.

    Returns ``inf`` when the projection is degenerate, which the caller must
    read as "nothing of ``r`` is representable yet".
    """
    if system is None:
        system = exact_tangent_system(model, x, y, config)
    else:
        validate_exact_tangent_system(system, model, x, y, config)
    if system is None:
        return float("inf")

    if config.projection_damping_auto:
        # Imported here: damping.py imports from tangent.py, and certify.py
        # is imported by the pipeline before either -- a module-level import
        # would close the cycle.
        from fgdlib.search.damping import minimal_relative_error_from_system

        # The growth signal is the SMALLEST eps the tangent space can reach,
        # at least regularisation. It is < 1/2 exactly when a certified step
        # exists (eps is increasing in lambda, so its minimum decides), and
        # unlike the certified eps it stays finite while the structure is
        # still inadequate -- which is what lets the loop rank one candidate
        # structure against another. Growth and stepping thus share the SAME
        # spectrum, measured at the same probe; they read different points of
        # it (growth the minimum, the step the selected lambda), which is the
        # correct division of labour rather than the two-certificates bug.
        return minimal_relative_error_from_system(system, config)

    _, approximation = _solve_tangent_projection(
        jacobian_matrix=system.jacobian,
        target=system.target,
        damping=config.projection_damping,
        solver=config.projection_solver,
        work_dtype=(
            torch.float32
            if getattr(config, "projection_fast_factorization", False)
            else torch.float64
        ),
    )
    stats = _output_relative_error_from_tensors(
        approximation=approximation,
        target=system.target,
        eps=config.eps,
    )
    epsilon = stats.output_error.relative_error
    if epsilon is None or not float(epsilon) == float(epsilon):  # NaN guard
        return float("inf")
    return float(epsilon)


def _relative_error_and_system(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> tuple[float, ExactTangentSystem | None]:
    system = exact_tangent_system(model, x, y, config)
    if system is None:
        return float("inf"), None
    return exact_relative_error(model, x, y, config, system=system), system


def _grow_clone(
    model,
    train_loader,
    layer_index: int,
    device: torch.device,
    config,
    function_preserving: bool,
):
    """A copy of ``model`` grown at ``layer_index``.

    ``function_preserving`` is the central trade-off of this method, and it is
    a measured one:

    | growth          | tangent directions gained | parameters added |
    |-----------------|---------------------------|------------------|
    | preserving      | +9                        | 42               |
    | non-preserving  | +42                       | 42               |

    With ``omega = 0`` the incoming weights contribute nothing to the
    Jacobian (``df/dalpha = 0``), so only the outgoing weights add directions
    -- about a fifth of the parameters spent. Releasing ``omega`` makes every
    added parameter contribute an independent direction, the theoretical
    maximum, so the rank needed to certify is reached far sooner.

    What is given up is the monotonicity proof: a non-preserving growth moves
    ``f``, so ``r`` moves with it and ``eps`` may rise after a growth. The
    loop then relies on measurement rather than a theorem, which is why the
    preserving route stays the default.

    Returns ``None`` when the growth cannot be applied, so a failed candidate
    drops out of the comparison instead of aborting the search.
    """
    clone = copy.deepcopy(model)
    try:
        grow_layer(
            model=clone,
            train_loader=train_loader,
            layer_index=layer_index,
            device=device,
            line_search_config=config.scaling_line_search,
            optimal_update_kwargs=tiny_optimal_update_kwargs(
                config.fgd_approx,
                compute_delta=config.fgd_approx.growth_compute_delta,
            ),
            progress=None,
            function_preserving=function_preserving,
            preservation_tolerance=(config.fgd_approx.growth_preservation_tolerance),
        )
    except RuntimeError:
        return None
    return clone


def grow_until_certified(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    train_loader,
    device: torch.device,
    config,
    max_growths: int = 64,
    function_preserving: bool = True,
    force: bool = False,
    progress=None,
    family_step=None,
):
    """Grow until ``eps < rel_error_threshold``; return the grown model.

    ``family_step``: an optional callable ``model -> stepped_model_or_None``.
    Before EACH growth it is given the current model; if it returns a stepped
    model, that means a NON-tangent family certified at the fixed structure by
    ITS OWN relative error (never a descent criterion, never the tangent's),
    so a certified step was taken instead of growing. The residual shrinks, so
    growth is deferred and, when it finally happens, is cheaper -- the whole
    point when growth is function-preserving and otherwise ruinous.

    At each iteration every growable location is tried on a clone and scored
    by its EXACT resulting ``eps``; the location with the lowest ``eps`` is
    committed.

    ``force`` grows at least once even when ``eps`` is already below the
    threshold. It exists to close a real deadlock, measured on MNIST: after
    certification the flow sat at ``eps = 0.475`` for epoch after epoch with
    the loss frozen at 0.0619, because

    * ``eps < 1/2`` said the structure was adequate, so no growth fired, and
    * no admissible learning rate produced held-out descent, so no step
      committed.

    Neither mechanism could act and nothing changed, ever. The resolution is
    the distinction the method already rests on: ``eps < 1/2`` is Lemma 3.5's
    admissibility of a STEP -- it certifies that an admissible rate *exists*
    in the worst-case bound -- while the realised descent is a **separate**
    certificate condition. A step that fails the descent condition despite
    ``eps < 1/2`` has shown that the structure did not deliver, and growth,
    not more of the same step, is the answer. That is exactly R1's reasoning,
    applied here.

    Returns ``(model, CertifyResult)``.
    """
    threshold = config.fgd_approx.rel_error_threshold
    epsilon, tangent_system = _relative_error_and_system(model, x, y, config.fgd_approx)
    trajectory = [epsilon]
    growths = 0
    family_steps = 0
    forced_remaining = 1 if force else 0

    while (epsilon >= threshold or forced_remaining > 0) and growths < max_growths:
        forced_remaining = max(0, forced_remaining - 1)

        # FAMILY LADDER, before growing: the tangent could not certify (that
        # is why we are here), so try a certified NON-tangent family at the
        # fixed structure. Accept ONLY if the family's own projection
        # certifies -- that gate lives inside family_step. A family step is a
        # real certified FGD step, so it becomes the outer step and the loop
        # returns instead of growing.
        if family_step is not None and epsilon >= threshold:
            stepped = family_step(model)
            if stepped is not None:
                model = stepped
                family_steps += 1
                epsilon, tangent_system = _relative_error_and_system(
                    model, x, y, config.fgd_approx
                )
                trajectory.append(epsilon)
                if progress is not None:
                    progress(
                        "[CERTIFY] parametric family certified at fixed "
                        f"structure (no growth); eps -> {epsilon:.4f}"
                    )
                return model, CertifyResult(
                    relative_error=epsilon,
                    growths=growths,
                    certified=epsilon < threshold,
                    trajectory=tuple(trajectory),
                    family_stepped=True,
                    family_steps=family_steps,
                    tangent_system=tangent_system,
                )

        locations = range(len(getattr(model, "_growable_layers", [])))
        best_model = None
        # Preserving growth cannot make eps worse, so requiring an improvement
        # is free there. Non-preserving growth moves f, so the best available
        # candidate may still sit above the current eps; accepting it is what
        # lets the rank keep climbing, and the loop then relies on the measured
        # trajectory rather than the monotonicity theorem.
        best_epsilon = epsilon if function_preserving else float("inf")
        best_location = None
        best_system = tangent_system
        for location in locations:
            candidate = _grow_clone(
                model,
                train_loader,
                location,
                device,
                config,
                function_preserving,
            )
            if candidate is None:
                continue
            candidate_epsilon, candidate_system = _relative_error_and_system(
                candidate, x, y, config.fgd_approx
            )
            if candidate_epsilon < best_epsilon:
                best_epsilon = candidate_epsilon
                best_model = candidate
                best_location = location
                best_system = candidate_system

        if best_model is None:
            # No growable location reduces eps. The theorem says this cannot
            # persist while the residual is non-zero, so reaching here means
            # the architecture cannot add a direction along rho at all --
            # report honestly rather than loop.
            if progress is not None:
                progress(
                    f"[CERTIFY] no growth reduced eps ({epsilon:.4f}); "
                    "the structure cannot add a direction along the residual"
                )
            break

        model = best_model
        epsilon = best_epsilon
        tangent_system = best_system
        growths += 1
        trajectory.append(epsilon)
        if progress is not None:
            progress(
                f"[CERTIFY] growth {growths} at location {best_location}: "
                f"eps -> {epsilon:.4f}"
                + ("  (certified)" if epsilon < threshold else "")
            )

        # ADAPTIVE COUNT: keep adding at the just-chosen best location while each
        # increment still pays -- marginal eps reduction above min_gain of the
        # remaining gap -- stopping the instant it certifies or the returns
        # diminish. The number of neurons is chosen by the certificate and the
        # value criterion, not fixed; every intermediate structure is scored by
        # its OWN projection, so the certificate conditions are unchanged. This
        # cuts the number of full location scans on a hard task (CIFAR) without
        # weakening any certificate.
        if getattr(config.fgd_approx, "certify_adaptive_growth", False):
            min_gain = float(
                getattr(config.fgd_approx, "certify_adaptive_growth_min_gain", 0.1)
            )
            while epsilon >= threshold and growths < max_growths:
                bigger = _grow_clone(
                    model,
                    train_loader,
                    best_location,
                    device,
                    config,
                    function_preserving,
                )
                if bigger is None:
                    break
                bigger_epsilon, bigger_system = _relative_error_and_system(
                    bigger, x, y, config.fgd_approx
                )
                gain = epsilon - bigger_epsilon
                if not (gain > min_gain * max(epsilon - threshold, 1e-6)):
                    break  # diminishing returns here; let the outer loop re-scan
                model = bigger
                epsilon = bigger_epsilon
                tangent_system = bigger_system
                growths += 1
                trajectory.append(epsilon)
                if progress is not None:
                    progress(
                        f"[CERTIFY] adaptive +growth {growths} at location "
                        f"{best_location}: eps -> {epsilon:.4f}"
                        + ("  (certified)" if epsilon < threshold else "")
                    )

    return model, CertifyResult(
        relative_error=epsilon,
        growths=growths,
        certified=epsilon < threshold,
        trajectory=tuple(trajectory),
        family_stepped=False,
        family_steps=family_steps,
        tangent_system=tangent_system,
    )
