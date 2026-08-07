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
import os
from dataclasses import dataclass

import torch

from fgdlib.profile import fallback, increment, profiled
from fgdlib.search.exact_where import (
    CandidateScoringError,
    UnsupportedGrowthStructure,
    exact_block_schur_score,
    factor_shared_base,
    identify_growth_columns,
    stream_shared_candidate_statistics,
)
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
from fgdlib.training_utils.loop import count_parameters

__all__ = [
    "CertifyResult",
    "certify_growth_target",
    "exact_relative_error",
    "grow_until_certified",
]


def _exact_block_schur_where_enabled(config: FGDApproxConfig) -> bool:
    value = os.environ.get("FGD_EXACT_BLOCK_SCHUR_WHERE", "")
    return bool(config.certify_exact_block_schur_where) or value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class CertifyResult:
    """Outcome of the grow-to-certify loop."""

    relative_error: float
    growths: int
    #: eps reached the GROWTH TARGET, which is ``rel_error_threshold`` unless
    #: ``certify_growth_target`` asks for more. NOT the step certificate: a loop
    #: that stops inside the voluntary band returns False here while
    #: ``lemma35_learning_rate`` still yields a rate, so this must stay what it
    #: has always been -- a log line, never a control-flow gate.
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
    #: Why the loop returned: "certified", "growth_target_unreachable",
    #: "parameter_budget", "no_growth_reduced_epsilon", "max_growths" or
    #: "family_stepped". Diagnostic; nothing branches on it.
    stop_reason: str = "certified"
    #: The eps the loop was chasing -- ``certify_growth_target`` clamped to the
    #: certificate, i.e. ``rel_error_threshold`` when the field is unset.
    growth_target: float | None = None


def certify_growth_target(config: FGDApproxConfig) -> float:
    """The eps the grow loop CHASES -- never laxer than the certificate.

    ``rel_error_threshold`` remains the STEP certificate; this is the loop's
    aspiration, and the clamp is ``min(target, rel_error_threshold)`` and NOT
    ``min(target, 0.5)``. A threshold above 1/2 is a deliberate looseness for
    the LOOP (``tests/test_fast_factorization.py`` grows at 0.7 on purpose, to
    reach an ill-conditioned structure), while the STEP clamps itself at 1/2 at
    every site that reads Lemma 3.5; clamping here too would silently tighten
    those runs for no gain.

    ``None`` returns the threshold, so the loop's behaviour is unchanged by
    construction rather than by a flag: the invariant ``eps >= target`` then
    implies ``eps >= min(threshold, 1/2)``, the regime where growth is
    unconditional.
    """
    target = getattr(config, "certify_growth_target", None)
    threshold = float(config.rel_error_threshold)
    if target is None:
        return threshold
    return min(float(target), threshold)


def _growth_pays(
    *,
    epsilon: float,
    candidate_epsilon: float,
    growth_target: float,
    certificate: float,
    min_gain: float,
) -> bool:
    """Whether the best available growth is worth the parameters it costs.

    Two regimes, split exactly where Lemma 3.5 splits them:

    * ``epsilon >= certificate``: no step exists at ANY damping (the bisection
      in ``damping.py`` returns None when even the least regularised solve
      misses the line), so growth is the only move the method has. True
      unconditionally -- which is also why this gate cannot deadlock the loop:
      it never refuses while the structure is still inadequate, so the loop can
      never stop in a state where no step certifies.
    * ``certificate > epsilon >= growth_target``: a step is already available
      and the extra accuracy is being chased VOLUNTARILY. The growth then has
      to close at least ``min_gain`` of the gap that is left, or it is buying
      nothing for its parameters.

    ``epsilon`` carries the projection in the denominator
    (``eps = ||g - r|| / ||g||``), so it is NOT bounded above and nothing here
    may assume ``eps in [0, 1]``. The floor mirrors the adaptive-count idiom
    below: a fraction of the remaining gap, with a tiny floor so that a gap of
    exactly zero still demands a real improvement.
    """
    if not (epsilon < certificate):
        return True
    floor = min_gain * max(epsilon - growth_target, 1e-6)
    return (epsilon - candidate_epsilon) > floor


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

    if system.factors is not None and config.projection_damping_auto:
        # Factored system: the spectrum arrives as (W, V) with J = W V^T, and
        # svd(J) = (A, S, VB) follows from an SVD of the (rows, k) factor
        # alone. Same quantity, same least-damping point -- the parameter
        # dimension simply never enters. MEASURED against the materialised J:
        # singular values to 5.1e-08, eps to 1.7e-04. Only the matrix-free
        # family sets factors, so the default path never reaches this.
        from fgdlib.search.damping import DAMPING_BRACKET
        from fgdlib.search.mffactored import factored_minimal_relative_error

        return factored_minimal_relative_error(
            system, config, DAMPING_BRACKET[0]
        )

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


def _full_candidate_score(
    candidate,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> tuple[float, ExactTangentSystem | None]:
    """The unchanged full-Jacobian candidate oracle."""
    increment("where_full_candidate_system_calls")
    return _relative_error_and_system(candidate, x, y, config)


@profiled("where_total_seconds")
def _select_growth_candidate(
    model,
    tangent_system: ExactTangentSystem | None,
    x: torch.Tensor,
    y: torch.Tensor,
    train_loader,
    device: torch.device,
    config,
    function_preserving: bool,
    current_epsilon: float,
    *,
    parameter_budget: int | None = None,
) -> tuple[object | None, float, int | None, ExactTangentSystem | None, bool]:
    """Select the same exhaustive argmin, with exact shared-base scoring if enabled.

    ``parameter_budget`` prices each candidate by its POST-growth count, the
    idiom the end-of-epoch path already argues for (``pipeline.py``): checking
    the budget only before growing lets a single input-layer widening (784
    parameters per neuron on MNIST) blow straight through it. Unaffordable
    clones are dropped BEFORE scoring, which also saves the O(P^3) exact solve
    each score costs. The trailing flag reports "every candidate was rejected
    as unaffordable", so the caller can name the budget as the reason it
    stopped instead of blaming the structure.
    """
    increment("where_scans")
    candidates: list[tuple[int, object]] = []
    unaffordable = 0
    for location in range(len(getattr(model, "_growable_layers", []))):
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
        if parameter_budget is not None and (
            count_parameters(candidate) > parameter_budget
        ):
            unaffordable += 1
            continue
        candidates.append((location, candidate))
    if unaffordable:
        increment("certify_budget_rejected_candidates", unaffordable)
    budget_exhausted = unaffordable > 0 and not candidates
    increment("where_candidates", len(candidates))

    best_model = None
    best_epsilon = current_epsilon if function_preserving else float("inf")
    best_location = None
    best_system = tangent_system
    enabled = (
        _exact_block_schur_where_enabled(config.fgd_approx)
        and function_preserving
        and tangent_system is not None
        and bool(candidates)
    )
    if not enabled:
        for location, candidate in candidates:
            candidate_epsilon, candidate_system = _full_candidate_score(
                candidate, x, y, config.fgd_approx
            )
            if candidate_epsilon < best_epsilon:
                best_epsilon = candidate_epsilon
                best_model = candidate
                best_location = location
                best_system = candidate_system
        return (
            best_model,
            best_epsilon,
            best_location,
            best_system,
            budget_exhausted,
        )

    increment("where_base_system_reuses")
    parity_tolerance = (
        1e-5
        if getattr(config.fgd_approx, "projection_fast_factorization", False)
        else 1e-9
    )
    layouts = []
    supported: list[tuple[int, object]] = []
    full_results: dict[int, tuple[float, ExactTangentSystem | None]] = {}
    for location, candidate in candidates:
        try:
            layout = identify_growth_columns(
                model,
                candidate,
                x,
                preservation_tolerance=(
                    config.fgd_approx.growth_preservation_tolerance
                ),
            )
        except UnsupportedGrowthStructure:
            increment("where_full_fallbacks")
            increment("where_unsupported_structure_fallbacks")
            full_results[location] = _full_candidate_score(
                candidate, x, y, config.fgd_approx
            )
        else:
            layouts.append(layout)
            supported.append((location, candidate))

    fast_results: dict[int, tuple[float, float]] = {}
    if layouts:
        try:
            shared = stream_shared_candidate_statistics(
                model,
                tangent_system,
                tuple(layouts),
                x,
                y,
                config.fgd_approx,
            )
            spectrum = factor_shared_base(
                tangent_system,
                reference_joint_factor=shared.candidates[0].joint_factor,
            )
        except (CandidateScoringError, RuntimeError):
            increment("where_full_fallbacks")
            increment("where_numerical_fallbacks")
            for location, candidate in supported:
                full_results[location] = _full_candidate_score(
                    candidate, x, y, config.fgd_approx
                )
        else:
            for (location, candidate), statistics in zip(
                supported, shared.candidates
            ):
                try:
                    score = exact_block_schur_score(
                        shared,
                        statistics,
                        spectrum,
                        eps=config.fgd_approx.eps,
                    )
                except (CandidateScoringError, RuntimeError):
                    increment("where_full_fallbacks")
                    increment("where_numerical_fallbacks")
                    full_results[location] = _full_candidate_score(
                        candidate, x, y, config.fgd_approx
                    )
                else:
                    increment("where_fast_candidate_scores")
                    fast_results[location] = (
                        score.relative_error,
                        score.error_bound,
                    )

    # A numerical interval that overlaps the provisional argmin is resolved by
    # the unchanged full oracle.  Strict enumeration order remains the tie-break.
    provisional = {
        location: (
            full_results[location][0]
            if location in full_results
            else fast_results[location][0]
        )
        for location, _ in candidates
    }
    if provisional:
        provisional_location = min(provisional, key=provisional.get)
        provisional_value = provisional[provisional_location]
        provisional_bound = fast_results.get(provisional_location, (0.0, 0.0))[1]
        for location, candidate in candidates:
            if location not in fast_results or location == provisional_location:
                continue
            value, bound = fast_results[location]
            if abs(value - provisional_value) <= (
                bound + provisional_bound + 2.0 * parity_tolerance
            ):
                increment("where_full_fallbacks")
                increment("where_ambiguous_fallbacks")
                full_results[location] = _full_candidate_score(
                    candidate, x, y, config.fgd_approx
                )

    for location, candidate in candidates:
        candidate_epsilon = (
            full_results[location][0]
            if location in full_results
            else fast_results[location][0]
        )
        if candidate_epsilon < best_epsilon:
            best_epsilon = candidate_epsilon
            best_model = candidate
            best_location = location
            best_system = (
                full_results[location][1] if location in full_results else None
            )

    if best_model is None:
        return None, best_epsilon, None, tangent_system, budget_exhausted
    if best_system is not None:
        return (
            best_model,
            best_epsilon,
            best_location,
            best_system,
            budget_exhausted,
        )

    # Required canonical validation: the selected grown model gets a complete
    # exact system before damping selection or realization can reuse it.
    increment("where_final_winner_full_validations")
    validated_epsilon, validated_system = _full_candidate_score(
        best_model, x, y, config.fgd_approx
    )
    if abs(validated_epsilon - best_epsilon) <= parity_tolerance:
        return (
            best_model,
            validated_epsilon,
            best_location,
            validated_system,
            budget_exhausted,
        )

    # The shared-base prediction did not match its full candidate oracle.
    # Discard every fast result and rerun the original exhaustive selection.
    increment("where_full_fallbacks")
    increment("where_winner_mismatch_fallbacks")
    if best_location is not None:
        full_results[best_location] = (validated_epsilon, validated_system)
    best_model = None
    best_epsilon = current_epsilon if function_preserving else float("inf")
    best_location = None
    best_system = tangent_system
    for location, candidate in candidates:
        if location in full_results:
            candidate_epsilon, candidate_system = full_results[location]
        else:
            candidate_epsilon, candidate_system = _full_candidate_score(
                candidate, x, y, config.fgd_approx
            )
        if candidate_epsilon < best_epsilon:
            best_epsilon = candidate_epsilon
            best_model = candidate
            best_location = location
            best_system = candidate_system
    return best_model, best_epsilon, best_location, best_system, budget_exhausted


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
    growth_warranted=None,
    layer_bottlenecks=None,
):
    """Grow until ``eps < certify_growth_target``; return the grown model.

    The target defaults to ``rel_error_threshold``, which is what this loop has
    always chased; ``certify_growth_target`` lets it aspire to a STRICTER eps
    without touching the threshold, because that threshold is simultaneously
    the step certificate and lowering it kills the step it is trying to earn
    (see ``FGDApproxConfig.certify_growth_target`` for the measured MNIST
    incident). Below the certificate the loop keeps growing only while each
    growth still pays -- ``_growth_pays``.

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
    # Three quantities, deliberately not one. ``certificate`` is Lemma 3.5's
    # admissibility of a STEP, clamped at 1/2 exactly as every step site clamps
    # it; ``target`` is what this loop CHASES and may be stricter; ``threshold``
    # is the raw configured value, which the adaptive-count block below keeps
    # using so that certify_adaptive_growth's meaning -- burst while no step
    # exists -- is left untouched by the split, and so that its bursts never
    # spend parameters inside the voluntary band behind _growth_pays's back.
    threshold = config.fgd_approx.rel_error_threshold
    certificate = min(threshold, 0.5)
    target = certify_growth_target(config.fgd_approx)
    # Named apart from the adaptive-count block's own min_gain, which rebinds
    # its local inside the loop body.
    growth_min_gain = float(
        getattr(config.fgd_approx, "certify_growth_min_gain", 0.1)
    )
    patience = max(
        1, int(getattr(config.fgd_approx, "certify_growth_min_gain_patience", 1))
    )
    parameter_budget = getattr(config.fgd_approx, "max_total_parameters", None)
    epsilon, tangent_system = _relative_error_and_system(model, x, y, config.fgd_approx)
    trajectory = [epsilon]
    growths = 0
    family_steps = 0
    forced_remaining = 1 if force else 0
    unpaid_growths = 0
    stop_reason: str | None = None

    # IS IT GROWTH'S TURN AT ALL? Asked ONCE per invocation, before the loop,
    # not before each growth -- the question is "grow now or train now", and
    # the answer only changes after something has moved. That placement is
    # what restores the interleave: a "no" returns immediately, the step
    # commits, training happens, and the NEXT outer step asks again. Asked
    # per growth instead it would double the cost of an already exhaustive
    # scan and still front-load, because nothing between two growths trains.
    #
    # The predicate itself is organic, which is the point: it does not test
    # eps against a constant -- on MNIST every growth moves eps ~1e-3, so any
    # fraction of any gap refuses all of them equally, and a parameter cap is
    # just another number pulled from the air. It trains a grown clone and a
    # stay clone the SAME number of steps and asks which reaches the lower
    # eps, i.e. whether the structure or the training is the binding
    # constraint. The scale it compares against is the problem's own, so it
    # transfers between datasets without a knob.
    if growth_warranted is not None and epsilon < certificate:
        if not growth_warranted(model):
            fallback("certify_growth_not_warranted", "training_beats_growing")
            if progress is not None:
                progress(
                    f"[CERTIFY] eps {epsilon:.4f} certifies and training "
                    "reaches a lower eps than growing would; stepping "
                    "instead of growing"
                )
            return model, CertifyResult(
                relative_error=epsilon,
                growths=0,
                certified=epsilon < target,
                trajectory=tuple(trajectory),
                tangent_system=tangent_system,
                stop_reason="training_beats_growing",
                growth_target=target,
            )

    # The budget exists (tangent.py) and is honoured at the end of each epoch,
    # but was invisible to the loop that spends the most -- this one. Checked
    # here before the first scan, because an exhausted budget makes every
    # candidate unaffordable and one exhaustive scan costs a full exact solve
    # per location.
    if parameter_budget is not None and count_parameters(model) >= parameter_budget:
        increment("certify_budget_stops")
        if progress is not None:
            progress(
                f"[CERTIFY] parameter budget {parameter_budget} already spent "
                f"({count_parameters(model)} params); no growth attempted at "
                f"eps {epsilon:.4f}"
            )
        return model, CertifyResult(
            relative_error=epsilon,
            growths=0,
            certified=epsilon < target,
            trajectory=tuple(trajectory),
            family_stepped=False,
            family_steps=0,
            tangent_system=tangent_system,
            stop_reason="parameter_budget",
            growth_target=target,
        )

    while (epsilon >= target or forced_remaining > 0) and growths < max_growths:
        # Captured BEFORE the decrement: a forced growth is the answer to a
        # measured deadlock and must not be vetoed by the value criterion,
        # which would restore the deadlock it was added to break.
        is_forced = forced_remaining > 0
        forced_remaining = max(0, forced_remaining - 1)

        # FAMILY LADDER, before growing: the tangent could not certify (that
        # is why we are here), so try a certified NON-tangent family at the
        # fixed structure. Accept ONLY if the family's own projection
        # certifies -- that gate lives inside family_step. A family step is a
        # real certified FGD step, so it becomes the outer step and the loop
        # returns instead of growing.
        #
        # Gated on the CERTIFICATE, not the target. Gating it on the target
        # was tried and MEASURED to be wrong: in the voluntary band the
        # ladder certifies on essentially every outer step (its own bar is
        # min(rel_error_threshold, 0.5), i.e. 30 degrees, and the clone
        # reaches 22), so it pre-empts BOTH the growth scan and the tangent
        # step -- and its step carries no Lemma 3.5 bound, because that bound
        # is produced on the tangent path it just displaced. MNIST then
        # diverged on held-out data, test loss 9.37 -> 44.53 over two epochs
        # while the train loss fell. The ladder is the remedy for "no step
        # certifies", so it belongs where no step certifies.
        family_floor = (
            target
            if getattr(
                config.fgd_approx, "certify_family_in_target_band", False
            )
            else certificate
        )
        if family_step is not None and epsilon >= family_floor:
            stepped = family_step(model)
            if stepped is not None:
                stepped_epsilon, stepped_system = _relative_error_and_system(
                    stepped, x, y, config.fgd_approx
                )
                # A certified family step DEFERS growth, so it has to earn the
                # deferral: the same marginal-return question growth answers.
                # MEASURED on MNIST, without this the ladder certified on all
                # 20 epochs and the structure never grew once (GRO=0), so the
                # net stayed at its 3x2 start and accuracy plateaued at 0.113
                # from epoch 9 -- worse than the 0.164 the growing path reached
                # by epoch 13. Certifying is not the same as progressing.
                family_min_gain = float(
                    getattr(config.fgd_approx, "certify_family_min_gain", 0.0)
                )
                family_pays = family_min_gain <= 0.0 or (
                    epsilon - stepped_epsilon
                    > family_min_gain * max(epsilon - target, 1e-6)
                )
                if not family_pays:
                    fallback(
                        "certify_family_deferral_refused",
                        "family_step_below_min_gain",
                    )
                    if progress is not None:
                        progress(
                            "[CERTIFY] family certified but only moved eps "
                            f"{epsilon:.4f} -> {stepped_epsilon:.4f} "
                            f"(needs {family_min_gain:g} of the gap to "
                            f"{target:g}); growing instead"
                        )
                if family_pays:
                    model = stepped
                    family_steps += 1
                    epsilon, tangent_system = stepped_epsilon, stepped_system
                    trajectory.append(epsilon)
                    if progress is not None:
                        progress(
                            "[CERTIFY] parametric family certified at fixed "
                            f"structure (no growth); eps -> {epsilon:.4f}"
                        )
                    return model, CertifyResult(
                        relative_error=epsilon,
                        growths=growths,
                        certified=epsilon < target,
                        trajectory=tuple(trajectory),
                        family_stepped=True,
                        family_steps=family_steps,
                        tangent_system=tangent_system,
                        stop_reason="family_stepped",
                        growth_target=target,
                    )

        # Preserving growth cannot make eps worse, so requiring an improvement
        # is free there. Non-preserving growth moves f, so the best available
        # candidate may still sit above the current eps; accepting it is what
        # lets the rank keep climbing, and the loop then relies on the measured
        # trajectory rather than the monotonicity theorem.
        # NOT replaceable by the expressivity bottleneck. MEASURED: swapping
        # this exhaustive eps argmin for the bottleneck argmax collapsed seed 1
        # to 46 parameters -- the 2-2-2 start -- after 70 epochs with the
        # budget untouched, mean 0.9285 -> 0.7380. The two answer different
        # questions: the bottleneck says WHERE WIDTH IS MISSING, this says
        # WHICH GROWTH UNSTICKS THE CERTIFICATE, and only the second guarantees
        # eps goes down, which is what the loop needs to terminate and what a
        # step needs in order to exist at all. They disagree in practice --
        # with the adaptive count gated on "is this layer still the argmax
        # bottleneck?" it fired ZERO times in four seeds.
        best_model, best_epsilon, best_location, best_system, budget_exhausted = (
            _select_growth_candidate(
                model,
                tangent_system,
                x,
                y,
                train_loader,
                device,
                config,
                function_preserving,
                epsilon,
                parameter_budget=parameter_budget,
            )
        )

        if best_model is None:
            if budget_exhausted:
                # Every candidate priced itself out of the budget. That is a
                # statement about what can be PAID for, not about what the
                # structure can represent, so it gets its own reason.
                increment("certify_budget_stops")
                if progress is not None:
                    progress(
                        f"[CERTIFY] parameter budget {parameter_budget} leaves "
                        f"no affordable growth at eps {epsilon:.4f} "
                        f"({count_parameters(model)} params)"
                    )
                stop_reason = "parameter_budget"
                break
            # No growable location reduces eps. The theorem says this cannot
            # persist while the residual is non-zero, so reaching here means
            # the architecture cannot add a direction along rho at all --
            # report honestly rather than loop.
            if progress is not None:
                progress(
                    f"[CERTIFY] no growth reduced eps ({epsilon:.4f}); "
                    "the structure cannot add a direction along the residual"
                )
            stop_reason = "no_growth_reduced_epsilon"
            break

        # THE VALUE GATE, evaluated on the candidate and BEFORE committing to
        # it: a refused growth must cost nothing, neither a parameter nor an
        # entry in the trajectory. It is the generalisation of the exit just
        # above -- from "no growth reduced eps" to "no growth reduced eps
        # enough" -- and it is exact rather than sampled, because
        # _select_growth_candidate is an exhaustive argmin over every location
        # scored by its own exact eps. Above the certificate it never fires
        # (see _growth_pays), so with an unset target this branch is dead by
        # the loop invariant, not by a flag.
        if not is_forced and not _growth_pays(
            epsilon=epsilon,
            candidate_epsilon=best_epsilon,
            growth_target=target,
            certificate=certificate,
            min_gain=growth_min_gain,
        ):
            # Loud rather than frozen: refusing here while no step certifies
            # would be the deadlock this whole design exists to avoid.
            assert epsilon < certificate, (
                "growth refused while no step can certify: "
                f"eps={epsilon!r} certificate={certificate!r}"
            )
            increment("certify_growth_target_stalls")
            unpaid_growths += 1
            if unpaid_growths >= patience:
                if progress is not None:
                    progress(
                        f"[CERTIFY] growth target {target} unreachable at a "
                        f"price worth paying: eps {epsilon:.4f}, gap "
                        f"{epsilon - target:.4f}, best growth gains "
                        f"{epsilon - best_epsilon:.6f} "
                        f"(needs {growth_min_gain * (epsilon - target):.6f}); "
                        f"stopping with the step certificate {certificate} "
                        "still met"
                    )
                stop_reason = "growth_target_unreachable"
                break
        else:
            unpaid_growths = 0

        model = best_model
        epsilon = best_epsilon
        tangent_system = best_system
        growths += 1
        trajectory.append(epsilon)
        if progress is not None:
            progress(
                f"[CERTIFY] growth {growths} at location {best_location}: "
                f"eps -> {epsilon:.4f}"
                + ("  (certified)" if epsilon < target else "")
            )

        # HOW MUCH, answered the same organic way as WHETHER. Once a step
        # already certifies, further growth is voluntary, so buy ONE increment
        # and hand the turn back: the step commits, training happens, and the
        # next outer step re-asks on a model that has actually moved. The
        # total is then decided by how many times "growing beats training"
        # keeps holding across genuinely different states -- no count, no
        # parameter cap, no eps fraction. MEASURED without this, gating only
        # the ENTRY still front-loaded: 19 growths against 8 committed steps,
        # because one "yes" licensed an unbounded run at an unreachable
        # target. Above the certificate nothing is voluntary -- no step exists
        # at any damping -- so the loop keeps growing there, unchanged.
        if growth_warranted is not None and epsilon < certificate:
            stop_reason = "growth_turn_taken"
            break

        # ADAPTIVE COUNT: keep adding at the just-chosen best location while each
        # increment still pays -- marginal eps reduction above min_gain of the
        # remaining gap -- stopping the instant it certifies or the returns
        # diminish. The number of neurons is chosen by the certificate and the
        # value criterion, not fixed; every intermediate structure is scored by
        # its OWN projection, so the certificate conditions are unchanged. This
        # cuts the number of full location scans on a hard task (CIFAR) without
        # weakening any certificate.
        # BY THE SAME CRITERION THAT CHOSE THE LOCATION, when one is supplied.
        # The gap rule below asks "did this neuron close 10% of what is left to
        # certify?", which is a question about eps and says nothing about
        # whether this layer is still the place to buy. It therefore keeps
        # buying HERE without re-asking WHERE -- MEASURED, that is exactly what
        # it did on seed 3: eps sat at 1.0310, twice the threshold, and it
        # bought a second neuron at location 0 anyway, which is the blind
        # purchase the expressivity criterion exists to avoid.
        #
        # With layer_bottlenecks the same question that picked the location
        # decides how many: keep buying while THIS layer is still the argmax
        # bottleneck, re-measured after every purchase. It is self-limiting --
        # widening a layer lowers its own bottleneck -- so it stops on its own
        # the moment another layer becomes the one that cannot express what is
        # asked of it, with no gap, no fraction and no constant.
        _adaptive_on = getattr(config.fgd_approx, "certify_adaptive_growth", False)
        if _adaptive_on and layer_bottlenecks is not None:
            while epsilon >= threshold and growths < max_growths:
                measured = layer_bottlenecks(model)
                if not measured or max(measured) <= 0.0:
                    break
                if measured.index(max(measured)) != best_location:
                    break  # another layer is the bottleneck now; re-scan
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
                model = bigger
                epsilon = bigger_epsilon
                tangent_system = bigger_system
                growths += 1
                trajectory.append(epsilon)
                if progress is not None:
                    progress(
                        f"[CERTIFY] adaptive +growth {growths} at location "
                        f"{best_location} [by=expressivity_bottleneck]: eps -> "
                        f"{epsilon:.4f}"
                        + ("  (certified)" if epsilon < threshold else "")
                    )
        elif _adaptive_on:
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

    if stop_reason is None:
        stop_reason = "certified" if epsilon < target else "max_growths"
    return model, CertifyResult(
        relative_error=epsilon,
        growths=growths,
        certified=epsilon < target,
        trajectory=tuple(trajectory),
        family_stepped=False,
        family_steps=family_steps,
        tangent_system=tangent_system,
        stop_reason=stop_reason,
        growth_target=target,
    )
