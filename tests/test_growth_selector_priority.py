"""Growth-layer selection: relative error outranks parameter count."""

from __future__ import annotations

import torch

from fgdlib.search.growth import GrowthResult
from fgdlib.tangent import FGDOutputRelError, FGDValidationCertificate
from stable_tiny.pipeline import _GrowthProbe, _select_growth_probe, _select_growth_probe_by_descent


def _certificate(relative_error: float | None) -> FGDValidationCertificate:
    output_error = (
        FGDOutputRelError(
            relative_error=relative_error,
            approximation_norm=1.0,
            target_norm=1.0,
            directional_cosine=1.0,
        )
        if relative_error is not None
        else None
    )
    return FGDValidationCertificate(
        learning_rate_upper_bound=None,
        max_valid_learning_rate=None,
        learning_rate_interval_valid=None,
        skipped_batches=0,
        relative_error_condition_valid=None,
        gradient_sq_norm=1.0,
        theory_descent_coefficient=None,
        relative_error=relative_error,
        output_relative_error=output_error,
        sensor_valid=True,
        sensor_invalid_batches=0,
    )


def _probe(
    *,
    layer_index: int,
    parameters: int,
    relative_error: float | None,
    improves: bool,
) -> _GrowthProbe:
    return _GrowthProbe(
        # A bias-free linear layer with `parameters` weights stands in for
        # the grown model: only its parameter count matters here.
        model=torch.nn.Linear(parameters, 1, bias=False),
        result=GrowthResult(
            layer_index=layer_index,
            best_scaling_factor=1.0,
            best_train_loss=1.0,
            line_search=[],
        ),
        certificate=_certificate(relative_error),
        improves_fgd=improves,
    )


def test_without_certifying_candidates_lowest_relative_error_wins() -> None:
    small_but_bad = _probe(
        layer_index=0,
        parameters=10,
        relative_error=0.9,
        improves=False,
    )
    big_but_good = _probe(
        layer_index=1,
        parameters=200,
        relative_error=0.6,
        improves=False,
    )
    chosen = _select_growth_probe([small_but_bad, big_but_good])
    assert chosen is big_but_good


def test_parameter_count_is_only_a_tie_breaker() -> None:
    lean = _probe(
        layer_index=1,
        parameters=10,
        relative_error=0.7,
        improves=False,
    )
    heavy = _probe(
        layer_index=0,
        parameters=200,
        relative_error=0.7,
        improves=False,
    )
    chosen = _select_growth_probe([heavy, lean])
    assert chosen is lean


def test_improving_candidates_keep_the_frugal_first_policy() -> None:
    frugal_improving = _probe(
        layer_index=2,
        parameters=20,
        relative_error=0.4,
        improves=True,
    )
    better_error_but_heavy = _probe(
        layer_index=0,
        parameters=300,
        relative_error=0.1,
        improves=True,
    )
    non_improving = _probe(
        layer_index=1,
        parameters=5,
        relative_error=0.05,
        improves=False,
    )
    chosen = _select_growth_probe(
        [better_error_but_heavy, non_improving, frugal_improving]
    )
    assert chosen is frugal_improving


def test_missing_relative_error_never_wins_over_a_measured_one() -> None:
    unmeasured = _probe(
        layer_index=0,
        parameters=10,
        relative_error=None,
        improves=False,
    )
    measured = _probe(
        layer_index=1,
        parameters=500,
        relative_error=0.99,
        improves=False,
    )
    chosen = _select_growth_probe([unmeasured, measured])
    assert chosen is measured


def test_empty_probe_list_selects_nothing() -> None:
    assert _select_growth_probe([]) is None


def test_prefer_lower_error_grows_the_impactful_expensive_layer() -> None:
    """With the flag, an improving probe with lower rel_err wins even if big."""
    cheap_weak = _probe(
        layer_index=1,
        parameters=20,
        relative_error=0.45,
        improves=True,
    )
    expensive_strong = _probe(
        layer_index=0,
        parameters=800,
        relative_error=0.20,
        improves=True,
    )
    # Default frugal-first keeps the cheap layer.
    assert _select_growth_probe([cheap_weak, expensive_strong]) is cheap_weak
    # prefer_lower_error picks the most impactful (input) layer.
    assert (
        _select_growth_probe(
            [cheap_weak, expensive_strong],
            prefer_lower_error=True,
        )
        is expensive_strong
    )


def _probe_with_descent(
    *, layer_index, added_params, descent
) -> _GrowthProbe:
    return _GrowthProbe(
        model=torch.nn.Linear(max(added_params, 1), 1, bias=False),
        result=GrowthResult(
            layer_index=layer_index,
            best_scaling_factor=1.0,
            best_train_loss=1.0,
            line_search=[],
        ),
        certificate=_certificate(1.5),  # rel_err jumped (blind); irrelevant
        improves_fgd=False,
        functional_descent=descent,
        added_parameters=added_params,
    )


def test_descent_per_parameter_selection_prefers_efficient_layer() -> None:
    from stable_tiny.pipeline import _select_growth_probe_by_descent

    # Input layer: big absolute descent but many params (low per-param).
    input_layer = _probe_with_descent(
        layer_index=0, added_params=1574, descent=1297.0
    )
    # Late layer: smaller absolute descent but tiny params (huge per-param).
    late_layer = _probe_with_descent(
        layer_index=2, added_params=26, descent=7516.0
    )
    chosen = _select_growth_probe_by_descent(
        [input_layer, late_layer], eps=1e-12
    )
    assert chosen is late_layer  # 289/param vs 0.8/param


def test_descent_selection_ignores_non_descending_growths() -> None:
    from stable_tiny.pipeline import _select_growth_probe_by_descent

    ascends = _probe_with_descent(
        layer_index=0, added_params=10, descent=-5.0
    )
    descends = _probe_with_descent(
        layer_index=1, added_params=100, descent=3.0
    )
    assert (
        _select_growth_probe_by_descent([ascends, descends], eps=1e-12)
        is descends
    )
    # No genuine descent anywhere -> no growth committed.
    assert (
        _select_growth_probe_by_descent([ascends], eps=1e-12) is None
    )


def _probe_with_epsilon(*, layer_index, added_params, epsilon_reduction):
    return _GrowthProbe(
        model=torch.nn.Linear(max(added_params, 1), 1, bias=False),
        result=GrowthResult(
            layer_index=layer_index,
            best_scaling_factor=1.0,
            best_train_loss=1.0,
            line_search=[],
        ),
        certificate=_certificate(1.5),
        improves_fgd=False,
        added_parameters=added_params,
        epsilon_reduction=epsilon_reduction,
    )


def test_epsilon_selection_buys_representability_per_parameter() -> None:
    """R2: rank by look-ahead eps reduction per added parameter."""
    from stable_tiny.pipeline import _select_growth_probe_by_epsilon

    # Mirrors the measured A/B: the expensive input layer barely moves eps,
    # the cheap late layer moves it a lot.
    expensive = _probe_with_epsilon(
        layer_index=0, added_params=1574, epsilon_reduction=0.10
    )
    cheap = _probe_with_epsilon(
        layer_index=2, added_params=26, epsilon_reduction=0.316
    )
    chosen = _select_growth_probe_by_epsilon([expensive, cheap], eps=1e-12)
    assert chosen is cheap


def test_no_growth_when_nothing_enlarges_the_reachable_set() -> None:
    """R3: 'no candidate reduces eps' is the termination condition."""
    from stable_tiny.pipeline import _select_growth_probe_by_epsilon

    # Every candidate makes eps WORSE -- the measured immediate-eps case.
    worse = [
        _probe_with_epsilon(
            layer_index=i, added_params=10 * (i + 1), epsilon_reduction=-0.05
        )
        for i in range(3)
    ]
    assert _select_growth_probe_by_epsilon(worse, eps=1e-12) is None
    # A single improving candidate is enough to keep searching.
    worse.append(
        _probe_with_epsilon(layer_index=3, added_params=20, epsilon_reduction=0.2)
    )
    assert _select_growth_probe_by_epsilon(worse, eps=1e-12) is worse[-1]
