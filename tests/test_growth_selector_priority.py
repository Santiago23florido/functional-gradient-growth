"""Growth-layer selection: relative error outranks parameter count."""

from __future__ import annotations

import torch

from fgdlib.growth import GrowthResult
from fgdlib.tangent import FGDOutputRelError, FGDValidationCertificate
from stable_tiny.pipeline import _GrowthProbe, _select_growth_probe


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
