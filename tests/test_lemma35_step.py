"""The paper-pure step: certify eps < 1/2, take 0.95 of the interval, apply.

Lemma 3.5 does not offer descent as a condition to check -- it derives it.
These tests pin that the implementation follows the lemma exactly: the rate
comes from the certified ``eps``, sits strictly inside the admissible
interval, and no step is ever taken when ``eps >= 1/2``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from fgdlib.tangent import (
    theoretical_learning_rate_upper_bound,
)
from stable_tiny.pipeline import (
    _apply_lemma35_step,
    lemma35_learning_rate,
    load_pipeline_config,
)


@pytest.fixture
def config():
    return load_pipeline_config("configs/fgd/certify_mnist.yaml").fgd_approx


def test_rate_is_the_configured_fraction_of_the_bound(config) -> None:
    """eta = theory_lr_safety * eta_bar(eps) -- nothing else."""
    epsilon = 0.3
    expected = config.theory_lr_safety * theoretical_learning_rate_upper_bound(
        epsilon, config
    )
    assert lemma35_learning_rate(epsilon, config) == pytest.approx(expected)


def test_rate_lies_strictly_inside_the_admissible_interval(config) -> None:
    """The lemma's guarantee is for the OPEN interval, so stay off both ends."""
    for epsilon in (0.0, 0.1, 0.25, 0.4, 0.49):
        rate = lemma35_learning_rate(epsilon, config)
        bound = theoretical_learning_rate_upper_bound(epsilon, config)
        assert rate is not None
        assert config.theory_lr_min < rate < bound


def test_no_rate_is_offered_once_the_certificate_fails(config) -> None:
    """eps >= 1/2 is outside Lemma 3.5 entirely -- there is no admissible rate."""
    for epsilon in (0.5, 0.501, 0.9, 1.7, float("inf")):
        assert lemma35_learning_rate(epsilon, config) is None


def test_a_stricter_configured_threshold_is_honoured(config) -> None:
    """rel_error_threshold below 1/2 tightens the certificate, never loosens it."""
    strict = replace(config, rel_error_threshold=0.2)
    assert lemma35_learning_rate(0.3, strict) is None
    assert lemma35_learning_rate(0.1, strict) is not None
    # And a threshold ABOVE 1/2 cannot buy back what the lemma forbids.
    loose = replace(config, rel_error_threshold=0.9)
    assert lemma35_learning_rate(0.7, loose) is None


def test_the_rate_widens_as_the_certificate_tightens(config) -> None:
    """eta_bar(eps) = 2(1-2 eps)/(L_s(1+2 eps)) is decreasing in eps."""
    rates = [lemma35_learning_rate(e, config) for e in (0.05, 0.2, 0.35, 0.45)]
    assert all(rate is not None for rate in rates)
    assert rates == sorted(rates, reverse=True)


class _Trial:
    """Minimal stand-in exposing only what the acceptance path reads."""

    def __init__(self, *, sensor_valid=True, descends=False):
        self.epoch_result = type(
            "R", (), {"sensor_valid": sensor_valid, "skipped_batches": 0}
        )()
        self.certificate = type("C", (), {"sensor_valid": sensor_valid})()
        # The point of the variant: this is False and the step commits anyway.
        self.all_conditions_valid = descends
        self.loss_descent_valid = descends


def test_a_certified_step_commits_even_without_observed_descent(config) -> None:
    """The whole point: descent is a CONCLUSION of the lemma, not a gate."""
    seen: list[float] = []

    def evaluate(rate: float) -> _Trial:
        seen.append(rate)
        return _Trial(descends=False)

    result = _apply_lemma35_step(
        relative_error=0.3, evaluate_trial=evaluate, config=config
    )
    assert result.accepted is not None
    assert result.accepted.all_conditions_valid is False   # explicitly ignored
    # Exactly ONE rate is tried: there is nothing to search over.
    assert len(seen) == 1
    assert seen[0] == pytest.approx(lemma35_learning_rate(0.3, config))


def test_an_uncertified_structure_takes_no_step_at_all(config) -> None:
    """eps >= 1/2: no evaluation happens, because no rate is admissible."""

    def evaluate(rate: float) -> _Trial:  # pragma: no cover - must not run
        raise AssertionError("a step was taken without a certificate")

    result = _apply_lemma35_step(
        relative_error=0.6, evaluate_trial=evaluate, config=config
    )
    assert result.accepted is None
    assert result.trial_count == 0
    assert result.sensor_failure is False   # not a numerical failure


def test_a_missing_relative_error_is_treated_as_uncertified(config) -> None:
    """No measurement is not a pass: absence of eps must block the step."""

    def evaluate(rate: float) -> _Trial:  # pragma: no cover - must not run
        raise AssertionError("a step was taken without a certificate")

    result = _apply_lemma35_step(
        relative_error=None, evaluate_trial=evaluate, config=config
    )
    assert result.accepted is None


def test_numerical_failure_still_blocks_the_step(config) -> None:
    """Sensors are arithmetic, not theory: a non-finite measurement rejects."""
    result = _apply_lemma35_step(
        relative_error=0.3,
        evaluate_trial=lambda rate: _Trial(sensor_valid=False),
        config=config,
    )
    assert result.accepted is None
    assert result.sensor_failure is True
    assert result.last_trial is not None    # kept for diagnostics
