"""The paper-pure step: take a rate inside the certified interval and apply.

Lemma 3.5 does not offer descent as a condition to check -- it derives it.
So the flow keeps the rate the certificate licenses and drops only the
empirical verification. These tests pin exactly that, and pin the boundary
the divergence lesson established: the rate must come from the certificate
that was measured, never recomputed from a different sample.
"""

from __future__ import annotations

import pytest

from stable_tiny.pipeline import (
    _apply_lemma35_step,
    certified_validation_learning_rate,
    load_pipeline_config,
)


@pytest.fixture
def config():
    return load_pipeline_config("configs/fgd/certify_mnist.yaml").fgd_approx


class _Trial:
    """Minimal stand-in exposing only what the acceptance path reads."""

    def __init__(self, *, sensor_valid=True, descends=False, skipped=0):
        self.epoch_result = type(
            "R", (), {"sensor_valid": sensor_valid, "skipped_batches": skipped}
        )()
        self.certificate = type("C", (), {"sensor_valid": sensor_valid})()
        # The point of the variant: this is False and the step commits anyway.
        self.all_conditions_valid = descends
        self.loss_descent_valid = descends


class _Certificate:
    def __init__(self, *, max_valid_learning_rate, sensor_valid=True):
        self.max_valid_learning_rate = max_valid_learning_rate
        self.sensor_valid = sensor_valid


def test_a_certified_step_commits_even_without_observed_descent(config) -> None:
    """The whole point: descent is a CONCLUSION of the lemma, not a gate."""
    seen: list[float] = []

    def evaluate(rate: float) -> _Trial:
        seen.append(rate)
        return _Trial(descends=False)

    result = _apply_lemma35_step(
        maximum_learning_rate=0.0925, evaluate_trial=evaluate, config=config
    )
    assert result.accepted is not None
    assert result.accepted.all_conditions_valid is False   # explicitly ignored


def test_exactly_one_rate_is_tried(config) -> None:
    """There is nothing to search over: any interior point is certified."""
    seen: list[float] = []
    _apply_lemma35_step(
        maximum_learning_rate=0.0925,
        evaluate_trial=lambda rate: (seen.append(rate), _Trial())[1],
        config=config,
    )
    assert seen == [0.0925]


def test_the_rate_is_the_certificate_s_own_bound(config) -> None:
    """It must be the SAME number the ordinary sweep would have started from.

    Recomputing it from a different sample is the mistake that diverged the
    run: eta = 0.95 * eta_bar(eps_train) at eps ~ 0.42 gave eta = 0.86
    against |u| = 15, far outside the linear regime in which the
    function-space lemma governs a parameter-space step.
    """
    certificate = _Certificate(max_valid_learning_rate=0.0925)
    expected = certified_validation_learning_rate(certificate, config)
    seen: list[float] = []
    _apply_lemma35_step(
        maximum_learning_rate=expected,
        evaluate_trial=lambda rate: (seen.append(rate), _Trial())[1],
        config=config,
    )
    assert seen == [expected]


def test_an_unlicensed_certificate_takes_no_step_at_all(config) -> None:
    """No admissible rate: no evaluation happens at all."""

    def evaluate(rate: float) -> _Trial:  # pragma: no cover - must not run
        raise AssertionError("a step was taken without a certified rate")

    result = _apply_lemma35_step(
        maximum_learning_rate=None, evaluate_trial=evaluate, config=config
    )
    assert result.accepted is None
    assert result.trial_count == 0
    assert result.sensor_failure is False   # not a numerical failure


def test_a_degenerate_interval_takes_no_step(config) -> None:
    """A rate at or below theory_lr_min is not strictly inside the interval."""

    def evaluate(rate: float) -> _Trial:  # pragma: no cover - must not run
        raise AssertionError("a step was taken outside the interval")

    for rate in (0.0, config.theory_lr_min, config.theory_lr_min / 2):
        result = _apply_lemma35_step(
            maximum_learning_rate=rate, evaluate_trial=evaluate, config=config
        )
        assert result.accepted is None


def test_the_certificate_withholds_a_rate_once_the_sensor_fails(config) -> None:
    """Upstream, an invalid sensor yields no rate -- so no step follows."""
    certificate = _Certificate(
        max_valid_learning_rate=0.0925, sensor_valid=False
    )
    assert certified_validation_learning_rate(certificate, config) is None


def test_numerical_failure_still_blocks_the_step(config) -> None:
    """Sensors are arithmetic, not theory: a non-finite measurement rejects."""
    result = _apply_lemma35_step(
        maximum_learning_rate=0.0925,
        evaluate_trial=lambda rate: _Trial(sensor_valid=False),
        config=config,
    )
    assert result.accepted is None
    assert result.sensor_failure is True
    assert result.last_trial is not None    # kept for diagnostics


def test_a_skipped_batch_blocks_the_step(config) -> None:
    """A skipped batch means the measurement is incomplete, not that it passed."""
    result = _apply_lemma35_step(
        maximum_learning_rate=0.0925,
        evaluate_trial=lambda rate: _Trial(skipped=1),
        config=config,
    )
    assert result.accepted is None
    assert result.sensor_failure is True
