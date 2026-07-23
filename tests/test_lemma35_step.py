"""The certified cycle: grow until eps < 1/2, then step at 0.95 of the interval.

The rule these pin, in full: once the relative-error criterion is satisfied,
train with the tangent approximation at ``eta = 0.95 * eta_bar(eps)`` and
ASSUME the remaining condition rather than verifying it -- its two premises
(the rate lies in the admissible interval, the relative error satisfies the
criterion) are exactly what has just been established. Keep stepping until
the relative error stops being satisfied; that is the signal to grow again.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from fgdlib.tangent import theoretical_learning_rate_upper_bound
from stable_tiny.pipeline import (
    _apply_lemma35_step,
    lemma35_learning_rate,
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


# --- the rate ------------------------------------------------------------


def test_rate_is_the_configured_fraction_of_the_bound(config) -> None:
    """eta = theory_lr_safety * eta_bar(eps) -- 0.95 of the interval."""
    assert config.theory_lr_safety == pytest.approx(0.95)
    epsilon = 0.3
    expected = config.theory_lr_safety * theoretical_learning_rate_upper_bound(
        epsilon, config
    )
    assert lemma35_learning_rate(epsilon, config) == pytest.approx(expected)


def test_rate_lies_strictly_inside_the_admissible_interval(config) -> None:
    """The guarantee is for the OPEN interval, so stay off both ends."""
    for epsilon in (0.0, 0.1, 0.25, 0.4, 0.49):
        rate = lemma35_learning_rate(epsilon, config)
        bound = theoretical_learning_rate_upper_bound(epsilon, config)
        assert rate is not None
        assert config.theory_lr_min < rate < bound


def test_the_rate_widens_as_the_certificate_tightens(config) -> None:
    """eta_bar(eps) = 2(1 - 2 eps)/(L_s (1 + 2 eps)) is decreasing in eps."""
    rates = [lemma35_learning_rate(e, config) for e in (0.05, 0.2, 0.35, 0.45)]
    assert all(rate is not None for rate in rates)
    assert rates == sorted(rates, reverse=True)


# --- where the cycle turns ----------------------------------------------


def test_no_rate_once_the_relative_error_criterion_fails(config) -> None:
    """eps >= 1/2 ends the training phase; the caller grows instead."""
    for epsilon in (0.5, 0.501, 0.9, 1.7, float("inf")):
        assert lemma35_learning_rate(epsilon, config) is None


def test_a_stricter_configured_threshold_is_honoured(config) -> None:
    """rel_error_threshold below 1/2 tightens the criterion, never loosens it."""
    strict = replace(config, rel_error_threshold=0.2)
    assert lemma35_learning_rate(0.3, strict) is None
    assert lemma35_learning_rate(0.1, strict) is not None
    # And a threshold ABOVE 1/2 cannot buy back what the lemma forbids.
    assert lemma35_learning_rate(0.7, replace(config, rel_error_threshold=0.9)) is None


def test_an_unmeasured_relative_error_is_not_a_pass(config) -> None:
    """Absence of a measurement blocks the step rather than licensing it."""
    assert lemma35_learning_rate(None, config) is None


# --- the step ------------------------------------------------------------


def test_a_certified_step_commits_without_verifying_descent(config) -> None:
    """The whole point: the remaining condition is assumed, not checked."""
    result = _apply_lemma35_step(
        relative_error=0.3,
        evaluate_trial=lambda rate: _Trial(descends=False),
        config=config,
    )
    assert result.accepted is not None
    assert result.accepted.all_conditions_valid is False   # explicitly ignored


def test_exactly_one_rate_is_tried_and_it_is_the_certified_one(config) -> None:
    """No sweep: any interior point is admissible, so 0.95 of the bound it is."""
    seen: list[float] = []
    _apply_lemma35_step(
        relative_error=0.3,
        evaluate_trial=lambda rate: (seen.append(rate), _Trial())[1],
        config=config,
    )
    assert seen == [lemma35_learning_rate(0.3, config)]


def test_an_uncertified_structure_takes_no_step_at_all(config) -> None:
    """eps >= 1/2: nothing is evaluated, because no rate is admissible."""

    def evaluate(rate: float) -> _Trial:  # pragma: no cover - must not run
        raise AssertionError("a step was taken without a certificate")

    result = _apply_lemma35_step(
        relative_error=0.6, evaluate_trial=evaluate, config=config
    )
    assert result.accepted is None
    assert result.trial_count == 0
    assert result.sensor_failure is False   # not a numerical failure


def test_a_held_out_sensor_failure_is_reported_but_does_not_block(config) -> None:
    """It is a statement about the fit, not about admissibility.

    Both sensors a trial carries are the HELD-OUT one: ``certificate`` is
    built from the validation measurement and ``epoch_result.sensor_valid``
    is copied from it, so there is no train-side sensor in a trial at all.
    With the projector invariants off there, the only test is finiteness --
    failing it says the MODEL produced non-finite values on unseen data.

    Letting it reject was the deadlock in its final form: MEASURED,
    eps = 0.4808 certified so no growth fired while this sensor rejected
    every step, and epochs 85-92 came out bit-identical at loss 0.1092, 1
    committed step against 21 growths.
    """
    result = _apply_lemma35_step(
        relative_error=0.3,
        evaluate_trial=lambda rate: _Trial(sensor_valid=False),
        config=config,
    )
    assert result.accepted is not None          # committed anyway
    assert result.sensor_failure is True        # and reported


def test_admissibility_comes_from_the_train_side_measurement(config) -> None:
    """A non-finite eps on the certified sample DOES block, via the rate.

    That is the guard the held-out sensor was standing in for, and it sits
    where the lemma puts it: NaN fails ``eps < 1/2``, so no rate is issued
    and no trial is ever evaluated.
    """

    def evaluate(rate: float) -> _Trial:  # pragma: no cover - must not run
        raise AssertionError("a step was taken without a finite eps")

    for epsilon in (float("nan"), float("inf"), None):
        result = _apply_lemma35_step(
            relative_error=epsilon, evaluate_trial=evaluate, config=config
        )
        assert result.accepted is None
        assert result.trial_count == 0


# --- the rate that maximises what the lemma guarantees ----------------------


def test_the_optimal_rate_is_the_interval_midpoint(config) -> None:
    """eta_bar is where the bracket VANISHES, so the parabola peaks at half.

    Lemma 3.5 bounds L(f_t+1) <= L(f_t) - eta*bracket(eta)*||grad L||^2, and
    the guaranteed decrease eta*bracket(eta) is a parabola whose upper root
    is eta_bar. Its vertex is therefore always eta_bar/2 -- the midpoint --
    independently of eps, L_s, alpha or beta.
    """
    optimal = replace(config, certify_optimal_rate=True)
    for epsilon in (0.0, 0.1, 0.25, 0.4, 0.49):
        bound = theoretical_learning_rate_upper_bound(epsilon, config)
        assert lemma35_learning_rate(epsilon, optimal) == pytest.approx(
            0.5 * bound
        )


def test_the_optimal_rate_is_smaller_than_the_edge_rate(config) -> None:
    """0.95 of the interval maximises the STEP and minimises what it buys."""
    optimal = replace(config, certify_optimal_rate=True)
    for epsilon in (0.05, 0.2, 0.45):
        assert lemma35_learning_rate(
            epsilon, optimal
        ) < lemma35_learning_rate(epsilon, config)


def test_the_optimal_rate_still_honours_the_certificate(config) -> None:
    """Choosing a better rate never licenses an uncertified step."""
    optimal = replace(config, certify_optimal_rate=True)
    for epsilon in (0.5, 0.7, float("inf"), None):
        assert lemma35_learning_rate(epsilon, optimal) is None


def test_sum_mse_lands_on_the_target_at_the_optimal_rate(config) -> None:
    """Why the midpoint matters, in the one case where it is exact.

    For sum-MSE the functional gradient is r = 2(f - y), so a step
    f <- f - eta r gives f_new - y = (1 - 2 eta)(f - y). At eps = 0 with
    L_s = 2 the interval is (0, 1): the midpoint 0.5 lands exactly on the
    target, while 0.95 gives -0.9(f - y) -- the error flips sign and shrinks
    by only 10 %, which is oscillation at the edge of stability.
    """
    mse = replace(
        config,
        functional_loss="mse",
        theory_smoothness_constant=2.0,
        certify_optimal_rate=True,
    )
    rate = lemma35_learning_rate(0.0, mse)
    assert rate == pytest.approx(0.5)
    assert abs(1.0 - 2.0 * rate) < 1e-12          # lands on the target
    edge = lemma35_learning_rate(0.0, replace(mse, certify_optimal_rate=False))
    assert 1.0 - 2.0 * edge == pytest.approx(-0.9)  # flips sign
