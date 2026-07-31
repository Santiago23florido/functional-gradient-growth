"""Forced-growth discrimination: finite failure vs. non-finite overflow.

``grow_until_certified``'s ``force`` closes a REAL deadlock (MEASURED on
MNIST: eps frozen at 0.475, certified, while no admissible learning rate
produced held-out descent -- nothing could act, ever). But forcing on
EVERY failed step over-fired just as badly (MEASURED: 262 growths against
7 committed steps, 242 with eps ALREADY certified, triggered by a
non-finite validation measurement -- the model overflowing on unseen
data). ``certify_force_growth_on_finite_step_failure`` discriminates the
two: force only when the failure's own measurement was finite.

This file pins two things in isolation, without running the full
pipeline: (1) the sensor now reports WHY it rejected, not just whether,
and non-finite is a distinct, named reason from the three finite
geometric violations; (2) the ``force`` decision itself, exercised
through the small pure helper ``_certify_force_growth``, is bit-identical
to today (always False) while the new config flag stays False.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from fgdlib.profile import reset, snapshot
from fgdlib.tangent import (
    FGDApproxConfig,
    FGDOutputRelError,
    _projection_sensor_reason,
    _projection_sensor_valid,
    _projection_step_sensor_reason,
    _projection_step_sensor_valid,
    _TangentProjectionStep,
)
from stable_tiny.pipeline import _certify_force_growth


def _valid_stats() -> dict[str, float]:
    """A dot/approx/target triple that satisfies every invariant."""
    return dict(dot_product=1.0, approximation_sq_norm=0.9, target_sq_norm=1.0)


# ---------------------------------------------------------------------------
# _projection_sensor_reason: one named cause per rejection, not just a bool.
# ---------------------------------------------------------------------------


def test_valid_projection_has_no_reason_and_passes_the_bool_sensor() -> None:
    stats = _valid_stats()
    assert _projection_sensor_reason(eps=1e-6, **stats) is None
    assert _projection_sensor_valid(eps=1e-6, **stats) is True


@pytest.mark.parametrize(
    "bad_field", ["dot_product", "approximation_sq_norm", "target_sq_norm"]
)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_inputs_report_non_finite_reason(
    bad_field: str, bad_value: float
) -> None:
    stats = _valid_stats()
    stats[bad_field] = bad_value
    assert _projection_sensor_reason(eps=1e-6, **stats) == "non_finite"
    assert _projection_sensor_valid(eps=1e-6, **stats) is False


def test_negative_norm_is_a_distinct_finite_reason() -> None:
    stats = _valid_stats()
    stats["approximation_sq_norm"] = -1.0
    reason = _projection_sensor_reason(eps=1e-6, **stats)
    assert reason == "negative_norm"
    assert reason != "non_finite"
    assert _projection_sensor_valid(eps=1e-6, **stats) is False


def test_negative_dot_product_is_a_distinct_finite_reason() -> None:
    stats = _valid_stats()
    stats["dot_product"] = -1.0
    reason = _projection_sensor_reason(eps=1e-6, **stats)
    assert reason == "negative_dot_product"
    assert reason not in ("non_finite", "negative_norm")
    assert _projection_sensor_valid(eps=1e-6, **stats) is False


def test_norm_overshoot_is_a_distinct_finite_reason() -> None:
    stats = _valid_stats()
    stats["approximation_sq_norm"] = 2.0  # > target_sq_norm(1.0) + tolerance
    reason = _projection_sensor_reason(eps=1e-6, **stats)
    assert reason == "norm_overshoot"
    assert reason not in ("non_finite", "negative_norm", "negative_dot_product")
    assert _projection_sensor_valid(eps=1e-6, **stats) is False


def test_every_rejection_reason_is_uniquely_named() -> None:
    """The four causes _projection_sensor_valid used to conflate into a
    single False must come back as four DIFFERENT strings -- otherwise a
    caller cannot tell the overfitting signal (non_finite) apart from a
    sane-but-invariant-violating measurement."""
    reasons = {
        "non_finite": _projection_sensor_reason(
            dot_product=float("nan"),
            approximation_sq_norm=1.0,
            target_sq_norm=1.0,
            eps=1e-6,
        ),
        "negative_norm": _projection_sensor_reason(
            dot_product=1.0,
            approximation_sq_norm=-1.0,
            target_sq_norm=1.0,
            eps=1e-6,
        ),
        "negative_dot_product": _projection_sensor_reason(
            dot_product=-1.0,
            approximation_sq_norm=1.0,
            target_sq_norm=1.0,
            eps=1e-6,
        ),
        "norm_overshoot": _projection_sensor_reason(
            dot_product=1.0,
            approximation_sq_norm=2.0,
            target_sq_norm=1.0,
            eps=1e-6,
        ),
    }
    for expected, actual in reasons.items():
        assert actual == expected
    assert len(set(reasons.values())) == 4


def _tangent_step(**stats: float) -> _TangentProjectionStep:
    return _TangentProjectionStep(
        output_error=FGDOutputRelError(0.0, 1.0, 1.0, 1.0),
        parameter_updates=(),
        learning_rate_used=0.0,
        loss_before=0.0,
        loss_after=0.0,
        descent_ok=True,
        **stats,
    )


def test_projection_step_reason_and_valid_wrap_the_scalar_sensor() -> None:
    """_projection_step_sensor_reason/_valid must agree with the underlying
    scalar sensor exactly -- they are thin wrappers, not a second policy."""
    config = FGDApproxConfig()
    valid_step = _tangent_step(**_valid_stats())
    assert _projection_step_sensor_reason(valid_step, config) is None
    assert _projection_step_sensor_valid(valid_step, config) is True

    invalid_step = _tangent_step(
        dot_product=float("nan"), approximation_sq_norm=1.0, target_sq_norm=1.0
    )
    assert _projection_step_sensor_reason(invalid_step, config) == "non_finite"
    assert _projection_step_sensor_valid(invalid_step, config) is False

    geometric_step = _tangent_step(
        dot_product=1.0, approximation_sq_norm=2.0, target_sq_norm=1.0
    )
    assert _projection_step_sensor_reason(geometric_step, config) == "norm_overshoot"
    assert _projection_step_sensor_valid(geometric_step, config) is False


# ---------------------------------------------------------------------------
# _certify_force_growth: the `force=` decision at the grow_until_certified
# call site, exercised directly so the pipeline's 6000+ line function does
# not have to be driven end to end to prove the discrimination rule.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("previous_step_committed", [True, False])
@pytest.mark.parametrize("previous_failure_non_finite", [True, False])
def test_default_flag_never_forces(
    monkeypatch, previous_step_committed: bool, previous_failure_non_finite: bool
) -> None:
    """certify_force_growth_on_finite_step_failure defaults to False, so
    force must be False for every combination of inputs -- this is the
    bit-identical-to-today guarantee the hard constraint asks for."""
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    config = FGDApproxConfig()
    assert config.certify_force_growth_on_finite_step_failure is False
    force = _certify_force_growth(
        config,
        previous_step_committed=previous_step_committed,
        previous_failure_non_finite=previous_failure_non_finite,
    )
    assert force is False
    values = snapshot()
    assert values["certify_forced_growths"] == 0
    assert values["certify_force_suppressed_nonfinite"] == 0
    reset()


def test_flag_on_forces_growth_on_a_finite_failed_step(monkeypatch) -> None:
    """The MNIST case: the previous step failed to commit and the
    measurement behind that failure was finite (sane numbers, no
    admissible rate) -- force must fire."""
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    config = replace(
        FGDApproxConfig(), certify_force_growth_on_finite_step_failure=True
    )
    force = _certify_force_growth(
        config,
        previous_step_committed=False,
        previous_failure_non_finite=False,
    )
    assert force is True
    values = snapshot()
    assert values["certify_forced_growths"] == 1
    assert values["certify_force_suppressed_nonfinite"] == 0
    reset()


def test_flag_on_suppresses_forcing_on_a_non_finite_failure(monkeypatch) -> None:
    """The over-firing case: the step failed to commit, but the trigger was
    a non-finite validation measurement (the model overflowing on unseen
    data). Forcing must be suppressed, and the suppression counted."""
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    config = replace(
        FGDApproxConfig(), certify_force_growth_on_finite_step_failure=True
    )
    force = _certify_force_growth(
        config,
        previous_step_committed=False,
        previous_failure_non_finite=True,
    )
    assert force is False
    values = snapshot()
    assert values["certify_forced_growths"] == 0
    assert values["certify_force_suppressed_nonfinite"] == 1
    reset()


@pytest.mark.parametrize("previous_failure_non_finite", [True, False])
def test_flag_on_never_forces_once_a_step_already_committed(
    monkeypatch, previous_failure_non_finite: bool
) -> None:
    """force only ever answers "the structure is stuck": if the previous
    step DID commit there is nothing to break out of, regardless of what
    the (irrelevant) failure-finiteness flag says."""
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    config = replace(
        FGDApproxConfig(), certify_force_growth_on_finite_step_failure=True
    )
    force = _certify_force_growth(
        config,
        previous_step_committed=True,
        previous_failure_non_finite=previous_failure_non_finite,
    )
    assert force is False
    values = snapshot()
    assert values["certify_forced_growths"] == 0
    assert values["certify_force_suppressed_nonfinite"] == 0
    reset()
