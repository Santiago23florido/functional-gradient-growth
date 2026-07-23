"""A rejected sensor must say WHAT was non-finite, and must not mean "grow".

With the projector invariants off -- as they are for the held-out
measurement, where the direction is a secant rather than a projection -- the
only test left is finiteness. So a rejection there says the MODEL produced
non-finite values on unseen data, not that our arithmetic was inexact. The
distinction matters: it is a symptom of the fit, and adding capacity makes
it worse.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from fgdlib.tangent import (
    FGDOutputRelError,
    _FunctionalStepStats,
    certificate_from_projection_stats,
)
from stable_tiny.pipeline import load_pipeline_config


@pytest.fixture
def config():
    return load_pipeline_config("configs/fgd/certify_smooth_sin_tiny.yaml").fgd_approx


def _stats(*, dot=1.0, approx_sq=1.0, target_sq=1.0):
    return _FunctionalStepStats(
        output_error=FGDOutputRelError(0.0, 1.0, 1.0, 1.0),
        dot_product=dot,
        approximation_sq_norm=approx_sq,
        target_sq_norm=target_sq,
    )


def test_an_overflowing_target_is_named(config) -> None:
    """sum-MSE gives target_sq_norm = 4 sum (f-y)^2 -- the model, overflowing."""
    certificate = certificate_from_projection_stats(
        stats=_stats(target_sq=math.inf),
        learning_rate=None,
        config=config,
        projection_sensor=False,
    )
    assert certificate.sensor_valid is False
    assert certificate.non_finite_quantities == ("target_sq_norm",)


def test_each_quantity_is_reported_by_name(config) -> None:
    for field, value in (
        ("dot", "dot_product"),
        ("approx_sq", "approximation_sq_norm"),
        ("target_sq", "target_sq_norm"),
    ):
        certificate = certificate_from_projection_stats(
            stats=_stats(**{field: math.nan}),
            learning_rate=None,
            config=config,
            projection_sensor=False,
        )
        assert certificate.non_finite_quantities == (value,)


def test_several_at_once_are_all_reported(config) -> None:
    certificate = certificate_from_projection_stats(
        stats=_stats(dot=math.inf, target_sq=math.inf),
        learning_rate=None,
        config=config,
        projection_sensor=False,
    )
    assert certificate.non_finite_quantities == (
        "dot_product",
        "target_sq_norm",
    )


def test_a_healthy_measurement_reports_nothing(config) -> None:
    certificate = certificate_from_projection_stats(
        stats=_stats(),
        learning_rate=None,
        config=config,
        projection_sensor=False,
    )
    assert certificate.sensor_valid is True
    assert certificate.non_finite_quantities == ()


def test_growth_is_not_forced_by_a_failed_step(config) -> None:
    """The certificate alone decides growth -- eps >= 1/2 and nothing else.

    Forcing growth whenever a step failed to commit was measured to produce
    262 growths against 7 committed steps, 242 of them with eps ALREADY
    certified and many at eps = 0.0000, driven by exactly the non-finite
    validation measurement above.
    """
    import inspect

    from stable_tiny import pipeline

    source = inspect.getsource(pipeline.run_pipeline)
    assert "force=not certify_previous_step_committed" not in source
    assert "force=False" in source
