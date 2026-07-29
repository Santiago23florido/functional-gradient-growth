"""Selected realization damping is explicit and default-off."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fgdlib.search.realize import realization_damping
from stable_tiny.pipeline import load_pipeline_config


def test_realization_uses_configured_damping_by_default() -> None:
    config = load_pipeline_config("configs/experiments/default.yaml").fgd_approx
    assert not config.certify_realize_use_selected_damping
    assert realization_damping(config, 0.123) == pytest.approx(
        config.projection_damping
    )


def test_realization_can_use_selected_absolute_damping() -> None:
    config = load_pipeline_config("configs/experiments/default.yaml").fgd_approx
    enabled = replace(config, certify_realize_use_selected_damping=True)
    assert realization_damping(enabled, 0.123) == pytest.approx(0.123)


def test_enabled_flag_without_a_selection_preserves_configured_damping() -> None:
    config = load_pipeline_config("configs/experiments/default.yaml").fgd_approx
    enabled = replace(config, certify_realize_use_selected_damping=True)
    assert realization_damping(enabled, None) == pytest.approx(
        config.projection_damping
    )
