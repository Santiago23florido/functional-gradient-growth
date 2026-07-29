"""Exact tangent systems are reusable only for their unchanged owner state."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from fgdlib.profile import reset, snapshot
from fgdlib.search.certify import _grow_clone, grow_until_certified
from fgdlib.search.damping import (
    minimal_relative_error,
    select_projection_damping,
)
from fgdlib.search.realize import realize_functional_step
from fgdlib.tangent import exact_tangent_system, validate_exact_tangent_system
from stable_tiny.pipeline import build_model, load_pipeline_config


def _setup(*, hidden_size: int = 32, samples: int = 8):
    config = load_pipeline_config("configs/experiments/default.yaml")
    config = replace(
        config,
        model=replace(
            config.model,
            hidden_size=hidden_size,
            number_hidden_layers=2,
        ),
        fgd_approx=replace(
            config.fgd_approx,
            projection_solver="exact",
            certify_realize_path=True,
            certify_realize_freeze_gram=True,
        ),
    )
    model = build_model(config, torch.device("cpu"))
    torch.manual_seed(0)
    x = torch.randn(samples, config.data.in_features)
    y = torch.randn(samples, config.data.out_features)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=samples,
    )
    return config, model, x, y, loader


@pytest.mark.parametrize("fast", [False, True])
def test_exact_system_numerical_and_damping_parity(fast: bool) -> None:
    config, model, x, y, _ = _setup()
    fa = replace(config.fgd_approx, projection_fast_factorization=fast)
    system = exact_tangent_system(model, x, y, fa)
    assert system is not None

    direct_error = minimal_relative_error(model, x, y, fa)
    reused_error = minimal_relative_error(model, x, y, fa, system=system)
    tolerance = 1e-5 if fast else 1e-10
    assert reused_error == pytest.approx(direct_error, abs=tolerance, rel=1e-8)

    direct = select_projection_damping(model, x, y, fa)
    reused = select_projection_damping(model, x, y, fa, system=system)
    assert direct is not None and reused is not None
    assert reused.candidate.relative_damping == pytest.approx(
        direct.candidate.relative_damping
    )
    assert reused.candidate.absolute_damping == pytest.approx(
        direct.candidate.absolute_damping
    )
    assert reused.candidate.relative_error == pytest.approx(
        direct.candidate.relative_error
    )
    assert reused.candidate.learning_rate == pytest.approx(
        direct.candidate.learning_rate
    )
    assert reused.candidate.guaranteed_decrease == pytest.approx(
        direct.candidate.guaranteed_decrease
    )
    assert reused.candidates == direct.candidates
    for actual, expected in zip(reused.parameter_updates, direct.parameter_updates):
        assert torch.allclose(actual, expected, atol=tolerance, rtol=1e-8)


def test_one_system_build_for_an_unchanged_outer_state(monkeypatch) -> None:
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    config, model, x, y, loader = _setup()
    grown, result = grow_until_certified(
        model,
        x,
        y,
        loader,
        torch.device("cpu"),
        config,
    )
    assert result.growths == 0
    assert result.tangent_system is not None
    assert snapshot()["exact_tangent_system_calls"] == 1

    choice = select_projection_damping(
        grown,
        x,
        y,
        config.fgd_approx,
        system=result.tangent_system,
    )
    assert choice is not None
    assert snapshot()["exact_tangent_system_calls"] == 1

    realize_functional_step(
        grown,
        x,
        y,
        choice.parameter_updates,
        choice.candidate.certified_learning_rate,
        config.fgd_approx,
        max_iterations=2,
        system=result.tangent_system,
    )
    assert snapshot()["exact_tangent_system_calls"] == 1
    with pytest.raises(ValueError, match="parameter values changed"):
        select_projection_damping(
            grown,
            x,
            y,
            config.fgd_approx,
            system=result.tangent_system,
        )


def test_system_rejects_parameter_probe_target_config_and_trainable_changes() -> None:
    config, model, x, y, _ = _setup()
    fa = config.fgd_approx
    system = exact_tangent_system(model, x, y, fa)
    assert system is not None

    changed_model = copy.deepcopy(model)
    with pytest.raises(ValueError, match="different model"):
        validate_exact_tangent_system(system, changed_model, x, y, fa)
    with pytest.raises(ValueError, match="probe or targets"):
        validate_exact_tangent_system(system, model, x.clone(), y, fa)
    with pytest.raises(ValueError, match="probe or targets"):
        validate_exact_tangent_system(system, model, x, y.clone(), fa)
    with pytest.raises(ValueError, match="configuration changed"):
        validate_exact_tangent_system(
            system,
            model,
            x,
            y,
            replace(fa, projection_damping=fa.projection_damping * 2),
        )

    next(model.parameters()).requires_grad_(False)
    with pytest.raises(ValueError, match="parameter ordering changed"):
        validate_exact_tangent_system(system, model, x, y, fa)


@pytest.mark.parametrize("function_preserving", [True, False])
def test_system_is_invalid_after_growth(function_preserving: bool) -> None:
    config, model, x, y, loader = _setup(hidden_size=3, samples=12)
    system = exact_tangent_system(model, x, y, config.fgd_approx)
    assert system is not None
    grown = _grow_clone(
        model,
        loader,
        0,
        torch.device("cpu"),
        config,
        function_preserving,
    )
    assert grown is not None
    with pytest.raises(ValueError, match="different model"):
        validate_exact_tangent_system(
            system,
            grown,
            x,
            y,
            config.fgd_approx,
        )


def test_family_step_returns_only_the_moved_models_system() -> None:
    config, model, x, y, loader = _setup(hidden_size=1, samples=20)
    initial_system = exact_tangent_system(model, x, y, config.fgd_approx)
    assert initial_system is not None

    def family_step(candidate):
        stepped = copy.deepcopy(candidate)
        with torch.no_grad():
            next(stepped.parameters()).add_(0.01)
        return stepped

    moved, result = grow_until_certified(
        model,
        x,
        y,
        loader,
        torch.device("cpu"),
        config,
        family_step=family_step,
    )
    assert result.family_stepped
    assert result.tangent_system is not None
    validate_exact_tangent_system(
        result.tangent_system,
        moved,
        x,
        y,
        config.fgd_approx,
    )
    with pytest.raises(ValueError, match="different model"):
        validate_exact_tangent_system(
            initial_system,
            moved,
            x,
            y,
            config.fgd_approx,
        )


def test_realization_parity_with_a_precomputed_system() -> None:
    config, original, x, y, _ = _setup()
    fa = config.fgd_approx
    optimized = copy.deepcopy(original)
    choice = select_projection_damping(original, x, y, fa)
    assert choice is not None
    updates = tuple(update.detach().clone() for update in choice.parameter_updates)
    rate = choice.candidate.certified_learning_rate
    assert rate is not None

    baseline_result = realize_functional_step(
        original,
        x,
        y,
        updates,
        rate,
        fa,
        max_iterations=4,
    )
    system = exact_tangent_system(optimized, x, y, fa)
    assert system is not None
    optimized_result = realize_functional_step(
        optimized,
        x,
        y,
        updates,
        rate,
        fa,
        max_iterations=4,
        system=system,
    )

    assert optimized_result.residual_fraction == pytest.approx(
        baseline_result.residual_fraction, rel=1e-10, abs=1e-10
    )
    assert optimized_result.realised_fraction == pytest.approx(
        baseline_result.realised_fraction, rel=1e-10, abs=1e-10
    )
    assert optimized_result.iterations == baseline_result.iterations
    assert optimized_result.parameter_displacement == pytest.approx(
        baseline_result.parameter_displacement, rel=1e-10, abs=1e-10
    )
    for actual, expected in zip(optimized.parameters(), original.parameters()):
        assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert torch.allclose(optimized(x), original(x), atol=1e-10, rtol=1e-10)
