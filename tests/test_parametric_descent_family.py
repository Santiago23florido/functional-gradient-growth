"""Measured-descent parametric family (Prop. 3.8 with measured coefficient)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.tangent import (
    FGDApproxConfig,
    ParametricDescentConfig,
    validate_family_order,
)
from stable_tiny.pipeline import (
    PipelineConfig,
    _FGDTheoryState,
    _search_parametric_descent_candidate,
    load_pipeline_config,
)

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_parametric_descent_config_validation() -> None:
    ParametricDescentConfig().validate()
    ParametricDescentConfig(optimizer="adamw").validate()
    with pytest.raises(ValueError):
        ParametricDescentConfig(optimizer="rmsprop").validate()
    with pytest.raises(ValueError):
        ParametricDescentConfig(min_progress=0.0).validate()
    with pytest.raises(ValueError):
        ParametricDescentConfig(min_cosine=1.5).validate()


def test_family_order_accepts_parametric_descent() -> None:
    validate_family_order(
        ("tangent", "rkhs_head", "parametric_gd", "parametric_descent")
    )


def _descent_problem(
    **descent_overrides: object,
) -> tuple[GrowingMLP, list, PipelineConfig, _FGDTheoryState, float]:
    torch.manual_seed(0)
    model = GrowingMLP(
        in_features=2,
        out_features=1,
        hidden_size=16,
        number_hidden_layers=1,
        device=torch.device("cpu"),
    )
    x = torch.randn(16, 2)
    y = torch.randn(16, 1) * 0.1
    batches = [(x, y)]
    defaults: dict[str, object] = {
        "optimizer": "adam",
        "inner_learning_rate": 1e-2,
        "inner_steps": (100,),
        "functional_learning_rates": (0.5,),
    }
    defaults.update(descent_overrides)
    config = PipelineConfig(
        # theory_mu = 2 and theory_loss_star = 0 are EXACT for sum-MSE in
        # function space; with them the certified contraction equals the
        # realized validation loss ratio.
        fgd_approx=FGDApproxConfig(theory_mu=2.0, theory_loss_star=0.0),
        parametric_descent=ParametricDescentConfig(**defaults),
    )
    model.eval()
    with torch.no_grad():
        loss = float(torch.sum((model(x) - y) ** 2))
    state = _FGDTheoryState(
        epoch_count=0,
        min_gradient_sq_norm=None,
        min_positive_learning_rate=None,
        min_descent_coefficient=None,
        global_contraction_product=1.0,
        previous_validation_functional_loss=loss,
    )
    return model, batches, config, state, loss


def _run_search(model, batches, config, state, loss):
    return _search_parametric_descent_candidate(
        base_model=model,
        train_batches=batches,
        validation_loader=batches,
        loss_function=torch.nn.MSELoss(),
        device=torch.device("cpu"),
        accuracy_tolerance=0.1,
        config=config,
        classification=False,
        theory_state=state,
        initial_functional_gap=loss,
        theory_loss_star=0.0,
    )


def test_measured_descent_certifies_and_contraction_is_realized_ratio() -> None:
    model, batches, config, state, loss = _descent_problem()
    result = _run_search(model, batches, config, state, loss)
    trial = result.accepted
    assert trial is not None
    assert trial.all_conditions_valid
    assert trial.loss_descent_valid
    assert trial.stationary_bound_valid is True
    assert trial.global_bound_valid is True
    # With mu = 2, L* = 0 the certified contraction is EXACTLY the realized
    # validation loss ratio L_after / L_before: the envelope is tight by
    # construction, which is the Prop. 3.8 identity for sum-MSE.
    assert trial.global_contraction == pytest.approx(
        trial.validation_functional_loss / loss, rel=1e-9
    )
    # The measured descent coefficient satisfies the descent inequality with
    # equality: D = eta* r |grad L|^2 with |grad L|^2 = 4 L.
    eta_star = trial.epoch_result.learning_rate
    coefficient = trial.certificate.theory_descent_coefficient
    assert coefficient is not None and coefficient > 0.0
    descent = loss - trial.validation_functional_loss
    assert descent == pytest.approx(
        eta_star * coefficient * 4.0 * loss, rel=1e-6
    )
    # Crel and the LR interval are diagnostics, never gates, for this family.
    assert trial.certificate.relative_error_condition_valid is None
    assert trial.certificate.learning_rate_interval_valid is None
    assert trial.certificate.relative_error is not None


def test_measured_descent_certifies_with_adamw_generator() -> None:
    """The certificate admits any generator; adamw must certify like adam."""
    model, batches, config, state, loss = _descent_problem(
        optimizer="adamw",
        inner_learning_rate=1e-2,
    )
    result = _run_search(model, batches, config, state, loss)
    trial = result.accepted
    assert trial is not None
    assert trial.all_conditions_valid
    assert trial.validation_functional_loss < loss


def test_measured_descent_rejects_when_progress_floor_unreachable() -> None:
    # min_progress = 1.0 demands D >= 4 L, impossible since D <= L.
    model, batches, config, state, loss = _descent_problem(min_progress=1.0)
    result = _run_search(model, batches, config, state, loss)
    assert result.accepted is None
    assert result.last_trial is not None
    assert result.last_trial.all_conditions_valid is False


def test_search_commits_the_largest_certified_progress() -> None:
    """Among certified candidates the search must pick the biggest step."""
    model, batches, config, state, loss = _descent_problem(
        inner_steps=(1, 200),
    )
    result = _run_search(model, batches, config, state, loss)
    trial = result.accepted
    assert trial is not None

    from stable_tiny.pipeline import _evaluate_parametric_descent_trial

    losses_by_budget = {}
    for steps in (1, 200):
        candidate = _evaluate_parametric_descent_trial(
            base_model=model,
            train_batches=batches,
            validation_loader=batches,
            loss_function=torch.nn.MSELoss(),
            device=torch.device("cpu"),
            functional_learning_rate=0.5,
            steps=steps,
            accuracy_tolerance=0.1,
            config=config,
            classification=False,
            theory_state=state,
            initial_functional_gap=loss,
            theory_loss_star=0.0,
        )
        assert candidate is not None
        losses_by_budget[steps] = candidate.validation_functional_loss
    assert losses_by_budget[200] < losses_by_budget[1]
    assert trial.validation_functional_loss == pytest.approx(
        losses_by_budget[200], rel=1e-6
    )


def test_measured_descent_direction_screen_rejects_everything() -> None:
    model, batches, config, state, loss = _descent_problem(min_cosine=1.0)
    result = _run_search(model, batches, config, state, loss)
    assert result.accepted is None
    # Screen failures never reach certification.
    assert result.last_trial is None


def test_all_families_config_wires_the_full_ladder() -> None:
    config = load_pipeline_config(
        REPO_ROOT / "configs" / "fgd" / "mnist_3x2_all_families.yaml"
    )
    # The ladder composition is a user toggle in this config; only check
    # that it is a valid order with the measured-descent family available.
    validate_family_order(config.fgd_approx.family_order)
    assert "parametric_descent" in config.fgd_approx.family_order
    # growth_function_preserving is a user toggle in this config; only check
    # that it parses as a boolean.
    assert isinstance(config.fgd_approx.growth_function_preserving, bool)
    assert config.parametric_descent.functional_learning_rates == (0.5, 0.2)
    assert config.parametric_descent.min_progress == pytest.approx(1e-3)


def test_base_config_keeps_families_commented() -> None:
    config = load_pipeline_config(
        REPO_ROOT / "configs" / "fgd" / "mnist_3x2_fp_growth.yaml"
    )
    assert config.fgd_approx.family_order == ("tangent",)
    config.parametric_descent.validate()
