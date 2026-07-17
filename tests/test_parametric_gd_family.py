"""Parametric-GD secant family: projection screen, calibration, ladder config."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.tangent import (
    FGDApproxConfig,
    ParametricGDConfig,
    validate_family_order,
)
from stable_tiny.pipeline import (
    PipelineConfig,
    _FGDTheoryState,
    _measure_secant_projection,
    _search_parametric_gd_candidate,
    load_pipeline_config,
)

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

class _FixedOutputModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output

def test_validate_family_order_rules() -> None:
    validate_family_order(("tangent",))
    validate_family_order(("tangent", "rkhs_head", "parametric_gd"))
    with pytest.raises(ValueError):
        validate_family_order(())
    with pytest.raises(ValueError):
        validate_family_order(("rkhs_head", "tangent"))
    with pytest.raises(ValueError):
        validate_family_order(("tangent", "unknown_family"))
    with pytest.raises(ValueError):
        validate_family_order(("tangent", "rkhs_head", "rkhs_head"))

def test_parametric_gd_config_validation() -> None:
    ParametricGDConfig().validate()
    with pytest.raises(ValueError):
        ParametricGDConfig(optimizer="rmsprop").validate()
    with pytest.raises(ValueError):
        ParametricGDConfig(min_cosine=0.0).validate()
    with pytest.raises(ValueError):
        ParametricGDConfig(inner_steps=()).validate()

def test_secant_projection_matches_exact_values() -> None:
    """cos and eta* must match the closed-form aggregates exactly."""
    torch.manual_seed(0)
    x = torch.randn(8, 3)
    y = torch.randn(8, 2).double()
    base_output = torch.randn(8, 2).double()
    delta = torch.randn(8, 2).double()
    base = _FixedOutputModel(base_output)
    candidate = _FixedOutputModel(base_output - delta)
    loader = [(x, y)]

    projection = _measure_secant_projection(
        base_model=base,
        candidate_model=candidate,
        validation_loader=loader,
        device=torch.device("cpu"),
        eps=1e-12,
    )
    assert projection is not None
    cosine, eta_star = projection

    r = 2.0 * (base_output - y)
    dot = float(torch.sum(delta * r))
    expected_cosine = dot / math.sqrt(
        float(torch.sum(delta * delta)) * float(torch.sum(r * r))
    )
    expected_eta_star = dot / float(torch.sum(r * r))
    assert cosine == pytest.approx(expected_cosine, abs=1e-12)
    assert eta_star == pytest.approx(expected_eta_star, abs=1e-12)

    # At eta* the secant relative error is exactly sqrt(1 - cos^2).
    approximation = delta / eta_star
    rel_error = float(torch.norm(approximation - r) / torch.norm(approximation))
    assert rel_error == pytest.approx(
        math.sqrt(max(0.0, 1.0 - expected_cosine**2)), abs=1e-9
    )

def _pgd_problem(
    min_cosine: float,
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
    config = PipelineConfig(
        fgd_approx=FGDApproxConfig(theory_lr_min=1e-6),
        parametric_gd=ParametricGDConfig(
            optimizer="adam",
            inner_learning_rate=1e-2,
            inner_steps=(200,),
            functional_learning_rates=(0.2,),
            min_cosine=min_cosine,
        ),
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

def test_parametric_gd_search_certifies_a_well_aligned_secant() -> None:
    """An overparameterized fit realizes the functional step and certifies."""
    model, batches, config, state, loss = _pgd_problem(min_cosine=0.9)
    result = _search_parametric_gd_candidate(
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
    trial = result.accepted
    assert trial is not None
    assert trial.all_conditions_valid
    assert trial.certificate.relative_error is not None
    assert trial.certificate.relative_error < 0.5
    assert trial.certificate.relative_error_condition_valid is True
    assert trial.certificate.learning_rate_interval_valid is True
    assert trial.loss_descent_valid
    assert trial.stationary_bound_valid is True
    assert trial.global_bound_valid is True
    # The declared learning rate is the calibrated eta*, not a nominal grid
    # value, and it must sit inside the certified interval.
    eta_star = trial.epoch_result.learning_rate
    assert eta_star > 0.0
    assert trial.certificate.max_valid_learning_rate is not None
    assert eta_star <= trial.certificate.max_valid_learning_rate + 1e-12

def test_parametric_gd_cosine_screen_rejects_before_certification() -> None:
    """An unreachable min_cosine must reject every candidate at the screen."""
    model, batches, config, state, loss = _pgd_problem(min_cosine=1.0)
    result = _search_parametric_gd_candidate(
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
    assert result.accepted is None
    # Screen failures never produce a certified last_trial either: the
    # candidate is dropped before the certificate is even computed.
    assert result.last_trial is None
    assert result.trial_count == 1

def test_fp_growth_config_wires_families_and_parametric_gd() -> None:
    config = load_pipeline_config(
        REPO_ROOT / "configs" / "fgd" / "mnist_3x2_fp_growth.yaml"
    )
    assert config.fgd_approx.family_order == ("tangent",)
    assert config.parametric_gd.optimizer == "sgd"
    assert config.parametric_gd.inner_steps == (16, 64)
    assert config.parametric_gd.functional_learning_rates == (0.2, 0.05)
    assert config.parametric_gd.min_cosine == pytest.approx(0.9)

def test_family_order_yaml_validation(tmp_path: Path) -> None:
    source = (
        REPO_ROOT / "configs" / "fgd" / "mnist_3x2_fp_growth.yaml"
    ).read_text(encoding="utf-8")
    bad = source.replace(
        "  family_order:\n    - tangent\n",
        "  family_order:\n    - tangent\n    - not_a_family\n",
    )
    assert bad != source
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_family"):
        load_pipeline_config(bad_path)
