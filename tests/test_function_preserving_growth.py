"""Function-preserving growth for the fgd_approx tangent flow."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.growth import ScalingLineSearchConfig, grow_layer
from fgdlib.tangent import FGDApproxConfig, tiny_optimal_update_kwargs
from stable_tiny.pipeline import load_pipeline_config

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent


def _model_and_loader(
    seed: int = 0,
) -> tuple[GrowingMLP, DataLoader, torch.Tensor]:
    torch.manual_seed(seed)
    model = GrowingMLP(
        in_features=4,
        out_features=2,
        hidden_size=2,
        number_hidden_layers=2,
        device=torch.device("cpu"),
    )
    x = torch.randn(64, 4)
    y = torch.randn(64, 2)
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
    return model, loader, x


def test_function_preserving_growth_keeps_outputs_and_adds_neurons() -> None:
    model, loader, x = _model_and_loader()
    model.eval()
    with torch.no_grad():
        outputs_before = model(x).clone()
    parameters_before = sum(parameter.numel() for parameter in model.parameters())

    result = grow_layer(
        model=model,
        train_loader=loader,
        layer_index=0,
        device=torch.device("cpu"),
        line_search_config=ScalingLineSearchConfig(),
        optimal_update_kwargs=tiny_optimal_update_kwargs(
            FGDApproxConfig(),
            compute_delta=False,
        ),
        function_preserving=True,
    )

    model.eval()
    with torch.no_grad():
        outputs_after = model(x)
    assert torch.allclose(outputs_after, outputs_before, atol=1e-6)
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        > parameters_before
    )
    assert result.best_scaling_factor == 1.0
    assert len(result.line_search) == 1


def test_function_preserving_growth_overrides_tiny_knobs() -> None:
    """omega_zero/compute_delta from the config must not defeat preservation."""
    model, loader, x = _model_and_loader(seed=1)
    model.eval()
    with torch.no_grad():
        outputs_before = model(x).clone()

    grow_layer(
        model=model,
        train_loader=loader,
        layer_index=1,
        device=torch.device("cpu"),
        line_search_config=ScalingLineSearchConfig(),
        optimal_update_kwargs=tiny_optimal_update_kwargs(
            FGDApproxConfig(tiny_omega_zero=False),
            compute_delta=True,
        ),
        function_preserving=True,
    )

    model.eval()
    with torch.no_grad():
        outputs_after = model(x)
    assert torch.allclose(outputs_after, outputs_before, atol=1e-6)


def test_default_growth_is_unchanged_without_the_flag() -> None:
    """The pre-existing line-search growth path stays byte-for-byte reachable."""
    model, loader, _ = _model_and_loader(seed=2)
    result = grow_layer(
        model=model,
        train_loader=loader,
        layer_index=0,
        device=torch.device("cpu"),
        line_search_config=ScalingLineSearchConfig(iterations=2),
        optimal_update_kwargs=tiny_optimal_update_kwargs(
            FGDApproxConfig(),
            compute_delta=False,
        ),
    )
    assert len(result.line_search) > 1


def test_fp_growth_config_wires_flag_and_disables_rkhs_phase() -> None:
    config = load_pipeline_config(
        REPO_ROOT / "configs" / "fgd" / "mnist_3x2_fp_growth.yaml"
    )
    assert config.training.method == "fgd_approx"
    assert config.fgd_approx.growth_function_preserving is True
    assert config.fgd_approx.growth_preservation_tolerance == pytest.approx(1e-6)
    assert config.secant_fgd.enabled is False
