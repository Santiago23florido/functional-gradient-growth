from __future__ import annotations

from dataclasses import replace

import torch

from stable_tiny.fgd_approx import (
    FGDApproxConfig,
    FGDOutputRelError,
    _projection_sensor_valid,
    _solve_tangent_projection,
    _TangentProjectionStep,
    evaluate_fgd_validation_certificate,
)
from stable_tiny.pipeline import build_dataloaders, load_pipeline_config


def _assert_projection_invariants(
    jacobian: torch.Tensor,
    target: torch.Tensor,
    damping: float,
) -> None:
    _, approximation = _solve_tangent_projection(jacobian, target, damping)
    dot_product = float(torch.sum(approximation * target).item())
    approximation_sq_norm = float(torch.sum(approximation**2).item())
    target_sq_norm = float(torch.sum(target**2).item())

    assert torch.isfinite(approximation).all()
    assert _projection_sensor_valid(
        dot_product=dot_product,
        approximation_sq_norm=approximation_sq_norm,
        target_sq_norm=target_sq_norm,
        eps=1e-12,
    )


def test_spectral_solver_preserves_projection_invariants() -> None:
    torch.manual_seed(0)
    jacobian = torch.randn(16, 8, dtype=torch.float32)
    target = torch.randn(16, dtype=torch.float32)

    _assert_projection_invariants(jacobian, target, damping=1e-2)


def test_spectral_solver_handles_ill_conditioned_jacobian() -> None:
    torch.manual_seed(1)
    left, _ = torch.linalg.qr(torch.randn(12, 12, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(12, 12, dtype=torch.float64))
    singular_values = torch.logspace(0, -10, 12, dtype=torch.float64)
    jacobian = (left @ torch.diag(singular_values) @ right.t()).to(torch.float32)
    target = torch.randn(12, dtype=torch.float32)

    _assert_projection_invariants(jacobian, target, damping=1e-2)


def test_invalid_sensor_stats_do_not_form_growth_condition(monkeypatch) -> None:
    invalid_step = _TangentProjectionStep(
        output_error=FGDOutputRelError(
            relative_error=0.0,
            approximation_norm=1.0,
            target_norm=1.0,
            directional_cosine=-1.0,
        ),
        parameter_updates=(),
        learning_rate_used=0.0,
        loss_before=0.0,
        loss_after=0.0,
        descent_ok=True,
        dot_product=-1.0,
        approximation_sq_norm=1.0,
        target_sq_norm=1.0,
    )

    monkeypatch.setattr(
        "stable_tiny.fgd_approx._compute_tangent_projection_step",
        lambda **_: invalid_step,
    )
    certificate = evaluate_fgd_validation_certificate(
        model=torch.nn.Linear(1, 1),
        data_loader=[(torch.zeros(1, 1), torch.zeros(1, 1))],
        device=torch.device("cpu"),
        config=FGDApproxConfig(),
        learning_rate=0.01,
    )

    assert certificate.sensor_valid is False
    assert certificate.relative_error_condition_valid is None
    assert certificate.relative_error_condition_valid is not False


def test_build_dataloaders_returns_distinct_validation_split() -> None:
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            train_batches=1,
            validation_batches=1,
            test_batches=1,
            batch_size=4,
        ),
        training=replace(config.training, device="cpu"),
    )

    train_loader, validation_loader, test_loader = build_dataloaders(
        config,
        torch.device("cpu"),
    )
    train_x, train_y = next(iter(train_loader))
    validation_x, validation_y = next(iter(validation_loader))
    test_x, test_y = next(iter(test_loader))

    assert len(train_loader) == 1
    assert len(validation_loader) == 1
    assert len(test_loader) == 1
    assert not torch.equal(train_x, validation_x)
    assert not torch.equal(train_y, validation_y)
    assert not torch.equal(validation_x, test_x)
    assert not torch.equal(validation_y, test_y)
