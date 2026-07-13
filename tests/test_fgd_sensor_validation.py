from __future__ import annotations

from dataclasses import replace

import torch
from torch.utils.data import DataLoader, TensorDataset

from stable_tiny.fgd_approx import (
    FGDApproxConfig,
    FGDApproxEpochResult,
    FGDOutputRelError,
    FGDValidationCertificate,
    _projection_sensor_valid,
    _solve_tangent_projection,
    _TangentProjectionStep,
    evaluate_fgd_validation_certificate,
    should_trigger_fgd_growth,
    train_one_epoch_fgd_approx,
)
from stable_tiny.pipeline import (
    build_dataloaders,
    build_model,
    load_pipeline_config,
    run_pipeline,
)


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


def test_one_invalid_validation_batch_invalidates_full_certificate(
    monkeypatch,
) -> None:
    invalid_step = _TangentProjectionStep(
        output_error=FGDOutputRelError(0.0, 1.0, 1.0, -1.0),
        parameter_updates=(),
        learning_rate_used=0.0,
        loss_before=0.0,
        loss_after=0.0,
        descent_ok=True,
        dot_product=-1.0,
        approximation_sq_norm=1.0,
        target_sq_norm=1.0,
    )
    valid_step = _TangentProjectionStep(
        output_error=FGDOutputRelError(0.0, 1.0, 1.0, 1.0),
        parameter_updates=(),
        learning_rate_used=0.0,
        loss_before=0.0,
        loss_after=0.0,
        descent_ok=True,
        dot_product=1.0,
        approximation_sq_norm=1.0,
        target_sq_norm=1.0,
    )
    steps = iter((invalid_step, valid_step))
    monkeypatch.setattr(
        "stable_tiny.fgd_approx._compute_tangent_projection_step",
        lambda **_: next(steps),
    )

    certificate = evaluate_fgd_validation_certificate(
        model=torch.nn.Linear(1, 1),
        data_loader=[
            (torch.zeros(1, 1), torch.zeros(1, 1)),
            (torch.ones(1, 1), torch.ones(1, 1)),
        ],
        device=torch.device("cpu"),
        config=FGDApproxConfig(),
        learning_rate=0.01,
    )

    assert certificate.sensor_valid is False
    assert certificate.sensor_invalid_batches == 1
    assert certificate.relative_error is None
    assert certificate.relative_error_condition_valid is None
    assert certificate.max_valid_learning_rate is None


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


def test_fgd_growth_is_not_delayed_by_epoch_or_dwell_constraints() -> None:
    config = replace(
        FGDApproxConfig(),
        rel_error_threshold=0.2,
        start_epoch=100,
        min_epochs_between_growth=100,
    )

    assert should_trigger_fgd_growth(
        relative_error=0.2,
        epoch=1,
        last_growth_epoch=1,
        config=config,
    )
    assert not should_trigger_fgd_growth(
        relative_error=0.19,
        epoch=100,
        last_growth_epoch=None,
        config=config,
    )


def test_train_epoch_does_not_evaluate_theory_conditions(monkeypatch) -> None:
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
        model=replace(config.model, hidden_size=2),
        training=replace(config.training, device="cpu"),
        fgd_approx=replace(
            config.fgd_approx,
            rel_error_threshold=0.0,
            projection_solver="exact_svd",
        ),
    )
    train_loader, _, test_loader = build_dataloaders(
        config,
        torch.device("cpu"),
    )
    model = build_model(config, torch.device("cpu"))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("training must not evaluate validation theory conditions")

    monkeypatch.setattr(
        "stable_tiny.fgd_approx.theoretical_learning_rate_upper_bound",
        fail_if_called,
    )
    monkeypatch.setattr(
        "stable_tiny.fgd_approx.theoretical_descent_coefficient",
        fail_if_called,
    )
    monkeypatch.setattr(
        "stable_tiny.fgd_approx.select_tiny_growth_layer_index",
        lambda **_: None,
    )

    result = train_one_epoch_fgd_approx(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_function=torch.nn.MSELoss(),
        device=torch.device("cpu"),
        learning_rate=0.01,
        accuracy_tolerance=1.0,
        config=config.fgd_approx,
    )

    assert result.min_positive_learning_rate == 0.01
    assert result.learning_rate_interval_valid is None
    assert result.relative_error_condition_valid is None
    assert result.theory_descent_coefficient is None
    assert result.learning_rate_clipped_batches == 0


def test_pipeline_clips_learning_rate_from_validation_certificate(
    monkeypatch,
    tmp_path,
) -> None:
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        training=replace(config.training, epochs=1, device="cpu", log_every=1),
        fgd_approx=replace(
            config.fgd_approx,
            theory_lr_initial=0.01,
            theory_lr_follow_bound=False,
            global_bound_action="ignore",
        ),
        growth_schedule=replace(config.growth_schedule, enabled=False),
        wandb=replace(config.wandb, enabled=False),
        run=replace(
            config.run,
            results_dir=tmp_path,
            save_plot=False,
            show_plot=False,
        ),
    )
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(4, config.data.in_features, generator=generator)
    y = torch.randn(4, config.data.out_features, generator=generator)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=4)
    validation_loader = DataLoader(TensorDataset(x + 1.0, y), batch_size=4)
    test_loader = DataLoader(TensorDataset(x + 2.0, y), batch_size=4)
    monkeypatch.setattr(
        "stable_tiny.pipeline.build_dataloaders",
        lambda *_: (train_loader, validation_loader, test_loader),
    )

    certificate_loaders = []

    def validation_certificate(**kwargs):
        certificate_loaders.append(kwargs["data_loader"])
        learning_rate = kwargs["learning_rate"]
        return FGDValidationCertificate(
            learning_rate_upper_bound=0.004,
            max_valid_learning_rate=0.004,
            learning_rate_interval_valid=(
                learning_rate is None or learning_rate <= 0.004
            ),
            skipped_batches=0,
            relative_error_condition_valid=True,
            gradient_sq_norm=1.0,
            theory_descent_coefficient=0.5,
            relative_error=0.1,
            output_relative_error=FGDOutputRelError(
                relative_error=0.1,
                approximation_norm=0.9,
                target_norm=1.0,
                directional_cosine=1.0,
            ),
            sensor_valid=True,
            sensor_invalid_batches=0,
        )

    monkeypatch.setattr(
        "stable_tiny.pipeline.evaluate_fgd_validation_certificate",
        validation_certificate,
    )
    observed_learning_rates = []

    def train_epoch(**kwargs):
        observed_learning_rates.append(kwargs["learning_rate"])
        return FGDApproxEpochResult(
            train_loss=1.0,
            train_accuracy=0.0,
            test_loss=1.0,
            test_accuracy=0.0,
            learning_rate=kwargs["learning_rate"],
            next_learning_rate=None,
            learning_rate_upper_bound=None,
            learning_rate_interval_valid=None,
            learning_rate_clipped_batches=0,
            skipped_batches=0,
            relative_error_condition_valid=None,
            loss_descent_valid=False,
            loss_non_descent_batches=99,
            gradient_sq_norm=None,
            theory_descent_coefficient=None,
            min_positive_learning_rate=kwargs["learning_rate"],
            relative_error=None,
            selected_layer_index=None,
            layer_relative_errors=[],
            output_relative_error=None,
            sensor_valid=True,
            sensor_invalid_batches=0,
        )

    monkeypatch.setattr(
        "stable_tiny.pipeline.train_one_epoch_fgd_approx",
        train_epoch,
    )

    result = run_pipeline(config, progress=None)

    assert observed_learning_rates == [0.004]
    assert certificate_loaders
    assert all(loader is validation_loader for loader in certificate_loaders)
    assert result.history[1].learning_rate == 0.004
    assert result.history[1].fgd_learning_rate_clipped_batches == 1
    assert result.history[1].fgd_loss_descent_valid is True
    assert result.history[1].fgd_loss_non_descent_batches == 0
