from __future__ import annotations

from dataclasses import replace

import torch
from torch.utils.data import DataLoader, TensorDataset

from fgdlib.tangent import (
    FGDApproxConfig,
    FGDApproxEpochResult,
    FGDOutputRelError,
    FGDValidationCertificate,
    _FunctionalStepStats,
    _output_relative_error_from_stats,
    _projection_sensor_valid,
    _solve_tangent_projection,
    _TangentProjectionStep,
    evaluate_fgd_validation_certificate,
    evaluate_secant_validation_certificate,
    should_trigger_fgd_growth,
    theoretical_learning_rate_upper_bound,
    train_one_epoch_fgd_approx,
)
from stable_tiny.pipeline import (
    _FGDTheoryState,
    _FGDTrial,
    _search_fgd_certified_trial,
    _search_secant_fgd_candidate,
    build_dataloaders,
    build_model,
    evaluate_functional_loss,
    load_pipeline_config,
    run_pipeline,
)
from fgdlib.growth import GrowthResult


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


def _direction_mocks(
    monkeypatch,
    *,
    rel_error_stats,
    direction_value: float | None = None,
    direction_from_gradient: bool = False,
):
    """Mock the shared-direction solve and its validation measurement.

    ``rel_error_stats`` is a (dot, approx_sq, target_sq) triple describing
    the direction's image on the validation probe; the certificate and the
    learning-rate search derive everything else from it. The direction is
    either a constant fill (``direction_value``) or the true loss gradient
    at the probe (``direction_from_gradient``), which guarantees strict
    descent of theta - eta * u for small eta.
    """
    dot, approx_sq, target_sq = rel_error_stats
    stats = _FunctionalStepStats(
        output_error=_output_relative_error_from_stats(
            dot_product=dot,
            approximation_sq_norm=approx_sq,
            target_sq_norm=target_sq,
            eps=1e-12,
        ),
        dot_product=dot,
        approximation_sq_norm=approx_sq,
        target_sq_norm=target_sq,
    )

    def direction_step(*, model, x, y, config):
        del config
        parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if direction_from_gradient:
            output = model(x)
            loss = torch.sum((output - y) ** 2)
            gradients = torch.autograd.grad(loss, parameters)
            updates = tuple(gradient.detach() for gradient in gradients)
        else:
            updates = tuple(
                torch.full_like(parameter, direction_value)
                for parameter in parameters
            )
        return _TangentProjectionStep(
            output_error=stats.output_error,
            parameter_updates=updates,
            learning_rate_used=0.0,
            loss_before=1.0,
            loss_after=1.0,
            descent_ok=True,
            dot_product=stats.dot_product,
            approximation_sq_norm=stats.approximation_sq_norm,
            target_sq_norm=stats.target_sq_norm,
        )

    monkeypatch.setattr(
        "stable_tiny.pipeline._compute_tangent_projection_step",
        direction_step,
    )
    monkeypatch.setattr(
        "stable_tiny.pipeline.measure_direction_projection",
        lambda *args, **kwargs: stats,
    )
    return stats


def _fgd_epoch_result(
    learning_rate: float,
    *,
    selected_layer_index: int | None = None,
) -> FGDApproxEpochResult:
    return FGDApproxEpochResult(
        train_loss=1.0,
        train_accuracy=0.0,
        test_loss=1.0,
        test_accuracy=0.0,
        learning_rate=learning_rate,
        next_learning_rate=None,
        learning_rate_upper_bound=None,
        learning_rate_interval_valid=None,
        learning_rate_clipped_batches=0,
        skipped_batches=0,
        relative_error_condition_valid=None,
        loss_descent_valid=None,
        loss_non_descent_batches=0,
        gradient_sq_norm=None,
        theory_descent_coefficient=None,
        min_positive_learning_rate=learning_rate,
        relative_error=None,
        selected_layer_index=selected_layer_index,
        layer_relative_errors=[],
        output_relative_error=None,
        sensor_valid=True,
        sensor_invalid_batches=0,
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
        "fgdlib.tangent._compute_tangent_projection_step",
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
        "fgdlib.tangent._compute_tangent_projection_step",
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


def test_secant_certificate_allows_non_projector_hilbert_approximation() -> None:
    base_model = torch.nn.Linear(1, 1)
    candidate_model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        base_model.weight.zero_()
        base_model.bias.zero_()
        candidate_model.weight.zero_()
        candidate_model.bias.fill_(0.24)
    loader = DataLoader(
        TensorDataset(torch.zeros(4, 1), torch.ones(4, 1)),
        batch_size=4,
    )

    certificate = evaluate_secant_validation_certificate(
        base_model=base_model,
        candidate_model=candidate_model,
        data_loader=loader,
        device=torch.device("cpu"),
        config=FGDApproxConfig(),
        learning_rate=0.1,
    )

    assert certificate.sensor_valid is True
    assert certificate.relative_error_condition_valid is True
    assert certificate.learning_rate_interval_valid is True
    assert certificate.output_relative_error is not None
    assert (
        certificate.output_relative_error.approximation_norm
        > certificate.output_relative_error.target_norm
    )


def test_secant_search_keeps_architecture_fixed() -> None:
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            train_batches=1,
            validation_batches=1,
            test_batches=1,
            batch_size=8,
        ),
        model=replace(config.model, hidden_size=2, number_hidden_layers=2),
        training=replace(config.training, device="cpu"),
        fgd_approx=replace(
            config.fgd_approx,
            theory_mu=1e-15,
            projection_group_auto=False,
        ),
    )
    device = torch.device("cpu")
    train_loader, _, _ = build_dataloaders(config, device)
    model = build_model(config, device)
    functional_loss = evaluate_functional_loss(model, train_loader, device)

    result = _search_secant_fgd_candidate(
        model=model,
        train_batches=list(train_loader),
        validation_loader=train_loader,
        loss_function=torch.nn.MSELoss(),
        device=device,
        accuracy_tolerance=config.training.accuracy_tolerance,
        config=config,
        classification=False,
        theory_state=_FGDTheoryState(
            0,
            None,
            None,
            None,
            1.0,
            functional_loss,
        ),
        initial_functional_gap=functional_loss,
        theory_loss_star=0.0,
    )

    assert result.accepted is not None
    assert sum(parameter.numel() for parameter in model.parameters()) == sum(
        parameter.numel() for parameter in result.accepted.model.parameters()
    )


def test_pipeline_uses_rkhs_phase_when_growth_does_not_improve_fgd(
    monkeypatch,
    tmp_path,
) -> None:
    """The certified RKHS head phase replaces the Hilbert-secant search."""
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(config.model, hidden_size=2, number_hidden_layers=2),
        training=replace(config.training, epochs=1, device="cpu", log_every=1),
        fgd_approx=replace(
            config.fgd_approx,
            projection_damping=1e6,
            projection_group_auto=False,
            projection_group_size=1,
            theory_mu=1e-15,
            theory_lr_search_steps=1,
            theory_lr_search_refinements=0,
        ),
        secant_fgd=replace(
            config.secant_fgd,
            search_steps=1,
            growth_min_relative_error_improvement=1e9,
            growth_min_learning_rate_improvement=1e9,
        ),
        scaling_line_search=replace(config.scaling_line_search, iterations=0),
        wandb=replace(config.wandb, enabled=False),
        run=replace(
            config.run,
            results_dir=tmp_path,
            save_plot=False,
            show_plot=False,
        ),
    )
    generator = torch.Generator().manual_seed(4)
    train_x = torch.randn(8, config.data.in_features, generator=generator)
    validation_x = torch.randn(8, config.data.in_features, generator=generator)
    targets = torch.zeros(8, config.data.out_features)
    train_loader = DataLoader(TensorDataset(train_x, targets), batch_size=8)
    validation_loader = DataLoader(
        TensorDataset(validation_x, targets),
        batch_size=8,
    )
    monkeypatch.setattr(
        "stable_tiny.pipeline.build_dataloaders",
        lambda *_: (train_loader, validation_loader, validation_loader),
    )

    result = run_pipeline(config, progress=None)

    initial_entry = next(entry for entry in result.history if entry.step_type == "INIT")
    phase_entry = next(entry for entry in result.history if entry.step_type == "RKHS")
    assert result.growth_events == []
    assert phase_entry.fgd_growth_probe_improved is False
    assert phase_entry.fgd_rkhs_phase_attempted is True
    assert phase_entry.fgd_rkhs_phase_accepted is True
    assert phase_entry.fgd_candidate_accepted is True
    assert phase_entry.fgd_approximation_kind == "rkhs_head"
    assert phase_entry.fgd_rkhs_loss_star is not None
    # The phase never changes the architecture: same parameter count.
    assert phase_entry.num_params == initial_entry.num_params


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


def test_lr_search_returns_largest_valid_rate_found() -> None:
    threshold = 0.003
    rates: list[float] = []

    def evaluate_trial(learning_rate: float) -> _FGDTrial:
        rates.append(learning_rate)
        condition_valid = learning_rate <= threshold
        certificate = FGDValidationCertificate(
            learning_rate_upper_bound=0.008 / 0.95,
            max_valid_learning_rate=0.008,
            learning_rate_interval_valid=True,
            skipped_batches=0,
            relative_error_condition_valid=True,
            gradient_sq_norm=1.0,
            theory_descent_coefficient=0.5,
            relative_error=0.1,
            output_relative_error=FGDOutputRelError(0.1, 0.9, 1.0, 1.0),
            sensor_valid=True,
            sensor_invalid_batches=0,
        )
        theory_state = _FGDTheoryState(1, 1.0, learning_rate, 0.5, 1.0, 1.0)
        return _FGDTrial(
            model=torch.nn.Linear(1, 1),
            epoch_result=_fgd_epoch_result(learning_rate),
            certificate=certificate,
            theory_state=theory_state,
            validation_functional_loss=1.0,
            loss_descent_valid=True,
            stationary_bound=1.0,
            stationary_bound_valid=True,
            global_bound=1.0,
            global_bound_valid=condition_valid,
            global_contraction=1.0,
            all_conditions_valid=condition_valid,
        )

    result = _search_fgd_certified_trial(
        maximum_learning_rate=0.008,
        evaluate_trial=evaluate_trial,
        config=replace(
            FGDApproxConfig(),
            theory_lr_search_steps=4,
            theory_lr_search_refinements=12,
        ),
    )

    assert rates[0] == 0.008
    assert result.accepted is not None
    accepted_rate = result.accepted.epoch_result.min_positive_learning_rate
    assert accepted_rate is not None
    assert 0.00299 <= accepted_rate <= threshold


def test_lr_search_rejects_rates_below_positive_theory_floor() -> None:
    rates: list[float] = []

    def evaluate_trial(learning_rate: float) -> _FGDTrial:
        rates.append(learning_rate)
        condition_valid = learning_rate < 1e-5
        certificate = FGDValidationCertificate(
            learning_rate_upper_bound=1e-3 / 0.95,
            max_valid_learning_rate=1e-3,
            learning_rate_interval_valid=True,
            skipped_batches=0,
            relative_error_condition_valid=True,
            gradient_sq_norm=1.0,
            theory_descent_coefficient=0.5,
            relative_error=0.1,
            output_relative_error=FGDOutputRelError(0.1, 0.9, 1.0, 1.0),
            sensor_valid=True,
            sensor_invalid_batches=0,
        )
        return _FGDTrial(
            model=torch.nn.Linear(1, 1),
            epoch_result=_fgd_epoch_result(learning_rate),
            certificate=certificate,
            theory_state=_FGDTheoryState(1, 1.0, learning_rate, 0.5, 1.0, 1.0),
            validation_functional_loss=1.0,
            loss_descent_valid=True,
            stationary_bound=1.0,
            stationary_bound_valid=True,
            global_bound=1.0,
            global_bound_valid=condition_valid,
            global_contraction=1.0,
            all_conditions_valid=condition_valid,
        )

    result = _search_fgd_certified_trial(
        maximum_learning_rate=1e-3,
        evaluate_trial=evaluate_trial,
        config=replace(
            FGDApproxConfig(),
            theory_lr_min=1e-5,
            theory_lr_search_steps=8,
            theory_lr_search_refinements=4,
        ),
    )

    assert result.accepted is None
    assert rates
    assert min(rates) > 1e-5


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
        "fgdlib.tangent.theoretical_learning_rate_upper_bound",
        fail_if_called,
    )
    monkeypatch.setattr(
        "fgdlib.tangent.theoretical_descent_coefficient",
        fail_if_called,
    )
    monkeypatch.setattr(
        "fgdlib.tangent.select_tiny_growth_layer_index",
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
            theory_mu=1e-15,
            theory_lr_search_steps=6,
            theory_lr_search_refinements=2,
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
    # Train and validation share the SAME tensors so the gradient direction
    # mocked below strictly descends the validation functional at small eta
    # (the strict-descent gate would reject a do-nothing direction).
    train_loader = DataLoader(TensorDataset(x, y), batch_size=4)
    validation_loader = DataLoader(TensorDataset(x, y), batch_size=4)
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
    stats = _direction_mocks(
        monkeypatch,
        direction_from_gradient=True,
        rel_error_stats=(0.9, 0.81, 1.0),
    )

    result = run_pipeline(config, progress=None)

    # The state certificate still clips the epoch's declared learning rate.
    assert certificate_loaders
    assert all(loader is validation_loader for loader in certificate_loaders)
    assert result.history[1].fgd_learning_rate_clipped_batches == 1
    # The committed step is ONE outer update at a rate inside the interval
    # certified for the ACTUAL direction (its validation relative error),
    # not an epoch of training at the state-certified rate.
    direction_bound = theoretical_learning_rate_upper_bound(
        stats.output_error.relative_error,
        config.fgd_approx,
    )
    assert direction_bound is not None
    maximum_learning_rate = config.fgd_approx.theory_lr_safety * direction_bound
    assert result.history[1].fgd_candidate_accepted is True
    accepted_learning_rate = result.history[1].learning_rate
    assert 0.0 < accepted_learning_rate <= maximum_learning_rate + 1e-12
    assert result.history[1].fgd_update_norm > 0.0
    assert result.history[1].fgd_loss_descent_valid is True
    assert result.history[1].fgd_loss_non_descent_batches == 0


def test_failed_lr_trials_do_not_modify_the_committed_model(
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
            theory_lr_search_steps=2,
            theory_lr_search_refinements=0,
            global_bound_action="lr_then_growth",
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
    # Targets equal the initial model's own outputs: the base loss is
    # exactly zero, so NO perturbation can strictly descend and every
    # candidate fails the local descent gate.
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(4, config.data.in_features, generator=generator)
    with torch.no_grad():
        y = build_model(config, torch.device("cpu"))(x).detach()
    train_loader = DataLoader(TensorDataset(x, y), batch_size=4)
    validation_loader = DataLoader(TensorDataset(x, y), batch_size=4)
    test_loader = DataLoader(TensorDataset(x, y), batch_size=4)
    monkeypatch.setattr(
        "stable_tiny.pipeline.build_dataloaders",
        lambda *_: (train_loader, validation_loader, test_loader),
    )

    def validation_certificate(**kwargs):
        maximum = 0.008
        learning_rate = kwargs["learning_rate"]
        return FGDValidationCertificate(
            learning_rate_upper_bound=maximum / 0.95,
            max_valid_learning_rate=maximum,
            learning_rate_interval_valid=(
                learning_rate is None or learning_rate <= maximum
            ),
            skipped_batches=0,
            relative_error_condition_valid=False,
            gradient_sq_norm=1.0,
            theory_descent_coefficient=0.5,
            relative_error=0.6,
            output_relative_error=FGDOutputRelError(0.6, 0.4, 1.0, 1.0),
            sensor_valid=True,
            sensor_invalid_batches=0,
        )

    monkeypatch.setattr(
        "stable_tiny.pipeline.evaluate_fgd_validation_certificate",
        validation_certificate,
    )
    # A NONZERO direction: every trial visibly mutates its clone, so any
    # leakage into the committed model would be detected below.
    _direction_mocks(
        monkeypatch,
        direction_value=1.0,
        rel_error_stats=(0.9, 0.81, 1.0),
    )

    initial_model = build_model(config, torch.device("cpu"))
    initial_state = {
        name: tensor.detach().clone()
        for name, tensor in initial_model.state_dict().items()
    }
    result = run_pipeline(config, progress=None)

    assert result.history[1].learning_rate == 0.0
    assert result.history[1].fgd_candidate_accepted is False
    assert result.history[1].fgd_lr_search_trials == 2
    assert not result.growth_events
    for name, tensor in result.model.state_dict().items():
        assert torch.equal(tensor, initial_state[name])


def test_growth_sets_lr_to_new_architecture_certified_maximum(
    monkeypatch,
    tmp_path,
) -> None:
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        training=replace(config.training, epochs=1, device="cpu", log_every=1),
        fgd_approx=replace(
            config.fgd_approx,
            rel_error_threshold=0.1,
            layer_selection="tiny_best",
            theory_lr_follow_bound=False,
            theory_lr_search_steps=2,
            theory_lr_search_refinements=0,
        ),
        growth_schedule=replace(config.growth_schedule, enabled=True),
        wandb=replace(config.wandb, enabled=False),
        run=replace(
            config.run,
            results_dir=tmp_path,
            save_plot=False,
            show_plot=False,
        ),
    )
    x = torch.zeros(4, config.data.in_features)
    y = torch.ones(4, config.data.out_features)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=4)
    validation_loader = DataLoader(TensorDataset(x + 1.0, y), batch_size=4)
    test_loader = DataLoader(TensorDataset(x + 2.0, y), batch_size=4)
    monkeypatch.setattr(
        "stable_tiny.pipeline.build_dataloaders",
        lambda *_: (train_loader, validation_loader, test_loader),
    )

    def validation_certificate(**kwargs):
        maximum = 0.007 if kwargs["learning_rate"] is None else 0.004
        return FGDValidationCertificate(
            learning_rate_upper_bound=maximum / 0.95,
            max_valid_learning_rate=maximum,
            learning_rate_interval_valid=True,
            skipped_batches=0,
            relative_error_condition_valid=False,
            gradient_sq_norm=1.0,
            theory_descent_coefficient=0.5,
            relative_error=0.2,
            output_relative_error=FGDOutputRelError(0.2, 0.8, 1.0, 1.0),
            sensor_valid=True,
            sensor_invalid_batches=0,
        )

    monkeypatch.setattr(
        "stable_tiny.pipeline.evaluate_fgd_validation_certificate",
        validation_certificate,
    )

    def train_epoch(**kwargs):
        learning_rate = kwargs["learning_rate"]
        assert kwargs["evaluate_test"] is False
        return _fgd_epoch_result(learning_rate, selected_layer_index=0)

    monkeypatch.setattr(
        "stable_tiny.pipeline.train_one_epoch_fgd_approx",
        train_epoch,
    )
    monkeypatch.setattr(
        "stable_tiny.pipeline.select_tiny_growth_layer_index",
        lambda **_: 0,
    )
    monkeypatch.setattr(
        "stable_tiny.pipeline.grow_layer",
        lambda **_: GrowthResult(0, 1.0, 1.0, []),
    )

    result = run_pipeline(config, progress=None)

    assert len(result.growth_events) == 1
    assert result.history[1].learning_rate == 0.0
    assert result.history[1].fgd_candidate_accepted is False
    assert result.history[-1].step_type == "GRO"
    assert result.history[-1].learning_rate == 0.007
    assert result.history[-1].fgd_max_valid_learning_rate == 0.007
