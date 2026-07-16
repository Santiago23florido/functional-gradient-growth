"""Tests for strict finite-space adaptive FGD (Algorithm 1)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fgdlib.adaptive import (
    AdaptiveFGDConfig,
    certify_empirical_secant,
    certify_empirical_secant_models,
    empirical_functional_loss,
    empirical_inner_product,
    empirical_norm,
    grow_layer_function_preserving,
    search_adaptive_fgd_step,
    theory_descent_coefficient,
    theory_learning_rate_upper_bound,
)
from fgdlib.gromo_setup import ensure_gromo_importable
from stable_tiny.pipeline import (
    build_model,
    load_pipeline_config,
    result_payload,
    run_pipeline,
)


ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP  # noqa: E402


def _small_model(seed: int = 0) -> GrowingMLP:
    torch.manual_seed(seed)
    return GrowingMLP(
        in_features=2,
        out_features=1,
        hidden_size=3,
        number_hidden_layers=1,
        device=torch.device("cpu"),
    )


def test_empirical_geometry_and_mse_constants() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[2.0, 0.0], [1.0, -1.0]])
    assert empirical_inner_product(left, right) == pytest.approx(0.5)
    assert empirical_norm(left) ** 2 == pytest.approx(15.0)
    assert empirical_functional_loss(left, torch.zeros_like(left)) == pytest.approx(7.5)
    assert theory_learning_rate_upper_bound(0.0) == pytest.approx(2.0)
    assert theory_descent_coefficient(0.0, 0.5) == pytest.approx(0.75)


def test_exact_functional_step_satisfies_every_certificate_condition() -> None:
    target = torch.zeros(4, 1, dtype=torch.float64)
    base = torch.tensor([[1.0], [2.0], [-1.0], [0.5]], dtype=torch.float64)
    eta = 0.5
    candidate = base - eta * (base - target)
    certificate = certify_empirical_secant(
        base_output=base,
        candidate_output=candidate,
        target=target,
        learning_rate=eta,
        epsilon=0.25,
    )
    assert certificate.accepted
    assert certificate.error_upper_bound == pytest.approx(0.0)
    assert certificate.relative_error == pytest.approx(0.0)
    assert certificate.directional_cosine == pytest.approx(1.0)
    assert certificate.smoothness == certificate.alpha == certificate.beta == 1.0
    assert certificate.mu == 1.0
    assert certificate.contraction == pytest.approx(0.25)
    assert certificate.loss_after <= certificate.predicted_loss_upper_bound + 1e-12


def test_batched_model_certificate_matches_dense_certificate() -> None:
    generator = torch.Generator().manual_seed(42)
    base_model = torch.nn.Linear(3, 2, dtype=torch.float64)
    candidate_model = torch.nn.Linear(3, 2, dtype=torch.float64)
    candidate_model.load_state_dict(base_model.state_dict())
    with torch.no_grad():
        candidate_model.weight.add_(
            0.01 * torch.randn(2, 3, generator=generator, dtype=torch.float64)
        )
    x = torch.randn(11, 3, generator=generator, dtype=torch.float64)
    y = torch.randn(11, 2, generator=generator, dtype=torch.float64)
    eta = 0.05
    with torch.no_grad():
        base_output = base_model(x)
        candidate_output = candidate_model(x)
    dense = certify_empirical_secant(
        base_output=base_output,
        candidate_output=candidate_output,
        target=y,
        learning_rate=eta,
        epsilon=0.9,
    )
    batched = certify_empirical_secant_models(
        base_model=base_model,
        candidate_model=candidate_model,
        x=x,
        y=y,
        learning_rate=eta,
        epsilon=0.9,
        batch_size=3,
    )
    for field in (
        "error_upper_bound",
        "approximation_norm",
        "target_norm",
        "relative_error",
        "directional_cosine",
        "algorithm_margin",
        "loss_before",
        "loss_after",
    ):
        assert getattr(batched, field) == pytest.approx(
            getattr(dense, field),
            rel=1e-12,
            abs=1e-12,
        )
    assert batched.accepted is dense.accepted
    assert batched.rejection_reason == dense.rejection_reason


def test_algorithm_one_strict_boundary_is_rejected() -> None:
    epsilon = 0.25
    base = torch.ones(1, 1, dtype=torch.float64)
    target = torch.zeros_like(base)
    eta = 0.1
    # q = |g-grad|/|g| = epsilon/(1+epsilon) exactly.
    approximation = torch.full_like(base, 1.0 / 1.2)
    candidate = base - eta * approximation
    certificate = certify_empirical_secant(
        base_output=base,
        candidate_output=candidate,
        target=target,
        learning_rate=eta,
        epsilon=epsilon,
        certificate_margin=1e-12,
    )
    assert certificate.algorithm_margin == pytest.approx(0.0, abs=1e-14)
    assert not certificate.relative_error_valid
    assert not certificate.accepted
    assert certificate.rejection_reason == "algorithm1_relative_error"


def test_opposite_or_zero_output_direction_never_certifies() -> None:
    base = torch.ones(2, 1)
    target = torch.zeros_like(base)
    zero = certify_empirical_secant(
        base_output=base,
        candidate_output=base.clone(),
        target=target,
        learning_rate=0.1,
        epsilon=0.25,
    )
    opposite = certify_empirical_secant(
        base_output=base,
        candidate_output=base + 0.1,
        target=target,
        learning_rate=0.1,
        epsilon=0.25,
    )
    assert zero.rejection_reason == "zero_approximation"
    assert not opposite.accepted
    assert opposite.directional_cosine == pytest.approx(-1.0)


def test_search_uses_real_secant_and_never_mutates_base_model() -> None:
    torch.manual_seed(4)
    model = _small_model(seed=4)
    x = torch.randn(10, 2)
    y = torch.randn(10, 1)
    state_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    result = search_adaptive_fgd_step(
        model=model,
        train_x=x,
        train_y=y,
        config=AdaptiveFGDConfig(
            epsilon=1e-8,
            family_order=("head_closed_form",),
            learning_rate_trials=2,
        ),
        step=1,
    )
    assert result.model is None
    assert result.attempts
    assert all(not attempt.certificate.accepted for attempt in result.attempts)
    for name, value in model.state_dict().items():
        assert torch.equal(value, state_before[name])


def test_tangent_family_can_generate_a_certified_real_secant() -> None:
    torch.manual_seed(3)
    model = _small_model(seed=3)
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)
    result = search_adaptive_fgd_step(
        model=model,
        train_x=x,
        train_y=y,
        config=AdaptiveFGDConfig(
            family_order=("tangent_least_squares",),
            tangent_damping=(0.0,),
            exact_jacobian_max_elements=100_000,
            learning_rate_trials=3,
        ),
        step=1,
    )
    assert result.model is not None
    assert result.certificate is not None
    assert result.certificate.accepted
    assert any(attempt.family == "tangent_least_squares" for attempt in result.attempts)


def test_chunked_matrix_free_cg_is_only_a_candidate_generator() -> None:
    torch.manual_seed(2)
    model = _small_model(seed=2)
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)
    seen_batch_sizes: list[int] = []

    def assert_bounded_batch(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        seen_batch_sizes.append(inputs[0].shape[0])
        assert inputs[0].shape[0] <= 3

    model.register_forward_pre_hook(assert_bounded_batch)
    result = search_adaptive_fgd_step(
        model=model,
        train_x=x,
        train_y=y,
        config=AdaptiveFGDConfig(
            family_order=("tangent_least_squares",),
            tangent_damping=(0.0,),
            tangent_cg_iterations=(64,),
            exact_jacobian_max_elements=0,
            computation_batch_size=3,
            learning_rate_trials=3,
        ),
        step=1,
    )
    assert result.model is not None
    accepted = next(
        attempt for attempt in result.attempts if attempt.certificate.accepted
    )
    assert accepted.solver == "cg"
    # Acceptance is still based on the finite secant, not the CG residual.
    assert accepted.certificate.sufficient_descent_valid
    assert seen_batch_sizes and max(seen_batch_sizes) <= 3


def test_screening_candidate_never_bypasses_full_train_certificate() -> None:
    torch.manual_seed(0)
    model = _small_model(seed=0)
    x = torch.randn(12, 2)
    y = torch.randn(12, 1)
    result = search_adaptive_fgd_step(
        model=model,
        train_x=x,
        train_y=y,
        config=AdaptiveFGDConfig(
            epsilon=0.9,
            screening_points=4,
            family_order=("tangent_least_squares",),
            tangent_damping=(0.0,),
            exact_jacobian_max_elements=100_000,
            learning_rate_trials=3,
        ),
        step=1,
    )
    assert any(
        attempt.certificate_scope == "screening" and attempt.certificate.accepted
        for attempt in result.attempts
    )
    promoted = [
        attempt
        for attempt in result.attempts
        if attempt.certificate_scope == "full_train"
    ]
    assert promoted and all(not attempt.certificate.accepted for attempt in promoted)
    assert result.model is None


def test_growth_adds_capacity_without_changing_the_function() -> None:
    torch.manual_seed(8)
    model = GrowingMLP(
        in_features=2,
        out_features=1,
        hidden_size=2,
        number_hidden_layers=2,
        device=torch.device("cpu"),
    )
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)
    with torch.no_grad():
        before = model(x).clone()
    grown, drift, added = grow_layer_function_preserving(
        model=model,
        train_x=x,
        train_y=y,
        layer_index=0,
        config=AdaptiveFGDConfig(computation_batch_size=8),
    )
    with torch.no_grad():
        after = grown(x)
    assert added > 0
    assert drift <= 1e-7
    assert torch.equal(before, after)
    assert sum(parameter.numel() for parameter in grown.parameters()) > sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_reference_config_exposes_new_method_and_tuple_fields() -> None:
    config = load_pipeline_config("configs/fgd/adaptive_grow_mnist.yaml")
    assert config.training.method == "fgd_adaptive_grow"
    assert config.fgd_adaptive.family_order == (
        "head_closed_form",
        "tangent_least_squares",
    )
    assert config.fgd_adaptive.screening_points == 256
    assert config.fgd_adaptive.tangent_damping == (0.1, 0.001)
    assert config.fgd_adaptive.tangent_cg_iterations == (64,)
    assert config.fgd_adaptive.nonlinear_steps == (16, 64)


def test_pipeline_stops_without_an_uncertified_step_when_growth_is_off() -> None:
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            train_batches=1,
            validation_batches=1,
            test_batches=1,
            batch_size=8,
            in_features=2,
            out_features=1,
            active_features=2,
        ),
        model=replace(config.model, hidden_size=2, number_hidden_layers=1),
        training=replace(
            config.training,
            method="fgd_adaptive_grow",
            epochs=1,
            device="cpu",
        ),
        fgd_adaptive=replace(
            config.fgd_adaptive,
            epsilon=1e-8,
            family_order=("head_closed_form",),
            learning_rate_trials=1,
        ),
        growth_schedule=replace(config.growth_schedule, enabled=False),
        wandb=replace(config.wandb, enabled=False),
        run=replace(config.run, save_plot=False, show_plot=False),
    )
    expected = build_model(config, torch.device("cpu"))
    result = run_pipeline(config, progress=None)
    assert result.termination_reason == "representation_exhausted"
    assert result.history[-1].fgd_candidate_accepted is False
    assert result.certificate_attempts
    for name, value in result.model.state_dict().items():
        assert torch.equal(value, expected.state_dict()[name])


def test_pipeline_growth_preserves_iterate_and_retries_same_step() -> None:
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            train_batches=1,
            validation_batches=1,
            test_batches=1,
            batch_size=8,
            in_features=2,
            out_features=1,
            active_features=2,
        ),
        model=replace(config.model, hidden_size=2, number_hidden_layers=2),
        training=replace(
            config.training,
            method="fgd_adaptive_grow",
            epochs=1,
            device="cpu",
        ),
        fgd_adaptive=replace(
            config.fgd_adaptive,
            family_order=("head_closed_form",),
            learning_rate_trials=1,
            max_growth_events=1,
            computation_batch_size=8,
            preservation_tolerance=2e-7,
        ),
        growth_schedule=replace(config.growth_schedule, enabled=True),
        wandb=replace(config.wandb, enabled=False),
        run=replace(config.run, save_plot=False, show_plot=False),
    )
    result = run_pipeline(config, progress=None)
    growth_entry = next(entry for entry in result.history if entry.step_type == "GRO")
    initial = result.history[0]
    assert growth_entry.step == 1  # growth does not consume the FGD step
    assert growth_entry.fgd_global_bound == pytest.approx(initial.fgd_global_bound)
    assert result.growth_events[0].output_drift <= 2e-7
    assert any(
        attempt.growth_layer_index is not None
        for attempt in result.certificate_attempts
    )


def test_validation_and_test_data_do_not_gate_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(21)
    train_x = torch.randn(8, 2, generator=generator)
    train_y = torch.randn(8, 1, generator=generator)
    train = DataLoader(TensorDataset(train_x, train_y), batch_size=4)
    validation_x = torch.randn(4, 2, generator=generator)
    validation_a = DataLoader(
        TensorDataset(validation_x, torch.zeros(4, 1)),
        batch_size=4,
    )
    validation_b = DataLoader(
        TensorDataset(validation_x, torch.full((4, 1), 1e6)),
        batch_size=4,
    )

    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        data=replace(config.data, in_features=2, out_features=1),
        model=replace(config.model, hidden_size=3, number_hidden_layers=1),
        training=replace(
            config.training,
            method="fgd_adaptive_grow",
            epochs=1,
            device="cpu",
        ),
        fgd_adaptive=replace(
            config.fgd_adaptive,
            family_order=("tangent_least_squares",),
            tangent_damping=(0.0,),
            exact_jacobian_max_elements=100_000,
            learning_rate_trials=3,
        ),
        growth_schedule=replace(config.growth_schedule, enabled=False),
        wandb=replace(config.wandb, enabled=False),
        run=replace(config.run, save_plot=False, show_plot=False),
    )

    monkeypatch.setattr(
        "stable_tiny.pipeline.build_dataloaders",
        lambda *_: (train, validation_a, validation_a),
    )
    result_a = run_pipeline(config, progress=None)
    monkeypatch.setattr(
        "stable_tiny.pipeline.build_dataloaders",
        lambda *_: (train, validation_b, validation_b),
    )
    result_b = run_pipeline(config, progress=None)

    assert result_a.termination_reason == result_b.termination_reason
    assert [
        attempt.certificate.accepted for attempt in result_a.certificate_attempts
    ] == [attempt.certificate.accepted for attempt in result_b.certificate_attempts]
    for name, value in result_a.model.state_dict().items():
        assert torch.equal(value, result_b.model.state_dict()[name])
    payload = result_payload(result_a)
    assert payload["certificate_attempts"]
    assert payload["termination_reason"] == result_a.termination_reason


def test_small_one_hot_classification_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(8, 2, generator=generator)
    labels = torch.arange(8) % 2
    y = torch.nn.functional.one_hot(labels, num_classes=2).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    monkeypatch.setattr(
        "stable_tiny.pipeline.build_dataloaders",
        lambda *_: (loader, loader, loader),
    )

    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        data=replace(config.data, kind="mnist", in_features=2, out_features=2),
        model=replace(config.model, hidden_size=3, number_hidden_layers=1),
        training=replace(
            config.training,
            method="fgd_adaptive_grow",
            epochs=1,
            device="cpu",
        ),
        fgd_adaptive=replace(
            config.fgd_adaptive,
            family_order=("tangent_least_squares",),
            tangent_damping=(0.0,),
            exact_jacobian_max_elements=100_000,
            learning_rate_trials=3,
        ),
        growth_schedule=replace(config.growth_schedule, enabled=False),
        wandb=replace(config.wandb, enabled=False),
        run=replace(config.run, save_plot=False, show_plot=False),
    )
    result = run_pipeline(config, progress=None)
    assert result.certificate_attempts
    assert all(math.isfinite(entry.train_accuracy) for entry in result.history)
