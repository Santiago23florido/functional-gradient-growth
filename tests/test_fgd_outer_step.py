"""One genuine certified FGD outer step: certify at f_t, then move once."""

from __future__ import annotations


import pytest
import torch

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.tangent import (
    FGDApproxConfig,
    _compute_tangent_projection_step,
    certificate_from_projection_stats,
    measure_direction_projection,
)
from stable_tiny.pipeline import (
    PipelineConfig,
    _FGDTheoryState,
    _evaluate_fgd_outer_trial,
)

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP  # noqa: E402


def _outer_problem(local_acceptance: bool = True):
    torch.manual_seed(0)
    model = GrowingMLP(
        in_features=2,
        out_features=1,
        hidden_size=32,
        number_hidden_layers=1,
        device=torch.device("cpu"),
    )
    x = torch.randn(12, 2)
    y = torch.randn(12, 1) * 0.5
    batches = [(x, y)]
    config = PipelineConfig(
        fgd_approx=FGDApproxConfig(
            projection_solver="exact_kernel_eigh",
            projection_damping=1e-4,
            probe_batches=1,
            # A tiny mu keeps Cglob lenient so the deterministic descent
            # assertion below is the binding check.
            theory_mu=1e-9,
            local_acceptance_conditions=local_acceptance,
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
    direction_step = _compute_tangent_projection_step(
        model=model,
        x=x,
        y=y,
        config=config.fgd_approx,
    )
    direction = direction_step.parameter_updates
    direction_stats = measure_direction_projection(
        model,
        direction,
        x,
        y,
        config.fgd_approx,
    )
    return model, batches, config, state, loss, direction, direction_stats


def _run_outer_trial(
    model,
    batches,
    config,
    state,
    loss,
    direction,
    direction_stats,
    learning_rate,
):
    return _evaluate_fgd_outer_trial(
        base_model=model,
        direction=direction,
        direction_stats=direction_stats,
        train_batches=batches,
        validation_loader=batches,
        loss_function=torch.nn.MSELoss(),
        device=torch.device("cpu"),
        learning_rate=learning_rate,
        accuracy_tolerance=0.1,
        config=config,
        classification=False,
        theory_state=state,
        initial_functional_gap=loss,
        theory_loss_star=0.0,
    )


def test_certificate_is_computed_before_the_outer_update() -> None:
    """The trial certifies pre-update stats and never touches the base model."""
    problem = _outer_problem()
    model, batches, config, state, loss, direction, direction_stats = problem
    base_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    trial = _run_outer_trial(*problem, learning_rate=0.05)

    # The base model is untouched: rejection is a rollback by construction.
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, base_state[name])
    # The trial's certificate is EXACTLY the pre-update direction
    # certificate: re-measuring the same direction at the (unchanged) base
    # model reproduces it, so it cannot describe the post-update state.
    reference = certificate_from_projection_stats(
        stats=measure_direction_projection(
            model,
            direction,
            batches[0][0],
            batches[0][1],
            config.fgd_approx,
        ),
        learning_rate=0.05,
        config=config.fgd_approx,
    )
    assert trial.certificate.relative_error == pytest.approx(
        reference.relative_error,
        rel=1e-9,
    )
    assert trial.certificate.max_valid_learning_rate == pytest.approx(
        reference.max_valid_learning_rate,
        rel=1e-9,
    )


def test_outer_trial_applies_exactly_one_shared_update() -> None:
    problem = _outer_problem()
    model, _, _, _, _, direction, _ = problem
    learning_rate = 0.05
    trial = _run_outer_trial(*problem, learning_rate=learning_rate)

    base_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    trial_parameters = [
        parameter
        for parameter in trial.model.parameters()
        if parameter.requires_grad
    ]
    assert len(base_parameters) == len(trial_parameters) == len(direction)
    for base, stepped, update in zip(
        base_parameters,
        trial_parameters,
        direction,
    ):
        assert torch.allclose(
            stepped,
            base - learning_rate * update,
            atol=1e-7,
        )


def test_accepted_outer_step_decreases_the_evaluated_loss() -> None:
    problem = _outer_problem()
    model, batches, config, state, loss, direction, direction_stats = problem
    certificate = certificate_from_projection_stats(
        stats=direction_stats,
        learning_rate=None,
        config=config.fgd_approx,
    )
    assert certificate.max_valid_learning_rate is not None
    # Small enough to stay in the linear regime of this ill-conditioned toy
    # direction; the pipeline's backtracking search finds such rates itself.
    learning_rate = 0.03 * certificate.max_valid_learning_rate
    trial = _run_outer_trial(*problem, learning_rate=learning_rate)

    assert trial.validation_functional_loss < loss
    assert trial.loss_descent_valid
    assert trial.all_conditions_valid


def test_local_conditions_accept_even_when_trajectory_bounds_fail() -> None:
    """Cstat/Cglob are trajectory diagnostics, never acceptance gates."""
    problem = _outer_problem()
    model, batches, config, state, loss, direction, direction_stats = problem
    # A poisoned accumulated history: the near-zero contraction product
    # makes the global bound unreachable and the long tiny-progress history
    # makes the stationary bound fail — neither may block a step whose four
    # LOCAL conditions hold.
    poisoned_state = _FGDTheoryState(
        epoch_count=1000,
        min_gradient_sq_norm=direction_stats.target_sq_norm,
        min_positive_learning_rate=1.0,
        min_descent_coefficient=1.0,
        global_contraction_product=1e-12,
        previous_validation_functional_loss=loss,
    )
    certificate = certificate_from_projection_stats(
        stats=direction_stats,
        learning_rate=None,
        config=config.fgd_approx,
    )
    assert certificate.max_valid_learning_rate is not None
    learning_rate = 0.03 * certificate.max_valid_learning_rate
    trial = _run_outer_trial(
        model,
        batches,
        config,
        poisoned_state,
        loss,
        direction,
        direction_stats,
        learning_rate=learning_rate,
    )

    # The four local conditions hold...
    assert trial.certificate.sensor_valid is True
    assert trial.certificate.relative_error_condition_valid is True
    assert trial.certificate.learning_rate_interval_valid is True
    assert trial.loss_descent_valid is True
    # ...the trajectory bounds are computed, logged, and FAILING...
    assert trial.stationary_bound is not None
    assert trial.stationary_bound_valid is False
    assert trial.global_bound is not None
    assert trial.global_bound_valid is False
    assert trial.global_contraction is not None
    # ...and the step is accepted anyway.
    assert trial.all_conditions_valid is True


def test_flag_off_restores_legacy_trajectory_gates() -> None:
    """Without the config flag the accumulated bounds gate acceptance."""
    # The flag defaults to OFF: existing configs keep the old behavior.
    assert FGDApproxConfig().local_acceptance_conditions is False

    problem = _outer_problem(local_acceptance=False)
    model, batches, config, state, loss, direction, direction_stats = problem
    poisoned_state = _FGDTheoryState(
        epoch_count=1000,
        min_gradient_sq_norm=direction_stats.target_sq_norm,
        min_positive_learning_rate=1.0,
        min_descent_coefficient=1.0,
        global_contraction_product=1e-12,
        previous_validation_functional_loss=loss,
    )
    certificate = certificate_from_projection_stats(
        stats=direction_stats,
        learning_rate=None,
        config=config.fgd_approx,
    )
    assert certificate.max_valid_learning_rate is not None
    trial = _run_outer_trial(
        model,
        batches,
        config,
        poisoned_state,
        loss,
        direction,
        direction_stats,
        learning_rate=0.03 * certificate.max_valid_learning_rate,
    )
    # Same step, same failing bounds — but in legacy mode they REJECT it.
    assert trial.loss_descent_valid is True
    assert trial.stationary_bound_valid is False
    assert trial.global_bound_valid is False
    assert trial.all_conditions_valid is False


def test_descent_strictness_follows_the_flag() -> None:
    """A no-progress step passes the legacy gate and fails the strict one."""
    for local_acceptance, expected_descent_valid in (
        (False, True),
        (True, False),
    ):
        problem = _outer_problem(local_acceptance=local_acceptance)
        model, batches, config, state, loss, direction, stats = problem
        zero_direction = tuple(
            torch.zeros_like(update) for update in direction
        )
        trial = _run_outer_trial(
            model,
            batches,
            config,
            state,
            loss,
            zero_direction,
            stats,
            learning_rate=0.02,
        )
        assert trial.validation_functional_loss == pytest.approx(loss)
        assert trial.loss_descent_valid is expected_descent_valid


def test_relative_error_condition_failure_rejects_the_step() -> None:
    problem = _outer_problem()
    model, batches, config, state, loss, direction, direction_stats = problem
    # Tighten the threshold below the direction's measured relative error:
    # only Crel fails (the LR interval still derives from the measured
    # relative error and the small step still strictly descends).
    from dataclasses import replace

    strict_config = replace(
        config,
        fgd_approx=replace(
            config.fgd_approx,
            rel_error_threshold=direction_stats.output_error.relative_error
            / 2.0,
        ),
    )
    trial = _run_outer_trial(
        model,
        batches,
        strict_config,
        state,
        loss,
        direction,
        direction_stats,
        learning_rate=0.02,
    )
    assert trial.certificate.relative_error_condition_valid is False
    assert trial.certificate.learning_rate_interval_valid is True
    assert trial.loss_descent_valid is True
    assert trial.all_conditions_valid is False


def test_learning_rate_outside_the_interval_rejects_the_step() -> None:
    problem = _outer_problem()
    model, batches, config, state, loss, direction, direction_stats = problem
    certificate = certificate_from_projection_stats(
        stats=direction_stats,
        learning_rate=None,
        config=config.fgd_approx,
    )
    assert certificate.max_valid_learning_rate is not None
    # Above the safe upper bound: eta < eta_bar fails.
    above = _run_outer_trial(
        model,
        batches,
        config,
        state,
        loss,
        direction,
        direction_stats,
        learning_rate=2.0 * certificate.max_valid_learning_rate,
    )
    assert above.certificate.learning_rate_interval_valid is False
    assert above.all_conditions_valid is False
    # Below theory_lr_min: eta > theory_lr_min fails while the step itself
    # still strictly descends.
    from dataclasses import replace

    floor_config = replace(
        config,
        fgd_approx=replace(config.fgd_approx, theory_lr_min=0.05),
    )
    below = _run_outer_trial(
        model,
        batches,
        floor_config,
        state,
        loss,
        direction,
        direction_stats,
        learning_rate=0.02,
    )
    assert below.certificate.learning_rate_interval_valid is False
    assert below.loss_descent_valid is True
    assert below.all_conditions_valid is False


def test_invalid_sensor_rejects_the_step() -> None:
    problem = _outer_problem()
    model, batches, config, state, loss, direction, direction_stats = problem
    from fgdlib.tangent import _FunctionalStepStats

    corrupted_stats = _FunctionalStepStats(
        output_error=direction_stats.output_error,
        dot_product=float("nan"),
        approximation_sq_norm=direction_stats.approximation_sq_norm,
        target_sq_norm=direction_stats.target_sq_norm,
    )
    trial = _run_outer_trial(
        model,
        batches,
        config,
        state,
        loss,
        direction,
        corrupted_stats,
        learning_rate=0.02,
    )
    assert trial.certificate.sensor_valid is False
    assert trial.all_conditions_valid is False


def test_outer_step_rejects_when_the_step_ascends() -> None:
    """Moving AGAINST the certified direction fails the descent gate."""
    problem = _outer_problem()
    model, batches, config, state, loss, direction, direction_stats = problem
    ascent_direction = tuple(-update for update in direction)
    trial = _evaluate_fgd_outer_trial(
        base_model=model,
        direction=ascent_direction,
        direction_stats=direction_stats,
        train_batches=batches,
        validation_loader=batches,
        loss_function=torch.nn.MSELoss(),
        device=torch.device("cpu"),
        learning_rate=0.1,
        accuracy_tolerance=0.1,
        config=config,
        classification=False,
        theory_state=state,
        initial_functional_gap=loss,
        theory_loss_star=0.0,
    )
    assert trial.validation_functional_loss > loss
    assert trial.loss_descent_valid is False
    assert trial.all_conditions_valid is False


def test_tangent_measured_descent_search_takes_a_big_certified_step() -> None:
    """Tangent direction + measured descent (Prop 3.8) beats the eps bound."""
    from dataclasses import replace
    from stable_tiny.pipeline import _search_tangent_measured_descent

    problem = _outer_problem(local_acceptance=True)
    model, batches, config, state, loss, direction, direction_stats = problem
    config = replace(
        config,
        fgd_approx=replace(
            config.fgd_approx,
            tangent_measured_descent=True,
            tangent_measured_max_lr=1.0,
            theory_lr_search_steps=8,
            theory_mu=1e-9,
        ),
    )
    result = _search_tangent_measured_descent(
        base_model=model,
        direction=direction,
        direction_stats=direction_stats,
        train_batches=batches,
        validation_loader=batches,
        loss_function=torch.nn.MSELoss(),
        device=torch.device("cpu"),
        accuracy_tolerance=0.1,
        config=config,
        theory_state=state,
        initial_functional_gap=loss,
        theory_loss_star=0.0,
    )
    trial = result.accepted
    assert trial is not None
    # A genuine measured descent was certified via Prop. 3.8 (not the
    # Lemma-3.5 epsilon interval, which is None-gated here).
    assert trial.validation_functional_loss < loss
    assert trial.loss_descent_valid is True
    assert trial.all_conditions_valid is True
    assert trial.certificate.relative_error_condition_valid is None
    assert trial.certificate.learning_rate_interval_valid is None
    # The committed step is the measured line-search optimum (positive).
    assert trial.epoch_result.learning_rate > 0.0
    # The measured descent coefficient is positive (Prop. 3.8 contraction).
    assert trial.certificate.theory_descent_coefficient is not None
    assert trial.certificate.theory_descent_coefficient > 0.0
