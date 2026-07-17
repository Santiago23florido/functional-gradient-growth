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


def _outer_problem():
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
