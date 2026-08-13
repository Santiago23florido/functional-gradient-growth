from pathlib import Path

import torch

from fgdlib.models.stack import build_stack_model
from fgdlib.training_utils.loop import count_parameters
from grid_search.run import (
    apply_dotted_overrides,
    build_trial_config,
    enumerate_trials,
    load_grid,
    parameter_count,
)
from stable_tiny.pipeline import PipelineConfig, build_model


def test_heterogeneous_stack_builds_exact_headline_architecture() -> None:
    architecture = (11, 19, 16)
    model = build_stack_model(
        stack=[{"mlp": width} for width in architecture],
        in_features=4,
        out_features=1,
        device=torch.device("cpu"),
    )

    dimensions = [(layer.in_features, layer.out_features) for layer in model.layers]
    assert dimensions == [(4, 11), (11, 19), (19, 16), (16, 1)]
    assert count_parameters(model) == parameter_count(architecture) == 620
    assert model(torch.randn(5, 4)).shape == (5, 1)


def test_trial_enumeration_is_deterministic_and_conditional() -> None:
    grid = {
        "architectures": [[2, 3], [4, 5]],
        "model_seeds": [0, 1],
        "search_spaces": [
            {"optimizer.name": ["adamw"], "optimizer.learning_rate": [0.1, 0.2]},
            {"optimizer.name": ["sgd"], "optimizer.momentum": [0.0, 0.9]},
        ],
    }
    first = enumerate_trials(grid)
    second = enumerate_trials(grid)

    assert len(first) == 16
    assert first == second
    assert len({trial.trial_id for trial in first}) == len(first)
    assert all(
        not (
            trial.overrides["optimizer.name"] == "adamw"
            and "optimizer.momentum" in trial.overrides
        )
        for trial in first
    )


def test_dotted_overrides_replace_only_requested_fields() -> None:
    config = PipelineConfig()
    updated = apply_dotted_overrides(
        config,
        {"training.epochs": 70, "optimizer.learning_rate": 0.003},
    )

    assert updated.training.epochs == 70
    assert updated.optimizer.learning_rate == 0.003
    assert updated.data == config.data
    assert updated.training.method == config.training.method


def test_default_grid_preserves_the_headline_data_protocol() -> None:
    grid = load_grid(Path("grid_search/fixed_architectures.yaml"))
    trials = enumerate_trials(grid)
    config = build_trial_config(grid, trials[0])

    assert len(trials) == 672
    assert config.data.train_batches * config.data.batch_size == 1024
    assert config.data.validation_batches * config.data.batch_size == 1024
    assert config.data.test_batches * config.data.batch_size == 8192
    assert config.data.train_seed == 0
    assert config.training.epochs == 70
    assert config.training.method == "normal"
    assert config.growth_schedule.enabled is False
    assert config.wandb.enabled is False


def test_stage2_follows_epoch_and_learning_rate_boundaries() -> None:
    grid = load_grid(Path("grid_search/fixed_architectures_stage2.yaml"))
    trials = enumerate_trials(grid)

    assert len(trials) == 960
    assert {trial.overrides["optimizer.name"] for trial in trials} == {"adamw"}
    assert {trial.overrides["optimizer.learning_rate"] for trial in trials} == {
        0.001,
        0.002,
        0.003,
        0.005,
        0.0075,
        0.01,
        0.015,
        0.02,
        0.03,
        0.04,
    }
    assert grid["results_dir"].endswith("_stage2_lr_sweep")
    assert grid["paired_same_seed_evaluation"] == {
        "protocol": "same_seed_retraining",
        "growth_runs": {
            0: {"architecture": [11, 19, 16], "test_accuracy": 0.949},
            1: {"architecture": [12, 17, 18], "test_accuracy": 0.940},
            2: {"architecture": [13, 19, 14], "test_accuracy": 0.927},
            3: {"architecture": [9, 16, 22], "test_accuracy": 0.958},
        },
    }

    config = build_trial_config(grid, trials[0])
    assert config.training.epochs == 400
    assert config.lr_scheduler.t_max == 400
    assert config.data.train_batches * config.data.batch_size == 1024
    assert config.data.test_batches * config.data.batch_size == 8192


def test_under600_snapshot_grid_reuses_the_extended_search_protocol() -> None:
    grid = load_grid(Path("grid_search/fixed_architectures_ladder_under600.yaml"))
    trials = enumerate_trials(grid)

    assert len(trials) == 960
    assert {trial.architecture for trial in trials} == {
        (16, 19, 9),
        (9, 18, 18),
        (13, 16, 15),
        (11, 17, 17),
    }
    assert {trial.overrides["optimizer.learning_rate"] for trial in trials} == {
        0.0075,
        0.01,
        0.015,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.08,
        0.1,
    }
    assert {trial.overrides["optimizer.weight_decay"] for trial in trials} == {
        0.0,
        0.001,
        0.01,
    }
    assert {trial.overrides["lr_scheduler.name"] for trial in trials} == {
        "none",
        "cosineannealing",
    }

    paired = grid["paired_same_seed_evaluation"]
    assert paired["reference_kind"] == "last_observed_state_below_600_parameters"
    expected = {
        0: ((16, 19, 9), 593, 28, 0.9522705078125),
        1: ((9, 18, 18), 586, 47, 0.9573974609375),
        2: ((13, 16, 15), 560, 34, 0.96142578125),
        3: ((11, 17, 17), 583, 56, 0.958740234375),
    }
    for seed, (architecture, parameters, epoch, test_accuracy) in expected.items():
        reference = paired["growth_runs"][seed]
        assert tuple(reference["architecture"]) == architecture
        assert parameter_count(architecture) == parameters
        assert reference["expected_parameters"] == parameters
        assert reference["source_epoch"] == epoch
        assert reference["test_accuracy"] == test_accuracy

    config = build_trial_config(grid, trials[0])
    assert config.training.epochs == 400
    assert config.lr_scheduler.t_max == 400
    assert config.data.train_batches * config.data.batch_size == 1024
    assert config.data.validation_batches * config.data.batch_size == 1024
    assert config.data.test_batches * config.data.batch_size == 8192
    assert config.data.train_seed == 0
    assert config.training.method == "normal"
    assert config.growth_schedule.enabled is False


def test_retraining_builds_a_fresh_deterministic_model_without_growth() -> None:
    grid = load_grid(Path("grid_search/fixed_architectures_stage2.yaml"))
    trial = enumerate_trials(grid)[0]
    config = build_trial_config(grid, trial)

    first = build_model(config, torch.device("cpu"))
    initial_state = {
        name: value.detach().clone() for name, value in first.state_dict().items()
    }
    with torch.no_grad():
        next(first.parameters()).add_(1.0)

    second = build_model(config, torch.device("cpu"))

    assert first is not second
    assert config.model.model_seed == trial.model_seed
    assert config.training.method == "normal"
    assert config.growth_schedule.enabled is False
    assert tuple(item["mlp"] for item in config.model.stack) == trial.architecture
    for name, value in second.state_dict().items():
        assert value.data_ptr() != first.state_dict()[name].data_ptr()
        assert torch.equal(value, initial_state[name])
