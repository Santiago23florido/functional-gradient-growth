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
from stable_tiny.pipeline import PipelineConfig


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

    assert len(trials) == 384
    assert {trial.overrides["optimizer.name"] for trial in trials} == {"adamw"}
    assert {trial.overrides["optimizer.learning_rate"] for trial in trials} == {
        0.003,
        0.01,
        0.02,
        0.03,
    }
    assert grid["results_dir"].endswith("_stage2")

    config = build_trial_config(grid, trials[0])
    assert config.training.epochs == 400
    assert config.lr_scheduler.t_max == 400
    assert config.data.train_batches * config.data.batch_size == 1024
    assert config.data.test_batches * config.data.batch_size == 8192
