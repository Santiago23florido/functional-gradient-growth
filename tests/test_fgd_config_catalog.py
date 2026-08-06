"""The small canonical FGD catalog and its matrix-free MNIST derivative."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from stable_tiny.pipeline import load_pipeline_config

CANONICAL_CONFIGS = {
    "cifar_streaming.yaml",
    "family_ladder_N1024.yaml",
    "family_ladder_matrix_free_N1024.yaml",
    "mnist_matrix_free.yaml",
    "mnist_streaming.yaml",
}


def _differing_fields(left, right) -> set[str]:
    return {
        field.name
        for field in dataclasses.fields(left)
        if getattr(left, field.name) != getattr(right, field.name)
    }


def test_fgd_contains_only_canonical_configs() -> None:
    present = {path.name for path in Path("configs/fgd").glob("*.yaml")}
    assert present == CANONICAL_CONFIGS


def test_mnist_matrix_free_changes_only_execution_and_matrix_free_gates() -> None:
    exact = load_pipeline_config("configs/fgd/mnist_streaming.yaml")
    matrix_free = load_pipeline_config("configs/fgd/mnist_matrix_free.yaml")

    assert _differing_fields(exact, matrix_free) == {
        "run",
        "fgd_approx",
        "parametric_gd",
        "wandb",
    }
    assert _differing_fields(exact.fgd_approx, matrix_free.fgd_approx) == {
        "family_order",
        "theory_lr_search_steps",
        "tangent_measured_max_lr",
        "certify_stream_gram",
        "probe_batches",
        "transactional_realized_descent",
        "transactional_max_retries",
        "transactional_descent_atol",
        "certify_functional_lr_cap",
    }
    assert _differing_fields(exact.parametric_gd, matrix_free.parametric_gd) == {
        "matrixfree_batch_size"
    }

    assert matrix_free.data == exact.data
    assert matrix_free.model == exact.model
    assert matrix_free.training == exact.training
    assert matrix_free.optimizer == exact.optimizer
    assert matrix_free.lr_scheduler == exact.lr_scheduler
    assert matrix_free.growth_schedule == exact.growth_schedule
    assert matrix_free.fgd_approx.family_order == ("matrix_free_tangent",)
    assert matrix_free.fgd_approx.certify_stream_gram is False
    assert matrix_free.fgd_approx.probe_batches == 2
    assert matrix_free.fgd_approx.max_total_parameters == 30000
    assert matrix_free.fgd_approx.transactional_realized_descent is True
    assert matrix_free.parametric_gd.matrixfree_batch_size == 128
