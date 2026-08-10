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
        # The probe floor. eps is an R^2-like ratio and least squares overfits
        # a finite probe, so with more parameters than rows it interpolates and
        # eps collapses. certify_probe_kappa sizes NK = kappa * rank(J) so the
        # probe stays above that floor as the net grows, and
        # max_total_parameters is coupled to it: at kappa 4 a budget of P needs
        # 4P rows, and this config only loads 10000 images.
        "certify_probe_kappa",
        "max_total_parameters",
        # The budget-free growth trigger and its two brakes. MEASURED without
        # them on this exact config: eps parks at 0.4975 against a bar of 0.5,
        # so `while epsilon >= target` never runs, and 150 epochs produce 2
        # growths and accuracy 0.109 with every certificate reporting health.
        # The lookahead could already say "training does not beat growing here"
        # and had no way to act on it.
        #
        # These three cannot appear in mnist_streaming.yaml even if wanted:
        # certify_growth_lookahead_entry is restricted to
        # family_order: [matrix_free_tangent], and that file is [tangent].
        "certify_growth_lookahead",
        "certify_growth_lookahead_entry",
        "growth_bottleneck_crossfold_folds",
        # 2, not the shared 10. Each lookahead call trains TWO clones for
        # `steps` passes over every batch of the loader -- 10 * 157 * 2 = 3140
        # forward/backward passes on MNIST, ~8 times per epoch. MEASURED: the
        # epoch went 11 -> 42 minutes at 10 and back to ~15 at 2, while the
        # predicate kept refusing at the same rate (10 yes / 5 no against
        # 19 / 11), so the horizon was cut without disabling the question.
        "growth_lookahead_steps",
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
    # 11 * 64 * 10 = 7040 rows against 1612 parameters before any growth
    # (784 in, 10 out), so NK/P = 4.37. At probe_batches 2 it was 1280 rows,
    # NK/P = 0.79, and MEASURED: eps read 0.0179 "certified" while every
    # transaction improved the probe and worsened the real loss, 20 of 30
    # epochs committed nothing and lr was exactly 0.
    assert matrix_free.fgd_approx.probe_batches == 11
    assert matrix_free.fgd_approx.certify_probe_kappa == 4.0
    assert matrix_free.fgd_approx.max_total_parameters == 5000
    assert matrix_free.fgd_approx.transactional_realized_descent is True
    # Removed: it forced a growth to break the "eps certifies but eta is below
    # theory_lr_min" deadlock, which is a symptom of the biased eps above.
    assert matrix_free.fgd_approx.certify_force_growth_on_finite_step_failure is False
    assert matrix_free.parametric_gd.matrixfree_batch_size == 128
