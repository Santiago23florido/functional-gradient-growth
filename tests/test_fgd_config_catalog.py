"""The small canonical FGD catalog and its matrix-free MNIST derivative."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from stable_tiny.pipeline import load_pipeline_config

CANONICAL_CONFIGS = {
    "cifar_streaming.yaml",
    "family_ladder_N1024.yaml",
    "family_ladder_matrix_free_N1024.yaml",
    "mnist_conv_matrix_free_N1024.yaml",
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


def test_mnist_conv_changes_only_data_model_and_the_probe_gates() -> None:
    """The conv config must not become a second, unaccountable tuning surface.

    It is config 1 (the matrix-free N1024 ladder) with the DATA swapped for
    MNIST images, the MODEL swapped for a conv stack, and the data-size knobs
    taken from config 2. Every other certified knob is inherited verbatim, and
    this test is what keeps it that way: a field that starts differing has to
    be added here with a reason.
    """
    ladder = load_pipeline_config(
        "configs/fgd/family_ladder_matrix_free_N1024.yaml"
    )
    conv = load_pipeline_config("configs/fgd/mnist_conv_matrix_free_N1024.yaml")

    assert _differing_fields(ladder, conv) == {"run", "data", "model", "fgd_approx"}
    assert _differing_fields(ladder.fgd_approx, conv.fgd_approx) == {
        # The probe floor, and the budget coupled to it. At kappa 4 the probe
        # holds NK = 4 * rank(J) only while rank(J) <= 2560, since 16 batches
        # of 64 with 10 outputs is 10240 rows. Measured at the start,
        # rank(J) = 192 at P = 202, so NK/rank = 13.3.
        "certify_probe_kappa",
        "max_total_parameters",
    }
    # Exactly two, and both are data-size knobs. Three more are written out
    # EXPLICITLY in the conv yaml with their reasons -- certify_stream_gram,
    # growth_bottleneck_crossfold_folds and growth_where_balance -- but they
    # do not appear above because config 1 already holds those values. Stating
    # them is documentation of a deliberate choice, not divergence, and this
    # assertion is what keeps the two categories apart.
    for field in (
        "certify_stream_gram",
        "growth_bottleneck_crossfold_folds",
        "growth_where_balance",
    ):
        assert getattr(conv.fgd_approx, field) == getattr(ladder.fgd_approx, field)
    assert conv.fgd_approx.family_order == ("matrix_free_tangent",)
    assert conv.fgd_approx.growth_where == "expressivity_bottleneck"
    assert conv.fgd_approx.tiny_use_fisher is False


def test_the_conv_stack_builds_the_architecture_the_comments_claim() -> None:
    import torch

    from stable_tiny.pipeline import build_dataloaders, build_model

    config = load_pipeline_config("configs/fgd/mnist_conv_matrix_free_N1024.yaml")
    model = build_model(config, torch.device("cpu"))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 202
    assert len(model._growable_layers) == 3
    growable = [
        index
        for index, layer in enumerate(model.layers)
        if any(layer is growing for growing in model._growable_layers)
    ]
    assert growable == [1, 3, 5]


def test_the_crossfold_test_really_cannot_measure_on_conv() -> None:
    """Records the fact the config comment asserts, instead of trusting it."""
    import torch

    from fgdlib.models.convstack import build_conv_stack_model
    from fgdlib.tangent import crossfold_bottleneck_significance

    torch.manual_seed(0)
    model = build_conv_stack_model(
        [
            {"conv": [2, 3]},
            {"conv": [2, 3]},
            {"avgpool": [3, 3]},
            "flatten",
            {"mlp": [2, 1]},
        ],
        in_features=784,
        out_features=10,
        device=torch.device("cpu"),
        input_shape=(1, 28, 28),
    )
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(64, 1, 28, 28, generator=generator)
    y = torch.zeros(64, 10)
    y[torch.arange(64), torch.randint(0, 10, (64,), generator=generator)] = 1.0
    config = load_pipeline_config(
        "configs/fgd/mnist_conv_matrix_free_N1024.yaml"
    ).fgd_approx
    for layer_index in range(len(model._growable_layers)):
        statistic, folds = crossfold_bottleneck_significance(
            model,
            layer_index,
            x,
            y,
            batch_size=32,
            device=torch.device("cpu"),
            config=config,
        )
        # inf is the reserved "NOT MEASURED" value, and inf > 1 would approve
        # every candidate -- a no-op wearing the costume of a criterion.
        assert statistic == float("inf"), (layer_index, statistic, folds)
