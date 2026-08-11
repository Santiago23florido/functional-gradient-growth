"""The small canonical FGD catalog and its matrix-free MNIST derivative."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from stable_tiny.pipeline import load_pipeline_config

CANONICAL_CONFIGS = {
    "cifar_streaming.yaml",
    "family_ladder_N1024.yaml",
    "family_ladder_matrix_free_N1024.yaml",
    "mnist_conv_matrix_free.yaml",
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


def test_mnist_conv_inherits_config_2s_optimisation_set() -> None:
    """The conv config is config 2 with a conv model. Nothing else.

    Config 1 (the N=1024 ladder) supplied the certified knobs originally, but
    the EXECUTION settings and the DATA have to come from config 2, because
    config 2 is the MNIST one and this is its conv counterpart. Deriving from
    config 1 and keeping its 16-batch probe, its 10-step lookahead, its 70
    epochs and 1024 samples was the mistake this test exists to stop
    recurring: it left the conv run with nothing comparable to measure
    against.

    Six fgd_approx fields may differ. Three are forced by convolution, two
    realizability diagnostics respond to the measured finite-step deadlock,
    and one permits measured float32/cuDNN roundoff. The linear config remains
    unchanged.
    """
    config_2 = load_pipeline_config("configs/fgd/mnist_matrix_free.yaml")
    conv = load_pipeline_config("configs/fgd/mnist_conv_matrix_free.yaml")

    assert _differing_fields(config_2.fgd_approx, conv.fgd_approx) == {
        # The validation certificate takes the EXACT route, whose jacrev vmaps
        # one backward per output row, each holding the whole probe's
        # activations. On an MLP that is (N, width); on a conv it is
        # (N, C, H, W). MEASURED: 256 OOMs the 8 GB card.
        "jacobian_row_chunk",
        # Same defect in the matrix-free route's range finder.
        "matrix_free_block_chunk",
        # GroMo's crossfold test cannot MEASURE on conv: its fitted extension
        # matrix assumes two linear layers, so with 4-D alpha/omega it returns
        # inf, and inf > 1 approves everything. Off beats looking live.
        "growth_bottleneck_crossfold_folds",
        # The live Conv run reached eps ~= 9e-4 while full-train transactions
        # rejected and linearization returned no useful eta. These two flags
        # expose the probe invariant and value the SAME bottleneck candidate
        # by certified finite progress; they do not change config 2.
        "certify_probe_diagnostics",
        "certify_realizable_progress_growth",
        # A zero-output-weight Conv extension is algebraically exact, but
        # changing channel shape changes float32/cuDNN reduction order. Keep
        # its measured roundoff allowance local to Conv.
        "growth_preservation_tolerance",
    }
    # The data differs only in the two ways a convolution requires.
    assert _differing_fields(config_2.data, conv.data) == {"kind", "input_shape"}
    assert conv.data.mnist_train_samples == config_2.data.mnist_train_samples
    assert conv.training.epochs == config_2.training.epochs
    # And the settings whose absence made the first attempt unaffordable.
    assert conv.fgd_approx.probe_batches == config_2.fgd_approx.probe_batches
    assert conv.fgd_approx.growth_lookahead_steps == 2
    assert conv.fgd_approx.certify_growth_lookahead is True
    assert conv.fgd_approx.certify_growth_lookahead_entry is True
    assert conv.fgd_approx.max_total_parameters == 5000
    assert conv.fgd_approx.certify_probe_kappa == 4.0
    assert config_2.fgd_approx.certify_probe_diagnostics is False
    assert config_2.fgd_approx.certify_realizable_progress_growth is False
    assert conv.fgd_approx.certify_probe_diagnostics is True
    assert conv.fgd_approx.certify_realizable_progress_growth is True
    assert config_2.fgd_approx.growth_preservation_tolerance == 1e-6
    assert conv.fgd_approx.growth_preservation_tolerance == 1e-5
    # The certified core is untouched.
    assert conv.fgd_approx.family_order == ("matrix_free_tangent",)
    assert conv.fgd_approx.growth_where == "expressivity_bottleneck"
    assert conv.fgd_approx.rel_error_threshold == 0.5
    assert conv.fgd_approx.tiny_use_fisher is False


def test_mnist_conv_declares_every_cluster_override() -> None:
    """The Slurm launcher must be able to patch its node-local YAML copy."""
    text = Path("configs/fgd/mnist_conv_matrix_free.yaml").read_text()
    for key in (
        "data_dir",
        "jacobian_row_chunk",
        "matrix_free_block_chunk",
        "growth_where_balance",
    ):
        assert re.search(rf"^  {key}:.*$", text, flags=re.MULTILINE), key


def test_the_conv_stack_builds_the_architecture_the_comments_claim() -> None:
    import torch

    from stable_tiny.pipeline import build_dataloaders, build_model

    config = load_pipeline_config("configs/fgd/mnist_conv_matrix_free.yaml")
    model = build_model(config, torch.device("cpu"))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 498
    assert len(model._growable_layers) == 3
    # Every width the SEARCH can reach starts at 2, exactly as the linear
    # reference does. The two the search cannot reach -- the last conv before
    # each shape change -- are configured, and the head therefore sees 8*9=72
    # rather than the 18 that strangled the first attempt.
    assert [int(layer.in_neurons) for layer in model._growable_layers] == [2, 2, 2]
    assert model.layers[4].in_features == 72
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
        "configs/fgd/mnist_conv_matrix_free.yaml"
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
