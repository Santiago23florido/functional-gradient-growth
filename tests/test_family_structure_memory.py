"""A family rejected at a structure is skipped until growth changes it."""

from __future__ import annotations

from dataclasses import replace

from fgdlib.tangent import ParametricGDConfig
from stable_tiny.pipeline import load_pipeline_config, run_pipeline


def test_rejected_family_is_skipped_until_growth(tmp_path) -> None:
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(config.model, hidden_size=2, number_hidden_layers=2),
        training=replace(config.training, epochs=3, device="cpu", log_every=1),
        fgd_approx=replace(
            config.fgd_approx,
            # The huge damping keeps the tangent certificate unusable, so the
            # family ladder is consulted every epoch.
            projection_damping=1e6,
            projection_group_auto=False,
            projection_group_size=1,
            theory_lr_search_steps=1,
            theory_lr_search_refinements=0,
            family_order=("tangent", "parametric_gd"),
        ),
        parametric_gd=ParametricGDConfig(
            inner_steps=(1,),
            functional_learning_rates=(0.2,),
            # Unreachable screen: the family declines at every attempt.
            min_cosine=1.0,
        ),
        secant_fgd=replace(
            config.secant_fgd,
            enabled=False,
            # The growth probe never improves, so the structure never grows
            # and the rejection memory must persist across epochs.
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

    lines: list[str] = []
    run_pipeline(config=config, progress=lines.append)

    declines = [line for line in lines if "no parametric_gd candidate" in line]
    skips = [line for line in lines if "skipping parametric_gd" in line]
    growths = [line for line in lines if "Committing layer" in line]
    assert not growths
    # Evaluated exactly once at this structure; skipped afterwards.
    assert len(declines) == 1
    assert len(skips) == 2
