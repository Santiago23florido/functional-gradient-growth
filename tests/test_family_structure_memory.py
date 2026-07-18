"""Family rejections are a cooldown in accepted outer steps, never permanent."""

from __future__ import annotations

from dataclasses import replace

from fgdlib.tangent import ParametricGDConfig
from stable_tiny.pipeline import (
    _family_rejection_active,
    load_pipeline_config,
    run_pipeline,
)


def test_family_rejection_cooldown_predicate() -> None:
    # Never rejected: always eligible.
    assert not _family_rejection_active(None, 10, 5)
    # Rejected at step 3: skipped while fewer than 5 steps committed since.
    assert _family_rejection_active(3, 3, 5)
    assert _family_rejection_active(3, 7, 5)
    # Reconsidered exactly once the cooldown has elapsed.
    assert not _family_rejection_active(3, 8, 5)
    # cooldown <= 0 disables the memory entirely.
    assert not _family_rejection_active(3, 3, 0)
    assert not _family_rejection_active(3, 3, -1)


def test_growth_reset_is_modeled_by_clearing_the_rejection_state() -> None:
    # Growth clears the dict, i.e. the predicate sees "never rejected".
    rejected_at: int | None = 4
    assert _family_rejection_active(rejected_at, 5, 5)
    rejected_at = None  # what family_rejection_step.clear() produces
    assert not _family_rejection_active(rejected_at, 5, 5)


def test_rejected_family_is_skipped_while_nothing_is_accepted(tmp_path) -> None:
    """With zero accepted outer steps the cooldown never elapses."""
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(config.model, hidden_size=2, number_hidden_layers=2),
        training=replace(config.training, epochs=3, device="cpu", log_every=1),
        fgd_approx=replace(
            config.fgd_approx,
            # The huge damping keeps the tangent certificate unusable, so the
            # family ladder is consulted every epoch and no outer step is
            # ever accepted.
            projection_damping=1e6,
            theory_lr_search_steps=1,
            theory_lr_search_refinements=0,
            family_order=("tangent", "parametric_gd"),
            family_rejection_cooldown=5,
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
            # and the rejection state is never cleared.
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
    # Evaluated exactly once; on cooldown afterwards (no accepted steps).
    assert len(declines) == 1
    assert len(skips) == 2


def test_zero_cooldown_retries_the_family_every_epoch(tmp_path) -> None:
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(config.model, hidden_size=2, number_hidden_layers=2),
        training=replace(config.training, epochs=3, device="cpu", log_every=1),
        fgd_approx=replace(
            config.fgd_approx,
            projection_damping=1e6,
            theory_lr_search_steps=1,
            theory_lr_search_refinements=0,
            family_order=("tangent", "parametric_gd"),
            family_rejection_cooldown=0,
        ),
        parametric_gd=ParametricGDConfig(
            inner_steps=(1,),
            functional_learning_rates=(0.2,),
            min_cosine=1.0,
        ),
        secant_fgd=replace(
            config.secant_fgd,
            enabled=False,
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
    assert len(declines) == 3
    assert not skips
