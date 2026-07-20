"""Per-epoch certified work dose and structure-burst growth patience."""

from __future__ import annotations

from dataclasses import replace

from fgdlib.tangent import ParametricGDConfig
from stable_tiny.pipeline import load_pipeline_config, run_pipeline


def test_multiple_certified_outer_steps_per_epoch(tmp_path) -> None:
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(config.model, hidden_size=2, number_hidden_layers=2),
        training=replace(config.training, epochs=1, device="cpu", log_every=1),
        fgd_approx=replace(
            config.fgd_approx,
            theory_lr_search_steps=2,
            theory_lr_search_refinements=0,
            outer_steps_per_epoch=3,
        ),
        wandb=replace(config.wandb, enabled=False),
        run=replace(
            config.run,
            results_dir=tmp_path,
            save_plot=False,
            show_plot=False,
        ),
    )

    lines: list[str] = []
    result = run_pipeline(config=config, progress=lines.append)

    step_lines = [line for line in lines if line.startswith("[FGD-STEP] Epoch 1:")]
    accepted_lines = [line for line in step_lines if "accepted=True" in line]
    # On smooth_sin the first tangent steps certify readily: the epoch must
    # attempt (and here commit) MORE than one genuine outer step.
    assert len(accepted_lines) >= 2
    # Still exactly one FGD history entry for the epoch.
    fgd_entries = [e for e in result.history if e.step_type == "FGD"]
    assert len(fgd_entries) == 1
    assert fgd_entries[0].fgd_candidate_accepted is True


def test_growth_patience_defers_the_probe(tmp_path) -> None:
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(config.model, hidden_size=2, number_hidden_layers=2),
        training=replace(config.training, epochs=3, device="cpu", log_every=1),
        fgd_approx=replace(
            config.fgd_approx,
            # Tangent unusable and the only fallback family always declines:
            # every epoch is fully exhausted.
            projection_damping=1e6,
            theory_lr_search_steps=1,
            theory_lr_search_refinements=0,
            family_order=("tangent", "parametric_gd"),
            family_rejection_cooldown=0,
            growth_patience=3,
        ),
        parametric_gd=ParametricGDConfig(
            inner_steps=(1,),
            functional_learning_rates=(0.2,),
            min_cosine=1.0,
        ),
        secant_fgd=replace(config.secant_fgd, enabled=False),
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

    deferrals = [line for line in lines if "growth deferred" in line]
    # Epochs 1 and 2 defer (1/3, 2/3); epoch 3 reaches the patience and may
    # probe growth.
    assert len(deferrals) == 2
    assert "(1/3" in deferrals[0]
    assert "(2/3" in deferrals[1]
    # With cooldown 0 the declining family is retried every epoch.
    declines = [line for line in lines if "no parametric_gd candidate" in line]
    assert len(declines) == 3


def test_growth_requires_lemma35_admissibility_failure(tmp_path) -> None:
    """Growth must not fire while the reachable set still represents r."""
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(config.model, hidden_size=2, number_hidden_layers=2),
        training=replace(config.training, epochs=3, device="cpu", log_every=1),
        fgd_approx=replace(
            config.fgd_approx,
            # An unreachable LR floor makes every transaction fail, but the
            # relative error stays low: Lemma 3.5 does NOT fail, so the
            # paper's structural criterion forbids growing.
            theory_lr_min=1e9,
            theory_lr_search_steps=1,
            theory_lr_search_refinements=0,
            family_order=("tangent",),
            growth_requires_admissibility_failure=True,
            rel_error_threshold=0.5,
        ),
        secant_fgd=replace(config.secant_fgd, enabled=False),
        scaling_line_search=replace(config.scaling_line_search, iterations=0),
        wandb=replace(config.wandb, enabled=False),
        run=replace(
            config.run,
            results_dir=tmp_path,
            save_plot=False,
            show_plot=False,
        ),
    )

    result = run_pipeline(config=config, progress=None)

    # Transactions fail every epoch, yet no capacity is added.
    assert not result.growth_events
    assert all(entry.step_type != "GRO" for entry in result.history)


def test_committed_family_does_not_veto_growth_when_lemma35_fails(tmp_path) -> None:
    """The family step is KEPT and growth still happens when eps >= 1/2.

    Under a functional whose infimum is not attained, some family step
    always certifies, so without this the structural step is postponed for
    ever while the network stays inadequate.
    """
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(config.model, hidden_size=2, number_hidden_layers=2),
        training=replace(config.training, epochs=2, device="cpu", log_every=1),
        fgd_approx=replace(
            config.fgd_approx,
            # Tangent cannot certify, so the ladder is consulted; the huge
            # damping also keeps the state relative error above 1/2.
            projection_damping=1e6,
            theory_lr_search_steps=1,
            theory_lr_search_refinements=0,
            family_order=("tangent", "parametric_descent"),
            family_rejection_cooldown=0,
            growth_patience=1,
            admissibility_failure_forces_growth=True,
            rel_error_threshold=0.5,
            # Judge the growth probe by certified descent, as the real
            # configuration does; the rel-error improvement gate is blind
            # to delta growth.
            growth_select_by_descent=True,
            growth_compute_delta=True,
            growth_function_preserving=False,
        ),
        parametric_descent=replace(
            config.parametric_descent,
            inner_steps=(1,),
            functional_learning_rates=(0.5,),
            min_progress=1e-12,     # accept almost anything: it must commit
        ),
        secant_fgd=replace(config.secant_fgd, enabled=False),
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
    result = run_pipeline(config=config, progress=lines.append)

    committed = [line for line in lines if "does not postpone" in line]
    # The family committed (its step was kept) ...
    assert committed, "expected a committed family step under Lemma-3.5 failure"
    # ... and the structural step happened anyway in the same run.
    assert result.growth_events
