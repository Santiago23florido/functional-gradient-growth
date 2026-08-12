"""Configuration and probe invariants for the full-MNIST matrix-free A/B."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fgdlib.search.damping import DampingCandidate, DampingChoice
from fgdlib.tangent import (
    validate_probe_refinement,
    validate_realizable_progress_growth,
)
from stable_tiny import pipeline
from stable_tiny.pipeline import load_pipeline_config


def _differing_paths(left, right, prefix: str = "") -> set[str]:
    differences: set[str] = set()
    for field in dataclasses.fields(left):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        path = f"{prefix}.{field.name}" if prefix else field.name
        if dataclasses.is_dataclass(left_value):
            differences.update(_differing_paths(left_value, right_value, path))
        elif left_value != right_value:
            differences.add(path)
    return differences


def _render_launcher_config(source: str, resample: bool, max_rows: int) -> str:
    """Reproduce the launcher's sed exactly, so the test measures what runs."""
    rendered = re.sub(
        r"^  data_dir:.*$",
        "  data_dir: /tmp/mnist-full-ab-test",
        source,
        flags=re.MULTILINE,
    )
    rendered = re.sub(
        r"^  certify_probe_base_resample:.*$",
        f"  certify_probe_base_resample: {str(resample).lower()}",
        rendered,
        flags=re.MULTILINE,
    )
    return re.sub(
        r"^  certify_probe_refine_max_rows:.*$",
        f"  certify_probe_refine_max_rows: {max_rows}",
        rendered,
        flags=re.MULTILINE,
    )


def test_probe_arms_differ_only_in_the_probe(tmp_path: Path) -> None:
    """Pin the effective YAML produced by the two-arm launcher.

    The A/B measures the SONDA, because that is what the diagnosis names: with
    a base fixed for the run the certified steps interpolate it and it stops
    representing the population (MEASURED, run 2kbo8rf4: 14.2x easier per image
    by transaction ~50). Both arms carry the realizable-progress abstention,
    which is not optional -- without it the biased arm simply freezes for 35
    epochs and measures nothing.

    The previous A/B on this launcher was transactional_family_step and is
    MEASURED as vacuous under probe refinement: the ON and OFF arms produced
    byte-identical logs with zero [FAMILY] lines, because eps sits at 0.23-0.25
    and the family ladder is never invoked.
    """
    source = Path("configs/experiments/mnist_full.yaml").read_text(encoding="utf-8")
    arms = []
    for resample, max_rows in ((True, 6400), (False, 0)):
        path = tmp_path / f"probe-{str(resample).lower()}.yaml"
        path.write_text(
            _render_launcher_config(source, resample, max_rows), encoding="utf-8"
        )
        arms.append(load_pipeline_config(path))

    fixed, biased = arms
    assert _differing_paths(fixed, biased) == {
        "fgd_approx.certify_probe_base_resample",
        "fgd_approx.certify_probe_refine_max_rows",
    }
    assert fixed.fgd_approx.certify_probe_base_resample is True
    assert fixed.fgd_approx.certify_probe_refine_max_rows == 6400
    assert biased.fgd_approx.certify_probe_base_resample is False
    assert biased.fgd_approx.certify_probe_refine_max_rows == 0
    # Everything the comparison holds fixed.
    for arm in arms:
        assert arm.fgd_approx.transactional_realized_descent is True
        assert arm.fgd_approx.certify_realizable_progress_growth is True
        assert arm.fgd_approx.certify_probe_refine_on_transaction_mismatch is True
        assert arm.fgd_approx.probe_resample is False

    launcher = Path("cluster/slurm/mnist_full_probe_ab.sbatch").read_text(
        encoding="utf-8"
    )
    assert "0) RESAMPLE=true;  MAX_ROWS=6400; ARM=probe-fixed" in launcher
    assert "1) RESAMPLE=false; MAX_ROWS=0;    ARM=probe-biased" in launcher
    assert "--gpus=1" in launcher
    assert 'RUN_SUFFIX="${RUN_SUFFIX:-}"' in launcher
    # The launcher must verify its own patch, or an arm silently runs the other
    # arm's config -- which is how the previous A/B could have gone unnoticed.
    assert "certify_probe_base_resample no se parcheo" in launcher
    assert "certify_probe_refine_max_rows no se parcheo" in launcher


def test_only_deadlocked_mnist_matrix_free_runs_enable_new_options() -> None:
    full = load_pipeline_config("configs/experiments/mnist_full.yaml").fgd_approx
    conv = load_pipeline_config("configs/fgd/mnist_conv_matrix_free.yaml").fgd_approx
    assert full.certify_probe_diagnostics is True
    assert full.certify_realizable_progress_growth is True
    assert conv.certify_probe_diagnostics is True
    assert conv.certify_realizable_progress_growth is True
    assert full.certify_probe_refine_on_transaction_mismatch is True
    assert full.certify_probe_refine_batches_per_round == 1
    assert full.certify_probe_refine_max_rounds == 16
    assert conv.certify_probe_refine_on_transaction_mismatch is True
    assert conv.certify_probe_refine_batches_per_round == 1
    assert conv.certify_probe_refine_max_rounds == 16
    validate_probe_refinement(conv)

    # The unbiased base and the row budget are the full-MNIST answer to the
    # MEASURED 14.2x probe/population gap. They are confined to that one
    # experiment: nothing under configs/fgd may reach them, conv included.
    assert full.certify_probe_base_resample is True
    assert full.certify_probe_refine_max_rows == 6400
    assert conv.certify_probe_base_resample is False
    assert conv.certify_probe_refine_max_rows == 0

    for path in Path("configs/fgd").glob("*.yaml"):
        config = load_pipeline_config(path).fgd_approx
        expected = path.name == "mnist_conv_matrix_free.yaml"
        assert config.certify_probe_diagnostics is expected, path
        assert config.certify_realizable_progress_growth is expected, path
        assert config.certify_probe_refine_on_transaction_mismatch is expected, path
        assert config.certify_probe_refine_batches_per_round == 1, path
        assert config.certify_probe_refine_max_rounds == (16 if expected else 0), path
        assert config.certify_probe_base_resample is False, path
        assert config.certify_probe_refine_max_rows == 0, path


@pytest.mark.parametrize(
    "path, family_order, transactional",
    [
        ("configs/fgd/family_ladder_N1024.yaml", ("tangent",), False),
        (
            "configs/fgd/family_ladder_matrix_free_N1024.yaml",
            ("matrix_free_tangent",),
            True,
        ),
    ],
)
def test_n1024_growth_settings_remain_at_their_defaults(
    path: str, family_order: tuple[str, ...], transactional: bool
) -> None:
    config = load_pipeline_config(path).fgd_approx
    assert config.family_order == family_order
    assert config.growth_where == "expressivity_bottleneck"
    assert config.growth_selection == "unified_expansion"
    assert config.certify_probe_kappa == 0.0
    assert config.certify_probe_diagnostics is False
    assert config.certify_realizable_progress_growth is False
    assert config.certify_probe_refine_on_transaction_mismatch is False
    assert config.certify_probe_refine_batches_per_round == 1
    assert config.certify_probe_refine_max_rounds == 0
    assert config.certify_probe_base_resample is False
    assert config.certify_probe_refine_max_rows == 0
    assert config.transactional_realized_descent is transactional


def test_realizable_growth_validation_rejects_another_where_rule() -> None:
    config = load_pipeline_config("configs/experiments/mnist_full.yaml").fgd_approx
    with pytest.raises(ValueError, match="expressivity_bottleneck"):
        validate_realizable_progress_growth(
            dataclasses.replace(config, growth_where="rank_ceiling")
        )
    with pytest.raises(ValueError, match="certify_realize_path"):
        validate_realizable_progress_growth(
            dataclasses.replace(config, certify_realize_path=False)
        )
    with pytest.raises(ValueError, match="must be authorised"):
        validate_realizable_progress_growth(
            dataclasses.replace(
                config, certify_force_growth_on_finite_step_failure=True
            )
        )


def test_probe_refinement_validation_is_matrix_free_and_bounded() -> None:
    config = load_pipeline_config("configs/experiments/mnist_full.yaml").fgd_approx
    validate_probe_refinement(config)

    # A row budget IS a budget, so dropping the round cap while it is set is
    # legal -- that is the point of it. Dropping BOTH is not.
    validate_probe_refinement(
        dataclasses.replace(config, certify_probe_refine_max_rounds=0)
    )
    with pytest.raises(ValueError, match="max_rounds"):
        validate_probe_refinement(
            dataclasses.replace(
                config,
                certify_probe_refine_max_rounds=0,
                certify_probe_refine_max_rows=0,
            )
        )
    with pytest.raises(ValueError, match="max_rows must be a non-negative"):
        validate_probe_refinement(
            dataclasses.replace(config, certify_probe_refine_max_rows=-1)
        )
    with pytest.raises(ValueError, match="positive integer"):
        validate_probe_refinement(
            dataclasses.replace(config, certify_probe_refine_batches_per_round=0)
        )
    with pytest.raises(ValueError, match="probe_resample"):
        validate_probe_refinement(dataclasses.replace(config, probe_resample=True))
    # The unbiased base is meaningless without the counterexample memory it is
    # designed to coexist with, and it must not leak onto the exact-tangent
    # path: the refinement it requires is already matrix-free only.
    with pytest.raises(ValueError, match="certify_probe_base_resample requires"):
        validate_probe_refinement(
            dataclasses.replace(
                config,
                certify_probe_refine_on_transaction_mismatch=False,
                certify_probe_refine_max_rows=0,
            )
        )


def test_probe_diagnostic_reports_rank_ratio_without_mutation() -> None:
    jacobian = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0]]
    )
    target = torch.zeros(3, 4)
    system = SimpleNamespace(jacobian=jacobian, target=target, factors=None)
    model = torch.nn.Linear(1, 2, bias=True)  # P = 4
    jacobian_before = jacobian.clone()
    target_before = target.clone()
    messages: list[str] = []

    pipeline._log_certification_probe(model, system, 0.0034, messages.append)

    assert len(messages) == 1
    assert "[CERTIFY-PROBE]" in messages[0]
    assert "P=4" in messages[0]
    assert "NK=12" in messages[0]
    assert "rank=2" in messages[0]
    assert "NK/rank=6.0000" in messages[0]
    assert "eps=0.0034" in messages[0]
    assert torch.equal(jacobian, jacobian_before)
    assert torch.equal(target, target_before)


def test_probe_diagnostic_flag_does_not_change_kappa_sizing(monkeypatch) -> None:
    config = load_pipeline_config("configs/experiments/mnist_full.yaml")
    rank = 1600
    monkeypatch.setattr(pipeline, "_estimate_certification_rank", lambda *args: rank)
    loader = [None] * 100

    batches = []
    for enabled in (False, True):
        pipeline._CERTIFICATION_RANK_CACHE.clear()
        candidate = dataclasses.replace(
            config,
            fgd_approx=dataclasses.replace(
                config.fgd_approx, certify_probe_diagnostics=enabled
            ),
        )
        batches.append(
            pipeline._bounded_probe_batches(
                candidate, torch.nn.Linear(1, 1), loader, torch.device("cpu")
            )
        )

    assert batches[0] == batches[1]
    rows = batches[0] * config.data.batch_size * config.data.out_features
    assert rows / rank >= config.fgd_approx.certify_probe_kappa


def _sized_batches(config, rank: int, parameters: int, monkeypatch) -> int:
    """Rows the sizer REQUESTS for a net of ``parameters`` at measured ``rank``."""
    monkeypatch.setattr(pipeline, "_estimate_certification_rank", lambda *args: rank)
    pipeline._CERTIFICATION_RANK_CACHE.clear()
    model = torch.nn.Linear(parameters - 1, 1)  # (P-1) weights + 1 bias = P
    assert sum(p.numel() for p in model.parameters()) == parameters
    return pipeline._bounded_probe_batches(
        config, model, [None] * 1000, torch.device("cpu")
    ) * config.data.batch_size * config.data.out_features


def test_the_parameter_floor_leaves_the_healthy_phase_identical(monkeypatch) -> None:
    """The measured trajectory of run unguzxkq, replayed through the sizer.

    The floor exists because sizing by rank alone has a FIXED POINT: rank(J) <=
    NK by construction and the rank is measured ON the probe, so a small probe
    reports a small rank, which reports that no more rows are needed. MEASURED
    closing while P triples -- rank 1249 -> 161, NK frozen at 5760, NK/P 0.45,
    eps collapsing 0.44 -> 0.009 with the validation relative error at 1.02.

    The user's constraint is that the phase which WAS working must not move, so
    that is what this pins: while the probe already clears the floor, the floor
    is inactive and the sizing is identical.
    """
    config = load_pipeline_config("configs/experiments/mnist_full.yaml")
    without = dataclasses.replace(
        config,
        fgd_approx=dataclasses.replace(
            config.fgd_approx, certify_probe_parameter_floor=0.0
        ),
    )
    assert config.fgd_approx.certify_probe_parameter_floor == 1.25

    # (P, rank) from the run, over the phase whose NK/P stayed >= 1.51.
    for parameters, rank in ((1612, 624), (2425, 646), (2503, 854), (3383, 1249)):
        assert _sized_batches(config, rank, parameters, monkeypatch) == _sized_batches(
            without, rank, parameters, monkeypatch
        ), (parameters, rank)


def test_the_parameter_floor_binds_exactly_where_the_run_degraded(monkeypatch) -> None:
    config = load_pipeline_config("configs/experiments/mnist_full.yaml")
    without = dataclasses.replace(
        config,
        fgd_approx=dataclasses.replace(
            config.fgd_approx, certify_probe_parameter_floor=0.0
        ),
    )
    floor = config.fgd_approx.certify_probe_parameter_floor

    # Past the healthy phase the rank-only sizer does not merely stop growing,
    # it COLLAPSES -- 640 rows requested at P=14487, against 18560 with the
    # floor. Only the monotone ratchet on the probe size held it at 5760, which
    # is how a certificate came to be read off 0.40 rows per parameter.
    for parameters, rank in ((5992, 862), (9383, 602), (12869, 161), (14487, 132)):
        with_floor = _sized_batches(config, rank, parameters, monkeypatch)
        rank_only = _sized_batches(without, rank, parameters, monkeypatch)
        assert with_floor > rank_only, (parameters, rank)
        # The invariant the floor exists to hold: rank(J) <= min(NK, P), so
        # NK > P forbids interpolation outright.
        assert with_floor >= floor * parameters, (parameters, with_floor)
    assert _sized_batches(without, 132, 14487, monkeypatch) < 14487


def test_the_parameter_floor_still_respects_the_dataset_cap(monkeypatch) -> None:
    """A floor cannot conjure rows the loader does not have."""
    config = load_pipeline_config("configs/experiments/mnist_full.yaml")
    monkeypatch.setattr(pipeline, "_estimate_certification_rank", lambda *args: 100)
    pipeline._CERTIFICATION_RANK_CACHE.clear()
    model = torch.nn.Linear(999_999, 1)
    batches = pipeline._bounded_probe_batches(
        config, model, [None] * 7, torch.device("cpu")
    )
    assert batches == 7


def test_probe_diagnostic_reports_the_true_rows_not_the_surrogate() -> None:
    """NK must be the probe's rows even when the system is a surrogate.

    Under certify_stream_gram the system carries a (rank+1) x P surrogate, so
    reading NK off system.target would silently report the surrogate's height
    in the very diagnostic that exists to catch NK falling below P.
    """
    system = SimpleNamespace(
        factors=(torch.eye(3), torch.eye(3)),
        target=torch.zeros(4),  # a surrogate height, not the probe's rows
    )
    messages: list[str] = []
    pipeline._log_certification_probe(
        torch.nn.Linear(4, 1), system, 0.5, messages.append, probe_rows=120
    )
    assert "NK=120" in messages[0]
    assert "NK=4" not in messages[0]
    # P = 4 weights + 1 bias = 5, so NK/P = 24 and the floor is comfortably met.
    assert "NK/P=24.0000" in messages[0]
    assert "BELOW INTERPOLATION FLOOR" not in messages[0]


def test_probe_diagnostic_shouts_when_the_probe_falls_below_the_floor() -> None:
    system = SimpleNamespace(factors=(torch.eye(2), torch.eye(2)), target=torch.zeros(2))
    messages: list[str] = []
    # P = 100*4 + 4 = 404 parameters against 120 probe rows: NK/P = 0.30, the
    # regime where run unguzxkq read eps 0.0131 while validation said 1.02.
    pipeline._log_certification_probe(
        torch.nn.Linear(100, 4), system, 0.0131, messages.append, probe_rows=120
    )
    assert "NK/P=0.2970" in messages[0]
    assert "BELOW INTERPOLATION FLOOR" in messages[0]


def test_realizable_progress_reuses_transaction_and_theorem_score(monkeypatch) -> None:
    """Only a transaction-approved rate contributes structural value."""
    model = torch.nn.Linear(1, 1)
    x = torch.ones(2, 1)
    y = torch.zeros(2, 1)
    system = SimpleNamespace(factors=(torch.ones(2, 1), torch.ones(2, 1)))
    candidate = DampingCandidate(
        relative_damping=1.0e-4,
        absolute_damping=2.0e-4,
        relative_error=0.2,
        update_norm=1.0,
        approximation_norm=1.0,
        certified_learning_rate=0.4,
        learning_rate=0.1,
        guaranteed_decrease=1.0,
        effective_dof=1.0,
        gcv=1.0,
    )
    choice = DampingChoice(
        candidate=candidate,
        parameter_updates=(torch.ones_like(model.weight),),
        candidates=(candidate,),
        tangent_system=system,
    )
    transaction_calls = []

    monkeypatch.setattr(pipeline, "exact_tangent_system", lambda *args: system)
    monkeypatch.setattr(
        pipeline, "select_projection_damping_factored", lambda *args, **kwargs: choice
    )

    def transaction(**kwargs):
        transaction_calls.append(kwargs)
        return pipeline._TransactionalRealization(
            base_model=kwargs["model"],
            candidate_model=kwargs["model"],
            direction=choice.parameter_updates,
            learning_rate=0.025,
            trials=(),
        )

    monkeypatch.setattr(pipeline, "_transactional_realize_functional_step", transaction)
    predicted_calls = []

    def predicted(**kwargs):
        predicted_calls.append(kwargs)
        return 3.5

    monkeypatch.setattr(pipeline, "_predicted_certified_decrease", predicted)
    config = SimpleNamespace(
        certify_realize_path=True,
        transactional_realized_descent=True,
    )

    measured = pipeline._measure_certified_realizable_progress(
        model=model,
        x=x,
        y=y,
        config=config,
        full_train_batches=[(x, y)],
        device=torch.device("cpu"),
        relative_error=0.0036,
    )

    assert measured.relative_error == pytest.approx(0.0036)
    assert measured.learning_rate == pytest.approx(0.025)
    assert measured.certified_progress == pytest.approx(3.5)
    assert transaction_calls[0]["progress"] is None
    full_train_batches = transaction_calls[0]["full_train_batches"]
    assert full_train_batches[0][0] is x
    assert full_train_batches[0][1] is y
    assert predicted_calls[0]["relative_error"] == pytest.approx(0.2)
    assert predicted_calls[0]["learning_rate"] == pytest.approx(0.025)
