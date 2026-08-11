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


def _render_launcher_config(source: str, guard: bool) -> str:
    rendered = re.sub(
        r"^  data_dir:.*$",
        "  data_dir: /tmp/mnist-full-ab-test",
        source,
        flags=re.MULTILINE,
    )
    return re.sub(
        r"^  transactional_realized_descent:.*$",
        "  transactional_realized_descent: true\n"
        f"  transactional_family_step: {str(guard).lower()}",
        rendered,
        flags=re.MULTILINE,
    )


def test_guard_arms_differ_only_in_family_transaction(tmp_path: Path) -> None:
    """Pin the effective YAML produced by the existing two-arm launcher."""
    source = Path("configs/experiments/mnist_full.yaml").read_text(encoding="utf-8")
    paths = []
    for guard in (True, False):
        path = tmp_path / f"guard-{str(guard).lower()}.yaml"
        path.write_text(_render_launcher_config(source, guard), encoding="utf-8")
        paths.append(path)

    guard_on = load_pipeline_config(paths[0])
    guard_off = load_pipeline_config(paths[1])
    assert _differing_paths(guard_on, guard_off) == {
        "fgd_approx.transactional_family_step"
    }
    assert guard_on.fgd_approx.transactional_realized_descent is True
    assert guard_off.fgd_approx.transactional_realized_descent is True
    assert guard_on.fgd_approx.certify_realizable_progress_growth is True
    assert guard_off.fgd_approx.certify_realizable_progress_growth is True
    assert (
        guard_on.fgd_approx.certify_probe_refine_on_transaction_mismatch is True
    )
    assert (
        guard_off.fgd_approx.certify_probe_refine_on_transaction_mismatch is True
    )

    launcher = Path("cluster/slurm/mnist_full_family_guard_ab.sbatch").read_text(
        encoding="utf-8"
    )
    assert "0) GUARD=true;  ARM=guard-on" in launcher
    assert "1) GUARD=false; ARM=guard-off" in launcher
    assert "--gpus=1" in launcher
    assert 'RUN_SUFFIX="${RUN_SUFFIX:-}"' in launcher


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
    assert conv.certify_probe_refine_on_transaction_mismatch is False

    for path in Path("configs/fgd").glob("*.yaml"):
        config = load_pipeline_config(path).fgd_approx
        expected = path.name == "mnist_conv_matrix_free.yaml"
        assert config.certify_probe_diagnostics is expected, path
        assert config.certify_realizable_progress_growth is expected, path
        assert config.certify_probe_refine_on_transaction_mismatch is False, path


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

    with pytest.raises(ValueError, match="max_rounds"):
        validate_probe_refinement(
            dataclasses.replace(config, certify_probe_refine_max_rounds=0)
        )
    with pytest.raises(ValueError, match="positive integer"):
        validate_probe_refinement(
            dataclasses.replace(config, certify_probe_refine_batches_per_round=0)
        )
    with pytest.raises(ValueError, match="probe_resample"):
        validate_probe_refinement(dataclasses.replace(config, probe_resample=True))


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
