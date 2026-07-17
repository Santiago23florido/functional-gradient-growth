"""Per-epoch metrics must describe the family that committed the step."""

from __future__ import annotations

from dataclasses import replace

import pytest

from stable_tiny.pipeline import HistoryEntry
from stable_tiny.plotting import plot_relative_error
from stable_tiny.wandb_logging import WandbConfig, WandbRunLogger


class _StubRun:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def log(self, payload: dict) -> None:
        self.payloads.append(payload)


def _entry(**overrides) -> HistoryEntry:
    base = HistoryEntry(
        step=5,
        step_type="FGD",
        train_loss=0.1,
        validation_loss=0.1,
        test_loss=0.1,
        train_accuracy=0.5,
        validation_accuracy=0.5,
        test_accuracy=0.5,
        learning_rate=0.0,
        num_params=100,
    )
    return replace(base, **overrides)


def _log(entry: HistoryEntry) -> dict:
    logger = WandbRunLogger(WandbConfig(enabled=True))
    stub = _StubRun()
    logger._run = stub
    logger.log_history_entry(entry)
    assert len(stub.payloads) == 1
    return stub.payloads[0]


def test_rejected_epoch_logs_state_diagnostic_not_step_metrics() -> None:
    payload = _log(
        _entry(
            rel_error=2.9,
            fgd_candidate_accepted=False,
            fgd_approximation_kind="tangent",
        )
    )
    assert payload["fgd/tangent_relative_error"] == pytest.approx(2.9)
    assert "fgd/relative_error" not in payload
    assert "fgd/family_index" not in payload


def test_committed_family_entry_logs_its_own_certificate() -> None:
    payload = _log(
        _entry(
            step_type="SEC",
            rel_error=0.99,
            learning_rate=0.005,
            fgd_candidate_accepted=True,
            fgd_approximation_kind="parametric_descent",
            fgd_global_contraction=0.97,
            fgd_global_bound=1.2,
            fgd_global_bound_valid=True,
            fgd_theory_descent_coefficient=0.4,
        )
    )
    assert payload["fgd/relative_error"] == pytest.approx(0.99)
    assert payload["fgd/family_index"] == 3
    assert payload["fgd/global_contraction"] == pytest.approx(0.97)
    assert payload["fgd/global_bound_valid"] is True
    # The tangent state diagnostic never comes from a family entry.
    assert "fgd/tangent_relative_error" not in payload


def test_committed_tangent_epoch_logs_both_series() -> None:
    payload = _log(
        _entry(
            rel_error=0.4,
            fgd_candidate_accepted=True,
            fgd_approximation_kind="tangent",
        )
    )
    assert payload["fgd/tangent_relative_error"] == pytest.approx(0.4)
    assert payload["fgd/relative_error"] == pytest.approx(0.4)
    assert payload["fgd/family_index"] == 0


def test_rejected_rkhs_entry_keeps_namespace_out_of_step_metrics() -> None:
    payload = _log(
        _entry(
            step_type="RKHS",
            rel_error=0.3,
            fgd_candidate_accepted=False,
            fgd_approximation_kind="rkhs_head",
            fgd_global_bound=9.9,
            fgd_rkhs_functional_loss=0.43,
        )
    )
    assert "fgd/relative_error" not in payload
    assert "fgd/global_bound" not in payload
    assert payload["fgd/rkhs_functional_loss"] == pytest.approx(0.43)


def test_plot_relative_error_groups_by_committed_family(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    history = [
        _entry(step=1, rel_error=0.7, fgd_candidate_accepted=False),
        _entry(
            step=1,
            step_type="SEC",
            rel_error=0.95,
            fgd_candidate_accepted=True,
            fgd_approximation_kind="parametric_descent",
        ),
        _entry(step=2, rel_error=0.45, fgd_candidate_accepted=True),
    ]
    saved = plot_relative_error(
        history,
        output_path=tmp_path / "rel_error.png",
        threshold=0.5,
    )
    assert saved is not None and saved.exists()
