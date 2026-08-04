from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from grid_search.run import paired_leave_one_seed_out_summary, summarize

ARCHITECTURES = ([2], [3], [4], [5])
GROWTH_TESTS = (0.60, 0.70, 0.80, 0.90)
FIXED_TESTS = (0.50, 0.80, 0.80, 0.95)


def _grid(tmp_path: Path | None = None, *, paired: bool = True) -> dict[str, Any]:
    grid: dict[str, Any] = {
        "base_config": "configs/fgd/family_ladder_N1024.yaml",
        "results_dir": str(tmp_path or "unused"),
        "architectures": ARCHITECTURES,
        "model_seeds": [0, 1, 2, 3],
        "search_spaces": [{"optimizer.learning_rate": [0.1, 0.2]}],
    }
    if paired:
        grid["paired_evaluation"] = {
            "protocol": "leave_one_seed_out",
            "growth_runs": {
                seed: {
                    "architecture": architecture,
                    "test_accuracy": GROWTH_TESTS[seed],
                }
                for seed, architecture in enumerate(ARCHITECTURES)
            },
        }
    return grid


def _result(
    architecture: list[int],
    seed: int,
    learning_rate: float,
    *,
    validation_accuracy: float,
    validation_loss: float,
    test_accuracy: float,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "trial": {
            "architecture": architecture,
            "model_seed": seed,
            "overrides": {"optimizer.learning_rate": learning_rate},
        },
        "best_validation_epoch": 10 + seed,
        "best": {
            "validation_accuracy": validation_accuracy,
            "validation_loss": validation_loss,
            "test_accuracy": test_accuracy,
        },
    }


def _completed() -> list[dict[str, Any]]:
    completed = []
    for held_out_seed, architecture in enumerate(ARCHITECTURES):
        for seed in range(4):
            completed.append(
                _result(
                    architecture,
                    seed,
                    0.1,
                    validation_accuracy=0.80,
                    validation_loss=0.20,
                    test_accuracy=FIXED_TESTS[held_out_seed]
                    if seed == held_out_seed
                    else 0.01,
                )
            )
            completed.append(
                _result(
                    architecture,
                    seed,
                    0.2,
                    # An exceptional held-out seed must not overturn the
                    # inferior mean on the three selection seeds.
                    validation_accuracy=0.99 if seed == held_out_seed else 0.70,
                    validation_loss=0.01 if seed == held_out_seed else 0.30,
                    # Test is deliberately enormous everywhere for the loser.
                    test_accuracy=1.0,
                )
            )
    return completed


def test_held_out_seed_and_test_do_not_influence_selection() -> None:
    summary = paired_leave_one_seed_out_summary(_grid(), _completed())

    for fold in summary["folds"]:
        assert fold["held_out_seed"] not in fold["selection_seeds"]
        assert fold["selected_overrides"] == {"optimizer.learning_rate": 0.1}
        assert fold["selection_mean_validation_accuracy"] == pytest.approx(0.80)
        assert fold["fixed_test_accuracy"] == pytest.approx(
            FIXED_TESTS[fold["held_out_seed"]]
        )


def test_paired_differences_and_aggregate_are_exact() -> None:
    summary = paired_leave_one_seed_out_summary(_grid(), _completed())
    differences = [fixed - growth for fixed, growth in zip(FIXED_TESTS, GROWTH_TESTS)]
    aggregate = summary["aggregate"]

    assert [fold["fixed_minus_growth"] for fold in summary["folds"]] == pytest.approx(
        differences
    )
    assert aggregate["number_of_folds"] == 4
    assert aggregate["fixed_mean_test_accuracy"] == pytest.approx(
        statistics.mean(FIXED_TESTS)
    )
    assert aggregate["growth_mean_test_accuracy"] == pytest.approx(
        statistics.mean(GROWTH_TESTS)
    )
    assert aggregate["mean_paired_difference"] == pytest.approx(
        statistics.mean(differences)
    )
    assert aggregate["std_paired_difference"] == pytest.approx(
        statistics.stdev(differences)
    )
    assert aggregate["standard_error_paired_difference"] == pytest.approx(
        statistics.stdev(differences) / 2
    )
    assert (aggregate["fixed_wins"], aggregate["growth_wins"], aggregate["ties"]) == (
        2,
        1,
        1,
    )


def test_validation_loss_then_overrides_break_selection_ties() -> None:
    completed = _completed()
    for result in completed:
        result["best"]["validation_accuracy"] = 0.8
        result["best"]["validation_loss"] = 0.2

    deterministic = paired_leave_one_seed_out_summary(_grid(), completed)
    assert all(
        fold["selected_overrides"] == {"optimizer.learning_rate": 0.1}
        for fold in deterministic["folds"]
    )

    for result in completed:
        if result["trial"]["overrides"]["optimizer.learning_rate"] == 0.2:
            result["best"]["validation_loss"] = 0.1
    loss_tiebreak = paired_leave_one_seed_out_summary(_grid(), completed)
    assert all(
        fold["selected_overrides"] == {"optimizer.learning_rate": 0.2}
        for fold in loss_tiebreak["folds"]
    )


def test_missing_seed_fails_with_fold_and_configuration_context() -> None:
    completed = [
        result
        for result in _completed()
        if not (
            result["trial"]["architecture"] == [2]
            and result["trial"]["model_seed"] == 2
            and result["trial"]["overrides"]["optimizer.learning_rate"] == 0.1
        )
    ]

    with pytest.raises(RuntimeError) as error:
        paired_leave_one_seed_out_summary(_grid(), completed)
    message = str(error.value)
    assert "architecture=[2]" in message
    assert "held_out_seed=0" in message
    assert "missing_seeds" in message
    assert "2" in message
    assert "optimizer.learning_rate" in message


def test_growth_references_must_match_seeds_and_architectures() -> None:
    missing_seed = _grid()
    del missing_seed["paired_evaluation"]["growth_runs"][3]
    with pytest.raises(ValueError, match="exactly model_seeds"):
        paired_leave_one_seed_out_summary(missing_seed, _completed())

    unknown_architecture = _grid()
    unknown_architecture["paired_evaluation"]["growth_runs"][0]["architecture"] = [99]
    with pytest.raises(ValueError, match="not present"):
        paired_leave_one_seed_out_summary(unknown_architecture, _completed())


def test_legacy_summary_still_writes_only_exploratory_output(tmp_path: Path) -> None:
    grid = {
        "base_config": "configs/fgd/family_ladder_N1024.yaml",
        "results_dir": str(tmp_path),
        "architectures": [[2]],
        "model_seeds": [0],
        "search_spaces": [{"optimizer.learning_rate": [0.1]}],
    }
    trial_dir = tmp_path / "trials"
    trial_dir.mkdir()
    payload = _result(
        [2],
        0,
        0.1,
        validation_accuracy=0.8,
        validation_loss=0.2,
        test_accuracy=0.7,
    )
    (trial_dir / "trial_00000.json").write_text(json.dumps(payload))

    output = summarize(grid)

    assert output == tmp_path / "summary.json"
    assert output.exists()
    assert not (tmp_path / "paired_leave_one_seed_out_summary.json").exists()


def test_summarize_writes_exploratory_and_paired_outputs(tmp_path: Path) -> None:
    grid = _grid(tmp_path)
    trial_dir = tmp_path / "trials"
    trial_dir.mkdir()
    for index, payload in enumerate(_completed()):
        (trial_dir / f"trial_{index:05d}.json").write_text(json.dumps(payload))

    exploratory_output = summarize(grid)
    paired_output = tmp_path / "paired_leave_one_seed_out_summary.json"

    assert exploratory_output.exists()
    assert paired_output.exists()
    paired = json.loads(paired_output.read_text())
    assert paired["aggregate"]["number_of_folds"] == 4
    assert len(paired["folds"]) == 4
