from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from grid_search.run import paired_same_seed_retraining_summary, summarize

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
        grid["paired_same_seed_evaluation"] = {
            "protocol": "same_seed_retraining",
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
    architecture_name = "-".join(map(str, architecture))
    return {
        "status": "complete",
        "trial": {
            "trial_id": f"trial_{architecture_name}_{seed}_{learning_rate}",
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
        "parameters": 100 + seed,
        "elapsed_seconds": 20.0 + seed,
    }


def _completed() -> list[dict[str, Any]]:
    completed = []
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        for seed in range(4):
            is_reference_pair = architecture_index == seed
            completed.append(
                _result(
                    architecture,
                    seed,
                    0.1,
                    validation_accuracy=0.80 if is_reference_pair else 0.99,
                    validation_loss=0.20,
                    test_accuracy=FIXED_TESTS[seed] if is_reference_pair else 1.0,
                )
            )
            completed.append(
                _result(
                    architecture,
                    seed,
                    0.2,
                    validation_accuracy=0.70,
                    validation_loss=0.30,
                    # The losing candidate has a perfect test score to prove
                    # that test never participates in selection.
                    test_accuracy=1.0,
                )
            )
    return completed


def test_selection_is_restricted_to_matching_architecture_and_seed() -> None:
    summary = paired_same_seed_retraining_summary(_grid(), _completed())

    for pair in summary["pairs"]:
        seed = pair["seed"]
        assert pair["architecture"] == list(ARCHITECTURES[seed])
        assert pair["selected_overrides"] == {"optimizer.learning_rate": 0.1}
        assert pair["fixed_test_accuracy"] == pytest.approx(FIXED_TESTS[seed])
        assert f"_{seed}_0.1" in pair["trial_id"]


def test_other_seed_and_other_architecture_cannot_win() -> None:
    completed = _completed()
    # These two distractors have better validation than seed 0's legal
    # candidates but violate one side of the exact (architecture, seed) pair.
    assert (
        next(
            result
            for result in completed
            if result["trial"]["architecture"] == [2]
            and result["trial"]["model_seed"] == 1
        )["best"]["validation_accuracy"]
        == 0.99
    )
    assert (
        next(
            result
            for result in completed
            if result["trial"]["architecture"] == [3]
            and result["trial"]["model_seed"] == 0
        )["best"]["validation_accuracy"]
        == 0.99
    )

    pair_zero = paired_same_seed_retraining_summary(_grid(), completed)["pairs"][0]
    assert pair_zero["architecture"] == [2]
    assert pair_zero["seed"] == 0
    assert pair_zero["validation_accuracy"] == pytest.approx(0.80)


def test_validation_not_test_selects_trial_and_reported_test_matches_it() -> None:
    summary = paired_same_seed_retraining_summary(_grid(), _completed())
    pair_zero = summary["pairs"][0]

    assert pair_zero["selected_overrides"] == {"optimizer.learning_rate": 0.1}
    assert pair_zero["fixed_test_accuracy"] == pytest.approx(0.50)
    assert pair_zero["trial_id"] == "trial_2_0_0.1"
    assert pair_zero["best_validation_epoch"] == 10
    assert pair_zero["parameters"] == 100
    assert pair_zero["elapsed_seconds"] == pytest.approx(20.0)


def test_validation_loss_then_overrides_break_ties_deterministically() -> None:
    completed = _completed()
    for result in completed:
        if result["trial"]["architecture"] == list(
            ARCHITECTURES[result["trial"]["model_seed"]]
        ):
            result["best"]["validation_accuracy"] = 0.8
            result["best"]["validation_loss"] = 0.2

    deterministic = paired_same_seed_retraining_summary(_grid(), completed)
    assert all(
        pair["selected_overrides"] == {"optimizer.learning_rate": 0.1}
        for pair in deterministic["pairs"]
    )

    for result in completed:
        if (
            result["trial"]["architecture"]
            == list(ARCHITECTURES[result["trial"]["model_seed"]])
            and result["trial"]["overrides"]["optimizer.learning_rate"] == 0.2
        ):
            result["best"]["validation_loss"] = 0.1
    loss_tiebreak = paired_same_seed_retraining_summary(_grid(), completed)
    assert all(
        pair["selected_overrides"] == {"optimizer.learning_rate": 0.2}
        for pair in loss_tiebreak["pairs"]
    )


def test_differences_and_aggregate_are_exact() -> None:
    summary = paired_same_seed_retraining_summary(_grid(), _completed())
    differences = [fixed - growth for fixed, growth in zip(FIXED_TESTS, GROWTH_TESTS)]
    aggregate = summary["aggregate"]

    assert [pair["fixed_minus_growth"] for pair in summary["pairs"]] == pytest.approx(
        differences
    )
    assert aggregate["number_of_pairs"] == 4
    assert aggregate["fixed_mean_test_accuracy"] == pytest.approx(
        statistics.mean(FIXED_TESTS)
    )
    assert aggregate["fixed_std_test_accuracy"] == pytest.approx(
        statistics.stdev(FIXED_TESTS)
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


def test_missing_or_incomplete_candidate_has_pair_context() -> None:
    completed = [
        result
        for result in _completed()
        if not (
            result["trial"]["architecture"] == [2]
            and result["trial"]["model_seed"] == 0
            and result["trial"]["overrides"]["optimizer.learning_rate"] == 0.1
        )
    ]

    with pytest.raises(RuntimeError) as error:
        paired_same_seed_retraining_summary(_grid(), completed)
    message = str(error.value)
    assert "seed=0" in message
    assert "expected_architecture=[2]" in message
    assert "complete_candidates=1/2" in message
    assert "missing_or_incomplete" in message


def test_growth_references_must_match_seeds_and_architectures() -> None:
    missing_seed = _grid()
    del missing_seed["paired_same_seed_evaluation"]["growth_runs"][3]
    with pytest.raises(ValueError, match="exactly model_seeds"):
        paired_same_seed_retraining_summary(missing_seed, _completed())

    unknown_architecture = _grid()
    unknown_architecture["paired_same_seed_evaluation"]["growth_runs"][0][
        "architecture"
    ] = [99]
    with pytest.raises(ValueError, match="not present"):
        paired_same_seed_retraining_summary(unknown_architecture, _completed())


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
    assert json.loads(output.read_text())["paired_reference_mean_test_accuracy"] is None
    assert not (tmp_path / "paired_same_seed_retraining_summary.json").exists()


def test_summarize_writes_exploratory_and_same_seed_outputs(tmp_path: Path) -> None:
    grid = _grid(tmp_path)
    trial_dir = tmp_path / "trials"
    trial_dir.mkdir()
    for index, payload in enumerate(_completed()):
        (trial_dir / f"trial_{index:05d}.json").write_text(json.dumps(payload))

    exploratory_output = summarize(grid)
    paired_output = tmp_path / "paired_same_seed_retraining_summary.json"

    assert exploratory_output.exists()
    assert paired_output.exists()
    exploratory = json.loads(exploratory_output.read_text())
    assert exploratory["paired_reference_mean_test_accuracy"] == pytest.approx(
        statistics.mean(GROWTH_TESTS)
    )
    paired = json.loads(paired_output.read_text())
    assert paired["aggregate"]["number_of_pairs"] == 4
    assert len(paired["pairs"]) == 4
