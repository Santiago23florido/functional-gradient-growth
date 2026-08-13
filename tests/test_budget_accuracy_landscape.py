import json
from pathlib import Path

import pytest

from grid_search.budget_search import GrowthReference
from grid_search.plot_budget_landscape import (
    accuracy_summary,
    load_completed_records,
    load_fixed_grid_references,
    plot_landscape,
)


def _record(trial_id: str, seed: int, accuracy: float, parameters: int = 500) -> dict:
    return {
        "trial_id": trial_id,
        "seed": seed,
        "parameters": parameters,
        "test_accuracy": accuracy,
        "status": "complete",
        "eligible_as_smaller_winner": True,
    }


def test_load_and_summary_position(tmp_path: Path) -> None:
    shard = tmp_path / "shards/stage1/shard_00000.jsonl"
    shard.parent.mkdir(parents=True)
    rows = [
        _record("a", 0, 0.80),
        _record("b", 0, 0.90),
        {**_record("failed", 0, 1.0), "status": "failed"},
        {**_record("control", 0, 1.0), "eligible_as_smaller_winner": False},
    ]
    shard.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loaded = load_completed_records(tmp_path, "stage1")
    reference = GrowthReference(0, (2, 2, 2), 50, 0.85)
    summary = accuracy_summary(loaded, [reference])[0]
    assert summary["completed_candidates"] == 2
    assert summary["candidate_median"] == pytest.approx(0.85)
    assert summary["growth_empirical_percentile"] == 50.0
    assert summary["candidates_better_than_growth"] == 1


def test_duplicate_trial_is_rejected(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards/stage1"
    shard_dir.mkdir(parents=True)
    content = json.dumps(_record("same", 0, 0.8)) + "\n"
    (shard_dir / "shard_00000.jsonl").write_text(content)
    (shard_dir / "shard_00001.jsonl").write_text(content)
    with pytest.raises(RuntimeError, match="duplicate trial_id"):
        load_completed_records(tmp_path, "stage1")


def test_plot_outputs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    references = [
        GrowthReference(seed, (4, 4, 4), 65, 0.85 + seed * 0.01) for seed in range(4)
    ]
    records = [
        _record(f"{seed}-{index}", seed, 0.70 + index * 0.01, 40 + index)
        for seed in range(4)
        for index in range(20)
    ]
    distribution, landscape = plot_landscape(records, references, tmp_path)
    assert distribution.stat().st_size > 0
    assert landscape.stat().st_size > 0


def test_load_under600_fixed_grid_references() -> None:
    references = load_fixed_grid_references(
        Path("grid_search/fixed_architectures_ladder_under600.yaml")
    )

    assert [reference.seed for reference in references] == [0, 1, 2, 3]
    assert [reference.architecture for reference in references] == [
        (16, 19, 9),
        (9, 18, 18),
        (13, 16, 15),
        (11, 17, 17),
    ]
    assert [reference.parameters for reference in references] == [593, 586, 560, 583]
    assert [reference.test_accuracy for reference in references] == [
        0.9522705078125,
        0.9573974609375,
        0.96142578125,
        0.958740234375,
    ]
