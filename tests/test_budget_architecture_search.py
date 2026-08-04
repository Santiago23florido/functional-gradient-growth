import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from grid_search import budget_search as search
from grid_search.run import enumerate_trials, load_grid, parameter_count

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "grid_search/smaller_architectures_400_stage1.yaml"


@pytest.fixture(scope="module")
def loaded() -> tuple[dict, search.SearchSettings]:
    config = search.load_search_config(CONFIG)
    return config, search.load_settings(config)


@pytest.fixture(scope="module")
def enumerated(
    loaded: tuple[dict, search.SearchSettings],
) -> dict[int, list[tuple[tuple[int, int, int], int]]]:
    return search.architectures_by_seed(loaded[1])


def make_paths(tmp_path: Path) -> search.RunPaths:
    paths = search.RunPaths(REPO_ROOT, tmp_path, tmp_path / "run")
    search.ensure_run_directories(paths)
    return paths


def result_for(entry: dict, validation: float, test: float, loss: float = 0.2) -> dict:
    return {
        "trial_id": entry["deterministic_trial_id"],
        "stage": entry["stage"],
        "seed": entry["seed"],
        "architecture": entry["architecture"],
        "parameters": entry["parameters"],
        "eligible_as_smaller_winner": entry.get(
            "eligible_as_smaller_winner", True
        ),
        "overrides": entry["overrides"],
        "best_validation_epoch": 10,
        "validation_accuracy": validation,
        "validation_loss": loss,
        "test_accuracy": test,
        "test_loss": 1.0 - test,
        "final_validation_accuracy": validation,
        "final_test_accuracy": test,
        "elapsed_seconds": 1.0,
        "status": "complete",
    }


def make_entry(
    index: int,
    seed: int,
    architecture: tuple[int, int, int],
    stage: str = "stage1",
    eligible: bool = True,
    overrides: dict | None = None,
) -> dict:
    entry = {
        "trial_index": index,
        "stage": stage,
        "seed": seed,
        "architecture": list(architecture),
        "parameters": parameter_count(architecture),
        "eligible_as_smaller_winner": eligible,
        "overrides": overrides or {"optimizer.learning_rate": 0.01},
    }
    entry["deterministic_trial_id"] = search.deterministic_trial_id(entry)
    return entry


def test_parameter_formula_and_reference_counts(
    loaded: tuple[dict, search.SearchSettings],
) -> None:
    settings = loaded[1]
    assert parameter_count((11, 19, 16)) == (4 + 1) * 11 + (11 + 1) * 19 + (
        19 + 1
    ) * 16 + (16 + 1)
    assert [reference.parameters for reference in settings.references] == [
        620,
        624,
        626,
        602,
    ]


def test_enumeration_is_valid_unique_and_deterministic(
    loaded: tuple[dict, search.SearchSettings],
    enumerated: dict[int, list[tuple[tuple[int, int, int], int]]],
) -> None:
    settings = loaded[1]
    expected_counts = {0: 19360, 1: 19850, 2: 20023, 3: 17413}
    assert {seed: len(rows) for seed, rows in enumerated.items()} == expected_counts
    for reference in settings.references:
        rows = enumerated[reference.seed]
        assert all(
            settings.minimum_parameters <= parameters < reference.parameters
            and min(architecture) >= settings.minimum_width
            for architecture, parameters in rows
        )
        assert len({architecture for architecture, _ in rows}) == len(rows)
        assert rows == sorted(rows, key=lambda item: (item[1], *item[0]))
        assert rows == search.enumerate_architectures(
            minimum_parameters=400,
            reference_parameters=reference.parameters,
            minimum_width=2,
        )


def test_manifest_is_seed_conditional_and_reproducible(
    loaded: tuple[dict, search.SearchSettings], tmp_path: Path
) -> None:
    config, settings = loaded
    first = search.build_stage1_manifest(config, settings, REPO_ROOT)
    second = search.build_stage1_manifest(config, settings, REPO_ROOT)
    one = search.write_jsonl_atomic(tmp_path / "one.jsonl", first)
    two = search.write_jsonl_atomic(tmp_path / "two.jsonl", second)
    assert one.read_bytes() == two.read_bytes()
    budgets = {reference.seed: reference.parameters for reference in settings.references}
    assert all(entry["parameters"] < budgets[entry["seed"]] for entry in first)
    assert len(first) == sum(
        len(
            search.enumerate_architectures(
                minimum_parameters=400,
                reference_parameters=budget,
                minimum_width=2,
            )
        )
        for budget in budgets.values()
    )


def test_trial_config_uses_own_seed_and_fresh_fixed_training(
    loaded: tuple[dict, search.SearchSettings], tmp_path: Path
) -> None:
    entry = make_entry(0, 2, (2, 3, 76))
    pipeline = search._trial_config(loaded[0], entry, make_paths(tmp_path))
    assert pipeline.model.model_seed == 2
    assert tuple(item["mlp"] for item in pipeline.model.stack) == (2, 3, 76)
    assert pipeline.training.method == "normal"
    assert pipeline.growth_schedule.enabled is False
    assert pipeline.wandb.enabled is False


def test_sharding_covers_every_trial_once() -> None:
    entries = [make_entry(index, index % 4, (2, 3, 76)) for index in range(97)]
    shards = [search.manifest_for_shard(entries, index, 11) for index in range(11)]
    flattened = [entry["trial_index"] for shard in shards for entry in shard]
    assert sorted(flattened) == list(range(97))
    assert len(flattened) == len(set(flattened))


def test_resume_skips_complete_and_can_retry_failed(
    loaded: tuple[dict, search.SearchSettings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = make_paths(tmp_path)
    entries = [make_entry(index, 0, (2, 3, 76 + index)) for index in range(2)]
    search.write_jsonl_atomic(paths.manifests / "stage1_manifest.jsonl", entries)
    calls: list[str] = []

    def fake_run(_config: dict, entry: dict, _paths: search.RunPaths) -> dict:
        calls.append(entry["deterministic_trial_id"])
        return result_for(entry, 0.7, 0.6)

    monkeypatch.setattr(search, "_run_compact_trial", fake_run)
    search.run_shard(CONFIG, paths, stage="stage1", shard_index=0, num_shards=1, retry_failed=False)
    search.run_shard(CONFIG, paths, stage="stage1", shard_index=0, num_shards=1, retry_failed=False)
    assert calls == [entry["deterministic_trial_id"] for entry in entries]

    failed = result_for(entries[1], 0.0, 0.0)
    failed.update(status="failed", error="synthetic")
    search.write_jsonl_atomic(
        paths.shards / "stage1/shard_00000.jsonl",
        [result_for(entries[0], 0.7, 0.6), failed],
    )
    calls.clear()
    search.run_shard(CONFIG, paths, stage="stage1", shard_index=0, num_shards=1, retry_failed=True)
    assert calls == [entries[1]["deterministic_trial_id"]]


def test_stage1_ranking_never_uses_test(
    loaded: tuple[dict, search.SearchSettings], tmp_path: Path
) -> None:
    paths = make_paths(tmp_path)
    entries = []
    records = []
    for seed in range(4):
        low_validation = make_entry(len(entries), seed, (2, 3, 76))
        entries.append(low_validation)
        records.append(result_for(low_validation, 0.70, 0.999))
        high_validation = make_entry(len(entries), seed, (2, 77, 2))
        entries.append(high_validation)
        records.append(result_for(high_validation, 0.80, 0.100))
    search.write_jsonl_atomic(paths.manifests / "stage1_manifest.jsonl", entries)
    search.write_jsonl_atomic(paths.shards / "stage1/shard_00000.jsonl", records)
    search.select_stage1(CONFIG, paths)
    selected = json.loads(
        (paths.selected / "stage1_top50_per_seed.json").read_text()
    )
    assert all(rows[0]["architecture"] == [2, 77, 2] for rows in selected["seeds"].values())
    assert (paths.selected / "stage1_top50_per_seed.csv").is_file()
    assert (paths.summaries / "stage1_summary.json").is_file()


def test_stage2_joint_selection_ignores_test_and_controls(
    loaded: tuple[dict, search.SearchSettings], tmp_path: Path
) -> None:
    paths = make_paths(tmp_path)
    stage1_entries = []
    stage1_records = []
    stage2_entries = []
    stage2_records = []
    for reference in loaded[1].references:
        screened = make_entry(len(stage1_entries), reference.seed, (2, 3, 76))
        stage1_entries.append(screened)
        stage1_records.append(result_for(screened, 0.7, 0.7))
        candidates = [
            (2, 3, 76),
            (2, 77, 2),
        ]
        for rank, architecture in enumerate(candidates):
            entry = make_entry(
                len(stage2_entries),
                reference.seed,
                architecture,
                stage="stage2",
                overrides={"optimizer.learning_rate": 0.01 + rank * 0.01},
            )
            stage2_entries.append(entry)
            stage2_records.append(
                result_for(entry, 0.90 if rank else 0.80, 0.20 if rank else 0.999)
            )
        control = make_entry(
            len(stage2_entries),
            reference.seed,
            reference.architecture,
            stage="stage2",
            eligible=False,
        )
        stage2_entries.append(control)
        stage2_records.append(result_for(control, 1.0, 1.0))
    search.write_jsonl_atomic(paths.manifests / "stage1_manifest.jsonl", stage1_entries)
    search.write_jsonl_atomic(paths.shards / "stage1/shard_00000.jsonl", stage1_records)
    search.write_jsonl_atomic(paths.manifests / "stage2_manifest.jsonl", stage2_entries)
    search.write_jsonl_atomic(paths.shards / "stage2/shard_00000.jsonl", stage2_records)
    search._atomic_json(
        paths.summaries / "stage1_summary.json",
        {
            "architectures_screened_by_seed": {str(seed): 1 for seed in range(4)},
            "total_trials": 4,
        },
    )
    final_path = search.finalize(CONFIG, paths)
    final = json.loads(final_path.read_text())
    assert all(row["best_smaller_architecture"] == [2, 77, 2] for row in final["seeds"])
    assert all(row["test_accuracy"] == 0.2 for row in final["seeds"])
    assert all(row["reference_architecture_retrained_test"] == 1.0 for row in final["seeds"])
    for name in ("stage2_summary.json", "stage2_summary.csv", "final_budget_comparison.csv"):
        assert (paths.summaries / name).is_file()


def test_scratch_preflight_rejects_repository_and_outside_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("FGG_SCRATCH_ROOT", str(scratch))
    with pytest.raises(ValueError, match="inside the repository"):
        search.resolve_run_paths(repo_root=REPO_ROOT, run_root=REPO_ROOT / "results")
    with pytest.raises(ValueError, match="child of SCRATCH_ROOT"):
        search.resolve_run_paths(repo_root=REPO_ROOT, run_root=tmp_path / "elsewhere")


def test_existing_grid_is_unchanged_and_exhaustive_count_is_explicit(
    loaded: tuple[dict, search.SearchSettings]
) -> None:
    existing = load_grid(REPO_ROOT / "grid_search/fixed_architectures.yaml")
    assert len(enumerate_trials(existing)) == 672
    config, settings = loaded
    exhaustive = search.SearchSettings(
        minimum_parameters=settings.minimum_parameters,
        minimum_width=settings.minimum_width,
        number_hidden_layers=settings.number_hidden_layers,
        top_k_per_seed=settings.top_k_per_seed,
        include_reference_control=settings.include_reference_control,
        search_mode="exhaustive",
        references=settings.references,
    )
    counts, total, stage2 = search.planned_counts(config, exhaustive, REPO_ROOT)
    assert total == (sum(counts.values()) + 4) * 60
    assert stage2 == 0


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck unavailable")
def test_slurm_scripts_pass_shellcheck() -> None:
    scripts = sorted((REPO_ROOT / "cluster/slurm").glob("budget_search*.sbatch"))
    scripts += sorted((REPO_ROOT / "cluster/slurm").glob("*smaller_architecture_search.sh"))
    subprocess.run(["shellcheck", *map(os.fspath, scripts)], check=True)
