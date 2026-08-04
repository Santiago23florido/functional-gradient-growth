"""Scratch-backed, sharded search for strictly smaller fixed architectures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from grid_search.run import apply_dotted_overrides, parameter_count
from stable_tiny.pipeline import (
    load_pipeline_config,
    run_pipeline,
    write_outputs,
)

DEFAULT_CONFIG = Path("grid_search/smaller_architectures_400_stage1.yaml")
DEFAULT_RUN_NAME = "smaller_architectures_400_v1"
METHODOLOGY_WARNING = (
    "The two-stage search does not tune all hyperparameters for every "
    "architecture. Every architecture is screened with one fixed recipe and "
    "only TOP_K proceeds to fine-tuning. The result establishes absence of a "
    "better candidate within this evaluated protocol, not mathematical "
    "non-existence of a better architecture."
)


@dataclass(frozen=True)
class GrowthReference:
    seed: int
    architecture: tuple[int, int, int]
    parameters: int
    test_accuracy: float


@dataclass(frozen=True)
class SearchSettings:
    minimum_parameters: int
    minimum_width: int
    number_hidden_layers: int
    top_k_per_seed: int
    include_reference_control: bool
    search_mode: str
    references: tuple[GrowthReference, ...]


@dataclass(frozen=True)
class RunPaths:
    repo_root: Path
    scratch_root: Path
    run_root: Path

    @property
    def manifests(self) -> Path:
        return self.run_root / "manifests"

    @property
    def shards(self) -> Path:
        return self.run_root / "shards"

    @property
    def summaries(self) -> Path:
        return self.run_root / "summaries"

    @property
    def selected(self) -> Path:
        return self.run_root / "selected"

    @property
    def metadata(self) -> Path:
        return self.run_root / "metadata"

    @property
    def failures(self) -> Path:
        return self.run_root / "failures"


def load_search_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {"base_config", "stage2_config", "architecture_budget_search"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"budget-search config is missing: {', '.join(missing)}")
    return payload


def load_settings(config: Mapping[str, Any]) -> SearchSettings:
    raw = config["architecture_budget_search"]
    if raw.get("protocol") != "same_seed_strictly_smaller":
        raise ValueError(
            "architecture_budget_search.protocol must be 'same_seed_strictly_smaller'"
        )
    growth_runs = raw.get("growth_runs")
    if not isinstance(growth_runs, Mapping) or not growth_runs:
        raise ValueError("architecture_budget_search.growth_runs is required")
    references = []
    for raw_seed, value in growth_runs.items():
        seed = int(raw_seed)
        architecture = tuple(int(width) for width in value["architecture"])
        if len(architecture) != 3:
            raise ValueError(f"seed {seed} reference must have three hidden layers")
        actual = parameter_count(architecture)
        expected = int(value["expected_parameters"])
        if actual != expected:
            raise ValueError(
                f"seed {seed} reference parameter mismatch: "
                f"architecture={list(architecture)}, expected={expected}, actual={actual}"
            )
        references.append(
            GrowthReference(
                seed=seed,
                architecture=architecture,
                parameters=actual,
                test_accuracy=float(value["test_accuracy"]),
            )
        )
    references.sort(key=lambda reference: reference.seed)
    search_mode = str(raw.get("search_mode", "two_stage"))
    if search_mode not in {"two_stage", "exhaustive"}:
        raise ValueError("search_mode must be 'two_stage' or 'exhaustive'")
    settings = SearchSettings(
        minimum_parameters=int(raw.get("minimum_parameters", 400)),
        minimum_width=int(raw.get("minimum_width", 2)),
        number_hidden_layers=int(raw.get("number_hidden_layers", 3)),
        top_k_per_seed=int(raw.get("top_k_per_seed", 50)),
        include_reference_control=bool(raw.get("include_reference_control", True)),
        search_mode=search_mode,
        references=tuple(references),
    )
    if settings.number_hidden_layers != 3:
        raise ValueError("budget search currently requires exactly three hidden layers")
    if settings.minimum_width < 1 or settings.minimum_parameters < 1:
        raise ValueError("minimum_width and minimum_parameters must be positive")
    if settings.top_k_per_seed < 1:
        raise ValueError("top_k_per_seed must be positive")
    return settings


def enumerate_architectures(
    *,
    minimum_parameters: int,
    reference_parameters: int,
    minimum_width: int,
) -> list[tuple[tuple[int, int, int], int]]:
    """Enumerate all valid three-layer widths without arbitrary upper bounds."""
    architectures: list[tuple[tuple[int, int, int], int]] = []
    w1 = minimum_width
    while parameter_count((w1, minimum_width, minimum_width)) < reference_parameters:
        w2 = minimum_width
        while parameter_count((w1, w2, minimum_width)) < reference_parameters:
            w3 = minimum_width
            while True:
                architecture = (w1, w2, w3)
                parameters = parameter_count(architecture)
                if parameters >= reference_parameters:
                    break
                if parameters >= minimum_parameters:
                    architectures.append((architecture, parameters))
                w3 += 1
            w2 += 1
        w1 += 1
    architectures.sort(key=lambda item: (item[1], *item[0]))
    if len({architecture for architecture, _ in architectures}) != len(architectures):
        raise AssertionError("architecture enumeration produced duplicates")
    return architectures


def architectures_by_seed(
    settings: SearchSettings,
) -> dict[int, list[tuple[tuple[int, int, int], int]]]:
    return {
        reference.seed: enumerate_architectures(
            minimum_parameters=settings.minimum_parameters,
            reference_parameters=reference.parameters,
            minimum_width=settings.minimum_width,
        )
        for reference in settings.references
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def deterministic_trial_id(entry: Mapping[str, Any]) -> str:
    identity = {
        "stage": entry["stage"],
        "seed": entry["seed"],
        "architecture": entry["architecture"],
        "overrides": entry["overrides"],
    }
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()[:16]


def _expand_search_spaces(spaces: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    combinations: list[dict[str, Any]] = []
    for space in spaces:
        keys = list(space)
        value_lists = [
            value if isinstance(value, list) else [value] for value in space.values()
        ]
        combinations.extend(
            dict(zip(keys, values)) for values in itertools.product(*value_lists)
        )
    return combinations


def stage2_overrides(
    config: Mapping[str, Any], repo_root: Path
) -> list[dict[str, Any]]:
    stage2_path = Path(config["stage2_config"])
    if not stage2_path.is_absolute():
        stage2_path = repo_root / stage2_path
    raw = yaml.safe_load(stage2_path.read_text(encoding="utf-8")) or {}
    fixed = dict(raw.get("fixed_overrides", {}))
    combinations = _expand_search_spaces(raw.get("search_spaces", [{}]))
    return [{**fixed, **combination} for combination in combinations]


def build_stage1_manifest(
    config: Mapping[str, Any], settings: SearchSettings, repo_root: Path
) -> list[dict[str, Any]]:
    per_seed = architectures_by_seed(settings)
    screening = dict(config.get("screening_overrides", {}))
    fine_grid = stage2_overrides(config, repo_root)
    entries: list[dict[str, Any]] = []
    for reference in settings.references:
        overrides_list = (
            fine_grid if settings.search_mode == "exhaustive" else [screening]
        )
        architectures = list(per_seed[reference.seed])
        if settings.search_mode == "exhaustive" and settings.include_reference_control:
            architectures.append((reference.architecture, reference.parameters))
        for architecture, parameters in architectures:
            for overrides in overrides_list:
                entry = {
                    "trial_index": len(entries),
                    "stage": "stage1",
                    "seed": reference.seed,
                    "architecture": list(architecture),
                    "parameters": parameters,
                    "eligible_as_smaller_winner": parameters < reference.parameters,
                    "overrides": overrides,
                }
                entry["deterministic_trial_id"] = deterministic_trial_id(entry)
                entries.append(entry)
    return entries


def _atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    return _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl_atomic(path: Path, entries: Sequence[Mapping[str, Any]]) -> Path:
    return _atomic_text(path, "".join(_canonical(entry) + "\n" for entry in entries))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)
    return path


def resolve_run_paths(
    *, repo_root: Path, run_root: Path | None = None, min_free_gb: float = 1.0
) -> RunPaths:
    repo_root = repo_root.expanduser().resolve()
    run_name = os.environ.get("RUN_NAME", DEFAULT_RUN_NAME)
    if not run_name.strip():
        raise ValueError("RUN_NAME must not be empty")
    scratch_candidate = Path(
        os.environ.get("FGG_SCRATCH_ROOT", str(Path.home() / "datasets"))
    ).expanduser()
    if not scratch_candidate.exists():
        raise FileNotFoundError(
            f"scratch root does not exist: {scratch_candidate} (expected ~/datasets)"
        )
    scratch_root = scratch_candidate.resolve()
    if not os.access(scratch_root, os.W_OK):
        raise PermissionError(f"scratch root is not writable: {scratch_root}")
    if run_root is None:
        explicit = os.environ.get("FGG_RUN_ROOT")
        run_root = (
            Path(explicit).expanduser()
            if explicit
            else scratch_root / "functional-gradient-growth" / run_name
        )
    run_root = run_root.expanduser().resolve()
    if run_root == repo_root or run_root.is_relative_to(repo_root):
        raise ValueError(f"RUN_ROOT must not be inside the repository: {run_root}")
    if run_root == scratch_root or not run_root.is_relative_to(scratch_root):
        raise ValueError(
            "RUN_ROOT must be a child of SCRATCH_ROOT "
            f"({scratch_root}): {run_root}"
        )
    free_gb = shutil.disk_usage(scratch_root).free / (1024**3)
    if free_gb < min_free_gb:
        raise OSError(
            f"insufficient scratch space: {free_gb:.2f} GiB free, "
            f"MIN_FREE_GB={min_free_gb:.2f}"
        )
    return RunPaths(repo_root, scratch_root, run_root)


def ensure_run_directories(paths: RunPaths) -> None:
    for directory in (
        paths.manifests,
        paths.shards / "stage1",
        paths.shards / "stage2",
        paths.summaries,
        paths.selected,
        paths.run_root / "slurm_logs",
        paths.metadata,
        paths.failures / "stage1",
        paths.failures / "stage2",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def manifest_for_shard(
    entries: Sequence[Mapping[str, Any]], shard_index: int, num_shards: int
) -> list[Mapping[str, Any]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard index/count")
    return [
        entry
        for entry in entries
        if int(entry["trial_index"]) % num_shards == shard_index
    ]


def _manifest_path(paths: RunPaths, stage: str) -> Path:
    return paths.manifests / f"{stage}_manifest.jsonl"


def _shard_path(paths: RunPaths, stage: str, shard_index: int) -> Path:
    return paths.shards / stage / f"shard_{shard_index:05d}.jsonl"


def print_preflight(
    paths: RunPaths,
    *,
    expected_stage1: int,
    expected_stage2: int,
    stage1_num_shards: int,
    stage2_num_shards: int,
    max_concurrent: int,
) -> None:
    print(f"pwd={Path.cwd()}")
    print(f"REPO_ROOT={paths.repo_root}")
    print(f"SCRATCH_ROOT={paths.scratch_root}")
    print(f"RUN_ROOT={paths.run_root}")
    subprocess.run(["df", "-h", str(paths.scratch_root)], check=True)
    print(f"expected_stage1_trials={expected_stage1}")
    print(f"expected_stage2_trials={expected_stage2}")
    print(f"stage1_num_shards={stage1_num_shards}")
    print(f"stage2_num_shards={stage2_num_shards}")
    print(f"max_concurrent={max_concurrent}")


def planned_counts(
    config: Mapping[str, Any], settings: SearchSettings, repo_root: Path
) -> tuple[dict[int, int], int, int]:
    counts = {
        seed: len(architectures)
        for seed, architectures in architectures_by_seed(settings).items()
    }
    combinations = len(stage2_overrides(config, repo_root))
    if settings.search_mode == "exhaustive":
        stage1 = sum(counts.values()) * combinations
        if settings.include_reference_control:
            stage1 += len(settings.references) * combinations
        stage2 = 0
    else:
        stage1 = sum(counts.values())
        per_seed = settings.top_k_per_seed + int(settings.include_reference_control)
        stage2 = len(settings.references) * per_seed * combinations
    return counts, stage1, stage2


def prepare_stage1(
    config_path: Path,
    paths: RunPaths,
    *,
    stage1_num_shards: int,
    stage2_num_shards: int,
    max_concurrent: int,
) -> Path:
    config = load_search_config(config_path)
    settings = load_settings(config)
    counts, stage1_count, stage2_count = planned_counts(
        config, settings, paths.repo_root
    )
    if settings.search_mode == "exhaustive":
        print(
            "WARNING: exhaustive mode applies the full hyperparameter grid to "
            f"every architecture ({stage1_count} trials)."
        )
    ensure_run_directories(paths)
    print_preflight(
        paths,
        expected_stage1=stage1_count,
        expected_stage2=stage2_count,
        stage1_num_shards=stage1_num_shards,
        stage2_num_shards=stage2_num_shards,
        max_concurrent=max_concurrent,
    )
    entries = build_stage1_manifest(config, settings, paths.repo_root)
    if len(entries) != stage1_count:
        raise AssertionError(
            f"planned {stage1_count} stage1 trials, built {len(entries)}"
        )
    manifest_path = _manifest_path(paths, "stage1")
    if manifest_path.exists():
        existing = manifest_path.read_bytes()
        candidate = "".join(_canonical(entry) + "\n" for entry in entries).encode()
        if existing != candidate:
            raise RuntimeError(
                f"existing manifest differs; refusing to overwrite {manifest_path}"
            )
        print(f"Reusing byte-identical manifest: {manifest_path}")
    else:
        write_jsonl_atomic(manifest_path, entries)
    _atomic_json(
        paths.metadata / "stage1_plan.json",
        {
            "config": str(config_path.resolve()),
            "architecture_counts_by_seed": counts,
            "expected_stage1_trials": stage1_count,
            "expected_stage2_trials": stage2_count,
            "stage1_num_shards": stage1_num_shards,
            "stage2_num_shards": stage2_num_shards,
            "max_concurrent": max_concurrent,
            "methodology_warning": METHODOLOGY_WARNING,
        },
    )
    print(f"Saved {manifest_path}")
    return manifest_path


def inspect_search(config_path: Path, repo_root: Path) -> None:
    config = load_search_config(config_path)
    settings = load_settings(config)
    _, stage1_count, stage2_count = planned_counts(config, settings, repo_root)
    print(f"search_mode={settings.search_mode}")
    for reference in settings.references:
        architectures = architectures_by_seed(settings)[reference.seed]
        print(
            f"seed={reference.seed}: count={len(architectures)}, "
            f"min_params={architectures[0][1]}, max_params={architectures[-1][1]}, "
            f"reference_params={reference.parameters}"
        )
        for architecture, parameters in architectures[:5]:
            print(f"  {list(architecture)} -> {parameters}")
    print(f"stage1_trials={stage1_count}")
    print(f"planned_stage2_trials={stage2_count}")
    if settings.search_mode == "exhaustive":
        print("WARNING: " + METHODOLOGY_WARNING)


def _trial_config(
    config: Mapping[str, Any], entry: Mapping[str, Any], paths: RunPaths
) -> Any:
    base_path = Path(config["base_config"])
    if not base_path.is_absolute():
        base_path = paths.repo_root / base_path
    pipeline = load_pipeline_config(base_path)
    pipeline = apply_dotted_overrides(pipeline, entry["overrides"])
    architecture = tuple(int(width) for width in entry["architecture"])
    return replace(
        pipeline,
        model=replace(
            pipeline.model,
            stack=tuple({"mlp": width} for width in architecture),
            model_seed=int(entry["seed"]),
        ),
        training=replace(pipeline.training, method="normal"),
        growth_schedule=replace(pipeline.growth_schedule, enabled=False),
        wandb=replace(pipeline.wandb, enabled=False),
        run=replace(
            pipeline.run,
            name=entry["deterministic_trial_id"],
            results_dir=paths.metadata,
            save_plot=False,
            show_plot=False,
        ),
    )


def _run_compact_trial(
    config: Mapping[str, Any], entry: Mapping[str, Any], paths: RunPaths
) -> dict[str, Any]:
    started = time.time()
    try:
        result = run_pipeline(_trial_config(config, entry, paths), progress=None)
        best = max(
            result.history,
            key=lambda item: (
                item.validation_accuracy,
                -item.validation_loss,
                -item.step,
            ),
        )
        final = result.history[-1]
        return {
            "trial_id": entry["deterministic_trial_id"],
            "stage": entry["stage"],
            "seed": entry["seed"],
            "architecture": entry["architecture"],
            "parameters": entry["parameters"],
            "eligible_as_smaller_winner": entry.get("eligible_as_smaller_winner", True),
            "overrides": entry["overrides"],
            "best_validation_epoch": best.step,
            "validation_accuracy": best.validation_accuracy,
            "validation_loss": best.validation_loss,
            "test_accuracy": best.test_accuracy,
            "test_loss": best.test_loss,
            "final_validation_accuracy": final.validation_accuracy,
            "final_test_accuracy": final.test_accuracy,
            "elapsed_seconds": time.time() - started,
            "status": "complete",
        }
    except Exception as error:  # noqa: BLE001 - a worker must persist any trial failure
        return {
            "trial_id": entry["deterministic_trial_id"],
            "stage": entry["stage"],
            "seed": entry["seed"],
            "architecture": entry["architecture"],
            "parameters": entry["parameters"],
            "eligible_as_smaller_winner": entry.get("eligible_as_smaller_winner", True),
            "overrides": entry["overrides"],
            "elapsed_seconds": time.time() - started,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }


def _records_by_id(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        trial_id = str(record["trial_id"])
        if trial_id in indexed:
            raise RuntimeError(f"duplicate trial_id detected: {trial_id}")
        indexed[trial_id] = record
    return indexed


def run_shard(
    config_path: Path,
    paths: RunPaths,
    *,
    stage: str,
    shard_index: int,
    num_shards: int,
    retry_failed: bool,
) -> None:
    config = load_search_config(config_path)
    entries = read_jsonl(_manifest_path(paths, stage))
    assigned = manifest_for_shard(entries, shard_index, num_shards)
    output = _shard_path(paths, stage, shard_index)
    existing = read_jsonl(output)
    indexed = _records_by_id(existing)
    if retry_failed:
        retained = [record for record in existing if record.get("status") == "complete"]
        if len(retained) != len(existing):
            write_jsonl_atomic(output, retained)
        indexed = _records_by_id(retained)
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = paths.failures / stage / f"shard_{shard_index:05d}.jsonl"
    failed = False
    with output.open("a", encoding="utf-8") as handle:
        for entry in assigned:
            trial_id = entry["deterministic_trial_id"]
            previous = indexed.get(trial_id)
            if previous is not None:
                if previous.get("status") == "failed":
                    failed = True
                continue
            record = _run_compact_trial(config, entry, paths)
            handle.write(_canonical(record) + "\n")
            handle.flush()
            indexed[trial_id] = record
            print(
                f"[{stage} shard {shard_index}] {trial_id} status={record['status']}",
                flush=True,
            )
            if record["status"] == "failed":
                failed = True
                failures.parent.mkdir(parents=True, exist_ok=True)
                with failures.open("a", encoding="utf-8") as failure_handle:
                    failure_handle.write(_canonical(record) + "\n")
                    failure_handle.flush()
    if failed:
        raise RuntimeError(
            f"shard {shard_index} contains failed trials; rerun with --retry-failed"
        )


def load_stage_results(paths: RunPaths, stage: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted((paths.shards / stage).glob("shard_*.jsonl")):
        records.extend(read_jsonl(path))
    _records_by_id(records)
    return records


def _selection_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(record["validation_accuracy"]),
        float(record["validation_loss"]),
        int(record["parameters"]),
        _canonical(
            {
                "architecture": record["architecture"],
                "overrides": record["overrides"],
            }
        ),
    )


def _stage1_selection_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(record["validation_accuracy"]),
        float(record["validation_loss"]),
        int(record["parameters"]),
        tuple(record["architecture"]),
    )


def _require_complete_manifest(
    entries: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    stage: str,
) -> None:
    indexed = _records_by_id(records)
    missing = []
    failed = []
    for entry in entries:
        record = indexed.get(entry["deterministic_trial_id"])
        if record is None:
            missing.append(entry["deterministic_trial_id"])
        elif record.get("status") != "complete":
            failed.append(entry["deterministic_trial_id"])
    if missing or failed:
        raise RuntimeError(
            f"{stage} is incomplete: missing={len(missing)} {missing[:10]}, "
            f"failed={len(failed)} {failed[:10]}"
        )


def select_stage1(config_path: Path, paths: RunPaths) -> Path:
    config = load_search_config(config_path)
    settings = load_settings(config)
    entries = read_jsonl(_manifest_path(paths, "stage1"))
    records = load_stage_results(paths, "stage1")
    _require_complete_manifest(entries, records, "stage1")

    per_seed: dict[int, list[dict[str, Any]]] = {}
    for reference in settings.references:
        seed_records = [
            record
            for record in records
            if int(record["seed"]) == reference.seed
            and record.get("eligible_as_smaller_winner", True)
        ]
        best_per_architecture: dict[tuple[int, ...], dict[str, Any]] = {}
        for record in seed_records:
            architecture = tuple(record["architecture"])
            current = best_per_architecture.get(architecture)
            if current is None or _selection_key(record) < _selection_key(current):
                best_per_architecture[architecture] = record
        ranked = sorted(best_per_architecture.values(), key=_stage1_selection_key)
        per_seed[reference.seed] = ranked[: settings.top_k_per_seed]

    selected_payload = {
        "top_k_per_seed": settings.top_k_per_seed,
        "search_mode": settings.search_mode,
        "seeds": {str(seed): rows for seed, rows in per_seed.items()},
        "selection": "validation_accuracy desc, validation_loss asc, parameters asc, architecture lexicographic",
        "methodology_warning": METHODOLOGY_WARNING,
    }
    selected_json = paths.selected / "stage1_top50_per_seed.json"
    _atomic_json(selected_json, selected_payload)
    selected_rows = []
    for seed, rows in per_seed.items():
        for rank, row in enumerate(rows, 1):
            selected_rows.append(
                {
                    "seed": seed,
                    "rank": rank,
                    "architecture": "-".join(map(str, row["architecture"])),
                    "parameters": row["parameters"],
                    "validation_accuracy": row["validation_accuracy"],
                    "validation_loss": row["validation_loss"],
                    "test_accuracy_recorded_not_selected": row["test_accuracy"],
                }
            )
    _atomic_csv(paths.selected / "stage1_top50_per_seed.csv", selected_rows)
    counts = {
        str(reference.seed): len(
            {
                tuple(record["architecture"])
                for record in records
                if int(record["seed"]) == reference.seed
                and record.get("eligible_as_smaller_winner", True)
            }
        )
        for reference in settings.references
    }
    _atomic_json(
        paths.summaries / "stage1_summary.json",
        {
            "architectures_screened_by_seed": counts,
            "total_trials": len(records),
            "total_elapsed_seconds": sum(
                float(record.get("elapsed_seconds", 0.0)) for record in records
            ),
            "methodology_warning": METHODOLOGY_WARNING,
        },
    )

    stage2_entries: list[dict[str, Any]] = []
    if settings.search_mode == "two_stage":
        fine_grid = stage2_overrides(config, paths.repo_root)
        for reference in settings.references:
            architectures = [
                (tuple(row["architecture"]), int(row["parameters"]), True)
                for row in per_seed[reference.seed]
            ]
            if settings.include_reference_control:
                architectures.append(
                    (reference.architecture, reference.parameters, False)
                )
            for architecture, parameters, eligible in architectures:
                for overrides in fine_grid:
                    entry = {
                        "trial_index": len(stage2_entries),
                        "stage": "stage2",
                        "seed": reference.seed,
                        "architecture": list(architecture),
                        "parameters": parameters,
                        "eligible_as_smaller_winner": eligible,
                        "overrides": overrides,
                    }
                    entry["deterministic_trial_id"] = deterministic_trial_id(entry)
                    stage2_entries.append(entry)
    write_jsonl_atomic(_manifest_path(paths, "stage2"), stage2_entries)
    print(f"Saved {selected_json}")
    print(f"Saved {_manifest_path(paths, 'stage2')} ({len(stage2_entries)} trials)")
    return selected_json


def finalize(config_path: Path, paths: RunPaths) -> Path:
    config = load_search_config(config_path)
    settings = load_settings(config)
    stage1_entries = read_jsonl(_manifest_path(paths, "stage1"))
    stage1_records = load_stage_results(paths, "stage1")
    _require_complete_manifest(stage1_entries, stage1_records, "stage1")
    if settings.search_mode == "two_stage":
        candidate_entries = read_jsonl(_manifest_path(paths, "stage2"))
        candidate_records = load_stage_results(paths, "stage2")
        _require_complete_manifest(candidate_entries, candidate_records, "stage2")
    else:
        candidate_records = stage1_records
    stage1_summary = json.loads(
        (paths.summaries / "stage1_summary.json").read_text(encoding="utf-8")
    )

    rows = []
    for reference in settings.references:
        seed_records = [
            record
            for record in candidate_records
            if int(record["seed"]) == reference.seed
        ]
        eligible = [
            record
            for record in seed_records
            if record.get("eligible_as_smaller_winner", True)
            and settings.minimum_parameters
            <= int(record["parameters"])
            < reference.parameters
        ]
        if not eligible:
            raise RuntimeError(
                f"seed {reference.seed} has no eligible completed candidates"
            )
        winner = min(eligible, key=_selection_key)
        controls = [
            record
            for record in seed_records
            if not record.get("eligible_as_smaller_winner", True)
            and tuple(record["architecture"]) == reference.architecture
        ]
        control = min(controls, key=_selection_key) if controls else None
        difference = float(winner["test_accuracy"]) - reference.test_accuracy
        rows.append(
            {
                "seed": reference.seed,
                "growth_architecture": list(reference.architecture),
                "growth_parameters": reference.parameters,
                "growth_test_accuracy": reference.test_accuracy,
                "stage1_architectures_screened": int(
                    stage1_summary["architectures_screened_by_seed"][
                        str(reference.seed)
                    ]
                ),
                "stage1_top_k": settings.top_k_per_seed,
                "best_smaller_architecture": winner["architecture"],
                "best_smaller_parameters": winner["parameters"],
                "selected_overrides": winner["overrides"],
                "best_validation_epoch": winner["best_validation_epoch"],
                "validation_accuracy": winner["validation_accuracy"],
                "validation_loss": winner["validation_loss"],
                "test_accuracy": winner["test_accuracy"],
                "smaller_minus_growth": difference,
                "smaller_has_better_test": difference > 0.0,
                "reference_architecture_retrained_test": (
                    control["test_accuracy"] if control else None
                ),
                "winner_trial_id": winner["trial_id"],
            }
        )

    differences = [row["smaller_minus_growth"] for row in rows]
    fixed_tests = [row["test_accuracy"] for row in rows]
    growth_tests = [row["growth_test_accuracy"] for row in rows]
    difference_std = statistics.stdev(differences) if len(differences) > 1 else 0.0
    stage2_records = load_stage_results(paths, "stage2")
    aggregate = {
        "growth_mean_test_accuracy": statistics.mean(growth_tests),
        "smaller_mean_test_accuracy": statistics.mean(fixed_tests),
        "mean_paired_difference": statistics.mean(differences),
        "std_paired_difference": difference_std,
        "standard_error_paired_difference": difference_std / math.sqrt(len(rows)),
        "smaller_wins": sum(value > 0.0 for value in differences),
        "growth_wins": sum(value < 0.0 for value in differences),
        "ties": sum(value == 0.0 for value in differences),
        "total_architectures_screened": sum(
            int(value)
            for value in stage1_summary["architectures_screened_by_seed"].values()
        ),
        "total_stage1_trials": len(stage1_records),
        "total_stage2_trials": len(stage2_records),
        "total_elapsed_seconds": sum(
            float(record.get("elapsed_seconds", 0.0))
            for record in [*stage1_records, *stage2_records]
        ),
    }
    payload = {
        "protocol": "same_seed_strictly_smaller",
        "selection": "validation only; test read after joint architecture/hyperparameter selection",
        "methodology_warning": METHODOLOGY_WARNING,
        "seeds": rows,
        "aggregate": aggregate,
    }
    final_json = paths.summaries / "final_budget_comparison.json"
    _atomic_json(final_json, payload)
    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat["growth_architecture"] = "-".join(map(str, row["growth_architecture"]))
        flat["best_smaller_architecture"] = "-".join(
            map(str, row["best_smaller_architecture"])
        )
        flat["selected_overrides"] = _canonical(row["selected_overrides"])
        flat_rows.append(flat)
    _atomic_csv(paths.summaries / "final_budget_comparison.csv", flat_rows)
    _atomic_json(
        paths.summaries / "stage2_summary.json", {"seeds": rows, "aggregate": aggregate}
    )
    _atomic_csv(paths.summaries / "stage2_summary.csv", flat_rows)
    print(f"Saved {final_json}")
    return final_json


def materialize_winners(config_path: Path, paths: RunPaths) -> None:
    config = load_search_config(config_path)
    final_path = paths.summaries / "final_budget_comparison.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    output_dir = paths.selected / "materialized_winners"
    for row in final["seeds"]:
        entry = {
            "stage": "materialized",
            "seed": row["seed"],
            "architecture": row["best_smaller_architecture"],
            "parameters": row["best_smaller_parameters"],
            "overrides": row["selected_overrides"],
            "deterministic_trial_id": f"winner_seed{row['seed']}",
        }
        pipeline = _trial_config(config, entry, paths)
        pipeline = replace(
            pipeline,
            run=replace(
                pipeline.run,
                name=f"budget_winner_seed{row['seed']}",
                results_dir=output_dir,
            ),
        )
        result = run_pipeline(pipeline, progress=print)
        written = write_outputs(result)
        print(f"Materialized seed {row['seed']}: {written}")


def incomplete_shards(paths: RunPaths, stage: str, num_shards: int) -> list[int]:
    entries = read_jsonl(_manifest_path(paths, stage))
    incomplete = []
    for shard_index in range(num_shards):
        assigned = manifest_for_shard(entries, shard_index, num_shards)
        records = read_jsonl(_shard_path(paths, stage, shard_index))
        indexed = _records_by_id(records)
        if any(
            entry["deterministic_trial_id"] not in indexed
            or indexed[entry["deterministic_trial_id"]].get("status") != "complete"
            for entry in assigned
        ):
            incomplete.append(shard_index)
    return incomplete


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--min-free-gb", type=float, default=float(os.environ.get("MIN_FREE_GB", "1"))
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "inspect",
        "prepare-stage1",
        "select-stage1",
        "finalize",
        "materialize-winners",
        "incomplete-shards",
    ):
        subparser = subparsers.add_parser(name)
        _common_arguments(subparser)
    run = subparsers.add_parser("run-shard")
    _common_arguments(run)
    run.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--num-shards", type=int, required=True)
    run.add_argument("--retry-failed", action="store_true")
    prepare = subparsers.choices["prepare-stage1"]
    prepare.add_argument("--stage1-num-shards", type=int, default=1000)
    prepare.add_argument("--stage2-num-shards", type=int, default=600)
    prepare.add_argument("--max-concurrent", type=int, default=30)
    incomplete = subparsers.choices["incomplete-shards"]
    incomplete.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    incomplete.add_argument("--num-shards", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    if args.command == "inspect":
        inspect_search(args.config, repo_root)
        return 0
    paths = resolve_run_paths(
        repo_root=repo_root, run_root=args.run_root, min_free_gb=args.min_free_gb
    )
    ensure_run_directories(paths)
    if args.command == "prepare-stage1":
        prepare_stage1(
            args.config,
            paths,
            stage1_num_shards=args.stage1_num_shards,
            stage2_num_shards=args.stage2_num_shards,
            max_concurrent=args.max_concurrent,
        )
    elif args.command == "run-shard":
        run_shard(
            args.config,
            paths,
            stage=args.stage,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            retry_failed=args.retry_failed,
        )
    elif args.command == "select-stage1":
        select_stage1(args.config, paths)
    elif args.command == "finalize":
        finalize(args.config, paths)
    elif args.command == "materialize-winners":
        materialize_winners(args.config, paths)
    elif args.command == "incomplete-shards":
        print(",".join(map(str, incomplete_shards(paths, args.stage, args.num_shards))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
