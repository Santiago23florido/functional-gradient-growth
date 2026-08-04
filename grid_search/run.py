"""Run and summarize restartable fixed-architecture grid-search trials.

Examples (from the repository root)::

    PYTHONPATH=src python -m grid_search.run list
    PYTHONPATH=src python -m grid_search.run run --trial-index 0
    PYTHONPATH=src python -m grid_search.run run --shard-index 0 --num-shards 8
    PYTHONPATH=src python -m grid_search.run summarize
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from fgdlib.training_utils.loop import count_parameters
from stable_tiny.pipeline import load_pipeline_config, run_pipeline, write_outputs

DEFAULT_GRID = Path("grid_search/fixed_architectures.yaml")


@dataclass(frozen=True)
class Trial:
    index: int
    trial_id: str
    architecture: tuple[int, ...]
    model_seed: int
    overrides: dict[str, Any]


def _product(space: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = list(space)
    values = [value if isinstance(value, list) else [value] for value in space.values()]
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def load_grid(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {"base_config", "results_dir", "architectures", "model_seeds"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"grid config is missing: {', '.join(missing)}")
    if not payload.get("search_spaces"):
        payload["search_spaces"] = [{}]
    return payload


def enumerate_trials(grid: Mapping[str, Any]) -> list[Trial]:
    combinations: list[dict[str, Any]] = []
    for space in grid["search_spaces"]:
        combinations.extend(_product(space or {}))

    trials: list[Trial] = []
    for architecture, model_seed, overrides in itertools.product(
        grid["architectures"], grid["model_seeds"], combinations
    ):
        architecture = tuple(int(width) for width in architecture)
        if not architecture or any(width < 1 for width in architecture):
            raise ValueError(f"invalid architecture: {architecture}")
        identity = {
            "architecture": architecture,
            "model_seed": int(model_seed),
            "overrides": overrides,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        index = len(trials)
        trials.append(
            Trial(
                index=index,
                trial_id=f"trial_{index:05d}_{digest}",
                architecture=architecture,
                model_seed=int(model_seed),
                overrides=dict(overrides),
            )
        )
    return trials


def _coerce_like(current: Any, value: Any) -> Any:
    if isinstance(current, Path):
        return Path(value)
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def apply_dotted_overrides(config: Any, overrides: Mapping[str, Any]) -> Any:
    """Return a replaced dataclass tree from ``section.field`` overrides."""
    grouped: dict[str, dict[str, Any]] = {}
    for key, value in overrides.items():
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(f"override must be section.field, got {key!r}")
        grouped.setdefault(parts[0], {})[parts[1]] = value

    replacements: dict[str, Any] = {}
    top_fields = {item.name for item in fields(config)}
    for section_name, section_values in grouped.items():
        if section_name not in top_fields:
            raise ValueError(f"unknown config section {section_name!r}")
        section = getattr(config, section_name)
        if not is_dataclass(section):
            raise ValueError(f"config section {section_name!r} is not replaceable")
        valid = {item.name for item in fields(section)}
        unknown = sorted(set(section_values) - valid)
        if unknown:
            raise ValueError(f"unknown {section_name} field(s): {', '.join(unknown)}")
        replacements[section_name] = replace(
            section,
            **{
                key: _coerce_like(getattr(section, key), value)
                for key, value in section_values.items()
            },
        )
    return replace(config, **replacements)


def parameter_count(
    architecture: Sequence[int], inputs: int = 4, outputs: int = 1
) -> int:
    widths = [inputs, *architecture, outputs]
    return sum((left + 1) * right for left, right in itertools.pairwise(widths))


def build_trial_config(grid: Mapping[str, Any], trial: Trial) -> Any:
    config = load_pipeline_config(Path(grid["base_config"]))
    overrides = dict(grid.get("fixed_overrides", {}))
    overrides.update(trial.overrides)
    config = apply_dotted_overrides(config, overrides)
    stack = tuple({"mlp": width} for width in trial.architecture)
    results_dir = Path(grid["results_dir"]) / "histories"
    return replace(
        config,
        model=replace(config.model, stack=stack, model_seed=trial.model_seed),
        training=replace(config.training, method="normal"),
        growth_schedule=replace(config.growth_schedule, enabled=False),
        wandb=replace(config.wandb, enabled=False),
        run=replace(
            config.run,
            name=trial.trial_id,
            results_dir=results_dir,
            save_plot=False,
            show_plot=False,
        ),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _best_entry(history: Sequence[Any]) -> Any:
    return max(
        history,
        key=lambda entry: (
            entry.validation_accuracy,
            -entry.validation_loss,
            -entry.step,
        ),
    )


def run_trial(grid: Mapping[str, Any], trial: Trial, force: bool = False) -> str:
    output = Path(grid["results_dir"]) / "trials" / f"{trial.trial_id}.json"
    if output.exists() and not force:
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"[skip] {trial.trial_id} already complete", flush=True)
            return "skipped"

    config = build_trial_config(grid, trial)
    expected_params = parameter_count(
        trial.architecture, config.data.in_features, config.data.out_features
    )
    started = time.time()
    print(
        f"[run] {trial.index}: arch={'-'.join(map(str, trial.architecture))} "
        f"seed={trial.model_seed} overrides={trial.overrides}",
        flush=True,
    )
    try:
        result = run_pipeline(config=config, progress=None)
        actual_params = count_parameters(result.model)
        if actual_params != expected_params:
            raise RuntimeError(
                f"parameter mismatch: expected {expected_params}, got {actual_params}"
            )
        output_paths = write_outputs(result)
        best = _best_entry(result.history)
        final = result.history[-1]
        payload = {
            "status": "complete",
            "trial": asdict(trial),
            "parameters": actual_params,
            "selection": "maximum validation_accuracy; test is not used for selection",
            "best_validation_epoch": best.step,
            "best": {
                "train_loss": best.train_loss,
                "validation_loss": best.validation_loss,
                "test_loss": best.test_loss,
                "train_accuracy": best.train_accuracy,
                "validation_accuracy": best.validation_accuracy,
                "test_accuracy": best.test_accuracy,
            },
            "final": {
                "epoch": final.step,
                "validation_accuracy": final.validation_accuracy,
                "test_accuracy": final.test_accuracy,
            },
            "history": str(output_paths["history"]),
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(output, payload)
        print(
            f"[done] {trial.trial_id}: val={best.validation_accuracy:.4f} "
            f"test={best.test_accuracy:.4f} epoch={best.step}",
            flush=True,
        )
        return "complete"
    except Exception as error:
        _atomic_json(
            output,
            {
                "status": "failed",
                "trial": asdict(trial),
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.time() - started,
            },
        )
        raise


def _group_key(payload: Mapping[str, Any]) -> str:
    trial = payload["trial"]
    identity = {
        "architecture": trial["architecture"],
        "overrides": trial["overrides"],
    }
    return json.dumps(identity, sort_keys=True)


def _overrides_key(overrides: Mapping[str, Any]) -> str:
    """Canonical representation used only for identity and deterministic ties."""
    return json.dumps(overrides, sort_keys=True, separators=(",", ":"))


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def paired_same_seed_retraining_summary(
    grid: Mapping[str, Any],
    trial_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare growth with fresh AdamW retraining on the corresponding seed.

    For seed ``s``, candidates are restricted simultaneously to the
    architecture produced by growth seed ``s`` and retraining seed ``s``.
    Hyperparameters are selected by validation, then test is read once.
    """
    paired = grid.get("paired_same_seed_evaluation")
    if not isinstance(paired, Mapping):
        raise TypeError("paired_same_seed_evaluation must be a mapping")
    if paired.get("protocol") != "same_seed_retraining":
        raise ValueError(
            "paired_same_seed_evaluation.protocol must be 'same_seed_retraining'"
        )

    model_seeds = tuple(sorted(int(seed) for seed in grid["model_seeds"]))
    if not model_seeds or len(set(model_seeds)) != len(model_seeds):
        raise ValueError("paired evaluation needs unique model_seeds")

    raw_growth_runs = paired.get("growth_runs")
    if not isinstance(raw_growth_runs, Mapping):
        raise TypeError("paired_same_seed_evaluation.growth_runs must be a mapping")
    growth_runs = {int(seed): value for seed, value in raw_growth_runs.items()}
    if set(growth_runs) != set(model_seeds):
        missing = sorted(set(model_seeds) - set(growth_runs))
        extra = sorted(set(growth_runs) - set(model_seeds))
        raise ValueError(
            "paired_same_seed_evaluation.growth_runs must contain exactly "
            f"model_seeds; missing={missing}, extra={extra}"
        )

    grid_architectures = {
        tuple(int(width) for width in architecture)
        for architecture in grid["architectures"]
    }
    references: dict[int, tuple[tuple[int, ...], float]] = {}
    for seed in model_seeds:
        reference = growth_runs[seed]
        if not isinstance(reference, Mapping):
            raise TypeError(f"growth_runs[{seed}] must be a mapping")
        architecture = tuple(int(width) for width in reference["architecture"])
        if architecture not in grid_architectures:
            raise ValueError(
                f"growth_runs[{seed}] architecture {list(architecture)} "
                "is not present in architectures"
            )
        references[seed] = (architecture, float(reference["test_accuracy"]))

    expected_overrides: dict[int, dict[str, dict[str, Any]]] = {
        seed: {} for seed in model_seeds
    }
    for trial in enumerate_trials(grid):
        reference_architecture, _ = references[trial.model_seed]
        if trial.architecture == reference_architecture:
            key = _overrides_key(trial.overrides)
            expected_overrides[trial.model_seed][key] = trial.overrides

    results: dict[tuple[tuple[int, ...], int, str], Mapping[str, Any]] = {}
    for payload in trial_payloads:
        trial = payload["trial"]
        architecture = tuple(int(width) for width in trial["architecture"])
        seed = int(trial["model_seed"])
        overrides_key = _overrides_key(trial["overrides"])
        identity = (architecture, seed, overrides_key)
        if seed not in references:
            continue
        expected_architecture, _ = references[seed]
        if architecture != expected_architecture:
            continue
        if overrides_key not in expected_overrides[seed]:
            continue
        if identity in results:
            raise RuntimeError(
                "duplicate same-seed retraining trial for "
                f"seed={seed}, expected_architecture={list(architecture)}, "
                f"overrides={overrides_key}"
            )
        results[identity] = payload

    incomplete_pairs: list[str] = []
    for seed, (architecture, _) in references.items():
        missing_or_incomplete = []
        complete_count = 0
        for key, overrides in expected_overrides[seed].items():
            payload = results.get((architecture, seed, key))
            status = payload.get("status") if payload is not None else "missing"
            if status == "complete":
                complete_count += 1
            else:
                missing_or_incomplete.append({"overrides": overrides, "status": status})
        if missing_or_incomplete or complete_count == 0:
            incomplete_pairs.append(
                f"seed={seed}, expected_architecture={list(architecture)}, "
                f"complete_candidates={complete_count}/"
                f"{len(expected_overrides[seed])}, "
                f"missing_or_incomplete={missing_or_incomplete}"
            )
    if incomplete_pairs:
        raise RuntimeError(
            "paired same-seed retraining evaluation is incomplete:\n- "
            + "\n- ".join(incomplete_pairs)
        )

    pairs: list[dict[str, Any]] = []
    for seed in model_seeds:
        architecture, growth_test_accuracy = references[seed]
        candidates = [
            results[(architecture, seed, key)] for key in expected_overrides[seed]
        ]
        selected = min(
            candidates,
            key=lambda candidate: (
                -float(candidate["best"]["validation_accuracy"]),
                float(candidate["best"]["validation_loss"]),
                _overrides_key(candidate["trial"]["overrides"]),
            ),
        )
        selected_trial = selected["trial"]
        selected_architecture = tuple(selected_trial["architecture"])
        selected_seed = int(selected_trial["model_seed"])
        if selected_architecture != architecture or selected_seed != seed:
            raise AssertionError(
                "same-seed selection escaped its pair: "
                f"seed={seed}, expected_architecture={list(architecture)}, "
                f"selected_seed={selected_seed}, "
                f"selected_architecture={list(selected_architecture)}"
            )
        fixed_test_accuracy = float(selected["best"]["test_accuracy"])
        pairs.append(
            {
                "seed": seed,
                "architecture": list(architecture),
                "selected_overrides": selected_trial["overrides"],
                "best_validation_epoch": int(selected["best_validation_epoch"]),
                "validation_accuracy": float(selected["best"]["validation_accuracy"]),
                "validation_loss": float(selected["best"]["validation_loss"]),
                "fixed_test_accuracy": fixed_test_accuracy,
                "growth_test_accuracy": growth_test_accuracy,
                "fixed_minus_growth": fixed_test_accuracy - growth_test_accuracy,
                "trial_id": selected_trial["trial_id"],
                "parameters": int(selected["parameters"]),
                "elapsed_seconds": float(selected["elapsed_seconds"]),
            }
        )

    fixed_tests = [pair["fixed_test_accuracy"] for pair in pairs]
    growth_tests = [pair["growth_test_accuracy"] for pair in pairs]
    differences = [pair["fixed_minus_growth"] for pair in pairs]
    difference_std = _sample_std(differences)
    aggregate = {
        "number_of_pairs": len(pairs),
        "fixed_mean_test_accuracy": statistics.mean(fixed_tests),
        "fixed_std_test_accuracy": _sample_std(fixed_tests),
        "growth_mean_test_accuracy": statistics.mean(growth_tests),
        "growth_std_test_accuracy": _sample_std(growth_tests),
        "mean_paired_difference": statistics.mean(differences),
        "std_paired_difference": difference_std,
        "standard_error_paired_difference": difference_std
        / math.sqrt(len(differences)),
        "fixed_wins": sum(difference > 0.0 for difference in differences),
        "growth_wins": sum(difference < 0.0 for difference in differences),
        "ties": sum(difference == 0.0 for difference in differences),
    }
    return {
        "protocol": "paired_same_seed_retraining",
        "selection": (
            "for each growth seed, restrict to its architecture and the same "
            "retraining seed; select by validation accuracy, validation loss, "
            "and canonical overrides; read test only after selection"
        ),
        "pairs": pairs,
        "aggregate": aggregate,
    }


def _print_same_seed_summary(summary: Mapping[str, Any]) -> None:
    print("\nPaired same-seed retraining")
    print(
        "Seed  Architecture  LR        WD        Scheduler          Val     "
        "Fixed test  Grow test  Delta"
    )
    print("-" * 105)
    for pair in summary["pairs"]:
        architecture = "-".join(map(str, pair["architecture"]))
        overrides = pair["selected_overrides"]
        print(
            f"{pair['seed']:<5} {architecture:<13} "
            f"{overrides.get('optimizer.learning_rate', 'n/a')!s:<9} "
            f"{overrides.get('optimizer.weight_decay', 'n/a')!s:<9} "
            f"{overrides.get('lr_scheduler.name', 'n/a')!s:<18} "
            f"{pair['validation_accuracy']:.4f}  "
            f"{pair['fixed_test_accuracy']:.4f}      "
            f"{pair['growth_test_accuracy']:.4f}    "
            f"{pair['fixed_minus_growth']:+.4f}"
        )
    print("-" * 105)
    aggregate = summary["aggregate"]
    print(
        "Fixed mean +/- std: "
        f"{aggregate['fixed_mean_test_accuracy']:.4f} +/- "
        f"{aggregate['fixed_std_test_accuracy']:.4f}"
    )
    print(
        "Grow mean +/- std:  "
        f"{aggregate['growth_mean_test_accuracy']:.4f} +/- "
        f"{aggregate['growth_std_test_accuracy']:.4f}"
    )
    print(
        "Paired difference mean +/- std: "
        f"{aggregate['mean_paired_difference']:+.4f} +/- "
        f"{aggregate['std_paired_difference']:.4f}"
    )
    print(
        "Fixed wins / Grow wins / Ties: "
        f"{aggregate['fixed_wins']} / {aggregate['growth_wins']} / "
        f"{aggregate['ties']}"
    )


def summarize(grid: Mapping[str, Any]) -> Path:
    trial_dir = Path(grid["results_dir"]) / "trials"
    trial_payloads = []
    completed = []
    failed = 0
    for path in sorted(trial_dir.glob("trial_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        trial_payloads.append(payload)
        if payload.get("status") == "complete":
            completed.append(payload)
        elif payload.get("status") == "failed":
            failed += 1
    if not completed:
        raise RuntimeError(f"no completed trials found in {trial_dir}")

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for payload in completed:
        groups.setdefault(_group_key(payload), []).append(payload)
    rankings = []
    for key, members in groups.items():
        identity = json.loads(key)
        validation = [member["best"]["validation_accuracy"] for member in members]
        test = [member["best"]["test_accuracy"] for member in members]
        rankings.append(
            {
                **identity,
                "seeds_completed": sorted(
                    member["trial"]["model_seed"] for member in members
                ),
                "runs": len(members),
                "mean_validation_accuracy": statistics.mean(validation),
                "std_validation_accuracy": statistics.stdev(validation)
                if len(validation) > 1
                else 0.0,
                "mean_test_accuracy": statistics.mean(test),
                "std_test_accuracy": statistics.stdev(test) if len(test) > 1 else 0.0,
            }
        )
    rankings.sort(key=lambda item: item["mean_validation_accuracy"], reverse=True)
    summary = {
        "selection": "groups ranked by mean validation accuracy across completed model seeds",
        "headline_reference_mean_test_accuracy": 0.9435,
        "completed_trials": len(completed),
        "failed_trials": failed,
        "expected_trials": len(enumerate_trials(grid)),
        "rankings": rankings,
    }
    output = Path(grid["results_dir"]) / "summary.json"
    _atomic_json(output, summary)
    best = rankings[0]
    print(
        f"Best by mean validation: arch={'-'.join(map(str, best['architecture']))} "
        f"val={best['mean_validation_accuracy']:.4f} "
        f"test={best['mean_test_accuracy']:.4f} over {best['runs']} run(s)"
    )
    print(f"Saved {output}")
    if grid.get("paired_same_seed_evaluation") is not None:
        paired_summary = paired_same_seed_retraining_summary(grid, trial_payloads)
        paired_output = (
            Path(grid["results_dir"]) / "paired_same_seed_retraining_summary.json"
        )
        _atomic_json(paired_output, paired_summary)
        _print_same_seed_summary(paired_summary)
        print(f"Saved {paired_output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "run", "summarize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--trial-index", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    grid = load_grid(args.config)
    trials = enumerate_trials(grid)
    if args.command == "list":
        print(f"{len(trials)} trials (indices 0..{len(trials) - 1})")
        for trial in trials[:10]:
            print(trial)
        if len(trials) > 10:
            print(f"... {len(trials) - 10} more")
        return 0
    if args.command == "summarize":
        summarize(grid)
        return 0

    if args.trial_index is not None:
        if not 0 <= args.trial_index < len(trials):
            raise SystemExit(f"--trial-index must be in 0..{len(trials) - 1}")
        selected = [trials[args.trial_index]]
    elif args.shard_index is not None or args.num_shards is not None:
        if args.shard_index is None or args.num_shards is None:
            raise SystemExit("--shard-index and --num-shards must be used together")
        if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
            raise SystemExit("invalid shard index/count")
        selected = [
            trial
            for trial in trials
            if trial.index % args.num_shards == args.shard_index
        ]
    else:
        selected = trials

    print(f"Selected {len(selected)} of {len(trials)} trials")
    for trial in selected:
        run_trial(grid, trial, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
