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


def paired_leave_one_seed_out_summary(
    grid: Mapping[str, Any],
    completed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the paired fixed-vs-growth comparison without seed selection.

    For fold ``s``, hyperparameters for growth architecture ``A_s`` are ranked
    on every configured model seed except ``s``.  Only after that choice is
    fixed do we read the held-out run's test accuracy.
    """
    paired = grid.get("paired_evaluation")
    if not isinstance(paired, Mapping):
        raise TypeError("paired_evaluation must be a mapping")
    if paired.get("protocol") != "leave_one_seed_out":
        raise ValueError("paired_evaluation.protocol must be 'leave_one_seed_out'")

    model_seeds = tuple(sorted(int(seed) for seed in grid["model_seeds"]))
    if len(model_seeds) < 2 or len(set(model_seeds)) != len(model_seeds):
        raise ValueError("paired evaluation needs at least two unique model_seeds")

    raw_growth_runs = paired.get("growth_runs")
    if not isinstance(raw_growth_runs, Mapping):
        raise TypeError("paired_evaluation.growth_runs must be a mapping")
    growth_runs = {int(seed): value for seed, value in raw_growth_runs.items()}
    if set(growth_runs) != set(model_seeds):
        missing = sorted(set(model_seeds) - set(growth_runs))
        extra = sorted(set(growth_runs) - set(model_seeds))
        raise ValueError(
            "paired_evaluation.growth_runs must contain exactly model_seeds; "
            f"missing={missing}, extra={extra}"
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

    expected_overrides: dict[tuple[int, ...], dict[str, dict[str, Any]]] = {
        architecture: {} for architecture in grid_architectures
    }
    for trial in enumerate_trials(grid):
        key = _overrides_key(trial.overrides)
        expected_overrides[trial.architecture][key] = trial.overrides

    results: dict[tuple[tuple[int, ...], str, int], Mapping[str, Any]] = {}
    for payload in completed:
        trial = payload["trial"]
        architecture = tuple(int(width) for width in trial["architecture"])
        seed = int(trial["model_seed"])
        overrides_key = _overrides_key(trial["overrides"])
        identity = (architecture, overrides_key, seed)
        if (
            architecture not in expected_overrides
            or overrides_key not in expected_overrides[architecture]
            or seed not in model_seeds
        ):
            continue
        if identity in results:
            raise RuntimeError(
                "duplicate completed paired trial for "
                f"architecture={list(architecture)}, seed={seed}, "
                f"overrides={overrides_key}"
            )
        results[identity] = payload

    incomplete_folds: list[str] = []
    for held_out_seed, (architecture, _) in references.items():
        incomplete_configurations = []
        for key, overrides in expected_overrides[architecture].items():
            missing_seeds = [
                seed for seed in model_seeds if (architecture, key, seed) not in results
            ]
            if missing_seeds:
                incomplete_configurations.append(
                    {
                        "overrides": overrides,
                        "missing_seeds": missing_seeds,
                    }
                )
        if incomplete_configurations:
            incomplete_folds.append(
                f"architecture={list(architecture)}, "
                f"held_out_seed={held_out_seed}, "
                f"incomplete_configurations={incomplete_configurations}"
            )
    if incomplete_folds:
        raise RuntimeError(
            "paired leave-one-seed-out evaluation is incomplete:\n- "
            + "\n- ".join(incomplete_folds)
        )

    folds: list[dict[str, Any]] = []
    for held_out_seed in model_seeds:
        architecture, growth_test_accuracy = references[held_out_seed]
        selection_seeds = [seed for seed in model_seeds if seed != held_out_seed]
        candidates = []
        for key, overrides in expected_overrides[architecture].items():
            selection_results = [
                results[(architecture, key, seed)] for seed in selection_seeds
            ]
            validation_accuracies = [
                float(result["best"]["validation_accuracy"])
                for result in selection_results
            ]
            validation_losses = [
                float(result["best"]["validation_loss"]) for result in selection_results
            ]
            candidates.append(
                {
                    "overrides_key": key,
                    "overrides": overrides,
                    "mean_validation_accuracy": statistics.mean(validation_accuracies),
                    "std_validation_accuracy": statistics.stdev(validation_accuracies),
                    "mean_validation_loss": statistics.mean(validation_losses),
                }
            )
        selected = min(
            candidates,
            key=lambda candidate: (
                -candidate["mean_validation_accuracy"],
                candidate["mean_validation_loss"],
                candidate["overrides_key"],
            ),
        )
        held_out = results[(architecture, selected["overrides_key"], held_out_seed)]
        fixed_test_accuracy = float(held_out["best"]["test_accuracy"])
        folds.append(
            {
                "held_out_seed": held_out_seed,
                "architecture": list(architecture),
                "selection_seeds": selection_seeds,
                "selected_overrides": selected["overrides"],
                "selection_mean_validation_accuracy": selected[
                    "mean_validation_accuracy"
                ],
                "selection_std_validation_accuracy": selected[
                    "std_validation_accuracy"
                ],
                "selection_mean_validation_loss": selected["mean_validation_loss"],
                "held_out_best_epoch": int(held_out["best_validation_epoch"]),
                "held_out_validation_accuracy": float(
                    held_out["best"]["validation_accuracy"]
                ),
                "fixed_test_accuracy": fixed_test_accuracy,
                "growth_test_accuracy": growth_test_accuracy,
                "fixed_minus_growth": (fixed_test_accuracy - growth_test_accuracy),
            }
        )

    fixed_tests = [fold["fixed_test_accuracy"] for fold in folds]
    growth_tests = [fold["growth_test_accuracy"] for fold in folds]
    differences = [fold["fixed_minus_growth"] for fold in folds]
    difference_std = statistics.stdev(differences)
    aggregate = {
        "number_of_folds": len(folds),
        "fixed_mean_test_accuracy": statistics.mean(fixed_tests),
        "fixed_std_test_accuracy": statistics.stdev(fixed_tests),
        "growth_mean_test_accuracy": statistics.mean(growth_tests),
        "growth_std_test_accuracy": statistics.stdev(growth_tests),
        "mean_paired_difference": statistics.mean(differences),
        "std_paired_difference": difference_std,
        "standard_error_paired_difference": difference_std
        / math.sqrt(len(differences)),
        "fixed_wins": sum(difference > 0.0 for difference in differences),
        "growth_wins": sum(difference < 0.0 for difference in differences),
        "ties": sum(difference == 0.0 for difference in differences),
    }
    return {
        "protocol": "paired_leave_one_seed_out",
        "selection": (
            "hyperparameters selected by mean validation accuracy on all "
            "non-held-out seeds; validation loss and canonical overrides "
            "break ties; test is read only after selection"
        ),
        "folds": folds,
        "aggregate": aggregate,
    }


def _print_paired_summary(summary: Mapping[str, Any]) -> None:
    print("\nPaired leave-one-seed-out")
    print("seed | architecture | selection seeds | fixed test | grow test | delta")
    for fold in summary["folds"]:
        architecture = "-".join(map(str, fold["architecture"]))
        selection_seeds = ",".join(map(str, fold["selection_seeds"]))
        print(
            f"{fold['held_out_seed']:>4} | {architecture:<12} | "
            f"{selection_seeds:<15} | {fold['fixed_test_accuracy']:.4f}     | "
            f"{fold['growth_test_accuracy']:.4f}    | "
            f"{fold['fixed_minus_growth']:+.4f}"
        )
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
        "Paired delta mean +/- std: "
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
    completed = []
    failed = 0
    for path in sorted(trial_dir.glob("trial_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
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
    if grid.get("paired_evaluation") is not None:
        paired_summary = paired_leave_one_seed_out_summary(grid, completed)
        paired_output = (
            Path(grid["results_dir"]) / "paired_leave_one_seed_out_summary.json"
        )
        _atomic_json(paired_output, paired_summary)
        _print_paired_summary(paired_summary)
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
