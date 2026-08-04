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
