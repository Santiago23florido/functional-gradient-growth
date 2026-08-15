"""Plot the full fixed-architecture accuracy landscape against train-and-grow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from grid_search.budget_search import (
    DEFAULT_CONFIG,
    GrowthReference,
    load_search_config,
    load_settings,
)
from grid_search.run import parameter_count


def load_fixed_grid_references(path: Path) -> list[GrowthReference]:
    """Load paired family-ladder references from a fixed-grid config."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    paired = payload.get("paired_same_seed_evaluation")
    if not isinstance(paired, Mapping):
        raise TypeError(f"{path} has no paired_same_seed_evaluation mapping")
    growth_runs = paired.get("growth_runs")
    if not isinstance(growth_runs, Mapping) or not growth_runs:
        raise ValueError(f"{path} has no paired growth_runs mapping")

    references: list[GrowthReference] = []
    for raw_seed, run in sorted(growth_runs.items(), key=lambda item: int(item[0])):
        if not isinstance(run, Mapping):
            raise TypeError(f"growth_runs[{raw_seed}] must be a mapping")
        seed = int(raw_seed)
        architecture = tuple(int(width) for width in run["architecture"])
        parameters = parameter_count(architecture)
        expected = run.get("expected_parameters")
        if expected is not None and int(expected) != parameters:
            raise ValueError(
                f"growth_runs[{seed}] expected {expected} parameters, "
                f"but architecture {architecture} has {parameters}"
            )
        references.append(
            GrowthReference(
                seed=seed,
                architecture=architecture,
                parameters=parameters,
                test_accuracy=float(run["test_accuracy"]),
            )
        )
    return references


def load_completed_records(run_root: Path, stage: str) -> list[dict[str, Any]]:
    """Load unique, completed, eligible rows from all shards in one stage."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    shard_dir = run_root / "shards" / stage
    for path in sorted(shard_dir.glob("shard_*.jsonl")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            trial_id = str(record["trial_id"])
            if trial_id in seen:
                raise RuntimeError(
                    f"duplicate trial_id {trial_id} in {path}:{line_number}"
                )
            seen.add(trial_id)
            if record.get("status") != "complete":
                continue
            if not record.get("eligible_as_smaller_winner", True):
                continue
            records.append(record)
    if not records:
        raise RuntimeError(f"no completed eligible records found under {shard_dir}")
    return records


def accuracy_summary(
    records: Sequence[Mapping[str, Any]], references: Sequence[Any]
) -> list[dict[str, Any]]:
    """Summarize where each growth score lies in its empirical distribution."""
    rows: list[dict[str, Any]] = []
    for reference in references:
        values = sorted(
            float(record["test_accuracy"])
            for record in records
            if int(record["seed"]) == reference.seed
        )
        if not values:
            raise RuntimeError(f"no completed records for seed {reference.seed}")
        below_or_equal = sum(value <= reference.test_accuracy for value in values)
        better = sum(value > reference.test_accuracy for value in values)
        rows.append(
            {
                "seed": reference.seed,
                "growth_architecture": list(reference.architecture),
                "growth_parameters": reference.parameters,
                "growth_test_accuracy": reference.test_accuracy,
                "completed_candidates": len(values),
                "candidate_mean": sum(values) / len(values),
                "candidate_median": _quantile(values, 0.5),
                "candidate_q1": _quantile(values, 0.25),
                "candidate_q3": _quantile(values, 0.75),
                "growth_empirical_percentile": 100.0 * below_or_equal / len(values),
                "candidates_better_than_growth": better,
                "fraction_better_than_growth": better / len(values),
            }
        )
    return rows


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def write_summary(path: Path, rows: Sequence[Mapping[str, Any]], stage: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "metric": "test_accuracy",
        "interpretation": (
            "Descriptive only: test accuracy was not used by the search ranking. "
            "Stage1 gives one common-recipe observation per architecture."
        ),
        "seeds": rows,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def plot_landscape(
    records: Sequence[Mapping[str, Any]], references: Sequence[Any], output_dir: Path
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    distributions = [
        [
            float(record["test_accuracy"])
            for record in records
            if int(record["seed"]) == reference.seed
        ]
        for reference in references
    ]
    positions = list(range(1, len(references) + 1))

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    violins = ax.violinplot(
        distributions,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.82,
    )
    for body in violins["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_edgecolor("#315778")
        body.set_alpha(0.35)
    ax.boxplot(
        distributions,
        positions=positions,
        widths=0.28,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "#315778"},
        medianprops={"color": "#1F2937", "linewidth": 1.6},
        whiskerprops={"color": "#315778"},
        capprops={"color": "#315778"},
    )
    growth_scores = [reference.test_accuracy for reference in references]
    ax.scatter(
        positions,
        growth_scores,
        marker="*",
        s=190,
        color="#D62728",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="Family-ladder reference",
    )
    for position, score in zip(positions, growth_scores):
        ax.annotate(
            f"{score:.3f}",
            (position, score),
            xytext=(7, 2),
            textcoords="offset points",
            fontsize=9,
            color="#9B1C1C",
        )
    ax.set_xticks(positions, [f"Seed {reference.seed}" for reference in references])
    ax.set_ylabel("Test accuracy")
    ax.set_title("Family-ladder reference within screened fixed-MLP accuracy")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower right")
    fig.tight_layout()
    distribution_path = output_dir / "test_accuracy_distribution.png"
    fig.savefig(distribution_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=True)
    scatter = None
    for ax, reference in zip(axes.ravel(), references):
        seed_records = [
            record for record in records if int(record["seed"]) == reference.seed
        ]
        scatter = ax.hexbin(
            [int(record["parameters"]) for record in seed_records],
            [float(record["test_accuracy"]) for record in seed_records],
            gridsize=(32, 28),
            mincnt=1,
            cmap="Blues",
            bins="log",
        )
        ax.scatter(
            [reference.parameters],
            [reference.test_accuracy],
            marker="*",
            s=210,
            color="#D62728",
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
            label="Family-ladder reference",
        )
        ax.set_title(
            f"Seed {reference.seed}: {reference.architecture[0]}-"
            f"{reference.architecture[1]}-{reference.architecture[2]}"
        )
        ax.set_xlabel("Trainable parameters")
        ax.set_ylabel("Test accuracy")
        ax.grid(alpha=0.15)
        ax.legend(loc="lower right", fontsize=8)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.82)
        colorbar.set_label("Candidate density (log count)")
    fig.suptitle("Accuracy landscape of all screened smaller architectures", y=0.99)
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.08, top=0.92, hspace=0.28)
    map_path = output_dir / "parameter_accuracy_landscape.png"
    fig.savefig(map_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return distribution_path, map_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--reference-config",
        type=Path,
        help=(
            "optional fixed-grid YAML whose paired_same_seed_evaluation "
            "replaces the original growth reference markers"
        ),
    )
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage1")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_dir = args.output_dir or run_root / "summaries" / "figures"
    if args.reference_config is None:
        references = load_settings(load_search_config(args.config)).references
    else:
        references = load_fixed_grid_references(args.reference_config)
    records = load_completed_records(run_root, args.stage)
    rows = accuracy_summary(records, references)
    summary_path = output_dir / "accuracy_position_summary.json"
    write_summary(summary_path, rows, args.stage)
    distribution_path, map_path = plot_landscape(records, references, output_dir)
    for row in rows:
        print(
            f"seed={row['seed']} n={row['completed_candidates']} "
            f"growth={row['growth_test_accuracy']:.4f} "
            f"percentile={row['growth_empirical_percentile']:.2f} "
            f"fixed_better={row['candidates_better_than_growth']} "
            f"({100 * row['fraction_better_than_growth']:.2f}%)"
        )
    print(f"Saved {distribution_path}")
    print(f"Saved {map_path}")
    print(f"Saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
