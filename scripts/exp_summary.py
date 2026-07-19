"""Tabulate experiment histories: params, best test accuracy, accepted steps.

Usage:
    python scripts/exp_summary.py [glob ...]

Each positional argument is a glob over results/*.json history files
(default: results/exp_*_history.json plus the reference baselines).
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter

DEFAULT_GLOBS = (
    "results/exp_*_history.json",
    "results/mnist_normal_adamw_3x18_15k_history.json",
    "results/mnist_fgd_all_families_history.json",
)


def summarize(path: str) -> dict | None:
    with open(path) as handle:
        data = json.load(handle)
    history = data["history"] if isinstance(data, dict) else data
    if not history:
        return None
    config = data.get("config", {}) if isinstance(data, dict) else {}
    best = max(history, key=lambda e: e.get("test_accuracy") or 0.0)
    accepted = Counter(
        entry.get("fgd_approximation_kind") or entry.get("step_type")
        for entry in history
        if entry.get("fgd_candidate_accepted")
    )
    growths = sum(1 for entry in history if entry.get("step_type") == "GRO")
    return {
        "name": config.get("run", {}).get("name") or path,
        "epochs": config.get("training", {}).get("epochs"),
        "final_params": history[-1].get("num_params"),
        "best_test": best.get("test_accuracy"),
        "best_epoch": best.get("step"),
        "params_at_best": best.get("num_params"),
        "growths": growths,
        "accepted": dict(accepted),
    }


def main() -> None:
    patterns = sys.argv[1:] or list(DEFAULT_GLOBS)
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)))
    rows = [row for row in (summarize(path) for path in paths) if row]
    rows.sort(key=lambda row: (-(row["best_test"] or 0.0)))
    header = (
        f"{'run':42s} {'best_test':>9s} {'@epoch':>6s} {'params@best':>11s} "
        f"{'final_params':>12s} {'grow':>4s}  accepted_steps"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name'][:42]:42s} "
            f"{row['best_test']:9.3f} "
            f"{row['best_epoch']:6d} "
            f"{row['params_at_best']:11d} "
            f"{row['final_params']:12d} "
            f"{row['growths']:4d}  "
            f"{row['accepted']}"
        )


if __name__ == "__main__":
    main()
