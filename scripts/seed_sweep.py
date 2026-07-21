"""Run a config across seeds and report mean +- std of test accuracy.

Motivated by a measurement, not by caution: the SENN and uniform *where*
criteria converged to the SAME architecture (784->8->8->10, 6552 params) and
still differed by 1.55 points of test accuracy (88.70 % against 90.25 %).
At that spread a single run cannot distinguish a 0.1-point margin over the
dense baseline from noise, so every architecture-search claim in this repo
needs a seed distribution behind it.

Usage:
    PYTHONPATH=src python scripts/seed_sweep.py configs/fgd/<config>.yaml 0 1 2
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import replace
from pathlib import Path

from stable_tiny.pipeline import load_pipeline_config, run_pipeline


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    config_path = argv[1]
    seeds = [int(value) for value in argv[2:]]

    base = load_pipeline_config(config_path)
    accuracies: list[float] = []
    parameters: list[int] = []

    for seed in seeds:
        config = replace(
            base,
            model=replace(base.model, model_seed=seed),
            data=replace(base.data, train_seed=seed),
            run=replace(
                base.run,
                name=f"{base.run.name}_seed{seed}",
                results_dir=Path(base.run.results_dir),
            ),
            wandb=replace(base.wandb, enabled=False),
        )
        result = run_pipeline(config=config, progress=None)
        best = max(
            (entry for entry in result.history if entry.test_accuracy is not None),
            key=lambda entry: entry.test_accuracy,
        )
        accuracies.append(best.test_accuracy)
        parameters.append(best.num_params)
        print(
            f"  seed {seed}: test {best.test_accuracy:.4f} "
            f"@ {best.num_params} params",
            flush=True,
        )

    spread = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    print(
        f"\n{config_path}: {statistics.mean(accuracies):.4f} +- {spread:.4f} "
        f"over {len(seeds)} seeds; params "
        f"{statistics.mean(parameters):.0f} "
        f"(min {min(parameters)}, max {max(parameters)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
