#!/usr/bin/env python
"""Entry point: run a growth experiment and plot the resulting curves.

Usage
-----
    python run.py                                  # uses configs/default.yaml
    python run.py --config configs/default.yaml
    python run.py --set task=mnist --set growth_steps=6   # override config keys

The script loads a YAML config, runs the train+grow loop (see
``stable_tiny.experiment``), saves the metric history as JSON, prints a
per-growth spike report, and writes a PNG with the loss/accuracy/params curves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from copy import deepcopy

import yaml

# Make ``src`` importable without installing the package.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stable_tiny.experiment import run_experiment  # noqa: E402
from stable_tiny.plotting import plot_comparison, plot_history, report_spikes  # noqa: E402


def _coerce(value: str):
    """Parse a CLI override string into bool/int/float/None/str."""
    low = value.lower()
    if low in ("null", "none"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(Path(__file__).parent / "configs" / "default.yaml")
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        help="Override a config key (repeatable).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for override in args.set:
        key, _, value = override.partition("=")
        cfg[key.strip()] = _coerce(value.strip())

    out_dir = Path(cfg.get("out_dir", "results"))
    methods = cfg.get("methods")
    if methods:
        comparison = []
        results = []
        base_run_name = cfg.get("run_name", "run")
        for method in methods:
            method_cfg = deepcopy(cfg)
            method_cfg.pop("methods", None)
            method_cfg["method"] = method
            method_cfg["run_name"] = f"{base_run_name}_{method}"
            result = run_experiment(method_cfg)
            results.append(result)
            report_spikes(result)
            h = result["history"]
            comparison.append(
                {
                    "method": method,
                    "final_train_loss": h["train_loss"][-1],
                    "final_train_acc": h["train_acc"][-1],
                    "final_test_loss": h["test_loss"][-1],
                    "final_test_acc": h["test_acc"][-1],
                    "final_params": result["final_params"],
                    "growth_events": len(result["growth_info"]),
                    "elapsed_sec": result["elapsed_sec"],
                }
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / f"{base_run_name}_comparison.json"
        with open(summary_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Saved comparison summary to {summary_path}")
        png_path = plot_comparison(
            results, out_dir / f"{base_run_name}_comparison_curves.png"
        )
        print(f"Saved comparison curves to {png_path}")
        return

    result = run_experiment(cfg)
    report_spikes(result)

    png_path = plot_history(result, out_dir / f"{cfg.get('run_name', 'run')}_curves.png")
    print(f"Saved curves to {png_path}")


if __name__ == "__main__":
    main()
