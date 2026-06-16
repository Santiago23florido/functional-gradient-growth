#!/usr/bin/env python
"""Run the configured methods and render the function-space loss landscape.

Each method is trained with function-space trajectory recording enabled, then all
trajectories are projected into a common 2D plane (PCA over probe logits) and the
exact functional loss is drawn as a topographic map with the descent paths
animated over it. See ``stable_tiny.landscape``.

Usage
-----
    python make_landscape.py                         # configs/landscape.yaml
    python make_landscape.py --config configs/landscape.yaml
    python make_landscape.py --static                # final-frame PNG only (fast)
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from stable_tiny.experiment import run_experiment  # noqa: E402
from stable_tiny.landscape import render_landscape  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "landscape.yaml"))
    parser.add_argument("--static", action="store_true", help="render only the final-frame PNG")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["record_trajectory"] = True

    out_dir = Path(cfg.get("out_dir", "results"))
    base_run_name = cfg.get("run_name", "run")
    methods = cfg.get("methods") or [cfg.get("method", "gromo_tiny")]

    trajectory_paths = []
    for method in methods:
        method_cfg = deepcopy(cfg)
        method_cfg.pop("methods", None)
        method_cfg["method"] = method
        method_cfg["run_name"] = f"{base_run_name}_{method}"
        result = run_experiment(method_cfg)
        trajectory_paths.append(result["trajectory_path"])

    out_path = render_landscape(
        trajectory_paths,
        out_dir / f"{base_run_name}_landscape.gif",
        static_only=args.static,
    )
    print(f"Saved landscape to {out_path}")


if __name__ == "__main__":
    main()
