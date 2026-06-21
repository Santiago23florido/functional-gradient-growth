#!/usr/bin/env python
"""Head-to-head: scheduled TINY vs certified growth in the Euclidean vs GGN metric.

Tests the theoretical claim that measuring the expressivity-bottleneck certificate
in the loss-induced (GGN/Fisher) metric -- the natural Hilbert structure for
cross-entropy -- makes certificate-driven growth target loss-relevant capacity and
improves the accuracy/parameter Pareto frontier over the Euclidean certificate.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))
from stable_tiny.experiment import run_experiment  # noqa: E402

ROOT = Path(__file__).parent


def _certified(metric: str, eps: float = 0.1) -> dict:
    return {
        "method": "functional_certified_tiny",
        "functional_certificate_scope": "fulldata",
        "functional_growth_layer_selection": "certifying",
        "functional_certificate_metric": metric,
        "functional_relative_error_tolerance": eps,
    }


VARIANTS = {
    "scheduled": {"method": "gromo_tiny"},
    "certified-euclidean": _certified("euclidean"),
    "certified-ggn": _certified("ggn"),
}


def run_one(base: dict, overrides: dict, seed: int) -> dict:
    cfg = deepcopy(base)
    cfg.update(overrides)
    cfg.pop("methods", None)
    cfg["device"] = "cuda"
    cfg["seed"] = seed
    cfg["run_name"] = f"cm_s{seed}"
    cfg["out_dir"] = "results/_metric"
    buf = io.StringIO()
    with redirect_stdout(buf):
        r = run_experiment(cfg)
    h = r["history"]
    return {
        "final_test_acc": h["test_acc"][-1],
        "final_test_loss": h["test_loss"][-1],
        "final_params": r["final_params"],
        "growth_events": len(r["growth_info"]),
    }


def agg(runs: list[dict]) -> dict:
    out = {}
    for k in runs[0]:
        vals = [r[k] for r in runs]
        out[k] = (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/compare_blobs_stable.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--out", default="results/_metric/summary.json")
    args = ap.parse_args()

    base = yaml.safe_load(open(ROOT / args.config))
    variants = dict(VARIANTS)
    variants["certified-euclidean"]["functional_relative_error_tolerance"] = args.eps
    variants["certified-ggn"]["functional_relative_error_tolerance"] = args.eps

    rows = {}
    for label, ov in variants.items():
        runs = [run_one(base, ov, s) for s in args.seeds]
        a = agg(runs)
        rows[label] = a
        print(
            f"{label:22s} acc {a['final_test_acc'][0]:.3f}±{a['final_test_acc'][1]:.3f}  "
            f"loss {a['final_test_loss'][0]:.3f}±{a['final_test_loss'][1]:.3f}  "
            f"params {a['final_params'][0]:.0f}±{a['final_params'][1]:.0f}  "
            f"grow {a['growth_events'][0]:.1f}",
            flush=True,
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
