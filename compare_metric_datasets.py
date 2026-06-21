#!/usr/bin/env python
"""Cross-dataset head-to-head of the Euclidean vs GGN certificate metric.

Reuses the ablation base configs (blobs / spiral / teacher) and compares
scheduled TINY against certificate-driven growth in the Euclidean and the
loss-induced GGN/Fisher metric, mean+/-std over seeds.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics as st
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from stable_tiny.experiment import run_experiment  # noqa: E402

import ablate  # noqa: E402  (provides BASES)


def variant(metric: str | None, eps: float):
    if metric is None:
        return {"method": "gromo_tiny"}
    return {
        "method": "functional_certified_tiny",
        "functional_certificate_scope": "fulldata",
        "functional_growth_layer_selection": "certifying",
        "functional_certificate_metric": metric,
        "functional_relative_error_tolerance": eps,
    }


def run_one(base, ov, seed):
    cfg = deepcopy(base)
    cfg.update(ov)
    cfg.pop("methods", None)
    cfg["device"] = "cuda"
    cfg["seed"] = seed
    cfg["run_name"] = f"cmd_s{seed}"
    cfg["out_dir"] = "results/_metric_ds"
    b = io.StringIO()
    with redirect_stdout(b):
        r = run_experiment(cfg)
    h = r["history"]
    return (h["test_acc"][-1], h["test_loss"][-1], r["final_params"], len(r["growth_info"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tasks", nargs="+", default=["blobs", "spiral", "teacher"])
    ap.add_argument("--out", default="results/_metric_ds/summary.json")
    args = ap.parse_args()

    variants = [
        ("scheduled", variant(None, 0.0)),
        ("euclidean eps=0.1", variant("euclidean", 0.1)),
        ("ggn eps=0.05", variant("ggn", 0.05)),
        ("ggn eps=0.02", variant("ggn", 0.02)),
    ]
    out = {}
    for task in args.tasks:
        base = ablate.BASES[task]
        out[task] = {}
        print(f"\n=== {task} ===", flush=True)
        for label, ov in variants:
            rs = [run_one(base, ov, s) for s in args.seeds]
            acc = (st.mean(x[0] for x in rs), st.pstdev([x[0] for x in rs]))
            lo = (st.mean(x[1] for x in rs), st.pstdev([x[1] for x in rs]))
            pa = st.mean(x[2] for x in rs)
            gr = st.mean(x[3] for x in rs)
            out[task][label] = {"acc": acc, "loss": lo, "params": pa, "grow": gr}
            print(
                f"  {label:20s} acc {acc[0]:.3f}±{acc[1]:.3f}  "
                f"loss {lo[0]:.3f}±{lo[1]:.3f}  params {pa:.0f}  grow {gr:.1f}",
                flush=True,
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
