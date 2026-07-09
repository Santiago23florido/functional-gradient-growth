#!/usr/bin/env python
"""How much does eps actually matter? Euclidean vs natural (GGN) certificate.

eps is the tolerance of the relative-error certificate; under the certificate stop
rule it sets the final model size, so it is a real knob. This script quantifies how
*sensitive* the outcome is to eps in each metric: it sweeps eps and reports, per
metric, the spread (max - min over the eps grid) of the final parameter count and
test accuracy. The claim under test is that the natural (GGN) metric -- which
discards the loss-irrelevant logit-shift direction and so produces a sharper
bottleneck that drops to a clear floor -- makes the result far less eps-sensitive.
"""
from __future__ import annotations

import io
import json
import statistics as st
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from stable_tiny.experiment import run_experiment  # noqa: E402

import ablate  # noqa: E402

SEEDS = [0, 1, 2]
EPS = [0.05, 0.1, 0.2, 0.3]
TASKS = ["blobs", "spiral"]


def ov(metric: str, eps: float) -> dict:
    return {
        "method": "functional_certified_tiny",
        "functional_certificate_scope": "fulldata",
        "functional_growth_layer_selection": "certifying",
        "functional_stop_rule": "certificate",
        "functional_certificate_metric": metric,
        "functional_relative_error_tolerance": eps,
    }


def run_one(base, o, seed):
    cfg = deepcopy(base)
    cfg.update(o)
    cfg.pop("methods", None)
    cfg["device"] = "cuda"
    cfg["seed"] = seed
    cfg["run_name"] = f"eps_s{seed}"
    cfg["out_dir"] = "results/_eps"
    b = io.StringIO()
    with redirect_stdout(b):
        r = run_experiment(cfg)
    h = r["history"]
    return h["test_acc"][-1], h["test_loss"][-1], r["final_params"]


def main():
    out = {}
    for task in TASKS:
        base = ablate.BASES[task]
        out[task] = {}
        print(f"\n=== {task} ===", flush=True)
        for metric in ("euclidean", "ggn"):
            params_by_eps, acc_by_eps = [], []
            per_eps = {}
            for eps in EPS:
                rs = [run_one(base, ov(metric, eps), s) for s in SEEDS]
                p = st.mean(x[2] for x in rs)
                a = st.mean(x[0] for x in rs)
                params_by_eps.append(p)
                acc_by_eps.append(a)
                per_eps[eps] = {"params": p, "acc": a,
                                "loss": st.mean(x[1] for x in rs)}
                print(f"  {metric:9s} eps={eps:<4} params {p:.0f}  acc {a:.3f}", flush=True)
            p_spread = max(params_by_eps) - min(params_by_eps)
            a_spread = max(acc_by_eps) - min(acc_by_eps)
            out[task][metric] = {
                "per_eps": per_eps,
                "param_spread": p_spread,
                "param_spread_frac": p_spread / max(1.0, st.mean(params_by_eps)),
                "acc_spread": a_spread,
            }
            print(f"  -> {metric}: param spread over eps = {p_spread:.0f} "
                  f"({100*out[task][metric]['param_spread_frac']:.0f}% of mean), "
                  f"acc spread = {a_spread:.3f}", flush=True)
    Path("results/_eps").mkdir(parents=True, exist_ok=True)
    Path("results/_eps/summary.json").write_text(json.dumps(out, indent=2))
    print("\nsaved results/_eps/summary.json", flush=True)


if __name__ == "__main__":
    main()
