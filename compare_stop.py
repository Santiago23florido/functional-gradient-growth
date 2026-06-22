#!/usr/bin/env python
"""Is the tolerance eps avoidable? Certificate-stop vs bottleneck-plateau stop.

With the certificate stopping rule (grow until (1+eps)||r|| < eps||g||), eps sets
the final model size: it is a real knob. This script tests a less heuristic
alternative within the same framework -- the *diminishing-returns* (marginal-
utility) stop: grow while each growth keeps reducing the deterministic expressivity
bottleneck and stop at the plateau (the "elbow" of the bottleneck-vs-capacity
curve). With min_gain=0 there is no tolerance to choose; the stopping point is set
by the data+model, not by eps.

For each task we report, in the natural (GGN) metric and over seeds {0,1,2}:
  - certificate stop, swept over eps  -> size/accuracy should vary with eps;
  - plateau stop (eps-free),  swept over eps -> size/accuracy should be ~constant.
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


def base_ggn(eps: float, plateau: bool) -> dict:
    ov = {
        "method": "functional_certified_tiny",
        "functional_certificate_scope": "fulldata",
        "functional_certificate_metric": "ggn",
        "functional_relative_error_tolerance": eps,
        "functional_growth_min_gain": 0.0,
    }
    if plateau:
        # eps-free stop: grow until the deterministic bottleneck plateaus, with an
        # eps-free growth selection (TINY first-order score). eps does not enter.
        ov["functional_stop_rule"] = "plateau"
        ov["functional_growth_layer_selection"] = "tiny_best"
    else:
        ov["functional_stop_rule"] = "certificate"
        ov["functional_growth_layer_selection"] = "certifying"
    return ov


def run_one(base, ov, seed):
    cfg = deepcopy(base)
    cfg.update(ov)
    cfg.pop("methods", None)
    cfg["device"] = "cuda"
    cfg["seed"] = seed
    cfg["run_name"] = f"stop_s{seed}"
    cfg["out_dir"] = "results/_stop"
    b = io.StringIO()
    with redirect_stdout(b):
        r = run_experiment(cfg)
    h = r["history"]
    return (h["test_acc"][-1], h["test_loss"][-1], r["final_params"], len(r["growth_info"]))


def summarize(rs):
    return {
        "acc": (st.mean(x[0] for x in rs), st.pstdev([x[0] for x in rs])),
        "loss": (st.mean(x[1] for x in rs), st.pstdev([x[1] for x in rs])),
        "params": (st.mean(x[2] for x in rs), st.pstdev([x[2] for x in rs])),
        "grow": st.mean(x[3] for x in rs),
    }


def main():
    seeds = [0, 1, 2]
    eps_list = [0.02, 0.05, 0.1]
    tasks = ["blobs", "spiral"]
    out = {}
    for task in tasks:
        base = ablate.BASES[task]
        out[task] = {}
        print(f"\n=== {task} (GGN metric) ===", flush=True)
        # scheduled baseline for reference
        rs = [run_one(base, {"method": "gromo_tiny"}, s) for s in seeds]
        out[task]["scheduled"] = summarize(rs)
        a = out[task]["scheduled"]
        print(f"  scheduled                 params {a['params'][0]:.0f}±{a['params'][1]:.0f}  "
              f"acc {a['acc'][0]:.3f}  loss {a['loss'][0]:.3f}", flush=True)
        for plateau in (False, True):
            tag = "plateau(eps-free)" if plateau else "certificate-stop"
            for eps in eps_list:
                rs = [run_one(base, base_ggn(eps, plateau), s) for s in seeds]
                key = f"{tag} eps={eps}"
                out[task][key] = summarize(rs)
                a = out[task][key]
                print(f"  {tag:18s} eps={eps:<4} params {a['params'][0]:.0f}±{a['params'][1]:.0f}  "
                      f"acc {a['acc'][0]:.3f}±{a['acc'][1]:.3f}  loss {a['loss'][0]:.3f}  "
                      f"grow {a['grow']:.1f}", flush=True)
    Path("results/_stop").mkdir(parents=True, exist_ok=True)
    Path("results/_stop/summary.json").write_text(json.dumps(out, indent=2))
    print("\nsaved results/_stop/summary.json", flush=True)


if __name__ == "__main__":
    main()
