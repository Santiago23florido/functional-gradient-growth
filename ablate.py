#!/usr/bin/env python
"""Ablation driver for certified functional growth.

Runs a grid of (dataset x method x certificate-scope x tolerance x seed)
configurations, aggregates the final metrics as mean +/- std over seeds, and
writes both a machine-readable JSON and a LaTeX ``tabular`` fragment that the
report includes directly.

The grid is defined in ``ABLATIONS`` below; each entry is a base config plus a
list of variants. Keeping it in code (not YAML) makes the matrix easy to read and
to extend. Runs are quiet by default (per-epoch logs suppressed).

Usage
-----
    python ablate.py                       # run everything, write results/ablation/*
    python ablate.py --only scope          # run a single named block
    python ablate.py --seeds 0 1 2         # override seed list
    python ablate.py --dry-run             # print the plan without running
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
OUT = ROOT / "results" / "ablation"


# --------------------------------------------------------------------------- #
# Base configs per task (kept minimal; variants below override the knobs)
# --------------------------------------------------------------------------- #
def _load(name: str) -> dict:
    with open(ROOT / "configs" / name) as f:
        return yaml.safe_load(f)


BASES = {
    "blobs": _load("compare_blobs_stable.yaml"),
}


def _spiral_base() -> dict:
    cfg = deepcopy(BASES["blobs"])
    cfg.update(
        task="spiral",
        out_features=3,
        points_per_class=1500,
        noise=0.12,
        revolutions=0.8,
        in_features=2,
        hidden_size=4,
        number_hidden_layers=3,
        maximum_added_neurons=8,
        functional_max_epochs=40,
        epochs_per_step=5,
        final_epochs=8,
    )
    for k in ("cluster_std", "center_scale", "n_train", "n_test"):
        cfg.pop(k, None)
    return cfg


def _teacher_base() -> dict:
    cfg = deepcopy(BASES["blobs"])
    cfg.update(
        task="teacher",
        in_features=20,
        out_features=10,
        teacher_hidden=128,
        n_train=20000,
        n_test=4000,
        label_noise=0.0,
        hidden_size=4,
        number_hidden_layers=2,
    )
    for k in ("cluster_std", "center_scale"):
        cfg.pop(k, None)
    return cfg


BASES["spiral"] = _spiral_base()
BASES["teacher"] = _teacher_base()


# --------------------------------------------------------------------------- #
# Variants: each is (label, overrides-dict). method is set per row.
# --------------------------------------------------------------------------- #
def _certifying(eps: float, **extra) -> dict:
    """Our method: deterministic fulldata certificate + certifying selection."""
    return {
        "method": "functional_certified_tiny",
        "functional_certificate_scope": "fulldata",
        "functional_growth_layer_selection": "certifying",
        "functional_marginal_utility_stop": False,
        "functional_relative_error_tolerance": eps,
        **extra,
    }


def _certifying_ggn(eps: float, **extra) -> dict:
    """Our method in the natural (loss-induced GGN/Fisher) metric."""
    return _certifying(eps, functional_certificate_metric="ggn", **extra)


def _minibatch_tiny(eps: float, **extra) -> dict:
    """Legacy noisy variant: mini-batch certificate + fraction threshold + hysteresis."""
    return {
        "method": "functional_certified_tiny",
        "functional_certificate_scope": "minibatch",
        "functional_growth_layer_selection": "tiny_best",
        "functional_relative_error_tolerance": eps,
        "functional_freeze_growth_after_certified": True,
        "functional_certify_threshold": 0.6,
        **extra,
    }


# Block: tolerance sensitivity on blobs. Does certificate-driven growth depend
# less on the tolerance than the legacy mini-batch heuristic?
TOLERANCE_BLOCK = {
    "task": "blobs",
    "variants": [
        ("TINY (scheduled)", {"method": "gromo_tiny"}),
        ("mini-batch $\\eps$=0.1", _minibatch_tiny(0.1)),
        ("mini-batch $\\eps$=0.3", _minibatch_tiny(0.3)),
        ("certifying $\\eps$=0.1", _certifying(0.1)),
        ("certifying $\\eps$=0.2", _certifying(0.2)),
        ("certifying $\\eps$=0.3", _certifying(0.3)),
        ("certifying $\\eps$=0.4", _certifying(0.4)),
    ],
}

# Block: head-to-head per dataset (certificate-driven growth vs scheduled TINY).
DATASET_BLOCK = {
    "tasks": ["blobs", "spiral", "teacher"],
    "variants": [
        ("TINY (scheduled)", {"method": "gromo_tiny"}),
        ("Certified ($\\eps$=0.2)", _certifying(0.2)),
    ],
}

# Block: Euclidean vs natural (GGN/Fisher) certificate metric, per dataset. Tests
# whether measuring the expressivity-bottleneck certificate in the loss-induced
# Hilbert metric improves the accuracy/parameter Pareto frontier (Sec. 4.x).
METRIC_BLOCK = {
    "tasks": ["blobs", "spiral", "teacher"],
    "variants": [
        ("TINY (scheduled)", {"method": "gromo_tiny"}),
        ("Euclidean $\\eps$=0.1", _certifying(0.1)),
        ("\\textbf{GGN} $\\eps$=0.05", _certifying_ggn(0.05)),
        ("\\textbf{GGN} $\\eps$=0.02", _certifying_ggn(0.02)),
    ],
}

ABLATIONS = {
    "tolerance": TOLERANCE_BLOCK,
    "dataset": DATASET_BLOCK,
    "metric": METRIC_BLOCK,
}


def _run_one(base: dict, overrides: dict, seed: int) -> dict:
    import torch
    assert torch.cuda.is_available(), "GPU required: no CUDA device available"
    cfg = deepcopy(base)
    cfg.update(overrides)
    cfg["device"] = "cuda"  # all ablation runs must be on GPU
    cfg["seed"] = seed
    cfg.pop("methods", None)
    cfg["run_name"] = f"ablate_s{seed}"
    cfg["out_dir"] = str(OUT / "_runs")
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = run_experiment(cfg)
    h = result["history"]
    return {
        "final_test_acc": h["test_acc"][-1],
        "final_test_loss": h["test_loss"][-1],
        "final_train_acc": h["train_acc"][-1],
        "final_params": result["final_params"],
        "growth_events": len(result["growth_info"]),
    }


def _agg(runs: list[dict]) -> dict:
    keys = runs[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[k] = {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        }
    return out


def run_block(name: str, block: dict, seeds: list[int]) -> list[dict]:
    tasks = block.get("tasks") or [block["task"]]
    rows = []
    for task in tasks:
        base = BASES[task]
        for label, overrides in block["variants"]:
            runs = []
            for seed in seeds:
                print(f"  [{name}/{task}] {label} seed={seed} ...", flush=True)
                runs.append(_run_one(base, overrides, seed))
            agg = _agg(runs)
            rows.append({"task": task, "label": label, "seeds": seeds, **agg})
            a = agg
            print(
                f"    -> acc {a['final_test_acc']['mean']:.3f}"
                f"+-{a['final_test_acc']['std']:.3f}"
                f"  loss {a['final_test_loss']['mean']:.3f}"
                f"+-{a['final_test_loss']['std']:.3f}"
                f"  params {a['final_params']['mean']:.0f}"
                f"  grow {a['growth_events']['mean']:.1f}",
                flush=True,
            )
    return rows


def _fmt(stat: dict, prec: int = 3) -> str:
    return f"{stat['mean']:.{prec}f}$\\pm${stat['std']:.{prec}f}"


def to_latex(name: str, rows: list[dict]) -> str:
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Task & Method & Test acc. & Test loss & Params & Growths \\\\",
        "\\midrule",
    ]
    last_task = None
    for r in rows:
        task = r["task"] if r["task"] != last_task else ""
        last_task = r["task"]
        lines.append(
            f"{task} & {r['label']} & {_fmt(r['final_test_acc'])} & "
            f"{_fmt(r['final_test_loss'])} & "
            f"{r['final_params']['mean']:.0f} & "
            f"{r['growth_events']['mean']:.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=list(ABLATIONS), help="run a single block")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    blocks = {args.only: ABLATIONS[args.only]} if args.only else ABLATIONS

    if args.dry_run:
        for name, block in blocks.items():
            tasks = block.get("tasks") or [block["task"]]
            n = len(tasks) * len(block["variants"]) * len(args.seeds)
            print(f"{name}: {n} runs ({tasks}, {len(block['variants'])} variants, "
                  f"seeds {args.seeds})")
        return

    for name, block in blocks.items():
        print(f"\n=== ablation block: {name} ===", flush=True)
        rows = run_block(name, block, args.seeds)
        (OUT / f"{name}.json").write_text(json.dumps(rows, indent=2))
        tex = to_latex(name, rows)
        (OUT / f"{name}.tex").write_text(tex)
        tables_dir = ROOT / "report" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        (tables_dir / f"{name}.tex").write_text(tex)
        print(f"Saved results/ablation/{name}.json and report/tables/{name}.tex")


if __name__ == "__main__":
    main()
