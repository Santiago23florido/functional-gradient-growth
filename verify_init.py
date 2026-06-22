#!/usr/bin/env python
"""Verify that all growth methods start from *identical* initial conditions.

A fair head-to-head between scheduled TINY and the certificate-driven method
requires that both begin from the same base network and the same data -- only the
*growth policy* should differ. ``run_experiment`` re-seeds with ``cfg['seed']``
before building the dataloaders and the model, and the data/architecture config is
shared across methods, so for a fixed seed the RNG state at model-construction time
is identical and the initial weights are bitwise-equal. This script checks that
invariant directly: it mirrors ``run_experiment``'s seed -> data -> model sequence
for ``gromo_tiny`` and ``functional_certified_tiny`` and asserts that the initial
parameters and the first training batch match exactly.

Usage:
    ../.venv/bin/python verify_init.py
    ../.venv/bin/python verify_init.py --configs configs/compare_blobs_natural.yaml
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))
from stable_tiny.data import get_dataloaders  # noqa: E402
from stable_tiny.experiment import _build_model  # noqa: E402


def init_state(cfg: dict, device: str):
    """Reproduce run_experiment's seed -> dataloaders -> model order exactly."""
    torch.manual_seed(cfg["seed"])
    dev = torch.device(device)
    train_loader, _test_loader, meta = get_dataloaders(cfg)
    model = _build_model(cfg, meta, dev)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    xb, yb = next(iter(train_loader))
    return state, xb, yb, meta


def compare(cfg_a: dict, cfg_b: dict, device: str, tag: str) -> bool:
    sd_a, x_a, y_a, meta = init_state(cfg_a, device)
    sd_b, x_b, y_b, _ = init_state(cfg_b, device)
    same_keys = sorted(sd_a) == sorted(sd_b)
    same_w = same_keys and all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    same_x = torch.equal(x_a.cpu(), x_b.cpu())
    same_y = torch.equal(y_a.cpu(), y_b.cpu())
    nparam = sum(v.numel() for v in sd_a.values())
    ok = same_w and same_x and same_y
    print(
        f"[{tag}/{device}] arch={meta['in_features']}->{meta['out_features']} "
        f"init_params={nparam} | same_init_weights={same_w} "
        f"same_first_x={same_x} same_first_y={same_y} -> {'OK' if ok else 'MISMATCH'}"
    )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/compare_blobs_natural.yaml",
            "configs/compare_blobs_stable.yaml",
            "configs/compare_cifar_natural.yaml",
        ],
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    all_ok = True
    for conf in args.configs:
        base = yaml.safe_load(open(conf))
        base.pop("methods", None)
        a = deepcopy(base)
        a.update(method="gromo_tiny", seed=args.seed)
        b = deepcopy(base)
        b.update(method="functional_certified_tiny", seed=args.seed)
        for dev in devices:
            all_ok &= compare(a, b, dev, Path(conf).name)

    print("\nAll methods share an identical starting point." if all_ok
          else "\nWARNING: starting points differ -- comparison is NOT controlled.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
