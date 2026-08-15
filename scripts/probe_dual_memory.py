"""Where do the 35 GB of the dual branch go? Measured stage by stage.

The parameter budget is capped at 16000 because a run at P=16304 peaked at
35.3 GB of a 40 GB A100. Only ~4.8 GB of that is the factors W and V; the rest
is transient, and it has never been profiled -- so every proposal to lower it
has been a guess.

This replays `dual_spectrum` + the caller's factor construction one step at a
time, reporting live and peak after each, so the peak can be attributed to a
specific tensor. It touches no production code.

The question it answers: what fraction is MOVABLE (factors and their copies,
which could live in host RAM -- 480 GB against 40) versus stuck inside
`torch.linalg.eigh`, which has to run somewhere.

    PYTHONPATH=src python scripts/probe_dual_memory.py
    PROBE_HIDDEN=10,14,20 PYTHONPATH=src python scripts/probe_dual_memory.py
"""

import dataclasses
import gc
import os
import sys

sys.path.insert(0, "src")

import torch

from fgdlib import tangent as T
from fgdlib.search.matrixfree import analytic_operators, dual_gram
from stable_tiny.pipeline import build_model, load_pipeline_config

# Never take the whole card: this is a diagnostic, not the experiment.
FRACTION = float(os.environ.get("PROBE_GPU_FRACTION", "0.9"))
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(FRACTION)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = os.environ.get("PROBE_CONFIG", "configs/experiments/mnist_full.yaml")
HIDDEN = [int(h) for h in os.environ.get("PROBE_HIDDEN", "10,14,20").split(",")]


def _snapshot() -> tuple[float, float]:
    if DEVICE.type != "cuda":
        return 0.0, 0.0
    torch.cuda.synchronize()
    return (
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.max_memory_allocated() / 1e9,
    )


def _stage(label: str, previous_peak: float) -> float:
    live, peak = _snapshot()
    print(
        f"  {label:34s} live={live:7.2f} GB  peak={peak:7.2f} GB"
        f"  (+{peak - previous_peak:6.2f})",
        flush=True,
    )
    return peak


def profile(hidden: int, base) -> None:
    config = dataclasses.replace(
        base, model=dataclasses.replace(base.model, hidden_size=hidden)
    )
    torch.manual_seed(0)
    model = build_model(config, DEVICE)
    model.eval()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # The probe the parameter floor forces: NK = 1.25 P, which is the regime
    # that made the dual Gram (NK x NK) larger than the primal (P x P).
    samples = max(1, int(1.25 * parameters / 10))
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(samples, 784, generator=generator).to(DEVICE)

    print(f"\n=== P={parameters}  NK={samples * 10}  NK/P=1.25 ===", flush=True)
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        with model.paused_computation():
            structure = T._supported_analytic_structure(
                model, x, capture_suspended=True
            )
            if structure is None:
                print("  analytic structure REFUSED -- dual route unavailable")
                return
            operators = analytic_operators(structure, x)
            peak = _stage("operators", 0.0)

            gram = dual_gram(operators)
            peak = _stage("dual_gram (NK x NK)", peak)

            values, vectors = torch.linalg.eigh(gram)
            peak = _stage("eigh", peak)

            largest = float(values.max())
            keep = values > max(
                largest * len(values) * 1e-14, config.fgd_approx.eps
            )
            kept_values = values[keep].flip(0)
            kept_vectors = vectors[:, keep].flip(1)
            peak = _stage("keep + flip (copy)", peak)
            rank = int(keep.sum())
            print(f"     k={rank}  k/P={rank / parameters:.3f}", flush=True)

            del gram, values, vectors
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            _stage("after freeing gram+vectors", peak)

            singular = kept_values.clamp_min(0.0).sqrt()
            right = (operators.apply_jt(kept_vectors.t()) / singular.unsqueeze(1)).t()
            left = kept_vectors * singular
            _stage("final factors W,V", peak)
            moved = (left.numel() + right.numel()) * left.element_size() / 1e9
            print(
                f"     W={tuple(left.shape)}  V={tuple(right.shape)}"
                f"  = {moved:.2f} GB  <- the movable part",
                flush=True,
            )
            del right, left, kept_vectors, singular, kept_values
    except RuntimeError as error:
        print(f"  OOM/ERROR: {str(error)[:80]}", flush=True)
    finally:
        del model, x
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    print(f"device={DEVICE}  config={CONFIG}  hidden={HIDDEN}", flush=True)
    if DEVICE.type == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"gpu={name}  total={total:.1f} GB  cap={FRACTION:.0%}", flush=True)
    base = load_pipeline_config(CONFIG)
    for hidden in HIDDEN:
        profile(hidden, base)


if __name__ == "__main__":
    main()
