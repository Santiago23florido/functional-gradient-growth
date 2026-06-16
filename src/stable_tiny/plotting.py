"""Plot loss/accuracy/parameter curves for single or comparative runs.

Single-method plots mark growth events with vertical dashed lines. Comparative
plots overlay methods in one PNG, using color for the method and line style for
train/test.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save to file
import matplotlib.pyplot as plt


_METHOD_LABELS = {
    "baseline_mlp": "MLP",
    "gromo_tiny": "Scheduled TINY + AdamW",
    "functional_triggered_tiny": "Functional certificate + TINY",
    "functional_certified_tiny": "Certified FGD + grow-until-certified",
}

_METHOD_COLORS = {
    "baseline_mlp": "tab:green",
    "gromo_tiny": "tab:blue",
    "functional_triggered_tiny": "tab:orange",
    "functional_certified_tiny": "tab:purple",
}


def plot_history(result: dict, out_path: str | Path) -> Path:
    h = result["history"]
    x = h["eval_idx"]
    growth_lines = result.get("growth_lines", [])

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    ax_loss, ax_acc, ax_params = axes

    # --- Panel 1: loss ----------------------------------------------------
    ax_loss.plot(x, h["train_loss"], "-o", ms=3, label="train loss", color="tab:blue")
    ax_loss.plot(x, h["test_loss"], "-o", ms=3, label="test loss", color="tab:red")
    ax_loss.set_ylabel("loss (cross-entropy)")
    ax_loss.set_title("Loss with growth events (look for spikes right after dashed lines)")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(alpha=0.3)

    # --- Panel 2: accuracy ------------------------------------------------
    ax_acc.plot(x, h["train_acc"], "-o", ms=3, label="train acc", color="tab:blue")
    ax_acc.plot(x, h["test_acc"], "-o", ms=3, label="test acc", color="tab:red")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend(loc="lower right")
    ax_acc.grid(alpha=0.3)

    # --- Panel 3: params --------------------------------------------------
    ax_params.step(x, h["params"], where="post", color="tab:green")
    ax_params.set_ylabel("# parameters")
    ax_params.set_xlabel("evaluation index")
    ax_params.set_title("Model size (capacity)")
    ax_params.grid(alpha=0.3)

    # --- growth event lines on every panel --------------------------------
    for gx in growth_lines:
        for ax in axes:
            ax.axvline(gx, color="gray", ls="--", lw=1, alpha=0.7)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_comparison(results: list[dict], out_path: str | Path) -> Path:
    """Plot several experiment histories in one PNG.

    Method is encoded by color. Train/test split is encoded by line style.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    ax_loss, ax_acc, ax_params = axes

    for index, result in enumerate(results):
        method = result["method"]
        h = result["history"]
        x = h["epoch"]
        label = _METHOD_LABELS.get(method, method)
        color = _METHOD_COLORS.get(method, f"C{index}")

        ax_loss.plot(
            x,
            h["train_loss"],
            "-o",
            ms=3,
            color=color,
            label=f"{label} train",
        )
        ax_loss.plot(
            x,
            h["test_loss"],
            "--o",
            ms=3,
            color=color,
            alpha=0.8,
            label=f"{label} test",
        )

        ax_acc.plot(
            x,
            h["train_acc"],
            "-o",
            ms=3,
            color=color,
            label=f"{label} train",
        )
        ax_acc.plot(
            x,
            h["test_acc"],
            "--o",
            ms=3,
            color=color,
            alpha=0.8,
            label=f"{label} test",
        )

        ax_params.step(
            x,
            h["params"],
            where="post",
            color=color,
            linewidth=2,
            label=label,
        )

        growth_epochs = [
            h["epoch"][idx]
            for idx, phase in enumerate(h["phase"])
            if phase == "post_grow"
        ]
        growth_params = [
            h["params"][idx]
            for idx, phase in enumerate(h["phase"])
            if phase == "post_grow"
        ]
        if growth_epochs:
            ax_params.scatter(
                growth_epochs,
                growth_params,
                color=color,
                marker="x",
                s=50,
                zorder=3,
            )

    ax_loss.set_ylabel("loss (cross-entropy)")
    ax_loss.set_title("Train/test loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc="best", fontsize=8)

    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Train/test accuracy")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(loc="best", fontsize=8)

    ax_params.set_ylabel("# parameters")
    ax_params.set_xlabel("epoch")
    ax_params.set_title("Parameter count")
    ax_params.grid(alpha=0.3)
    ax_params.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def report_spikes(result: dict) -> None:
    """Report loss behaviour around each growth event.

    Two effects are distinguished:

    - *instant*: pre-growth SGD eval -> post-growth eval. With train-loss line
      search this is (almost) always a decrease, so it rarely shows a spike.
    - *rebound*: the worst loss reached during the SGD epochs that follow the
      growth, relative to the post-growth value. This is where the real
      post-growth spike usually shows up (transient instability), especially in
      the test loss / accuracy.
    """
    h = result["history"]
    phases = h["phase"]
    n = len(phases)
    print("\n=== Growth events: instant change & post-growth rebound ===")
    for i, phase in enumerate(phases):
        if phase != "post_grow":
            continue
        pre_tr, pre_te = h["train_loss"][i - 1], h["test_loss"][i - 1]
        post_tr, post_te = h["train_loss"][i], h["test_loss"][i]
        d_tr, d_te = post_tr - pre_tr, post_te - pre_te

        # Training window after this growth (until the next growth or the end).
        j = i + 1
        while j < n and phases[j] in {"sgd", "fgd", "warmup"}:
            j += 1
        window_tr = h["train_loss"][i + 1 : j]
        window_te = h["test_loss"][i + 1 : j]
        # Rebound = how far the loss climbs above the post-growth value during
        # the following SGD window; <= 0 means it only kept decreasing.
        reb_tr = max(0.0, (max(window_tr) - post_tr) if window_tr else 0.0)
        reb_te = max(0.0, (max(window_te) - post_te) if window_te else 0.0)
        flag = "  <-- SPIKE" if (reb_tr > 0 or reb_te > 0) else ""

        print(
            f"  growth@eval{i}: "
            f"instant train Δ{d_tr:+.4f} test Δ{d_te:+.4f} | "
            f"rebound train +{reb_tr:.4f} test +{reb_te:.4f}{flag}"
        )
