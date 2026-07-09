"""Plotting helpers for GroMo baseline histories."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class HistoryEntryLike(Protocol):
    step: int
    step_type: str
    test_loss: float
    num_params: int


def plot_history(
    history: Sequence[HistoryEntryLike],
    output_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Plot test loss and parameter count over training/growth steps."""
    import matplotlib.pyplot as plt

    steps = [entry.step for entry in history]
    losses = [entry.test_loss for entry in history]
    params = [entry.num_params for entry in history]
    sgd_indices = [
        i for i, entry in enumerate(history) if entry.step_type in {"INIT", "SGD"}
    ]
    gro_indices = [i for i, entry in enumerate(history) if entry.step_type == "GRO"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Test Loss", color="tab:blue")
    ax1.plot(steps, losses, color="tab:blue", alpha=0.5, linewidth=1)
    ax1.scatter(
        [steps[i] for i in sgd_indices],
        [losses[i] for i in sgd_indices],
        color="tab:blue",
        marker="o",
        s=70,
        label="SGD Loss",
        zorder=3,
    )
    ax1.scatter(
        [steps[i] for i in gro_indices],
        [losses[i] for i in gro_indices],
        color="tab:blue",
        marker="*",
        s=170,
        label="Growth Loss",
        zorder=3,
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Trainable Parameters", color="tab:orange")
    ax2.plot(steps, params, color="tab:orange", alpha=0.5, linewidth=1)
    ax2.scatter(
        [steps[i] for i in sgd_indices],
        [params[i] for i in sgd_indices],
        color="tab:orange",
        marker="o",
        s=70,
        label="SGD Params",
        zorder=3,
    )
    ax2.scatter(
        [steps[i] for i in gro_indices],
        [params[i] for i in gro_indices],
        color="tab:orange",
        marker="*",
        s=170,
        label="Growth Params",
        zorder=3,
    )
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Model Performance and Capacity Evolution")

    fig.tight_layout()

    saved_path = None
    if output_path is not None:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=160)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path
