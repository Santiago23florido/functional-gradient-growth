"""Plotting helpers for GroMo baseline histories."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class HistoryEntryLike(Protocol):
    step: int
    step_type: str
    train_loss: float
    test_loss: float
    train_accuracy: float
    test_accuracy: float
    learning_rate: float
    num_params: int


def plot_history(
    history: Sequence[HistoryEntryLike],
    output_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Plot temporal metric evolution without point markers."""
    import matplotlib.pyplot as plt

    steps = [entry.step for entry in history]
    train_losses = [entry.train_loss for entry in history]
    test_losses = [entry.test_loss for entry in history]
    train_accuracies = [entry.train_accuracy for entry in history]
    test_accuracies = [entry.test_accuracy for entry in history]
    learning_rates = [entry.learning_rate for entry in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    loss_ax, train_acc_ax, test_acc_ax, lr_ax = axes.ravel()

    loss_ax.plot(steps, train_losses, linewidth=1.8, label="Train Loss")
    loss_ax.plot(steps, test_losses, linewidth=1.8, label="Test Loss")
    loss_ax.set_title("Loss")
    loss_ax.set_ylabel("MSE")
    loss_ax.legend(loc="best")

    train_acc_ax.plot(steps, train_accuracies, linewidth=1.8, color="tab:green")
    train_acc_ax.set_title("Train Accuracy")
    train_acc_ax.set_ylabel("Tolerance Accuracy")
    train_acc_ax.set_ylim(0.0, 1.0)

    test_acc_ax.plot(steps, test_accuracies, linewidth=1.8, color="tab:purple")
    test_acc_ax.set_title("Test Accuracy")
    test_acc_ax.set_xlabel("Epoch")
    test_acc_ax.set_ylabel("Tolerance Accuracy")
    test_acc_ax.set_ylim(0.0, 1.0)

    lr_ax.plot(steps, learning_rates, linewidth=1.8, color="tab:red")
    lr_ax.set_title("Learning Rate")
    lr_ax.set_xlabel("Epoch")
    lr_ax.set_ylabel("LR")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)

    fig.suptitle("GroMo Baseline Metrics", fontsize=14)
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


def plot_parameters(
    history: Sequence[HistoryEntryLike],
    output_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Plot trainable parameter count over time."""
    import matplotlib.pyplot as plt

    steps = [entry.step for entry in history]
    num_params = [entry.num_params for entry in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, num_params, linewidth=1.8, color="tab:orange")
    ax.set_title("Trainable Parameters")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameters")
    ax.grid(True, alpha=0.25)
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
