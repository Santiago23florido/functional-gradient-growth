"""Training and evaluation helpers for the GroMo pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from stable_tiny.gromo_setup import ensure_gromo_importable


ensure_gromo_importable()

import torch

from gromo.utils.training_utils import evaluate_model, gradient_descent


@dataclass(frozen=True)
class RegressionMetrics:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class TrainEpochResult:
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_loss(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
) -> float:
    """Evaluate a model and return the average loss."""
    loss, _ = evaluate_model(
        model,
        dataloader,
        loss_function,
        device=device,
    )
    return float(loss)


@torch.no_grad()
def evaluate_regression_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    accuracy_tolerance: float,
) -> RegressionMetrics:
    """Evaluate loss and tolerance-based regression accuracy.

    Accuracy is the element-wise fraction of predictions whose absolute error is
    less than or equal to ``accuracy_tolerance``.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_values = 0
    total_values = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_function(y_pred, y)

        batch_size = x.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size

        correct_values += int((y_pred - y).abs().le(accuracy_tolerance).sum().item())
        total_values += y.numel()

    return RegressionMetrics(
        loss=total_loss / max(1, total_samples),
        accuracy=correct_values / max(1, total_values),
    )


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    device: torch.device,
    accuracy_tolerance: float,
) -> TrainEpochResult:
    """Run one gradient-descent epoch and evaluate on the test loader."""
    gradient_descent(
        model,
        train_loader,
        optimizer,
        scheduler=None,
        loss_function=loss_function,
        device=device,
    )
    train_metrics = evaluate_regression_metrics(
        model,
        train_loader,
        loss_function,
        device=device,
        accuracy_tolerance=accuracy_tolerance,
    )
    test_metrics = evaluate_regression_metrics(
        model,
        test_loader,
        loss_function,
        device=device,
        accuracy_tolerance=accuracy_tolerance,
    )
    return TrainEpochResult(
        train_loss=train_metrics.loss,
        train_accuracy=train_metrics.accuracy,
        test_loss=test_metrics.loss,
        test_accuracy=test_metrics.accuracy,
    )


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Return the first parameter group's learning rate."""
    return float(optimizer.param_groups[0]["lr"])
