"""Training and evaluation helpers for the GroMo pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from stable_tiny.gromo_setup import ensure_gromo_importable


ensure_gromo_importable()

import torch

from gromo.utils.training_utils import evaluate_model, gradient_descent


@dataclass(frozen=True)
class TrainEpochResult:
    train_loss: float
    test_loss: float


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


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    device: torch.device,
) -> TrainEpochResult:
    """Run one gradient-descent epoch and evaluate on the test loader."""
    train_loss, _ = gradient_descent(
        model,
        train_loader,
        optimizer,
        scheduler=None,
        loss_function=loss_function,
        device=device,
    )
    test_loss = evaluate_loss(
        model,
        test_loader,
        loss_function,
        device=device,
    )
    return TrainEpochResult(train_loss=float(train_loss), test_loss=test_loss)
