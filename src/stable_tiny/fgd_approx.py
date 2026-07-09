"""Approximate functional-gradient-descent training."""

from __future__ import annotations

from dataclasses import dataclass

from stable_tiny.gromo_setup import ensure_gromo_importable
from stable_tiny.train import RegressionMetrics, evaluate_regression_metrics


ensure_gromo_importable()

import torch


@dataclass(frozen=True)
class FGDApproxConfig:
    rel_error_threshold: float = 0.5
    start_epoch: int = 1
    min_epochs_between_growth: int = 1
    eps: float = 1e-12


@dataclass(frozen=True)
class FGDApproxEpochResult:
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    relative_error: float


def mse_functional_gradient(
    y_pred: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return the MSE functional-gradient direction up to a constant factor."""
    return y_pred.detach() - y.detach()


def relative_l2_error(
    approximation: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> float:
    """Compute ||approximation - target|| / ||approximation|| on a batch."""
    numerator = torch.sqrt(torch.mean((approximation - target) ** 2))
    denominator = torch.sqrt(torch.mean(approximation**2)).clamp_min(eps)
    return float((numerator / denominator).detach().item())


def should_trigger_fgd_growth(
    relative_error: float,
    epoch: int,
    last_growth_epoch: int | None,
    config: FGDApproxConfig,
) -> bool:
    """Return whether the FGD approximation error should trigger GroMo growth."""
    if epoch < config.start_epoch:
        return False

    if last_growth_epoch is not None:
        if epoch - last_growth_epoch < config.min_epochs_between_growth:
            return False

    return relative_error > config.rel_error_threshold


def train_one_epoch_fgd_approx(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    device: torch.device,
    learning_rate: float,
    accuracy_tolerance: float,
    gradient_clip_norm: float | None,
    config: FGDApproxConfig,
) -> FGDApproxEpochResult:
    """Train one epoch and estimate FGD relative error.

    The functional direction is approximated from the realized parameter update:
    ``g_t,theta approx (f_t - f_{t+1}) / eta``.
    """
    model.train()
    rel_error_sum = 0.0
    rel_error_count = 0
    step_scale = max(abs(learning_rate), config.eps)

    for batch_index, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_before = model(x)
        loss = loss_function(y_before, y)
        if not torch.isfinite(loss).all():
            raise RuntimeError(
                "Non-finite FGD training loss detected "
                f"(loss={loss.item()}, batch_index={batch_index})."
            )

        functional_gradient = mse_functional_gradient(y_before, y)
        y_before_detached = y_before.detach()

        loss.backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        with torch.no_grad():
            y_after = model(x)
            fgd_direction = (y_before_detached - y_after) / step_scale
            relative_error = relative_l2_error(
                approximation=fgd_direction,
                target=functional_gradient,
                eps=config.eps,
            )

        batch_size = x.size(0)
        rel_error_sum += relative_error * batch_size
        rel_error_count += batch_size

    train_metrics: RegressionMetrics = evaluate_regression_metrics(
        model,
        train_loader,
        loss_function,
        device=device,
        accuracy_tolerance=accuracy_tolerance,
    )
    test_metrics: RegressionMetrics = evaluate_regression_metrics(
        model,
        test_loader,
        loss_function,
        device=device,
        accuracy_tolerance=accuracy_tolerance,
    )

    return FGDApproxEpochResult(
        train_loss=train_metrics.loss,
        train_accuracy=train_metrics.accuracy,
        test_loss=test_metrics.loss,
        test_accuracy=test_metrics.accuracy,
        relative_error=rel_error_sum / max(1, rel_error_count),
    )
