"""Learning-rate scheduler configuration and application."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from stable_tiny.gromo_setup import ensure_gromo_importable


ensure_gromo_importable()

import torch


SchedulerName = Literal["none", "cosineannealing"]


@dataclass(frozen=True)
class LRSchedulerConfig:
    name: SchedulerName = "cosineannealing"
    t_max: int | None = None
    eta_min: float = 0.0
    restart_on_growth: bool = True


def _normalise_name(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def learning_rate_for_epoch(
    config: LRSchedulerConfig,
    base_learning_rate: float,
    epoch: int,
    total_epochs: int,
    growth_every: int | None = None,
    first_growth_epoch: int | None = None,
) -> float:
    """Return the learning rate for an epoch.

    With ``restart_on_growth=True``, cosine annealing is computed inside each
    growth interval instead of across the full training run.
    """
    scheduler_name = _normalise_name(config.name)
    if scheduler_name in {"none", "constant"}:
        return base_learning_rate

    if scheduler_name == "cosineannealing":
        cycle_position = max(0, epoch)
        default_t_max = total_epochs

        if (
            config.restart_on_growth
            and growth_every is not None
            and growth_every > 0
            and first_growth_epoch is not None
        ):
            default_t_max = growth_every
            if epoch > first_growth_epoch:
                relative_epoch = epoch - first_growth_epoch
                cycle_position = relative_epoch % growth_every
                if cycle_position == 0:
                    cycle_position = growth_every

        t_max = config.t_max if config.t_max is not None else default_t_max
        t_max = max(1, t_max)
        position = min(cycle_position, t_max)
        cosine = 0.5 * (1.0 + math.cos(math.pi * position / t_max))
        return config.eta_min + (base_learning_rate - config.eta_min) * cosine

    raise ValueError(
        f"Unsupported LR scheduler '{config.name}'. Use one of: none, cosineannealing."
    )


def apply_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    """Set the same learning rate on all optimizer parameter groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
