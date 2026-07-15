"""Optimizer configuration and construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fgdlib.gromo_setup import ensure_gromo_importable


ensure_gromo_importable()

import torch


OptimizerName = Literal["sgd", "adam", "adamw"]


@dataclass(frozen=True)
class OptimizerConfig:
    name: OptimizerName = "sgd"
    learning_rate: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


def build_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Create an optimizer from YAML-backed config."""
    optimizer_name = config.name.lower()

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer '{config.name}'. Use one of: sgd, adam, adamw."
    )


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Return the first parameter group's learning rate."""
    return float(optimizer.param_groups[0]["lr"])
