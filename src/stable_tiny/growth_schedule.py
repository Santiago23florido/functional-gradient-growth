"""Growth scheduling policies for GroMo experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LayerPolicy = Literal["round_robin", "first"]


@dataclass(frozen=True)
class GrowthScheduleConfig:
    enabled: bool = True
    every: int = 50
    first_epoch: int = 50
    layer_policy: LayerPolicy = "round_robin"


def should_grow(epoch: int, config: GrowthScheduleConfig) -> bool:
    """Return whether a growth step should run after this epoch."""
    if not config.enabled or config.every <= 0:
        return False

    if epoch < config.first_epoch:
        return False

    return (epoch - config.first_epoch) % config.every == 0


def layer_index_for_growth(
    growth_count: int,
    number_hidden_layers: int,
    config: GrowthScheduleConfig,
) -> int:
    """Select the next growable layer index."""
    if number_hidden_layers <= 0:
        raise ValueError("number_hidden_layers must be positive")

    if config.layer_policy == "first":
        return 0

    if config.layer_policy == "round_robin":
        return growth_count % number_hidden_layers

    raise ValueError(
        f"Unsupported growth layer policy '{config.layer_policy}'. "
        "Use one of: round_robin, first."
    )
