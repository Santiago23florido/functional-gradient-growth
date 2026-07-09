"""GroMo growth step for the baseline pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from stable_tiny.gromo_setup import ensure_gromo_importable


ensure_gromo_importable()

import torch

from gromo.containers.growing_mlp import GrowingMLP
from gromo.utils.training_utils import compute_statistics, evaluate_model


ProgressFn = Callable[[str], None]


@dataclass(frozen=True)
class LineSearchPoint:
    scaling_factor: float
    train_loss: float


@dataclass(frozen=True)
class GrowthResult:
    layer_index: int
    best_scaling_factor: float
    best_train_loss: float
    line_search: list[LineSearchPoint]


def grow_layer(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    layer_index: int,
    device: torch.device,
    scaling_factors: Sequence[float],
    progress: ProgressFn | None = None,
) -> GrowthResult:
    """Grow one GroMo layer and apply the best line-search update.

    ``layer_index`` follows GroMo's local API: it is zero-based over
    ``model._growable_layers``.
    """
    criterion_sum = torch.nn.MSELoss(reduction="sum")
    criterion_mean = torch.nn.MSELoss(reduction="mean")

    model.set_growing_layers(index=layer_index)
    compute_statistics(
        model,
        train_loader,
        loss_function=criterion_sum,
        device=device,
    )

    model.compute_optimal_updates()
    model.reset_computation()
    model.dummy_select_update()

    best_loss = float("inf")
    best_value = float(scaling_factors[0])
    line_search: list[LineSearchPoint] = []

    for value in scaling_factors:
        model.set_scaling_factor(value)
        loss, _ = evaluate_model(
            model,
            train_loader,
            criterion_mean,
            use_extended_model=True,
            device=device,
        )
        train_loss = float(loss)
        line_search.append(LineSearchPoint(float(value), train_loss))

        if progress is not None:
            progress(f"  scaling={value:.4g}, train_loss={train_loss:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            best_value = float(value)

    model.set_scaling_factor(best_value)
    model.apply_change()

    return GrowthResult(
        layer_index=layer_index,
        best_scaling_factor=best_value,
        best_train_loss=float(best_loss),
        line_search=line_search,
    )
