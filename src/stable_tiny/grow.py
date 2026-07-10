"""GroMo growth step for the baseline pipeline."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from stable_tiny.gromo_setup import ensure_gromo_importable


ensure_gromo_importable()

import torch

from gromo.containers.growing_mlp import GrowingMLP
from gromo.utils.training_utils import compute_statistics, evaluate_model


ProgressFn = Callable[[str], None]
LineSearchMethod = Literal["golden_section"]


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


@dataclass(frozen=True)
class ScalingLineSearchConfig:
    method: LineSearchMethod = "golden_section"
    min_value: float = 0.0
    max_value: float = 1.0
    iterations: int = 12
    tolerance: float = 1e-3


def _evaluate_scaling_factor(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    scaling_factor: float,
    evaluated: dict[float, LineSearchPoint],
    line_search: list[LineSearchPoint],
    progress: ProgressFn | None,
) -> LineSearchPoint:
    key = round(float(scaling_factor), 12)
    if key in evaluated:
        return evaluated[key]

    model.set_scaling_factor(float(scaling_factor))
    loss, _ = evaluate_model(
        model,
        train_loader,
        criterion,
        use_extended_model=True,
        device=device,
    )
    point = LineSearchPoint(scaling_factor=float(scaling_factor), train_loss=float(loss))
    evaluated[key] = point
    line_search.append(point)

    if progress is not None:
        progress(
            f"  scaling={point.scaling_factor:.6g}, "
            f"train_loss={point.train_loss:.4f}"
        )

    return point


def _golden_section_line_search(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    config: ScalingLineSearchConfig,
    progress: ProgressFn | None,
) -> tuple[float, float, list[LineSearchPoint]]:
    if config.max_value < config.min_value:
        raise ValueError("scaling_line_search.max_value must be >= min_value")

    line_search: list[LineSearchPoint] = []
    evaluated: dict[float, LineSearchPoint] = {}

    a = float(config.min_value)
    b = float(config.max_value)
    if math.isclose(a, b):
        point = _evaluate_scaling_factor(
            model,
            train_loader,
            criterion,
            device,
            a,
            evaluated,
            line_search,
            progress,
        )
        return point.scaling_factor, point.train_loss, line_search

    _evaluate_scaling_factor(
        model, train_loader, criterion, device, a, evaluated, line_search, progress
    )
    _evaluate_scaling_factor(
        model, train_loader, criterion, device, b, evaluated, line_search, progress
    )

    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    c_point = _evaluate_scaling_factor(
        model, train_loader, criterion, device, c, evaluated, line_search, progress
    )
    d_point = _evaluate_scaling_factor(
        model, train_loader, criterion, device, d, evaluated, line_search, progress
    )

    for _ in range(max(0, config.iterations)):
        if abs(b - a) <= config.tolerance:
            break

        if c_point.train_loss <= d_point.train_loss:
            b = d
            d = c
            d_point = c_point
            c = b - inv_phi * (b - a)
            c_point = _evaluate_scaling_factor(
                model,
                train_loader,
                criterion,
                device,
                c,
                evaluated,
                line_search,
                progress,
            )
        else:
            a = c
            c = d
            c_point = d_point
            d = a + inv_phi * (b - a)
            d_point = _evaluate_scaling_factor(
                model,
                train_loader,
                criterion,
                device,
                d,
                evaluated,
                line_search,
                progress,
            )

    best_point = min(line_search, key=lambda point: point.train_loss)
    return best_point.scaling_factor, best_point.train_loss, line_search


def grow_layer(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    layer_index: int,
    device: torch.device,
    line_search_config: ScalingLineSearchConfig,
    optimal_update_kwargs: dict[str, Any] | None = None,
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

    model.compute_optimal_updates(**(optimal_update_kwargs or {}))
    model.reset_computation()
    model.dummy_select_update()

    if line_search_config.method != "golden_section":
        raise ValueError(
            f"Unsupported scaling line-search method '{line_search_config.method}'."
        )

    best_value, best_loss, line_search = _golden_section_line_search(
        model=model,
        train_loader=train_loader,
        criterion=criterion_mean,
        device=device,
        config=line_search_config,
        progress=progress,
    )

    model.set_scaling_factor(best_value)
    model.apply_change()

    return GrowthResult(
        layer_index=layer_index,
        best_scaling_factor=best_value,
        best_train_loss=float(best_loss),
        line_search=line_search,
    )
