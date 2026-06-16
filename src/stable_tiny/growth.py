"""One growth step of a GrowingMLP using gromo's TINY procedure.

This mirrors the official gromo container tutorial
(``examples/plot_growing_container_tutorial.py``) but:

- uses ``CrossEntropyLoss`` (classification) instead of MSE,
- lets you cap how many neurons are added per growth
  (``maximum_added_neurons``), which controls how strongly the function is
  perturbed -- and therefore how visible any post-growth spike is,
- returns the chosen scaling factor and the resulting layer size.

Nothing here implements the stable-growth (S-orthogonal) theory; this is the
*plain* gromo growth, exactly the regime where we want to see whether spikes
appear.
"""

from __future__ import annotations

import torch
import torch.utils.data

from gromo.containers.growing_mlp import GrowingMLP
from gromo.utils.training_utils import compute_statistics, evaluate_model


@torch.no_grad()
def _eval_extended_loss(
    model: GrowingMLP,
    loader: torch.utils.data.DataLoader,
    criterion_mean: torch.nn.Module,
    device: torch.device,
) -> float:
    loss, _ = evaluate_model(
        model, loader, criterion_mean, use_extended_model=True, device=device
    )
    return loss


def grow_step(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    layer_to_grow: int,
    device: torch.device,
    *,
    maximum_added_neurons: int | None = None,
    line_search_factors: tuple[float, ...] | list[float] = (0.0, 0.1, 0.5, 1.0),
) -> dict:
    """Grow ``layer_to_grow`` and return diagnostic info about the step."""
    criterion_sum = torch.nn.CrossEntropyLoss(reduction="sum")
    criterion_mean = torch.nn.CrossEntropyLoss(reduction="mean")

    model.set_growing_layers(scheduling_method="sequential", index=layer_to_grow)

    # Accumulate gradient statistics over the whole training set.
    compute_statistics(
        model, train_loader, loss_function=criterion_sum, device=device
    )
    model.compute_optimal_updates(maximum_added_neurons=maximum_added_neurons)
    model.reset_computation()
    model.dummy_select_update()

    # Line search for the scaling factor that minimises *train* loss.
    best_loss = float("inf")
    best_value = 0.0
    search = {}
    for value in line_search_factors:
        model.set_scaling_factor(value)
        loss = _eval_extended_loss(model, train_loader, criterion_mean, device)
        search[value] = loss
        if loss < best_loss:
            best_loss = loss
            best_value = value

    model.set_scaling_factor(best_value)
    model.apply_change()

    return {
        "layer": layer_to_grow,
        "scaling_factor": best_value,
        "line_search": search,
        "extended_train_loss": best_loss,
    }


def select_tiny_growth_layer(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    maximum_added_neurons: int | None = None,
) -> tuple[int, dict]:
    """Select the layer with the largest TINY first-order improvement.

    This is a read-only scoring pass: it computes the same statistics and
    optimal updates used by GroMo/TINY, records the predicted improvement of
    every growable layer, deletes those temporary updates, and returns the best
    layer index. The actual growth is still performed by ``grow_step``.
    """
    criterion_sum = torch.nn.CrossEntropyLoss(reduction="sum")

    model.set_growing_layers(scheduling_method="all")
    compute_statistics(model, train_loader, loss_function=criterion_sum, device=device)
    model.compute_optimal_updates(maximum_added_neurons=maximum_added_neurons)

    raw_info = model.update_information()
    scores = {
        int(index): float(info["update_value"].detach().cpu())
        for index, info in raw_info.items()
    }
    parameter_scores = {
        int(index): float(info["parameter_improvement"].detach().cpu())
        for index, info in raw_info.items()
    }
    best_layer = max(scores, key=scores.get)

    for layer in model._growing_layers:
        layer.delete_update(include_previous=True, delete_output=True)
    model.reset_computation()
    model.currently_updated_layer_index = None

    return best_layer, {
        "selection": "tiny_best_first_order_improvement",
        "tiny_scores": scores,
        "tiny_parameter_scores": parameter_scores,
    }
