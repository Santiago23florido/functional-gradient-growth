"""GroMo growth step for the baseline pipeline."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from fgdlib.gromo_setup import ensure_gromo_importable


ensure_gromo_importable()

import torch

from gromo.containers.growing_mlp import GrowingMLP
from gromo.utils.training_utils import compute_statistics, evaluate_model


ProgressFn = Callable[[str], None]
LineSearchMethod = Literal["golden_section"]

# Sample cap for the function-preservation drift check; inputs are cached
# before growth so a shuffling loader cannot invalidate the comparison.
_PRESERVATION_CHECK_SAMPLES = 4096


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


def _function_preserving_growth(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    layer_index: int,
    device: torch.device,
    optimal_update_kwargs: dict[str, Any],
    preservation_tolerance: float,
    progress: ProgressFn | None,
) -> GrowthResult:
    """Grow one layer without changing the represented function.

    TINY statistics still select the incoming weights of the new neurons,
    but their outgoing weights are exactly zero and no delta touches the
    existing weights, so the committed function is unchanged: growth only
    refines the representation (enlarges the tangent image) and is not an
    optimization step. The measured output drift must stay within
    ``preservation_tolerance``.
    """
    criterion_sum = torch.nn.MSELoss(reduction="sum")
    criterion_mean = torch.nn.MSELoss(reduction="mean")

    model.eval()
    reference: list[tuple[torch.Tensor, torch.Tensor]] = []
    cached_samples = 0
    with torch.no_grad():
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            reference.append((batch_x, model(batch_x).detach().clone()))
            cached_samples += batch_x.shape[0]
            if cached_samples >= _PRESERVATION_CHECK_SAMPLES:
                break

    model.set_growing_layers(index=layer_index)
    compute_statistics(
        model,
        train_loader,
        loss_function=criterion_sum,
        device=device,
    )
    model.compute_optimal_updates(
        **{
            **optimal_update_kwargs,
            "compute_delta": False,
            "omega_zero": True,
        }
    )
    model.reset_computation()
    model.dummy_select_update()

    growing_layer = model.currently_updated_layer
    growing_layer.apply_change(
        apply_delta=False,
        apply_extension=True,
        input_extension_scaling=1.0,
        output_extension_scaling=1.0,
    )
    growing_layer.delete_update()
    model.currently_updated_layer_index = None

    model.eval()
    drift = 0.0
    with torch.no_grad():
        for batch_x, output_before in reference:
            batch_drift = float(
                torch.max(torch.abs(model(batch_x) - output_before)).item()
            )
            drift = max(drift, batch_drift)
    if not math.isfinite(drift) or drift > preservation_tolerance:
        raise RuntimeError(
            "Function-preserving growth exceeded its output tolerance: "
            f"{drift:.3e} > {preservation_tolerance:.3e}."
        )

    train_loss, _ = evaluate_model(
        model,
        train_loader,
        criterion_mean,
        use_extended_model=False,
        device=device,
    )
    point = LineSearchPoint(scaling_factor=1.0, train_loss=float(train_loss))
    if progress is not None:
        progress(
            f"  function-preserving growth: drift={drift:.3e}, "
            f"train_loss={point.train_loss:.4f}"
        )
    return GrowthResult(
        layer_index=layer_index,
        best_scaling_factor=1.0,
        best_train_loss=float(train_loss),
        line_search=[point],
    )


def growable_neuron_costs(
    model: GrowingMLP, input_features: int
) -> list[int]:
    """Parameter cost of ONE neuron added at each growable location.

    Growing ``_growable_layers[i]`` widens its *input* dimension, so each new
    neuron costs its incoming weights and bias in the preceding layer, plus
    its outgoing weights in this one::

        cost_i = fan_in_i + 1 + growable[i].out_features

    The spread is the whole point: on MNIST from 3x2 this is 787 parameters
    at the input projection against 5 and 13 later -- a factor of ~150 that
    an absolute singular-value threshold cannot see.
    """
    growable = list(getattr(model, "_growable_layers", []))
    costs: list[int] = []
    for index, layer in enumerate(growable):
        fan_in = (
            input_features if index == 0 else growable[index - 1].in_features
        )
        costs.append(int(fan_in) + 1 + int(layer.out_features))
    return costs


def expansion_spectrum(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    layer_index: int,
    device: torch.device,
    optimal_update_kwargs: dict[str, Any] | None = None,
) -> list[float]:
    """Per-neuron expansion scores at ``layer_index``: the ``s_i^2``.

    :func:`rank_layer_expansion_score` returns their sum, which is the
    location's total first-order loss decrease. This returns the individual
    terms, so candidate *neurons* can be compared across locations rather
    than whole layers -- the granularity at which a cost correction escapes
    the starvation that per-layer ranking produced (R2).

    Same cost as the ranking: one statistics pass and one SVD, no line
    search and no model clone. The model is left untouched.
    """
    model.set_growing_layers(index=layer_index)
    compute_statistics(
        model,
        train_loader,
        loss_function=torch.nn.MSELoss(reduction="sum"),
        device=device,
    )
    model.compute_optimal_updates(**(optimal_update_kwargs or {}))

    spectrum: list[float] = []
    for layer in getattr(model, "_growing_layers", []):
        eigenvalues = getattr(layer, "eigenvalues_extension", None)
        if eigenvalues is not None:
            spectrum.extend(float(value) ** 2 for value in eigenvalues)

    model.reset_computation()
    for layer in getattr(model, "_growing_layers", []):
        if hasattr(layer, "delete_update"):
            layer.delete_update(include_previous=True)
    model.currently_updated_layer_index = None
    model.zero_grad(set_to_none=True)
    return spectrum


def allocate_by_expansion_per_parameter(
    spectra: list[list[float]],
    costs: list[int],
    total_neurons: int,
) -> list[int]:
    """Spend ``total_neurons`` on the neurons that pay for themselves best.

    Every candidate neuron from every location is pooled and ranked by

        s_i^2 / cost(location)

    the certified first-order loss decrease per parameter it costs, and the
    budget is spent down that list. Returns how many neurons each location
    won.

    This is a *neuron*-level criterion evaluated with all locations pooled,
    which is what distinguishes it from R2. R2 ranked whole layers by
    decrease per parameter and therefore always bought the cheap late layer,
    starving the input projection until the run collapsed (784->2->2->14,
    64.4%). Here a location that holds one genuinely valuable direction
    still wins its slot even when its neurons are expensive, because the
    comparison is per candidate rather than per layer.

    The budget replaces a threshold: no tuned constant decides what "pays
    for itself" means, only how much is spent per growth event.
    """
    pooled: list[tuple[float, int]] = []
    for location, spectrum in enumerate(spectra):
        cost = max(costs[location], 1)
        pooled.extend((value / cost, location) for value in spectrum)
    pooled.sort(key=lambda item: -item[0])

    allocation = [0] * len(spectra)
    for value, location in pooled[:total_neurons]:
        if value <= 0.0:
            break
        allocation[location] += 1
    return allocation


def rank_layer_expansion_score(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    layer_index: int,
    device: torch.device,
    optimal_update_kwargs: dict[str, Any] | None = None,
) -> float:
    """SENN's natural expansion score for growing ``layer_index``.

    Returns ``sum(s_i^2)`` over the retained TINY singular values, which
    GroMo documents (``growing_module.py``) as the extension's first-order
    effect on the loss::

        L(A + dA) = L(A) - t * sigma'(0) * (eigenvalues_extension ** 2).sum()

    That first-order decrease is exactly SENN's expansion-score increase for
    this location (arXiv:2307.04526, Theorem 3.2), computed from the layer's
    Kronecker factors: ``tensor_s_growth()`` is the input activation second
    moment (KFAC's ``A``) and, with ``use_fisher=True`` in
    ``optimal_update_kwargs``, ``covariance_loss_gradient()`` supplies the
    output-side factor ``S``. Without that flag the score is TINY's, in the
    plain Euclidean output metric rather than SENN's Fisher one.

    The point of this helper is cost. It stops after the statistics pass and
    the SVD, so ranking L candidate layers costs L statistics passes instead
    of L * (1 + line_search.iterations) passes plus L model clones -- the
    golden-section search is then paid once, on the winner, inside
    :func:`grow_layer`. This is why SENN can afford to answer *where* from
    curvature instead of from trial growths.

    The model is left with its update tensors cleared, so a subsequent
    :func:`grow_layer` on the chosen layer starts from a clean state.
    """
    model.set_growing_layers(index=layer_index)
    compute_statistics(
        model,
        train_loader,
        loss_function=torch.nn.MSELoss(reduction="sum"),
        device=device,
    )
    model.compute_optimal_updates(**(optimal_update_kwargs or {}))

    score = 0.0
    for layer in getattr(model, "_growing_layers", []):
        eigenvalues = getattr(layer, "eigenvalues_extension", None)
        if eigenvalues is not None:
            score += float(eigenvalues.pow(2).sum())

    model.reset_computation()
    for layer in getattr(model, "_growing_layers", []):
        if hasattr(layer, "delete_update"):
            layer.delete_update(include_previous=True)
    model.currently_updated_layer_index = None
    model.zero_grad(set_to_none=True)
    return score


def grow_layer(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    layer_index: int,
    device: torch.device,
    line_search_config: ScalingLineSearchConfig,
    optimal_update_kwargs: dict[str, Any] | None = None,
    progress: ProgressFn | None = None,
    function_preserving: bool = False,
    preservation_tolerance: float = 1e-6,
    line_search_loader: torch.utils.data.DataLoader | None = None,
) -> GrowthResult:
    """Grow one GroMo layer and apply the best line-search update.

    ``layer_index`` follows GroMo's local API: it is zero-based over
    ``model._growable_layers``. With ``function_preserving=True`` the
    scaling line search is skipped and the extension is applied with zero
    outgoing weights, leaving the represented function exactly unchanged.

    ``line_search_loader`` selects the data the scaling factor is chosen on.
    The GroMo default minimizes the TRAIN loss, which makes the magnitude of
    the structural step an uncertified, train-fitting choice; passing the
    held-out loader instead makes the growth's magnitude follow the same
    held-out functional descent that Proposition 3.8 certifies for every
    other step.
    """
    if function_preserving:
        return _function_preserving_growth(
            model=model,
            train_loader=train_loader,
            layer_index=layer_index,
            device=device,
            optimal_update_kwargs=dict(optimal_update_kwargs or {}),
            preservation_tolerance=preservation_tolerance,
            progress=progress,
        )

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
        train_loader=(
            line_search_loader if line_search_loader is not None else train_loader
        ),
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
