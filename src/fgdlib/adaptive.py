"""Strict adaptive FGD over the finite empirical output space.

This module implements Algorithm 1 of arXiv:2606.16926 for the empirical
Hilbert space

    H = B = R^(n x m),  <u, v> = (1/n) sum_i <u_i, v_i>.

For ``L(f) = 1/2 ||f-y||_B^2`` the functional gradient is ``f-y`` and the
theory constants are derived, not configured: ``K = alpha = beta = mu = 1``
and ``L* = 0``.  Parameter-space procedures are only proposal generators.
Every accepted direction is the *finite secant actually realised* by the
candidate network, measured on the complete empirical training set.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.func import functional_call, jacrev, jvp

from fgdlib.tangent import (
    _clear_inaccessible_tensor_caches,
    _conjugate_gradient,
)


ApproximationFamily = Literal[
    "head_closed_form",
    "scaled_parameter_gradient",
    "tangent_least_squares",
    "nonlinear_secant",
]


@dataclass(frozen=True)
class AdaptiveFGDConfig:
    """Configuration of strict empirical adaptive FGD.

    The functional-space constants deliberately do not appear here: the
    empirical MSE backend derives K = alpha = beta = mu = 1.
    """

    epsilon: float = 0.25
    learning_rate_initial: float = 0.5
    learning_rate_backtrack: float = 0.5
    learning_rate_min: float = 1e-6
    learning_rate_trials: int = 7
    screening_points: int | None = 256
    family_order: tuple[str, ...] = (
        "head_closed_form",
        "tangent_least_squares",
    )
    tangent_damping: tuple[float, ...] = (1e-1, 1e-3)
    tangent_cg_iterations: tuple[int, ...] = (64,)
    tangent_cg_tolerance: float = 1e-8
    exact_jacobian_max_elements: int = 2_000_000
    nonlinear_steps: tuple[int, ...] = (16, 64)
    nonlinear_learning_rate: float = 1e-2
    computation_batch_size: int = 1024
    certificate_margin: float = 1e-10
    numerical_tolerance: float = 1e-7
    gradient_tolerance: float = 1e-12
    preservation_tolerance: float = 1e-7
    max_growth_events: int = 32
    max_hidden_size: int | None = None
    growth_neurons_per_event: int = 1

    def validate(self) -> None:
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("fgd_adaptive.epsilon must lie in (0, 1).")
        if self.learning_rate_initial <= 0.0:
            raise ValueError("fgd_adaptive.learning_rate_initial must be positive.")
        if not 0.0 < self.learning_rate_backtrack < 1.0:
            raise ValueError("fgd_adaptive.learning_rate_backtrack must lie in (0, 1).")
        if self.learning_rate_min <= 0.0:
            raise ValueError("fgd_adaptive.learning_rate_min must be positive.")
        if self.learning_rate_initial < self.learning_rate_min:
            raise ValueError(
                "fgd_adaptive.learning_rate_initial must be >= learning_rate_min."
            )
        if self.learning_rate_trials < 1:
            raise ValueError("fgd_adaptive.learning_rate_trials must be >= 1.")
        if self.screening_points is not None and self.screening_points < 1:
            raise ValueError("fgd_adaptive.screening_points must be positive.")
        supported = {
            "head_closed_form",
            "scaled_parameter_gradient",
            "tangent_least_squares",
            "nonlinear_secant",
        }
        unknown = sorted(set(self.family_order) - supported)
        if unknown:
            raise ValueError("Unsupported fgd_adaptive family: " + ", ".join(unknown))
        if not self.family_order:
            raise ValueError("fgd_adaptive.family_order cannot be empty.")
        if any(value < 0.0 for value in self.tangent_damping):
            raise ValueError("fgd_adaptive.tangent_damping must be non-negative.")
        if any(value < 1 for value in self.tangent_cg_iterations):
            raise ValueError(
                "fgd_adaptive.tangent_cg_iterations must contain positive values."
            )
        if self.tangent_cg_tolerance <= 0.0:
            raise ValueError("fgd_adaptive.tangent_cg_tolerance must be positive.")
        if self.exact_jacobian_max_elements < 0:
            raise ValueError(
                "fgd_adaptive.exact_jacobian_max_elements must be non-negative."
            )
        if any(value < 1 for value in self.nonlinear_steps):
            raise ValueError(
                "fgd_adaptive.nonlinear_steps must contain positive values."
            )
        if self.nonlinear_learning_rate <= 0.0:
            raise ValueError("fgd_adaptive.nonlinear_learning_rate must be positive.")
        if self.computation_batch_size < 1:
            raise ValueError("fgd_adaptive.computation_batch_size must be positive.")
        if any(
            value < 0.0
            for value in (
                self.certificate_margin,
                self.numerical_tolerance,
                self.gradient_tolerance,
                self.preservation_tolerance,
            )
        ):
            raise ValueError("fgd_adaptive tolerances must be non-negative.")
        if self.max_growth_events < 0:
            raise ValueError("fgd_adaptive.max_growth_events must be non-negative.")
        if self.max_hidden_size is not None and self.max_hidden_size < 1:
            raise ValueError("fgd_adaptive.max_hidden_size must be positive when set.")
        if self.growth_neurons_per_event < 1:
            raise ValueError("fgd_adaptive.growth_neurons_per_event must be positive.")


@dataclass(frozen=True)
class AdaptiveFGDCertificate:
    """All theoretical and diagnostic quantities for one finite secant."""

    learning_rate: float
    error_upper_bound: float
    approximation_norm: float
    target_norm: float
    relative_error: float
    directional_cosine: float | None
    algorithm_margin: float
    relative_error_valid: bool
    learning_rate_upper_bound: float | None
    learning_rate_valid: bool
    descent_coefficient: float | None
    contraction: float | None
    loss_before: float
    loss_after: float
    predicted_loss_upper_bound: float | None
    sufficient_descent_valid: bool
    accepted: bool
    rejection_reason: str | None
    smoothness: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    mu: float = 1.0
    loss_star: float = 0.0


@dataclass(frozen=True)
class AdaptiveFGDAttemptRecord:
    """Persistent record of an accepted or rejected approximation attempt."""

    step: int
    attempt: int
    family: str
    subspace: str
    solver: str | None
    damping: float | None
    solver_iterations: int | None
    certificate: AdaptiveFGDCertificate
    growth_layer_index: int | None = None
    certificate_scope: str = "full_train"
    certificate_points: int | None = None


@dataclass(frozen=True)
class AdaptiveFGDSearchResult:
    """Result of the representation ladder at one fixed functional iterate."""

    model: nn.Module | None
    certificate: AdaptiveFGDCertificate | None
    attempts: list[AdaptiveFGDAttemptRecord]
    converged: bool
    best_relative_error: float | None


@dataclass(frozen=True)
class AdaptiveGrowthResult:
    """A function-preserving representation refinement."""

    layer_index: int
    added_parameters: int
    output_drift: float
    certified_candidate_available: bool
    best_relative_error: float | None


@dataclass(frozen=True)
class _ParameterSubspace:
    name: str
    parameter_names: tuple[str, ...]


def empirical_inner_product(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return ``(1/n) sum_i <left_i, right_i>`` in float64."""
    if left.shape != right.shape or left.ndim < 1:
        raise ValueError("Empirical vectors must have equal non-scalar shapes.")
    n = left.shape[0]
    return float(torch.sum(left.to(torch.float64) * right.to(torch.float64)).item() / n)


def empirical_sq_norm(value: torch.Tensor) -> float:
    return empirical_inner_product(value, value)


def empirical_norm(value: torch.Tensor) -> float:
    return math.sqrt(max(empirical_sq_norm(value), 0.0))


def empirical_functional_loss(output: torch.Tensor, target: torch.Tensor) -> float:
    """Return ``1/(2n) sum_i ||output_i-target_i||^2``."""
    return 0.5 * empirical_sq_norm(output - target)


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def _iter_tensor_batches(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
):
    """Yield a fixed empirical design in bounded device-sized chunks."""
    if x.ndim < 1 or y.ndim < 1 or x.shape[0] != y.shape[0]:
        raise ValueError("Empirical inputs and targets need equal non-zero axes.")
    if x.shape[0] == 0:
        raise ValueError("The empirical training design cannot be empty.")
    if batch_size < 1:
        raise ValueError("The empirical computation batch size must be positive.")
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        yield (
            x[start:stop].to(device=device, non_blocking=True),
            y[start:stop].to(device=device, non_blocking=True),
        )


def empirical_model_functional_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
) -> float:
    """Measure empirical functional loss without a full-train forward."""
    model.eval()
    device = _model_device(model)
    squared_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in _iter_tensor_batches(
            x,
            y,
            batch_size=batch_size,
            device=device,
        ):
            residual = model(batch_x).to(torch.float64) - batch_y.to(torch.float64)
            if not bool(torch.isfinite(residual).all()):
                return float("nan")
            squared_sum += float(torch.sum(residual.square()).item())
            count += batch_x.shape[0]
    return 0.5 * squared_sum / count


def theory_learning_rate_upper_bound(relative_error: float) -> float | None:
    """Prop. 3.8 LR ceiling for K = alpha = beta = 1."""
    if not math.isfinite(relative_error) or not 0.0 <= relative_error < 0.5:
        return None
    numerator = 2.0 * (1.0 - 2.0 * relative_error)
    denominator = 2.0 * relative_error + 1.0
    return numerator / denominator


def theory_descent_coefficient(
    relative_error: float,
    learning_rate: float,
) -> float | None:
    """Lemma 3.5 coefficient for K = alpha = beta = 1."""
    if (
        not math.isfinite(relative_error)
        or not 0.0 <= relative_error < 1.0
        or learning_rate <= 0.0
    ):
        return None
    ratio = relative_error / (1.0 - relative_error)
    return 1.0 - 0.5 * learning_rate - (1.0 + 1.5 * learning_rate) * ratio


def _certificate_from_measurements(
    *,
    learning_rate: float,
    epsilon: float,
    error_bound: float,
    approximation_norm: float,
    target_norm: float,
    approximation_target_inner_product: float,
    loss_before: float,
    loss_after: float,
    certificate_margin: float,
    numerical_tolerance: float,
) -> AdaptiveFGDCertificate:
    """Apply the strict certificate to globally accumulated measurements."""
    if approximation_norm == 0.0:
        relative_error = 0.0 if target_norm == 0.0 else float("inf")
    else:
        relative_error = error_bound / approximation_norm

    if approximation_norm > 0.0 and target_norm > 0.0:
        cosine = approximation_target_inner_product / (approximation_norm * target_norm)
        cosine = min(max(cosine, -1.0), 1.0)
    else:
        cosine = None

    algorithm_margin = epsilon * approximation_norm - (1.0 + epsilon) * error_bound
    margin_scale = 1.0 + epsilon * approximation_norm + error_bound
    relative_error_valid = (
        math.isfinite(relative_error)
        and algorithm_margin > certificate_margin * margin_scale
    )

    lr_upper = theory_learning_rate_upper_bound(relative_error)
    learning_rate_valid = (
        lr_upper is not None
        and learning_rate < lr_upper - certificate_margin * (1.0 + abs(lr_upper))
    )
    descent = theory_descent_coefficient(relative_error, learning_rate)
    contraction = 1.0 - 2.0 * learning_rate * descent if descent is not None else None
    contraction_valid = (
        descent is not None
        and descent > 0.0
        and contraction is not None
        and 0.0 <= contraction < 1.0
    )

    predicted_loss_upper = (
        loss_before - learning_rate * descent * target_norm**2
        if descent is not None
        else None
    )
    loss_tolerance = numerical_tolerance * (1.0 + abs(loss_before))
    sufficient_descent_valid = (
        predicted_loss_upper is not None
        and loss_after <= predicted_loss_upper + loss_tolerance
    )

    reason: str | None = None
    if approximation_norm == 0.0 and target_norm > 0.0:
        reason = "zero_approximation"
    elif not relative_error_valid:
        reason = "algorithm1_relative_error"
    elif not learning_rate_valid:
        reason = "learning_rate_interval"
    elif descent is None or descent <= 0.0:
        reason = "nonpositive_descent_coefficient"
    elif not contraction_valid:
        reason = "invalid_contraction"
    elif not sufficient_descent_valid:
        reason = "sufficient_descent_violation"

    return AdaptiveFGDCertificate(
        learning_rate=learning_rate,
        error_upper_bound=error_bound,
        approximation_norm=approximation_norm,
        target_norm=target_norm,
        relative_error=relative_error,
        directional_cosine=cosine,
        algorithm_margin=algorithm_margin,
        relative_error_valid=relative_error_valid,
        learning_rate_upper_bound=lr_upper,
        learning_rate_valid=learning_rate_valid,
        descent_coefficient=descent,
        contraction=contraction,
        loss_before=loss_before,
        loss_after=loss_after,
        predicted_loss_upper_bound=predicted_loss_upper,
        sufficient_descent_valid=sufficient_descent_valid,
        accepted=reason is None,
        rejection_reason=reason,
    )


def certify_empirical_secant(
    *,
    base_output: torch.Tensor,
    candidate_output: torch.Tensor,
    target: torch.Tensor,
    learning_rate: float,
    epsilon: float,
    certificate_margin: float = 1e-10,
    numerical_tolerance: float = 1e-7,
) -> AdaptiveFGDCertificate:
    """Certify the exact realised secant before it can be committed.

    ``target`` is the supervised target ``y``.  The functional gradient is
    therefore ``base_output - target`` and the approximation is the exact
    finite secant ``(base_output - candidate_output) / eta``.
    """
    if learning_rate <= 0.0:
        raise ValueError("An adaptive FGD candidate needs a positive learning rate.")

    values_finite = (
        torch.isfinite(base_output).all()
        and torch.isfinite(candidate_output).all()
        and torch.isfinite(target).all()
    )
    if not bool(values_finite):
        return AdaptiveFGDCertificate(
            learning_rate=learning_rate,
            error_upper_bound=float("inf"),
            approximation_norm=float("nan"),
            target_norm=float("nan"),
            relative_error=float("inf"),
            directional_cosine=None,
            algorithm_margin=float("-inf"),
            relative_error_valid=False,
            learning_rate_upper_bound=None,
            learning_rate_valid=False,
            descent_coefficient=None,
            contraction=None,
            loss_before=float("nan"),
            loss_after=float("nan"),
            predicted_loss_upper_bound=None,
            sufficient_descent_valid=False,
            accepted=False,
            rejection_reason="non_finite_values",
        )

    functional_gradient = base_output.to(torch.float64) - target.to(torch.float64)
    approximation = (
        base_output.to(torch.float64) - candidate_output.to(torch.float64)
    ) / learning_rate
    error = approximation - functional_gradient
    error_bound = empirical_norm(error)  # equality: this U is exact on finite B
    approximation_norm = empirical_norm(approximation)
    target_norm = empirical_norm(functional_gradient)

    loss_before = empirical_functional_loss(base_output, target)
    loss_after = empirical_functional_loss(candidate_output, target)
    return _certificate_from_measurements(
        learning_rate=learning_rate,
        epsilon=epsilon,
        error_bound=error_bound,
        approximation_norm=approximation_norm,
        target_norm=target_norm,
        approximation_target_inner_product=empirical_inner_product(
            approximation,
            functional_gradient,
        ),
        loss_before=loss_before,
        loss_after=loss_after,
        certificate_margin=certificate_margin,
        numerical_tolerance=numerical_tolerance,
    )


def _functional_gradient_measurements(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[float, float]:
    """Return full-design gradient norm and loss via batch accumulation."""
    model.eval()
    device = _model_device(model)
    gradient_sq_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in _iter_tensor_batches(
            x,
            y,
            batch_size=batch_size,
            device=device,
        ):
            gradient = model(batch_x).to(torch.float64) - batch_y.to(torch.float64)
            if not bool(torch.isfinite(gradient).all()):
                return float("nan"), float("nan")
            gradient_sq_sum += float(torch.sum(gradient.square()).item())
            count += batch_x.shape[0]
    squared_norm = gradient_sq_sum / count
    return math.sqrt(max(squared_norm, 0.0)), 0.5 * squared_norm


def certify_empirical_secant_models(
    *,
    base_model: nn.Module,
    candidate_model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float,
    epsilon: float,
    batch_size: int,
    certificate_margin: float = 1e-10,
    numerical_tolerance: float = 1e-7,
) -> AdaptiveFGDCertificate:
    """Certify a realised secant over all samples using bounded batches."""
    if learning_rate <= 0.0:
        raise ValueError("An adaptive FGD candidate needs a positive learning rate.")
    base_model.eval()
    candidate_model.eval()
    device = _model_device(base_model)
    if _model_device(candidate_model) != device:
        raise ValueError("Base and candidate models must be on the same device.")

    error_sq_sum = 0.0
    approximation_sq_sum = 0.0
    gradient_sq_sum = 0.0
    approximation_gradient_sum = 0.0
    candidate_residual_sq_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in _iter_tensor_batches(
            x,
            y,
            batch_size=batch_size,
            device=device,
        ):
            base_output = base_model(batch_x).to(torch.float64)
            candidate_output = candidate_model(batch_x).to(torch.float64)
            target = batch_y.to(torch.float64)
            if (
                base_output.shape != candidate_output.shape
                or base_output.shape != target.shape
            ):
                raise ValueError(
                    "Model outputs and empirical targets must have equal shapes."
                )
            if not bool(
                torch.isfinite(base_output).all()
                and torch.isfinite(candidate_output).all()
                and torch.isfinite(target).all()
            ):
                return _failed_certificate(
                    learning_rate,
                    "non_finite_values",
                    loss_before=float("nan"),
                    target_norm=float("nan"),
                )

            functional_gradient = base_output - target
            approximation = (base_output - candidate_output) / learning_rate
            error = approximation - functional_gradient
            candidate_residual = candidate_output - target
            error_sq_sum += float(torch.sum(error.square()).item())
            approximation_sq_sum += float(torch.sum(approximation.square()).item())
            gradient_sq_sum += float(torch.sum(functional_gradient.square()).item())
            approximation_gradient_sum += float(
                torch.sum(approximation * functional_gradient).item()
            )
            candidate_residual_sq_sum += float(
                torch.sum(candidate_residual.square()).item()
            )
            count += batch_x.shape[0]

    inverse_count = 1.0 / count
    error_bound = math.sqrt(max(error_sq_sum * inverse_count, 0.0))
    approximation_norm = math.sqrt(max(approximation_sq_sum * inverse_count, 0.0))
    target_norm = math.sqrt(max(gradient_sq_sum * inverse_count, 0.0))
    return _certificate_from_measurements(
        learning_rate=learning_rate,
        epsilon=epsilon,
        error_bound=error_bound,
        approximation_norm=approximation_norm,
        target_norm=target_norm,
        approximation_target_inner_product=(approximation_gradient_sum * inverse_count),
        loss_before=0.5 * gradient_sq_sum * inverse_count,
        loss_after=0.5 * candidate_residual_sq_sum * inverse_count,
        certificate_margin=certificate_margin,
        numerical_tolerance=numerical_tolerance,
    )


def _named_trainable_parameters(model: nn.Module) -> OrderedDict[str, nn.Parameter]:
    return OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def parameter_subspaces(model: nn.Module) -> list[_ParameterSubspace]:
    """Return nested suffix spaces: head, progressively earlier blocks, all."""
    named = _named_trainable_parameters(model)
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) == 0:
        return [_ParameterSubspace("all", tuple(named))]

    spaces: list[_ParameterSubspace] = []
    last = len(layers) - 1
    for start in range(last, -1, -1):
        names = tuple(
            name
            for name in named
            if any(
                name.startswith(f"layers.{index}.") for index in range(start, last + 1)
            )
        )
        if not names:
            continue
        label = "head" if start == last else f"suffix_from_layer_{start}"
        if names == tuple(named):
            label = "all"
        if not spaces or names != spaces[-1].parameter_names:
            spaces.append(_ParameterSubspace(label, names))
    if not spaces or spaces[-1].parameter_names != tuple(named):
        spaces.append(_ParameterSubspace("all", tuple(named)))
    return spaces


def _learning_rates(config: AdaptiveFGDConfig) -> list[float]:
    rates: list[float] = []
    value = config.learning_rate_initial
    for index in range(config.learning_rate_trials):
        if value < config.learning_rate_min:
            break
        if (
            index == config.learning_rate_trials - 1
            and value > config.learning_rate_min
        ):
            value = config.learning_rate_min
        rates.append(value)
        if value <= config.learning_rate_min:
            break
        value *= config.learning_rate_backtrack
    return rates


def _clone_model(model: nn.Module) -> nn.Module:
    _clear_inaccessible_tensor_caches(model)
    return copy.deepcopy(model)


def _apply_direction(
    model: nn.Module,
    direction: dict[str, torch.Tensor],
    learning_rate: float,
) -> nn.Module:
    candidate = _clone_model(model)
    named = dict(candidate.named_parameters())
    with torch.no_grad():
        for name, update in direction.items():
            named[name].add_(update.to(named[name]), alpha=-learning_rate)
    return candidate


def _head_closed_form_candidate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float,
    batch_size: int,
) -> nn.Module:
    """Fit the affine head from streamed sufficient statistics."""
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("Closed-form head fitting requires model.layers.")
    candidate = _clone_model(model)
    candidate.eval()
    model.eval()
    device = _model_device(model)
    gram: torch.Tensor | None = None
    cross: torch.Tensor | None = None
    with torch.no_grad():
        for batch_x, batch_y in _iter_tensor_batches(
            x,
            y,
            batch_size=batch_size,
            device=device,
        ):
            base_output = model(batch_x)
            functional_target = base_output - learning_rate * (base_output - batch_y)
            features = (
                candidate.flatten(batch_x) if hasattr(candidate, "flatten") else batch_x
            )
            for layer in candidate.layers[:-1]:
                features = layer(features)
            design = torch.cat(
                [
                    features.to(torch.float64),
                    torch.ones(
                        features.shape[0],
                        1,
                        device=features.device,
                        dtype=torch.float64,
                    ),
                ],
                dim=1,
            )
            target_work = functional_target.to(torch.float64)
            batch_gram = design.T @ design
            batch_cross = design.T @ target_work
            gram = batch_gram if gram is None else gram + batch_gram
            cross = batch_cross if cross is None else cross + batch_cross
        if gram is None or cross is None:
            raise RuntimeError("Closed-form head fitting received no samples.")
        gram = 0.5 * (gram + gram.T)
        solution = torch.linalg.lstsq(gram, cross).solution
        head = candidate.layers[-1]
        linear = getattr(head, "layer", head)
        linear.weight.copy_(solution[:-1].T.to(linear.weight))
        if linear.bias is None:
            if not torch.allclose(solution[-1], torch.zeros_like(solution[-1])):
                raise RuntimeError("The closed-form target needs a head bias.")
        else:
            linear.bias.copy_(solution[-1].to(linear.bias))
    return candidate


def _functional_call_for_subspace(
    model: nn.Module,
    x: torch.Tensor,
    subspace: _ParameterSubspace,
):
    all_parameters = OrderedDict(model.named_parameters())
    selected = tuple(all_parameters[name] for name in subspace.parameter_names)
    fixed = OrderedDict(
        (name, parameter)
        for name, parameter in all_parameters.items()
        if name not in subspace.parameter_names
    )
    buffers = OrderedDict(model.named_buffers())

    def call(parameter_values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        state: OrderedDict[str, torch.Tensor] = OrderedDict(fixed)
        state.update(zip(subspace.parameter_names, parameter_values))
        state.update(buffers)
        return functional_call(model, state, (x,))

    return selected, call


def _flatten_tensors(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _unflatten_tensor(
    vector: torch.Tensor,
    references: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    result: list[torch.Tensor] = []
    offset = 0
    for reference in references:
        size = reference.numel()
        result.append(vector[offset : offset + size].reshape_as(reference))
        offset += size
    return tuple(result)


def _scaled_parameter_gradient_direction(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    subspace: _ParameterSubspace,
    eps: float,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Return the calibrated parameter gradient accumulated over all batches."""
    model.eval()
    device = _model_device(model)
    named = OrderedDict(model.named_parameters())
    selected = tuple(named[name] for name in subspace.parameter_names)
    direction = tuple(torch.zeros_like(parameter) for parameter in selected)
    n = x.shape[0]

    for batch_x, batch_y in _iter_tensor_batches(
        x,
        y,
        batch_size=batch_size,
        device=device,
    ):
        output = model(batch_x)
        residual = output - batch_y
        objective = 0.5 * torch.sum(residual.square()) / n
        gradients = torch.autograd.grad(objective, selected, allow_unused=True)
        direction = tuple(
            accumulated
            + (torch.zeros_like(parameter) if gradient is None else gradient.detach())
            for accumulated, parameter, gradient in zip(
                direction,
                selected,
                gradients,
            )
        )

    dot_sum = 0.0
    sq_norm_sum = 0.0
    try:
        for batch_x, batch_y in _iter_tensor_batches(
            x,
            y,
            batch_size=batch_size,
            device=device,
        ):
            parameters, call = _functional_call_for_subspace(
                model,
                batch_x,
                subspace,
            )
            output = model(batch_x)
            residual = output - batch_y
            _, output_direction = jvp(call, (parameters,), (direction,))
            dot_sum += float(
                torch.sum(
                    output_direction.to(torch.float64)
                    * residual.detach().to(torch.float64)
                ).item()
            )
            sq_norm_sum += float(
                torch.sum(output_direction.to(torch.float64).square()).item()
            )
    finally:
        _clear_inaccessible_tensor_caches(model)
    dot = dot_sum / n
    sq_norm = sq_norm_sum / n
    if sq_norm <= eps or dot <= eps:
        raise RuntimeError("Projected parameter gradient is not a descent direction.")
    scale = dot / sq_norm
    return {
        name: scale * update
        for name, update in zip(subspace.parameter_names, direction)
    }


def _exact_tangent_direction(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    subspace: _ParameterSubspace,
    damping: float,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Accumulate exact parameter Gram statistics one batch at a time."""
    model.eval()
    device = _model_device(model)
    named = OrderedDict(model.named_parameters())
    selected = tuple(named[name] for name in subspace.parameter_names)
    parameter_count = sum(parameter.numel() for parameter in selected)
    gram = torch.zeros(
        parameter_count,
        parameter_count,
        dtype=torch.float64,
        device=device,
    )
    rhs = torch.zeros(parameter_count, dtype=torch.float64, device=device)
    try:
        for batch_x, batch_y in _iter_tensor_batches(
            x,
            y,
            batch_size=batch_size,
            device=device,
        ):
            parameters, call = _functional_call_for_subspace(
                model,
                batch_x,
                subspace,
            )
            output = model(batch_x)
            functional_gradient = (output - batch_y).detach().reshape(-1)
            jacobian = jacrev(call)(parameters)
            matrix = (
                torch.cat(
                    [
                        block.reshape(functional_gradient.numel(), -1)
                        for block in jacobian
                    ],
                    dim=1,
                )
                .detach()
                .to(torch.float64)
            )
            target = functional_gradient.to(torch.float64)
            gram.add_(matrix.T @ matrix)
            rhs.add_(matrix.T @ target)
    finally:
        _clear_inaccessible_tensor_caches(model)

    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0.0)
    coefficients = eigenvectors.T @ rhs
    if damping > 0.0:
        inverse = (eigenvalues + damping).reciprocal()
    else:
        maximum = max(float(eigenvalues.max().item()), 1.0)
        threshold = torch.finfo(torch.float64).eps * parameter_count * maximum
        inverse = torch.where(
            eigenvalues > threshold,
            eigenvalues.reciprocal(),
            torch.zeros_like(eigenvalues),
        )
    flat = (eigenvectors @ (inverse * coefficients)).to(selected[0])
    updates = _unflatten_tensor(flat, selected)
    return dict(zip(subspace.parameter_names, updates))


def _cg_tangent_direction(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    subspace: _ParameterSubspace,
    damping: float,
    iterations: int,
    tolerance: float,
    eps: float,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Solve the tangent normal equations without retaining full-train graphs."""
    model.eval()
    device = _model_device(model)
    named = OrderedDict(model.named_parameters())
    selected = tuple(named[name] for name in subspace.parameter_names)

    def output_vjp(
        output: torch.Tensor,
        parameters: tuple[torch.Tensor, ...],
        vector: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        gradients = torch.autograd.grad(
            output,
            parameters,
            grad_outputs=vector,
            allow_unused=True,
        )
        return tuple(
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for parameter, gradient in zip(parameters, gradients)
        )

    rhs = torch.zeros(
        sum(parameter.numel() for parameter in selected),
        device=device,
        dtype=selected[0].dtype,
    )
    for batch_x, batch_y in _iter_tensor_batches(
        x,
        y,
        batch_size=batch_size,
        device=device,
    ):
        parameters, call = _functional_call_for_subspace(
            model,
            batch_x,
            subspace,
        )
        output = call(parameters)
        target_chunk = (output - batch_y).detach()
        rhs = rhs + _flatten_tensors(output_vjp(output, parameters, target_chunk))

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        product = torch.zeros_like(vector)
        for batch_x, _ in _iter_tensor_batches(
            x,
            y,
            batch_size=batch_size,
            device=device,
        ):
            parameters, call = _functional_call_for_subspace(
                model,
                batch_x,
                subspace,
            )
            output = call(parameters)
            parameter_vector = _unflatten_tensor(vector, parameters)
            _, tangent = jvp(call, (parameters,), (parameter_vector,))
            product = product + _flatten_tensors(
                output_vjp(output, parameters, tangent.detach())
            )
        if damping > 0.0:
            product = product + damping * vector
        return product

    try:
        flat = _conjugate_gradient(
            matvec,
            rhs,
            max_iterations=iterations,
            tolerance=tolerance,
            eps=eps,
        )
    finally:
        _clear_inaccessible_tensor_caches(model)
    updates = _unflatten_tensor(flat, selected)
    return dict(zip(subspace.parameter_names, updates))


def _nonlinear_secant_candidate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    subspace: _ParameterSubspace,
    *,
    functional_learning_rate: float,
    steps: int,
    learning_rate: float,
    batch_size: int,
) -> nn.Module:
    candidate = _clone_model(model)
    named = OrderedDict(candidate.named_parameters())
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in named.items()
    }
    selected_names = set(subspace.parameter_names)
    for name, parameter in named.items():
        parameter.requires_grad_(name in selected_names)
    selected = [named[name] for name in subspace.parameter_names]
    optimizer = torch.optim.Adam(selected, lr=learning_rate)
    n = x.shape[0]
    device = _model_device(model)
    model.eval()
    candidate.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for batch_x, batch_y in _iter_tensor_batches(
            x,
            y,
            batch_size=batch_size,
            device=device,
        ):
            with torch.no_grad():
                base_output = model(batch_x)
                functional_target = base_output - functional_learning_rate * (
                    base_output - batch_y
                )
            output = candidate(batch_x)
            difference = output - functional_target
            objective = 0.5 * torch.sum(difference.square()) / n
            if not torch.isfinite(objective):
                raise RuntimeError("Non-finite nonlinear secant objective.")
            objective.backward()
        optimizer.step()
    for name, parameter in named.items():
        parameter.requires_grad_(original_requires_grad[name])
        parameter.grad = None
    candidate.eval()
    return candidate


def _failed_certificate(
    learning_rate: float,
    reason: str,
    *,
    loss_before: float,
    target_norm: float,
) -> AdaptiveFGDCertificate:
    return AdaptiveFGDCertificate(
        learning_rate=learning_rate,
        error_upper_bound=float("inf"),
        approximation_norm=float("nan"),
        target_norm=target_norm,
        relative_error=float("inf"),
        directional_cosine=None,
        algorithm_margin=float("-inf"),
        relative_error_valid=False,
        learning_rate_upper_bound=None,
        learning_rate_valid=False,
        descent_coefficient=None,
        contraction=None,
        loss_before=loss_before,
        loss_after=float("nan"),
        predicted_loss_upper_bound=None,
        sufficient_descent_valid=False,
        accepted=False,
        rejection_reason=reason,
    )


def search_adaptive_fgd_step(
    *,
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    config: AdaptiveFGDConfig,
    step: int,
    progress: Callable[[str], None] | None = None,
) -> AdaptiveFGDSearchResult:
    """Try the representation ladder using full-design batch reductions."""
    config.validate()
    model.eval()
    target_norm, base_loss = _functional_gradient_measurements(
        model,
        train_x,
        train_y,
        batch_size=config.computation_batch_size,
    )
    if not math.isfinite(target_norm) or not math.isfinite(base_loss):
        raise RuntimeError("Non-finite functional gradient on the training design.")
    if target_norm <= config.gradient_tolerance:
        return AdaptiveFGDSearchResult(None, None, [], True, 0.0)

    rates = _learning_rates(config)
    spaces = parameter_subspaces(model)
    screening_count = min(
        train_x.shape[0],
        config.screening_points or train_x.shape[0],
    )
    proposal_x = train_x[:screening_count]
    proposal_y = train_y[:screening_count]
    uses_screening = screening_count < train_x.shape[0]
    attempts: list[AdaptiveFGDAttemptRecord] = []
    attempt_index = 0
    best_relative_error: float | None = None

    def evaluate(
        candidate: nn.Module,
        *,
        family: str,
        subspace: str,
        eta: float,
        solver: str | None = None,
        damping: float | None = None,
        iterations: int | None = None,
    ) -> AdaptiveFGDSearchResult | None:
        nonlocal attempt_index, best_relative_error
        candidate.eval()

        def measure_and_record(
            measure_x: torch.Tensor,
            measure_y: torch.Tensor,
            *,
            scope: str,
        ) -> AdaptiveFGDCertificate:
            nonlocal attempt_index, best_relative_error
            try:
                certificate = certify_empirical_secant_models(
                    base_model=model,
                    candidate_model=candidate,
                    x=measure_x,
                    y=measure_y,
                    learning_rate=eta,
                    epsilon=config.epsilon,
                    batch_size=config.computation_batch_size,
                    certificate_margin=config.certificate_margin,
                    numerical_tolerance=config.numerical_tolerance,
                )
            except Exception as error:
                certificate = _failed_certificate(
                    eta,
                    f"certificate_failure:{type(error).__name__}",
                    loss_before=base_loss,
                    target_norm=target_norm,
                )
            attempt_index += 1
            attempts.append(
                AdaptiveFGDAttemptRecord(
                    step=step,
                    attempt=attempt_index,
                    family=family,
                    subspace=subspace,
                    solver=solver,
                    damping=damping,
                    solver_iterations=iterations,
                    certificate=certificate,
                    certificate_scope=scope,
                    certificate_points=measure_x.shape[0],
                )
            )
            if math.isfinite(certificate.relative_error):
                best_relative_error = (
                    certificate.relative_error
                    if best_relative_error is None
                    else min(best_relative_error, certificate.relative_error)
                )
            return certificate

        if uses_screening:
            screening_certificate = measure_and_record(
                proposal_x,
                proposal_y,
                scope="screening",
            )
            if not screening_certificate.accepted:
                return None
            if progress is not None:
                progress(
                    f"[ADAPT] step {step}: promoting {family}/{subspace} "
                    f"eta={eta:.4g}, screening_q="
                    f"{screening_certificate.relative_error:.4f} to full train"
                )

        certificate = measure_and_record(
            train_x,
            train_y,
            scope="full_train",
        )
        if certificate.accepted:
            return AdaptiveFGDSearchResult(
                candidate,
                certificate,
                attempts,
                False,
                best_relative_error,
            )
        return None

    def record_generation_failure(
        *,
        family: str,
        subspace: str,
        eta: float,
        reason: str,
        solver: str | None = None,
        damping: float | None = None,
        iterations: int | None = None,
    ) -> None:
        nonlocal attempt_index
        attempt_index += 1
        attempts.append(
            AdaptiveFGDAttemptRecord(
                step=step,
                attempt=attempt_index,
                family=family,
                subspace=subspace,
                solver=solver,
                damping=damping,
                solver_iterations=iterations,
                certificate=_failed_certificate(
                    eta,
                    reason,
                    loss_before=base_loss,
                    target_norm=target_norm,
                ),
                certificate_scope="proposal_generation",
                certificate_points=screening_count,
            )
        )

    for family in config.family_order:
        if progress is not None:
            progress(
                f"[ADAPT] step {step}: proposal family={family}, "
                f"screening_points={screening_count}"
            )
        if family == "head_closed_form":
            for eta in rates:
                try:
                    candidate = _head_closed_form_candidate(
                        model,
                        proposal_x,
                        proposal_y,
                        eta,
                        config.computation_batch_size,
                    )
                except Exception as error:
                    record_generation_failure(
                        family=family,
                        subspace="head",
                        eta=eta,
                        reason=f"proposal_failure:{type(error).__name__}",
                    )
                    continue
                result = evaluate(
                    candidate,
                    family=family,
                    subspace="head",
                    eta=eta,
                    solver="lstsq",
                )
                if result is not None:
                    return result

        elif family == "scaled_parameter_gradient":
            for subspace in spaces:
                try:
                    direction = _scaled_parameter_gradient_direction(
                        model,
                        proposal_x,
                        proposal_y,
                        subspace,
                        config.gradient_tolerance,
                        config.computation_batch_size,
                    )
                except Exception as error:
                    record_generation_failure(
                        family=family,
                        subspace=subspace.name,
                        eta=rates[0],
                        reason=f"proposal_failure:{type(error).__name__}",
                        solver="autograd_jvp",
                    )
                    continue
                for eta in rates:
                    candidate = _apply_direction(model, direction, eta)
                    result = evaluate(
                        candidate,
                        family=family,
                        subspace=subspace.name,
                        eta=eta,
                        solver="autograd_jvp",
                    )
                    if result is not None:
                        return result

        elif family == "tangent_least_squares":
            for subspace in spaces:
                parameter_count = sum(
                    dict(model.named_parameters())[name].numel()
                    for name in subspace.parameter_names
                )
                batch_output_elements = min(
                    config.computation_batch_size,
                    proposal_x.shape[0],
                ) * (proposal_y[0].numel())
                exact = (
                    parameter_count * parameter_count
                    <= config.exact_jacobian_max_elements
                    and batch_output_elements * parameter_count
                    <= config.exact_jacobian_max_elements
                )
                for damping in config.tangent_damping:
                    budgets = (None,) if exact else config.tangent_cg_iterations
                    for budget in budgets:
                        solver = "exact_parameter_eigh" if exact else "cg"
                        try:
                            direction = (
                                _exact_tangent_direction(
                                    model,
                                    proposal_x,
                                    proposal_y,
                                    subspace,
                                    damping,
                                    config.computation_batch_size,
                                )
                                if exact
                                else _cg_tangent_direction(
                                    model,
                                    proposal_x,
                                    proposal_y,
                                    subspace,
                                    damping,
                                    int(budget),
                                    config.tangent_cg_tolerance,
                                    config.gradient_tolerance,
                                    config.computation_batch_size,
                                )
                            )
                        except Exception as error:
                            record_generation_failure(
                                family=family,
                                subspace=subspace.name,
                                eta=rates[0],
                                reason=(f"proposal_failure:{type(error).__name__}"),
                                solver=solver,
                                damping=damping,
                                iterations=budget,
                            )
                            continue
                        for eta in rates:
                            candidate = _apply_direction(model, direction, eta)
                            result = evaluate(
                                candidate,
                                family=family,
                                subspace=subspace.name,
                                eta=eta,
                                solver=solver,
                                damping=damping,
                                iterations=budget,
                            )
                            if result is not None:
                                return result

        elif family == "nonlinear_secant":
            for subspace in spaces:
                for eta in rates:
                    for budget in config.nonlinear_steps:
                        # Each budget is intentionally an independent clone: a failed
                        # candidate never becomes state for a later accepted step.
                        try:
                            candidate = _nonlinear_secant_candidate(
                                model,
                                proposal_x,
                                proposal_y,
                                subspace,
                                functional_learning_rate=eta,
                                steps=budget,
                                learning_rate=config.nonlinear_learning_rate,
                                batch_size=config.computation_batch_size,
                            )
                        except Exception as error:
                            record_generation_failure(
                                family=family,
                                subspace=subspace.name,
                                eta=eta,
                                reason=(f"proposal_failure:{type(error).__name__}"),
                                solver="adam",
                                iterations=budget,
                            )
                            continue
                        result = evaluate(
                            candidate,
                            family=family,
                            subspace=subspace.name,
                            eta=eta,
                            solver="adam",
                            iterations=budget,
                        )
                        if result is not None:
                            return result

    return AdaptiveFGDSearchResult(
        None,
        None,
        attempts,
        False,
        best_relative_error,
    )


def grow_layer_function_preserving(
    *,
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    layer_index: int,
    config: AdaptiveFGDConfig,
) -> tuple[nn.Module, float, int]:
    """Return a grown clone whose represented function is unchanged.

    GroMo is used only to find useful incoming features.  Existing-weight
    deltas are disabled and the outgoing weights of every new neuron are
    exactly zero.  Consequently this operation refines the representation
    without constituting an optimization step.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from gromo.utils.training_utils import compute_statistics

    candidate = _clone_model(model)
    candidate.eval()
    parameters_before = sum(parameter.numel() for parameter in candidate.parameters())

    candidate.set_growing_layers(index=layer_index)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(config.computation_batch_size, train_x.shape[0]),
        shuffle=False,
    )
    compute_statistics(
        candidate,
        loader,
        loss_function=torch.nn.MSELoss(reduction="sum"),
        device=_model_device(candidate),
    )
    candidate.compute_optimal_updates(
        compute_delta=False,
        maximum_added_neurons=config.growth_neurons_per_event,
        alpha_zero=False,
        omega_zero=True,
        use_projection=True,
    )
    candidate.reset_computation()
    candidate.dummy_select_update()
    growing_layer = candidate.currently_updated_layer
    growing_layer.apply_change(
        apply_delta=False,
        apply_extension=True,
        input_extension_scaling=1.0,
        output_extension_scaling=1.0,
    )
    growing_layer.delete_update()
    candidate.currently_updated_layer_index = None
    candidate.eval()

    model.eval()
    device = _model_device(model)
    drift = 0.0
    with torch.no_grad():
        for batch_x, _ in _iter_tensor_batches(
            train_x,
            train_y,
            batch_size=config.computation_batch_size,
            device=device,
        ):
            output_before = model(batch_x).to(torch.float64)
            output_after = candidate(batch_x).to(torch.float64)
            batch_drift = float(
                torch.max(torch.abs(output_after - output_before)).item()
            )
            drift = max(drift, batch_drift)
    added = sum(parameter.numel() for parameter in candidate.parameters()) - (
        parameters_before
    )
    if added <= 0:
        raise RuntimeError("GroMo did not add a neuron to the requested layer.")
    if not math.isfinite(drift) or drift > config.preservation_tolerance:
        raise RuntimeError(
            "Function-preserving growth exceeded its output tolerance: "
            f"{drift:.3e} > {config.preservation_tolerance:.3e}."
        )
    return candidate, drift, added
