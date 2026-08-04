"""Minibatch-only nonlinear primary approximation family.

This module deliberately has no dependency on the tangent-system, projection,
or realization helpers.  A disposable AdamW clone is trained toward the
functional target ``f_t - eta_f r_t`` and its realized secant displacement is
certified by streaming dot products and squared norms over a loader.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import torch

from fgdlib.profile import increment, timed
from fgdlib.tangent import (
    FGDApproxConfig,
    ParametricGDConfig,
    batch_functional_loss,
    functional_gradient,
)

__all__ = [
    "NonlinearCandidate",
    "NonlinearCertificateStats",
    "scale_parameter_displacement",
    "stream_nonlinear_certificate",
    "train_nonlinear_candidate",
]


def _sync_cuda() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


@dataclass(frozen=True)
class NonlinearCandidate:
    """A disposable AdamW candidate and its generation diagnostics."""

    model: object | None
    functional_learning_rate: float
    inner_steps: int
    batches_seen: int
    final_objective: float | None
    sensor_valid: bool
    training_seconds: float


@dataclass(frozen=True)
class NonlinearCertificateStats:
    """Global nonlinear directional certificate accumulated over batches."""

    dot_product: float
    displacement_sq_norm: float
    gradient_sq_norm: float
    cosine: float | None
    relative_error: float | None
    base_loss: float
    candidate_loss: float
    batches_seen: int
    sensor_valid: bool
    certified: bool
    certification_seconds: float


def _parameters_are_finite(model: torch.nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    )


def train_nonlinear_candidate(
    *,
    base_model: torch.nn.Module,
    train_loader,
    device: torch.device,
    functional_learning_rate: float,
    inner_steps: int,
    config: ParametricGDConfig,
    fgd_config: FGDApproxConfig,
) -> NonlinearCandidate:
    """Train one disposable AdamW clone using ordinary loader minibatches.

    ``inner_steps`` is an optimizer-step budget for the nonlinear primary
    family. The loader is re-iterated only when the budget exceeds one pass;
    no batch, target, or output from a previous step is retained.
    """
    if config.optimizer != "adamw":
        raise ValueError(
            "The nonlinear primary family requires parametric_gd.optimizer='adamw'."
        )
    increment("nonlinear_ladder_attempts")
    _sync_cuda()
    started = time.perf_counter()
    batches_seen = 0
    final_objective: float | None = None
    sensor_valid = True
    candidate: torch.nn.Module | None = copy.deepcopy(base_model)

    base_was_training = base_model.training
    base_model.eval()
    # Keep dropout and batch-normalization state fixed: the certificate is a
    # statement about the represented eval-mode function, while gradients and
    # AdamW remain fully available in eval mode.
    candidate.eval()
    base_parameters = {
        name: parameter.detach().clone()
        for name, parameter in base_model.named_parameters()
    }
    trainable = [
        parameter for parameter in candidate.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.inner_learning_rate,
        weight_decay=config.weight_decay,
    )

    plateau_stop = bool(fgd_config.certify_family_plateau_stop)
    plateau_tol = float(fgd_config.certify_family_plateau_tol)
    best_epoch_objective = math.inf
    stale_epochs = 0

    with timed("nonlinear_candidate_training_seconds"):
        try:
            step_budget = max(1, int(inner_steps))
            while batches_seen < step_budget:
                epoch_objective = 0.0
                epoch_batches = 0
                for x, y in train_loader:
                    if batches_seen >= step_budget:
                        break
                    batches_seen += 1
                    epoch_batches += 1
                    x = x.to(device)
                    y = y.to(device)
                    with torch.no_grad():
                        base_output = base_model(x)
                        residual = functional_gradient(
                            base_output,
                            y,
                            fgd_config.functional_loss,
                        )
                        functional_target = (
                            base_output - functional_learning_rate * residual
                        )
                    if not (
                        torch.isfinite(base_output).all()
                        and torch.isfinite(residual).all()
                        and torch.isfinite(functional_target).all()
                    ):
                        sensor_valid = False
                        break

                    optimizer.zero_grad(set_to_none=True)
                    candidate_output = candidate(x)
                    objective = torch.mean((candidate_output - functional_target) ** 2)
                    if config.parameter_penalty > 0.0:
                        penalty = torch.zeros((), device=device)
                        for name, parameter in candidate.named_parameters():
                            penalty = penalty + torch.mean(
                                (parameter - base_parameters[name]) ** 2
                            )
                        objective = objective + config.parameter_penalty * penalty
                    if not (
                        torch.isfinite(candidate_output).all()
                        and torch.isfinite(objective)
                    ):
                        sensor_valid = False
                        break

                    objective.backward()
                    if config.gradient_clip_norm is not None:
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            trainable,
                            config.gradient_clip_norm,
                        )
                        if not torch.isfinite(gradient_norm):
                            sensor_valid = False
                            break
                    elif any(
                        parameter.grad is not None
                        and not bool(torch.isfinite(parameter.grad).all())
                        for parameter in trainable
                    ):
                        sensor_valid = False
                        break
                    optimizer.step()
                    if not _parameters_are_finite(candidate):
                        sensor_valid = False
                        break
                    value = float(objective.detach())
                    epoch_objective += value
                    final_objective = value

                if not sensor_valid or epoch_batches == 0:
                    sensor_valid = False
                    break
                if plateau_stop:
                    epoch_mean = epoch_objective / epoch_batches
                    improvement = best_epoch_objective - epoch_mean
                    required = plateau_tol * max(abs(best_epoch_objective), 1.0)
                    if math.isfinite(best_epoch_objective) and improvement <= required:
                        stale_epochs += 1
                    else:
                        best_epoch_objective = epoch_mean
                        stale_epochs = 0
                if stale_epochs >= 3:
                    break
        finally:
            base_model.train(base_was_training)

    _sync_cuda()
    elapsed = time.perf_counter() - started
    if not sensor_valid:
        candidate = None
    return NonlinearCandidate(
        model=candidate,
        functional_learning_rate=float(functional_learning_rate),
        inner_steps=int(inner_steps),
        batches_seen=batches_seen,
        final_objective=final_objective,
        sensor_valid=sensor_valid,
        training_seconds=elapsed,
    )


@torch.no_grad()
def stream_nonlinear_certificate(
    *,
    base_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    certification_loader,
    device: torch.device,
    config: FGDApproxConfig,
    max_batches: int | None = None,
) -> NonlinearCertificateStats:
    """Accumulate one global cosine; never average per-batch cosines."""
    _sync_cuda()
    started = time.perf_counter()
    base_was_training = base_model.training
    candidate_was_training = candidate_model.training
    base_model.eval()
    candidate_model.eval()
    dot_product = 0.0
    displacement_sq_norm = 0.0
    gradient_sq_norm = 0.0
    base_loss = 0.0
    candidate_loss = 0.0
    batches_seen = 0
    sensor_valid = True

    with timed("nonlinear_certification_seconds"):
        try:
            for x, y in certification_loader:
                if max_batches is not None and batches_seen >= max_batches:
                    break
                batches_seen += 1
                x = x.to(device)
                y = y.to(device)
                base_output = base_model(x).to(torch.float64)
                candidate_output = candidate_model(x).to(torch.float64)
                y64 = y.to(torch.float64)
                residual = functional_gradient(
                    base_output,
                    y64,
                    config.functional_loss,
                )
                displacement = base_output - candidate_output
                batch_base_loss = batch_functional_loss(
                    base_output,
                    y64,
                    config.functional_loss,
                )
                batch_candidate_loss = batch_functional_loss(
                    candidate_output,
                    y64,
                    config.functional_loss,
                )
                if not (
                    torch.isfinite(base_output).all()
                    and torch.isfinite(candidate_output).all()
                    and torch.isfinite(residual).all()
                    and torch.isfinite(displacement).all()
                    and torch.isfinite(batch_base_loss)
                    and torch.isfinite(batch_candidate_loss)
                ):
                    sensor_valid = False
                    break
                dot_product += float(torch.sum(displacement * residual))
                displacement_sq_norm += float(torch.sum(displacement.square()))
                gradient_sq_norm += float(torch.sum(residual.square()))
                base_loss += float(batch_base_loss)
                candidate_loss += float(batch_candidate_loss)
        finally:
            base_model.train(base_was_training)
            candidate_model.train(candidate_was_training)

    finite = all(
        math.isfinite(value)
        for value in (
            dot_product,
            displacement_sq_norm,
            gradient_sq_norm,
            base_loss,
            candidate_loss,
        )
    )
    sensor_valid = (
        sensor_valid
        and finite
        and batches_seen > 0
        and displacement_sq_norm > config.eps
        and gradient_sq_norm > config.eps
    )
    cosine: float | None = None
    relative_error: float | None = None
    if sensor_valid:
        cosine = dot_product / math.sqrt(displacement_sq_norm * gradient_sq_norm)
        cosine = max(-1.0, min(1.0, cosine))
        relative_error = math.sqrt(max(0.0, 1.0 - max(cosine, 0.0) ** 2))
        threshold = min(config.rel_error_threshold, 0.5)
        # Canonicalize the floating-point representation of the strict edge:
        # sqrt(1 - (sqrt(3)/2)^2) can round to 0.4999999999999999. Treating
        # that as below 0.5 would turn an exact equality into an acceptance.
        if abs(relative_error - threshold) <= config.eps:
            relative_error = threshold
        sensor_valid = math.isfinite(cosine) and math.isfinite(relative_error)

    threshold = min(config.rel_error_threshold, 0.5)
    certified = bool(
        sensor_valid
        and cosine is not None
        and cosine > 0.0
        and relative_error is not None
        and relative_error < threshold
    )
    _sync_cuda()
    elapsed = time.perf_counter() - started
    return NonlinearCertificateStats(
        dot_product=dot_product,
        displacement_sq_norm=displacement_sq_norm,
        gradient_sq_norm=gradient_sq_norm,
        cosine=cosine,
        relative_error=relative_error,
        base_loss=base_loss,
        candidate_loss=candidate_loss,
        batches_seen=batches_seen,
        sensor_valid=sensor_valid,
        certified=certified,
        certification_seconds=elapsed,
    )


def scale_parameter_displacement(
    *,
    base_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    rate: float,
) -> torch.nn.Module:
    """Return ``theta + rate * (theta_candidate - theta)`` transactionally."""
    committed = copy.deepcopy(base_model)
    base_parameters = dict(base_model.named_parameters())
    candidate_parameters = dict(candidate_model.named_parameters())
    committed_parameters = dict(committed.named_parameters())
    if not (
        base_parameters.keys()
        == candidate_parameters.keys()
        == committed_parameters.keys()
    ):
        raise RuntimeError("Nonlinear candidate parameter structure changed.")
    with torch.no_grad():
        for name, parameter in committed_parameters.items():
            base = base_parameters[name]
            moved = candidate_parameters[name]
            parameter.copy_(base + rate * (moved - base))
    committed.eval()
    return committed
