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
    "InterpolatedStep",
    "NonlinearCandidate",
    "NonlinearCertificateStats",
    "scale_parameter_displacement",
    "search_interpolated_step",
    "stream_nonlinear_certificate",
    "train_nonlinear_candidate",
]


def _sync_cuda() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


@dataclass(frozen=True)
class NonlinearCandidate:
    """A disposable AdamW candidate and its generation diagnostics.

    The budget fields are reported separately on purpose: ``inner_steps`` is
    the CONFIGURED number in the configured unit, while ``optimizer_steps``,
    ``epochs``, ``batches_seen`` and ``examples_seen`` are what the run
    actually spent. Comparing a nonlinear-primary candidate with a ladder
    candidate is only meaningful through ``examples_seen``.
    """

    model: object | None
    functional_learning_rate: float
    inner_steps: int
    batches_seen: int
    final_objective: float | None
    sensor_valid: bool
    training_seconds: float
    #: AdamW updates actually applied.
    optimizer_steps: int = 0
    #: Complete passes over the training source.
    epochs: float = 0.0
    #: Example-gradients consumed (sum of batch sizes over all updates).
    examples_seen: int = 0
    #: Candidate objective before the first update and after the last one.
    initial_objective: float | None = None
    #: Unit the configured ``inner_steps`` was interpreted in.
    step_unit: str = "minibatch"

    @property
    def objective_reduction(self) -> float | None:
        """``initial_objective - final_objective``; positive means progress."""
        if self.initial_objective is None or self.final_objective is None:
            return None
        return self.initial_objective - self.final_objective


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

    @property
    def effective_secant_rate(self) -> float | None:
        """``eta* = <Delta, r> / |r|^2`` -- the scale the candidate REALIZED.

        This is a fourth, distinct quantity from the ``eta_f`` that generated
        the target, from the interpolation ``alpha``, and from the Lemma 3.5
        admissible rate. It is the functional distance the displacement
        actually travelled along ``r``, and it is the scale at which
        ``sqrt(1 - cos^2)`` is the achievable relative error.
        """
        if not self.sensor_valid or self.gradient_sq_norm <= 0.0:
            return None
        return self.dot_product / self.gradient_sq_norm


def _parameters_are_finite(model: torch.nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    )


def _candidate_objective(
    outputs: torch.Tensor,
    target: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Parametric objective toward the functional target.

    ``"sum"`` is the ladder's convention and matches the certified sum-MSE
    functional exactly; ``"mean"`` divides it by ``batch * out_features``.
    AdamW absorbs most of that rescaling but NOT ``weight_decay`` or
    ``parameter_penalty``, which are applied in parameter space.
    """
    squared = (outputs - target) ** 2
    return torch.sum(squared) if reduction == "sum" else torch.mean(squared)


def train_nonlinear_candidate(
    *,
    base_model: torch.nn.Module,
    train_loader,
    device: torch.device,
    functional_learning_rate: float,
    inner_steps: int,
    config: ParametricGDConfig,
    fgd_config: FGDApproxConfig,
    probe: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> NonlinearCandidate:
    """Train one disposable AdamW clone toward ``f - eta_f r``.

    ``inner_steps`` is interpreted in ``config.inner_step_unit``:

    * ``"probe"`` -- full-batch AdamW steps on ``probe``, the ladder's
      ``certify_parametric_step`` semantics. The functional target is
      computed ONCE from the frozen base outputs, exactly as the ladder does.
    * ``"epoch"`` -- complete passes over ``train_loader``.
    * ``"minibatch"`` -- individual optimizer steps on loader minibatches.

    Under ``"epoch"`` and ``"minibatch"`` the target is recomputed per batch
    from the frozen base model, which is the same function of ``x`` -- only
    the budget accounting differs. No batch, target or output is retained
    between steps.
    """
    if config.optimizer != "adamw":
        raise ValueError(
            "The nonlinear primary family requires parametric_gd.optimizer='adamw'."
        )
    if config.inner_step_unit == "probe" and probe is None:
        raise ValueError(
            "parametric_gd.inner_step_unit='probe' requires a certification probe."
        )
    increment("nonlinear_ladder_attempts")
    _sync_cuda()
    started = time.perf_counter()
    batches_seen = 0
    optimizer_steps = 0
    examples_seen = 0
    epochs = 0.0
    final_objective: float | None = None
    initial_objective: float | None = None
    sensor_valid = True
    candidate: torch.nn.Module | None = copy.deepcopy(base_model)

    base_was_training = base_model.training
    base_model.eval()
    # The certificate is a statement about the represented eval-mode function.
    # The ladder nevertheless TRAINS its clone in train() mode, so the mode is
    # configurable and only observable when dropout / batch norm are present.
    candidate.train() if config.candidate_train_mode else candidate.eval()
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
    reduction = config.candidate_objective

    def _apply_update(x: torch.Tensor, y: torch.Tensor, target=None):
        """One AdamW update. Returns the objective, or ``None`` if non-finite."""
        nonlocal examples_seen
        if target is None:
            with torch.no_grad():
                base_output = base_model(x)
                residual = functional_gradient(
                    base_output,
                    y,
                    fgd_config.functional_loss,
                )
                target = base_output - functional_learning_rate * residual
            if not (
                torch.isfinite(base_output).all()
                and torch.isfinite(residual).all()
                and torch.isfinite(target).all()
            ):
                return None
        optimizer.zero_grad(set_to_none=True)
        candidate_output = candidate(x)
        objective = _candidate_objective(candidate_output, target, reduction)
        if config.parameter_penalty > 0.0:
            penalty = torch.zeros((), device=x.device)
            for name, parameter in candidate.named_parameters():
                penalty = penalty + torch.mean((parameter - base_parameters[name]) ** 2)
            objective = objective + config.parameter_penalty * penalty
        if not (torch.isfinite(candidate_output).all() and torch.isfinite(objective)):
            return None
        objective.backward()
        if config.gradient_clip_norm is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                config.gradient_clip_norm,
            )
            if not torch.isfinite(gradient_norm):
                return None
        elif any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        ):
            return None
        optimizer.step()
        if not _parameters_are_finite(candidate):
            return None
        examples_seen += int(x.shape[0])
        return float(objective.detach())

    plateau_stop = bool(fgd_config.certify_family_plateau_stop)
    plateau_tol = float(fgd_config.certify_family_plateau_tol)
    best_epoch_objective = math.inf
    stale_epochs = 0

    with timed("nonlinear_candidate_training_seconds"):
        try:
            budget = max(1, int(inner_steps))
            if config.inner_step_unit == "probe":
                # Ladder parity: one frozen target, full-batch steps.
                probe_x = probe[0].to(device)
                probe_y = probe[1].to(device)
                with torch.no_grad():
                    base_output = base_model(probe_x)
                    residual = functional_gradient(
                        base_output,
                        probe_y,
                        fgd_config.functional_loss,
                    )
                    probe_target = (
                        base_output - functional_learning_rate * residual
                    ).detach()
                if not torch.isfinite(probe_target).all():
                    sensor_valid = False
                for _ in range(budget if sensor_valid else 0):
                    value = _apply_update(probe_x, probe_y, target=probe_target)
                    if value is None:
                        sensor_valid = False
                        break
                    optimizer_steps += 1
                    batches_seen += 1
                    if initial_objective is None:
                        initial_objective = value
                    final_objective = value
                    if plateau_stop:
                        improvement = best_epoch_objective - value
                        required = plateau_tol * max(abs(best_epoch_objective), 1.0)
                        if math.isfinite(best_epoch_objective) and (
                            improvement <= required
                        ):
                            stale_epochs += 1
                        else:
                            best_epoch_objective = value
                            stale_epochs = 0
                        if stale_epochs >= 3:
                            break
                epochs = float(optimizer_steps)
            else:
                by_epoch = config.inner_step_unit == "epoch"
                completed_epochs = 0
                while sensor_valid and (
                    completed_epochs < budget if by_epoch else optimizer_steps < budget
                ):
                    epoch_objective = 0.0
                    epoch_batches = 0
                    for x, y in train_loader:
                        if not by_epoch and optimizer_steps >= budget:
                            break
                        x = x.to(device)
                        y = y.to(device)
                        value = _apply_update(x, y)
                        if value is None:
                            sensor_valid = False
                            break
                        optimizer_steps += 1
                        batches_seen += 1
                        epoch_batches += 1
                        epoch_objective += value
                        if initial_objective is None:
                            initial_objective = value
                        final_objective = value

                    if not sensor_valid or epoch_batches == 0:
                        sensor_valid = False
                        break
                    completed_epochs += 1
                    if plateau_stop:
                        epoch_mean = epoch_objective / epoch_batches
                        improvement = best_epoch_objective - epoch_mean
                        required = plateau_tol * max(abs(best_epoch_objective), 1.0)
                        if math.isfinite(best_epoch_objective) and (
                            improvement <= required
                        ):
                            stale_epochs += 1
                        else:
                            best_epoch_objective = epoch_mean
                            stale_epochs = 0
                    if stale_epochs >= 3:
                        break
                epochs = float(completed_epochs)
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
        optimizer_steps=optimizer_steps,
        epochs=epochs,
        examples_seen=examples_seen,
        initial_objective=initial_objective,
        step_unit=config.inner_step_unit,
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


@dataclass(frozen=True)
class InterpolatedStep:
    """One interpolation ``theta + alpha (theta' - theta)`` and ITS OWN certificate.

    ``stats`` is always re-measured on the interpolated model. It is never
    inherited from the full candidate: for a nonlinear network
    ``f_theta - f_{theta + alpha d} != alpha (f_theta - f_{theta + d})``, so a
    certificate earned by the full candidate says nothing about the direction
    of a shorter parameter-space step.
    """

    alpha: float
    model: object
    stats: NonlinearCertificateStats
    rejection_reason: str | None = None

    @property
    def certified(self) -> bool:
        return self.stats.certified

    @property
    def functional_descent(self) -> float:
        """``L(f_base) - L(f_alpha)`` measured on the certification split."""
        return self.stats.base_loss - self.stats.candidate_loss


def search_interpolated_step(
    *,
    base_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    certification_loader,
    device: torch.device,
    config: FGDApproxConfig,
    alpha_grid,
    policy: str = "largest_certified",
    max_batches: int | None = None,
    progress=None,
) -> tuple[InterpolatedStep | None, tuple[InterpolatedStep, ...]]:
    """Re-certify every interpolation and return the one the policy selects.

    The step that is APPLIED must be the step that was CERTIFIED. Scaling a
    certified parameter displacement by ``alpha`` and keeping the certificate
    is unsound; each ``alpha`` here is measured again from scratch on the same
    probe against the same ``min(rel_error_threshold, 1/2)``.

    Policies:

    * ``"largest_certified"`` -- the largest certified ``alpha`` (longest
      certified step).
    * ``"max_descent"``       -- among certified alphas, the largest measured
      functional descent.
    * ``"full_only"``         -- only ``alpha = 1``.

    Returns ``(selected, all_trials)``; ``selected`` is ``None`` when no
    interpolation certifies, and ``all_trials`` keeps every attempt so the
    caller can log why each was rejected rather than only the last failure.
    """
    alphas = tuple(dict.fromkeys(float(a) for a in (alpha_grid or ())))
    if policy == "full_only" or not alphas:
        alphas = (1.0,)
    trials: list[InterpolatedStep] = []
    for alpha in alphas:
        stepped = scale_parameter_displacement(
            base_model=base_model,
            candidate_model=candidate_model,
            rate=alpha,
        )
        stats = stream_nonlinear_certificate(
            base_model=base_model,
            candidate_model=stepped,
            certification_loader=certification_loader,
            device=device,
            config=config,
            max_batches=max_batches,
        )
        reason: str | None = None
        if not stats.sensor_valid:
            reason = "sensor_invalid"
        elif not stats.certified:
            reason = "relative_error_above_threshold"
        elif stats.base_loss - stats.candidate_loss <= 0.0:
            reason = "no_functional_descent"
        trials.append(InterpolatedStep(alpha, stepped, stats, reason))
        if progress is not None:
            cosine = "n/a" if stats.cosine is None else f"{stats.cosine:.4f}"
            epsilon = (
                "n/a" if stats.relative_error is None else f"{stats.relative_error:.4f}"
            )
            progress(
                f"[NONLINEAR-ALPHA] alpha={alpha:g}, cos={cosine}, eps={epsilon}, "
                f"descent={stats.base_loss - stats.candidate_loss:+.6e}, "
                f"certified={stats.certified}, "
                f"rejected={reason or 'no'}"
            )

    admissible = [trial for trial in trials if trial.rejection_reason is None]
    if not admissible:
        return None, tuple(trials)
    if policy == "max_descent":
        selected = max(admissible, key=lambda trial: trial.functional_descent)
    else:
        selected = max(admissible, key=lambda trial: trial.alpha)
    return selected, tuple(trials)
