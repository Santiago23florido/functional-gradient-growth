"""Empirical-PL certified training of the full network.

This module implements the reconciliation between "satisfy the paper's
criteria" and "actually train the network's weights": instead of *assuming*
the Polyak-Lojasiewicz constant of the nonconvex parametric objective
(which cannot be certified a priori -- any saddle point, e.g. all-zero
weights, violates PL for every candidate constant), the constant is
*measured* along the trajectory.

Mathematical background
-----------------------
For the empirical MSE ``L(theta) = (1/(2n)) sum_i ||f_theta(X_i) - Y_i||^2``
with residual vector ``r`` (stacked over samples and output channels) and
per-sample Jacobian ``J`` (rows indexed by (sample, output) pairs, columns
by parameters), the parameter gradient is ``grad L = J^T r / n`` exactly,
hence

    ||grad L||^2 = (1/n^2) r^T (J J^T) r >= (2 lambda_min(J J^T) / n) L.

This is an algebraic identity, not an assumption: the loss satisfies PL at
``theta_t`` with the *measured* constant ``mu_t = lambda_min(K_t)/n`` where
``K_t = J_t J_t^T`` is the empirical tangent (NTK) Gram at the current
weights. Combined with the realized per-step descent coefficient

    r_t = (L_t - L_{t+1}) / (eta_t ||grad L_t||^2),

(which absorbs the second-order remainder of the parametrization a
posteriori -- steps are rejected and the learning rate backtracked until
``r_t >= r_min``), one obtains the measured analogue of Prop. 3.8 of
arXiv:2606.16926:

    L_T <= L_0 * prod_t (1 - 2 eta_t mu_t r_t).

If the measured ``mu_t`` stays bounded away from zero until the loss is
(numerically) zero, the run carries an a-posteriori certificate of global
optimality *for the full nonconvex training* -- this is exactly the regime
that NTK theory (Jacot et al.; Du et al.) proves is reachable for
sufficiently wide networks, here verified instead of assumed. When the
network is too narrow, ``mu_t`` collapses to zero (the Gram is rank
deficient whenever the parameter count is below the number of certificate
equations); the certificate then honestly switches off, and the collapse is
the principled, certificate-driven trigger for growing the network.

Honest scope: the certificate is conditional-but-verified (it certifies
the run that happened; it cannot promise in advance that ``mu_t`` will stay
positive for a narrow network), and it applies to the empirical loss on the
certificate subset (``certificate_points`` samples) that the Gram is
computed on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True)
class EmpiricalPLConfig:
    """Hyperparameters of the empirical-PL certified trainer.

    ``learning_rate`` is only the initial trial step: every step is
    validated by the realized descent coefficient and backtracked until it
    certifies (``r_min``, ``backtrack_factor``, ``max_backtracks``), and
    gently re-expanded after accepted steps (``lr_recovery``).
    ``certificate_points`` bounds the size of the tangent Gram (the
    certificate applies to the empirical loss on that subset);
    ``mu_collapse_threshold`` defines when the measured PL constant is
    declared collapsed, which is the growth trigger.
    """

    learning_rate: float = 1.0
    r_min: float = 0.5
    backtrack_factor: float = 0.5
    max_backtracks: int = 12
    lr_recovery: float = 1.25
    steps_per_epoch: int = 10
    certificate_points: int = 256
    certificate_seed: int = 0
    mu_collapse_threshold: float = 1e-8
    gradient_tolerance: float = 1e-16
    loss_tolerance: float = 1e-12
    eps: float = 1e-12
    growth_cooldown_epochs: int = 3
    growth_max_events: int = 8
    growth_max_hidden_size: int | None = None
    # Growth must be EARNED: it fires only when mu is collapsed AND the
    # relative per-epoch loss improvement fell below this threshold (the
    # current structure stopped paying off). This changes only the growth
    # timing; the per-step descent certificates are untouched.
    growth_min_progress: float = 0.01
    # Growth is ARBITRATED by the certified ceiling: every eligible layer
    # is trial-grown and the candidate with the lowest closed-form head
    # optimum L* wins; growth is skipped when no candidate improves the
    # current ceiling by at least this relative amount.
    growth_min_ceiling_improvement: float = 0.0
    # Optional ridge term: the objective becomes F = L + (ridge/2)||theta||^2.
    # HONEST certificate semantics: the global vs-zero envelope
    # L_0 prod(1 - 2 eta mu r) is only available for the pure data loss
    # (ridge = 0); with ridge > 0 the augmented residual's Gram is rank
    # deficient in the vs-zero comparison, so what remains certified is
    # per-step sufficient descent of F (r_t >= r_min) and convergence to
    # stationarity, while mu keeps being measured on the data Gram as the
    # growth sensor.
    ridge: float = 0.0
    # Declare convergence after this many consecutive fully-rejected steps
    # (descent below measurement resolution: stationary at precision). A
    # fully-rejected step CONTINUES the eta search: the persistent
    # learning rate keeps shrinking across steps until acceptance or the
    # floor below.
    max_rejected_steps: int = 3
    learning_rate_floor: float = 1e-12
    # Keep a snapshot of the best-validation model (deployment choice made
    # by the pipeline; does not alter the certified trajectory).
    keep_best_validation: bool = True


@dataclass(frozen=True)
class EmpiricalPLStepRecord:
    """One accepted (or exhausted) full-batch step with its certificates."""

    step: int
    learning_rate: float
    loss_before: float
    loss_after: float
    gradient_sq_norm: float
    descent_coefficient: float
    accepted: bool
    backtracks: int
    converged: bool


@dataclass(frozen=True)
class EmpiricalPLEpochResult:
    step_records: list[EmpiricalPLStepRecord] = field(default_factory=list)
    mu: float = 0.0
    mu_valid: bool = False
    mu_collapsed: bool = True
    train_loss: float = 0.0
    envelope: float = 0.0
    envelope_enabled: bool = True
    envelope_valid: bool | None = None
    converged: bool = False


class EmpiricalPLTrainer:
    """Full-weight certified trainer: measured PL constant + validated steps.

    Trains *all* parameters of ``model`` by full-batch gradient descent on
    the MSE. Every step must realize a descent coefficient ``r_t >= r_min``
    (backtracking otherwise), and the measured envelope
    ``L_0 prod (1 - 2 eta mu r)`` is validated against the true loss every
    epoch. The trainer never mutates the architecture; when ``mu``
    collapses the caller is expected to grow the network and build a fresh
    trainer for the new structure (certificates are per structure).
    """

    def __init__(
        self,
        model: nn.Module,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        config: EmpiricalPLConfig,
        device: torch.device | None = None,
    ) -> None:
        if not 0.0 < config.r_min < 1.0:
            raise ValueError("fgd_pl.r_min must lie in (0, 1).")
        if not 0.0 < config.backtrack_factor < 1.0:
            raise ValueError("fgd_pl.backtrack_factor must lie in (0, 1).")
        if config.learning_rate <= 0.0:
            raise ValueError("fgd_pl.learning_rate must be positive.")
        if config.steps_per_epoch < 1:
            raise ValueError("fgd_pl.steps_per_epoch must be >= 1.")
        if config.certificate_points < 1:
            raise ValueError("fgd_pl.certificate_points must be >= 1.")
        if config.lr_recovery < 1.0:
            raise ValueError("fgd_pl.lr_recovery must be >= 1.")
        if config.ridge < 0.0:
            raise ValueError("fgd_pl.ridge must be >= 0.")
        if config.max_rejected_steps < 1:
            raise ValueError("fgd_pl.max_rejected_steps must be >= 1.")
        if config.growth_min_progress < 0.0:
            raise ValueError("fgd_pl.growth_min_progress must be >= 0.")
        if config.growth_min_ceiling_improvement < 0.0:
            raise ValueError(
                "fgd_pl.growth_min_ceiling_improvement must be >= 0."
            )
        if config.learning_rate_floor <= 0.0:
            raise ValueError("fgd_pl.learning_rate_floor must be positive.")

        self.config = config
        device = device or next(model.parameters()).device
        self.device = device
        self.model = model.to(device)
        x = train_x.reshape(train_x.shape[0], -1).to(device=device)
        y = train_y.reshape(train_y.shape[0], -1).to(device=device)
        if x.shape[0] != y.shape[0]:
            raise ValueError("train_x and train_y must have matching lengths.")
        self.train_x = x
        self.train_y = y

        n = x.shape[0]
        keep = torch.randperm(
            n,
            generator=torch.Generator().manual_seed(config.certificate_seed),
        )[: min(config.certificate_points, n)].to(device)
        self.certificate_indices = keep

        self.learning_rate = config.learning_rate
        self.total_steps = 0
        self.converged = False
        # The global vs-zero envelope is only sound for the pure data loss
        # (zero is its universal lower bound and the Gram identity compares
        # against it); with ridge the certified statements are per-step
        # sufficient descent and stationarity.
        self.envelope_enabled = config.ridge == 0.0
        self._consecutive_rejections = 0
        self.converged_reason = ""
        self.initial_loss = self._loss()
        # Measured Prop. 3.8-style envelope: L0 * prod (1 - 2 eta mu r).
        self.contraction_product = 1.0
        self.current_mu = 0.0
        self.current_mu_valid = False

    # ------------------------------------------------------------------
    # Loss and gradient primitives (full batch, exact).
    # ------------------------------------------------------------------
    def _ridge_penalty(self) -> float:
        if self.config.ridge == 0.0:
            return 0.0
        return 0.5 * self.config.ridge * float(
            sum(
                parameter.detach().to(torch.float64).square().sum().item()
                for parameter in self.model.parameters()
            )
        )

    @torch.no_grad()
    def _loss(self) -> float:
        """Objective value measured in float64.

        float64 measurement keeps the acceptance test meaningful for
        descents below float32 resolution (which otherwise stall the run
        as spurious rejections).
        """
        predictions = self.model(self.train_x).to(torch.float64)
        residual = predictions.reshape(self.train_y.shape) - self.train_y.to(
            torch.float64
        )
        data_loss = float(residual.square().sum().item()) / (
            2.0 * residual.shape[0]
        )
        return data_loss + self._ridge_penalty()

    def _loss_and_gradients(self) -> tuple[float, list[torch.Tensor]]:
        self.model.zero_grad(set_to_none=True)
        predictions = self.model(self.train_x)
        residual = predictions.reshape(self.train_y.shape) - self.train_y
        objective = residual.square().sum() / (2.0 * residual.shape[0])
        if self.config.ridge > 0.0:
            objective = objective + 0.5 * self.config.ridge * sum(
                parameter.square().sum()
                for parameter in self.model.parameters()
            )
        objective.backward()
        gradients = [
            (
                parameter.grad.detach().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            )
            for parameter in self.model.parameters()
        ]
        return self._loss(), gradients

    # ------------------------------------------------------------------
    # The measured PL constant: mu = lambda_min(J J^T) / n_cert.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_update(self, gradients: list[torch.Tensor], scale: float) -> None:
        for parameter, gradient in zip(self.model.parameters(), gradients):
            parameter.add_(gradient, alpha=scale)

    def _jacobian_rows_loop(self, x: torch.Tensor) -> torch.Tensor:
        """Per-(sample, output) Jacobian rows by plain autograd (robust)."""
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        rows: list[torch.Tensor] = []
        for i in range(x.shape[0]):
            output = self.model(x[i : i + 1]).reshape(-1)
            for c in range(output.shape[0]):
                grads = torch.autograd.grad(
                    output[c],
                    parameters,
                    retain_graph=(c < output.shape[0] - 1),
                    allow_unused=True,
                )
                row = torch.cat(
                    [
                        (
                            g.reshape(-1)
                            if g is not None
                            else torch.zeros(p.numel(), device=x.device)
                        )
                        for g, p in zip(grads, parameters)
                    ]
                ).to(torch.float64)
                rows.append(row)
        return torch.stack(rows)

    def _jacobian_rows_fast(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized Jacobian rows via torch.func (vmap + jacrev)."""
        from torch.func import functional_call, jacrev, vmap

        params = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

        def network(p: dict, sample: torch.Tensor) -> torch.Tensor:
            return functional_call(
                self.model,
                p,
                (sample.unsqueeze(0),),
            ).reshape(-1)

        jac = vmap(jacrev(network), in_dims=(None, 0))(params, x)
        n = x.shape[0]
        m = self.train_y.shape[1]
        # jac[name]: (n, m, *param.shape) -> rows ordered (sample, channel),
        # columns concatenated in parameter registration order (same layout
        # as the autograd loop).
        blocks = [jac[name].reshape(n, m, -1) for name in params]
        jacobian = torch.cat(blocks, dim=2).reshape(n * m, -1)
        return jacobian.to(torch.float64)

    def _jacobian_rows(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self._jacobian_rows_fast(x)
        except Exception:
            # torch.func can fail on exotic modules; the loop is always
            # correct, just slower.
            return self._jacobian_rows_loop(x)

    def measure_mu(self) -> tuple[float, bool]:
        """Exact smallest eigenvalue of the tangent Gram on the certificate set.

        Returns ``(mu, valid)`` with ``mu = lambda_min(J J^T)/n_cert``.
        Rank shortcut: the Gram has ``n_cert * out_features`` rows and rank
        at most the parameter count ``P``; when ``P`` is smaller,
        ``lambda_min = 0`` exactly and no Jacobian needs to be built (the
        certificate is honestly off: underparametrized structure).
        """
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        parameter_count = sum(p.numel() for p in parameters)
        x = self.train_x[self.certificate_indices]
        n_cert = x.shape[0]
        gram_rows = n_cert * self.train_y.shape[1]
        if parameter_count < gram_rows:
            self.current_mu = 0.0
            self.current_mu_valid = False
            return 0.0, False

        jacobian = self._jacobian_rows(x)  # (n_cert * m, P)
        gram = jacobian @ jacobian.T
        eigenvalues = torch.linalg.eigvalsh(gram)
        lambda_min = max(float(eigenvalues.min().item()), 0.0)
        mu = lambda_min / n_cert
        valid = mu > self.config.mu_collapse_threshold
        self.current_mu = mu
        self.current_mu_valid = valid
        return mu, valid

    # ------------------------------------------------------------------
    # One certified step: backtrack until r_t >= r_min, then commit.
    # ------------------------------------------------------------------
    def step(self) -> EmpiricalPLStepRecord:
        config = self.config
        loss_before, gradients = self._loss_and_gradients()
        gradient_sq = float(
            sum(g.square().sum().item() for g in gradients)
        )
        if (
            gradient_sq <= config.gradient_tolerance
            or loss_before <= config.loss_tolerance
        ):
            self.converged = True
            self.converged_reason = "numerical_zero"
            return EmpiricalPLStepRecord(
                step=self.total_steps,
                learning_rate=0.0,
                loss_before=loss_before,
                loss_after=loss_before,
                gradient_sq_norm=gradient_sq,
                descent_coefficient=0.0,
                accepted=True,
                backtracks=0,
                converged=True,
            )

        learning_rate = self.learning_rate
        last_tried_rate = learning_rate
        backtracks = 0
        while True:
            self._apply_update(gradients, -learning_rate)
            loss_after = self._loss()
            descent = (loss_before - loss_after) / (
                learning_rate * gradient_sq
            )
            if descent >= config.r_min:
                accepted = True
                break
            # Reject: undo, shrink, retry. The realized coefficient absorbs
            # the second-order remainder of the parametrization, so this
            # loop terminates for small enough eta (descent -> 1).
            self._apply_update(gradients, learning_rate)
            last_tried_rate = learning_rate
            backtracks += 1
            if backtracks > config.max_backtracks:
                accepted = False
                loss_after = loss_before
                descent = 0.0
                learning_rate = 0.0
                break
            learning_rate *= config.backtrack_factor

        if accepted and learning_rate > 0.0:
            self._consecutive_rejections = 0
            self.learning_rate = learning_rate * config.lr_recovery
            if self.envelope_enabled:
                # Measured contraction factor of the Prop. 3.8 envelope.
                factor = 1.0 - 2.0 * learning_rate * self.current_mu * max(
                    descent,
                    0.0,
                )
                self.contraction_product *= min(max(factor, 0.0), 1.0)
        else:
            self._consecutive_rejections += 1
            # Continue the eta search where it left off: the next step
            # starts below the smallest rate just rejected, so consecutive
            # rejections keep shrinking eta instead of retrying the same
            # cascade.
            self.learning_rate = max(
                last_tried_rate * config.backtrack_factor,
                config.learning_rate_floor,
            )
            if (
                self.learning_rate <= config.learning_rate_floor
                or self._consecutive_rejections >= config.max_rejected_steps
            ):
                # No measurable certified descent at any admissible eta:
                # stationary at the achievable precision.
                self.converged = True
                self.converged_reason = "stationary_at_precision"
        self.total_steps += 1
        return EmpiricalPLStepRecord(
            step=self.total_steps,
            learning_rate=learning_rate,
            loss_before=loss_before,
            loss_after=loss_after,
            gradient_sq_norm=gradient_sq,
            descent_coefficient=descent,
            accepted=accepted,
            backtracks=backtracks,
            converged=False,
        )

    def envelope(self) -> float:
        """Measured envelope L_0 * prod (1 - 2 eta mu r)."""
        return self.initial_loss * self.contraction_product

    def run_epoch(self) -> EmpiricalPLEpochResult:
        mu, mu_valid = self.measure_mu()
        records: list[EmpiricalPLStepRecord] = []
        for _ in range(self.config.steps_per_epoch):
            record = self.step()
            records.append(record)
            if record.converged or self.converged:
                break
        train_loss = records[-1].loss_after if records else self._loss()
        bound = self.envelope()
        tolerance = self.config.eps * (1.0 + self.initial_loss)
        envelope_valid = (
            train_loss <= bound + tolerance
            if (mu_valid and self.envelope_enabled)
            else None
        )
        return EmpiricalPLEpochResult(
            step_records=records,
            mu=mu,
            mu_valid=mu_valid,
            mu_collapsed=not mu_valid,
            train_loss=train_loss,
            envelope=bound,
            envelope_enabled=self.envelope_enabled,
            envelope_valid=envelope_valid,
            converged=self.converged,
        )
