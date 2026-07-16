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
        self.initial_loss = self._loss()
        # Measured Prop. 3.8-style envelope: L0 * prod (1 - 2 eta mu r).
        self.contraction_product = 1.0
        self.current_mu = 0.0
        self.current_mu_valid = False

    # ------------------------------------------------------------------
    # Loss and gradient primitives (full batch, exact).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _loss(self) -> float:
        predictions = self.model(self.train_x)
        residual = predictions.reshape(self.train_y.shape) - self.train_y
        return float(residual.square().sum().item()) / (
            2.0 * residual.shape[0]
        )

    def _loss_and_gradients(self) -> tuple[float, list[torch.Tensor]]:
        self.model.zero_grad(set_to_none=True)
        predictions = self.model(self.train_x)
        residual = predictions.reshape(self.train_y.shape) - self.train_y
        loss = residual.square().sum() / (2.0 * residual.shape[0])
        loss.backward()
        gradients = [
            (
                parameter.grad.detach().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            )
            for parameter in self.model.parameters()
        ]
        return float(loss.item()), gradients

    # ------------------------------------------------------------------
    # The measured PL constant: mu = lambda_min(J J^T) / n_cert.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _apply_update(self, gradients: list[torch.Tensor], scale: float) -> None:
        for parameter, gradient in zip(self.model.parameters(), gradients):
            parameter.add_(gradient, alpha=scale)

    def measure_mu(self) -> tuple[float, bool]:
        """Exact smallest eigenvalue of the tangent Gram on the certificate set.

        Builds the per-(sample, output) Jacobian rows by plain autograd
        (robust for any module, including GroMo layers) and returns
        ``(mu, valid)`` with ``mu = lambda_min(J J^T)/n_cert``. The Gram is
        rank deficient whenever the parameter count is smaller than
        ``n_cert * out_features`` -- then ``mu = 0`` and the certificate is
        honestly off (underparametrized structure).
        """
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        x = self.train_x[self.certificate_indices]
        n_cert = x.shape[0]
        rows: list[torch.Tensor] = []
        for i in range(n_cert):
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
        jacobian = torch.stack(rows)  # (n_cert * m, P)
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
            backtracks += 1
            if backtracks > config.max_backtracks:
                accepted = False
                loss_after = loss_before
                descent = 0.0
                learning_rate = 0.0
                break
            learning_rate *= config.backtrack_factor

        if accepted and learning_rate > 0.0:
            self.learning_rate = learning_rate * config.lr_recovery
            # Measured contraction factor of the Prop. 3.8-style envelope.
            factor = 1.0 - 2.0 * learning_rate * self.current_mu * max(
                descent,
                0.0,
            )
            self.contraction_product *= min(max(factor, 0.0), 1.0)
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
            if record.converged:
                break
        train_loss = records[-1].loss_after if records else self._loss()
        bound = self.envelope()
        tolerance = self.config.eps * (1.0 + self.initial_loss)
        envelope_valid = (
            train_loss <= bound + tolerance if mu_valid else None
        )
        return EmpiricalPLEpochResult(
            step_records=records,
            mu=mu,
            mu_valid=mu_valid,
            mu_collapsed=not mu_valid,
            train_loss=train_loss,
            envelope=bound,
            envelope_valid=envelope_valid,
            converged=self.converged,
        )
