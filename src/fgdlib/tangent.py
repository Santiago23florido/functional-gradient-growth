"""Approximate functional-gradient-descent training."""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Literal

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.growth import ScalingLineSearchConfig, grow_layer
from fgdlib.training import RegressionMetrics, evaluate_regression_metrics


ensure_gromo_importable()

import torch
from torch.func import functional_call, jacrev, jvp

from gromo.containers.growing_mlp import GrowingMLP
from gromo.utils.training_utils import compute_statistics


RelErrorMode = Literal["tangent_projection"]
LayerSelection = Literal["certifying", "tiny_best", "min_rel_error"]
ProjectionSolver = Literal[
    "gromo_layer",
    "cg",
    "exact",
    "exact_svd",
    "exact_kernel_eigh",
]
LearningRatePolicy = Literal["scheduler", "theory_interval"]
GrowthLimitCriterion = Literal["progress_floor", "epsilon_stationary"]
GrowthSelection = Literal[
    "descent_per_parameter",
    "epsilon_lookahead",
    # SENN (arXiv:2307.04526): rank by the RAW increase in the natural
    # expansion score. Identical to "epsilon_lookahead" except that it does
    # not divide by the added parameter count. See fgdlib/senn.py.
    "natural_expansion",
    # Pool every candidate NEURON from every location and rank by certified
    # first-order decrease per parameter it costs, s_i^2 / cost. Neuron-level
    # and pooled, which is what separates it from the refuted per-LAYER
    # ranking. See fgdlib/growth.py:allocate_by_expansion_per_parameter.
    "expansion_per_parameter",
]
FunctionalLoss = Literal["mse", "cross_entropy"]
GlobalBoundAction = Literal["lr_then_growth", "grow", "ignore"]


@dataclass(frozen=True)
class FGDApproxConfig:
    rel_error_threshold: float = 0.5
    rel_error_mode: RelErrorMode = "tangent_projection"
    layer_selection: LayerSelection = "certifying"
    projection_solver: ProjectionSolver = "exact"
    learning_rate_policy: LearningRatePolicy = "theory_interval"
    projection_damping: float = 1e-2
    cg_max_iterations: int = 16
    cg_tolerance: float = 1e-4
    theory_smoothness_constant: float = 2.0
    theory_alpha: float = 1.0
    theory_beta: float = 1.0
    theory_mu: float = 1.0
    theory_loss_star: float = 0.0
    theory_lr_safety: float = 0.95
    theory_lr_initial: float = 0.01
    theory_lr_min: float = 1e-5
    theory_lr_follow_bound: bool = False
    theory_lr_search_steps: int = 8
    theory_lr_search_refinements: int = 4
    sufficient_descent_c: float | None = 0.1
    lr_backtrack: float = 0.5
    lr_min_factor: float = 1e-3
    global_bound_action: GlobalBoundAction = "lr_then_growth"
    # Retained for config compatibility; FGD certificate growth is immediate.
    global_bound_lr_patience: int = 5
    rel_error_compute_delta: bool = True
    growth_compute_delta: bool = False
    # Retained for config compatibility; only scheduled growth uses timing gates.
    start_epoch: int = 1
    min_epochs_between_growth: int = 1
    # DEPRECATED: projection_group_* configured independent per-group
    # projections whose norms were aggregated into one certificate. That is
    # not a valid shared tangent direction and is no longer computed
    # anywhere; the fields are parsed for config compatibility only (the
    # legacy "scheduler" training path still concatenates batches with
    # projection_group_size). Certificates now use probe_batches.
    projection_group_size: int = 1
    projection_group_auto: bool = True
    projection_group_max: int | None = None
    # Number of mini-batches concatenated into the FIXED certification
    # probe. Every certificate solves ONE joint projection over the probe:
    # a single shared parameter direction for all probe batches.
    probe_batches: int = 4
    stall_min_epoch_decrease: float = 2e-3
    stall_patience: int = 5
    eps: float = 1e-12
    tiny_use_covariance: bool = True
    tiny_alpha_zero: bool = False
    tiny_omega_zero: bool = False
    tiny_use_projection: bool = True
    tiny_ignore_singular_values: bool = False
    tiny_use_fisher: bool = False
    tiny_maximum_added_neurons: int | None = None
    tiny_numerical_threshold: float = 1e-6
    tiny_statistical_threshold: float = 1e-3
    # Function-preserving growth: new neurons keep zero outgoing weights, no
    # delta and no scaling line search, so growth never changes the function.
    growth_function_preserving: bool = False
    growth_preservation_tolerance: float = 1e-6
    # Ordered ladder of approximation families. "tangent" must come first (it
    # is the epoch's main transactional search); the remaining entries are
    # tried in the listed order only after the previous family fails to
    # certify, and structural growth is probed only after every listed family
    # fails. Every family commits through the same full FGD certificate.
    # Supported: "tangent", "rkhs_head", "parametric_gd".
    family_order: tuple[str, ...] = ("tangent",)
    # Certify the tangent outer step by MEASURED validation descent
    # (Prop. 3.8) instead of the Lemma-3.5 relative-error interval. The
    # step direction is still the paper's functional-gradient projection
    # g = P_T r; only the step SIZE is chosen by a measured nonlinear line
    # search rather than the worst-case bound eta_max(eps), which is far
    # too conservative (measured optimum ~0.5 vs eta_max ~0.06 at eps 0.45).
    # Both certificates are in the paper; measured descent unlocks the
    # nonlinear step the linear bound forbids. Default False (legacy
    # eps-bounded step).
    tangent_measured_descent: bool = False
    # Largest eta tried by the measured tangent line search (the grid
    # descends geometrically from here to theory_lr_min). Only used when
    # tangent_measured_descent is True.
    tangent_measured_max_lr: float = 1.0
    # Certified outer steps attempted per epoch: each pass re-solves the
    # shared direction at the CURRENT model and certifies it independently
    # (k applications of the same per-step theorem). The epoch stops at the
    # first rejected attempt. 1 = one outer step per epoch (legacy).
    outer_steps_per_epoch: int = 1
    # The certified functional. Every certificate in this file is a
    # statement about THIS loss. "mse" is the legacy default with the exact
    # identity ||r||^2 = 4L (mu = 2, L* = 0). "cross_entropy" keeps
    # convexity and smoothness (so the method and every per-step
    # certificate transfer) but has NO global Polyak-Lojasiewicz constant,
    # so the linear C_glob envelope is unavailable for it -- see
    # report/CROSS_ENTROPY_FGD.md.
    functional_loss: FunctionalLoss = "mse"
    # How the growth LAYER is chosen.
    #
    #   "descent_per_parameter"  legacy: largest certified functional
    #                            descent per added parameter.
    #   "epsilon_lookahead"      R2: largest reduction of the Lemma-3.5
    #                            relative error per added parameter,
    #                            measured AFTER one certified family step on
    #                            the grown clone. Spending a parameter is
    #                            worthwhile exactly when it enlarges what the
    #                            reachable set can express, and that only
    #                            shows once the new capacity has been used:
    #                            measured on a trained 3x2 the IMMEDIATE eps
    #                            gets worse for every candidate (the delta
    #                            perturbs the function first), while after one
    #                            family step exactly one layer improves it --
    #                            the same layer that yields the best accuracy.
    #                            "no candidate reduces eps" is then the
    #                            termination condition: the structure is
    #                            already minimal-adequate.
    growth_selection: GrowthSelection = "descent_per_parameter"
    # Certified family steps taken on each grown clone before its eps is
    # judged, for growth_selection = "epsilon_lookahead".
    growth_lookahead_steps: int = 10
    # How the structure's LIMIT is recognised, i.e. when more training can
    # no longer substitute for more capacity.
    #
    #   "progress_floor"      the legacy reading: the structure is exhausted
    #                         when no family certifies progress above the
    #                         C_prog floor. Sound for sum-MSE, whose infimum
    #                         is attained at f = Y.
    #   "epsilon_stationary"  the reinterpretation: the structure is at its
    #                         REPRESENTATION limit when a certified step no
    #                         longer reduces the held-out relative error
    #                         eps = ||g - r|| / ||g||. Required for any
    #                         functional whose infimum is NOT attained
    #                         (cross-entropy: more confidence always lowers
    #                         the loss, so the progress floor never fires and
    #                         the structure never grows). It is a monotonicity
    #                         comparison of the Lemma-3.5 quantity between two
    #                         measured states -- no threshold, no window, no
    #                         budget -- and it is defined for EVERY certified
    #                         functional, which is what makes the method
    #                         loss-agnostic.
    growth_limit_criterion: GrowthLimitCriterion = "progress_floor"
    # Let the fallback families run whenever the tangent outer step fails,
    # independently of whether growth is due. They are nested inside the
    # growth trigger otherwise, so once the structure becomes adequate
    # (eps < 1/2 -> growth correctly not requested) the ladder is skipped
    # and only the tangent remains, freezing the flow. Default False.
    families_available_without_growth: bool = False
    # When Lemma 3.5 fails on the committed state (eps >=
    # rel_error_threshold) a successful family step does NOT cancel the
    # growth probe. Needed for any functional whose infimum is not attained
    # -- under cross-entropy some family step always certifies (more
    # confidence always lowers the loss), so growth would be postponed for
    # ever while the structure stays inadequate. Self-limiting through eps,
    # so it needs no parameter budget: once the reachable set can represent
    # the functional gradient again, eps drops below the threshold and the
    # override switches itself off. Default False.
    admissibility_failure_forces_growth: bool = False
    # Require the paper's structural criterion before growing: capacity is
    # increased only when Lemma 3.5 fails on the committed state, i.e. when
    # the relative error reaches rel_error_threshold and the reachable set
    # genuinely cannot represent the functional gradient. Without this, ANY
    # failed transaction requests growth -- including failures caused by the
    # step size or by a loss plateau at eps far below 1/2 -- and with no
    # parameter budget the run grows without bound. Default False keeps the
    # legacy trigger.
    growth_requires_admissibility_failure: bool = False
    # Choose the growth's scaling factor by held-out (validation) loss
    # instead of the GroMo default train loss. The magnitude of the
    # structural step is otherwise the one part of growth that escapes the
    # certificate: minimizing TRAIN loss over the scaling makes each growth
    # a train-fitting move, and the overfit accumulates over many growths.
    # On validation the growth's magnitude follows the same held-out
    # functional descent Prop. 3.8 certifies for every other step.
    # Default False (legacy GroMo behavior).
    growth_scaling_on_validation: bool = False
    # Grow EVERY hidden layer together (uniform widening) instead of
    # selecting one layer. This sidesteps the input-layer credit-assignment
    # problem of greedy per-layer growth: any greedy criterion (descent or
    # relative error) undervalues the input layer, whose benefit is latent,
    # so incremental growth from a tiny net keeps layer 0 narrow and caps
    # accuracy. Uniform growth traces the balanced dense nets (3xk) from a
    # tiny start, so the certified family training matches fixed AdamW.
    # Default False.
    growth_uniform: bool = False
    # Select the growth layer by the largest CERTIFIED functional descent
    # per added parameter (Prop. 3.8 measured descent), instead of the
    # relative-error certificate. Required with delta growth (function-
    # preserving False, compute_delta True): there the GroMo optimal update
    # reduces the loss but jumps the tangent linearization, so the
    # rel-error certificate is blind to which layer actually helps. This is
    # the paper's structural step made parameter-efficient. Default False.
    growth_select_by_descent: bool = False
    # Growth layer selection among probes that improve the certificate.
    # False (default): frugal-first (fewest added parameters, then lowest
    # post-growth relative error). True: lowest post-growth relative error
    # first, so growth widens the most impactful layer even when it is the
    # expensive input layer — required on MNIST, where layer-0 width drives
    # accuracy and the frugal tie-break otherwise starves it.
    growth_prefer_lower_error: bool = False
    # Hard parameter budget: once the model has at least this many total
    # parameters, structural growth is suppressed and the flow keeps
    # training the fixed structure through the certified families. None
    # means no cap. Keeps a grow-and-train run inside a target budget.
    max_total_parameters: int | None = None
    # Structure-burst patience: the growth probe runs only after this many
    # CONSECUTIVE epochs in which no family committed a step. With a value
    # above 1, combine with family_rejection_cooldown: 0 so the stochastic
    # parametric generators actually retry during the patience window.
    # 1 = probe growth on the first fully-failed epoch (legacy).
    growth_patience: int = 1
    # Acceptance mode. When true, an outer step commits on its four LOCAL
    # conditions only — valid sensor, strict Crel, strict LR interval
    # (theory_lr_min < eta < eta_bar), and STRICT realized descent of the
    # validation functional loss — while the stationary and global bounds
    # are computed and logged as trajectory diagnostics. When false (the
    # default), the legacy gates apply: non-strict descent and Cstat/Cglob
    # as acceptance conditions.
    local_acceptance_conditions: bool = False
    # Cooldown for rejected fallback families, measured in ACCEPTED outer
    # steps: a family rejected at theta_t is skipped until this many model
    # updates have been committed since the rejection (weight updates change
    # the tangent space, so rejection must never be permanent at a fixed
    # architecture). Growth clears all rejection state immediately; 0
    # disables the memory (families are retried every epoch).
    family_rejection_cooldown: int = 5
    # In-ladder rkhs_head acceptance margin: the committed head must improve
    # the FULL validation functional loss by at least this relative amount.
    # The phase's internal acceptance compares subsampled train losses
    # (fgd_rkhs.max_train_points), so without this external margin the
    # per-epoch subsample jitter re-certifies an epsilon "improvement"
    # forever and starves every family below rkhs_head in the ladder.
    rkhs_family_min_relative_improvement: float = 1e-3


SUPPORTED_FGD_FAMILIES = (
    "tangent",
    "rkhs_head",
    "parametric_gd",
    "parametric_descent",
)


def validate_functional_loss(functional_loss: str) -> None:
    """Reject an unknown certified functional at config-load time."""
    if functional_loss not in _FUNCTIONAL_LOSSES:
        raise ValueError(
            f"Unsupported fgd_approx.functional_loss '{functional_loss}'. "
            f"Use one of: {', '.join(sorted(_FUNCTIONAL_LOSSES))}."
        )


def validate_family_order(family_order: tuple[str, ...]) -> None:
    """Reject malformed fgd_approx.family_order values early."""
    if not family_order:
        raise ValueError("fgd_approx.family_order cannot be empty.")
    if family_order[0] != "tangent":
        raise ValueError(
            "fgd_approx.family_order must start with 'tangent' (the epoch's "
            "main transactional search)."
        )
    unknown = sorted(set(family_order) - set(SUPPORTED_FGD_FAMILIES))
    if unknown:
        raise ValueError(
            "Unsupported fgd_approx.family_order entries: "
            + ", ".join(unknown)
            + f". Supported: {', '.join(SUPPORTED_FGD_FAMILIES)}."
        )
    if len(set(family_order)) != len(family_order):
        raise ValueError("fgd_approx.family_order entries must be unique.")


@dataclass(frozen=True)
class ParametricGDConfig:
    """Parametric-GD secant family (calibrated projection at the output).

    A disposable clone is trained parametrically toward the functional target
    f - eta_nominal * r. The realized output displacement Delta = F(base) -
    F(candidate) is then screened on validation: its directional cosine
    against the functional gradient must reach ``min_cosine`` (with the
    scale-optimal declared learning rate eta* = <Delta, r>/|r|^2 the best
    achievable relative error is exactly sqrt(1 - cos^2), so cosines below
    sqrt(1 - eps^2) can never satisfy Crel). Surviving candidates are
    certified at eta* with the SAME secant certificate and full transactional
    conditions as every other family.
    """

    optimizer: str = "sgd"
    inner_learning_rate: float = 0.05
    inner_steps: tuple[int, ...] = (16, 64)
    functional_learning_rates: tuple[float, ...] = (0.2, 0.05)
    min_cosine: float = 0.9
    parameter_penalty: float = 1e-6
    gradient_clip_norm: float | None = 1.0
    # Decoupled weight decay for the adam/adamw inner optimizer. Higher
    # values regularize the generated candidate so its realized
    # displacement generalizes to validation (closing the train/val gap
    # that caps the certified acceptance).
    weight_decay: float = 0.0

    def validate(self) -> None:
        if self.optimizer not in ("sgd", "adam", "adamw"):
            raise ValueError(
                "parametric_gd.optimizer must be 'sgd', 'adam' or 'adamw'."
            )
        if self.weight_decay < 0.0:
            raise ValueError(
                "parametric_gd.weight_decay must be non-negative."
            )
        if self.inner_learning_rate <= 0.0:
            raise ValueError(
                "parametric_gd.inner_learning_rate must be positive."
            )
        if not self.inner_steps or any(v < 1 for v in self.inner_steps):
            raise ValueError(
                "parametric_gd.inner_steps must contain positive integers."
            )
        if not self.functional_learning_rates or any(
            v <= 0.0 for v in self.functional_learning_rates
        ):
            raise ValueError(
                "parametric_gd.functional_learning_rates must be positive."
            )
        if not 0.0 < self.min_cosine <= 1.0:
            raise ValueError("parametric_gd.min_cosine must lie in (0, 1].")
        if self.parameter_penalty < 0.0:
            raise ValueError(
                "parametric_gd.parameter_penalty must be non-negative."
            )


@dataclass(frozen=True)
class ParametricDescentConfig:
    """Parametric family certified by MEASURED functional descent.

    Same candidate generation as parametric_gd (clone trained toward the
    functional target f - eta_nominal * r; note eta_nominal = 0.5 makes the
    target exactly y, i.e. plain parametric loss descent). Acceptance does
    NOT go through the relative-error route: for the empirical sum-MSE
    functional the function-space PL inequality is the exact identity
    |grad L|^2 = 4 L (mu = 2, L* = 0, the configured theory constants), so
    Proposition 3.8's contraction only needs the per-step descent
    inequality — which is MEASURED on validation instead of lower-bounded
    via epsilon. The measured descent coefficient
    r_t = (L_t - L_{t+1}) / (eta* |grad L_t|^2) plugs into the same Cprog,
    Cstat and Cglob accumulators; the contraction it certifies equals the
    realized loss ratio exactly.
    """

    optimizer: str = "sgd"
    inner_learning_rate: float = 0.05
    inner_steps: tuple[int, ...] = (16, 64)
    functional_learning_rates: tuple[float, ...] = (0.5, 0.2)
    # Optional direction screen. 0.0 only requires a descent direction in
    # function space (eta* > 0); raise it to demand tangent-like alignment.
    min_cosine: float = 0.0
    parameter_penalty: float = 1e-6
    gradient_clip_norm: float | None = 1.0
    # Decoupled weight decay for the adam/adamw inner optimizer. Higher
    # values regularize the generated candidate so its realized
    # displacement generalizes to validation (closing the train/val gap
    # that caps the certified acceptance).
    weight_decay: float = 0.0
    # Cprog floor on the measured progress eta* r_t = D_t / |grad L_t|^2
    # (= D/4L). Every accepted step must remove at least 4*min_progress of
    # the remaining loss; below that the structure is treated as exhausted
    # for this family, which is what lets growth take over instead of the
    # family certifying ever-smaller crumbs forever.
    min_progress: float = 1e-3

    def validate(self) -> None:
        if self.optimizer not in ("sgd", "adam", "adamw"):
            raise ValueError(
                "parametric_descent.optimizer must be 'sgd', 'adam' or "
                "'adamw'."
            )
        if self.inner_learning_rate <= 0.0:
            raise ValueError(
                "parametric_descent.inner_learning_rate must be positive."
            )
        if self.weight_decay < 0.0:
            raise ValueError(
                "parametric_descent.weight_decay must be non-negative."
            )
        if not self.inner_steps or any(v < 1 for v in self.inner_steps):
            raise ValueError(
                "parametric_descent.inner_steps must contain positive "
                "integers."
            )
        if not self.functional_learning_rates or any(
            v <= 0.0 for v in self.functional_learning_rates
        ):
            raise ValueError(
                "parametric_descent.functional_learning_rates must be "
                "positive."
            )
        if not 0.0 <= self.min_cosine <= 1.0:
            raise ValueError(
                "parametric_descent.min_cosine must lie in [0, 1]."
            )
        if self.parameter_penalty < 0.0:
            raise ValueError(
                "parametric_descent.parameter_penalty must be non-negative."
            )
        if self.min_progress <= 0.0:
            raise ValueError(
                "parametric_descent.min_progress must be positive."
            )


@dataclass(frozen=True)
class SecantFGDConfig:
    enabled: bool = True
    inner_steps: int = 8
    inner_learning_rate: float = 1e-2
    parameter_penalty: float = 1e-6
    gradient_clip_norm: float | None = 1.0
    search_steps: int = 3
    max_learning_rate: float = 0.1
    min_learning_rate_factor: float = 1e-2
    growth_min_relative_error_improvement: float = 1e-3
    growth_min_learning_rate_improvement: float = 0.05


@dataclass(frozen=True)
class FGDLayerRelError:
    layer_index: int
    relative_error: float
    approximation_norm: float
    target_norm: float
    directional_cosine: float


@dataclass(frozen=True)
class FGDOutputRelError:
    relative_error: float
    approximation_norm: float
    target_norm: float
    directional_cosine: float


@dataclass(frozen=True)
class FGDApproxEpochResult:
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    learning_rate: float | None
    next_learning_rate: float | None
    learning_rate_upper_bound: float | None
    learning_rate_interval_valid: bool | None
    learning_rate_clipped_batches: int
    skipped_batches: int
    relative_error_condition_valid: bool | None
    loss_descent_valid: bool | None
    loss_non_descent_batches: int
    gradient_sq_norm: float | None
    theory_descent_coefficient: float | None
    min_positive_learning_rate: float | None
    relative_error: float | None
    selected_layer_index: int | None
    layer_relative_errors: list[FGDLayerRelError]
    output_relative_error: FGDOutputRelError | None
    sensor_valid: bool
    sensor_invalid_batches: int


@dataclass(frozen=True)
class FGDValidationCertificate:
    learning_rate_upper_bound: float | None
    max_valid_learning_rate: float | None
    learning_rate_interval_valid: bool | None
    skipped_batches: int
    relative_error_condition_valid: bool | None
    gradient_sq_norm: float | None
    theory_descent_coefficient: float | None
    relative_error: float | None
    output_relative_error: FGDOutputRelError | None
    sensor_valid: bool
    sensor_invalid_batches: int


@dataclass(frozen=True)
class _FunctionalStepStats:
    output_error: FGDOutputRelError
    dot_product: float
    approximation_sq_norm: float
    target_sq_norm: float


@dataclass(frozen=True)
class _TangentProjectionStep:
    output_error: FGDOutputRelError
    parameter_updates: tuple[torch.Tensor, ...]
    learning_rate_used: float
    loss_before: float
    loss_after: float
    descent_ok: bool
    dot_product: float
    approximation_sq_norm: float
    target_sq_norm: float


def mse_functional_gradient(
    y_pred: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return the output gradient for GroMo/TINY sum-MSE convention."""
    return 2.0 * (y_pred.detach() - y.detach())


def batch_functional_mse_loss(
    y_pred: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return the sum-MSE loss used by GroMo/TINY statistics."""
    return torch.sum((y_pred - y) ** 2)


def cross_entropy_functional_gradient(
    y_pred: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return the functional gradient of summed softmax cross-entropy.

    With logits ``f`` and one-hot targets ``Y``,
    ``L(f) = sum_i [logsumexp(f_i) - f_{i,c_i}]`` and

        r = grad_f L = softmax(f) - Y.

    Unlike the sum-MSE gradient ``2(f-Y)``, this vanishes only as the
    correct-class logit diverges, so it never opposes additional confidence
    on an already-correct sample. See report/CROSS_ENTROPY_FGD.md §4.
    """
    return torch.softmax(y_pred.detach(), dim=-1) - y.detach()


def batch_functional_cross_entropy_loss(
    y_pred: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return summed softmax cross-entropy against one-hot targets."""
    log_probabilities = torch.log_softmax(y_pred, dim=-1)
    return -torch.sum(y * log_probabilities)


# The certified functional. Every certificate in this module is a statement
# about THIS loss; see report/CROSS_ENTROPY_FGD.md for which parts of the
# theory transfer between the two (convexity and smoothness do, the
# Polyak-Lojasiewicz constant does not).

_FUNCTIONAL_GRADIENTS = {
    "mse": mse_functional_gradient,
    "cross_entropy": cross_entropy_functional_gradient,
}
_FUNCTIONAL_LOSSES = {
    "mse": batch_functional_mse_loss,
    "cross_entropy": batch_functional_cross_entropy_loss,
}
# Smoothness constant L_s of the functional in output space:
#   MSE:           Hessian = 2 Id                      -> L_s = 2
#   cross-entropy: Hessian = diag(p) - p p^T, PSD,     -> L_s = 1/2
#                  lambda_max <= 1/2 (verified numerically)
# Lemma 3.5's admissible interval is eta_bar = 2(1-2eps)/(L_s(1+2eps)), so
# the cross-entropy interval is four times wider.
FUNCTIONAL_SMOOTHNESS = {"mse": 2.0, "cross_entropy": 0.5}
# Whether the functional admits a global Polyak-Lojasiewicz constant. Only
# sum-MSE does (||r||^2 = 4L exactly, mu = 2, L* = 0). Cross-entropy has
# ||r||^2 = Theta(L^2) as p_c -> 1 and bounded ||r||^2 with unbounded L as
# p_c -> 0, so no mu > 0 exists and the LINEAR contraction of Prop. 3.8
# (the C_glob envelope) is not available for it.
FUNCTIONAL_HAS_PL_CONSTANT = {"mse": True, "cross_entropy": False}


def functional_gradient(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    functional_loss: FunctionalLoss = "mse",
) -> torch.Tensor:
    """Return r = grad_f L for the configured certified functional."""
    try:
        return _FUNCTIONAL_GRADIENTS[functional_loss](y_pred, y)
    except KeyError:
        raise ValueError(
            f"Unsupported fgd_approx.functional_loss '{functional_loss}'. "
            f"Use one of: {', '.join(_FUNCTIONAL_GRADIENTS)}."
        ) from None


def batch_functional_loss(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    functional_loss: FunctionalLoss = "mse",
) -> torch.Tensor:
    """Return L(f) on a batch for the configured certified functional."""
    try:
        return _FUNCTIONAL_LOSSES[functional_loss](y_pred, y)
    except KeyError:
        raise ValueError(
            f"Unsupported fgd_approx.functional_loss '{functional_loss}'. "
            f"Use one of: {', '.join(_FUNCTIONAL_LOSSES)}."
        ) from None


def relative_l2_error(
    approximation: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> float:
    """Compute ||approximation - target|| / ||approximation|| on a batch."""
    numerator = torch.sqrt(torch.mean((approximation - target) ** 2))
    denominator = torch.sqrt(torch.mean(approximation**2)).clamp_min(eps)
    return float((numerator / denominator).detach().item())


def _output_relative_error_from_tensors(
    approximation: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> _FunctionalStepStats:
    dot_product = float(torch.sum(approximation * target).detach().item())
    approximation_sq_norm = float(torch.sum(approximation**2).detach().item())
    target_sq_norm = float(torch.sum(target**2).detach().item())
    output_error = _output_relative_error_from_stats(
        dot_product=dot_product,
        approximation_sq_norm=approximation_sq_norm,
        target_sq_norm=target_sq_norm,
        eps=eps,
    )
    return _FunctionalStepStats(
        output_error=output_error,
        dot_product=dot_product,
        approximation_sq_norm=approximation_sq_norm,
        target_sq_norm=target_sq_norm,
    )


def tiny_optimal_update_kwargs(
    config: FGDApproxConfig,
    *,
    compute_delta: bool,
) -> dict[str, object]:
    """Return GroMo options matching the TINY-style optimal layer update."""
    return {
        "compute_delta": compute_delta,
        "use_covariance": config.tiny_use_covariance,
        "alpha_zero": config.tiny_alpha_zero,
        "omega_zero": config.tiny_omega_zero,
        "use_projection": config.tiny_use_projection,
        "ignore_singular_values": config.tiny_ignore_singular_values,
        "use_fisher": config.tiny_use_fisher,
        "maximum_added_neurons": config.tiny_maximum_added_neurons,
        "numerical_threshold": config.tiny_numerical_threshold,
        "statistical_threshold": config.tiny_statistical_threshold,
    }


def _cleanup_tiny_update(model: GrowingMLP) -> None:
    model.reset_computation()
    for layer in getattr(model, "_growing_layers", []):
        if hasattr(layer, "delete_update"):
            layer.delete_update(include_previous=True)
    model.currently_updated_layer_index = None
    model.zero_grad(set_to_none=True)


def _forward_with_tiny_update(
    model: GrowingMLP,
    x: torch.Tensor,
) -> torch.Tensor:
    """Forward the temporary layer delta all the way to the network output."""
    x = model.flatten(x)
    x_ext = None
    for layer in model.layers:
        x, x_ext = layer.extended_forward(
            x,
            x_ext,
            use_optimal_delta=True,
            use_extended_input=False,
            use_extended_output=False,
        )
    return x


def _relative_error_from_stats(
    *,
    layer_index: int,
    dot_product: float,
    approximation_sq_norm: float,
    target_sq_norm: float,
    config: FGDApproxConfig,
) -> FGDLayerRelError:
    numerator_sq = (
        approximation_sq_norm
        - 2.0 * dot_product
        + target_sq_norm
    )
    numerator = math.sqrt(max(0.0, numerator_sq))

    denominator_sq = approximation_sq_norm
    denominator = max(math.sqrt(max(0.0, denominator_sq)), config.eps)
    approximation_norm = math.sqrt(max(0.0, approximation_sq_norm))
    target_norm = math.sqrt(max(0.0, target_sq_norm))

    cosine_denominator = max(
        math.sqrt(max(0.0, approximation_sq_norm * target_sq_norm)),
        config.eps,
    )
    directional_cosine = dot_product / cosine_denominator

    return FGDLayerRelError(
        layer_index=layer_index,
        relative_error=numerator / denominator,
        approximation_norm=approximation_norm,
        target_norm=target_norm,
        directional_cosine=directional_cosine,
    )


def _output_relative_error_from_stats(
    *,
    dot_product: float,
    approximation_sq_norm: float,
    target_sq_norm: float,
    eps: float,
) -> FGDOutputRelError:
    numerator_sq = approximation_sq_norm - 2.0 * dot_product + target_sq_norm
    numerator = math.sqrt(max(0.0, numerator_sq))
    denominator = max(math.sqrt(max(0.0, approximation_sq_norm)), eps)
    approximation_norm = math.sqrt(max(0.0, approximation_sq_norm))
    target_norm = math.sqrt(max(0.0, target_sq_norm))
    cosine_denominator = max(
        math.sqrt(max(0.0, approximation_sq_norm * target_sq_norm)),
        eps,
    )
    directional_cosine = dot_product / cosine_denominator
    return FGDOutputRelError(
        relative_error=numerator / denominator,
        approximation_norm=approximation_norm,
        target_norm=target_norm,
        directional_cosine=directional_cosine,
    )


def _projection_sensor_valid(
    *,
    dot_product: float,
    approximation_sq_norm: float,
    target_sq_norm: float,
    eps: float,
    relative_tolerance: float = 1e-4,
) -> bool:
    """Return whether the damped projection satisfies exact-operator invariants."""
    values = (dot_product, approximation_sq_norm, target_sq_norm)
    if not all(math.isfinite(value) for value in values):
        return False
    if approximation_sq_norm < -eps or target_sq_norm < -eps:
        return False

    approximation_sq_norm = max(0.0, approximation_sq_norm)
    target_sq_norm = max(0.0, target_sq_norm)
    norm_product = math.sqrt(approximation_sq_norm * target_sq_norm)
    dot_tolerance = relative_tolerance * max(norm_product, eps)
    if dot_product < -dot_tolerance:
        return False

    norm_tolerance = relative_tolerance * max(target_sq_norm, eps)
    if approximation_sq_norm > target_sq_norm + norm_tolerance:
        return False

    return True


def _projection_step_sensor_valid(
    step: _TangentProjectionStep,
    config: FGDApproxConfig,
) -> bool:
    return _projection_sensor_valid(
        dot_product=step.dot_product,
        approximation_sq_norm=step.approximation_sq_norm,
        target_sq_norm=step.target_sq_norm,
        eps=config.eps,
    )


def certified_smoothness_constant(config: FGDApproxConfig) -> float:
    """Return the smoothness L_s of the CERTIFIED functional.

    Lemma 3.5's admissible interval is eta_bar = 2(1-2eps)/(L_s(1+2eps)),
    so L_s must describe the loss actually being certified: 2 for sum-MSE
    (Hessian 2*Id) and 1/2 for softmax cross-entropy (Hessian
    diag(p)-p p^T, lambda_max <= 1/2), which makes the cross-entropy
    interval four times wider. When theory_smoothness_constant is left at
    its MSE default the constant of the configured functional is used;
    an explicitly non-default value always wins, so the knob stays usable.
    """
    configured = float(config.theory_smoothness_constant)
    default_mse = FUNCTIONAL_SMOOTHNESS["mse"]
    if configured == default_mse:
        return FUNCTIONAL_SMOOTHNESS.get(config.functional_loss, configured)
    return configured


def theoretical_learning_rate_upper_bound(
    relative_error: float,
    config: FGDApproxConfig,
) -> float | None:
    """Return the FGD learning-rate upper bound from the current RelErr."""
    if relative_error < 0.0 or relative_error >= 0.5:
        return None

    smoothness = float(certified_smoothness_constant(config))
    if smoothness <= 0.0:
        raise ValueError("fgd_approx.theory_smoothness_constant must be positive.")

    return 2.0 * (1.0 - 2.0 * relative_error) / (
        smoothness * (2.0 * relative_error + 1.0)
    )


def theoretical_descent_coefficient(
    relative_error: float,
    learning_rate: float,
    config: FGDApproxConfig,
) -> float | None:
    """Return the proof coefficient r for the approximate FGD step."""
    if relative_error < 0.0 or relative_error >= 1.0:
        return None
    if learning_rate <= 0.0:
        return None

    smoothness = float(certified_smoothness_constant(config))
    alpha = float(config.theory_alpha)
    beta = float(config.theory_beta)
    if smoothness <= 0.0:
        raise ValueError("fgd_approx.theory_smoothness_constant must be positive.")
    if alpha <= 0.0:
        raise ValueError("fgd_approx.theory_alpha must be positive.")
    if beta <= 0.0:
        raise ValueError("fgd_approx.theory_beta must be positive.")

    error_ratio = relative_error / max(1.0 - relative_error, config.eps)
    return (
        alpha
        - 0.5 * smoothness * learning_rate
        - beta * error_ratio
        - 1.5 * smoothness * learning_rate * error_ratio
    )


def _clear_inaccessible_tensor_caches(model: torch.nn.Module) -> None:
    """Drop functorch-wrapped tensors cached by GroMo modules during jacrev."""
    for module in model.modules():
        for name, value in list(vars(module).items()):
            if torch.is_tensor(value):
                try:
                    value.untyped_storage()
                except (NotImplementedError, RuntimeError):
                    setattr(module, name, None)


def _trainable_named_parameters(
    model: torch.nn.Module,
) -> OrderedDict[str, torch.nn.Parameter]:
    return OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def _flatten_jacobian(
    jacobian: tuple[torch.Tensor, ...],
    output_numel: int,
) -> torch.Tensor:
    columns = [item.reshape(output_numel, -1) for item in jacobian]
    return torch.cat(columns, dim=1).detach()


def _unflatten_parameter_update(
    flat_update: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor, ...]:
    updates: list[torch.Tensor] = []
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        updates.append(flat_update[offset : offset + size].reshape_as(parameter))
        offset += size
    return tuple(updates)


def _flatten_parameter_tensors(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _zero_tangent_projection(
    jacobian_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_numel, parameter_numel = jacobian_matrix.shape
    original_dtype = jacobian_matrix.dtype
    return (
        torch.zeros(
            parameter_numel,
            device=jacobian_matrix.device,
            dtype=original_dtype,
        ),
        torch.zeros(
            output_numel,
            device=jacobian_matrix.device,
            dtype=original_dtype,
        ),
    )


def _solve_tangent_projection_svd(
    jacobian_matrix: torch.Tensor,
    target: torch.Tensor,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the damped tangent projection with a float64 SVD of J."""
    output_numel, parameter_numel = jacobian_matrix.shape
    damping = max(float(damping), 0.0)
    original_dtype = jacobian_matrix.dtype
    work_dtype = torch.float64
    jacobian_work = jacobian_matrix.to(dtype=work_dtype)
    target_work = target.reshape(-1).to(dtype=work_dtype)
    if output_numel == 0 or parameter_numel == 0:
        return _zero_tangent_projection(jacobian_matrix)

    u, singular_values, vh = torch.linalg.svd(jacobian_work, full_matrices=False)
    if singular_values.numel() == 0:
        return _zero_tangent_projection(jacobian_matrix)

    coefficients = u.t() @ target_work
    eigenvalues = singular_values.square()

    if damping > 0.0:
        denominator = eigenvalues + damping
        output_factors = eigenvalues / denominator
        update_factors = singular_values / denominator
    else:
        max_singular = torch.max(singular_values)
        threshold = (
            torch.finfo(work_dtype).eps
            * max(output_numel, parameter_numel)
            * max(float(max_singular.detach().item()), 1.0)
        )
        nonzero = singular_values > threshold
        output_factors = torch.where(
            nonzero,
            torch.ones_like(singular_values),
            torch.zeros_like(singular_values),
        )
        update_factors = torch.where(
            nonzero,
            singular_values.reciprocal(),
            torch.zeros_like(singular_values),
        )

    approximation_work = u @ (output_factors * coefficients)
    flat_update_work = vh.t() @ (update_factors * coefficients)
    return (
        flat_update_work.to(dtype=original_dtype),
        approximation_work.to(dtype=original_dtype),
    )


def _solve_tangent_projection_kernel_eigh(
    jacobian_matrix: torch.Tensor,
    target: torch.Tensor,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the damped tangent projection with a float64 eigendecomposition of K."""
    output_numel, parameter_numel = jacobian_matrix.shape
    damping = max(float(damping), 0.0)
    original_dtype = jacobian_matrix.dtype
    work_dtype = torch.float64
    jacobian_work = jacobian_matrix.to(dtype=work_dtype)
    target_work = target.reshape(-1).to(dtype=work_dtype)
    if output_numel == 0 or parameter_numel == 0:
        return _zero_tangent_projection(jacobian_matrix)

    kernel = jacobian_work @ jacobian_work.t()
    kernel = 0.5 * (kernel + kernel.t())
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    eigenvalues = eigenvalues.clamp_min(0.0)
    if eigenvalues.numel() == 0:
        return _zero_tangent_projection(jacobian_matrix)

    coefficients = eigenvectors.t() @ target_work
    if damping > 0.0:
        denominator = eigenvalues + damping
        output_factors = eigenvalues / denominator
        dual_factors = denominator.reciprocal()
    else:
        max_eigenvalue = torch.max(eigenvalues)
        threshold = (
            torch.finfo(work_dtype).eps
            * output_numel
            * max(float(max_eigenvalue.detach().item()), 1.0)
        )
        nonzero = eigenvalues > threshold
        output_factors = torch.where(
            nonzero,
            torch.ones_like(eigenvalues),
            torch.zeros_like(eigenvalues),
        )
        dual_factors = torch.where(
            nonzero,
            eigenvalues.reciprocal(),
            torch.zeros_like(eigenvalues),
        )

    dual_solution_work = eigenvectors @ (dual_factors * coefficients)
    approximation_work = eigenvectors @ (output_factors * coefficients)
    flat_update_work = jacobian_work.t() @ dual_solution_work
    return (
        flat_update_work.to(dtype=original_dtype),
        approximation_work.to(dtype=original_dtype),
    )


def _solve_tangent_projection(
    jacobian_matrix: torch.Tensor,
    target: torch.Tensor,
    damping: float,
    solver: ProjectionSolver = "exact",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return parameter update and projected output gradient."""
    if solver in {"exact", "exact_svd"}:
        return _solve_tangent_projection_svd(
            jacobian_matrix=jacobian_matrix,
            target=target,
            damping=damping,
        )
    if solver == "exact_kernel_eigh":
        return _solve_tangent_projection_kernel_eigh(
            jacobian_matrix=jacobian_matrix,
            target=target,
            damping=damping,
        )
    raise ValueError(
        f"Unsupported exact projection solver '{solver}'. "
        "Use one of: exact, exact_svd, exact_kernel_eigh."
    )


def _conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    max_iterations: int,
    tolerance: float,
    eps: float,
) -> torch.Tensor:
    """Solve A x = rhs using conjugate gradient with an implicit SPD matvec."""
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rhs_norm = torch.linalg.norm(rhs).clamp_min(eps)
    residual_sq = torch.dot(r, r)
    if torch.sqrt(residual_sq) <= tolerance * rhs_norm:
        return x

    for _ in range(max(1, max_iterations)):
        ap = matvec(p)
        denominator = torch.dot(p, ap)
        if torch.abs(denominator) <= eps:
            break

        alpha = residual_sq / denominator
        x = x + alpha * p
        r = r - alpha * ap
        next_residual_sq = torch.dot(r, r)
        if torch.sqrt(next_residual_sq) <= tolerance * rhs_norm:
            break

        beta = next_residual_sq / residual_sq.clamp_min(eps)
        p = r + beta * p
        residual_sq = next_residual_sq

    return x


def _compute_exact_tangent_projection_step(
    model: GrowingMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> _TangentProjectionStep:
    """Compute g = P_T grad L by explicitly materializing the full Jacobian."""
    named_parameters = _trainable_named_parameters(model)
    if not named_parameters:
        raise RuntimeError("FGD tangent projection requires trainable parameters.")

    parameter_names = tuple(named_parameters.keys())
    parameters = tuple(named_parameters.values())
    buffers = OrderedDict(model.named_buffers())

    output = model(x)
    loss = batch_functional_loss(output, y, config.functional_loss)
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"Non-finite FGD loss detected before projection: {loss}.")

    target_tensor = torch.autograd.grad(loss, output)[0].detach()
    target = target_tensor.reshape(-1)
    output_numel = target.numel()

    if torch.linalg.norm(target) <= torch.finfo(target.dtype).eps:
        zero_updates = tuple(torch.zeros_like(parameter) for parameter in parameters)
        output_error = _output_relative_error_from_stats(
            dot_product=0.0,
            approximation_sq_norm=0.0,
            target_sq_norm=0.0,
            eps=config.eps,
        )
        return _TangentProjectionStep(
            output_error=output_error,
            parameter_updates=zero_updates,
            learning_rate_used=0.0,
            loss_before=float(loss.detach().item()),
            loss_after=float(loss.detach().item()),
            descent_ok=True,
            dot_product=0.0,
            approximation_sq_norm=0.0,
            target_sq_norm=0.0,
        )

    def call_with_parameters(
        parameter_values: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        state = OrderedDict(zip(parameter_names, parameter_values))
        state.update(buffers)
        return functional_call(model, state, (x,)).reshape(-1)

    try:
        jacobian = jacrev(call_with_parameters)(parameters)
    finally:
        _clear_inaccessible_tensor_caches(model)

    jacobian_matrix = _flatten_jacobian(jacobian, output_numel)
    flat_update, approximation = _solve_tangent_projection(
        jacobian_matrix=jacobian_matrix,
        target=target,
        damping=config.projection_damping,
        solver=config.projection_solver,
    )
    if not torch.isfinite(flat_update).all() or not torch.isfinite(approximation).all():
        raise RuntimeError("Non-finite FGD tangent projection update detected.")

    stats = _output_relative_error_from_tensors(
        approximation=approximation,
        target=target,
        eps=config.eps,
    )
    parameter_updates = _unflatten_parameter_update(flat_update, parameters)
    return _TangentProjectionStep(
        output_error=stats.output_error,
        parameter_updates=parameter_updates,
        learning_rate_used=0.0,
        loss_before=float(loss.detach().item()),
        loss_after=float(loss.detach().item()),
        descent_ok=True,
        dot_product=stats.dot_product,
        approximation_sq_norm=stats.approximation_sq_norm,
        target_sq_norm=stats.target_sq_norm,
    )


def _compute_cg_tangent_projection_step(
    model: GrowingMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> _TangentProjectionStep:
    """Compute g = P_T grad L with implicit CG products through the Jacobian."""
    named_parameters = _trainable_named_parameters(model)
    if not named_parameters:
        raise RuntimeError("FGD tangent projection requires trainable parameters.")

    parameter_names = tuple(named_parameters.keys())
    parameters = tuple(named_parameters.values())
    buffers = OrderedDict(model.named_buffers())

    output = model(x)
    loss = batch_functional_loss(output, y, config.functional_loss)
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"Non-finite FGD loss detected before projection: {loss}.")

    target_tensor = torch.autograd.grad(loss, output, retain_graph=True)[0].detach()
    target = target_tensor.reshape(-1)
    output_numel = target.numel()
    parameter_numel = sum(parameter.numel() for parameter in parameters)

    if torch.linalg.norm(target) <= torch.finfo(target.dtype).eps:
        zero_updates = tuple(torch.zeros_like(parameter) for parameter in parameters)
        output_error = _output_relative_error_from_stats(
            dot_product=0.0,
            approximation_sq_norm=0.0,
            target_sq_norm=0.0,
            eps=config.eps,
        )
        return _TangentProjectionStep(
            output_error=output_error,
            parameter_updates=zero_updates,
            learning_rate_used=0.0,
            loss_before=float(loss.detach().item()),
            loss_after=float(loss.detach().item()),
            descent_ok=True,
            dot_product=0.0,
            approximation_sq_norm=0.0,
            target_sq_norm=0.0,
        )

    def call_with_parameters(
        parameter_values: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        state = OrderedDict(zip(parameter_names, parameter_values))
        state.update(buffers)
        return functional_call(model, state, (x,))

    def jvp_parameters_to_output(
        parameter_tangent: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _, tangent_output = jvp(
            call_with_parameters,
            (parameters,),
            (parameter_tangent,),
        )
        return tangent_output.reshape(-1).detach()

    def vjp_output_to_parameters(vector: torch.Tensor) -> tuple[torch.Tensor, ...]:
        gradients = torch.autograd.grad(
            output,
            parameters,
            grad_outputs=vector.reshape_as(output),
            retain_graph=True,
            allow_unused=True,
        )
        return tuple(
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for parameter, gradient in zip(parameters, gradients)
        )

    def output_kernel_matvec(vector: torch.Tensor) -> torch.Tensor:
        parameter_tangent = vjp_output_to_parameters(vector)
        return jvp_parameters_to_output(parameter_tangent)

    damping = max(float(config.projection_damping), 0.0)

    def damped_output_kernel_matvec(vector: torch.Tensor) -> torch.Tensor:
        product = output_kernel_matvec(vector)
        if damping > 0.0:
            product = product + damping * vector
        return product

    try:
        if output_numel <= parameter_numel:
            dual_solution = _conjugate_gradient(
                damped_output_kernel_matvec,
                target,
                max_iterations=config.cg_max_iterations,
                tolerance=config.cg_tolerance,
                eps=config.eps,
            )
            approximation = output_kernel_matvec(dual_solution)
            parameter_updates = vjp_output_to_parameters(dual_solution)
        else:
            rhs = _flatten_parameter_tensors(vjp_output_to_parameters(target))

            def damped_parameter_gram_matvec(vector: torch.Tensor) -> torch.Tensor:
                parameter_tangent = _unflatten_parameter_update(vector, parameters)
                output_tangent = jvp_parameters_to_output(parameter_tangent)
                product = _flatten_parameter_tensors(
                    vjp_output_to_parameters(output_tangent)
                )
                if damping > 0.0:
                    product = product + damping * vector
                return product

            flat_update = _conjugate_gradient(
                damped_parameter_gram_matvec,
                rhs,
                max_iterations=config.cg_max_iterations,
                tolerance=config.cg_tolerance,
                eps=config.eps,
            )
            parameter_updates = _unflatten_parameter_update(flat_update, parameters)
            approximation = jvp_parameters_to_output(parameter_updates)
    finally:
        _clear_inaccessible_tensor_caches(model)

    if (
        not torch.isfinite(approximation).all()
        or any(not torch.isfinite(update).all() for update in parameter_updates)
    ):
        raise RuntimeError("Non-finite FGD CG tangent projection update detected.")

    stats = _output_relative_error_from_tensors(
        approximation=approximation,
        target=target,
        eps=config.eps,
    )
    return _TangentProjectionStep(
        output_error=stats.output_error,
        parameter_updates=parameter_updates,
        learning_rate_used=0.0,
        loss_before=float(loss.detach().item()),
        loss_after=float(loss.detach().item()),
        descent_ok=True,
        dot_product=stats.dot_product,
        approximation_sq_norm=stats.approximation_sq_norm,
        target_sq_norm=stats.target_sq_norm,
    )


def _grouped_batches(
    data_loader: torch.utils.data.DataLoader,
    group_size: int,
):
    """Yield concatenations of ``group_size`` consecutive mini-batches."""
    if group_size <= 1:
        yield from data_loader
        return

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in data_loader:
        xs.append(x)
        ys.append(y)
        if len(xs) == group_size:
            yield torch.cat(xs), torch.cat(ys)
            xs, ys = [], []
    if xs:
        yield torch.cat(xs), torch.cat(ys)


def build_projection_probe(
    data_loader: torch.utils.data.DataLoader,
    probe_batches: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate the first ``probe_batches`` mini-batches into one probe.

    The probe is the joint certification sample: every certificate solves a
    SINGLE shared-direction projection over it (conceptually the stacked
    system J_probe u = r_probe). Materialize the probe once per data source
    and pass the same tensors to every certificate evaluation, so family
    comparisons, growth-layer trials and consecutive epochs all measure on
    the same fixed probe.
    """
    if probe_batches < 1:
        raise ValueError("fgd_approx.probe_batches must be a positive integer.")
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in data_loader:
        xs.append(x)
        ys.append(y)
        if len(xs) == probe_batches:
            break
    if not xs:
        raise ValueError(
            "Cannot build a projection probe from an empty data loader."
        )
    x = torch.cat(xs)
    y = torch.cat(ys)
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    return x, y


def _compute_tangent_projection_step(
    model: GrowingMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> _TangentProjectionStep:
    """Compute g = P_T grad L and the parameter update that realizes it."""
    if config.projection_solver in {"exact", "exact_svd", "exact_kernel_eigh"}:
        return _compute_exact_tangent_projection_step(
            model=model,
            x=x,
            y=y,
            config=config,
        )

    if config.projection_solver == "cg":
        return _compute_cg_tangent_projection_step(
            model=model,
            x=x,
            y=y,
            config=config,
        )

    raise ValueError(
        f"Unsupported fgd_approx.projection_solver '{config.projection_solver}'. "
        "Use one of: cg, exact, exact_svd, exact_kernel_eigh."
    )


def _apply_tangent_projection_step(
    model: GrowingMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    step: _TangentProjectionStep,
    learning_rate: float,
    config: FGDApproxConfig,
) -> _TangentProjectionStep:
    named_parameters = _trainable_named_parameters(model)
    parameters = tuple(named_parameters.values())
    base_parameters = tuple(parameter.detach().clone() for parameter in parameters)
    base_loss = step.loss_before
    directional_derivative = step.dot_product

    def apply(step_learning_rate: float) -> None:
        with torch.no_grad():
            for parameter, base, update in zip(
                parameters,
                base_parameters,
                step.parameter_updates,
            ):
                parameter.copy_(base)
                parameter.add_(update, alpha=-step_learning_rate)

    @torch.no_grad()
    def trial_loss() -> float:
        return float(
            batch_functional_loss(model(x), y, config.functional_loss)
            .detach()
            .item()
        )

    if learning_rate <= config.eps or directional_derivative <= config.eps:
        apply(0.0)
        return _TangentProjectionStep(
            output_error=step.output_error,
            parameter_updates=step.parameter_updates,
            learning_rate_used=0.0,
            loss_before=step.loss_before,
            loss_after=trial_loss(),
            descent_ok=directional_derivative <= config.eps,
            dot_product=step.dot_product,
            approximation_sq_norm=step.approximation_sq_norm,
            target_sq_norm=step.target_sq_norm,
        )

    if config.learning_rate_policy == "theory_interval":
        # The theoretical interval is certified on held-out validation data by
        # the pipeline. Training batches only realize that certified step.
        apply(learning_rate)
        accepted_loss = trial_loss()
        if not math.isfinite(accepted_loss):
            apply(0.0)
            accepted_loss = trial_loss()
            return _TangentProjectionStep(
                output_error=step.output_error,
                parameter_updates=step.parameter_updates,
                learning_rate_used=0.0,
                loss_before=step.loss_before,
                loss_after=accepted_loss,
                descent_ok=False,
                dot_product=step.dot_product,
                approximation_sq_norm=step.approximation_sq_norm,
                target_sq_norm=step.target_sq_norm,
            )
        return _TangentProjectionStep(
            output_error=step.output_error,
            parameter_updates=step.parameter_updates,
            learning_rate_used=learning_rate,
            loss_before=step.loss_before,
            loss_after=accepted_loss,
            descent_ok=True,
            dot_product=step.dot_product,
            approximation_sq_norm=step.approximation_sq_norm,
            target_sq_norm=step.target_sq_norm,
        )

    if config.sufficient_descent_c is None:
        apply(learning_rate)
        accepted_loss = trial_loss()
        return _TangentProjectionStep(
            output_error=step.output_error,
            parameter_updates=step.parameter_updates,
            learning_rate_used=learning_rate,
            loss_before=step.loss_before,
            loss_after=accepted_loss,
            descent_ok=accepted_loss <= step.loss_before + config.eps,
            dot_product=step.dot_product,
            approximation_sq_norm=step.approximation_sq_norm,
            target_sq_norm=step.target_sq_norm,
        )

    trial_learning_rate = learning_rate
    min_learning_rate = learning_rate * max(config.lr_min_factor, 0.0)
    accepted_learning_rate: float | None = None
    accepted_loss = base_loss
    while trial_learning_rate >= min_learning_rate:
        apply(trial_learning_rate)
        current_loss = trial_loss()
        sufficient_decrease = (
            base_loss
            - config.sufficient_descent_c
            * trial_learning_rate
            * directional_derivative
        )
        if current_loss <= sufficient_decrease:
            accepted_learning_rate = trial_learning_rate
            accepted_loss = current_loss
            break
        trial_learning_rate *= config.lr_backtrack

    descent_ok = accepted_learning_rate is not None
    if accepted_learning_rate is None:
        accepted_learning_rate = max(trial_learning_rate, min_learning_rate)
        apply(accepted_learning_rate)
        accepted_loss = trial_loss()

    return _TangentProjectionStep(
        output_error=step.output_error,
        parameter_updates=step.parameter_updates,
        learning_rate_used=accepted_learning_rate,
        loss_before=step.loss_before,
        loss_after=accepted_loss,
        descent_ok=descent_ok,
        dot_product=step.dot_product,
        approximation_sq_norm=step.approximation_sq_norm,
        target_sq_norm=step.target_sq_norm,
    )


@torch.no_grad()
def _layer_functional_error(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    layer_index: int,
    device: torch.device,
    config: FGDApproxConfig,
) -> FGDLayerRelError:
    model.set_scaling_factor(1.0)
    model.eval()

    dot_product = 0.0
    approximation_sq_norm = 0.0
    target_sq_norm = 0.0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        y_before = model(x)
        y_candidate = _forward_with_tiny_update(model, x)

        approximation = y_before - y_candidate
        target = functional_gradient(y_before, y, config.functional_loss)

        dot_product += float(torch.sum(approximation * target).detach().item())
        approximation_sq_norm += float(torch.sum(approximation**2).detach().item())
        target_sq_norm += float(torch.sum(target**2).detach().item())

    return _relative_error_from_stats(
        layer_index=layer_index,
        dot_product=dot_product,
        approximation_sq_norm=approximation_sq_norm,
        target_sq_norm=target_sq_norm,
        config=config,
    )


def compute_tiny_layer_relative_errors(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: FGDApproxConfig,
) -> list[FGDLayerRelError]:
    """Compute TINY optimal update and functional RelErr for every growable layer."""
    layer_errors: list[FGDLayerRelError] = []
    update_kwargs = tiny_optimal_update_kwargs(
        config,
        compute_delta=config.rel_error_compute_delta,
    )

    for layer_index in range(len(model._growable_layers)):
        model.set_growing_layers(index=layer_index)
        try:
            compute_statistics(
                model,
                train_loader,
                loss_function=batch_functional_mse_loss,
                device=device,
            )
            model.compute_optimal_updates(**update_kwargs)
            model.reset_computation()
            model.dummy_select_update()
            layer_errors.append(
                _layer_functional_error(
                    model=model,
                    train_loader=train_loader,
                    layer_index=layer_index,
                    device=device,
                    config=config,
                )
            )
        finally:
            _cleanup_tiny_update(model)

    return layer_errors


def apply_fgd_parameter_update(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    selected_layer: FGDLayerRelError,
    device: torch.device,
    learning_rate: float,
    config: FGDApproxConfig,
) -> None:
    """Apply one FGD parameter update from GroMo's optimal delta approximation."""
    model.set_growing_layers(index=selected_layer.layer_index)
    try:
        compute_statistics(
            model,
            train_loader,
            loss_function=batch_functional_mse_loss,
            device=device,
        )
        model.compute_optimal_updates(
            **tiny_optimal_update_kwargs(config, compute_delta=True)
        )
        model.reset_computation()
        model.dummy_select_update()
        model.currently_updated_layer.apply_change(
            apply_delta=True,
            apply_extension=False,
            optimal_delta_scaling=learning_rate,
        )
    finally:
        _cleanup_tiny_update(model)


def select_fgd_growth_layer(
    layer_errors: list[FGDLayerRelError],
    config: FGDApproxConfig,
) -> FGDLayerRelError | None:
    if not layer_errors:
        return None
    if config.layer_selection == "min_rel_error":
        return min(layer_errors, key=lambda item: item.relative_error)
    raise ValueError(
        f"Unsupported fgd_approx.layer_selection '{config.layer_selection}'. "
        "Use one of: tiny_best, min_rel_error."
    )


def _output_error_from_layer_error(
    layer_error: FGDLayerRelError,
) -> FGDOutputRelError:
    return FGDOutputRelError(
        relative_error=layer_error.relative_error,
        approximation_norm=layer_error.approximation_norm,
        target_norm=layer_error.target_norm,
        directional_cosine=layer_error.directional_cosine,
    )


def train_one_epoch_gromo_layer_proxy(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    learning_rate: float,
    accuracy_tolerance: float,
    config: FGDApproxConfig,
    classification: bool = False,
) -> FGDApproxEpochResult:
    """Train with GroMo's algebraic per-layer optimal update as the FGD proxy."""
    model.train()
    layer_errors = compute_tiny_layer_relative_errors(
        model=model,
        train_loader=train_loader,
        device=device,
        config=config,
    )
    selected_layer = (
        min(layer_errors, key=lambda item: item.relative_error)
        if layer_errors
        else None
    )

    if selected_layer is not None:
        apply_fgd_parameter_update(
            model=model,
            train_loader=train_loader,
            selected_layer=selected_layer,
            device=device,
            learning_rate=learning_rate,
            config=config,
        )

    train_metrics: RegressionMetrics = evaluate_regression_metrics(
        model,
        train_loader,
        loss_function,
        device=device,
        accuracy_tolerance=accuracy_tolerance,
        classification=classification,
    )
    test_metrics: RegressionMetrics = evaluate_regression_metrics(
        model,
        test_loader,
        loss_function,
        device=device,
        accuracy_tolerance=accuracy_tolerance,
        classification=classification,
    )

    output_error = (
        _output_error_from_layer_error(selected_layer)
        if selected_layer is not None
        else None
    )
    return FGDApproxEpochResult(
        train_loss=train_metrics.loss,
        train_accuracy=train_metrics.accuracy,
        test_loss=test_metrics.loss,
        test_accuracy=test_metrics.accuracy,
        learning_rate=learning_rate,
        next_learning_rate=learning_rate,
        learning_rate_upper_bound=None,
        learning_rate_interval_valid=None,
        learning_rate_clipped_batches=0,
        skipped_batches=0,
        relative_error_condition_valid=None,
        loss_descent_valid=None,
        loss_non_descent_batches=0,
        gradient_sq_norm=None,
        theory_descent_coefficient=None,
        min_positive_learning_rate=learning_rate if learning_rate > 0.0 else None,
        relative_error=selected_layer.relative_error
        if selected_layer is not None
        else None,
        selected_layer_index=selected_layer.layer_index
        if selected_layer is not None
        else None,
        layer_relative_errors=layer_errors,
        output_relative_error=output_error,
        sensor_valid=True,
        sensor_invalid_batches=0,
    )


def select_tiny_growth_layer_index(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: FGDApproxConfig,
) -> int | None:
    """Read-only TINY scoring pass used to choose a growth layer."""
    if not getattr(model, "_growable_layers", None):
        return None

    model.set_growing_layers(scheduling_method="all")
    try:
        compute_statistics(
            model,
            train_loader,
            loss_function=batch_functional_mse_loss,
            device=device,
        )
        model.compute_optimal_updates(
            **tiny_optimal_update_kwargs(
                config,
                compute_delta=config.growth_compute_delta,
            )
        )
        scores = {
            int(index): float(info["update_value"].detach().cpu())
            for index, info in model.update_information().items()
        }
        if not scores:
            return None
        return max(scores, key=scores.get)
    finally:
        _cleanup_tiny_update(model)


def compute_tangent_projection_error(
    model: GrowingMLP,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: FGDApproxConfig,
) -> FGDOutputRelError:
    """Measure the global tangent-projection certificate without updating."""
    certificate = evaluate_fgd_validation_certificate(
        model=model,
        data_loader=data_loader,
        device=device,
        config=config,
        learning_rate=None,
    )
    if certificate.output_relative_error is None:
        raise RuntimeError("Invalid FGD tangent projection sensor measurement.")
    return certificate.output_relative_error


def _count_all_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def select_certifying_growth_layer_index(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: FGDApproxConfig,
    line_search_config: ScalingLineSearchConfig,
    probe: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> int | None:
    """Trial-grow with train data and choose by validation certificate."""
    if not getattr(model, "_growable_layers", None):
        return None

    if probe is None:
        # One fixed probe shared by every layer trial, so their certificates
        # are directly comparable.
        probe = build_projection_probe(validation_loader, config.probe_batches)
    _clear_inaccessible_tensor_caches(model)
    _cleanup_tiny_update(model)
    base_parameter_count = _count_all_parameters(model)
    trials: list[dict[str, float | int | bool]] = []
    optimal_update_kwargs = tiny_optimal_update_kwargs(
        config,
        compute_delta=config.growth_compute_delta,
    )

    for layer_index in range(len(model._growable_layers)):
        trial_model = copy.deepcopy(model)
        growth_result = grow_layer(
            model=trial_model,
            train_loader=train_loader,
            layer_index=layer_index,
            device=device,
            line_search_config=line_search_config,
            optimal_update_kwargs=optimal_update_kwargs,
            progress=None,
        )
        effective_growth = growth_result.best_scaling_factor > max(
            line_search_config.tolerance,
            config.eps,
        )
        certificate = evaluate_fgd_validation_certificate(
            model=trial_model,
            data_loader=validation_loader,
            device=device,
            config=config,
            learning_rate=None,
            probe=probe,
        )
        relative_error = certificate.relative_error
        certified = (
            effective_growth
            and certificate.sensor_valid
            and relative_error is not None
            and relative_error <= config.rel_error_threshold
        )
        trials.append(
            {
                "layer_index": layer_index,
                "relative_error": relative_error
                if relative_error is not None
                else float("inf"),
                "added_parameters": _count_all_parameters(trial_model)
                - base_parameter_count,
                "certified": certified,
                "sensor_valid": certificate.sensor_valid and effective_growth,
            }
        )
        del trial_model

    if not trials:
        return None

    certifying_trials = [trial for trial in trials if bool(trial["certified"])]
    if certifying_trials:
        # Explicit certified policy: fewest added parameters, then lowest
        # relative error, then layer index (deterministic).
        chosen = min(
            certifying_trials,
            key=lambda trial: (
                int(trial["added_parameters"]),
                float(trial["relative_error"]),
                int(trial["layer_index"]),
            ),
        )
    else:
        valid_trials = [trial for trial in trials if bool(trial["sensor_valid"])]
        if not valid_trials:
            return None
        # No candidate certifies: the lowest post-growth relative error
        # wins; parameter count is only a tie-breaker.
        chosen = min(
            valid_trials,
            key=lambda trial: (
                float(trial["relative_error"]),
                int(trial["added_parameters"]),
                int(trial["layer_index"]),
            ),
        )
    return int(chosen["layer_index"])


def should_trigger_fgd_growth(
    relative_error: float,
    epoch: int,
    last_growth_epoch: int | None,
    config: FGDApproxConfig,
) -> bool:
    """Return whether an FGD certificate requests immediate GroMo growth.

    FGD certificate failures are not delayed by epoch or dwell constraints.
    The pipeline-level ``growth_schedule.enabled`` flag remains the global
    switch that can disable architecture changes.
    """
    del epoch, last_growth_epoch
    return relative_error >= config.rel_error_threshold


def certificate_from_projection_stats(
    *,
    stats: _FunctionalStepStats,
    learning_rate: float | None,
    config: FGDApproxConfig,
    projection_sensor: bool = True,
) -> FGDValidationCertificate:
    """Build the FGD certificate from ONE joint probe measurement.

    ``stats`` describes a single approximation g of the functional gradient
    r over the whole probe (one shared direction). ``projection_sensor``
    additionally enforces the exact-projector invariants (<g, r> >= 0 and
    |g| <= |r|), which hold for tangent projections but not for general
    Hilbert secants.
    """
    values = (
        stats.dot_product,
        stats.approximation_sq_norm,
        stats.target_sq_norm,
    )
    finite = all(math.isfinite(value) for value in values)
    sensor_valid = finite and (
        not projection_sensor
        or _projection_sensor_valid(
            dot_product=stats.dot_product,
            approximation_sq_norm=stats.approximation_sq_norm,
            target_sq_norm=stats.target_sq_norm,
            eps=config.eps,
        )
    )
    if not sensor_valid:
        return FGDValidationCertificate(
            learning_rate_upper_bound=None,
            max_valid_learning_rate=None,
            learning_rate_interval_valid=None,
            skipped_batches=0,
            relative_error_condition_valid=None,
            gradient_sq_norm=None,
            theory_descent_coefficient=None,
            relative_error=None,
            output_relative_error=None,
            sensor_valid=False,
            sensor_invalid_batches=1,
        )

    output_error = stats.output_error
    relative_error = output_error.relative_error
    relative_error_condition_valid = relative_error < min(
        config.rel_error_threshold,
        0.5,
    )

    learning_rate_upper_bound: float | None = None
    max_valid_learning_rate: float | None = None
    learning_rate_interval_valid: bool | None = None
    theory_descent_coefficient: float | None = None
    skipped_batches = 0
    if config.learning_rate_policy == "theory_interval":
        learning_rate_upper_bound = theoretical_learning_rate_upper_bound(
            relative_error,
            config,
        )
        if learning_rate_upper_bound is None:
            learning_rate_interval_valid = False
            skipped_batches = 1
        else:
            safe_upper_bound = config.theory_lr_safety * learning_rate_upper_bound
            interval_ok = safe_upper_bound > config.theory_lr_min + config.eps
            if interval_ok:
                max_valid_learning_rate = safe_upper_bound
            if learning_rate is not None and learning_rate > config.eps:
                if config.local_acceptance_conditions:
                    # Strict interval: theory_lr_min < eta < eta_bar, with
                    # eps as the only numerical tolerance.
                    upper_bound_ok = (
                        learning_rate < safe_upper_bound + config.eps
                    )
                else:
                    upper_bound_ok = (
                        learning_rate <= safe_upper_bound + config.eps
                    )
                interval_ok = (
                    interval_ok
                    and learning_rate > config.theory_lr_min
                    and upper_bound_ok
                )
            learning_rate_interval_valid = interval_ok
            if not interval_ok:
                skipped_batches = 1
            elif learning_rate is not None and learning_rate > config.eps:
                theory_descent_coefficient = theoretical_descent_coefficient(
                    relative_error,
                    learning_rate,
                    config,
                )

    return FGDValidationCertificate(
        learning_rate_upper_bound=learning_rate_upper_bound,
        max_valid_learning_rate=max_valid_learning_rate,
        learning_rate_interval_valid=learning_rate_interval_valid,
        skipped_batches=skipped_batches,
        relative_error_condition_valid=relative_error_condition_valid,
        gradient_sq_norm=output_error.target_norm**2,
        theory_descent_coefficient=theory_descent_coefficient,
        relative_error=relative_error,
        output_relative_error=output_error,
        sensor_valid=True,
        sensor_invalid_batches=0,
    )


def measure_direction_projection(
    model: GrowingMLP,
    parameter_updates: tuple[torch.Tensor, ...],
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> _FunctionalStepStats:
    """Measure g = J u against r = grad L on a probe for one SHARED direction.

    This is the certificate measurement for an outer step: ``u`` is the
    direction that will actually move the model, evaluated at the CURRENT
    parameters through a Jacobian-vector product (the Jacobian is never
    materialized).
    """
    named_parameters = _trainable_named_parameters(model)
    if not named_parameters:
        raise RuntimeError(
            "FGD direction measurement requires trainable parameters."
        )
    parameter_names = tuple(named_parameters.keys())
    parameters = tuple(named_parameters.values())
    buffers = OrderedDict(model.named_buffers())

    with torch.no_grad():
        output = model(x)
    target = functional_gradient(output, y, config.functional_loss).reshape(-1)

    def call_with_parameters(
        parameter_values: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        state = OrderedDict(zip(parameter_names, parameter_values))
        state.update(buffers)
        return functional_call(model, state, (x,)).reshape(-1)

    try:
        _, approximation = jvp(
            call_with_parameters,
            (parameters,),
            (tuple(parameter_updates),),
        )
    finally:
        _clear_inaccessible_tensor_caches(model)
    return _output_relative_error_from_tensors(
        approximation=approximation.detach(),
        target=target,
        eps=config.eps,
    )


def evaluate_fgd_validation_certificate(
    model: GrowingMLP,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: FGDApproxConfig,
    learning_rate: float | None,
    probe: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> FGDValidationCertificate:
    """Evaluate FGD certificate conditions on a fixed held-out probe.

    The certificate solves ONE joint tangent projection over the probe (the
    concatenation of ``config.probe_batches`` mini-batches of
    ``data_loader``, unless an explicit ``probe`` is given): a single shared
    parameter direction u* = argmin_u |J_probe u - r_probe|^2, certified
    through g_probe = J_probe u*. Independent per-batch projections combined
    through their aggregated norms are no longer computed anywhere.
    """
    if config.rel_error_mode != "tangent_projection":
        raise ValueError(
            f"Unsupported fgd_approx.rel_error_mode '{config.rel_error_mode}'. "
            "Use tangent_projection."
        )
    if config.projection_solver == "gromo_layer":
        raise ValueError(
            "Validation certificates require projection_solver to be cg, exact, "
            "exact_svd, or exact_kernel_eigh."
        )

    model.eval()
    if probe is None:
        probe = build_projection_probe(data_loader, config.probe_batches)
    x, y = probe
    x = x.to(device)
    y = y.to(device)
    projection_step = _compute_tangent_projection_step(
        model=model,
        x=x,
        y=y,
        config=config,
    )
    stats = _FunctionalStepStats(
        output_error=projection_step.output_error,
        dot_product=projection_step.dot_product,
        approximation_sq_norm=projection_step.approximation_sq_norm,
        target_sq_norm=projection_step.target_sq_norm,
    )
    return certificate_from_projection_stats(
        stats=stats,
        learning_rate=learning_rate,
        config=config,
    )


def evaluate_secant_validation_certificate(
    base_model: GrowingMLP,
    candidate_model: GrowingMLP,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: FGDApproxConfig,
    learning_rate: float,
    probe: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> FGDValidationCertificate:
    """Certify a finite functional secant in the output Hilbert space.

    The realized displacement Delta = F(base) - F(candidate) is one shared
    direction by construction, so the certificate measures it jointly on the
    same fixed probe used by the tangent certificate. Secants are not
    projections, so the exact-projector sensor invariants are not enforced
    (only finiteness).
    """
    if learning_rate <= config.eps:
        raise ValueError("A secant FGD learning rate must be positive.")

    base_model.eval()
    candidate_model.eval()
    if probe is None:
        probe = build_projection_probe(data_loader, config.probe_batches)
    x, y = probe
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        base_output = base_model(x)
        candidate_output = candidate_model(x)
    target = functional_gradient(base_output, y, config.functional_loss)
    approximation = (base_output - candidate_output) / learning_rate
    if not (
        torch.isfinite(target).all() and torch.isfinite(approximation).all()
    ):
        return FGDValidationCertificate(
            learning_rate_upper_bound=None,
            max_valid_learning_rate=None,
            learning_rate_interval_valid=None,
            skipped_batches=0,
            relative_error_condition_valid=None,
            gradient_sq_norm=None,
            theory_descent_coefficient=None,
            relative_error=None,
            output_relative_error=None,
            sensor_valid=False,
            sensor_invalid_batches=1,
        )

    stats = _output_relative_error_from_tensors(
        approximation,
        target,
        config.eps,
    )
    return certificate_from_projection_stats(
        stats=stats,
        learning_rate=learning_rate,
        config=config,
        projection_sensor=False,
    )


def train_one_epoch_fgd_approx(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    learning_rate: float,
    accuracy_tolerance: float,
    config: FGDApproxConfig,
    projection_group_size: int = 1,
    classification: bool = False,
    evaluate_test: bool = True,
) -> FGDApproxEpochResult:
    """Train one FGD epoch with the configured functional-gradient proxy.

    ``exact`` computes the global tangent projection with the full Jacobian.
    ``cg`` and ``gromo_layer`` remain available as cheaper proxy variants, but
    they are not the default training path.
    """
    if config.rel_error_mode != "tangent_projection":
        raise ValueError(
            f"Unsupported fgd_approx.rel_error_mode '{config.rel_error_mode}'. "
            "Use tangent_projection."
        )
    if config.learning_rate_policy not in {"scheduler", "theory_interval"}:
        raise ValueError(
            f"Unsupported fgd_approx.learning_rate_policy "
            f"'{config.learning_rate_policy}'. Use one of: scheduler, "
            "theory_interval."
        )

    if config.projection_solver == "gromo_layer":
        return train_one_epoch_gromo_layer_proxy(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_function=loss_function,
            device=device,
            learning_rate=learning_rate,
            accuracy_tolerance=accuracy_tolerance,
            config=config,
            classification=classification,
        )

    model.train()
    learning_rate_sum = 0.0
    batch_count = 0
    skipped_batches = 0
    min_positive_learning_rate: float | None = None
    sensor_invalid_batches = 0

    if config.learning_rate_policy == "theory_interval":
        if config.theory_lr_safety <= 0.0 or config.theory_lr_safety > 1.0:
            raise ValueError("fgd_approx.theory_lr_safety must be in (0, 1].")
        if config.theory_lr_min < 0.0:
            raise ValueError("fgd_approx.theory_lr_min must be non-negative.")
    if config.theory_mu <= 0.0:
        raise ValueError("fgd_approx.theory_mu must be positive.")

    for x, y in _grouped_batches(train_loader, projection_group_size):
        x = x.to(device)
        y = y.to(device)

        projection_step = _compute_tangent_projection_step(
            model=model,
            x=x,
            y=y,
            config=config,
        )
        if not _projection_step_sensor_valid(projection_step, config):
            sensor_invalid_batches += 1
            skipped_batches += 1
            batch_count += 1
            continue

        applied_step = _apply_tangent_projection_step(
            model=model,
            x=x,
            y=y,
            step=projection_step,
            learning_rate=learning_rate,
            config=config,
        )
        learning_rate_sum += applied_step.learning_rate_used
        batch_count += 1
        if (
            applied_step.learning_rate_used <= config.eps
            and learning_rate > config.eps
        ):
            skipped_batches += 1
        if applied_step.learning_rate_used > config.eps:
            min_positive_learning_rate = (
                applied_step.learning_rate_used
                if min_positive_learning_rate is None
                else min(min_positive_learning_rate, applied_step.learning_rate_used)
            )

    selected_layer_index = select_tiny_growth_layer_index(
        model=model,
        train_loader=train_loader,
        device=device,
        config=config,
    )

    train_metrics: RegressionMetrics = evaluate_regression_metrics(
        model,
        train_loader,
        loss_function,
        device=device,
        accuracy_tolerance=accuracy_tolerance,
        classification=classification,
    )
    test_metrics = (
        evaluate_regression_metrics(
            model,
            test_loader,
            loss_function,
            device=device,
            accuracy_tolerance=accuracy_tolerance,
            classification=classification,
        )
        if evaluate_test
        else RegressionMetrics(loss=float("nan"), accuracy=float("nan"))
    )

    return FGDApproxEpochResult(
        train_loss=train_metrics.loss,
        train_accuracy=train_metrics.accuracy,
        test_loss=test_metrics.loss,
        test_accuracy=test_metrics.accuracy,
        learning_rate=learning_rate_sum / max(1, batch_count),
        next_learning_rate=None,
        learning_rate_upper_bound=None,
        learning_rate_interval_valid=None,
        learning_rate_clipped_batches=0,
        skipped_batches=skipped_batches,
        relative_error_condition_valid=None,
        loss_descent_valid=None,
        loss_non_descent_batches=0,
        gradient_sq_norm=None,
        theory_descent_coefficient=None,
        min_positive_learning_rate=min_positive_learning_rate,
        relative_error=None,
        selected_layer_index=selected_layer_index,
        layer_relative_errors=[],
        output_relative_error=None,
        sensor_valid=sensor_invalid_batches == 0,
        sensor_invalid_batches=sensor_invalid_batches,
    )
