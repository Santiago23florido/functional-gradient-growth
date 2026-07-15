"""Certified approximate functional gradient descent in an RKHS.

Third training method of the repository, independent from both plain SGD
(``training.method: normal``) and the growth-coupled tangent projection
(``training.method: fgd_approx``). It is a faithful implementation of
Algorithm 1 of

    D. Csillag et al., "Functional Gradient Descent with Adaptive
    Representations", arXiv:2606.16926 (2026),

specialized to the empirical regression functional of Section 4.1 so that
*every* assumption of the paper holds with exact, computable constants and
the global-optimality certificate of Theorem 3.10(iii) applies.

Mathematical setting (fixed structure)
--------------------------------------
Fix the training design ``X_1, ..., X_n`` and a kernel ``k``. The *fixed
network structure* is the kernel dictionary

    f(x) = sum_i B_i k(X_i, x),

with trainable linear output coefficients ``B``. Two kernels are supported:

- ``kernel: gaussian`` -- ``k(x, x') = exp(-gamma ||phi(x) - phi(x')||^2)``:
  the structure is a shallow network with ``n`` frozen RBF units.
- ``kernel: linear``  -- ``k(x, x') = phi(x)^T phi(x')`` where ``phi`` is a
  *frozen* fully-connected feature map (``feature_hidden_layers`` hidden
  layers of ``feature_hidden_size`` units). Every dictionary function
  collapses to ``f(x) = W^T phi(x)`` with ``W = Phi_centers^T B``, so the
  trained model is *exactly* a fixed multilayer perceptron -- e.g. three
  hidden layers of 18 neurons -- whose hidden weights are part of the fixed
  structure and whose output layer is trained by certified FGD. Freezing
  the hidden weights is what keeps the loss quadratic over ``H``; training
  them would make the parametrization nonconvex, and no method can certify
  global optimality there (the paper's theory lives in Hilbert space).

The hypothesis space is the (product) RKHS ``H = (H_k)^m`` with
``||f||_H^2 = sum_c ||f_c||^2``, and we take the Banach space ``B = H``
(allowed by the paper; see the remark after Assumptions 3.3-3.4). The loss
is

    L(f) = (1/(2n)) * sum_i ||f(X_i) - Y_i||^2.

The empirical loss only depends on ``f`` through its values at the ``X_i``,
so the infimum of ``L`` over the *whole* space ``H`` equals the infimum over
the fixed dictionary span: the certified optimum is the global optimum of
the function space and of the fixed structure simultaneously. For the
strictly positive definite Gaussian kernel that infimum is ``L* = 0``; for
the linear kernel it is the exact least-squares optimum of the fixed
structure, computed in closed form at setup.

Verified assumptions and constants (all computed, none assumed)
---------------------------------------------------------------
- Assumption 3.1 (extension):     trivial, since B = H.
- Assumption 3.2 (K-smoothness):  the empirical loss is an exact quadratic
  on H with constant Hessian ``(1/n) sum_i k(X_i,.) (x) k(X_i,.)``, so
      gaussian: ``K_s = kappa = sup_x k(x, x) = 1``;
      linear:   ``K_s = lambda_max(Phi^T Phi) / n`` (exact operator norm).
- Assumption 3.3 (H descends):    alpha = 1   (B = H).
- Assumption 3.4 (compatibility): beta = 1    (B = H).
- Assumption 3.7 (Polyak-Lojasiewicz): with residual matrix R,
      L - L* <= n ||grad L||_H^2 / (2 lambda_min+)  where lambda_min+ is
  the smallest (positive) eigenvalue of the Gram operator, so
      gaussian: ``mu = lambda_min(K) / n`` (n x n eigendecomposition),
      linear:   ``mu = lambda_min(Phi^T Phi) / n`` (p x p, numerically
                clean because p is the feature dimension, e.g. 18).
  In the linear case PL holds relative to the exact ``L* > 0`` because the
  predictions always remain in the column space of ``Phi``.
- Functional gradient (Prop. 4.1): grad L(f) = sum_i C_i k(X_i, .) with
  coefficients ``C = R / n`` -- exactly representable in the dictionary,
  for either kernel.

Approximation family (Algorithm 1)
----------------------------------
The gradient representations are nested spans of a *fixed* dictionary: a
permutation of the centers is drawn once, and level ``q`` exposes the first
``q`` centers. The level-q approximation is the H-orthogonal projection of
``grad L`` onto that span. The paper's upper bound ``U`` (Equation 5) is
computed *exactly* here:

    U^2 = ||g - grad L||_H^2
        = tr(A^T K_qq A) - 2 tr(A^T (K C)_q) + tr(C^T K C),

which is a valid upper bound for *any* coefficients ``A`` (in particular it
stays valid when the linear solve is damped or inaccurate). At the top level
the dictionary contains all centers, ``g = grad L`` identically and
``U = 0``, so the refinement loop of Algorithm 1 terminates unconditionally
(Lemma 3.9 holds by construction).

Certificates (Theorem 3.10)
---------------------------
The step is accepted only when ``(1 + eps) U < eps ||g||_H``, which enforces
``RelErr <= eps/(1+eps) =: eps_bar < 1/2`` (Theorem 3.10(i)). The learning
rate is a strict fraction of ``2 (alpha - (alpha+beta) eps_bar) /
(K_s (2 eps_bar + 1))`` as required by Proposition 3.8, whose linear rate

    L(f_T) - L* <= (1 - 2 eta beta^-2 mu r)^T (L(f_0) - L*)

is tracked and checked every epoch, together with the per-step sufficient
descent inequality of Lemma 3.5.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class FGDRKHSConfig:
    """Hyperparameters of the certified RKHS FGD method.

    ``epsilon`` is the tolerance of Algorithm 1 and must lie in (0, 1);
    the certified relative error is then ``epsilon / (1 + epsilon) < 1/2``.
    ``lr_safety`` in (0, 1) keeps the learning rate strictly inside the
    admissible interval of Proposition 3.8. ``kernel_gamma = None`` selects
    the median heuristic. ``levels = None`` builds a geometric ladder ending
    at the full dictionary (the last level must always be the full
    dictionary; that is what guarantees termination of the refinement loop).
    ``solver_jitter`` only damps the projection solve; the error certificate
    ``U`` is computed from the realized coefficients, so damping can never
    invalidate the certificate, only trigger extra refinement.

    ``kernel`` selects the fixed structure: ``"gaussian"`` is the RBF
    dictionary; ``"linear"`` composes a *frozen* MLP feature map (built from
    ``feature_hidden_layers`` / ``feature_hidden_size`` / ``feature_activation``
    / ``feature_seed``) with a trainable linear output layer, so the trained
    model is exactly that fixed MLP. ``feature_hidden_layers = 0`` applies
    the kernel directly to the raw inputs. ``kernel_gamma`` is only
    meaningful for the Gaussian kernel (``None`` selects the median
    heuristic, computed on the features when a feature map is present).

    ``feature_whitening`` (linear kernel only) composes the frozen features
    with the fixed linear map ``T = V diag(lambda)^{-1/2}`` computed once
    from the training design. This is a linear reparametrization of the
    output layer: it changes neither the function class of the fixed
    structure nor its optimum ``L*``, only the inner product of ``H``
    (which the paper leaves free to choose). It makes the Gram spectrum
    near-isotropic, so the certified constants ``K_s`` and ``mu`` -- still
    *measured* on the realized features, never assumed -- become sharp and
    the Prop. 3.8 envelope contracts fast instead of being near-vacuous.
    """

    kernel_gamma: float | None = None
    epsilon: float = 0.25
    lr_safety: float = 0.9
    steps_per_epoch: int = 1
    levels: tuple[int, ...] | None = None
    max_train_points: int = 4096
    centers_seed: int = 0
    solver_jitter: float = 1e-10
    min_kernel_eigenvalue: float = 1e-12
    gradient_tolerance: float = 1e-14
    eps: float = 1e-12
    kernel: str = "gaussian"
    feature_hidden_layers: int = 0
    feature_hidden_size: int = 0
    feature_activation: str = "tanh"
    feature_seed: int = 0
    feature_whitening: bool = True
    # Train-and-grow cycle (training.method: fgd_rkhs_grow): after each
    # growth step the grown hidden layers are frozen as the feature map and
    # the head is trained to the certified optimum of that structure. The
    # cycle stops when the closed-form ceiling L* of the newly grown
    # structure improves by less than growth_min_ceiling_improvement
    # (relative), when every hidden layer reached growth_max_hidden_size,
    # or after growth_max_cycles growth events.
    growth_max_cycles: int = 4
    growth_epochs_per_cycle: int = 40
    growth_min_ceiling_improvement: float = 0.01
    growth_max_hidden_size: int | None = None


@dataclass(frozen=True)
class FGDRKHSTheory:
    """Exact constants of the paper's assumptions for this problem."""

    train_points: int
    output_features: int
    kernel_kind: str
    feature_dimension: int
    kernel_gamma: float
    kappa: float
    smoothness: float
    alpha: float
    beta: float
    kernel_lambda_min: float
    pl_mu: float
    pl_certificate_valid: bool
    loss_star: float
    epsilon: float
    epsilon_bar: float
    learning_rate_upper_bound: float
    learning_rate: float
    descent_coefficient: float
    contraction: float
    initial_loss: float


@dataclass(frozen=True)
class FGDRKHSStepRecord:
    """Certificates measured for one accepted FGD step (Algorithm 1)."""

    step: int
    level: int
    dictionary_size: int
    refinements: int
    gradient_sq_norm: float
    approximation_sq_norm: float
    error_upper_bound: float
    relative_error: float
    relative_error_condition_valid: bool
    loss_before: float
    loss_after: float
    descent_bound: float
    descent_valid: bool
    converged: bool


@dataclass(frozen=True)
class FGDRKHSEpochResult:
    step_records: list[FGDRKHSStepRecord] = field(default_factory=list)
    train_functional_loss: float = 0.0
    global_bound: float | None = None
    global_bound_valid: bool | None = None
    converged: bool = False


_FEATURE_ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
}


class FrozenMLPFeatureMap(nn.Module):
    """Frozen fully-connected feature map ``phi`` (part of the fixed structure).

    The hidden weights are drawn once from a seeded CPU generator and stored
    as *buffers*, never as parameters: the network structure -- including
    these weights -- is fixed, and only the linear output layer on top of
    the last hidden layer is trained (by certified functional gradient
    steps). Freezing keeps the training functional an exact quadratic over
    the induced RKHS; training the hidden weights would make the
    parametrization nonconvex and void every certificate of the paper.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        hidden_layers: int,
        activation: str = "tanh",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("feature_hidden_layers must be >= 1.")
        if hidden_size < 1:
            raise ValueError("feature_hidden_size must be >= 1.")
        if activation not in _FEATURE_ACTIVATIONS:
            raise ValueError(
                "feature_activation must be one of "
                f"{sorted(_FEATURE_ACTIVATIONS)}, got {activation!r}."
            )
        self.in_features = int(in_features)
        self.hidden_size = int(hidden_size)
        self.hidden_layers = int(hidden_layers)
        self.out_features = int(hidden_size)
        self.activation_name = activation
        generator = torch.Generator().manual_seed(seed)
        fan_in = in_features
        for layer in range(hidden_layers):
            bound = 1.0 / math.sqrt(fan_in)
            weight = bound * (
                2.0
                * torch.rand(
                    hidden_size,
                    fan_in,
                    generator=generator,
                    dtype=torch.float64,
                )
                - 1.0
            )
            bias = bound * (
                2.0
                * torch.rand(hidden_size, generator=generator, dtype=torch.float64)
                - 1.0
            )
            self.register_buffer(f"weight_{layer}", weight)
            self.register_buffer(f"bias_{layer}", bias)
            fan_in = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_weight = self.get_buffer("weight_0")
        z = x.reshape(x.shape[0], -1).to(
            device=first_weight.device,
            dtype=torch.float64,
        )
        activation = _FEATURE_ACTIVATIONS[self.activation_name]
        for layer in range(self.hidden_layers):
            weight = self.get_buffer(f"weight_{layer}")
            bias = self.get_buffer(f"bias_{layer}")
            z = activation(z @ weight.T + bias)
        return z

    def __str__(self) -> str:
        return (
            "FrozenMLPFeatureMap("
            f"in={self.in_features}, "
            f"hidden_layers={self.hidden_layers}, "
            f"hidden_size={self.hidden_size}, "
            f"activation={self.activation_name})"
        )


class FrozenAffineFeatureMap(nn.Module):
    """Frozen feature map snapshotted from existing affine layers.

    Freezes the hidden layers of an externally trained network (e.g. a
    GroMo-grown MLP) as the fixed structure ``phi``: weights and biases are
    copied into float64 buffers, so they can never be trained. With
    ``append_one=True`` a constant 1 feature is appended, which makes a
    linear head on ``phi`` exactly an affine output layer (weight + bias):
    the certified optimum over H is then the true global optimum of the
    donor network's output layer for its frozen hidden weights.
    """

    def __init__(
        self,
        weights: Sequence[torch.Tensor],
        biases: Sequence[torch.Tensor | None],
        activations: Sequence[nn.Module],
        append_one: bool = True,
    ) -> None:
        super().__init__()
        if not weights:
            raise ValueError("FrozenAffineFeatureMap needs at least one layer.")
        if len(weights) != len(biases) or len(weights) != len(activations):
            raise ValueError(
                "weights, biases and activations must have equal lengths."
            )
        self.depth = len(weights)
        self.append_one = bool(append_one)
        self.activations = nn.ModuleList(
            copy.deepcopy(module) for module in activations
        )
        for parameter in self.activations.parameters():
            parameter.requires_grad_(False)
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            if weight.ndim != 2:
                raise ValueError("Each layer weight must be 2D (out, in).")
            weight64 = weight.detach().to(dtype=torch.float64)
            if bias is None:
                bias64 = torch.zeros(
                    weight.shape[0],
                    dtype=torch.float64,
                    device=weight64.device,
                )
            else:
                bias64 = bias.detach().to(dtype=torch.float64)
            self.register_buffer(f"weight_{index}", weight64)
            self.register_buffer(f"bias_{index}", bias64)
        self.in_features = int(weights[0].shape[1])
        self.out_features = int(weights[-1].shape[0]) + int(self.append_one)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_weight = self.get_buffer("weight_0")
        z = x.reshape(x.shape[0], -1).to(
            device=first_weight.device,
            dtype=torch.float64,
        )
        for index in range(self.depth):
            weight = self.get_buffer(f"weight_{index}")
            bias = self.get_buffer(f"bias_{index}")
            z = self.activations[index](z @ weight.T + bias)
        if self.append_one:
            ones = torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
            z = torch.cat([z, ones], dim=1)
        return z

    def __str__(self) -> str:
        widths = "->".join(
            str(self.get_buffer(f"weight_{index}").shape[0])
            for index in range(self.depth)
        )
        return (
            "FrozenAffineFeatureMap("
            f"in={self.in_features}, widths={widths}, "
            f"append_one={self.append_one})"
        )


class KernelDictionaryModel(nn.Module):
    """Fixed-structure dictionary network ``x -> K(x, centers) @ B``.

    The (feature-space) centers and the optional frozen feature map are the
    fixed structure; the output coefficients are the only parameters and are
    updated by the trainer through certified functional gradient steps.

    With ``kernel="linear"`` and a deep feature map ``phi``, the forward pass
    ``(phi(x) @ Phi_c^T) @ B`` collapses to ``phi(x) @ W`` with
    ``W = Phi_c^T @ B`` (see :meth:`linear_head_weight`): the model *is* the
    fixed MLP ``phi`` followed by a trained linear output layer.
    """

    def __init__(
        self,
        centers: torch.Tensor,
        kernel_gamma: float | None,
        out_features: int,
        *,
        kernel: str = "gaussian",
        feature_map: nn.Module | None = None,
        feature_transform: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if centers.ndim != 2:
            raise ValueError("KernelDictionaryModel centers must be 2D (n, d).")
        if kernel not in ("gaussian", "linear"):
            raise ValueError("kernel must be 'gaussian' or 'linear'.")
        if kernel == "gaussian":
            if kernel_gamma is None or kernel_gamma <= 0.0:
                raise ValueError("kernel_gamma must be positive.")
            self.kernel_gamma = float(kernel_gamma)
        else:
            self.kernel_gamma = None
        self.kernel_kind = kernel
        self.feature_map = feature_map
        if feature_transform is not None:
            # Fixed linear reparametrization of the feature space (e.g.
            # whitening); composes with the output layer, so the deployed
            # model is unchanged as a network architecture.
            self.register_buffer(
                "feature_transform",
                feature_transform.detach().to(dtype=torch.float64),
            )
        else:
            self.feature_transform = None
        with torch.no_grad():
            center_features = self.features(centers.detach())
        self.register_buffer("centers", center_features)
        self.coefficients = nn.Parameter(
            torch.zeros(
                centers.shape[0],
                out_features,
                dtype=torch.float64,
                device=center_features.device,
            )
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_map is not None:
            feats = self.feature_map(x)
        else:
            feats = x.reshape(x.shape[0], -1).to(dtype=torch.float64)
        if self.feature_transform is not None:
            feats = feats.to(device=self.feature_transform.device)
            feats = feats @ self.feature_transform
        return feats

    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x).to(device=self.centers.device)
        if self.kernel_kind == "gaussian":
            distances_sq = torch.cdist(feats, self.centers).square()
            return torch.exp(-self.kernel_gamma * distances_sq)
        return feats @ self.centers.T

    def linear_head_weight(self) -> torch.Tensor:
        """Collapsed output-layer weight ``W`` with ``model(x) = phi(x) @ W``,
        where ``phi`` is the raw frozen feature map (the MLP's last hidden
        layer). Any fixed feature transform is folded into ``W``."""
        if self.kernel_kind != "linear":
            raise RuntimeError(
                "linear_head_weight is only defined for kernel='linear'."
            )
        head = self.centers.T @ self.coefficients
        if self.feature_transform is not None:
            head = self.feature_transform @ head
        return head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.kernel(x) @ self.coefficients
        return output.to(dtype=x.dtype if x.is_floating_point() else torch.float32)

    def __str__(self) -> str:
        feature_note = (
            f", feature_map={self.feature_map}" if self.feature_map is not None else ""
        )
        gamma_note = (
            f", gamma={self.kernel_gamma:.6g}" if self.kernel_gamma is not None else ""
        )
        return (
            "KernelDictionaryModel("
            f"kernel={self.kernel_kind}, "
            f"centers={self.centers.shape[0]}, "
            f"center_dim={self.centers.shape[1]}, "
            f"out_features={self.coefficients.shape[1]}"
            f"{gamma_note}{feature_note})"
        )


def median_heuristic_gamma(x: torch.Tensor, max_points: int = 2048) -> float:
    """Return ``1 / median(||x_i - x_j||^2)`` over a deterministic subsample."""
    flat = x.reshape(x.shape[0], -1).to(dtype=torch.float64)
    if flat.shape[0] > max_points:
        flat = flat[:max_points]
    distances_sq = torch.cdist(flat, flat).square()
    off_diagonal = distances_sq[
        ~torch.eye(flat.shape[0], dtype=torch.bool, device=flat.device)
    ]
    median = float(off_diagonal.median().item())
    if median <= 0.0:
        raise ValueError(
            "Median pairwise distance is zero; the training inputs contain "
            "duplicates, so the kernel matrix cannot be strictly positive "
            "definite. Deduplicate the data or set kernel_gamma explicitly."
        )
    return 1.0 / median


def default_level_ladder(n: int) -> tuple[int, ...]:
    """Geometric ladder of nested dictionary sizes ending at ``n``."""
    levels: list[int] = []
    size = max(1, n // 16)
    while size < n:
        levels.append(size)
        size = min(n, size * 2)
    levels.append(n)
    return tuple(levels)


def theory_learning_rate_upper_bound(
    epsilon_bar: float,
    smoothness: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Admissible learning-rate bound of Proposition 3.8 (strict)."""
    if not 0.0 <= epsilon_bar < min(0.5, alpha / (alpha + beta)):
        raise ValueError(
            "epsilon_bar must satisfy the Prop. 3.6 condition "
            "eps < min{1/2, alpha/(alpha+beta)}."
        )
    return 2.0 * (alpha - (alpha + beta) * epsilon_bar) / (
        smoothness * (2.0 * epsilon_bar + 1.0)
    )


def theory_descent_coefficient(
    relative_error: float,
    learning_rate: float,
    smoothness: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Coefficient ``r`` of Lemma 3.5 / Prop. 3.6 at a given relative error."""
    if relative_error < 0.0 or relative_error >= 1.0:
        raise ValueError("relative_error must lie in [0, 1).")
    error_ratio = relative_error / (1.0 - relative_error)
    return (
        alpha
        - 0.5 * smoothness * learning_rate
        - (beta + 1.5 * smoothness * learning_rate) * error_ratio
    )


class FGDRKHSTrainer:
    """Runs Algorithm 1 on the empirical RKHS regression functional."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        config: FGDRKHSConfig,
        device: torch.device | None = None,
        feature_map: nn.Module | None = None,
    ) -> None:
        if not 0.0 < config.epsilon < 1.0:
            raise ValueError("fgd_rkhs.epsilon must lie in (0, 1).")
        if not 0.0 < config.lr_safety < 1.0:
            raise ValueError("fgd_rkhs.lr_safety must lie in (0, 1).")
        if config.steps_per_epoch < 1:
            raise ValueError("fgd_rkhs.steps_per_epoch must be >= 1.")
        if config.kernel not in ("gaussian", "linear"):
            raise ValueError("fgd_rkhs.kernel must be 'gaussian' or 'linear'.")
        if config.feature_hidden_layers < 0:
            raise ValueError("fgd_rkhs.feature_hidden_layers must be >= 0.")
        if config.feature_hidden_layers > 0 and config.feature_hidden_size < 1:
            raise ValueError(
                "fgd_rkhs.feature_hidden_size must be >= 1 when "
                "feature_hidden_layers > 0."
            )

        self.config = config
        device = device or train_x.device
        x = train_x.reshape(train_x.shape[0], -1).to(
            device=device,
            dtype=torch.float64,
        )
        y = train_y.reshape(train_y.shape[0], -1).to(
            device=device,
            dtype=torch.float64,
        )
        if x.shape[0] != y.shape[0]:
            raise ValueError("train_x and train_y must have matching lengths.")
        if x.shape[0] > config.max_train_points:
            keep = torch.randperm(
                x.shape[0],
                generator=torch.Generator().manual_seed(config.centers_seed),
            )[: config.max_train_points].to(device)
            x = x[keep]
            y = y[keep]
        self.train_x = x
        self.train_y = y
        n = x.shape[0]

        if feature_map is not None:
            # Externally provided frozen feature map (e.g. the snapshotted
            # hidden layers of a grown network): the fixed structure is the
            # donor architecture, not a random seeded MLP.
            feature_map = feature_map.to(device)
        elif config.feature_hidden_layers > 0:
            feature_map = FrozenMLPFeatureMap(
                in_features=x.shape[1],
                hidden_size=config.feature_hidden_size,
                hidden_layers=config.feature_hidden_layers,
                activation=config.feature_activation,
                seed=config.feature_seed,
            ).to(device)

        feature_transform: torch.Tensor | None = None
        whitening_condition = 1.0
        if config.kernel == "gaussian":
            with torch.no_grad():
                heuristic_input = feature_map(x) if feature_map is not None else x
            gamma = (
                config.kernel_gamma
                if config.kernel_gamma is not None
                else median_heuristic_gamma(heuristic_input)
            )
        else:
            gamma = None
            if config.feature_whitening:
                # Fixed whitening map T = V diag(lambda)^{-1/2} of the raw
                # training features. A linear reparametrization of the
                # output layer: same function class, same L*, but the Gram
                # spectrum becomes near-isotropic, so the certified
                # constants (measured below on the realized features) are
                # sharp. Eigenvalues are floored to keep T finite; floored
                # directions simply stay poorly scaled and are reported as
                # such by the measured spectrum.
                with torch.no_grad():
                    raw = feature_map(x) if feature_map is not None else x.to(
                        dtype=torch.float64
                    )
                    covariance = raw.T @ raw
                    eigvals, eigvecs = torch.linalg.eigh(covariance)
                    floor = max(
                        config.min_kernel_eigenvalue,
                        float(eigvals.max().item()) * 1e-12,
                    )
                    feature_transform = eigvecs * eigvals.clamp_min(
                        floor
                    ).rsqrt().unsqueeze(0)
                    whitening_condition = float(eigvals.max().item()) / floor
                    if float(eigvals.min().item()) > floor:
                        whitening_condition = float(eigvals.max().item()) / float(
                            eigvals.min().item()
                        )
        self.model = KernelDictionaryModel(
            centers=x,
            kernel_gamma=gamma,
            out_features=y.shape[1],
            kernel=config.kernel,
            feature_map=feature_map,
            feature_transform=feature_transform,
        ).to(device)
        self.kernel_matrix = self.model.kernel(x)

        # Numerical slack for the certificate checks. With the whitened
        # linear kernel the Hessian spectrum is near-isotropic, so Lemma 3.5
        # holds with near-EQUALITY and float64 roundoff (matmul
        # accumulation, eigendecomposition resolution) decides the sign; a
        # relative floor of 1e-9 absorbs that wobble while remaining many
        # orders of magnitude below any genuine violation (a wrong constant
        # or learning rate shows up at >= 1e-3 relative). The slack also
        # grows with the conditioning of the raw feature covariance and is
        # capped so a catastrophically conditioned structure fails loudly
        # instead of hiding behind an absurd tolerance.
        float64_eps = float(torch.finfo(torch.float64).eps)
        tightness_floor = (
            1e-9 if (config.kernel == "linear" and config.feature_whitening)
            else config.eps
        )
        self.certificate_tolerance = min(
            max(config.eps, tightness_floor, 16.0 * whitening_condition * float64_eps),
            1e-6,
        )

        # Fixed nested representation family: one permutation drawn once.
        permutation = torch.randperm(
            n,
            generator=torch.Generator().manual_seed(config.centers_seed + 1),
        ).to(device)
        self.permutation = permutation
        levels = (
            tuple(int(level) for level in config.levels)
            if config.levels is not None
            else default_level_ladder(n)
        )
        levels = tuple(min(level, n) for level in levels)
        if list(levels) != sorted(set(levels)):
            raise ValueError("fgd_rkhs.levels must be strictly increasing.")
        if any(level <= 0 for level in levels):
            raise ValueError("fgd_rkhs.levels must be positive.")
        if levels[-1] != n:
            # The full dictionary level realizes U = 0 exactly and is what
            # guarantees termination of the refinement loop (Lemma 3.9).
            levels = levels + (n,)
        self.levels = levels
        self.level_index = 0
        self._cholesky_cache: dict[int, torch.Tensor | None] = {}
        self._cached_predictions: torch.Tensor | None = None

        if config.kernel == "gaussian":
            eigenvalues = torch.linalg.eigvalsh(self.kernel_matrix)
            lambda_min = max(float(eigenvalues.min().item()), 0.0)
            kappa = 1.0  # Gaussian kernel: sup_x k(x, x) = 1.
            smoothness = kappa
            # Strictly positive definite kernel: inf over H is exact
            # interpolation, L* = 0.
            loss_star = 0.0
            feature_dimension = int(self.model.centers.shape[1])
        else:
            # Linear kernel over the (feature-mapped) design: the Gram
            # operator on H is (1/n) Phi^T Phi, a p x p matrix whose exact
            # spectrum gives the smoothness and PL constants, and L* is the
            # exact least-squares optimum of the fixed structure (computed
            # by spectral pseudo-inverse; directions below the eigenvalue
            # floor are dropped, which can only *increase* the reported L*,
            # keeping the envelope a valid upper bound).
            features = self.model.centers
            feature_dimension = int(features.shape[1])
            gram = features.T @ features
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            lambda_min = max(float(eigenvalues.min().item()), 0.0)
            lambda_max = float(eigenvalues.max().item())
            kappa = float(self.kernel_matrix.diagonal().max().item())
            smoothness = lambda_max / n
            inverse = torch.where(
                eigenvalues > config.min_kernel_eigenvalue,
                1.0 / eigenvalues,
                torch.zeros_like(eigenvalues),
            )
            head_star = eigenvectors @ (
                inverse.unsqueeze(1) * (eigenvectors.T @ (features.T @ y))
            )
            loss_star = self._functional_loss(features @ head_star)
        pl_valid = lambda_min > config.min_kernel_eigenvalue
        alpha = 1.0
        beta = 1.0
        epsilon_bar = config.epsilon / (1.0 + config.epsilon)
        lr_bound = theory_learning_rate_upper_bound(
            epsilon_bar,
            smoothness,
            alpha=alpha,
            beta=beta,
        )
        learning_rate = config.lr_safety * lr_bound
        descent = theory_descent_coefficient(
            epsilon_bar,
            learning_rate,
            smoothness,
            alpha=alpha,
            beta=beta,
        )
        if descent <= 0.0:
            raise RuntimeError(
                "Non-positive descent coefficient r; this cannot happen for "
                "lr_safety < 1 and epsilon < 1."
            )
        mu = lambda_min / n
        contraction = min(max(1.0 - 2.0 * learning_rate * mu * descent, 0.0), 1.0)

        initial_loss = self._functional_loss(self._predictions())
        self.theory = FGDRKHSTheory(
            train_points=n,
            output_features=y.shape[1],
            kernel_kind=config.kernel,
            feature_dimension=feature_dimension,
            kernel_gamma=gamma if gamma is not None else 0.0,
            kappa=kappa,
            smoothness=smoothness,
            alpha=alpha,
            beta=beta,
            kernel_lambda_min=lambda_min,
            pl_mu=mu,
            pl_certificate_valid=pl_valid,
            loss_star=loss_star,
            epsilon=config.epsilon,
            epsilon_bar=epsilon_bar,
            learning_rate_upper_bound=lr_bound,
            learning_rate=learning_rate,
            descent_coefficient=descent,
            contraction=contraction,
            initial_loss=initial_loss,
        )
        self.total_steps = 0
        self.converged = False

    # ------------------------------------------------------------------
    # Loss / gradient primitives (all in float64, all exact quadratics).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _predictions(self) -> torch.Tensor:
        if self._cached_predictions is not None:
            return self._cached_predictions
        return self.kernel_matrix @ self.model.coefficients

    def _functional_loss(self, predictions: torch.Tensor) -> float:
        residual = predictions - self.train_y
        return float(residual.square().sum().item()) / (2.0 * residual.shape[0])

    def _level_cholesky(self, level: int) -> torch.Tensor | None:
        if level in self._cholesky_cache:
            return self._cholesky_cache[level]
        indices = self.permutation[:level]
        block = self.kernel_matrix[indices][:, indices]
        if self.config.solver_jitter > 0.0:
            jitter = self.config.solver_jitter * float(
                block.diagonal().mean().item()
            )
            block = block + jitter * torch.eye(
                level,
                dtype=block.dtype,
                device=block.device,
            )
        try:
            factor = torch.linalg.cholesky(block)
        except RuntimeError:
            # Ill-conditioned block: skip the level. The certificate cannot
            # be affected -- a missing level only forces refinement, and the
            # full-dictionary level never needs a factorization.
            factor = None
        self._cholesky_cache[level] = factor
        return factor

    @torch.no_grad()
    def gradient_approximation(
        self,
        level: int,
        coefficients: torch.Tensor,
        kernel_times_coefficients: torch.Tensor,
        gradient_sq_norm: float,
    ) -> tuple[torch.Tensor, float, float] | None:
        """Level-``level`` representation of the functional gradient.

        Returns ``(update, ||g||_H^2, ||g - grad L||_H^2)`` where ``update``
        holds the dictionary coefficients of ``g`` on the first ``level``
        centers of the fixed permutation. Both norms are exact RKHS
        quadratic forms, so the second value is the paper's upper bound
        ``U^2`` (Equation 5) with equality -- valid for any coefficients the
        solver returns. Returns ``None`` when the level's Cholesky
        factorization fails, which simply forces a refinement.
        """
        n = self.kernel_matrix.shape[0]
        indices = self.permutation[:level]
        if level >= n:
            # Full dictionary: g = grad L identically (same coefficients up
            # to permutation), hence U = 0 exactly.
            return coefficients[indices], gradient_sq_norm, 0.0

        factor = self._level_cholesky(level)
        if factor is None:
            return None
        rhs = kernel_times_coefficients[indices]
        update = torch.cholesky_solve(rhs, factor)
        block = self.kernel_matrix[indices][:, indices]
        approximation_sq_norm = float((update * (block @ update)).sum().item())
        cross_term = float((update * rhs).sum().item())
        error_sq = max(
            approximation_sq_norm - 2.0 * cross_term + gradient_sq_norm,
            0.0,
        )
        return update, approximation_sq_norm, error_sq

    @torch.no_grad()
    def step(self) -> FGDRKHSStepRecord:
        """One certified FGD step: refine until Algorithm 1 accepts, then move."""
        config = self.config
        n = self.theory.train_points
        predictions = self._predictions()
        loss_before = self._functional_loss(predictions)
        residual = predictions - self.train_y

        # grad L(f) = sum_i C_i k(X_i, .) with C = R / n (Prop. 4.1).
        coefficients = residual / n
        kernel_times_coefficients = self.kernel_matrix @ coefficients
        gradient_sq_norm = float(
            (coefficients * kernel_times_coefficients).sum().item()
        )
        if gradient_sq_norm <= config.gradient_tolerance:
            # PL (Assumption 3.7): L - L* <= ||grad L||^2 / (2 mu) ~ 0.
            self.converged = True
            self._cached_predictions = predictions
            return FGDRKHSStepRecord(
                step=self.total_steps,
                level=self.level_index,
                dictionary_size=self.levels[self.level_index],
                refinements=0,
                gradient_sq_norm=gradient_sq_norm,
                approximation_sq_norm=0.0,
                error_upper_bound=0.0,
                relative_error=0.0,
                relative_error_condition_valid=True,
                loss_before=loss_before,
                loss_after=loss_before,
                descent_bound=loss_before,
                descent_valid=True,
                converged=True,
            )

        refinements = 0
        while True:
            level = self.levels[self.level_index]
            indices = self.permutation[:level]
            approximation = self.gradient_approximation(
                level,
                coefficients,
                kernel_times_coefficients,
                gradient_sq_norm,
            )
            if approximation is None:
                self.level_index += 1
                refinements += 1
                continue
            update, approximation_sq_norm, error_sq = approximation

            approximation_norm = math.sqrt(max(approximation_sq_norm, 0.0))
            error_upper_bound = math.sqrt(error_sq)
            accepted = (
                (1.0 + config.epsilon) * error_upper_bound
                < config.epsilon * approximation_norm
            )
            if accepted:
                break
            if self.level_index + 1 >= len(self.levels):
                raise RuntimeError(
                    "Algorithm 1 failed to certify at the full dictionary "
                    "level; this is impossible because U = 0 there. "
                    "Check the data for non-finite values."
                )
            self.level_index += 1
            refinements += 1

        relative_error = (
            error_upper_bound / approximation_norm
            if approximation_norm > 0.0
            else 0.0
        )
        relative_error_valid = relative_error <= self.theory.epsilon_bar + config.eps

        learning_rate = self.theory.learning_rate
        self.model.coefficients.data[indices] -= learning_rate * update
        self._cached_predictions = (
            self.kernel_matrix @ self.model.coefficients
        )
        loss_after = self._functional_loss(self._cached_predictions)

        # Sufficient descent (Lemma 3.5) at the measured relative error.
        measured_descent = theory_descent_coefficient(
            min(relative_error, self.theory.epsilon_bar),
            learning_rate,
            self.theory.smoothness,
            alpha=self.theory.alpha,
            beta=self.theory.beta,
        )
        descent_bound = loss_before - learning_rate * measured_descent * (
            gradient_sq_norm
        )
        descent_tolerance = self.certificate_tolerance * (
            1.0 + abs(loss_before) + learning_rate * gradient_sq_norm
        )
        descent_valid = loss_after <= descent_bound + descent_tolerance

        self.total_steps += 1
        return FGDRKHSStepRecord(
            step=self.total_steps,
            level=self.level_index,
            dictionary_size=self.levels[self.level_index],
            refinements=refinements,
            gradient_sq_norm=gradient_sq_norm,
            approximation_sq_norm=approximation_sq_norm,
            error_upper_bound=error_upper_bound,
            relative_error=relative_error,
            relative_error_condition_valid=relative_error_valid,
            loss_before=loss_before,
            loss_after=loss_after,
            descent_bound=descent_bound,
            descent_valid=descent_valid,
            converged=False,
        )

    def global_bound(self) -> float:
        """Prop. 3.8 envelope after ``total_steps`` certified steps."""
        theory = self.theory
        gap = max(theory.initial_loss - theory.loss_star, 0.0)
        return theory.loss_star + gap * theory.contraction**self.total_steps

    @torch.no_grad()
    def run_epoch(self) -> FGDRKHSEpochResult:
        records: list[FGDRKHSStepRecord] = []
        for _ in range(self.config.steps_per_epoch):
            record = self.step()
            records.append(record)
            if record.converged:
                break
        train_loss = records[-1].loss_after if records else self._functional_loss(
            self._predictions()
        )
        bound = self.global_bound()
        bound_tolerance = self.certificate_tolerance * (
            1.0 + self.theory.initial_loss
        )
        bound_valid = (
            train_loss <= bound + bound_tolerance
            if self.theory.pl_certificate_valid
            else None
        )
        return FGDRKHSEpochResult(
            step_records=records,
            train_functional_loss=train_loss,
            global_bound=bound if self.theory.pl_certificate_valid else None,
            global_bound_valid=bound_valid,
            converged=self.converged,
        )
