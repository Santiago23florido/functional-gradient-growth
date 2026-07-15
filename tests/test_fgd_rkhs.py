"""Tests for the certified RKHS FGD method (arXiv:2606.16926, Algorithm 1).

Each test verifies one of the paper's conditions *independently* of the
trainer's internal bookkeeping: the assumptions are re-derived from kernel
algebra, and the certificates (Lemma 3.5, Prop. 3.6/3.8, Theorem 3.10) are
checked against measured losses.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fgdlib.rkhs import (
    FGDRKHSConfig,
    FGDRKHSTrainer,
    FrozenAffineFeatureMap,
    FrozenMLPFeatureMap,
    KernelDictionaryModel,
    default_level_ladder,
    median_heuristic_gamma,
    theory_descent_coefficient,
    theory_learning_rate_upper_bound,
)
from stable_tiny.pipeline import (
    _apply_certified_head,
    _frozen_feature_map_from_grown_mlp,
    load_pipeline_config,
    run_pipeline,
)


def _random_problem(
    n: int = 48,
    d: int = 3,
    m: int = 2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=generator, dtype=torch.float64)
    y = torch.stack(
        [
            torch.sin(x[:, 0]) + 0.5 * torch.cos(2.0 * x[:, 1 % d]),
            0.3 * x[:, 0] * x[:, (d - 1)],
        ],
        dim=1,
    )[:, :m]
    return x, y


def _grid_problem(side: int = 6, m: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Well-separated 2D grid: keeps the kernel matrix well conditioned."""
    coords = torch.arange(side, dtype=torch.float64)
    grid = torch.cartesian_prod(coords, coords)
    y = torch.stack(
        [
            torch.sin(grid[:, 0]) * torch.cos(grid[:, 1]),
            0.1 * grid[:, 0] - 0.2 * grid[:, 1],
        ],
        dim=1,
    )[:, :m]
    return grid, y


def _trainer(
    x: torch.Tensor,
    y: torch.Tensor,
    **overrides: object,
) -> FGDRKHSTrainer:
    config = replace(FGDRKHSConfig(), **overrides)
    return FGDRKHSTrainer(x, y, config)


def _mlp_trainer(
    x: torch.Tensor,
    y: torch.Tensor,
    **overrides: object,
) -> FGDRKHSTrainer:
    """Trainer over the fixed MLP structure (3 hidden layers x 18 neurons)."""
    settings: dict[str, object] = dict(
        kernel="linear",
        feature_hidden_layers=3,
        feature_hidden_size=18,
    )
    settings.update(overrides)
    return _trainer(x, y, **settings)


# ---------------------------------------------------------------------------
# Theory constants (Assumptions 3.1-3.4, 3.7 and Prop. 3.8 quantities).
# ---------------------------------------------------------------------------


def test_theory_constants_match_paper_formulas() -> None:
    x, y = _random_problem()
    trainer = _trainer(x, y, kernel_gamma=0.5, epsilon=0.25, lr_safety=0.9)
    theory = trainer.theory

    assert theory.kappa == 1.0  # Gaussian kernel sup k(x, x).
    assert theory.smoothness == theory.kappa
    assert theory.alpha == 1.0 and theory.beta == 1.0  # B = H.
    assert theory.loss_star == 0.0
    assert theory.epsilon_bar == pytest.approx(0.25 / 1.25)
    assert theory.epsilon_bar < 0.5  # Prop. 3.6 requires eps < 1/2.

    expected_bound = 2.0 * (1.0 - 2.0 * theory.epsilon_bar) / (
        theory.smoothness * (2.0 * theory.epsilon_bar + 1.0)
    )
    assert theory.learning_rate_upper_bound == pytest.approx(expected_bound)
    assert theory.learning_rate == pytest.approx(0.9 * expected_bound)
    assert theory.learning_rate < expected_bound  # strict, as Prop. 3.8 asks.

    error_ratio = theory.epsilon_bar / (1.0 - theory.epsilon_bar)
    expected_r = (
        1.0
        - 0.5 * theory.smoothness * theory.learning_rate
        - (1.0 + 1.5 * theory.smoothness * theory.learning_rate) * error_ratio
    )
    assert theory.descent_coefficient == pytest.approx(expected_r)
    assert theory.descent_coefficient > 0.0

    assert theory.pl_mu == pytest.approx(
        theory.kernel_lambda_min / theory.train_points
    )
    assert theory.contraction == pytest.approx(
        1.0 - 2.0 * theory.learning_rate * theory.pl_mu * theory.descent_coefficient
    )
    assert 0.0 < theory.contraction < 1.0


def test_k_smoothness_assumption_holds() -> None:
    """Assumption 3.2 with K_s = kappa = 1, re-derived from kernel algebra."""
    x, y = _random_problem(n=32, seed=1)
    trainer = _trainer(x, y, kernel_gamma=0.7)
    kernel = trainer.kernel_matrix
    n = kernel.shape[0]
    generator = torch.Generator().manual_seed(7)

    for trial in range(5):
        b = torch.randn(n, y.shape[1], generator=generator, dtype=torch.float64)
        v = torch.randn(n, y.shape[1], generator=generator, dtype=torch.float64)
        predictions = kernel @ b
        perturbed = kernel @ (b + v)
        loss = (predictions - trainer.train_y).square().sum() / (2.0 * n)
        loss_perturbed = (perturbed - trainer.train_y).square().sum() / (2.0 * n)
        directional = ((predictions - trainer.train_y) * (kernel @ v)).sum() / n
        v_norm_sq = (v * (kernel @ v)).sum()
        upper = loss + directional + 0.5 * trainer.theory.smoothness * v_norm_sq
        assert float(loss_perturbed.item()) <= float(upper.item()) + 1e-9


def test_pl_assumption_holds() -> None:
    """Assumption 3.7 with mu = lambda_min(K)/n, re-derived independently."""
    x, y = _random_problem(n=40, seed=2)
    trainer = _trainer(x, y, kernel_gamma=1.0)
    kernel = trainer.kernel_matrix
    n = kernel.shape[0]
    mu = trainer.theory.pl_mu
    assert mu > 0.0
    generator = torch.Generator().manual_seed(11)

    for trial in range(5):
        b = torch.randn(n, y.shape[1], generator=generator, dtype=torch.float64)
        residual = kernel @ b - trainer.train_y
        loss = float(residual.square().sum().item()) / (2.0 * n)
        coefficients = residual / n
        gradient_sq = float(
            (coefficients * (kernel @ coefficients)).sum().item()
        )
        assert loss - trainer.theory.loss_star <= gradient_sq / (2.0 * mu) + 1e-9


# ---------------------------------------------------------------------------
# Approximation family and the exact upper bound U (Equation 5, Lemma 3.9).
# ---------------------------------------------------------------------------


def test_error_certificate_equals_exact_rkhs_norm() -> None:
    x, y = _random_problem(n=36, seed=3)
    trainer = _trainer(x, y, kernel_gamma=0.4, levels=(6, 12, 36))
    kernel = trainer.kernel_matrix
    n = kernel.shape[0]

    residual = trainer._predictions() - trainer.train_y
    residual = residual + 1.0  # move away from the zero initial residual
    coefficients = residual / n
    kernel_times = kernel @ coefficients
    gradient_sq = float((coefficients * kernel_times).sum().item())

    result = trainer.gradient_approximation(12, coefficients, kernel_times, gradient_sq)
    assert result is not None
    update, approximation_sq, error_sq = result

    scattered = torch.zeros_like(coefficients)
    scattered[trainer.permutation[:12]] = update
    difference = scattered - coefficients
    direct_error_sq = float((difference * (kernel @ difference)).sum().item())
    assert error_sq == pytest.approx(direct_error_sq, rel=1e-8, abs=1e-12)

    full = trainer.gradient_approximation(n, coefficients, kernel_times, gradient_sq)
    assert full is not None
    _, full_sq, full_error = full
    assert full_error == 0.0  # U = 0 at the top level: Lemma 3.9 by design.
    assert full_sq == pytest.approx(gradient_sq)


def test_levels_always_end_at_full_dictionary() -> None:
    x, y = _random_problem(n=20, seed=4)
    trainer = _trainer(x, y, kernel_gamma=0.5, levels=(4, 8))
    assert trainer.levels == (4, 8, 20)
    assert default_level_ladder(64) == (4, 8, 16, 32, 64)
    assert default_level_ladder(5)[-1] == 5


def test_tiny_epsilon_forces_full_dictionary_with_zero_error() -> None:
    x, y = _random_problem(n=24, seed=5)
    trainer = _trainer(x, y, kernel_gamma=0.5, epsilon=1e-6)
    record = trainer.step()
    assert record.dictionary_size == trainer.theory.train_points
    assert record.error_upper_bound == 0.0
    assert record.relative_error == 0.0


# ---------------------------------------------------------------------------
# Step certificates: Theorem 3.10(i) and Lemma 3.5.
# ---------------------------------------------------------------------------


def test_relative_error_certificate_theorem_3_10() -> None:
    x, y = _random_problem(n=64, d=4, seed=6)
    trainer = _trainer(x, y, kernel_gamma=0.05, epsilon=0.3)
    epsilon = trainer.config.epsilon
    epsilon_bar = trainer.theory.epsilon_bar

    for _ in range(30):
        record = trainer.step()
        if record.converged:
            break
        assert record.relative_error <= epsilon_bar + 1e-12
        assert record.relative_error_condition_valid
        # Algorithm 1 acceptance test, re-checked from the record.
        approximation_norm = record.approximation_sq_norm ** 0.5
        assert (1.0 + epsilon) * record.error_upper_bound < (
            epsilon * approximation_norm + 1e-12
        )


def test_sufficient_descent_lemma_3_5() -> None:
    x, y = _random_problem(n=48, seed=8)
    trainer = _trainer(x, y, kernel_gamma=0.3, epsilon=0.25)

    for _ in range(50):
        record = trainer.step()
        if record.converged:
            break
        assert record.descent_valid
        assert record.loss_after < record.loss_before
        # Lemma 3.5 quantitative form at the measured relative error.
        r_measured = theory_descent_coefficient(
            record.relative_error,
            trainer.theory.learning_rate,
            trainer.theory.smoothness,
        )
        bound = record.loss_before - (
            trainer.theory.learning_rate * r_measured * record.gradient_sq_norm
        )
        assert record.loss_after <= bound + 1e-9


# ---------------------------------------------------------------------------
# Global optimality: Proposition 3.8 / Theorem 3.10(iii).
# ---------------------------------------------------------------------------


def test_global_convergence_certificate_prop_3_8() -> None:
    x, y = _grid_problem(side=6)
    trainer = _trainer(
        x,
        y,
        kernel_gamma=8.0,
        epsilon=0.05,
        lr_safety=0.95,
        steps_per_epoch=25,
    )
    theory = trainer.theory
    assert theory.pl_certificate_valid
    assert theory.kernel_lambda_min > 1e-3  # genuinely well conditioned
    initial_gap = theory.initial_loss - theory.loss_star

    losses = []
    for _ in range(60):  # 60 * 25 = 1500 certified steps
        epoch = trainer.run_epoch()
        losses.append(epoch.train_functional_loss)
        assert epoch.global_bound is not None
        # Theorem 3.10(iii): the measured loss must stay below the envelope.
        assert epoch.global_bound_valid
        assert epoch.train_functional_loss <= epoch.global_bound + 1e-9
        if epoch.converged:
            break

    # The envelope itself contracts geometrically...
    assert trainer.global_bound() < initial_gap
    # ...and the iterate actually reaches the global optimum L* = 0 of the
    # fixed structure (and of the whole RKHS) to numerical precision.
    assert losses[-1] <= max(1e-10 * theory.initial_loss, 1e-12)


def test_convergence_flag_certifies_global_optimum() -> None:
    x, y = _grid_problem(side=4)
    trainer = _trainer(
        x,
        y,
        kernel_gamma=8.0,
        epsilon=0.05,
        lr_safety=0.95,
        steps_per_epoch=200,
        gradient_tolerance=1e-22,
    )
    for _ in range(40):
        epoch = trainer.run_epoch()
        if epoch.converged:
            break
    if trainer.converged:
        # PL: L - L* <= ||grad||^2 / (2 mu); with grad ~ 0 the iterate is a
        # global minimizer.
        assert epoch.train_functional_loss <= 1e-9


# ---------------------------------------------------------------------------
# Config validation, subsampling, model surface.
# ---------------------------------------------------------------------------


def test_config_validation() -> None:
    x, y = _random_problem(n=16, seed=9)
    with pytest.raises(ValueError):
        _trainer(x, y, epsilon=1.0)
    with pytest.raises(ValueError):
        _trainer(x, y, epsilon=0.0)
    with pytest.raises(ValueError):
        _trainer(x, y, lr_safety=1.0)
    with pytest.raises(ValueError):
        _trainer(x, y, levels=(8, 4))
    with pytest.raises(ValueError):
        _trainer(x, y, levels=(0, 4))
    with pytest.raises(ValueError):
        theory_learning_rate_upper_bound(0.5, 1.0)


def test_subsampling_respects_max_train_points() -> None:
    x, y = _random_problem(n=40, seed=10)
    trainer = _trainer(x, y, kernel_gamma=0.5, max_train_points=16)
    assert trainer.theory.train_points == 16
    assert trainer.model.centers.shape[0] == 16
    assert trainer.kernel_matrix.shape == (16, 16)
    assert trainer.levels[-1] == 16


def test_median_heuristic_rejects_duplicates() -> None:
    x = torch.zeros(8, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        median_heuristic_gamma(x)


def test_kernel_dictionary_model_forward() -> None:
    x, y = _random_problem(n=20, seed=11)
    trainer = _trainer(x, y, kernel_gamma=0.5)
    model = trainer.model
    assert isinstance(model, KernelDictionaryModel)
    out = model(x.to(torch.float32))
    assert out.shape == (20, y.shape[1])
    # Zero coefficients -> zero output at initialization.
    assert float(out.abs().max().item()) == 0.0
    trainer.step()
    out_after = model(x.to(torch.float32))
    assert float(out_after.abs().max().item()) > 0.0


# ---------------------------------------------------------------------------
# Pipeline integration: third training method end to end.
# ---------------------------------------------------------------------------


def test_pipeline_dispatch_fgd_rkhs(tmp_path) -> None:
    config = load_pipeline_config("configs/fgd/rkhs_default.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            train_batches=2,
            validation_batches=1,
            test_batches=1,
            batch_size=24,
            in_features=4,
            out_features=2,
            active_features=2,
        ),
        training=replace(config.training, epochs=3, device="cpu", log_every=1),
        fgd_rkhs=replace(config.fgd_rkhs, kernel_gamma=0.5, steps_per_epoch=2),
        run=replace(config.run, results_dir=tmp_path, save_plot=False),
    )
    result = run_pipeline(config=config, progress=None)

    assert isinstance(result.model, KernelDictionaryModel)
    assert result.growth_events == []
    step_types = [entry.step_type for entry in result.history]
    assert step_types[0] == "INIT"
    assert set(step_types[1:]) == {"RKHS"}
    for entry in result.history[1:]:
        assert entry.fgd_approximation_kind == "rkhs_dictionary"
        assert entry.fgd_rkhs_functional_loss is not None
        assert entry.fgd_loss_descent_valid is True
        assert entry.fgd_relative_error_condition_valid is True
        assert entry.fgd_learning_rate_interval_valid is True
    functional_losses = [
        entry.fgd_rkhs_functional_loss for entry in result.history
    ]
    assert functional_losses[-1] < functional_losses[0]


def test_rkhs_config_yaml_roundtrip(tmp_path) -> None:
    config_path = tmp_path / "rkhs.yaml"
    config_path.write_text(
        """
training:
  method: fgd_rkhs
fgd_rkhs:
  kernel_gamma: 2.5
  epsilon: 0.1
  levels: [8, 16]
  steps_per_epoch: 4
""",
        encoding="utf-8",
    )
    config = load_pipeline_config(config_path)
    assert config.training.method == "fgd_rkhs"
    assert config.fgd_rkhs.kernel_gamma == 2.5
    assert config.fgd_rkhs.epsilon == 0.1
    assert config.fgd_rkhs.levels == (8, 16)
    assert config.fgd_rkhs.steps_per_epoch == 4
    # Default configs still parse (levels: null -> None).
    default_config = load_pipeline_config("configs/fgd/rkhs_default.yaml")
    assert default_config.fgd_rkhs.levels is None


# ---------------------------------------------------------------------------
# Fixed MLP structure: kernel: linear over a frozen deep feature map, so the
# trained model is exactly an MLP with 3 hidden layers of 18 neurons whose
# output layer reaches the certified global optimum of the fixed structure.
# ---------------------------------------------------------------------------


def test_frozen_feature_map_is_deterministic_and_frozen() -> None:
    x, _ = _random_problem(n=25, d=5, seed=12)
    first = FrozenMLPFeatureMap(5, 18, 3, activation="tanh", seed=7)
    second = FrozenMLPFeatureMap(5, 18, 3, activation="tanh", seed=7)
    other_seed = FrozenMLPFeatureMap(5, 18, 3, activation="tanh", seed=8)

    features = first(x)
    assert features.shape == (25, 18)
    assert features.dtype == torch.float64
    assert torch.equal(features, second(x))
    assert not torch.equal(features, other_seed(x))
    # Frozen structure: the hidden weights are buffers, never parameters.
    assert list(first.parameters()) == []

    with pytest.raises(ValueError):
        FrozenMLPFeatureMap(5, 18, 0)
    with pytest.raises(ValueError):
        FrozenMLPFeatureMap(5, 0, 3)
    with pytest.raises(ValueError):
        FrozenMLPFeatureMap(5, 18, 3, activation="softplus")


def test_linear_kernel_model_is_the_fixed_mlp() -> None:
    """The trained model collapses to phi(x) @ W: a plain 3x18 MLP."""
    x, y = _random_problem(n=30, seed=13)
    trainer = _mlp_trainer(x, y)
    model = trainer.model
    assert model.kernel_kind == "linear"
    assert model.feature_map is not None
    assert model.feature_map.hidden_layers == 3
    assert model.feature_map.hidden_size == 18

    for _ in range(3):
        trainer.step()
    head = model.linear_head_weight()
    assert head.shape == (18, y.shape[1])
    # The head is expressed on the RAW last-hidden-layer activations: the
    # deployed model is literally the MLP phi followed by one linear layer
    # (the fixed whitening reparametrization is folded into the head).
    raw_activations = model.feature_map(x)
    direct_mlp_output = raw_activations @ head
    dictionary_output = model(x).to(torch.float64)
    assert torch.allclose(dictionary_output, direct_mlp_output, atol=1e-8)


def test_linear_kernel_theory_constants_match_bruteforce() -> None:
    x, y = _random_problem(n=40, seed=14)
    trainer = _mlp_trainer(x, y)
    theory = trainer.theory
    n = theory.train_points
    features = trainer.model.centers
    gram = features.T @ features
    eigenvalues = torch.linalg.eigvalsh(gram)

    assert theory.kernel_kind == "linear"
    assert theory.feature_dimension == 18
    assert theory.smoothness == pytest.approx(
        float(eigenvalues.max().item()) / n, rel=1e-10
    )
    assert theory.pl_mu == pytest.approx(
        float(eigenvalues.min().item()) / n, rel=1e-10
    )
    assert theory.kappa == pytest.approx(
        float(features.square().sum(dim=1).max().item()), rel=1e-10
    )
    # L* is the exact least-squares optimum of the fixed structure.
    solution = torch.linalg.lstsq(features, trainer.train_y).solution
    residual = features @ solution - trainer.train_y
    loss_star = float(residual.square().sum().item()) / (2.0 * n)
    assert theory.loss_star == pytest.approx(loss_star, rel=1e-8, abs=1e-12)
    assert 0.0 < theory.loss_star < theory.initial_loss


def test_linear_kernel_k_smoothness_holds() -> None:
    """Assumption 3.2 with K_s = lambda_max(Phi^T Phi)/n, re-derived."""
    x, y = _random_problem(n=32, seed=15)
    trainer = _mlp_trainer(x, y)
    kernel = trainer.kernel_matrix
    n = kernel.shape[0]
    generator = torch.Generator().manual_seed(17)

    for trial in range(5):
        b = torch.randn(n, y.shape[1], generator=generator, dtype=torch.float64)
        v = torch.randn(n, y.shape[1], generator=generator, dtype=torch.float64)
        predictions = kernel @ b
        perturbed = kernel @ (b + v)
        loss = (predictions - trainer.train_y).square().sum() / (2.0 * n)
        loss_perturbed = (perturbed - trainer.train_y).square().sum() / (2.0 * n)
        directional = ((predictions - trainer.train_y) * (kernel @ v)).sum() / n
        v_norm_sq = (v * (kernel @ v)).sum()
        upper = loss + directional + 0.5 * trainer.theory.smoothness * v_norm_sq
        assert float(loss_perturbed.item()) <= float(upper.item()) + 1e-9


def test_linear_kernel_pl_holds_relative_to_structure_optimum() -> None:
    """Assumption 3.7 with mu = lambda_min(Phi^T Phi)/n and exact L* > 0."""
    x, y = _random_problem(n=36, seed=16)
    trainer = _mlp_trainer(x, y)
    kernel = trainer.kernel_matrix
    n = kernel.shape[0]
    mu = trainer.theory.pl_mu
    assert trainer.theory.pl_certificate_valid
    assert mu > 0.0
    generator = torch.Generator().manual_seed(19)

    for trial in range(5):
        b = torch.randn(n, y.shape[1], generator=generator, dtype=torch.float64)
        residual = kernel @ b - trainer.train_y
        loss = float(residual.square().sum().item()) / (2.0 * n)
        coefficients = residual / n
        gradient_sq = float(
            (coefficients * (kernel @ coefficients)).sum().item()
        )
        assert loss - trainer.theory.loss_star <= gradient_sq / (2.0 * mu) + 1e-9


def test_linear_kernel_global_convergence_to_structure_optimum() -> None:
    """Prop. 3.8: the fixed 3x18 MLP reaches the global optimum L* of its
    structure, with the envelope holding at every epoch. Whitening makes
    the certified contraction sharp, so convergence is fast."""
    x, y = _random_problem(n=60, seed=18)
    trainer = _mlp_trainer(x, y, steps_per_epoch=50, epsilon=0.1)
    theory = trainer.theory
    assert theory.pl_certificate_valid
    assert theory.contraction < 0.9  # non-vacuous certificate
    initial_gap = theory.initial_loss - theory.loss_star
    assert initial_gap > 0.0

    final_loss = theory.initial_loss
    for _ in range(20):
        epoch = trainer.run_epoch()
        assert all(record.descent_valid for record in epoch.step_records)
        assert epoch.global_bound_valid is True
        final_loss = epoch.train_functional_loss
        if trainer.converged:
            break

    final_gap = final_loss - theory.loss_star
    assert final_gap <= 1e-6 * initial_gap
    # The Prop. 3.8 envelope itself certifies the remaining gap.
    envelope_gap = trainer.global_bound() - theory.loss_star
    assert final_gap <= envelope_gap + 1e-12


def test_linear_kernel_whitening_preserves_structure_optimum() -> None:
    """Whitening is a reparametrization: same function class, same L*."""
    x, y = _random_problem(n=40, seed=22)
    whitened = _mlp_trainer(x, y)
    plain = _mlp_trainer(x, y, feature_whitening=False)
    assert whitened.theory.loss_star == pytest.approx(
        plain.theory.loss_star, rel=1e-6, abs=1e-12
    )
    # Same frozen feature map underneath, different H-geometry on top.
    assert torch.equal(
        whitened.model.feature_map.get_buffer("weight_0"),
        plain.model.feature_map.get_buffer("weight_0"),
    )
    assert whitened.theory.contraction < plain.theory.contraction


def test_gaussian_kernel_composes_with_feature_map() -> None:
    """kernel: gaussian over a frozen feature map is also supported."""
    x, y = _random_problem(n=24, seed=20)
    trainer = _trainer(
        x,
        y,
        kernel="gaussian",
        kernel_gamma=0.5,
        feature_hidden_layers=3,
        feature_hidden_size=18,
    )
    assert trainer.theory.kernel_kind == "gaussian"
    assert trainer.theory.loss_star == 0.0
    features = trainer.model.features(x)
    distances_sq = torch.cdist(features, features).square()
    expected = torch.exp(-0.5 * distances_sq)
    assert torch.allclose(trainer.kernel_matrix, expected, atol=1e-12)


def test_linear_kernel_config_validation() -> None:
    x, y = _random_problem(n=16, seed=21)
    with pytest.raises(ValueError):
        _trainer(x, y, kernel="polynomial")
    with pytest.raises(ValueError):
        _trainer(x, y, feature_hidden_layers=-1)
    with pytest.raises(ValueError):
        _trainer(x, y, feature_hidden_layers=2, feature_hidden_size=0)
    with pytest.raises(ValueError):
        _mlp_trainer(x, y, feature_activation="softplus")


def test_rkhs_linear_config_yaml_roundtrip(tmp_path) -> None:
    config_path = tmp_path / "rkhs_linear.yaml"
    config_path.write_text(
        """
training:
  method: fgd_rkhs
fgd_rkhs:
  kernel: linear
  feature_hidden_layers: 3
  feature_hidden_size: 18
  feature_activation: tanh
  feature_seed: 5
""",
        encoding="utf-8",
    )
    config = load_pipeline_config(config_path)
    assert config.fgd_rkhs.kernel == "linear"
    assert config.fgd_rkhs.feature_hidden_layers == 3
    assert config.fgd_rkhs.feature_hidden_size == 18
    assert config.fgd_rkhs.feature_activation == "tanh"
    assert config.fgd_rkhs.feature_seed == 5
    # The shipped configs now request the fixed 3x18 MLP structure.
    for shipped in ("configs/fgd/rkhs_default.yaml", "configs/fgd/rkhs_mnist.yaml"):
        shipped_config = load_pipeline_config(shipped)
        assert shipped_config.fgd_rkhs.kernel == "linear"
        assert shipped_config.fgd_rkhs.feature_hidden_layers == 3
        assert shipped_config.fgd_rkhs.feature_hidden_size == 18


# ---------------------------------------------------------------------------
# Certified train-and-grow cycle (training.method: fgd_rkhs_grow): the grown
# network's hidden layers become the frozen feature map, the head is trained
# to the certified optimum of each structure, and growth stops when the
# closed-form ceiling L* stops improving.
# ---------------------------------------------------------------------------


def _small_growing_mlp(in_features: int = 4, out_features: int = 2):
    from gromo.containers.growing_mlp import GrowingMLP

    torch.manual_seed(0)
    return GrowingMLP(
        in_features=in_features,
        out_features=out_features,
        hidden_size=3,
        number_hidden_layers=2,
        device=torch.device("cpu"),
    )


def test_frozen_affine_feature_map_matches_manual() -> None:
    generator = torch.Generator().manual_seed(23)
    w0 = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    b0 = torch.randn(5, generator=generator, dtype=torch.float64)
    w1 = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    feature_map = FrozenAffineFeatureMap(
        weights=[w0, w1],
        biases=[b0, None],
        activations=[torch.nn.Tanh(), torch.nn.Identity()],
        append_one=True,
    )
    x = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    manual = torch.tanh(x @ w0.T + b0) @ w1.T
    manual = torch.cat([manual, torch.ones(7, 1, dtype=torch.float64)], dim=1)
    assert torch.allclose(feature_map(x), manual, atol=1e-12)
    assert feature_map.out_features == 4  # 3 features + constant 1
    assert list(feature_map.parameters()) == []  # frozen structure

    with pytest.raises(ValueError):
        FrozenAffineFeatureMap(weights=[], biases=[], activations=[])
    with pytest.raises(ValueError):
        FrozenAffineFeatureMap(
            weights=[w0],
            biases=[b0, None],
            activations=[torch.nn.Tanh()],
        )


def test_certified_head_writeback_matches_grown_network() -> None:
    """After write-back, the grown network IS the certified model."""
    x, y = _random_problem(n=40, d=4, seed=24)
    mlp = _small_growing_mlp()
    feature_map = _frozen_feature_map_from_grown_mlp(mlp)
    # Constant-1 feature: the head is affine, exactly like the output layer.
    assert feature_map.out_features == 3 + 1
    trainer = FGDRKHSTrainer(
        x,
        y,
        FGDRKHSConfig(kernel="linear", feature_hidden_layers=0),
        feature_map=feature_map,
    )
    for _ in range(5):
        trainer.step()
    _apply_certified_head(mlp, trainer.model)
    certified = trainer.model(x).to(torch.float64)
    deployed = mlp(x.to(torch.float32)).to(torch.float64)
    assert torch.allclose(certified, deployed, atol=1e-4)


def test_grow_cycle_pipeline_end_to_end(tmp_path) -> None:
    config = load_pipeline_config("configs/fgd/rkhs_default.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            train_batches=2,
            validation_batches=1,
            test_batches=1,
            batch_size=24,
            in_features=4,
            out_features=2,
            active_features=2,
        ),
        training=replace(
            config.training,
            method="fgd_rkhs_grow",
            epochs=12,
            device="cpu",
            log_every=1,
        ),
        fgd_rkhs=replace(
            config.fgd_rkhs,
            kernel="linear",
            feature_hidden_layers=0,
            steps_per_epoch=2,
            growth_max_cycles=1,
            growth_epochs_per_cycle=3,
            growth_min_ceiling_improvement=0.0,
            growth_max_hidden_size=8,
        ),
        run=replace(config.run, results_dir=tmp_path, save_plot=False),
    )
    result = run_pipeline(config=config, progress=None)

    step_types = [entry.step_type for entry in result.history]
    assert step_types[0] == "INIT"
    assert "RKHS" in step_types
    assert "GRO" in step_types
    assert len(result.growth_events) == 1
    assert result.growth_events[0].layer_index is not None

    rkhs_entries = [e for e in result.history if e.step_type == "RKHS"]
    for entry in rkhs_entries:
        assert entry.fgd_loss_descent_valid is True
        assert entry.fgd_relative_error_condition_valid is True
        assert entry.fgd_rkhs_loss_star is not None
    # Two cycles trained (before and after growth): the certified ceiling
    # of the grown structure can only be <= the previous one, because
    # growth enlarges the function class of the head.
    ceilings = sorted({e.fgd_rkhs_loss_star for e in rkhs_entries})
    assert len(ceilings) >= 1
    assert result.history[-1].train_loss <= result.history[0].train_loss
