"""Tests for the empirical-PL certified trainer (fgdlib.empirical_pl).

The certificates are re-derived independently: the measured PL constant is
compared against a brute-force Jacobian Gram, the PL inequality is checked
as the algebraic identity it is, step acceptance is exercised under an
aggressive learning rate, and the envelope is validated end to end in the
over- and under-parametrized regimes.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fgdlib.empirical_pl import EmpiricalPLConfig, EmpiricalPLTrainer
from stable_tiny.pipeline import load_pipeline_config, run_pipeline


def _problem(n: int = 12, d: int = 3, m: int = 2, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=generator)
    y = torch.randn(n, m, generator=generator)
    return x, y


def _mlp(d: int = 3, hidden: int = 8, m: int = 2, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(d, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, m),
    )


def _brute_force_mu(model: torch.nn.Module, x: torch.Tensor) -> float:
    parameters = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for i in range(x.shape[0]):
        out = model(x[i : i + 1]).reshape(-1)
        for c in range(out.shape[0]):
            grads = torch.autograd.grad(
                out[c], parameters, retain_graph=True, allow_unused=True
            )
            rows.append(
                torch.cat(
                    [
                        g.reshape(-1) if g is not None else torch.zeros(p.numel())
                        for g, p in zip(grads, parameters)
                    ]
                ).to(torch.float64)
            )
    jacobian = torch.stack(rows)
    gram = jacobian @ jacobian.T
    eigenvalues = torch.linalg.eigvalsh(gram)
    return max(float(eigenvalues.min().item()), 0.0) / x.shape[0]


def test_measured_mu_matches_bruteforce() -> None:
    x, y = _problem()
    model = _mlp()
    trainer = EmpiricalPLTrainer(
        model, x, y, EmpiricalPLConfig(certificate_points=12)
    )
    mu, valid = trainer.measure_mu()
    expected = _brute_force_mu(model, trainer.train_x[trainer.certificate_indices])
    # The vectorized and looped Jacobians differ by float32 accumulation
    # order; the eigenvalue agrees to that precision.
    assert mu == pytest.approx(expected, rel=1e-4, abs=1e-12)
    # Overparametrized here: P = 3*8+8+8*2+2 = 50 >= 12*2 = 24 rows.
    assert valid and mu > 0.0


def test_pl_inequality_is_an_identity() -> None:
    """||grad L||^2 >= 2 mu L holds by algebra at the measurement point."""
    x, y = _problem(n=10, seed=3)
    model = _mlp(seed=3)
    trainer = EmpiricalPLTrainer(
        model, x, y, EmpiricalPLConfig(certificate_points=10)
    )
    mu, _ = trainer.measure_mu()
    # Loss and gradient restricted to the certificate subset.
    subset_x = trainer.train_x[trainer.certificate_indices]
    subset_y = trainer.train_y[trainer.certificate_indices]
    predictions = model(subset_x)
    residual = predictions - subset_y
    loss = residual.square().sum() / (2.0 * subset_x.shape[0])
    grads = torch.autograd.grad(loss, list(model.parameters()))
    gradient_sq = float(sum(g.square().sum().item() for g in grads))
    assert gradient_sq >= 2.0 * mu * float(loss.item()) - 1e-9


def test_steps_certify_descent_under_aggressive_learning_rate() -> None:
    x, y = _problem(n=16, seed=5)
    model = _mlp(seed=5)
    trainer = EmpiricalPLTrainer(
        model,
        x,
        y,
        EmpiricalPLConfig(learning_rate=64.0, certificate_points=16),
    )
    trainer.measure_mu()
    losses = [trainer.initial_loss]
    for _ in range(20):
        record = trainer.step()
        assert record.accepted
        if not record.converged:
            assert record.descent_coefficient >= trainer.config.r_min
        losses.append(record.loss_after)
    assert all(b <= a + 1e-12 for a, b in zip(losses, losses[1:]))
    assert losses[-1] < losses[0]


def test_envelope_certifies_overparametrized_convergence() -> None:
    """mu > 0 throughout => measured Prop. 3.8 envelope holds and bites."""
    x, y = _problem(n=8, d=2, m=1, seed=7)
    model = _mlp(d=2, hidden=32, m=1, seed=7)  # P >> n*m: verified NTK regime
    trainer = EmpiricalPLTrainer(
        model,
        x,
        y,
        EmpiricalPLConfig(certificate_points=8, steps_per_epoch=20),
    )
    initial = trainer.initial_loss
    for _ in range(30):
        result = trainer.run_epoch()
        assert result.mu_valid, "certificate must stay on in this regime"
        assert result.envelope_valid is True
        if result.converged:
            break
    assert result.train_loss < 1e-4 * initial
    assert result.envelope < initial  # the envelope actually contracted


def test_strict_mode_refuses_descent_without_pl() -> None:
    """Strict mode: no descent step may run while mu is collapsed."""
    x, y = _problem(n=40, d=3, m=2, seed=19)
    model = _mlp(hidden=2, seed=19)  # underparametrized -> mu = 0
    trainer = EmpiricalPLTrainer(
        model,
        x,
        y,
        EmpiricalPLConfig(certificate_points=40, strict_certificates=True),
    )
    loss_before = trainer.initial_loss
    result = trainer.run_epoch()
    assert result.mu_collapsed
    assert result.step_records == []  # not a single uncertified step
    assert result.train_loss == pytest.approx(loss_before, rel=1e-12)


def test_mu_is_zero_when_underparametrized() -> None:
    x, y = _problem(n=40, d=3, m=2, seed=9)
    model = _mlp(hidden=2, seed=9)  # P = 3*2+2+2*2+2 = 14 < 40*2 rows
    trainer = EmpiricalPLTrainer(
        model, x, y, EmpiricalPLConfig(certificate_points=40)
    )
    mu, valid = trainer.measure_mu()
    assert mu == pytest.approx(0.0, abs=1e-10)
    assert not valid  # certificate honestly off -> growth trigger


def test_fast_jacobian_matches_autograd_loop() -> None:
    """The vectorized torch.func path must match the plain autograd loop."""
    x, y = _problem(n=6, seed=15)
    model = _mlp(seed=15)
    trainer = EmpiricalPLTrainer(
        model, x, y, EmpiricalPLConfig(certificate_points=6)
    )
    subset = trainer.train_x[trainer.certificate_indices]
    fast = trainer._jacobian_rows_fast(subset)
    loop = trainer._jacobian_rows_loop(subset)
    assert fast.shape == loop.shape
    assert torch.allclose(fast, loop, atol=1e-6)


def test_rank_shortcut_skips_jacobian_when_underparametrized() -> None:
    """P < certificate rows => mu = 0 exactly, without building J."""
    x, y = _problem(n=40, seed=17)
    model = _mlp(hidden=2, seed=17)  # P = 14 < 80 rows
    trainer = EmpiricalPLTrainer(
        model, x, y, EmpiricalPLConfig(certificate_points=40)
    )
    calls = {"count": 0}
    original = trainer._jacobian_rows

    def counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    trainer._jacobian_rows = counting  # type: ignore[method-assign]
    mu, valid = trainer.measure_mu()
    assert mu == 0.0 and not valid
    assert calls["count"] == 0  # the shortcut avoided all Jacobian work


def test_ridge_gradient_and_envelope_semantics() -> None:
    """With ridge: gradient includes lambda*theta; envelope honestly off."""
    x, y = _problem(n=10, seed=11)
    model = _mlp(seed=11)
    ridge = 0.05
    trainer = EmpiricalPLTrainer(
        model,
        x,
        y,
        EmpiricalPLConfig(certificate_points=10, ridge=ridge),
    )
    assert trainer.envelope_enabled is False
    objective, gradients = trainer._loss_and_gradients()
    # Brute-force gradient of F = L + (ridge/2)||theta||^2.
    predictions = model(trainer.train_x)
    data_loss = (predictions - trainer.train_y).square().sum() / (
        2.0 * trainer.train_x.shape[0]
    )
    penalty = 0.5 * ridge * sum(
        p.square().sum() for p in model.parameters()
    )
    expected = torch.autograd.grad(data_loss + penalty, list(model.parameters()))
    for got, want in zip(gradients, expected):
        assert torch.allclose(got, want, atol=1e-6)
    assert objective == pytest.approx(
        float((data_loss + penalty).item()), rel=1e-5
    )
    trainer.measure_mu()
    result = trainer.run_epoch()
    assert result.envelope_enabled is False
    assert result.envelope_valid is None
    # Certified sufficient descent still holds on the regularized objective.
    assert all(r.accepted for r in result.step_records)


def test_stall_declares_stationarity() -> None:
    """Consecutive fully-rejected steps mark convergence at precision."""
    x, y = _problem(n=10, seed=13)
    model = _mlp(seed=13)
    trainer = EmpiricalPLTrainer(
        model,
        x,
        y,
        EmpiricalPLConfig(
            learning_rate=1e6,
            max_backtracks=0,  # every oversized step is rejected outright
            max_rejected_steps=3,
            certificate_points=10,
        ),
    )
    trainer.measure_mu()
    for _ in range(3):
        record = trainer.step()
        assert not record.accepted
    assert trainer.converged


def test_config_validation() -> None:
    x, y = _problem()
    model = _mlp()
    for bad in (
        {"r_min": 0.0},
        {"r_min": 1.0},
        {"backtrack_factor": 1.0},
        {"learning_rate": 0.0},
        {"steps_per_epoch": 0},
        {"certificate_points": 0},
        {"lr_recovery": 0.5},
        {"ridge": -0.1},
        {"max_rejected_steps": 0},
    ):
        with pytest.raises(ValueError):
            EmpiricalPLTrainer(
                model, x, y, replace(EmpiricalPLConfig(), **bad)
            )


def test_pipeline_dispatch_fgd_pl_with_growth(tmp_path) -> None:
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
        model=replace(config.model, hidden_size=2, number_hidden_layers=2),
        training=replace(
            config.training,
            method="fgd_pl",
            epochs=10,
            device="cpu",
            log_every=1,
        ),
        fgd_pl=replace(
            config.fgd_pl,
            certificate_points=24,
            steps_per_epoch=5,
            growth_cooldown_epochs=1,
            growth_max_events=2,
            growth_max_hidden_size=8,
            # Declare any progress "stalled" so the mu trigger drives
            # growth in this short run.
            growth_min_progress=10.0,
        ),
        run=replace(config.run, results_dir=tmp_path, save_plot=False),
    )
    result = run_pipeline(config=config, progress=None)

    step_types = [entry.step_type for entry in result.history]
    assert step_types[0] == "INIT"
    assert "PL" in step_types
    pl_entries = [e for e in result.history if e.step_type == "PL"]
    for entry in pl_entries:
        assert entry.fgd_pl_mu is not None
        assert entry.fgd_loss_descent_valid is True
    # Start 2x2 is underparametrized for 48*2 certificate rows -> mu = 0
    # -> the certificate-driven growth trigger must have fired.
    assert "GRO" in step_types
    assert len(result.growth_events) >= 1
    # The ceiling arbiter records the certified L* of the grown structure.
    growth_entries = [e for e in result.history if e.step_type == "GRO"]
    for entry in growth_entries:
        assert entry.fgd_rkhs_loss_star is not None
    assert result.history[-1].train_loss <= result.history[0].train_loss


def test_growth_must_be_earned(tmp_path) -> None:
    """With growth_min_progress=0 the loss never counts as stalled
    (relative improvement >= 0 on accepted steps), so mu collapse alone
    must NOT trigger growth."""
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
        model=replace(config.model, hidden_size=2, number_hidden_layers=2),
        training=replace(
            config.training,
            method="fgd_pl",
            epochs=6,
            device="cpu",
            log_every=1,
        ),
        fgd_pl=replace(
            config.fgd_pl,
            certificate_points=24,
            steps_per_epoch=5,
            growth_cooldown_epochs=1,
            growth_max_events=2,
            growth_min_progress=0.0,
            # The earned-progress gate only governs non-strict mode.
            strict_certificates=False,
        ),
        run=replace(config.run, results_dir=tmp_path, save_plot=False),
    )
    result = run_pipeline(config=config, progress=None)
    step_types = [entry.step_type for entry in result.history]
    assert "GRO" not in step_types
    assert result.growth_events == []


def test_pl_config_yaml_roundtrip(tmp_path) -> None:
    config_path = tmp_path / "pl.yaml"
    config_path.write_text(
        """
training:
  method: fgd_pl
fgd_pl:
  learning_rate: 2.0
  r_min: 0.25
  certificate_points: 64
  growth_max_hidden_size: 18
""",
        encoding="utf-8",
    )
    config = load_pipeline_config(config_path)
    assert config.training.method == "fgd_pl"
    assert config.fgd_pl.learning_rate == 2.0
    assert config.fgd_pl.r_min == 0.25
    assert config.fgd_pl.certificate_points == 64
    assert config.fgd_pl.growth_max_hidden_size == 18
    shipped = load_pipeline_config("configs/fgd/pl_mnist.yaml")
    assert shipped.training.method == "fgd_pl"
    assert shipped.fgd_pl.growth_max_hidden_size == 18
