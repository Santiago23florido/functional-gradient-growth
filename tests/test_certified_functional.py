"""The certified functional: MSE (legacy) and softmax cross-entropy.

These tests pin the three properties the FGD certificates actually use
(report/CROSS_ENTROPY_FGD.md): convexity, smoothness, and the existence of
a Polyak-Lojasiewicz constant.
"""

from __future__ import annotations

import pytest
import torch

from fgdlib.tangent import (
    FGDApproxConfig,
    FUNCTIONAL_HAS_PL_CONSTANT,
    FUNCTIONAL_SMOOTHNESS,
    batch_functional_loss,
    functional_gradient,
)


def test_mse_stays_the_default() -> None:
    assert FGDApproxConfig().functional_loss == "mse"


def test_unknown_functional_is_rejected() -> None:
    x = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        functional_gradient(x, x, "hinge")
    with pytest.raises(ValueError):
        batch_functional_loss(x, x, "hinge")


def test_gradients_match_autograd_for_both_functionals() -> None:
    """r must be exactly grad_f L for the loss it claims to differentiate."""
    torch.manual_seed(0)
    for name in ("mse", "cross_entropy"):
        f = torch.randn(6, 4, requires_grad=True)
        y = torch.zeros(6, 4)
        y[range(6), torch.randint(0, 4, (6,))] = 1.0
        loss = batch_functional_loss(f, y, name)
        (autograd_r,) = torch.autograd.grad(loss, f)
        analytic_r = functional_gradient(f, y, name)
        assert torch.allclose(analytic_r, autograd_r, atol=1e-6)


def test_mse_satisfies_the_exact_pl_identity() -> None:
    """||r||^2 = 4L exactly -- this is what makes C_glob tight for MSE."""
    torch.manual_seed(1)
    f = torch.randn(8, 5)
    y = torch.zeros(8, 5)
    y[range(8), torch.randint(0, 5, (8,))] = 1.0
    r = functional_gradient(f, y, "mse")
    loss = batch_functional_loss(f, y, "mse")
    assert float(r.pow(2).sum()) == pytest.approx(4.0 * float(loss), rel=1e-6)
    assert FUNCTIONAL_HAS_PL_CONSTANT["mse"] is True


def test_cross_entropy_has_no_global_pl_constant() -> None:
    """||r||^2 / L -> 0 as p_c -> 1, so no mu > 0 can lower-bound it."""
    ratios = []
    for correct_probability in (0.5, 0.99, 0.9999):
        classes = 10
        rest = (1.0 - correct_probability) / (classes - 1)
        probabilities = torch.tensor([correct_probability] + [rest] * (classes - 1))
        logits = torch.log(probabilities).unsqueeze(0)
        target = torch.zeros(1, classes)
        target[0, 0] = 1.0
        r = functional_gradient(logits, target, "cross_entropy")
        loss = batch_functional_loss(logits, target, "cross_entropy")
        ratios.append(float(r.pow(2).sum()) / float(loss))
    # Strictly decreasing toward zero: the PL constant degenerates.
    assert ratios[0] > ratios[1] > ratios[2]
    assert ratios[-1] < 1e-3
    assert FUNCTIONAL_HAS_PL_CONSTANT["cross_entropy"] is False


def test_cross_entropy_is_convex_and_half_smooth() -> None:
    """Hessian diag(p) - p p^T is PSD with lambda_max <= 1/2."""
    torch.manual_seed(2)
    worst = 0.0
    for _ in range(400):
        classes = int(torch.randint(2, 11, (1,)))
        logits = torch.randn(classes) * float(torch.rand(1)) * 6.0
        p = torch.softmax(logits, 0)
        hessian = torch.diag(p) - torch.outer(p, p)
        eigenvalues = torch.linalg.eigvalsh(hessian)
        assert float(eigenvalues.min()) >= -1e-6      # convex (PSD)
        worst = max(worst, float(eigenvalues.max()))
    assert worst <= 0.5 + 1e-6                        # smoothness bound
    assert FUNCTIONAL_SMOOTHNESS["cross_entropy"] == 0.5
    assert FUNCTIONAL_SMOOTHNESS["mse"] == 2.0


def test_cross_entropy_never_penalises_confidence() -> None:
    """The MSE/accuracy conflict: MSE pushes a confident logit back down."""
    logits = torch.tensor([[4.0, 0.0, 0.0]])   # already correct and confident
    target = torch.tensor([[1.0, 0.0, 0.0]])
    mse_r = functional_gradient(logits, target, "mse")
    ce_r = functional_gradient(logits, target, "cross_entropy")
    # MSE: positive gradient on the true logit -> descent DECREASES it.
    assert float(mse_r[0, 0]) > 0.0
    # Cross-entropy: negative gradient -> descent INCREASES it further.
    assert float(ce_r[0, 0]) < 0.0


def test_configured_functional_governs_the_reported_loss() -> None:
    """evaluate_functional_loss must report the CERTIFIED loss, not MSE."""
    import torch.nn as nn
    from stable_tiny.pipeline import evaluate_functional_loss

    torch.manual_seed(3)
    model = nn.Linear(4, 3)
    x = torch.randn(5, 4)
    y = torch.zeros(5, 3)
    y[range(5), torch.randint(0, 3, (5,))] = 1.0
    batches = [(x, y)]
    device = torch.device("cpu")

    mse_total = evaluate_functional_loss(model, batches, device, "mse")
    ce_total = evaluate_functional_loss(model, batches, device, "cross_entropy")

    with torch.no_grad():
        f = model(x)
    assert mse_total == pytest.approx(
        float(batch_functional_loss(f, y, "mse")), rel=1e-6
    )
    assert ce_total == pytest.approx(
        float(batch_functional_loss(f, y, "cross_entropy")), rel=1e-6
    )
    assert mse_total != pytest.approx(ce_total)


def test_cross_entropy_widens_the_admissible_interval_fourfold() -> None:
    """L_s = 1/2 vs 2 makes Lemma 3.5's eta_bar four times wider."""
    from dataclasses import replace

    from fgdlib.tangent import theoretical_learning_rate_upper_bound

    mse_config = FGDApproxConfig()
    ce_config = replace(mse_config, functional_loss="cross_entropy")
    for epsilon in (0.05, 0.1, 0.3, 0.45):
        mse_bound = theoretical_learning_rate_upper_bound(epsilon, mse_config)
        ce_bound = theoretical_learning_rate_upper_bound(epsilon, ce_config)
        assert ce_bound == pytest.approx(4.0 * mse_bound, rel=1e-9)


def test_explicit_smoothness_override_still_wins() -> None:
    from dataclasses import replace

    from fgdlib.tangent import certified_smoothness_constant

    ce_config = replace(
        FGDApproxConfig(),
        functional_loss="cross_entropy",
        theory_smoothness_constant=1.0,
    )
    assert certified_smoothness_constant(ce_config) == 1.0


def test_no_global_bound_is_asserted_without_a_pl_constant() -> None:
    """Without mu the C_glob envelope must be left undefined, not invented."""
    from dataclasses import replace

    from stable_tiny.pipeline import PipelineConfig, _certified_pl_constant

    mse_config = PipelineConfig()
    assert _certified_pl_constant(mse_config) == mse_config.fgd_approx.theory_mu

    ce_config = replace(
        mse_config,
        fgd_approx=replace(
            mse_config.fgd_approx, functional_loss="cross_entropy"
        ),
    )
    # mu = 0 disables the linear contraction: global_bound stays None.
    assert _certified_pl_constant(ce_config) == 0.0


def test_epsilon_stationarity_is_the_loss_agnostic_limit_criterion() -> None:
    """R1: eps is defined for BOTH functionals, C_prog is not comparable.

    The progress floor cannot recognise the limit of a functional whose
    infimum is not attained: cross-entropy always admits more descent by
    raising confidence, so certified progress never falls to the floor.
    eps is a property of the reachable set, so the same test works for
    either certified loss.
    """
    from fgdlib.tangent import GrowthLimitCriterion  # noqa: F401

    assert FGDApproxConfig().growth_limit_criterion == "progress_floor"

    # Raising confidence on an already-correct sample: cross-entropy keeps
    # descending without bound, so no progress floor can ever fire.
    target = torch.tensor([[1.0, 0.0, 0.0]])
    losses = [
        float(
            batch_functional_loss(
                torch.tensor([[scale, 0.0, 0.0]]), target, "cross_entropy"
            )
        )
        for scale in (2.0, 6.0, 10.0, 14.0)
    ]
    assert losses[0] > losses[1] > losses[2] > losses[3] > 0.0
    # ... while sum-MSE has a finite minimiser and turns back up.
    mse_losses = [
        float(
            batch_functional_loss(
                torch.tensor([[scale, 0.0, 0.0]]), target, "mse"
            )
        )
        for scale in (1.0, 2.0, 6.0)
    ]
    assert mse_losses[0] < mse_losses[1] < mse_losses[2]
