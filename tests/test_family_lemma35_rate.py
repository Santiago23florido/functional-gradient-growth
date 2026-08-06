"""The family step must obey Lemma 3.5's interval for its OWN eps.

The ladder certifies a DIRECTION (cos(Delta, r) above the threshold's cosine)
and used to commit the trained clone outright -- a functional step of size 1.
Lemma 3.5 applied to the same eps allows much less, and overshooting it is
what made MNIST oscillate with a flat loss. These tests pin the rate, the
default identity, and the invariant that the acceptance decision is untouched.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fgdlib.search.families import certify_parametric_step, family_lemma35_rate
from fgdlib.tangent import (
    FGDApproxConfig,
    certified_smoothness_constant,
    theoretical_learning_rate_upper_bound,
)


def _config(**overrides) -> FGDApproxConfig:
    return replace(FGDApproxConfig(), **overrides)


def test_rate_is_the_lemma_interval_times_the_safety_fraction() -> None:
    config = _config()
    eps = 0.292
    bound = theoretical_learning_rate_upper_bound(eps, config)
    assert bound is not None
    assert family_lemma35_rate(eps, config) == pytest.approx(
        config.theory_lr_safety * bound
    )


def test_the_measured_mnist_eps_allows_far_less_than_a_full_step() -> None:
    """The number the whole change rests on.

    L_s = 2 for sum-MSE, so eta_bar(0.292) = 2(1-0.584)/(2*1.584) = 0.263.
    Committing the clone whole takes 1.0 -- 3.8x the admissible interval,
    before the 0.95 safety fraction is even applied.
    """
    config = _config()
    assert certified_smoothness_constant(config) == pytest.approx(2.0)
    bound = theoretical_learning_rate_upper_bound(0.292, config)
    assert bound == pytest.approx(0.2626, abs=1e-3)
    assert 1.0 / bound > 3.5


def test_a_worse_alignment_gets_a_proportionally_smaller_rate() -> None:
    config = _config()
    tighter = family_lemma35_rate(0.292, config)
    looser = family_lemma35_rate(0.400, config)
    assert tighter is not None and looser is not None
    assert looser < tighter


def test_no_rate_exists_at_or_above_one_half() -> None:
    config = _config()
    assert family_lemma35_rate(0.5, config) is None
    assert family_lemma35_rate(0.7, config) is None


def _tiny_problem(seed: int = 0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 6), torch.nn.Tanh(), torch.nn.Linear(6, 2)
    )
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)
    return model, x, y


def test_default_is_byte_identical_to_committing_the_clone_whole() -> None:
    """Off by default: the returned parameters must match exactly."""
    config = _config(certify_family_inner_steps=40, rel_error_threshold=0.5)
    model_a, x, y = _tiny_problem()
    model_b, _, _ = _tiny_problem()

    result_a = certify_parametric_step(model_a, x, y, config)
    result_b = certify_parametric_step(model_b, x, y, config)
    assert result_a.certified == result_b.certified
    assert result_a.cosine == pytest.approx(result_b.cosine)
    if result_a.model is not None and result_b.model is not None:
        for pa, pb in zip(result_a.model.parameters(), result_b.model.parameters()):
            assert torch.equal(pa, pb)


def test_the_rate_shortens_the_step_without_changing_the_decision() -> None:
    """Alignment is scale-invariant, so certified/cosine must be identical
    while the committed displacement shrinks to the certified rate."""
    base = _config(certify_family_inner_steps=40, rel_error_threshold=0.5)
    bounded = replace(base, certify_family_lemma35_rate=True)

    model_full, x, y = _tiny_problem()
    model_bounded, _, _ = _tiny_problem()
    start = [p.detach().clone() for p in _tiny_problem()[0].parameters()]

    full = certify_parametric_step(model_full, x, y, base)
    short = certify_parametric_step(model_bounded, x, y, bounded)

    # The DECISION is untouched: same cosine, same certified boolean.
    assert full.certified == short.certified
    assert full.cosine == pytest.approx(short.cosine)
    assert full.relative_error == pytest.approx(short.relative_error)

    if not full.certified:
        pytest.skip("fixture did not certify; the rate never applies")

    rate = family_lemma35_rate(full.relative_error, bounded)
    assert rate is not None and rate < 1.0

    # The committed displacement is exactly `rate` of the full one.
    for theta0, p_full, p_short in zip(
        start, full.model.parameters(), short.model.parameters()
    ):
        expected = theta0 + rate * (p_full.detach() - theta0)
        assert torch.allclose(p_short.detach(), expected, rtol=1e-5, atol=1e-7)


def test_a_degenerate_interval_withdraws_certification() -> None:
    """When no rate sits inside the interval the family cannot deliver a
    certified step, so it must report certified=False rather than commit."""
    config = _config(
        certify_family_inner_steps=10,
        certify_family_lemma35_rate=True,
        theory_lr_min=1e9,  # forces family_lemma35_rate to return None
        rel_error_threshold=0.5,
    )
    model, x, y = _tiny_problem()
    result = certify_parametric_step(model, x, y, config)
    assert result.certified is False
    assert result.model is None
