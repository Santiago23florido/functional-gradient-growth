"""Validation of SENN's theory against the certificates already in use.

Reference: Mitchell et al., "Self-Expanding Neural Networks",
arXiv:2307.04526v3. These tests pin the claims `report/SENN_EXPANSION_SCORE.md`
makes, so that adopting SENN's *where* cannot silently weaken Lemma 3.5.
"""

from __future__ import annotations

import pytest
import torch

from fgdlib.senn import (
    admissible_expansion_score,
    expansion_score_from_relative_error,
    expansion_score_increase_lower_bound,
    natural_expansion_score,
    relative_error_from_expansion_score,
    residual_gradient,
)


def _tangent_projection(jacobian: torch.Tensor, target: torch.Tensor):
    """Our shared-direction probe: P(r) = J argmin ||J u - r||^2."""
    solution, *_ = torch.linalg.lstsq(jacobian, target)
    projection = jacobian @ solution
    relative_error = float((projection - target).norm() / projection.norm())
    return projection, relative_error


def test_expansion_score_is_the_squared_tangent_projection() -> None:
    """The bridge: N*eta = ||P(r)||^2 = ||r||^2 / (1 + eps^2).

    This is the identity the whole integration rests on -- SENN's score and
    Lemma 3.5's relative error are the same projection in two coordinates.
    """
    torch.manual_seed(0)
    for _ in range(20):
        n_outputs = int(torch.randint(12, 60, (1,)))
        n_parameters = int(torch.randint(2, 10, (1,)))
        jacobian = torch.randn(n_outputs, n_parameters, dtype=torch.float64)
        target = torch.randn(n_outputs, dtype=torch.float64)

        projection, relative_error = _tangent_projection(jacobian, target)
        measured = float(projection.pow(2).sum())
        predicted = expansion_score_from_relative_error(
            relative_error=relative_error,
            gradient_sq_norm=float(target.pow(2).sum()),
        )
        assert measured == pytest.approx(predicted, rel=1e-9)


def test_the_bridge_inverts() -> None:
    torch.manual_seed(1)
    jacobian = torch.randn(40, 6, dtype=torch.float64)
    target = torch.randn(40, dtype=torch.float64)
    projection, relative_error = _tangent_projection(jacobian, target)
    recovered = relative_error_from_expansion_score(
        expansion_score=float(projection.pow(2).sum()),
        gradient_sq_norm=float(target.pow(2).sum()),
    )
    assert recovered == pytest.approx(relative_error, rel=1e-9)


def test_lemma35_admissibility_is_eighty_percent_of_gradient_energy() -> None:
    """eps < 1/2  <=>  N*eta > 0.8 ||r||^2. Same condition, two coordinates."""
    gradient_sq_norm = 7.5
    threshold = admissible_expansion_score(
        gradient_sq_norm=gradient_sq_norm, rel_error_threshold=0.5
    )
    assert threshold == pytest.approx(0.8 * gradient_sq_norm)

    # Below the relative-error threshold => above the score threshold.
    for relative_error in (0.0, 0.1, 0.3, 0.4999):
        assert (
            expansion_score_from_relative_error(
                relative_error=relative_error,
                gradient_sq_norm=gradient_sq_norm,
            )
            > threshold
        )
    # And above it => below.
    for relative_error in (0.5001, 0.8, 2.0):
        assert (
            expansion_score_from_relative_error(
                relative_error=relative_error,
                gradient_sq_norm=gradient_sq_norm,
            )
            < threshold
        )


def test_no_projection_means_infinite_relative_error() -> None:
    assert relative_error_from_expansion_score(
        expansion_score=0.0, gradient_sq_norm=1.0
    ) == float("inf")


def test_residual_gradient_is_orthogonal_to_current_activations() -> None:
    """Lemma A.6's g_r must carry nothing the layer already predicts."""
    torch.manual_seed(2)
    activations = torch.randn(200, 5, dtype=torch.float64)
    gradients = torch.randn(200, 3, dtype=torch.float64)
    residual = residual_gradient(
        activations=activations,
        output_gradients=gradients,
        damping=0.0,
    )
    # E[g_r a_c^T] = 0 by construction.
    cross = (residual.transpose(-2, -1) @ activations) / activations.shape[0]
    assert torch.allclose(cross, torch.zeros_like(cross), atol=1e-9)


def test_theorem_32_is_a_genuine_lower_bound_on_delta_eta() -> None:
    """Delta eta' <= Delta eta, checked against the exact KFAC score.

    The exact increase is recomputed by scoring the layer with and without
    the proposed neurons concatenated; Theorem A.8 says the cheap trace can
    never exceed it.
    """
    torch.manual_seed(3)
    damping = 1e-8
    for _ in range(25):
        n_samples = 300
        n_current = int(torch.randint(2, 7, (1,)))
        n_proposed = int(torch.randint(1, 4, (1,)))
        n_out = int(torch.randint(2, 6, (1,)))

        current = torch.randn(n_samples, n_current, dtype=torch.float64)
        # Correlate the proposals with the current neurons: the interesting
        # regime is partial redundancy, which is what A_p^-1 corrects for.
        mixing = torch.randn(n_current, n_proposed, dtype=torch.float64)
        proposed = current @ mixing + 0.5 * torch.randn(
            n_samples, n_proposed, dtype=torch.float64
        )
        gradients = torch.randn(n_samples, n_out, dtype=torch.float64)

        before = natural_expansion_score(
            activations=current,
            output_gradients=gradients,
            damping=damping,
        )
        after = natural_expansion_score(
            activations=torch.cat([current, proposed], dim=1),
            output_gradients=gradients,
            damping=damping,
        )
        exact_increase = after - before
        bound = expansion_score_increase_lower_bound(
            current_activations=current,
            proposed_activations=proposed,
            output_gradients=gradients,
            damping=damping,
        )
        assert bound <= exact_increase + 1e-6
        assert bound >= -1e-9          # Lemma A.1: the score is non-negative


def test_a_redundant_proposal_scores_near_zero() -> None:
    """A neuron that duplicates an existing one buys no expansion score."""
    torch.manual_seed(4)
    current = torch.randn(400, 4, dtype=torch.float64)
    gradients = torch.randn(400, 3, dtype=torch.float64)
    # Exact duplicate of an existing activation: nothing new is expressible.
    redundant = current[:, :1].clone()
    bound = expansion_score_increase_lower_bound(
        current_activations=current,
        proposed_activations=redundant,
        output_gradients=gradients,
        damping=1e-8,
    )
    assert bound == pytest.approx(0.0, abs=1e-6)


def test_an_informative_proposal_scores_above_a_redundant_one() -> None:
    """The criterion must rank a genuinely new direction first."""
    torch.manual_seed(5)
    n = 500
    current = torch.randn(n, 3, dtype=torch.float64)
    fresh = torch.randn(n, 1, dtype=torch.float64)
    # Build gradients that genuinely depend on the fresh direction.
    gradients = fresh @ torch.randn(1, 2, dtype=torch.float64) + 0.1 * torch.randn(
        n, 2, dtype=torch.float64
    )
    informative = expansion_score_increase_lower_bound(
        current_activations=current,
        proposed_activations=fresh,
        output_gradients=gradients,
        damping=1e-8,
    )
    redundant = expansion_score_increase_lower_bound(
        current_activations=current,
        proposed_activations=current[:, :1].clone(),
        output_gradients=gradients,
        damping=1e-8,
    )
    assert informative > redundant
