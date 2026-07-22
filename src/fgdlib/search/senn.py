"""SENN's natural expansion score, as a second answer to *where* to grow.

Reference: Mitchell, Menzenbach, Kersting, Mundt, "Self-Expanding Neural
Networks", arXiv:2307.04526v3.

Why this belongs in a certified FGD flow
----------------------------------------
SENN scores a proposed expansion by the *natural expansion score* (their
eq. 3),

    eta = g^T F^{-1} g = (1/N) ||P_Theta(g_y)||^2 ,

where ``g_y`` is the gradient with respect to the concatenated network
outputs, ``J`` the parameters-to-outputs Jacobian, ``F = (1/N) J^T J`` the
Fisher matrix under the Euclidean output metric (the choice SENN states in
section 3.4), and ``P_Theta`` the orthogonal projection onto ``range(J)``.

That projection is *the same object* this codebase already certifies.
``g_y`` is our functional gradient ``r``, ``range(J)`` is our tangent space,
and our shared-direction probe solves exactly ``P(r) = J u*`` with
``u* = argmin ||J u - r||^2``. Lemma 3.5's relative error is
``eps = ||P(r) - r|| / ||P(r)||``, so orthogonality of the projection gives
``||r||^2 = ||P(r)||^2 + ||r - P(r)||^2`` and hence the exact bridge

    N * eta = ||P(r)||^2 = ||r||^2 / (1 + eps^2) .                     (*)

Two consequences make SENN usable here without weakening any certificate:

1. Lemma 3.5's admissibility ``eps < 1/2`` is *identical* to
   ``N * eta > 0.8 ||r||^2``. The two frameworks state the same condition in
   different coordinates, so adopting SENN's score introduces no new
   assumption.
2. Maximising ``Delta eta`` is, at fixed ``||r||^2``, exactly minimising
   ``eps``. SENN's *where* is therefore the epsilon-reduction criterion --
   but obtained in closed form from KFAC statistics instead of by building
   and training a probe model per layer.

What is deliberately NOT adopted: SENN answers *when* to grow with two tuned
thresholds (relative ``tau`` and absolute ``alpha``, their Ingredient 4).
This flow keeps its own loss-agnostic answer -- Lemma 3.5 proposes and R1
governs postponement -- because a tuned threshold does not transfer to a
dataset that has never been trained on. Only *where* is taken from SENN.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "expansion_score_from_relative_error",
    "relative_error_from_expansion_score",
    "admissible_expansion_score",
    "kfac_factors",
    "residual_gradient",
    "natural_expansion_score",
    "expansion_score_increase_lower_bound",
]


def _inverse(matrix: torch.Tensor, damping: float) -> torch.Tensor:
    """Damped inverse. Damping is Tikhonov, as SENN uses for its factors."""
    eye = torch.eye(
        matrix.shape[-1], dtype=matrix.dtype, device=matrix.device
    )
    return torch.linalg.solve(matrix + damping * eye, eye)


# --------------------------------------------------------------------------
# The bridge between SENN's score and Lemma 3.5's relative error.
# --------------------------------------------------------------------------


def expansion_score_from_relative_error(
    *, relative_error: float, gradient_sq_norm: float
) -> float:
    """Return ``N * eta = ||r||^2 / (1 + eps^2)`` -- identity (*) above.

    ``gradient_sq_norm`` is ``||r||^2``. The result is the *unnormalised*
    score ``||P(r)||^2``; divide by the sample count for SENN's ``eta``.
    """
    return gradient_sq_norm / (1.0 + relative_error * relative_error)


def relative_error_from_expansion_score(
    *, expansion_score: float, gradient_sq_norm: float
) -> float:
    """Invert (*): ``eps = sqrt(||r||^2 / ||P(r)||^2 - 1)``.

    Returns ``inf`` for a degenerate (zero) projection, which is the correct
    reading: nothing of ``r`` is representable.
    """
    if expansion_score <= 0.0:
        return float("inf")
    ratio = gradient_sq_norm / expansion_score
    # Orthogonality forces ratio >= 1; clamp away float error only. Kept in
    # Python floats: routing through a default-dtype tensor would silently
    # truncate a float64 certificate to float32.
    return math.sqrt(max(ratio - 1.0, 0.0))


def admissible_expansion_score(
    *, gradient_sq_norm: float, rel_error_threshold: float = 0.5
) -> float:
    """The score Lemma 3.5 demands: ``||r||^2 / (1 + threshold^2)``.

    At the default threshold this is ``0.8 ||r||^2``: admissibility and "the
    expansion score covers 80 % of the gradient energy" are the same
    statement.
    """
    return gradient_sq_norm / (1.0 + rel_error_threshold * rel_error_threshold)


# --------------------------------------------------------------------------
# KFAC statistics (SENN section 3.4).
# --------------------------------------------------------------------------


def kfac_factors(
    activations: torch.Tensor, output_gradients: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Kronecker factors ``(A, S)`` of ``F ~= S (x) A``.

    ``A = E[a a^T]`` is the input activation second moment and
    ``S = E[g g^T]`` the pre-activation gradient second moment, with the
    expectation over the sample rows.

    Parameters
    ----------
    activations
        ``(n_samples, n_in)`` inputs to the layer.
    output_gradients
        ``(n_samples, n_out)`` gradients w.r.t. the layer pre-activations.
    """
    n = activations.shape[0]
    a_factor = (activations.transpose(-2, -1) @ activations) / n
    s_factor = (output_gradients.transpose(-2, -1) @ output_gradients) / n
    return a_factor, s_factor


def residual_gradient(
    *,
    activations: torch.Tensor,
    output_gradients: torch.Tensor,
    damping: float = 1e-6,
) -> torch.Tensor:
    """SENN Lemma A.6: ``g_r = g - E[g a_c^T] A_c^{-1} a_c``.

    The part of the output gradient *not* predicted by the activations the
    layer already has. Everything a new neuron can contribute lives here,
    which is why Theorem 3.2 can be written in terms of it.
    """
    n = activations.shape[0]
    a_factor = (activations.transpose(-2, -1) @ activations) / n
    cross = (output_gradients.transpose(-2, -1) @ activations) / n
    a_inverse = _inverse(a_factor, damping)
    return output_gradients - activations @ a_inverse @ cross.transpose(-2, -1)


# --------------------------------------------------------------------------
# The score itself, and Theorem 3.2's cheap lower bound on its increase.
# --------------------------------------------------------------------------


def natural_expansion_score(
    *,
    activations: torch.Tensor,
    output_gradients: torch.Tensor,
    damping: float = 1e-6,
) -> float:
    """A layer's contribution to ``eta``: ``Tr[S^-1 dW A^-1 dW^T]``.

    With ``dW = E[g a^T]`` the weight gradient (SENN section 3.4). This is
    the exact KFAC expression, not the lower bound.
    """
    n = activations.shape[0]
    a_factor, s_factor = kfac_factors(activations, output_gradients)
    weight_gradient = (output_gradients.transpose(-2, -1) @ activations) / n
    a_inverse = _inverse(a_factor, damping)
    s_inverse = _inverse(s_factor, damping)
    product = (
        s_inverse
        @ weight_gradient
        @ a_inverse
        @ weight_gradient.transpose(-2, -1)
    )
    return float(torch.diagonal(product, dim1=-2, dim2=-1).sum())


def expansion_score_increase_lower_bound(
    *,
    current_activations: torch.Tensor,
    proposed_activations: torch.Tensor,
    output_gradients: torch.Tensor,
    damping: float = 1e-6,
) -> float:
    """SENN Theorem 3.2 / A.8: a cheap lower bound on ``Delta eta``.

        Delta eta >= Delta eta' = Tr[S^-1 E[g_r a_p^T] A_p^-1 E[a_p g_r^T]]

    The bound is what makes the criterion affordable: ``g_r`` and ``S^-1``
    depend only on the *existing* layer, so they are computed once and then
    reused to score many candidate expansions ``a_p`` -- no model clone and
    no probe training per candidate, which is what the descent- and
    epsilon-based selectors currently pay for.

    Parameters
    ----------
    current_activations
        ``(n_samples, n_current)`` activations of the neurons already there.
    proposed_activations
        ``(n_samples, n_proposed)`` activations the candidate neurons would
        produce on the same samples.
    output_gradients
        ``(n_samples, n_out)`` gradients w.r.t. the layer pre-activations.
    """
    n = current_activations.shape[0]
    _, s_factor = kfac_factors(current_activations, output_gradients)
    residual = residual_gradient(
        activations=current_activations,
        output_gradients=output_gradients,
        damping=damping,
    )
    # E[g_r a_p^T], shape (n_out, n_proposed).
    cross = (residual.transpose(-2, -1) @ proposed_activations) / n
    proposed_moment = (
        proposed_activations.transpose(-2, -1) @ proposed_activations
    ) / n
    s_inverse = _inverse(s_factor, damping)
    proposed_inverse = _inverse(proposed_moment, damping)
    product = (
        s_inverse @ cross @ proposed_inverse @ cross.transpose(-2, -1)
    )
    return float(torch.diagonal(product, dim1=-2, dim2=-1).sum())
