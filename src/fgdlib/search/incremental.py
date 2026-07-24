"""The exact ``where`` eps as a low-rank update of a shared base factorisation.

The grow-to-certify loop chooses WHERE to grow by measuring the resulting
min-damping ``eps`` EXACTLY on every candidate, then committing the argmin
(``fgdlib/search/certify.py``). Measured on the headline run that recompute is
the dominant cost of a step: each candidate rebuilds the full Jacobian of a
grown clone and takes its SVD, ``O(NK * P^2)`` per candidate per growth.

Under **function-preserving** growth that work is almost entirely redundant.
A preserving growth adds a neuron with outgoing weight ``omega = 0``: ``f`` is
exactly unchanged, ``df/dalpha = 0``, and only ``df/domega`` contributes a
small block of NEW columns ``C``. Every existing parameter's column is
untouched, so

    J_grown = [ J_base | C ]      with J_base IDENTICAL across candidates.

So the base factorisation is computed ONCE per growth and only the thin ``C``
(a handful of columns) differs per location. This module turns the per-
candidate ``eps`` into a low-rank update of that shared factorisation, exact
to the same min-damping projection the full solver computes.

The algebra (min damping = orthogonal projection onto ``range(J)`` in the
limit the solver uses, ``lambda = 1e-16 sigma_max^2``). Let ``U`` be an
orthonormal basis of ``range(J_base)`` and ``g_base = P_base r`` the base
projection, with base residual ``rho = r - g_base`` (so ``rho _|_ range(U)``).
Adding ``C`` enlarges the tangent space by ``range(C_tilde)`` where
``C_tilde = (I - U U^T) C`` is the part of ``C`` orthogonal to the base range,
and the grown projection splits orthogonally,

    g_grown = g_base + P_{C_tilde} rho .

Because ``P_{C_tilde} rho _|_ g_base`` and ``P`` is an orthogonal projection,
the three scalars the relative-error sensor needs update ADDITIVELY by the
single quantity ``cap = ||P_{C_tilde} rho||^2``:

    <g_grown, r>   = <g_base, r>   + cap
    ||g_grown||^2  = ||g_base||^2  + cap
    ||r||^2        unchanged

and ``cap`` is computed from ``C`` alone, with ``rho _|_ range(U)`` giving
``C_tilde^T rho = C^T rho``:

    cap = (C^T rho)^T [ C^T C - (U^T C)^T (U^T C) ]^+ (C^T rho) .

Feeding ``(<g_grown,r>, ||g_grown||^2, ||r||^2)`` to the SAME sensor the full
solver uses makes the returned ``eps`` identical, and ``argmin_L eps`` reduces
to ``argmax_L cap`` -- the location whose new directions capture the most of
the base residual. Verified against the full recompute in
``tests/test_incremental_where.py`` (identical location, eps to ~1e-6).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fgdlib.tangent import _output_relative_error_from_stats

__all__ = [
    "BaseProjection",
    "factor_base_projection",
    "low_rank_relative_error",
]


@dataclass(frozen=True)
class BaseProjection:
    """The shared base factorisation, computed ONCE per growth.

    Holds everything a candidate needs to score its own ``eps`` from just its
    new columns ``C``: the orthonormal base range ``U``, the base residual
    ``rho`` it must capture, and the two base sensor scalars the update adds to.
    """

    #: Orthonormal basis of ``range(J_base)`` (the left singular vectors kept
    #: above the rank tolerance), shape ``(NK, r)``.
    basis: torch.Tensor
    #: Base residual ``rho = r - P_base r``, orthogonal to ``range(basis)``.
    residual: torch.Tensor
    #: ``<g_base, r>`` for the base projection (= ``||g_base||^2`` here).
    base_dot: float
    #: ``||g_base||^2`` for the base projection.
    base_approximation_sq: float
    #: ``||r||^2`` -- unchanged by any preserving growth.
    target_sq: float
    #: The sensor's numerical floor, threaded through unchanged.
    eps: float


def factor_base_projection(
    jacobian: torch.Tensor,
    target: torch.Tensor,
    eps: float,
    *,
    rank_tolerance: float = 1e-7,
    work_dtype: torch.dtype = torch.float32,
) -> BaseProjection | None:
    """Factor ``J_base`` once so every candidate scores from its ``C`` alone.

    Reproduces the min-damping projection of :func:`minimal_relative_error`
    (``lambda = 1e-16 sigma_max^2``, which the retained-rank orthogonal
    projection matches to full precision) and packages the pieces a low-rank
    update needs. ``work_dtype`` matches the fast-where float32 path; the
    ranking is an argmax separated by far more than float32's round-off.

    ``None`` when the base system is degenerate.
    """
    j = jacobian.to(work_dtype)
    r = target.reshape(-1).to(work_dtype)
    if j.numel() == 0 or r.numel() == 0:
        return None

    left, singular_values, _ = torch.linalg.svd(j, full_matrices=False)
    if singular_values.numel() == 0:
        return None
    sigma_max = float(singular_values.max())
    if not sigma_max > 0.0:
        return None

    # Keep the well-conditioned range: the min-damping filter
    # sigma^2/(sigma^2 + 1e-16 sigma_max^2) is ~1 above this and ~0 below, so
    # the retained-rank orthogonal projection equals the filtered one.
    keep = singular_values > rank_tolerance * sigma_max
    basis = left[:, keep]                       # (NK, r), orthonormal columns
    coefficients = basis.t() @ r                # U^T r
    projection = basis @ coefficients           # g_base = P_base r
    base_approx_sq = float(coefficients.square().sum())  # ||g_base||^2
    residual = r - projection                   # rho _|_ range(basis)

    return BaseProjection(
        basis=basis,
        residual=residual,
        base_dot=base_approx_sq,                 # <g_base, r> = ||g_base||^2
        base_approximation_sq=base_approx_sq,
        target_sq=float(r.square().sum()),
        eps=eps,
    )


def _captured_energy(base: BaseProjection, new_columns: torch.Tensor) -> float:
    """``cap = ||P_{C_tilde} rho||^2`` -- how much of ``rho`` the new columns catch.

    Uses ``C_tilde = (I - U U^T) C`` only through the small Gram system, so the
    per-candidate cost is ``O(NK * r * k)`` (the ``U^T C`` product) rather than
    a fresh SVD. ``rho _|_ range(U)`` makes ``C_tilde^T rho = C^T rho``.
    """
    c = new_columns.to(base.basis.dtype)
    if c.ndim != 2 or c.shape[1] == 0:
        return 0.0
    ctr = c.t() @ base.residual                  # C^T rho, shape (k,)
    utc = base.basis.t() @ c                     # U^T C, shape (r, k)
    gram = c.t() @ c - utc.t() @ utc             # C_tilde^T C_tilde, (k, k)
    # cap = ctr^T gram^+ ctr. Solve in the small k-space; pinv is robust to a
    # rank-deficient C_tilde (a new direction already inside range(U)).
    try:
        solution = torch.linalg.lstsq(gram, ctr.unsqueeze(-1)).solution
    except RuntimeError:
        solution = torch.linalg.pinv(gram) @ ctr.unsqueeze(-1)
    cap = float((ctr.unsqueeze(-1) * solution).sum())
    return max(0.0, cap)


def low_rank_relative_error(
    base: BaseProjection,
    new_columns: torch.Tensor,
) -> float:
    """The grown min-damping ``eps`` from the base factorisation and ``C``.

    Identical to recomputing :func:`minimal_relative_error` on the grown model,
    at ``O(NK * r * k)`` instead of a full Jacobian and SVD. The three sensor
    scalars update additively by ``cap`` and go through the SAME sensor, so the
    returned value -- and hence the ``argmin`` over candidates -- matches.
    """
    cap = _captured_energy(base, new_columns)
    stats = _output_relative_error_from_stats(
        dot_product=base.base_dot + cap,
        approximation_sq_norm=base.base_approximation_sq + cap,
        target_sq_norm=base.target_sq,
        eps=base.eps,
    )
    value = stats.relative_error
    if value is None or not float(value) == float(value):
        return float("inf")
    return float(value)
