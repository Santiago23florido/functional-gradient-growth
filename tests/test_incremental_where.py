"""The low-rank ``where`` eps equals the full min-damping recompute.

Function-preserving growth gives ``J_grown = [J_base | C]`` with ``J_base``
identical across candidates, so :mod:`fgdlib.search.incremental` scores each
candidate's min-damping ``eps`` as a low-rank update of a shared base
factorisation instead of a fresh Jacobian + SVD. These tests pin that the
update is EXACT -- same eps as recomputing the full projection on the grown
system -- and, what the ``where`` actually needs, DECISION-PRESERVING: the
``argmin`` over candidate locations is unchanged.
"""

from __future__ import annotations

import torch

from fgdlib.search.damping import DAMPING_BRACKET
from fgdlib.search.incremental import (
    factor_base_projection,
    low_rank_relative_error,
)
from fgdlib.tangent import _output_relative_error_from_stats


def _full_min_relative_error(
    jacobian: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.float64,
) -> float:
    """``minimal_relative_error``'s min-damping eps, computed directly on ``J``."""
    j = jacobian.to(dtype)
    r = target.reshape(-1).to(dtype)
    left, singular_values, _ = torch.linalg.svd(j, full_matrices=False)
    scale = float(singular_values.max()) ** 2
    absolute = DAMPING_BRACKET[0] * scale
    coefficients = left.t() @ r
    denominator = singular_values.square() + absolute
    approximation = left @ (singular_values.square() / denominator * coefficients)
    return _output_relative_error_from_stats(
        dot_product=float((approximation * r).sum()),
        approximation_sq_norm=float((approximation**2).sum()),
        target_sq_norm=float((r**2).sum()),
        eps=eps,
    ).relative_error


def _rank_deficient_base(nk: int, rank: int, extra: int, seed: int):
    """A rank-``rank`` base Jacobian (``eps > 0``) and a residual target."""
    generator = torch.Generator().manual_seed(seed)
    left = torch.randn(nk, rank, generator=generator, dtype=torch.float64)
    right = torch.randn(rank, rank + extra, generator=generator, dtype=torch.float64)
    jacobian = left @ right                       # rank <= `rank` << nk
    target = torch.randn(nk, generator=generator, dtype=torch.float64)
    return jacobian, target


def test_low_rank_eps_matches_full_recompute():
    """Low-rank eps == full min-damping eps, on a rank-deficient base."""
    jacobian, target = _rank_deficient_base(nk=256, rank=90, extra=60, seed=1)
    base = factor_base_projection(
        jacobian, target, eps=1e-12, work_dtype=torch.float64
    )
    assert base is not None

    generator = torch.Generator().manual_seed(7)
    for _ in range(8):
        new_columns = torch.randn(256, 9, generator=generator, dtype=torch.float64)
        grown = torch.cat([jacobian, new_columns], dim=1)
        full = _full_min_relative_error(grown, target)
        low = low_rank_relative_error(base, new_columns)
        assert abs(full - low) < 1e-6, (full, low)


def test_low_rank_preserves_argmin_over_locations():
    """The chosen location (argmin eps) matches the full recompute exactly."""
    jacobian, target = _rank_deficient_base(nk=256, rank=90, extra=60, seed=1)
    base = factor_base_projection(
        jacobian, target, eps=1e-12, work_dtype=torch.float64
    )
    assert base is not None
    residual = base.residual.to(torch.float64)

    full_eps, low_eps = [], []
    for location in range(6):
        generator = torch.Generator().manual_seed(100 + location)
        new_columns = torch.randn(256, 9, generator=generator, dtype=torch.float64)
        # Give two locations extra alignment with the residual so the argmin is
        # a genuine, non-trivial choice rather than a near-tie.
        if location in (2, 4):
            new_columns[:, 0] = new_columns[:, 0] + (1.0 + location) * residual
        grown = torch.cat([jacobian, new_columns], dim=1)
        full_eps.append(_full_min_relative_error(grown, target))
        low_eps.append(low_rank_relative_error(base, new_columns))

    assert full_eps.index(min(full_eps)) == low_eps.index(min(low_eps))
    # Full ranking, not just the winner.
    assert sorted(range(6), key=lambda i: full_eps[i]) == sorted(
        range(6), key=lambda i: low_eps[i]
    )


def test_new_columns_inside_base_range_do_not_help():
    """Columns already in ``range(J_base)`` add no capture (cap = 0)."""
    jacobian, target = _rank_deficient_base(nk=256, rank=90, extra=60, seed=3)
    base = factor_base_projection(
        jacobian, target, eps=1e-12, work_dtype=torch.float64
    )
    assert base is not None

    base_eps = low_rank_relative_error(base, torch.zeros(256, 1, dtype=torch.float64))
    # Recombine existing columns: nothing new is added to the span.
    generator = torch.Generator().manual_seed(5)
    mixture = jacobian @ torch.randn(
        jacobian.shape[1], 4, generator=generator, dtype=torch.float64
    )
    redundant_eps = low_rank_relative_error(base, mixture)
    assert abs(base_eps - redundant_eps) < 1e-6
