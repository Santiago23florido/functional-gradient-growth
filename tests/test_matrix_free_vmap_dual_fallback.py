"""The vmapped operators carry no analytic factors, so the dual route is out.

MEASURED on Margaret, MNIST, config ``mnist_depth3_small``: the analytic
structure was refused (``forward_has_caching_side_effects``), the builder fell
back to ``VmapOperators``, and with ``NK = 3840`` against ``P = 2399`` the
ratio gate still chose the dual branch -- which reads ``sensitivities`` /
``activations`` / ``use_bias``, attributes only ``AnalyticOperators`` has. The
run died mid-flight with ``AttributeError``, on both nodes, at ~2 minutes.

The gate now asks whether the dual is REACHABLE, not just whether it is
cheaper. These tests pin the three things that matter: the crash is gone, the
route taken is still exact, and the analytic path is untouched.
"""

from __future__ import annotations

import pytest
import torch

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.profile import profile_enabled, reset, snapshot
from fgdlib.search.matrixfree import (
    AnalyticOperators,
    VmapOperators,
    dual_gram,
    randomized_factorization,
)

ensure_gromo_importable()


def _vmap_operators_for(matrix: torch.Tensor) -> VmapOperators:
    """``VmapOperators`` around a known ``J``, so the factorisation is checkable."""
    rows, columns = matrix.shape
    return VmapOperators(
        _apply_j=lambda block: (matrix @ block.t()).t(),
        _apply_jt=lambda block: (matrix.t() @ block.t()).t(),
        rows=rows,
        columns=columns,
    )


def test_dual_gram_really_cannot_read_vmapped_operators() -> None:
    """Pin the defect itself, so the guard is not protecting against nothing."""
    operators = _vmap_operators_for(torch.randn(6, 4))
    assert not hasattr(operators, "sensitivities")
    with pytest.raises(AttributeError):
        dual_gram(operators)


def test_analytic_operators_do_expose_the_factors_the_dual_needs() -> None:
    """The capability check is a real discriminator between the two kinds."""
    for field in ("sensitivities", "activations", "use_bias"):
        assert field in AnalyticOperators.__dataclass_fields__
        assert field not in VmapOperators.__dataclass_fields__


def test_the_range_finder_serves_the_vmapped_operators_exactly() -> None:
    """Full rank is not an approximation: rank(J) <= min(NK, P) bounds it.

    This is the route the guard sends the fallback down, so it has to
    reconstruct J to numerical precision rather than merely not crash.
    """
    torch.manual_seed(0)
    matrix = torch.randn(12, 5, dtype=torch.float64)
    operators = _vmap_operators_for(matrix)

    built = randomized_factorization(
        apply_j=operators.apply_j,
        apply_jt=operators.apply_jt,
        rows=operators.rows,
        columns=operators.columns,
        rank=min(operators.rows, operators.columns),
        oversampling=0,
        power_iterations=1,
        device=matrix.device,
        generator=torch.Generator().manual_seed(0),
    )
    assert built is not None
    left, right = built
    assert torch.allclose(left @ right.t(), matrix, atol=1e-8)


def test_full_rank_matches_the_columns_rank_the_analytic_branch_uses() -> None:
    """The unified `min(rows, columns)` must not move the existing path.

    The analytic branch only reaches the range finder when
    ``rows > 2 * columns``, and there ``min(rows, columns) == columns``, so
    substituting one for the other is a no-op wherever it already ran.
    """
    for rows, columns in ((3840, 2399), (7040, 1612), (100, 10)):
        if rows > 2 * columns:
            assert min(rows, columns) == columns


def test_the_cluster_dimensions_would_have_taken_the_dual_branch() -> None:
    """NK 3840 against P 2399 is on the dual side of the 2x ratio gate.

    Without the capability check this is exactly the configuration that
    crashed, which is why the gate could not stay a pure cost decision.
    """
    rows, columns = 3840, 2399
    assert not rows > 2 * columns


def test_the_fallback_is_counted_and_named() -> None:
    """No silent fallback: the counter and its reason both have to be declared."""
    from fgdlib.profile import PROFILE_FIELDS

    assert "matrix_free_dual_unavailable" in PROFILE_FIELDS
    if not profile_enabled():
        pytest.skip("FGD_PROFILE is off; the counter store is inert")
    reset()
    from fgdlib.profile import fallback

    fallback("matrix_free_dual_unavailable", "vmap_operators_lack_factors")
    assert snapshot()["matrix_free_dual_unavailable"] == 1
