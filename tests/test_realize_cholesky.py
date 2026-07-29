"""Frozen realization solves reuse one accurate float64 Cholesky factor."""

from __future__ import annotations

import pytest
import torch

from fgdlib.profile import reset, snapshot
from fgdlib.search.realize import (
    _factorize_frozen_gram,
    _solve_frozen_gram,
)


@pytest.mark.parametrize(
    ("jacobian", "damping"),
    [
        (torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)), 0.25),
        (
            torch.diag(torch.tensor([1.0, 1e-7, 0.0], dtype=torch.float64)),
            1e-12,
        ),
    ],
)
def test_cholesky_matches_direct_solve_for_several_rhs(
    jacobian: torch.Tensor,
    damping: float,
) -> None:
    factorization = _factorize_frozen_gram(jacobian, damping)
    assert factorization.matrix.dtype == torch.float64
    assert factorization.cholesky is not None
    right_hand_sides = torch.tensor(
        [
            [1.0, -2.0, 0.5],
            [0.25, 3.0, -1.0],
            [-4.0, 0.75, 2.0],
        ],
        dtype=torch.float64,
    )

    actual = _solve_frozen_gram(factorization, right_hand_sides)
    expected = torch.linalg.solve(factorization.matrix, right_hand_sides)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    for rhs in right_hand_sides.t():
        actual_vector = _solve_frozen_gram(factorization, rhs)
        expected_vector = torch.linalg.solve(factorization.matrix, rhs)
        assert torch.allclose(
            actual_vector,
            expected_vector,
            atol=1e-10,
            rtol=1e-10,
        )


def test_failed_cholesky_uses_the_previous_direct_solve(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()

    def fail_cholesky(matrix, *, check_errors):
        del check_errors
        return torch.zeros_like(matrix), torch.ones(
            (), dtype=torch.int32, device=matrix.device
        )

    monkeypatch.setattr(torch.linalg, "cholesky_ex", fail_cholesky)
    jacobian = torch.eye(3, dtype=torch.float64)
    factorization = _factorize_frozen_gram(jacobian, 0.1)
    assert factorization.cholesky is None
    rhs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    actual = _solve_frozen_gram(factorization, rhs)
    expected = torch.linalg.solve(factorization.matrix, rhs)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert snapshot()["realize_cholesky_fallbacks"] == 1
