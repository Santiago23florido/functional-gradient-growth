"""Chunking the exact Jacobian: same matrix, bounded memory.

jacrev vmaps one backward pass per output row and holds the batched
activations for all rows at once, so its peak scales with
(output rows) x (activations), not with the size of J. A 3840 x ~600
Jacobian is only ~9 MB and still ran the GPU out of memory in one call.
Chunking rebuilds the SAME matrix block by block, which is what makes
certifying over a whole dataset affordable instead of over a subsample.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fgdlib.tangent import exact_tangent_system
from stable_tiny.pipeline import build_model, load_pipeline_config


@pytest.fixture
def setup():
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(config.model, hidden_size=6, number_hidden_layers=2),
        fgd_approx=replace(config.fgd_approx, projection_solver="exact"),
    )
    model = build_model(config, torch.device("cpu"))
    torch.manual_seed(0)
    x = torch.randn(20, config.data.in_features)
    y = torch.randn(20, config.data.out_features)
    return model, x, y, config.fgd_approx


def test_chunked_jacobian_equals_the_single_pass_one(setup) -> None:
    """Exactness is the whole point: the matrix must be identical."""
    model, x, y, fa = setup
    whole = exact_tangent_system(model, x, y, fa)
    for chunk in (1, 7, 13, 60):
        chunked = exact_tangent_system(
            model, x, y, replace(fa, jacobian_row_chunk=chunk)
        )
        assert chunked.jacobian.shape == whole.jacobian.shape
        assert torch.allclose(chunked.jacobian, whole.jacobian, atol=1e-6)


def test_a_chunk_at_or_above_the_row_count_is_a_no_op(setup) -> None:
    """No silent change of path when chunking cannot help."""
    model, x, y, fa = setup
    whole = exact_tangent_system(model, x, y, fa)
    rows = whole.jacobian.shape[0]
    for chunk in (rows, rows + 1, 10_000):
        chunked = exact_tangent_system(
            model, x, y, replace(fa, jacobian_row_chunk=chunk)
        )
        assert torch.equal(chunked.jacobian, whole.jacobian)


def test_chunking_is_off_by_default(setup) -> None:
    """Existing configs keep the single-pass path bit-for-bit."""
    _, _, _, fa = setup
    assert fa.jacobian_row_chunk == 0


def test_the_target_is_unaffected_by_chunking(setup) -> None:
    """Only J is assembled in blocks; r is one backward pass either way."""
    model, x, y, fa = setup
    whole = exact_tangent_system(model, x, y, fa)
    chunked = exact_tangent_system(model, x, y, replace(fa, jacobian_row_chunk=5))
    assert torch.equal(chunked.target, whole.target)
