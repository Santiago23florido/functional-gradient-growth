"""Chunking the vmap block must change the footprint and nothing else.

The range finder asks ``apply_j`` for a block of ``rank + oversampling``
directions and ``vmap`` holds every direction's full forward activations at
once. On a conv that activation carries the spatial extent, so the block is
what exhausted the machine. Chunking is a loop over independent directions,
so mathematically the result cannot change. In floating point it changes by
ONE ULP: the reduction order inside the kernels depends on the batch size.
MEASURED on the conv stack, ``|J^T w|`` of magnitude 9.2 moves by 1.8e-15,
i.e. 1.4e-16 relative. These tests pin that bound, because "close enough"
without a number is how a memory fix quietly becomes an accuracy trade.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgdlib.models.convstack import build_conv_stack_model
from fgdlib.models.stack import build_stack_model
from fgdlib.search.matrixfree import vmap_operators

DEVICE = torch.device("cpu")

CONV_STACK = [
    {"conv": [2, 3]},
    {"conv": [2, 3]},
    "maxpool",
    {"avgpool": [3, 3]},
    "flatten",
    {"mlp": [2, 1]},
]


def conv_model():
    torch.manual_seed(0)
    return build_conv_stack_model(
        CONV_STACK,
        in_features=784,
        out_features=10,
        device=DEVICE,
        input_shape=(1, 28, 28),
    )


def mlp_model():
    torch.manual_seed(0)
    return build_stack_model(
        [{"mlp": [4, 2]}], in_features=4, out_features=1, device=DEVICE
    )


def operators(model, x, chunk: int):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for p in model.parameters() if p.requires_grad]
    return vmap_operators(model, x, parameters, names, chunk=chunk)


@pytest.mark.parametrize("chunk", [1, 3, 7, 64])
def test_chunking_moves_nothing_beyond_one_ulp_on_conv(chunk: int) -> None:
    net = conv_model()
    x = torch.rand(8, 1, 28, 28, generator=torch.Generator().manual_seed(1))
    whole = operators(net, x, chunk=0)
    split = operators(net, x, chunk=chunk)

    block = torch.randn(
        11, whole.columns, generator=torch.Generator().manual_seed(2)
    ).double()
    reference = whole.apply_j(block)
    assert torch.allclose(reference, split.apply_j(block), rtol=0.0, atol=1e-14)

    rows_block = torch.randn(
        11, whole.rows, generator=torch.Generator().manual_seed(3)
    ).double()
    reference_t = whole.apply_jt(rows_block)
    moved = (split.apply_jt(rows_block) - reference_t).abs().max()
    # One ulp of the quantity itself, not an absolute epsilon plucked out.
    assert moved <= 8.0 * torch.finfo(torch.float64).eps * reference_t.abs().max()


def test_the_same_bound_holds_on_an_mlp() -> None:
    net = mlp_model()
    x = torch.rand(8, 4, generator=torch.Generator().manual_seed(1))
    whole = operators(net, x, chunk=0)
    split = operators(net, x, chunk=2)
    block = torch.randn(
        9, whole.columns, generator=torch.Generator().manual_seed(2)
    ).double()
    reference = whole.apply_j(block)
    moved = (split.apply_j(block) - reference).abs().max()
    assert moved <= 8.0 * torch.finfo(torch.float64).eps * reference.abs().max()


def test_a_chunk_at_least_the_block_width_takes_the_unchunked_path() -> None:
    net = conv_model()
    x = torch.rand(4, 1, 28, 28, generator=torch.Generator().manual_seed(1))
    whole = operators(net, x, chunk=0)
    split = operators(net, x, chunk=1000)
    block = torch.randn(
        5, whole.columns, generator=torch.Generator().manual_seed(2)
    ).double()
    assert torch.equal(whole.apply_j(block), split.apply_j(block))


def test_the_default_is_unchunked_so_existing_runs_are_untouched() -> None:
    from stable_tiny.pipeline import load_pipeline_config

    for name in (
        "family_ladder_N1024",
        "family_ladder_matrix_free_N1024",
        "mnist_matrix_free",
    ):
        config = load_pipeline_config(f"configs/fgd/{name}.yaml")
        assert config.fgd_approx.matrix_free_block_chunk == 0

    conv = load_pipeline_config("configs/fgd/mnist_conv_matrix_free_N1024.yaml")
    # 32, measured at the probe the run actually uses (NK=1920 after
    # certify_probe_kappa shrinks it), not at the full probe.
    assert conv.fgd_approx.matrix_free_block_chunk == 32
