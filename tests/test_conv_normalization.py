"""Batch-norm on a conv layer must keep up with growth, exactly.

In its own file rather than appended to ``test_regularized_mlp.py``: that
file guards the MLP arrangement, and a conv regression should not be able to
hide among its cases.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgdlib.models.convstack import build_conv_stack_model
from fgdlib.models.regularized_mlp import sync_normalization
from fgdlib.search.growth import grow_layer
from fgdlib.tangent import ScalingLineSearchConfig
from gromo.modules.growing_normalisation import GrowingBatchNorm1d, GrowingBatchNorm2d

DEVICE = torch.device("cpu")

NORMED_STACK = [
    {"conv": [2, 3]},
    {"conv": [2, 3]},
    "batchnorm",
    "maxpool",
    {"conv": [2, 3]},
    {"conv": [2, 3]},
    "batchnorm",
    {"avgpool": [3, 3]},
    "flatten",
    {"mlp": [2, 1]},
    "batchnorm",
]

UPDATE_KWARGS = {
    "compute_delta": False,
    "use_covariance": True,
    "alpha_zero": False,
    "omega_zero": True,
    "use_projection": True,
    "ignore_singular_values": False,
    "use_fisher": False,
    "maximum_added_neurons": 1,
    "numerical_threshold": 1e-6,
    "statistical_threshold": 1e-3,
}


def model() -> object:
    torch.manual_seed(0)
    return build_conv_stack_model(
        NORMED_STACK,
        in_features=784,
        out_features=10,
        device=DEVICE,
        input_shape=(1, 28, 28),
    )


def loader(n: int = 128, batch: int = 64):
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(n, 1, 28, 28, generator=generator)
    y = torch.zeros(n, 10)
    y[torch.arange(n), torch.randint(0, 10, (n,), generator=generator)] = 1.0
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=batch
    )


def test_the_stack_pairs_each_norm_with_its_own_rank() -> None:
    net = model()
    assert isinstance(net.layers[1].post_layer_function[0], GrowingBatchNorm2d)
    assert isinstance(net.layers[3].post_layer_function[0], GrowingBatchNorm2d)
    assert isinstance(net.layers[4].post_layer_function[0], GrowingBatchNorm1d)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_growth_syncs_the_norm_and_preserves_the_function(index: int) -> None:
    net = model()
    net.eval()
    x = next(iter(loader()))[0]
    with torch.no_grad():
        before = net(x).clone()

    trial = copy.deepcopy(net)
    grow_layer(
        model=trial,
        train_loader=loader(),
        layer_index=index,
        device=DEVICE,
        line_search_config=ScalingLineSearchConfig(),
        optimal_update_kwargs=UPDATE_KWARGS,
        function_preserving=True,
        preservation_tolerance=1e-5,
    )
    trial.eval()

    for layer in trial.layers:
        post = getattr(layer, "post_layer_function", None)
        if isinstance(post, nn.Sequential) and post and hasattr(post[0], "num_features"):
            assert int(post[0].num_features) == int(layer.out_features)

    with torch.no_grad():
        drift = float((trial(x) - before).abs().max())
    assert drift < 1e-5


def test_a_mismatched_norm_rank_is_refused_loudly() -> None:
    net = model()
    # A 1-D norm hung on a conv layer grows to the right width and then fails
    # far from here; sync_normalization must say so at the point of the error.
    net.layers[1].post_layer_function = nn.Sequential(
        GrowingBatchNorm1d(2, device=DEVICE), nn.SELU()
    )
    with pytest.raises(TypeError, match="GrowingBatchNorm2d"):
        sync_normalization(net)


def test_sync_is_still_a_no_op_on_a_plain_mlp() -> None:
    from fgdlib.models.stack import build_stack_model

    torch.manual_seed(0)
    mlp = build_stack_model(
        [{"mlp": [3, 2]}], in_features=4, out_features=1, device=DEVICE
    )
    before = [p.clone() for p in mlp.parameters()]
    sync_normalization(mlp)
    assert all(
        torch.equal(a, b) for a, b in zip(before, mlp.parameters())
    )
