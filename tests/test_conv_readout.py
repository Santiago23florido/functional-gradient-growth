"""The readout finder must pick the right layer on a conv stack.

``_readout_and_features`` identifies the output layer by EXECUTION order in a
real forward pass. On a conv stack that pass now runs convolutions, pooling
and a flatten first, so the property is worth pinning rather than assuming.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgdlib.models.convstack import build_conv_stack_model
from fgdlib.search.nonlinear import _readout_and_features, solve_readout_least_squares

DEVICE = torch.device("cpu")

BASE_STACK = [
    {"conv": [2, 3]},
    {"conv": [2, 3]},
    "maxpool",
    {"conv": [2, 3]},
    {"conv": [2, 3]},
    "maxpool",
    {"avgpool": [3, 3]},
    "flatten",
    {"mlp": [2, 1]},
]


def model():
    torch.manual_seed(0)
    return build_conv_stack_model(
        BASE_STACK,
        in_features=784,
        out_features=10,
        device=DEVICE,
        input_shape=(1, 28, 28),
    )


def test_the_readout_is_the_output_layer_and_its_input_is_flat() -> None:
    net = model()
    x = torch.rand(8, 1, 28, 28)
    layer, features = _readout_and_features(net, x)
    assert layer is net.layers[-1].layer
    assert features is not None
    # (batch, last hidden width) -- the flatten and the pools already ran.
    assert features.shape == (8, 2)


def test_the_closed_form_readout_solve_runs_on_a_conv_stack() -> None:
    net = model()
    x = torch.rand(16, 1, 28, 28)
    target = torch.rand(16, 10)
    with torch.no_grad():
        before = float(((net(x) - target) ** 2).sum())
    assert solve_readout_least_squares(model=net, probe_x=x, target=target)
    with torch.no_grad():
        after = float(((net(x) - target) ** 2).sum())
    assert after <= before


def test_a_model_with_no_linear_declines_instead_of_guessing() -> None:
    """A pure CNN has no readout to fit, and the solve says so by failing.

    Not raising: returning False is the caller's existing "this family could
    not act" path, and the conv configs run ``family_order:
    [matrix_free_tangent]`` where the nonlinear family never fires at all.
    Turning this into an exception would change a handled outcome into a
    crash on speculation.
    """
    from torch import nn

    pure_conv = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.Flatten())
    x = torch.rand(4, 1, 28, 28)
    assert _readout_and_features(pure_conv, x) == (None, None)
    assert not solve_readout_least_squares(
        model=pure_conv, probe_x=x, target=torch.rand(4, 1568)
    )
