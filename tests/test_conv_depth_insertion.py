"""Inserting a conv layer must change nothing, and unlock something.

Two properties, and the second is why the depth axis exists at all:

* the represented function is unchanged EXACTLY (not to a tolerance -- a
  centred delta kernel with same-padding is the identity in float32);
* the insertion makes the predecessor's width growable, which is the only
  way a width pinned by GroMo's topology can ever be relieved.
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
from fgdlib.search.depth import (
    IdentityHomotopyActivation,
    insert_identity_layer,
    inserted_layer_cost,
)
from fgdlib.search.growth import grow_layer, growable_neuron_costs
from fgdlib.tangent import ScalingLineSearchConfig
from gromo.modules.conv2d_growing_module import Conv2dGrowingModule
from gromo.utils.training_utils import compute_statistics

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


def model():
    torch.manual_seed(0)
    return build_conv_stack_model(
        BASE_STACK,
        in_features=784,
        out_features=10,
        device=DEVICE,
        input_shape=(1, 28, 28),
    )


def images(n: int = 256):
    return torch.rand(n, 1, 28, 28, generator=torch.Generator().manual_seed(1))


def loader(n: int = 128, batch: int = 64):
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(n, 1, 28, 28, generator=generator)
    y = torch.zeros(n, 10)
    y[torch.arange(n), torch.randint(0, 10, (n,), generator=generator)] = 1.0
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=batch
    )


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def growable_indices(net) -> list[int]:
    return [
        index
        for index, layer in enumerate(net.layers)
        if any(layer is growing for growing in net._growable_layers)
    ]


@pytest.mark.parametrize("position", [1, 2, 3, 4, 5])
def test_insertion_is_exactly_function_preserving(position: int) -> None:
    """Exactly 0.0, not a tolerance: the identity is exact in float32."""
    net = model()
    net.eval()
    x = images()
    with torch.no_grad():
        before = net(x).clone()

    trial = copy.deepcopy(net)
    insert_identity_layer(trial, position=position, device=DEVICE)
    trial.eval()
    with torch.no_grad():
        after = trial(x)
    assert float((after - before).abs().max()) == 0.0


def test_the_inserted_layer_dispatches_to_the_container() -> None:
    trial = model()
    inserted = insert_identity_layer(trial, position=2, device=DEVICE)
    assert isinstance(inserted, Conv2dGrowingModule)
    assert isinstance(inserted.post_layer_function, IdentityHomotopyActivation)
    assert float(inserted.post_layer_function.alpha.detach()) == 0.0
    assert len(trial.layers) == 6 + 1


def test_insertion_unlocks_a_width_that_was_pinned() -> None:
    """The property the whole depth axis exists for.

    conv B's consumer sits behind a pool, so B's output width can never be
    bought: growable is [1, 3, 5] and none of those widens B. Inserting a
    conv immediately after B -- before the pool -- creates a layer whose
    predecessor IS B on the same grid, so B's width becomes purchasable.
    """
    net = model()
    assert growable_indices(net) == [1, 3, 5]

    insert_identity_layer(net, position=2, device=DEVICE)
    assert growable_indices(net) == [1, 2, 4, 6]

    # And buying at the new location really does widen B's output.
    before = int(net.layers[1].out_channels)
    grow_layer(
        model=net,
        train_loader=loader(),
        layer_index=1,  # the inserted layer, in _growable_layers order
        device=DEVICE,
        line_search_config=ScalingLineSearchConfig(),
        optimal_update_kwargs=UPDATE_KWARGS,
        function_preserving=True,
        preservation_tolerance=1e-5,
    )
    assert int(net.layers[1].out_channels) > before


def test_the_inserted_layer_is_priced_as_it_costs() -> None:
    net = model()
    before = count_parameters(net)
    insert_identity_layer(net, position=2, device=DEVICE)
    # 2*2*9 weight + 2 bias + 1 homotopy alpha
    assert count_parameters(net) - before == inserted_layer_cost(2, 9) == 39


def test_a_linear_position_inserts_a_linear_identity() -> None:
    from gromo.modules.linear_growing_module import LinearGrowingModule

    net = model()
    inserted = insert_identity_layer(net, position=5, device=DEVICE)
    assert isinstance(inserted, LinearGrowingModule)
    assert count_parameters(net) - 202 == inserted_layer_cost(2) == 7


@pytest.mark.parametrize(
    "kernel,stride,padding",
    [(3, 2, 1), (3, 1, 0), (4, 1, 2)],
)
def test_a_geometry_that_cannot_be_the_identity_is_refused(
    kernel: int, stride: int, padding: int
) -> None:
    """The caller in the growth loop already treats ValueError as 'skip'."""
    from gromo.modules.conv2d_growing_module import RestrictedConv2dGrowingModule

    net = model()
    net.layers[1] = RestrictedConv2dGrowingModule(
        in_channels=2,
        out_channels=2,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        use_bias=True,
        post_layer_function=nn.SELU(),
        device=DEVICE,
    )
    with pytest.raises(ValueError):
        insert_identity_layer(net, position=2, device=DEVICE)


def test_the_position_must_be_inside_the_chain() -> None:
    net = model()
    with pytest.raises(ValueError, match="position"):
        insert_identity_layer(net, position=0, device=DEVICE)
    with pytest.raises(ValueError, match="position"):
        insert_identity_layer(net, position=len(net.layers), device=DEVICE)


def test_a_one_by_one_insertion_is_also_exact_and_cheaper() -> None:
    net = model()
    net.eval()
    x = images()
    with torch.no_grad():
        before = net(x).clone()
    net.insert_identity_at(2, device=DEVICE, kernel_size=1)
    net.eval()
    with torch.no_grad():
        assert float((net(x) - before).abs().max()) == 0.0
    assert count_parameters(net) - 202 == inserted_layer_cost(2, 1) == 7


def test_the_mlp_container_path_is_untouched() -> None:
    """GrowingMLP has no insert_identity_at, so it keeps the flat rebuild."""
    from fgdlib.models.stack import build_stack_model

    torch.manual_seed(0)
    mlp = build_stack_model(
        [{"mlp": [3, 2]}], in_features=4, out_features=1, device=DEVICE
    )
    assert not hasattr(mlp, "insert_identity_at")
    x = torch.rand(8, 4)
    mlp.eval()
    with torch.no_grad():
        before = mlp(x).clone()
    insert_identity_layer(mlp, position=1, device=DEVICE)
    mlp.eval()
    with torch.no_grad():
        assert float((mlp(x) - before).abs().max()) == 0.0


def test_statistics_still_run_after_an_insertion() -> None:
    """The links were rebuilt, so GroMo's graph is the real one."""
    net = model()
    insert_identity_layer(net, position=2, device=DEVICE)
    net.set_growing_layers(index=1)
    compute_statistics(
        net, loader(), loss_function=nn.MSELoss(reduction="sum"), device=DEVICE
    )
    net.compute_optimal_updates(**UPDATE_KWARGS)
    # The inserted layer's own predecessor is conv B, whose unfolded
    # fan-in is 2*9=18, so buying there costs 18+1+2*9 = 37.
    assert growable_neuron_costs(net, 784) == [28, 37, 37, 29]
    net.reset_computation()
