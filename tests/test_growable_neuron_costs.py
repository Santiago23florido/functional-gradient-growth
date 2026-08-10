"""The neuron cost model: unchanged on MLPs, and MEASURED on conv.

Cost is the denominator of the growth ranking and the quantity
``max_total_parameters`` enforces, so getting it wrong does not raise -- it
quietly buys the wrong thing. These tests therefore (a) prove the rewrite is
the old formula on every MLP, and (b) check the conv figures against what
growing actually costs, rather than against the derivation that produced
them.
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
from fgdlib.models.stack import build_stack_model
from fgdlib.search.depth import inserted_layer_cost
from fgdlib.search.growth import growable_neuron_costs
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


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def conv_model():
    torch.manual_seed(0)
    return build_conv_stack_model(
        BASE_STACK,
        in_features=784,
        out_features=10,
        device=DEVICE,
        input_shape=(1, 28, 28),
    )


def image_loader(n: int = 128, batch: int = 64):
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(n, 1, 28, 28, generator=generator)
    y = torch.zeros(n, 10)
    y[torch.arange(n), torch.randint(0, 10, (n,), generator=generator)] = 1.0
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=batch
    )


@pytest.mark.parametrize(
    "stack,in_features",
    [
        ([{"mlp": [3, 3]}], 4),
        ([{"mlp": [5, 1]}, {"mlp": [2, 1]}, {"mlp": [9, 1]}], 7),
        ([{"mlp": [2, 4]}], 11),
        ([{"mlp": [17, 1]}, {"mlp": [3, 2]}], 784),
    ],
)
def test_the_rewrite_is_the_old_formula_on_every_mlp(stack, in_features) -> None:
    """Byte-identity proof against the literal previous implementation."""
    torch.manual_seed(0)
    model = build_stack_model(
        stack, in_features=in_features, out_features=1, device=DEVICE
    )
    growable = list(model._growable_layers)
    legacy = [
        (in_features if index == 0 else growable[index - 1].in_features)
        + 1
        + int(layer.out_features)
        for index, layer in enumerate(growable)
    ]
    assert growable_neuron_costs(model, in_features) == legacy


def test_the_conv_costs_are_the_derived_ones() -> None:
    assert growable_neuron_costs(conv_model(), 784) == [28, 37, 29]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_the_predicted_cost_is_what_growing_actually_costs(index: int) -> None:
    """The check that catches a change in GroMo's 1x1 padding convention.

    ``RestrictedConv2dGrowingModule`` zero-pads its 1x1 extension into a full
    ``k x k`` kernel, so one channel allocates ``out_channels * k_h * k_w``
    parameters in the receiving layer. If that convention ever changes, the
    derivation above stays plausible and this assertion fails.
    """
    model = conv_model()
    predicted = growable_neuron_costs(model, 784)[index]
    before = count_parameters(model)

    trial = copy.deepcopy(model)
    trial.set_growing_layers(index=index)
    compute_statistics(
        trial, image_loader(), loss_function=nn.MSELoss(reduction="sum"), device=DEVICE
    )
    trial.compute_optimal_updates(**UPDATE_KWARGS)
    trial.reset_computation()
    trial.dummy_select_update()
    target = trial.currently_updated_layer
    target.apply_change(
        apply_delta=False,
        apply_extension=True,
        input_extension_scaling=1.0,
        output_extension_scaling=1.0,
    )
    target.delete_update()
    trial.currently_updated_layer_index = None

    assert count_parameters(trial) - before == predicted


def test_growing_scratch_never_reaches_the_parameter_count() -> None:
    """The bordering convolution is scratch; it must not be priced as capacity.

    Asserted on ``requires_grad`` directly rather than on a parameter total,
    because ``compute_optimal_updates`` legitimately leaves PENDING extension
    layers registered -- real capacity in flight, which a total cannot tell
    apart from scratch.
    """
    model = conv_model()
    model.set_growing_layers(index=0)
    compute_statistics(
        model, image_loader(), loss_function=nn.MSELoss(reduction="sum"), device=DEVICE
    )
    model.compute_optimal_updates(**UPDATE_KWARGS)

    scratch = [
        module.bordering_convolution
        for module in model.modules()
        if getattr(module, "bordering_convolution", None) is not None
    ]
    assert scratch, "the probe did not allocate the scratch, so this proves nothing"
    for helper in scratch:
        assert not any(p.requires_grad for p in helper.parameters())
        # And it is what we think it is: a constant depthwise delta kernel.
        assert helper.groups == helper.in_channels
    # So none of it is in the Jacobian's columns either.
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    for helper in scratch:
        assert not any(id(p) in trainable for p in helper.parameters())
    model.reset_computation()


def test_inserted_layer_cost_defaults_to_the_linear_figure() -> None:
    assert inserted_layer_cost(7) == 7 * 7 + 7
    assert inserted_layer_cost(2, 9) == 2 * 2 * 9 + 2 == 38
