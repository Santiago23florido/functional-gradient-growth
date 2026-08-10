"""The conv container, and the two facts of GroMo that shape it.

These tests are deliberately assertions about GroMo's behaviour, not just
about our code. If a GroMo upgrade lifts the topology restriction or changes
the activation-derivative lookup, the flow's growth decisions change
underneath us, and we want that to fail loudly here rather than show up as a
worse accuracy number three sessions later.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgdlib.models.convstack import (
    ConvStackModel,
    build_conv_stack_model,
    is_conv_stack,
    parse_conv_stack,
)
from fgdlib.models.stack import build_stack_model

# The plan's base stack: 202 parameters, three growable locations, matching
# the linear reference's three (hidden_size 2, number_hidden_layers 3).
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

SELU_GRADIENT_AT_ZERO_PLUS = 1.0507


def build(stack=None, **kwargs) -> ConvStackModel:
    torch.manual_seed(0)
    return build_conv_stack_model(
        stack if stack is not None else BASE_STACK,
        in_features=784,
        out_features=10,
        device=torch.device("cpu"),
        input_shape=(1, 28, 28),
        **kwargs,
    )


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_the_base_stack_has_the_expected_size() -> None:
    assert count_parameters(build()) == 202


def test_forward_maps_a_batch_of_images_to_class_scores() -> None:
    model = build()
    assert model(torch.rand(4, 1, 28, 28)).shape == (4, 10)


def test_growable_layers_are_exactly_the_ones_gromo_allows() -> None:
    """Pins GroMo's topology restriction as an executable statement.

    A conv is growable iff its predecessor is a conv with no shape change in
    between (``bordered_unfolded_extended_prev_input`` shares the spatial
    grid), and a Linear is never growable behind a conv
    (``linear_growing_module.py:585``). For the base stack that is layers
    1 (conv B), 3 (conv D) and 5 (the output fc) -- and NOT 0, 2 or 4.
    """
    model = build()
    growable = [
        index
        for index, layer in enumerate(model.layers)
        if any(layer is growing for growing in model._growable_layers)
    ]
    assert growable == [1, 3, 5]


def test_pooling_did_not_end_up_inside_a_post_layer_function() -> None:
    """The arrangement that keeps ``activation_gradient`` exact.

    GroMo walks ``post_layer_function`` looking up each entry's derivative at
    0+ and warns-then-numerically-differentiates anything it does not know
    (``growing_module.py:1116-1144``). A pooling module there returns either a
    crash or a garbage number -- and the winning ``where`` rule is
    ``activation_gradient * sum(s_i**2)``, so garbage would silently corrupt
    every growth decision rather than fail.
    """
    model = build()
    for layer in model._growable_layers:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value = float(layer.activation_gradient)
        assert value == pytest.approx(SELU_GRADIENT_AT_ZERO_PLUS, abs=1e-4)


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_dropout_does_not_randomise_the_activation_derivative(mode: str) -> None:
    """Dropout in the post-function must not turn the where-rule into a lottery.

    Left unregistered, GroMo estimates dropout's derivative at 0+ by
    differentiating it on a single 0-D scalar: measured 1.0507 in eval but
    0.0 in train whenever that one element is the dropped one -- and the
    value is memoised. The where-rule multiplies by it, so a cached 0.0 would
    silently retire a layer for the rest of the run.
    """
    model = build(
        [
            {"conv": [2, 3]},
            {"conv": [2, 3]},
            {"dropout": 0.25},
            {"avgpool": [3, 3]},
            "flatten",
            {"mlp": [2, 1]},
            {"dropout": 0.25},
        ]
    )
    getattr(model, mode)()
    for layer in model._growable_layers:
        layer._activation_gradient_previous_module = None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value = float(layer.activation_gradient)
        assert value == pytest.approx(SELU_GRADIENT_AT_ZERO_PLUS, abs=1e-4)


def test_extended_forward_matches_forward_without_an_extension() -> None:
    """Pins the ``x_ext`` threading through pooling and flatten."""
    model = build()
    x = torch.rand(4, 1, 28, 28)
    with torch.no_grad():
        assert torch.equal(model.extended_forward(x), model(x))


def test_batchnorm_and_dropout_attach_to_the_module_above() -> None:
    model = build(
        [
            {"conv": [2, 3]},
            {"conv": [2, 3]},
            "batchnorm",
            "maxpool",
            {"avgpool": [3, 3]},
            "flatten",
            {"mlp": [2, 1]},
            {"dropout": 0.25},
        ]
    )
    from gromo.modules.growing_dropout import GrowingDropout
    from gromo.modules.growing_normalisation import GrowingBatchNorm2d

    post_conv = model.layers[1].post_layer_function
    assert any(isinstance(m, GrowingBatchNorm2d) for m in post_conv)
    post_fc = model.layers[2].post_layer_function
    assert any(isinstance(m, GrowingDropout) for m in post_fc)
    assert model(torch.rand(4, 1, 28, 28)).shape == (4, 10)


def test_the_growth_scheme_is_selectable() -> None:
    from gromo.modules.conv2d_growing_module import (
        FullConv2dGrowingModule,
        RestrictedConv2dGrowingModule,
    )

    assert isinstance(build().layers[0], RestrictedConv2dGrowingModule)
    assert isinstance(
        build(conv_growth_scheme="full").layers[0], FullConv2dGrowingModule
    )
    with pytest.raises(ValueError, match="conv_growth_scheme"):
        build(conv_growth_scheme="sideways")


def test_the_flatten_width_is_derived_from_the_pooling_arithmetic() -> None:
    """avgpool (3,3) on 2 channels -> 18 features into the head."""
    assert build().layers[4].in_features == 18
    wider = build(
        [
            {"conv": [2, 3]},
            {"avgpool": [4, 4]},
            "flatten",
            {"mlp": [2, 1]},
        ]
    )
    assert wider.layers[1].in_features == 32


def test_the_stack_is_rejected_when_it_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="flatten"):
        build([{"conv": [2, 3]}, {"mlp": [2, 1]}])
    with pytest.raises(ValueError, match="conv cannot follow the flatten"):
        build([{"conv": [2, 3]}, "flatten", {"conv": [2, 3]}, {"mlp": [2, 1]}])
    with pytest.raises(ValueError, match="odd kernel"):
        build([{"conv": [2, 4]}, "flatten", {"mlp": [2, 1]}])
    with pytest.raises(ValueError, match="input_shape"):
        build_conv_stack_model(
            BASE_STACK,
            in_features=784,
            out_features=10,
            device=torch.device("cpu"),
            input_shape=None,
        )


def test_the_dispatch_splits_the_catalog_the_way_it_claims() -> None:
    """Exactly one canonical config takes the conv path; the rest cannot.

    The MLP branch of ``build_stack_model`` is therefore provably unreachable
    from the conv config, and vice versa -- which is what makes "the linear
    runs are untouched" a fact about the code and not a hope.
    """
    from stable_tiny.pipeline import load_pipeline_config

    root = Path(__file__).resolve().parents[1] / "configs" / "fgd"
    conv_configs = set()
    for path in sorted(root.glob("*.yaml")):
        stack = load_pipeline_config(str(path)).model.stack
        if is_conv_stack(list(stack) if stack else None):
            conv_configs.add(path.name)
    assert conv_configs == {"mnist_conv_matrix_free_N1024.yaml"}


def test_the_mlp_stack_path_is_untouched() -> None:
    from gromo.containers.growing_mlp import GrowingMLP

    torch.manual_seed(0)
    model = build_stack_model(
        [{"mlp": [3, 2]}], in_features=4, out_features=1,
        device=torch.device("cpu"),
    )
    assert isinstance(model, GrowingMLP)


def test_parse_reports_the_ops_in_order() -> None:
    from fgdlib.models.convstack import ConvSpec, MlpSpec, OpSpec

    specs = parse_conv_stack(BASE_STACK)
    kinds = [
        spec.kind if isinstance(spec, OpSpec)
        else ("conv" if isinstance(spec, ConvSpec) else "mlp")
        for spec in specs
    ]
    assert kinds == [
        "conv", "conv", "maxpool", "conv", "conv", "maxpool",
        "avgpool", "flatten", "mlp",
    ]
    assert isinstance(specs[-1], MlpSpec)
