"""Declarative model stacks: place mlp / dropout / batchnorm from the config."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from fgdlib.models.regularized_mlp import _hidden_norm
from fgdlib.models.stack import LayerSpec, build_stack_model, parse_stack
from stable_tiny.pipeline import build_model, load_pipeline_config


def test_parse_expands_mlp_blocks_and_attaches_regularizers() -> None:
    specs = parse_stack(
        [{"mlp": [2, 1]}, "batchnorm", {"mlp": [3, 2]}, {"dropout": 0.2}]
    )
    # mlp [2,1] -> one width-2 layer with batch-norm; mlp [3,2] -> two width-3
    # layers, dropout on the LAST (the one whose output it sees).
    assert specs == [
        LayerSpec(width=2, batchnorm=True, dropout_rate=0.0),
        LayerSpec(width=3, batchnorm=False, dropout_rate=0.0),
        LayerSpec(width=3, batchnorm=False, dropout_rate=0.2),
    ]


def test_mlp_shorthand_and_dict_forms() -> None:
    assert parse_stack([{"mlp": 4}]) == [LayerSpec(width=4)]
    assert parse_stack([{"mlp": {"width": 4, "layers": 2}}]) == [
        LayerSpec(width=4),
        LayerSpec(width=4),
    ]


def test_regularizer_without_a_preceding_mlp_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_stack(["batchnorm"])
    with pytest.raises(ValueError):
        parse_stack([{"dropout": 0.1}])


def test_build_places_regularizers_exactly_where_asked() -> None:
    device = torch.device("cpu")
    model = build_stack_model(
        stack=[{"mlp": [2, 1]}, "batchnorm", {"mlp": [2, 1]}, {"dropout": 0.2}],
        in_features=784,
        out_features=10,
        device=device,
    )
    hidden = list(model.layers)[:-1]
    assert len(hidden) == 2
    # Layer 0: batch-norm present.
    assert _hidden_norm(hidden[0].post_layer_function) is not None
    # Layer 1: dropout, no batch-norm.
    assert _hidden_norm(hidden[1].post_layer_function) is None
    assert isinstance(hidden[1].post_layer_function, nn.Sequential)
    # Output layer never regularized.
    assert isinstance(model.layers[-1].post_layer_function, nn.Identity)
    # All hidden layers are growable.
    assert len(model._growable_layers) == 2


def test_mixed_starting_widths_are_rejected_for_now() -> None:
    with pytest.raises(ValueError, match="same"):
        build_stack_model(
            stack=[{"mlp": [2, 1]}, {"mlp": [4, 1]}],
            in_features=8,
            out_features=2,
            device=torch.device("cpu"),
        )


def test_stack_config_builds_through_build_model() -> None:
    """The pipeline entry point routes a `model.stack` to the stack builder."""
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(
            config.model,
            stack=({"mlp": [2, 2]}, "batchnorm"),
        ),
    )
    model = build_model(config, torch.device("cpu"))
    hidden = list(model.layers)[:-1]
    assert len(hidden) == 2
    # batch-norm attaches to the LAST layer of the block.
    assert _hidden_norm(hidden[-1].post_layer_function) is not None


def test_stack_grows_and_forwards() -> None:
    """A stack model is a first-class growable GrowingMLP."""
    from fgdlib.search.growth import grow_layer, rank_layer_expansion_score
    from fgdlib.tangent import tiny_optimal_update_kwargs

    device = torch.device("cpu")
    config = load_pipeline_config("configs/fgd/default.yaml")
    model = build_stack_model(
        stack=[{"mlp": [3, 1]}, "batchnorm", {"mlp": [3, 1]}],
        in_features=config.data.in_features,
        out_features=config.data.out_features,
        device=device,
    )
    torch.manual_seed(0)
    inputs = torch.randn(48, config.data.in_features)
    targets = torch.randn(48, config.data.out_features)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets), batch_size=24
    )
    kwargs = tiny_optimal_update_kwargs(config.fgd_approx, compute_delta=True)

    assert rank_layer_expansion_score(model, loader, 0, device, kwargs) >= 0.0
    grow_layer(
        model=model,
        train_loader=loader,
        layer_index=0,
        device=device,
        line_search_config=config.scaling_line_search,
        optimal_update_kwargs=kwargs,
        function_preserving=True,
        preservation_tolerance=config.fgd_approx.growth_preservation_tolerance,
    )
    model(torch.randn(4, config.data.in_features))     # no dimension error
