"""Oracle invariants for function-preserving exhaustive ``where`` candidates."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.search.growth import ScalingLineSearchConfig, grow_layer
from fgdlib.tangent import (
    FGDApproxConfig,
    exact_tangent_system,
    tiny_optimal_update_kwargs,
)


ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP  # noqa: E402


def _prefix_coordinate_map(
    base: torch.nn.Module,
    candidate: torch.nn.Module,
) -> tuple[int, ...]:
    """Map base flattened coordinates into prefix-preserving grown tensors."""
    candidate_parameters = dict(candidate.named_parameters())
    mapped: list[int] = []
    candidate_offset = 0
    for name, base_parameter in base.named_parameters():
        grown_parameter = candidate_parameters[name]
        assert base_parameter.ndim == grown_parameter.ndim
        assert all(
            old <= new
            for old, new in zip(base_parameter.shape, grown_parameter.shape)
        )
        for old_flat_index in range(base_parameter.numel()):
            old_coordinate = torch.unravel_index(
                torch.tensor(old_flat_index), base_parameter.shape
            )
            grown_flat_index = 0
            for dimension, coordinate in enumerate(old_coordinate):
                stride = (
                    int(torch.tensor(grown_parameter.shape[dimension + 1 :]).prod())
                    if dimension + 1 < grown_parameter.ndim
                    else 1
                )
                grown_flat_index += int(coordinate) * stride
            mapped.append(candidate_offset + grown_flat_index)
        candidate_offset += grown_parameter.numel()
    return tuple(mapped)


def _problem(out_features: int):
    torch.manual_seed(7)
    model = GrowingMLP(
        in_features=4,
        out_features=out_features,
        hidden_size=2,
        number_hidden_layers=2,
        device=torch.device("cpu"),
    )
    x = torch.randn(12, 4)
    y = torch.randn(12, out_features)
    loader = DataLoader(TensorDataset(x, y), batch_size=6, shuffle=False)
    config = FGDApproxConfig(
        projection_fast_factorization=False,
        certify_stream_gram=False,
    )
    return model, x, y, loader, config


@pytest.mark.parametrize("out_features", [1, 2])
def test_function_preserving_candidates_have_shared_old_columns(
    out_features: int,
) -> None:
    """Every actual growable location has ``[J_base | C]`` up to roundoff."""
    model, x, y, loader, config = _problem(out_features)
    base_output = model(x).detach()
    base_system = exact_tangent_system(model, x, y, config)
    assert base_system is not None

    for location in range(len(model._growable_layers)):
        candidate = copy.deepcopy(model)
        grow_layer(
            model=candidate,
            train_loader=loader,
            layer_index=location,
            device=torch.device("cpu"),
            line_search_config=ScalingLineSearchConfig(),
            optimal_update_kwargs=tiny_optimal_update_kwargs(
                config,
                compute_delta=False,
            ),
            function_preserving=True,
        )
        candidate_system = exact_tangent_system(candidate, x, y, config)
        assert candidate_system is not None

        old_coordinates = _prefix_coordinate_map(model, candidate)
        assert len(old_coordinates) == base_system.jacobian.shape[1]
        assert tuple(sorted(old_coordinates)) == old_coordinates
        assert torch.allclose(candidate(x), base_output, atol=2e-6, rtol=2e-6)
        assert torch.allclose(
            candidate_system.jacobian[:, old_coordinates],
            base_system.jacobian,
            atol=5e-7,
            rtol=2e-6,
        )


@pytest.mark.parametrize("location", [0, 1])
def test_new_column_partition_contains_all_and_only_nonzero_columns(
    location: int,
) -> None:
    """The complement of the old-coordinate map is the complete new block."""
    model, x, y, loader, config = _problem(out_features=2)
    candidate = copy.deepcopy(model)
    grow_layer(
        model=candidate,
        train_loader=loader,
        layer_index=location,
        device=torch.device("cpu"),
        line_search_config=ScalingLineSearchConfig(),
        optimal_update_kwargs=tiny_optimal_update_kwargs(
            config,
            compute_delta=False,
        ),
        function_preserving=True,
    )
    candidate_system = exact_tangent_system(candidate, x, y, config)
    assert candidate_system is not None

    old_coordinates = set(_prefix_coordinate_map(model, candidate))
    new_coordinates = tuple(
        index
        for index in range(candidate_system.jacobian.shape[1])
        if index not in old_coordinates
    )
    nonzero_coordinates = tuple(
        index
        for index in new_coordinates
        if torch.count_nonzero(candidate_system.jacobian[:, index]).item() != 0
    )
    zero_coordinates = tuple(
        index for index in new_coordinates if index not in nonzero_coordinates
    )

    assert new_coordinates
    assert nonzero_coordinates
    assert zero_coordinates
    assert torch.count_nonzero(
        candidate_system.jacobian[:, zero_coordinates]
    ).item() == 0
    reconstructed = torch.cat(
        [
            candidate_system.jacobian[
                :, _prefix_coordinate_map(model, candidate)
            ],
            candidate_system.jacobian[:, nonzero_coordinates],
        ],
        dim=1,
    )
    assert reconstructed.shape[1] == (
        len(old_coordinates) + len(nonzero_coordinates)
    )
