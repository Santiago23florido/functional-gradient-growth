"""Exact streamed shared-base statistics match materialised Jacobian products."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.search.exact_where import (
    identify_growth_columns,
    stream_shared_candidate_statistics,
)
from fgdlib.search.growth import ScalingLineSearchConfig, grow_layer
from fgdlib.tangent import (
    FGDApproxConfig,
    exact_tangent_system,
    tiny_optimal_update_kwargs,
)

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP


def _grown_candidates(out_features: int = 2):
    torch.manual_seed(23)
    model = GrowingMLP(
        in_features=3,
        out_features=out_features,
        hidden_size=2,
        number_hidden_layers=2,
        device=torch.device("cpu"),
    )
    x = torch.randn(10, 3)
    y = torch.randn(10, out_features)
    loader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=False)
    config = FGDApproxConfig(
        projection_fast_factorization=False,
        certify_stream_gram=False,
        certify_stream_chunk=3,
        growth_preservation_tolerance=1e-6,
    )
    candidates = []
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
        candidates.append(candidate)
    return model, candidates, x, y, config


@pytest.mark.parametrize("out_features", [1, 2])
def test_streamed_cross_statistics_equal_explicit_products(
    out_features: int,
) -> None:
    model, candidates, x, y, config = _grown_candidates(out_features)
    base_system = exact_tangent_system(model, x, y, config)
    assert base_system is not None
    layouts = tuple(
        identify_growth_columns(
            model,
            candidate,
            x,
            preservation_tolerance=config.growth_preservation_tolerance,
        )
        for candidate in candidates
    )
    shared = stream_shared_candidate_statistics(
        model,
        base_system,
        layouts,
        x,
        y,
        config,
    )

    j_base = base_system.jacobian.to(torch.float64)
    r = base_system.target.to(torch.float64)
    assert torch.allclose(shared.gram, j_base.t() @ j_base, atol=1e-11, rtol=1e-9)
    assert torch.allclose(shared.rhs, j_base.t() @ r, atol=1e-11, rtol=1e-9)
    assert shared.target_sq_norm == pytest.approx(
        float(torch.dot(r, r)), abs=1e-11, rel=1e-9
    )

    for candidate, layout, statistics in zip(candidates, layouts, shared.candidates):
        candidate_system = exact_tangent_system(candidate, x, y, config)
        assert candidate_system is not None
        c_full = candidate_system.jacobian[
            :, statistics.new_candidate_coordinates
        ].to(torch.float64)
        assert torch.allclose(
            statistics.cross_gram,
            j_base.t() @ c_full,
            atol=2e-6,
            rtol=2e-6,
        )
        assert torch.allclose(
            statistics.new_gram,
            c_full.t() @ c_full,
            atol=2e-6,
            rtol=2e-6,
        )
        assert torch.allclose(
            statistics.new_rhs,
            c_full.t() @ r,
            atol=2e-6,
            rtol=2e-6,
        )
        omitted = set(layout.new_candidate_coordinates) - set(
            statistics.new_candidate_coordinates
        )
        assert omitted
        assert torch.count_nonzero(
            candidate_system.jacobian[:, tuple(sorted(omitted))]
        ).item() == 0


def test_streamed_base_system_is_reconstructed_only_once(monkeypatch) -> None:
    """A surrogate base still streams one shared J, never one J per candidate."""
    model, candidates, x, y, config = _grown_candidates()
    streamed_config = replace(
        config,
        certify_stream_gram=True,
        certify_stream_chunk=3,
    )
    base_system = exact_tangent_system(model, x, y, streamed_config)
    assert base_system is not None
    assert base_system.jacobian.shape[0] != base_system.full_target.numel()
    layouts = tuple(
        identify_growth_columns(
            model,
            candidate,
            x,
            preservation_tolerance=config.growth_preservation_tolerance,
        )
        for candidate in candidates
    )

    from fgdlib.search import exact_where

    original = exact_where._base_jacobian_block
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(exact_where, "_base_jacobian_block", counted)
    stream_shared_candidate_statistics(
        model,
        base_system,
        layouts,
        x,
        y,
        streamed_config,
    )
    assert calls == 4  # ceil(10 samples / chunk size 3), independent of candidates
