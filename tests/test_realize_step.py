"""Realising the certified functional step by integration rather than a jump."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fgdlib.search.realize import realize_functional_step
from stable_tiny.pipeline import build_model, load_pipeline_config


@pytest.fixture
def setup():
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(config.model, hidden_size=16, number_hidden_layers=2),
        fgd_approx=replace(
            config.fgd_approx,
            projection_solver="exact",
            certify_linearization_tolerance=0.1,
        ),
    )
    device = torch.device("cpu")
    model = build_model(config, device)
    torch.manual_seed(0)
    x = torch.randn(8, config.data.in_features)
    y = torch.randn(8, config.data.out_features)
    updates = tuple(torch.randn_like(p) * 0.01 for p in model.parameters())
    return model, x, y, updates, config.fgd_approx


def test_it_travels_toward_the_target(setup) -> None:
    """The point of integrating: actually cover the intended displacement."""
    model, x, y, updates, fa = setup
    result = realize_functional_step(model, x, y, updates, 0.5, fa)
    assert result.realised_fraction > 0.0
    assert result.iterations > 0


def test_a_zero_step_is_a_no_op(setup) -> None:
    """No intended displacement means nothing to realise and nothing to move."""
    model, x, y, updates, fa = setup
    before = [p.detach().clone() for p in model.parameters()]
    result = realize_functional_step(model, x, y, updates, 0.0, fa)
    assert result.iterations == 0
    for parameter, original in zip(model.parameters(), before):
        assert torch.equal(parameter.detach(), original)


def test_it_performs_the_step_rather_than_proposing_one(setup) -> None:
    """It mutates the model in place -- callers must not re-apply anything."""
    model, x, y, updates, fa = setup
    before = [p.detach().clone() for p in model.parameters()]
    realize_functional_step(model, x, y, updates, 0.5, fa)
    assert any(
        not torch.equal(p.detach(), o)
        for p, o in zip(model.parameters(), before)
    )


def test_training_mode_is_restored(setup) -> None:
    """Measurement runs in eval; the caller's mode must survive it."""
    model, x, y, updates, fa = setup
    model.train()
    realize_functional_step(model, x, y, updates, 0.5, fa)
    assert model.training


def test_the_residual_and_realised_fractions_are_reported_honestly(setup) -> None:
    """A partial realisation must say so rather than claim the full step."""
    model, x, y, updates, fa = setup
    result = realize_functional_step(
        model, x, y, updates, 0.5, fa, max_iterations=1
    )
    assert 0.0 <= result.realised_fraction
    assert result.residual_fraction >= 0.0
    assert result.iterations <= 1
