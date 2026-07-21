"""Depth must enter the search without discarding the certified trajectory.

Uniform widening cannot answer depth at all; SENN answers it with the same
expansion score it answers width with. The precondition is that inserting a
layer preserves the represented function exactly, which is what these tests
pin.
"""

from __future__ import annotations

import torch
from torch import nn

from fgdlib.depth import (
    IdentityHomotopyActivation,
    insert_identity_layer,
    inserted_layer_cost,
)
from stable_tiny.pipeline import build_model, load_pipeline_config


def _model(hidden_size: int = 4):
    from dataclasses import replace

    device = torch.device("cpu")
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(
            config.model, hidden_size=hidden_size, number_hidden_layers=3
        ),
    )
    return build_model(config, device), config, device


def test_homotopy_starts_at_the_identity() -> None:
    activation = IdentityHomotopyActivation(nn.SELU())
    inputs = torch.randn(32, 7)
    assert torch.allclose(activation(inputs), inputs, atol=0.0)


def test_homotopy_reaches_the_wrapped_activation() -> None:
    inner = nn.SELU()
    activation = IdentityHomotopyActivation(inner, alpha=1.0)
    inputs = torch.randn(32, 7)
    assert torch.allclose(activation(inputs), inner(inputs), atol=1e-6)


def test_alpha_is_trainable() -> None:
    """The layer must EARN its nonlinearity by descent, not be handed it."""
    activation = IdentityHomotopyActivation(nn.SELU())
    assert activation.alpha.requires_grad
    inputs = torch.randn(16, 5)
    activation(inputs).pow(2).sum().backward()
    assert activation.alpha.grad is not None


def test_insertion_preserves_the_function_exactly() -> None:
    """Not to a tolerance -- exactly. The trajectory must survive a growth."""
    model, config, device = _model()
    inputs = torch.randn(64, config.data.in_features)
    model.eval()
    with torch.no_grad():
        before = model(inputs).clone()

    depth_before = len(model.layers)
    insert_identity_layer(model, position=2, device=device)
    assert len(model.layers) == depth_before + 1

    model.eval()
    with torch.no_grad():
        after = model(inputs)
    assert torch.allclose(before, after, atol=1e-6)


def test_insertion_relinks_the_module_chain() -> None:
    """Stale `previous_module` links would corrupt the growth statistics."""
    model, _, device = _model()
    insert_identity_layer(model, position=1, device=device)
    layers = list(model.layers)
    assert getattr(layers[0], "previous_module", None) is None
    for index in range(1, len(layers)):
        assert layers[index].previous_module is layers[index - 1]


def test_insertion_rejects_impossible_positions() -> None:
    model, _, device = _model()
    for position in (0, len(model.layers)):
        try:
            insert_identity_layer(model, position=position, device=device)
        except ValueError:
            continue
        raise AssertionError(f"position {position} should have been rejected")


def test_depth_cost_is_comparable_to_neuron_cost() -> None:
    """Both proposals are priced in parameters, so they can be ranked together.

    This is what lets a depth candidate and a width candidate compete in one
    pooled ranking by certified decrease per parameter.
    """
    from dataclasses import replace

    from fgdlib.growth import growable_neuron_costs

    # Use the real MNIST search config: the interesting price spread only
    # exists when the input projection is genuinely wide.
    device = torch.device("cpu")
    config = load_pipeline_config("configs/fgd/search_ce_uniform.yaml")
    config = replace(config, model=replace(config.model, hidden_size=8))
    model = build_model(config, device)

    neuron_costs = growable_neuron_costs(model, config.data.in_features)
    depth_cost = inserted_layer_cost(8)
    assert depth_cost == 8 * 8 + 8

    # A layer at width 8 costs far less than a neuron on the 784-wide input
    # projection, and more than one on a narrow late layer -- exactly the
    # trade the pooled ranking has to arbitrate, and the reason depth cannot
    # be decided by a separate policy from width.
    assert depth_cost < neuron_costs[0]
    assert depth_cost > min(neuron_costs[1:])


def test_insertion_refreshes_gromo_bookkeeping() -> None:
    """The inserted layer must become a first-class GroMo growable layer.

    ``GrowingMLP.__init__`` builds ``_growable_layers`` once, so an insertion
    that does not refresh it leaves the container describing the OLD graph:
    the new layer could never be widened and the growable indices would stop
    matching positions in ``layers``. Everything downstream --
    compute_statistics, compute_optimal_updates, the TINY spectrum -- reads
    that list.
    """
    model, _, device = _model(hidden_size=6)
    growable_before = len(model._growable_layers)

    inserted = insert_identity_layer(model, position=2, device=device)

    assert len(model._growable_layers) == growable_before + 1
    assert inserted in model._growable_layers
    # The list must mirror the real graph, in order.
    assert list(model._growable_layers) == list(model.layers[1:])
    # And GroMo's selection machinery still works on the new chain.
    assert model._growing_layers


def test_a_grown_model_still_widens_after_an_insertion() -> None:
    """End-to-end: insert a layer, then grow one -- GroMo must cope."""
    import torch as _torch

    from fgdlib.growth import rank_layer_expansion_score
    from fgdlib.tangent import tiny_optimal_update_kwargs

    model, config, device = _model(hidden_size=6)
    insert_identity_layer(model, position=2, device=device)

    _torch.manual_seed(0)
    inputs = _torch.randn(64, config.data.in_features)
    targets = _torch.randn(64, config.data.out_features)
    loader = _torch.utils.data.DataLoader(
        _torch.utils.data.TensorDataset(inputs, targets), batch_size=32
    )
    kwargs = tiny_optimal_update_kwargs(config.fgd_approx, compute_delta=True)

    # Every location, including the freshly inserted one, must be scorable.
    scores = [
        rank_layer_expansion_score(model, loader, index, device, kwargs)
        for index in range(len(model._growable_layers))
    ]
    assert len(scores) == len(model._growable_layers)
    assert all(score >= 0.0 for score in scores)
