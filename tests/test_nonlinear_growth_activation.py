"""Function-preserving growth must hand the next candidate usable capacity.

A function-preserving growth adds a neuron whose OUTGOING weights are zero.
That is what keeps ``f`` fixed, and it is also a degenerate starting point:
with a zero outgoing weight the gradient reaching the new unit's INCOMING
weights is zero, so if the outgoing weight never moves the unit stays asleep
forever and every subsequent growth is pure parameter inflation.

It does un-stick, but only because the gradient w.r.t. the OUTGOING weight is
proportional to the new unit's activation, not to its outgoing weight. These
tests pin that: the outgoing weight leaves zero, and the incoming weights move
once it has.
"""

from __future__ import annotations

import torch

from fgdlib.search.nonlinear import build_nonlinear_probe, train_nonlinear_candidate
from stable_tiny import pipeline


def _config():
    from dataclasses import replace

    from tests.test_nonlinear_primary_family import _tiny_pipeline_config

    base = _tiny_pipeline_config()
    return replace(
        base,
        parametric_gd=replace(
            base.parametric_gd,
            inner_learning_rate=0.05,
            inner_step_unit="probe",
            candidate_objective="sum",
            parameter_penalty=0.0,
            gradient_clip_norm=None,
        ),
    )


def _grow_once(config, device, train_loader, model, monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "compute_expressivity_bottlenecks",
        lambda model, loader, device, config: [0.25, 2.0],
    )
    outcome = pipeline._apply_nonlinear_primary_growth(
        model=model,
        train_loader=train_loader,
        device=device,
        config=config,
        epoch=1,
        progress=None,
    )
    assert outcome.model is not None
    return outcome


def _grown_layer_weights(model, layer_index: int):
    """Weights on either side of the unit added at ``layer_index``.

    ``_growable_layers[i]`` widens its INPUT, so the new unit is produced by
    ``model.layers[i]`` (a new output row) and consumed by
    ``_growable_layers[i]`` (a new input column, zeroed by FP growth).
    """
    incoming = model.layers[layer_index].layer.weight
    outgoing = model._growable_layers[layer_index].layer.weight
    return incoming, outgoing


def test_new_unit_starts_with_zero_outgoing_weight(monkeypatch) -> None:
    config = _config()
    device = torch.device("cpu")
    train_loader, _, _ = pipeline.build_dataloaders(config, device)
    model = pipeline.build_model(config, device)
    outcome = _grow_once(config, device, train_loader, model, monkeypatch)

    _, outgoing = _grown_layer_weights(outcome.model, 1)
    # The newly added column is the last one; it is what keeps f fixed.
    new_column = outgoing[:, -1].detach()
    assert float(new_column.abs().max()) == 0.0


def test_candidate_training_wakes_the_new_unit(monkeypatch) -> None:
    """Both the outgoing AND the incoming weights of the new unit must move."""
    config = _config()
    device = torch.device("cpu")
    train_loader, _, _ = pipeline.build_dataloaders(config, device)
    model = pipeline.build_model(config, device)
    outcome = _grow_once(config, device, train_loader, model, monkeypatch)
    grown = outcome.model

    incoming_before, outgoing_before = _grown_layer_weights(grown, 1)
    incoming_before = incoming_before.detach().clone()
    outgoing_before = outgoing_before.detach().clone()

    probe = build_nonlinear_probe(train_loader, config.fgd_approx.probe_batches, device)
    candidate = train_nonlinear_candidate(
        base_model=grown,
        train_loader=train_loader,
        device=device,
        functional_learning_rate=1.0,
        inner_steps=50,
        config=config.parametric_gd,
        fgd_config=config.fgd_approx,
        probe=probe,
    )
    assert candidate.sensor_valid
    incoming_after, outgoing_after = _grown_layer_weights(candidate.model, 1)

    outgoing_after = outgoing_after.detach()
    incoming_after = incoming_after.detach()
    outgoing_moved = float(
        (outgoing_after[:, -1] - outgoing_before[:, -1]).abs().max()
    )
    incoming_moved = float(
        (incoming_after[-1, :] - incoming_before[-1, :]).abs().max()
    )
    assert outgoing_moved > 1e-8, "the new unit's outgoing weight never left zero"
    assert incoming_moved > 1e-8, (
        "the new unit's incoming weights never moved: it is asleep, so the "
        "added capacity is pure parameter inflation"
    )


def test_growth_preserves_the_function_within_tolerance(monkeypatch) -> None:
    config = _config()
    device = torch.device("cpu")
    train_loader, _, _ = pipeline.build_dataloaders(config, device)
    model = pipeline.build_model(config, device)
    probe_x, _ = next(iter(train_loader))
    with torch.no_grad():
        before = model(probe_x).clone()
    outcome = _grow_once(config, device, train_loader, model, monkeypatch)
    with torch.no_grad():
        after = outcome.model(probe_x)
    torch.testing.assert_close(
        before,
        after,
        rtol=0.0,
        atol=config.fgd_approx.growth_preservation_tolerance,
    )


def test_every_parameter_of_the_grown_model_stays_trainable(monkeypatch) -> None:
    """Nothing in the grown clone is frozen -- growth must not detach layers."""
    config = _config()
    device = torch.device("cpu")
    train_loader, _, _ = pipeline.build_dataloaders(config, device)
    model = pipeline.build_model(config, device)
    outcome = _grow_once(config, device, train_loader, model, monkeypatch)
    assert all(
        parameter.requires_grad for parameter in outcome.model.parameters()
    ), "a grown model has non-trainable parameters"
