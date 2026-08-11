"""Counterexample-guided refinement of the full-MNIST tangent probe."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from fgdlib.search.realize import RealizationResult
from fgdlib.tangent import ExactTangentSystem, FGDApproxConfig, build_projection_probe
from stable_tiny import pipeline


def _config(**overrides) -> FGDApproxConfig:
    values = {
        "family_order": ("matrix_free_tangent",),
        "certify_realize_path": True,
        "certify_apply_in_interval": True,
        "projection_damping_auto": True,
        "transactional_realized_descent": True,
        "transactional_max_retries": 2,
        "transactional_backtrack_factor": 0.5,
        "transactional_descent_atol": 0.0,
        "transactional_min_predicted_decrease_fraction": 0.0,
        "certify_realizable_progress_growth": True,
        "certify_probe_refine_on_transaction_mismatch": True,
        "certify_probe_refine_batches_per_round": 1,
        "certify_probe_refine_max_rounds": 4,
    }
    values.update(overrides)
    return replace(FGDApproxConfig(), **values)


def _batch(value: float) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([[value]]), torch.tensor([[value + 0.5]])


def _mismatch(*violating: tuple[int, float], outer_step: int = 7):
    return SimpleNamespace(
        trial=SimpleNamespace(outer_step_global_index=outer_step),
        violating_batches=tuple(violating),
    )


def _system(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> ExactTangentSystem:
    named_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    named_buffers = tuple(model.named_buffers())
    return ExactTangentSystem(
        jacobian=torch.empty((0, sum(p.numel() for p in named_parameters.values()))),
        target=torch.ones(1),
        parameters=tuple(named_parameters.values()),
        loss=1.0,
        factors=(torch.ones((1, 1)), torch.ones((1, 1))),
        owner_model=model,
        parameter_names=tuple(named_parameters),
        parameter_versions=tuple(p._version for p in named_parameters.values()),
        buffer_names=tuple(name for name, _ in named_buffers),
        buffers=tuple(buffer for _, buffer in named_buffers),
        buffer_versions=tuple(buffer._version for _, buffer in named_buffers),
        probe_x=x,
        probe_y=y,
        probe_versions=(x._version, y._version),
        config_signature=repr(config),
        evaluation_state=tuple(
            (name, module.training) for name, module in model.named_modules()
        ),
    )


def test_true_mismatch_requires_probe_descent_and_full_train_ascent() -> None:
    assert pipeline._is_probe_full_train_mismatch(
        probe_before=10.0,
        probe_after=9.0,
        full_train_before=100.0,
        full_train_after=101.0,
        tolerance=1.0e-10,
    )
    assert not pipeline._is_probe_full_train_mismatch(
        probe_before=10.0,
        probe_after=9.0,
        full_train_before=100.0,
        full_train_after=99.0,
        tolerance=0.0,
    )
    assert not pipeline._is_probe_full_train_mismatch(
        probe_before=10.0,
        probe_after=float("nan"),
        full_train_before=100.0,
        full_train_after=101.0,
        tolerance=0.0,
    )


def test_violating_batches_rank_largest_delta_then_index() -> None:
    ranked = pipeline._rank_violating_full_train_batches(
        (1.0, 1.0, 1.0, 1.0),
        (1.5, 0.5, 1.5, 1.2),
    )
    assert ranked == ((0, 0.5), (2, 0.5), (3, pytest.approx(0.2)))


def test_initial_probe_is_exact_and_refinement_is_monotone_without_duplicates() -> None:
    batches = [_batch(float(index)) for index in range(5)]
    expected = build_projection_probe(batches, 2, torch.device("cpu"))
    state = pipeline._new_adaptive_certification_probe(batches, 2)
    probe = pipeline._compose_adaptive_certification_probe(
        state, _config(), torch.device("cpu")
    )
    assert torch.equal(probe[0], expected[0])
    assert torch.equal(probe[1], expected[1])

    model = torch.nn.Linear(1, 1)
    refined = pipeline._refine_adaptive_certification_probe(
        state=state,
        mismatch=_mismatch((3, 2.0), (2, 1.0)),
        frozen_train_batches=batches,
        config=_config(),
        model=model,
        device=torch.device("cpu"),
        progress=None,
    )
    assert refined is not None
    assert torch.equal(refined[0][: expected[0].shape[0]], expected[0])
    assert torch.equal(refined[0][-1:], batches[3][0])
    assert [
        (item.outer_step_global_index, item.batch_index)
        for item in state.counterexamples
    ] == [(7, 3)]

    # The same content cannot enter B twice.
    duplicate = pipeline._refine_adaptive_certification_probe(
        state=state,
        mismatch=_mismatch((3, 3.0)),
        frozen_train_batches=batches,
        config=_config(),
        model=model,
        device=torch.device("cpu"),
        progress=None,
    )
    assert duplicate is None
    assert len(state.counterexamples) == 1


def test_rank_base_growth_keeps_discovered_counterexample() -> None:
    batches = [_batch(float(index)) for index in range(6)]
    state = pipeline._new_adaptive_certification_probe(batches[:2], 2)
    model = torch.nn.Linear(1, 1)
    assert (
        pipeline._refine_adaptive_certification_probe(
            state=state,
            mismatch=_mismatch((4, 2.0)),
            frozen_train_batches=batches,
            config=_config(),
            model=model,
            device=torch.device("cpu"),
            progress=None,
        )
        is not None
    )

    pipeline._extend_adaptive_probe_base(state, batches, 3)
    probe = pipeline._compose_adaptive_certification_probe(
        state, _config(), torch.device("cpu")
    )
    assert len(state.base_batches) == 3
    assert torch.equal(probe[0][-1:], batches[4][0])
    assert state.counterexamples[0].batch_index == 4


def test_refined_probe_changes_functional_gradient_and_tangent_rows() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.25)
    batches = [_batch(1.0), _batch(2.0), _batch(4.0)]
    state = pipeline._new_adaptive_certification_probe(batches, 2)
    config = _config(family_order=("tangent",))
    old_probe = pipeline._compose_adaptive_certification_probe(
        state, config, torch.device("cpu")
    )
    old_gradient = pipeline.functional_gradient(model(old_probe[0]), old_probe[1])
    old_system = pipeline.exact_tangent_system(model, *old_probe, config)

    refined = pipeline._refine_adaptive_certification_probe(
        state=state,
        mismatch=_mismatch((2, 1.0)),
        frozen_train_batches=batches,
        config=config,
        model=model,
        device=torch.device("cpu"),
        progress=None,
    )
    assert refined is not None
    new_gradient = pipeline.functional_gradient(model(refined[0]), refined[1])
    new_system = pipeline.exact_tangent_system(model, *refined, config)
    assert new_gradient.numel() > old_gradient.numel()
    assert torch.equal(new_gradient[: old_gradient.numel()], old_gradient)
    assert old_system is not None and new_system is not None
    assert new_system.target.numel() > old_system.target.numel()


def test_transaction_returns_smallest_mismatch_and_restores_same_theta(
    monkeypatch,
) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    original = model.weight.detach().clone()
    x, y = _batch(1.0)
    config = _config()
    system = _system(model, x, y, config)

    def fake_realize(walker, _x, _y, _updates, rate, _config, **_kwargs):
        with torch.no_grad():
            walker.weight.add_(rate)
        return RealizationResult(
            residual_fraction=0.1,
            realised_fraction=0.9,
            iterations=1,
            parameter_displacement=rate,
            functional_before=10.0,
            functional_after=9.0,
            functional_delta=-1.0,
            aligned_realised_fraction=0.8,
            effective_learning_rate=0.8 * rate,
        )

    measurements = iter(
        [
            (3.0, 3, (1.0, 1.0, 1.0)),
            (4.0, 3, (1.1, 1.7, 1.2)),
            (3.6, 3, (1.1, 1.4, 1.1)),
            (3.3, 3, (1.0, 1.3, 1.0)),
        ]
    )
    monkeypatch.setattr(pipeline, "realize_functional_step", fake_realize)
    monkeypatch.setattr(
        pipeline,
        "_evaluate_transactional_full_train_batch_losses",
        lambda *_args, **_kwargs: next(measurements),
    )

    result = pipeline._transactional_realize_functional_step(
        model=model,
        x=x,
        y=y,
        updates=(torch.ones_like(model.weight),),
        proposed_learning_rate=0.2,
        capped_learning_rate=0.1,
        relative_error=0.1,
        system=system,
        selected_relative_damping=1.0e-3,
        selected_absolute_damping=1.0e-2,
        config=config,
        epoch=4,
        outer_step_index=0,
        outer_step_global_index=11,
        full_train_batches=[_batch(1.0), _batch(2.0), _batch(3.0)],
        full_train_device=torch.device("cpu"),
    )

    assert result.candidate_model is None
    assert result.probe_mismatch is not None
    assert result.probe_mismatch.trial.trial_learning_rate == pytest.approx(0.025)
    assert result.probe_mismatch.violating_batches[0] == (1, pytest.approx(0.3))
    assert torch.equal(model.weight, original)


def test_disabled_transaction_does_not_collect_per_batch_losses(monkeypatch) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    x, y = _batch(1.0)
    config = _config(certify_probe_refine_on_transaction_mismatch=False)
    system = _system(model, x, y, config)

    monkeypatch.setattr(
        pipeline,
        "_evaluate_transactional_full_train_batch_losses",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled refinement collected batch losses")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "realize_functional_step",
        lambda *_args, **_kwargs: RealizationResult(
            residual_fraction=0.1,
            realised_fraction=0.9,
            iterations=1,
            parameter_displacement=0.1,
            functional_before=2.0,
            functional_after=1.0,
            functional_delta=-1.0,
            aligned_realised_fraction=0.8,
            effective_learning_rate=0.08,
        ),
    )
    result = pipeline._transactional_realize_functional_step(
        model=model,
        x=x,
        y=y,
        updates=(torch.ones_like(model.weight),),
        proposed_learning_rate=0.1,
        capped_learning_rate=0.1,
        relative_error=0.1,
        system=system,
        selected_relative_damping=1.0e-3,
        selected_absolute_damping=1.0e-2,
        config=config,
        epoch=1,
        outer_step_index=0,
        outer_step_global_index=1,
        full_train_batches=[_batch(1.0)],
        full_train_device=torch.device("cpu"),
    )
    assert result.probe_mismatch is None


def test_refinement_limit_logs_explicit_exhaustion() -> None:
    batches = [_batch(float(index)) for index in range(4)]
    state = pipeline._new_adaptive_certification_probe(batches[:1], 1)
    model = torch.nn.Linear(1, 1)
    messages: list[str] = []
    config = _config(certify_probe_refine_max_rounds=1)

    assert (
        pipeline._refine_adaptive_certification_probe(
            state=state,
            mismatch=_mismatch((1, 2.0)),
            frozen_train_batches=batches,
            config=config,
            model=model,
            device=torch.device("cpu"),
            progress=messages.append,
        )
        is not None
    )
    assert (
        pipeline._refine_adaptive_certification_probe(
            state=state,
            mismatch=_mismatch((2, 3.0)),
            frozen_train_batches=batches,
            config=config,
            model=model,
            device=torch.device("cpu"),
            progress=messages.append,
        )
        is None
    )
    assert state.exhausted is True
    assert any("probe_refinement_exhausted" in message for message in messages)


def test_refined_probe_must_be_transaction_checked_before_growth() -> None:
    config = _config()
    assert pipeline._realizable_growth_is_active(
        config,
        previous_step_committed=False,
        previous_failure_non_finite=False,
        probe_requires_consistency_check=False,
    )
    assert not pipeline._realizable_growth_is_active(
        config,
        previous_step_committed=False,
        previous_failure_non_finite=False,
        probe_requires_consistency_check=True,
    )
