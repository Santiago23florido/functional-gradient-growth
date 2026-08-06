"""Opt-in realized-descent transaction for the matrix-free tangent path."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from fgdlib.search.realize import RealizationResult, realize_functional_step
from fgdlib.tangent import ExactTangentSystem, FGDApproxConfig
from stable_tiny import pipeline
from stable_tiny.pipeline import (
    FGDTransactionalTrialRecord,
    PipelineConfig,
    PipelineResult,
    _transactional_realize_functional_step,
    load_pipeline_config,
    result_payload,
)
from stable_tiny.wandb_logging import WandbConfig, WandbRunLogger


EXACT_CONFIG_SHA256 = "ca28df64d314e0a5c538dc27815ac2ddc4b5c22448f514f472bc3e9ed0f615a5"


def _transaction_config(**overrides) -> FGDApproxConfig:
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
    }
    values.update(overrides)
    return replace(FGDApproxConfig(), **values)


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


def _realization(before: float, after: float, rate: float) -> RealizationResult:
    return RealizationResult(
        residual_fraction=0.1,
        realised_fraction=0.9,
        iterations=1,
        parameter_displacement=0.2,
        functional_before=before,
        functional_after=after,
        functional_delta=after - before,
        aligned_realised_fraction=0.8,
        effective_learning_rate=0.8 * rate,
    )


def _run_transaction(model, x, y, config, system):
    return _transactional_realize_functional_step(
        model=model,
        x=x,
        y=y,
        updates=(torch.ones_like(model.weight),),
        proposed_learning_rate=0.2,
        capped_learning_rate=0.1,
        relative_error=0.1,
        system=system,
        selected_relative_damping=1e-3,
        selected_absolute_damping=1e-2,
        config=config,
        epoch=4,
        outer_step_index=2,
        outer_step_global_index=11,
    )


def test_exact_config_file_and_transaction_defaults_are_unchanged() -> None:
    path = "configs/fgd/family_ladder_N1024.yaml"
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == EXACT_CONFIG_SHA256
    exact = load_pipeline_config(path).fgd_approx
    assert exact.transactional_realized_descent is False
    assert exact.transactional_max_retries == 0
    assert exact.certify_functional_lr_cap is None


def test_canonical_matrix_free_config_preserves_the_70_epoch_experiment() -> None:
    base = load_pipeline_config(
        "configs/experiments/mftangent_ladder_N1024_25e.yaml"
    )
    experiment = load_pipeline_config(
        "configs/fgd/family_ladder_matrix_free_N1024.yaml"
    )
    normalized = replace(
        experiment,
        run=replace(experiment.run, name=base.run.name),
        training=replace(experiment.training, epochs=base.training.epochs),
    )
    assert normalized == base
    assert experiment.training.epochs == 70
    assert experiment.run.name == "family_ladder_matrix_free_N1024"
    assert experiment.fgd_approx.transactional_realized_descent is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("transactional_max_retries", -1, "max_retries"),
        ("transactional_backtrack_factor", 1.0, "backtrack_factor"),
        ("transactional_descent_atol", -1e-9, "descent_atol"),
        (
            "transactional_min_predicted_decrease_fraction",
            1.1,
            "min_predicted_decrease_fraction",
        ),
        ("certify_functional_lr_cap", 0.0, "functional_lr_cap"),
    ],
)
def test_transactional_config_rejects_invalid_bounds(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    import yaml

    raw = yaml.safe_load(
        Path("configs/fgd/family_ladder_matrix_free_N1024.yaml").read_text()
    )
    raw["fgd_approx"][field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=message):
        load_pipeline_config(path)


def test_disabled_realization_does_not_compute_new_functional_diagnostics(
    monkeypatch,
) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    x = torch.tensor([[1.0]])
    y = torch.tensor([[0.0]])
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("disabled path evaluated transactional diagnostics")

    monkeypatch.setattr("fgdlib.search.realize.batch_functional_loss", forbidden)
    result = realize_functional_step(
        model,
        x,
        y,
        (torch.ones_like(model.weight),),
        0.1,
        FGDApproxConfig(),
        max_iterations=0,
    )
    assert calls == 0
    assert result.functional_before is None
    assert result.functional_after is None


def test_diagnostics_use_existing_outputs_without_an_extra_forward() -> None:
    class CountingLinear(torch.nn.Linear):
        forwards = 0

        def forward(self, inputs):
            self.forwards += 1
            return super().forward(inputs)

    model = CountingLinear(1, 1, bias=False)
    x = torch.tensor([[1.0]])
    y = torch.tensor([[0.0]])
    config = _transaction_config()
    result = realize_functional_step(
        model,
        x,
        y,
        (torch.ones_like(model.weight),),
        0.1,
        config,
        max_iterations=0,
    )
    # start, predicted_displacement and final are the historical forwards.
    assert model.forwards == 3
    assert result.functional_before is not None
    assert result.functional_after == pytest.approx(result.functional_before)
    assert result.functional_delta == pytest.approx(0.0)


def test_retry_restores_original_reuses_system_and_allocates_one_checkpoint(
    monkeypatch,
) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    original = model.weight.detach().clone()
    x = torch.tensor([[1.0]])
    y = torch.tensor([[0.0]])
    config = _transaction_config()
    system = _system(model, x, y, config)
    starts: list[torch.Tensor] = []
    systems: list[ExactTangentSystem] = []

    def fake_realize(walker, _x, _y, _updates, rate, _config, **kwargs):
        starts.append(walker.weight.detach().clone())
        systems.append(kwargs["system"])
        with torch.no_grad():
            walker.weight.add_(1.0)
        if len(starts) == 1:
            return _realization(1.0, 2.0, rate)
        return _realization(1.0, 0.5, rate)

    real_deepcopy = copy.deepcopy
    copies = 0

    def counted_deepcopy(value):
        nonlocal copies
        copies += 1
        return real_deepcopy(value)

    monkeypatch.setattr(pipeline, "realize_functional_step", fake_realize)
    monkeypatch.setattr(pipeline.copy, "deepcopy", counted_deepcopy)
    outcome = _run_transaction(model, x, y, config, system)

    assert copies == 1
    assert len(starts) == 2
    assert all(torch.equal(start, original) for start in starts)
    assert systems[0].target is systems[1].target is system.target
    assert systems[0].factors is systems[1].factors is system.factors
    assert [record.trial_learning_rate for record in outcome.trials] == [0.1, 0.05]
    assert [record.accepted for record in outcome.trials] == [False, True]
    assert all(record.transactional_trial_count == 2 for record in outcome.trials)
    assert outcome.candidate_model is model
    assert outcome.learning_rate == pytest.approx(0.05)
    assert torch.equal(outcome.base_model.weight, original)
    assert outcome.direction is not None
    reconstructed = outcome.base_model.weight - outcome.learning_rate * outcome.direction[0]
    assert torch.equal(reconstructed, model.weight)


def test_all_retries_fail_and_leave_original_model_bit_exact(monkeypatch) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(math.pi)
    original = model.weight.detach().clone()
    x = torch.tensor([[1.0]])
    y = torch.tensor([[0.0]])
    config = _transaction_config(transactional_max_retries=2)
    system = _system(model, x, y, config)
    starts: list[torch.Tensor] = []

    def always_increases(walker, _x, _y, _updates, rate, _config, **kwargs):
        starts.append(walker.weight.detach().clone())
        with torch.no_grad():
            walker.weight.mul_(3.0)
        return _realization(1.0, 1.1, rate)

    monkeypatch.setattr(pipeline, "realize_functional_step", always_increases)
    outcome = _run_transaction(model, x, y, config, system)

    assert len(starts) == 3
    assert all(torch.equal(start, original) for start in starts)
    assert torch.equal(model.weight, original)
    assert outcome.candidate_model is None
    assert outcome.direction is None
    assert outcome.learning_rate is None
    assert all(record.rejected for record in outcome.trials)


def test_realization_exception_rolls_back_before_propagating(monkeypatch) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    original = model.weight.detach().clone()
    x = torch.tensor([[1.0]])
    y = torch.tensor([[0.0]])
    config = _transaction_config()
    system = _system(model, x, y, config)

    def explode(walker, *args, **kwargs):
        with torch.no_grad():
            walker.weight.add_(7.0)
        raise RuntimeError("realization failed")

    monkeypatch.setattr(pipeline, "realize_functional_step", explode)
    with pytest.raises(RuntimeError, match="realization failed"):
        _run_transaction(model, x, y, config, system)
    assert torch.equal(model.weight, original)


def _record() -> FGDTransactionalTrialRecord:
    return FGDTransactionalTrialRecord(
        epoch=1,
        outer_step_index=0,
        outer_step_global_index=1,
        retry_index=0,
        transactional_trial_count=1,
        proposed_learning_rate=0.2,
        capped_learning_rate=0.1,
        trial_learning_rate=0.1,
        committed_learning_rate=0.1,
        realized_effective_learning_rate=0.08,
        realized_functional_before=1.0,
        realized_functional_after=0.5,
        realized_functional_delta=-0.5,
        predicted_certified_decrease=0.4,
        realized_decrease_ratio=1.25,
        realization_residual_fraction=0.1,
        realization_realized_fraction=0.9,
        realization_iterations=2,
        selected_relative_damping=1e-3,
        selected_absolute_damping=1e-2,
        accepted=True,
        rejected=False,
        rejection_reason=None,
    )


def test_wandb_transaction_logging_is_scalar_only() -> None:
    class Run:
        def __init__(self):
            self.payload = None

        def log(self, payload):
            self.payload = payload

    logger = WandbRunLogger(WandbConfig(enabled=True))
    run = Run()
    logger._run = run
    logger.log_transactional_trial(_record())
    assert run.payload["fgd/realized_functional_delta"] == pytest.approx(-0.5)
    assert run.payload["fgd/transactional_rejection_reason"] == "accepted"
    assert all(
        value is None or isinstance(value, (bool, int, float, str))
        for value in run.payload.values()
    )


def test_result_payload_adds_trials_only_when_present() -> None:
    base = PipelineResult(
        config=PipelineConfig(),
        history=[],
        growth_events=[],
        model=torch.nn.Linear(1, 1),
        device="cpu",
    )
    assert "fgd_outer_steps" not in result_payload(base)
    with_record = replace(base, fgd_outer_steps=[_record()])
    assert result_payload(with_record)["fgd_outer_steps"][0]["accepted"] is True
