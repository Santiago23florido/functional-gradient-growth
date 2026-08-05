"""Focused tests for the standalone nonlinear primary approximation family."""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fgdlib import tangent
from fgdlib.profile import reset, snapshot
from fgdlib.search import certify, linearization, realize
from fgdlib.search import damping as damping_search
from fgdlib.search import nonlinear as nonlinear_module
from fgdlib.search.nonlinear import (
    NonlinearCandidate,
    NonlinearCertificateStats,
    scale_parameter_displacement,
    stream_nonlinear_certificate,
    train_nonlinear_candidate,
)
from fgdlib.search.schedule import GrowthScheduleConfig
from fgdlib.tangent import (
    FGDApproxConfig,
    FGDApproxEpochResult,
    FGDOutputRelError,
    FGDValidationCertificate,
    ParametricGDConfig,
    validate_family_order,
)
from stable_tiny import pipeline
from stable_tiny.wandb_logging import WandbConfig, WandbRunLogger


class _Zero(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)


class _NegativeInput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -x[:, :1]


def _certificate_loader(delta: tuple[float, float]) -> DataLoader:
    x = torch.tensor(delta, dtype=torch.float64).reshape(2, 1)
    # base output is zero, hence r = 2(0-y) = (1, 0).
    y = torch.tensor((-0.5, 0.0), dtype=torch.float64).reshape(2, 1)
    return DataLoader(TensorDataset(x, y), batch_size=1, shuffle=False)


def _tiny_pipeline_config(*, epochs: int = 1, threshold: float = 0.0):
    return pipeline.PipelineConfig(
        data=pipeline.DataConfig(
            kind="smooth_sin",
            in_features=2,
            out_features=1,
            train_batches=2,
            validation_batches=2,
            test_batches=2,
            batch_size=8,
            active_features=2,
        ),
        model=pipeline.ModelConfig(
            hidden_size=2,
            number_hidden_layers=2,
            model_seed=0,
        ),
        training=pipeline.TrainingConfig(
            method="fgd_approx",
            epochs=epochs,
            device="cpu",
            log_every=1,
        ),
        fgd_approx=FGDApproxConfig(
            family_order=("nonlinear",),
            rel_error_threshold=threshold,
            local_acceptance_conditions=True,
            growth_where="expressivity_bottleneck",
            growth_selection="unified_expansion",
            tiny_maximum_added_neurons=1,
            certify_max_growths=max(1, epochs),
        ),
        parametric_gd=ParametricGDConfig(
            optimizer="adamw",
            inner_learning_rate=0.001,
            inner_steps=(1,),
            functional_learning_rates=(0.5,),
            gradient_clip_norm=1.0,
        ),
        growth_schedule=GrowthScheduleConfig(enabled=True),
    )


def test_family_order_accepts_standalone_nonlinear_and_legacy_ladders() -> None:
    validate_family_order(("nonlinear",))
    validate_family_order(("tangent",))
    validate_family_order(
        ("tangent", "rkhs_head", "parametric_gd", "parametric_descent")
    )


def test_dedicated_n1024_yaml_selects_nonlinear_2x2x2() -> None:
    config = pipeline.load_pipeline_config(
        Path("configs/fgd/nonlinear_family_ladder_N1024.yaml")
    )
    assert config.fgd_approx.family_order == ("nonlinear",)
    assert config.model.hidden_size == 2
    assert config.model.number_hidden_layers == 3
    assert config.data.train_seed == 0
    assert config.parametric_gd.optimizer == "adamw"
    assert config.parametric_gd.inner_steps == (16, 64, 256)
    assert config.parametric_gd.certification_batches is None


@pytest.mark.parametrize(
    "family_order",
    [
        ("tangent", "nonlinear"),
        ("nonlinear", "tangent"),
        ("nonlinear", "parametric_gd"),
        ("nonlinear", "rkhs_head"),
    ],
)
def test_family_order_rejects_nonlinear_combinations(family_order) -> None:
    with pytest.raises(ValueError, match="standalone primary family"):
        validate_family_order(family_order)


def test_nonlinear_certification_batch_budget_must_be_positive_or_null() -> None:
    ParametricGDConfig(certification_batches=None).validate()
    ParametricGDConfig(certification_batches=2).validate()
    with pytest.raises(ValueError, match="certification_batches"):
        ParametricGDConfig(certification_batches=0).validate()


def test_nonlinear_pipeline_requires_adamw_and_expressivity_growth() -> None:
    base = _tiny_pipeline_config(epochs=0)
    with pytest.raises(ValueError, match="optimizer='adamw'"):
        pipeline.run_pipeline(
            replace(
                base,
                parametric_gd=replace(base.parametric_gd, optimizer="sgd"),
            ),
            progress=None,
        )
    with pytest.raises(ValueError, match="expressivity_bottleneck"):
        pipeline.run_pipeline(
            replace(
                base,
                fgd_approx=replace(base.fgd_approx, growth_where="rank_ceiling"),
            ),
            progress=None,
        )


def test_nonlinear_training_iterates_minibatches_without_materializing_loader() -> None:
    class RecordingLinear(torch.nn.Linear):
        def __init__(self) -> None:
            super().__init__(2, 1)
            self.batch_sizes: list[int] = []

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(inputs.shape[0])
            return super().forward(inputs)

    torch.manual_seed(3)
    base = RecordingLinear()
    x = torch.randn(6, 2)
    y = torch.randn(6, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    candidate = train_nonlinear_candidate(
        base_model=base,
        train_loader=loader,
        device=torch.device("cpu"),
        functional_learning_rate=0.5,
        inner_steps=3,
        config=ParametricGDConfig(
            optimizer="adamw",
            inner_learning_rate=0.01,
            inner_steps=(3,),
            functional_learning_rates=(0.5,),
        ),
        fgd_config=FGDApproxConfig(family_order=("nonlinear",)),
    )
    assert candidate.sensor_valid
    assert candidate.model is not None
    assert candidate.batches_seen == 3
    assert base.batch_sizes == [2, 2, 2]


def test_streaming_certificate_matches_direct_concatenation() -> None:
    loader = _certificate_loader((1.0, 0.25))
    stats = stream_nonlinear_certificate(
        base_model=_Zero(),
        candidate_model=_NegativeInput(),
        certification_loader=loader,
        device=torch.device("cpu"),
        config=FGDApproxConfig(family_order=("nonlinear",)),
    )
    delta = torch.tensor((1.0, 0.25), dtype=torch.float64)
    residual = torch.tensor((1.0, 0.0), dtype=torch.float64)
    cosine = float(torch.dot(delta, residual) / (delta.norm() * residual.norm()))
    relative_error = math.sqrt(1.0 - cosine**2)
    assert stats.batches_seen == 2
    assert stats.cosine == pytest.approx(cosine)
    assert stats.relative_error == pytest.approx(relative_error)


def test_strict_nonlinear_certificate_rejects_exactly_one_half() -> None:
    stats = stream_nonlinear_certificate(
        base_model=_Zero(),
        candidate_model=_NegativeInput(),
        certification_loader=_certificate_loader((math.sqrt(3.0) / 2.0, 0.5)),
        device=torch.device("cpu"),
        config=FGDApproxConfig(family_order=("nonlinear",)),
    )
    assert stats.relative_error == pytest.approx(0.5)
    assert not stats.certified


def test_nonlinear_certificate_accepts_below_half() -> None:
    stats = stream_nonlinear_certificate(
        base_model=_Zero(),
        candidate_model=_NegativeInput(),
        certification_loader=_certificate_loader((1.0, 0.1)),
        device=torch.device("cpu"),
        config=FGDApproxConfig(family_order=("nonlinear",)),
    )
    assert stats.relative_error is not None and stats.relative_error < 0.5
    assert stats.certified


def test_nonlinear_certificate_rejects_above_half() -> None:
    stats = stream_nonlinear_certificate(
        base_model=_Zero(),
        candidate_model=_NegativeInput(),
        certification_loader=_certificate_loader((0.5, 1.0)),
        device=torch.device("cpu"),
        config=FGDApproxConfig(family_order=("nonlinear",)),
    )
    assert stats.relative_error is not None and stats.relative_error > 0.5
    assert not stats.certified


def test_nonlinear_certificate_rejects_zero_displacement() -> None:
    stats = stream_nonlinear_certificate(
        base_model=_Zero(),
        candidate_model=_Zero(),
        certification_loader=_certificate_loader((1.0, 0.0)),
        device=torch.device("cpu"),
        config=FGDApproxConfig(family_order=("nonlinear",)),
    )
    assert not stats.sensor_valid
    assert stats.relative_error is None
    assert not stats.certified


def test_nonlinear_certificate_respects_streaming_batch_cap() -> None:
    stats = stream_nonlinear_certificate(
        base_model=_Zero(),
        candidate_model=_NegativeInput(),
        certification_loader=_certificate_loader((1.0, 0.25)),
        device=torch.device("cpu"),
        config=FGDApproxConfig(family_order=("nonlinear",)),
        max_batches=1,
    )
    assert stats.batches_seen == 1


def test_nonlinear_certificate_rejects_nonpositive_alignment() -> None:
    stats = stream_nonlinear_certificate(
        base_model=_Zero(),
        candidate_model=_NegativeInput(),
        certification_loader=_certificate_loader((-1.0, 0.0)),
        device=torch.device("cpu"),
        config=FGDApproxConfig(family_order=("nonlinear",)),
    )
    assert stats.cosine == pytest.approx(-1.0)
    assert stats.relative_error == pytest.approx(1.0)
    assert not stats.certified


def test_nonlinear_certificate_rejects_nonfinite_outputs() -> None:
    stats = stream_nonlinear_certificate(
        base_model=_Zero(),
        candidate_model=_NegativeInput(),
        certification_loader=_certificate_loader((float("inf"), 0.0)),
        device=torch.device("cpu"),
        config=FGDApproxConfig(family_order=("nonlinear",)),
    )
    assert not stats.sensor_valid
    assert not stats.certified
    assert stats.relative_error is None


def test_scaled_commit_uses_certified_rate_not_full_candidate() -> None:
    base = torch.nn.Linear(1, 1, bias=False)
    candidate = copy.deepcopy(base)
    with torch.no_grad():
        base.weight.zero_()
        candidate.weight.fill_(2.0)
    committed = scale_parameter_displacement(
        base_model=base,
        candidate_model=candidate,
        rate=0.25,
    )
    assert float(committed.weight.detach()) == pytest.approx(0.5)
    assert float(committed.weight.detach()) != pytest.approx(
        float(candidate.weight.detach())
    )


def test_nonlinear_search_certifies_on_validation_loader_with_configured_cap(
    monkeypatch,
) -> None:
    base = torch.nn.Linear(1, 1)
    train_loader = DataLoader(
        TensorDataset(torch.zeros(2, 1), torch.zeros(2, 1)),
        batch_size=1,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.ones(2, 1)),
        batch_size=1,
    )
    generated = NonlinearCandidate(
        model=copy.deepcopy(base),
        functional_learning_rate=0.5,
        inner_steps=1,
        batches_seen=1,
        final_objective=1.0,
        sensor_valid=True,
        training_seconds=0.0,
    )
    captured = {}

    def stream(**kwargs):
        captured.update(kwargs)
        return NonlinearCertificateStats(
            dot_product=0.0,
            displacement_sq_norm=1.0,
            gradient_sq_norm=1.0,
            cosine=0.0,
            relative_error=1.0,
            base_loss=1.0,
            candidate_loss=1.0,
            batches_seen=1,
            sensor_valid=True,
            certified=False,
            certification_seconds=0.0,
        )

    monkeypatch.setattr(
        pipeline, "train_nonlinear_candidate", lambda **kwargs: generated
    )
    monkeypatch.setattr(pipeline, "stream_nonlinear_certificate", stream)
    # The streaming cap is a property of the VALIDATION certificate path, so
    # this test asks for that split explicitly rather than relying on a
    # default (which is now "train", matching the ladder).
    config = replace(
        _tiny_pipeline_config(threshold=0.5),
        parametric_gd=replace(
            _tiny_pipeline_config().parametric_gd,
            certification_batches=1,
            certificate_split="validation",
        ),
    )
    pipeline._search_nonlinear_primary_candidate(
        base_model=base,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=validation_loader,
        loss_function=torch.nn.MSELoss(),
        device=torch.device("cpu"),
        accuracy_tolerance=0.1,
        config=config,
        classification=False,
        theory_state=pipeline._FGDTheoryState(0, None, None, None, 1.0, 1.0),
        initial_functional_gap=1.0,
        theory_loss_star=0.0,
        progress=None,
    )
    assert captured["certification_loader"] is validation_loader
    assert captured["max_batches"] == 1


def test_required_validation_descent_rejects_nonlinear_candidate(
    monkeypatch,
) -> None:
    torch.manual_seed(0)
    base = torch.nn.Linear(1, 1)
    x = torch.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    with torch.no_grad():
        y = base(x).detach()
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    moved = copy.deepcopy(base)
    with torch.no_grad():
        moved.bias.add_(10.0)

    generated = NonlinearCandidate(
        model=moved,
        functional_learning_rate=0.5,
        inner_steps=1,
        batches_seen=4,
        final_objective=0.0,
        sensor_valid=True,
        training_seconds=0.0,
    )
    stats = NonlinearCertificateStats(
        dot_product=1.0,
        displacement_sq_norm=1.0,
        gradient_sq_norm=1.0,
        cosine=1.0,
        relative_error=0.0,
        base_loss=0.0,
        candidate_loss=0.0,
        batches_seen=4,
        sensor_valid=True,
        certified=True,
        certification_seconds=0.0,
    )
    monkeypatch.setattr(
        pipeline,
        "train_nonlinear_candidate",
        lambda **kwargs: generated,
    )
    monkeypatch.setattr(
        pipeline,
        "stream_nonlinear_certificate",
        lambda **kwargs: stats,
    )
    # The committed step now comes from the interpolation search, and each
    # alpha carries its OWN re-measured certificate. Hand it one certified
    # interpolation whose realized eta* sits inside the Lemma 3.5 interval,
    # so the only thing left that can reject it is the transactional descent
    # condition on validation -- which is what this test is about.
    admissible = nonlinear_module.InterpolatedStep(
        alpha=1.0,
        model=moved,
        stats=replace(stats, dot_product=0.5, displacement_sq_norm=0.25),
        rejection_reason=None,
    )
    monkeypatch.setattr(
        pipeline,
        "search_interpolated_step",
        lambda **kwargs: (admissible, (admissible,)),
    )
    config = replace(
        _tiny_pipeline_config(epochs=1, threshold=0.5),
        parametric_gd=replace(
            _tiny_pipeline_config().parametric_gd,
            inner_steps=(1,),
            functional_learning_rates=(0.5,),
            certificate_split="validation",
        ),
    )
    theory_state = pipeline._FGDTheoryState(
        epoch_count=0,
        min_gradient_sq_norm=None,
        min_positive_learning_rate=None,
        min_descent_coefficient=None,
        global_contraction_product=1.0,
        previous_validation_functional_loss=0.0,
    )
    result = pipeline._search_nonlinear_primary_candidate(
        base_model=base,
        train_loader=loader,
        validation_loader=loader,
        test_loader=loader,
        loss_function=torch.nn.MSELoss(),
        device=torch.device("cpu"),
        accuracy_tolerance=0.1,
        config=config,
        classification=False,
        theory_state=theory_state,
        initial_functional_gap=0.0,
        theory_loss_star=0.0,
        progress=None,
    )
    assert result.accepted is None
    assert result.last_trial is not None
    assert not result.last_trial.loss_descent_valid


def _forbid_tangent_operations(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("tangent operation called in nonlinear mode")

    for name in (
        "build_projection_probe",
        "_materialize_dataset",
        "_bounded_probe_batches",
        "_estimate_certification_rank",
        "certificate_from_projection_stats",
        "evaluate_fgd_validation_certificate",
        "grow_until_certified",
        "select_projection_damping",
        "realize_functional_step",
        "measure_direction_projection",
        "_compute_tangent_projection_step",
        "exact_relative_error",
        "certified_linear_learning_rate",
        "train_one_epoch_fgd_approx",
        "_probe_fgd_growth",
        "_growth_reduces_lookahead_epsilon",
        "select_tiny_growth_layer_index",
        "rank_candidates_by_certified_gain",
    ):
        monkeypatch.setattr(pipeline, name, forbidden)
    for module, names in (
        (
            tangent,
            (
                "exact_tangent_system",
                "exact_relative_error",
                "compute_tangent_projection_error",
                "evaluate_fgd_validation_certificate",
                "measure_direction_projection",
                "_compute_tangent_projection_step",
                "_solve_tangent_projection",
                "jacrev",
                "jvp",
            ),
        ),
        (certify, ("exact_relative_error", "grow_until_certified")),
        (damping_search, ("select_projection_damping",)),
        (realize, ("realize_functional_step",)),
        (linearization, ("certified_linear_learning_rate",)),
    ):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)


def test_nonlinear_mode_never_calls_tangent_operations_and_grows(
    monkeypatch,
) -> None:
    _forbid_tangent_operations(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "compute_expressivity_bottlenecks",
        lambda model, loader, device, config: [1.0, 0.5],
    )
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    result = pipeline.run_pipeline(_tiny_pipeline_config(epochs=2), progress=None)
    values = snapshot()
    reset()
    assert len(result.growth_events) == 2
    assert result.history[-1].fgd_approximation_kind == "nonlinear"
    assert result.history[-1].nonlinear_growth_requested
    assert values["nonlinear_ladder_attempts"] == 2
    assert values["nonlinear_failed_ladders"] == 2
    assert values["nonlinear_growth_events"] == 2
    assert values["exact_tangent_system_calls"] == 0
    assert values["tangent_system_calls"] == 0
    assert values["tangent_projection_solve_calls"] == 0
    nonlinear_steps = [entry for entry in result.history if entry.step_type == "FGD"]
    assert nonlinear_steps[0].architecture_widths == (2, 2)
    assert nonlinear_steps[1].architecture_widths == (3, 2)


def test_certified_nonlinear_candidate_prevents_growth(monkeypatch) -> None:
    _forbid_tangent_operations(monkeypatch)
    stats = NonlinearCertificateStats(
        dot_product=1.0,
        displacement_sq_norm=1.0,
        gradient_sq_norm=1.0,
        cosine=1.0,
        relative_error=0.0,
        base_loss=1.0,
        candidate_loss=0.5,
        batches_seen=2,
        sensor_valid=True,
        certified=True,
        certification_seconds=0.0,
    )

    def accepted_search(**kwargs):
        model = kwargs["base_model"]
        theory_state = kwargs["theory_state"]
        output_error = FGDOutputRelError(0.0, 1.0, 1.0, 1.0)
        certificate = FGDValidationCertificate(
            learning_rate_upper_bound=1.0,
            max_valid_learning_rate=0.25,
            learning_rate_interval_valid=True,
            skipped_batches=0,
            relative_error_condition_valid=True,
            gradient_sq_norm=1.0,
            theory_descent_coefficient=0.5,
            relative_error=0.0,
            output_relative_error=output_error,
            sensor_valid=True,
            sensor_invalid_batches=0,
        )
        epoch_result = FGDApproxEpochResult(
            train_loss=0.1,
            train_accuracy=1.0,
            test_loss=0.1,
            test_accuracy=1.0,
            learning_rate=0.25,
            next_learning_rate=0.25,
            learning_rate_upper_bound=1.0,
            learning_rate_interval_valid=True,
            learning_rate_clipped_batches=0,
            skipped_batches=0,
            relative_error_condition_valid=True,
            loss_descent_valid=True,
            loss_non_descent_batches=0,
            gradient_sq_norm=1.0,
            theory_descent_coefficient=0.5,
            min_positive_learning_rate=0.25,
            relative_error=0.0,
            selected_layer_index=None,
            layer_relative_errors=[],
            output_relative_error=output_error,
            sensor_valid=True,
            sensor_invalid_batches=0,
        )
        trial = pipeline._FGDTrial(
            model=model,
            epoch_result=epoch_result,
            certificate=certificate,
            theory_state=replace(
                theory_state,
                epoch_count=theory_state.epoch_count + 1,
                min_gradient_sq_norm=1.0,
                min_positive_learning_rate=0.25,
                min_descent_coefficient=0.5,
            ),
            validation_functional_loss=(
                theory_state.previous_validation_functional_loss - 0.1
            ),
            loss_descent_valid=True,
            stationary_bound=1.0,
            stationary_bound_valid=True,
            global_bound=1.0,
            global_bound_valid=True,
            global_contraction=0.5,
            all_conditions_valid=True,
        )
        generated = NonlinearCandidate(
            model=model,
            functional_learning_rate=0.5,
            inner_steps=1,
            batches_seen=2,
            final_objective=0.1,
            sensor_valid=True,
            training_seconds=0.0,
        )
        return pipeline._NonlinearPrimaryResult(
            accepted=trial,
            last_trial=trial,
            certificate=certificate,
            stats=stats,
            candidate=generated,
            attempts=1,
            candidate_training_seconds=0.0,
            certification_seconds=0.0,
            update_norm=0.1,
        )

    monkeypatch.setattr(
        pipeline,
        "_search_nonlinear_primary_candidate",
        accepted_search,
    )
    monkeypatch.setattr(
        pipeline,
        "_apply_nonlinear_primary_growth",
        lambda **kwargs: pytest.fail("growth followed a certified candidate"),
    )
    result = pipeline.run_pipeline(
        _tiny_pipeline_config(epochs=1, threshold=0.5),
        progress=None,
    )
    assert not result.growth_events
    assert result.history[-1].fgd_candidate_accepted is True


def test_wandb_payload_names_nonlinear_metrics_without_tangent_claim() -> None:
    class StubRun:
        def __init__(self) -> None:
            self.payloads = []

        def log(self, payload) -> None:
            self.payloads.append(payload)

    entry = pipeline.HistoryEntry(
        step=1,
        step_type="FGD",
        train_loss=0.1,
        validation_loss=0.1,
        test_loss=0.1,
        train_accuracy=1.0,
        validation_accuracy=1.0,
        test_accuracy=1.0,
        learning_rate=0.2,
        num_params=25,
        fgd_candidate_accepted=True,
        fgd_approximation_kind="nonlinear",
        nonlinear_functional_learning_rate=0.5,
        nonlinear_inner_steps=40,
        nonlinear_adamw_learning_rate=0.003,
        nonlinear_weight_decay=0.05,
        nonlinear_cosine=math.sqrt(1.0 - 0.3**2),
        nonlinear_relative_error=0.3,
        nonlinear_certificate_valid=True,
        nonlinear_validation_descent_valid=True,
        nonlinear_committed_rate=0.2,
        nonlinear_ladder_attempts=3,
        nonlinear_accepted_steps=1,
        nonlinear_failed_ladders=2,
        nonlinear_growth_events=2,
        architecture_widths=(2, 2, 2),
    )
    logger = WandbRunLogger(WandbConfig(enabled=True))
    stub = StubRun()
    logger._run = stub
    logger.log_history_entry(entry)
    payload = stub.payloads[0]
    assert payload["fgd/approximation_kind"] == "nonlinear"
    assert payload["fgd/family_index"] == 4
    assert payload["fgd/nonlinear_relative_error"] == pytest.approx(0.3)
    assert payload["nonlinear/relative_error"] == pytest.approx(0.3)
    assert payload["nonlinear/full_jacobian_calls"] == 0
    assert payload["nonlinear/tangent_system_calls"] == 0
    assert payload["nonlinear/tangent_projection_solves"] == 0
    assert payload["fgd/nonlinear_certificate_valid"] is True
    assert payload["model/architecture_widths"] == [2, 2, 2]
    assert "fgd/tangent_relative_error" not in payload


def test_growth_is_function_preserving_and_uses_configured_argmax(
    monkeypatch,
) -> None:
    config = _tiny_pipeline_config()
    device = torch.device("cpu")
    train_loader, _, _ = pipeline.build_dataloaders(config, device)
    model = pipeline.build_model(config, device)
    probe_x, _ = next(iter(train_loader))
    with torch.no_grad():
        before = model(probe_x).clone()
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
    with torch.no_grad():
        original_after = model(probe_x)
        grown_after = outcome.model(probe_x)
    assert outcome.result is not None
    assert outcome.layer_index == 1
    assert pipeline._architecture_widths(model) == (2, 2)
    assert pipeline._architecture_widths(outcome.model) == (2, 3)
    torch.testing.assert_close(before, original_after, rtol=0.0, atol=0.0)
    torch.testing.assert_close(before, grown_after, rtol=1e-6, atol=1e-6)
    with torch.no_grad():
        for batch_x, _ in train_loader:
            torch.testing.assert_close(
                model(batch_x),
                outcome.model(batch_x),
                rtol=0.0,
                atol=config.fgd_approx.growth_preservation_tolerance,
            )


def test_failed_preservation_check_discards_grown_clone(monkeypatch) -> None:
    config = _tiny_pipeline_config()
    device = torch.device("cpu")
    train_loader, _, _ = pipeline.build_dataloaders(config, device)
    model = pipeline.build_model(config, device)
    widths_before = pipeline._architecture_widths(model)
    monkeypatch.setattr(
        pipeline,
        "compute_expressivity_bottlenecks",
        lambda model, loader, device, config: [2.0, 1.0],
    )
    monkeypatch.setattr(
        pipeline,
        "_stream_max_function_drift",
        lambda **kwargs: config.fgd_approx.growth_preservation_tolerance * 2.0,
    )
    outcome = pipeline._apply_nonlinear_primary_growth(
        model=model,
        train_loader=train_loader,
        device=device,
        config=config,
        epoch=1,
        progress=None,
    )
    assert outcome.model is None
    assert outcome.result is None
    assert outcome.preservation_valid is False
    assert pipeline._architecture_widths(model) == widths_before


def test_failed_ladder_retries_after_growth(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "compute_expressivity_bottlenecks",
        lambda model, loader, device, config: [1.0, 0.5],
    )
    result = pipeline.run_pipeline(
        _tiny_pipeline_config(epochs=2),
        progress=None,
    )
    nonlinear_entries = [
        entry
        for entry in result.history
        if entry.fgd_approximation_kind == "nonlinear" and entry.step_type == "FGD"
    ]
    assert len(nonlinear_entries) == 2
    assert len(result.growth_events) == 2


def test_default_tangent_configuration_does_not_instantiate_nonlinear(
    monkeypatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("nonlinear family instantiated by tangent config")

    monkeypatch.setattr(
        pipeline,
        "_run_nonlinear_pipeline",
        forbidden,
    )
    legacy_calls = 0
    original_bounded_probe_batches = pipeline._bounded_probe_batches

    def legacy_probe(*args, **kwargs):
        nonlocal legacy_calls
        legacy_calls += 1
        return original_bounded_probe_batches(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_bounded_probe_batches", legacy_probe)
    config = replace(
        _tiny_pipeline_config(epochs=0),
        fgd_approx=FGDApproxConfig(
            family_order=("tangent",),
            probe_batches=1,
            projection_solver="exact",
        ),
    )
    result = pipeline.run_pipeline(config, progress=None)
    assert config.fgd_approx.family_order == ("tangent",)
    assert len(result.history) == 1
    assert legacy_calls == 2
