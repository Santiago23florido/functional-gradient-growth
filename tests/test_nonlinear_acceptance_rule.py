"""The three acceptance bars, and which one the ladder actually uses.

``certify_parametric_step`` returns on its cosine test alone:
``certify_family_lemma35_rate`` defaults off, and ``grow_until_certified``
commits the returned model with no transactional gate. So the ladder's
nonlinear family accepts on ``eps < 1/2`` and says nothing about step LENGTH.

That is safe *as a fallback*, where the tangent path supplies the bounded
steps. Promoted to the primary family it is not: MEASURED on N=1024 it accepted
a well-aligned step (cos 0.91) at epoch 9 and the training loss went from
0.1532 to 27.6955.

These tests pin which rule enforces what, so the bars cannot drift back into
being implicit.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from fgdlib.search import nonlinear as nonlinear_module
from fgdlib.tangent import ParametricGDConfig
from stable_tiny import pipeline

from tests.test_nonlinear_primary_family import _tiny_pipeline_config


def test_acceptance_rule_is_validated() -> None:
    for rule in ("theory_interval", "measured_descent", "direction_only"):
        ParametricGDConfig(acceptance_rule=rule).validate()
    with pytest.raises(ValueError, match="acceptance_rule"):
        ParametricGDConfig(acceptance_rule="whatever").validate()


def test_ladder_family_certifies_on_the_cosine_alone() -> None:
    """Pins the ladder's bar: no length test, no transactional gate.

    If this ever fails, the ladder changed and the comparison in
    docs/nonlinear_primary_vs_ladder.md needs revisiting.
    """
    from fgdlib.tangent import FGDApproxConfig

    config = FGDApproxConfig(functional_loss="mse", rel_error_threshold=0.5)
    assert config.certify_family_lemma35_rate is False


def _run_with_rule(monkeypatch, rule: str, *, relative_error: float, descends: bool):
    """Drive the search with one stubbed candidate under the given rule."""
    base = torch.nn.Linear(1, 1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.zeros(2, 1), torch.zeros(2, 1)),
        batch_size=1,
    )
    moved = copy.deepcopy(base)
    with torch.no_grad():
        moved.bias.add_(1.0)

    # eta* = dot/grad_sq = 5.0 -- far outside any Lemma 3.5 interval, which is
    # the "well aligned but far too long" step that diverged in the real run.
    stats = nonlinear_module.NonlinearCertificateStats(
        dot_product=5.0,
        displacement_sq_norm=25.0,
        gradient_sq_norm=1.0,
        cosine=1.0,
        relative_error=relative_error,
        base_loss=1.0,
        candidate_loss=(0.5 if descends else 2.0),
        batches_seen=1,
        sensor_valid=True,
        certified=relative_error < 0.5,
        certification_seconds=0.0,
    )
    # Mirror what the real search does: an uncertified direction is inadmissible
    # before any acceptance rule is consulted.
    step = nonlinear_module.InterpolatedStep(
        1.0,
        moved,
        stats,
        None if stats.certified else "relative_error_above_threshold",
    )

    monkeypatch.setattr(
        pipeline,
        "train_nonlinear_candidate",
        lambda **kwargs: nonlinear_module.NonlinearCandidate(
            model=moved,
            functional_learning_rate=1.0,
            inner_steps=1,
            batches_seen=1,
            final_objective=1.0,
            sensor_valid=True,
            training_seconds=0.0,
        ),
    )
    monkeypatch.setattr(pipeline, "stream_nonlinear_certificate", lambda **k: stats)
    monkeypatch.setattr(
        pipeline,
        "search_interpolated_step",
        lambda **k: ((step if step.rejection_reason is None else None), (step,)),
    )
    # all_conditions_valid ANDs in learning_rate_interval_valid, which for this
    # over-long step (eta* = 5.0) is False. Modelling that faithfully is the
    # whole point: a stub that returned all_conditions_valid=True whenever the
    # step descended hid the fact that "measured_descent" was re-imposing the
    # Lemma 3.5 interval through the shared gate.
    monkeypatch.setattr(
        pipeline,
        "_certify_fgd_candidate",
        lambda **k: pipeline._FGDTrial(
            model=moved,
            epoch_result=k["epoch_result"],
            certificate=k["certificate"],
            theory_state=k["theory_state"],
            validation_functional_loss=1.0,
            loss_descent_valid=descends,
            stationary_bound=None,
            stationary_bound_valid=None,
            global_bound=None,
            global_bound_valid=None,
            global_contraction=None,
            # eta* = 5.0 sits outside the Lemma 3.5 interval, so the shared
            # gate's conjunction is False EVEN WHEN the step descends.
            all_conditions_valid=False,
        ),
    )

    base_config = _tiny_pipeline_config(threshold=0.5)
    config = replace(
        base_config,
        parametric_gd=replace(
            base_config.parametric_gd,
            functional_learning_rates=(1.0,),
            inner_steps=(1,),
            certificate_split="validation",
            acceptance_rule=rule,
        ),
    )
    return pipeline._search_nonlinear_primary_candidate(
        base_model=base,
        train_loader=loader,
        validation_loader=loader,
        test_loader=loader,
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


def test_direction_only_accepts_an_over_long_step(monkeypatch) -> None:
    """The ladder's bar: aligned is enough, length is not examined."""
    result = _run_with_rule(
        monkeypatch, "direction_only", relative_error=0.3, descends=False
    )
    assert result.accepted is not None
    assert result.committed_alpha == 1.0


def test_measured_descent_rejects_a_step_that_does_not_descend(monkeypatch) -> None:
    result = _run_with_rule(
        monkeypatch, "measured_descent", relative_error=0.3, descends=False
    )
    assert result.accepted is None


def test_measured_descent_accepts_an_over_long_step_that_descends(
    monkeypatch,
) -> None:
    """No theoretical length bound -- only real, measured descent."""
    result = _run_with_rule(
        monkeypatch, "measured_descent", relative_error=0.3, descends=True
    )
    assert result.accepted is not None


def test_theory_interval_rejects_the_over_long_step_even_when_it_descends(
    monkeypatch,
) -> None:
    """eta* = 5.0 is outside the interval eps 0.3 admits, so it is refused."""
    result = _run_with_rule(
        monkeypatch, "theory_interval", relative_error=0.3, descends=True
    )
    assert result.accepted is None


def test_no_rule_accepts_an_uncertified_direction(monkeypatch) -> None:
    for rule in ("direction_only", "measured_descent", "theory_interval"):
        result = _run_with_rule(
            monkeypatch, rule, relative_error=0.75, descends=True
        )
        assert result.accepted is None, rule
