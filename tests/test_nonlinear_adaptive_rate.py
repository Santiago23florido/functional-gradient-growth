"""The functional rate must be derived from the eps that was measured.

A well-fitted clone realises ``eta* ~ eta_f``, while Lemma 3.5 admits only
``eta <= 2(1 - 2 eps)/(L_s(1 + 2 eps))``. That bound depends on the ``eps`` the
clone happens to reach, which is not knowable before training it, so a fixed
grid of functional rates cannot be placed correctly in advance -- MEASURED on
N=1024, ``eta_f`` 0.25 reaches eps 0.4045 (bound 0.1056) while ``eta_f``
0.0625 reaches eps 0.4779 (bound 0.0215): the target moves as fast as the
guess does.

The retry is a fixed-point iteration seeded from the MOST RECENT measurement.
Seeding it from the best eps seen overstates the bound, because the smallest
eps belongs to the LARGEST rate.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fgdlib.tangent import ParametricGDConfig, theoretical_learning_rate_upper_bound
from stable_tiny import pipeline

from tests.test_nonlinear_primary_family import _tiny_pipeline_config


def test_adaptive_rate_retries_must_be_non_negative() -> None:
    ParametricGDConfig(adaptive_rate_retries=0).validate()
    ParametricGDConfig(adaptive_rate_retries=3).validate()
    with pytest.raises(ValueError, match="adaptive_rate_retries"):
        ParametricGDConfig(adaptive_rate_retries=-1).validate()


def _record_rates(monkeypatch, *, retries: int, relative_errors):
    """Run the search with stubbed candidates and collect the rates tried."""
    tried: list[float] = []
    errors = iter(relative_errors)
    base = torch.nn.Linear(1, 1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.zeros(2, 1), torch.zeros(2, 1)),
        batch_size=1,
    )

    def fake_train(**kwargs):
        tried.append(kwargs["functional_learning_rate"])
        return pipeline.NonlinearCandidate(
            model=base,
            functional_learning_rate=kwargs["functional_learning_rate"],
            inner_steps=1,
            batches_seen=1,
            final_objective=1.0,
            sensor_valid=True,
            training_seconds=0.0,
        )

    def fake_stream(**kwargs):
        try:
            relative_error = next(errors)
        except StopIteration:
            relative_error = 0.9
        return pipeline.NonlinearCertificateStats(
            dot_product=0.0,
            displacement_sq_norm=1.0,
            gradient_sq_norm=1.0,
            cosine=0.1,
            relative_error=relative_error,
            base_loss=1.0,
            candidate_loss=1.0,
            batches_seen=1,
            sensor_valid=True,
            certified=False,
            certification_seconds=0.0,
        )

    monkeypatch.setattr(pipeline, "train_nonlinear_candidate", fake_train)
    monkeypatch.setattr(pipeline, "stream_nonlinear_certificate", fake_stream)

    base_config = _tiny_pipeline_config(threshold=0.5)
    config = replace(
        base_config,
        parametric_gd=replace(
            base_config.parametric_gd,
            functional_learning_rates=(1.0,),
            inner_steps=(1,),
            certificate_split="validation",
            adaptive_rate_retries=retries,
        ),
    )
    pipeline._search_nonlinear_primary_candidate(
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
    return tried, config


def test_no_retries_leaves_the_fixed_grid_alone(monkeypatch) -> None:
    tried, _ = _record_rates(monkeypatch, retries=0, relative_errors=[0.3])
    assert tried == [1.0]


def test_retry_asks_for_the_rate_the_measured_eps_admits(monkeypatch) -> None:
    tried, config = _record_rates(monkeypatch, retries=1, relative_errors=[0.3, 0.3])
    assert len(tried) == 2
    bound = theoretical_learning_rate_upper_bound(0.3, config.fgd_approx)
    assert tried[1] == pytest.approx(config.fgd_approx.theory_lr_safety * bound)


def test_iteration_is_seeded_from_the_latest_eps_not_the_best(monkeypatch) -> None:
    """Second retry must use the SECOND eps, not the smaller first one."""
    tried, config = _record_rates(
        monkeypatch,
        retries=2,
        relative_errors=[0.30, 0.40, 0.40],
    )
    assert len(tried) == 3
    safety = config.fgd_approx.theory_lr_safety
    from_first = safety * theoretical_learning_rate_upper_bound(0.30, config.fgd_approx)
    from_second = safety * theoretical_learning_rate_upper_bound(
        0.40, config.fgd_approx
    )
    assert tried[1] == pytest.approx(from_first)
    # Seeding from the best (0.30) would repeat from_first; the iteration must
    # contract using the eps the previous retry actually achieved.
    assert tried[2] == pytest.approx(from_second)
    assert tried[2] < tried[1]


def test_a_rate_is_never_tried_twice(monkeypatch) -> None:
    """A retry that lands on an already-tried rate must not re-run it."""
    tried, _ = _record_rates(monkeypatch, retries=3, relative_errors=[0.4, 0.4, 0.4])
    assert len(tried) == len(set(tried))
