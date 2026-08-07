"""Explicit optimization-budget semantics for the nonlinear primary family.

The ladder's nonlinear family spends ``certify_family_inner_steps`` FULL-BATCH
AdamW steps on the certification probe. The nonlinear primary originally spent
the same integer on loader MINIBATCH steps -- on N=1024 with batch 64 that is
64x fewer example-gradients per candidate. Two implementations in which the
same number means a different amount of optimization are not comparable, so
the unit is now named and the spend is reported.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from fgdlib.search.nonlinear import train_nonlinear_candidate
from fgdlib.tangent import FGDApproxConfig, ParametricGDConfig


def _fgd_config() -> FGDApproxConfig:
    return FGDApproxConfig(functional_loss="mse", rel_error_threshold=0.5)


class _LinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _budget_loader(batches: int = 4, batch_size: int = 8) -> DataLoader:
    total = batches * batch_size
    x = torch.linspace(-1.0, 1.0, total, dtype=torch.float64).reshape(total, 1)
    y = torch.sin(x)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _budget_config(unit: str, **overrides) -> ParametricGDConfig:
    base = {
        "optimizer": "adamw",
        "inner_learning_rate": 0.01,
        "inner_steps": (4,),
        "functional_learning_rates": (1.0,),
        "gradient_clip_norm": None,
        "parameter_penalty": 0.0,
        "inner_step_unit": unit,
    }
    base.update(overrides)
    return ParametricGDConfig(**base)


def test_minibatch_unit_spends_exactly_that_many_optimizer_steps() -> None:
    loader = _budget_loader(batches=4, batch_size=8)
    candidate = train_nonlinear_candidate(
        base_model=_LinearModel(),
        train_loader=loader,
        device=torch.device("cpu"),
        functional_learning_rate=1.0,
        inner_steps=6,
        config=_budget_config("minibatch"),
        fgd_config=_fgd_config(),
    )
    assert candidate.optimizer_steps == 6
    assert candidate.examples_seen == 6 * 8
    assert candidate.step_unit == "minibatch"


def test_epoch_unit_spends_complete_loader_passes() -> None:
    loader = _budget_loader(batches=4, batch_size=8)
    candidate = train_nonlinear_candidate(
        base_model=_LinearModel(),
        train_loader=loader,
        device=torch.device("cpu"),
        functional_learning_rate=1.0,
        inner_steps=3,
        config=_budget_config("epoch"),
        fgd_config=_fgd_config(),
    )
    assert candidate.epochs == 3.0
    assert candidate.optimizer_steps == 3 * 4
    # 3 complete passes over 32 examples -- 4x what 3 minibatch steps would be.
    assert candidate.examples_seen == 3 * 32


def test_probe_unit_matches_the_ladder_full_batch_budget() -> None:
    """Ladder parity: N full-batch steps, each consuming the WHOLE probe."""
    loader = _budget_loader(batches=4, batch_size=8)
    probe_x = torch.linspace(-1.0, 1.0, 32, dtype=torch.float64).reshape(32, 1)
    probe_y = torch.sin(probe_x)
    candidate = train_nonlinear_candidate(
        base_model=_LinearModel(),
        train_loader=loader,
        device=torch.device("cpu"),
        functional_learning_rate=1.0,
        inner_steps=10,
        config=_budget_config("probe"),
        fgd_config=_fgd_config(),
        probe=(probe_x, probe_y),
    )
    assert candidate.optimizer_steps == 10
    assert candidate.examples_seen == 10 * 32
    assert candidate.step_unit == "probe"


def test_probe_unit_requires_a_probe() -> None:
    loader = _budget_loader()
    try:
        train_nonlinear_candidate(
            base_model=_LinearModel(),
            train_loader=loader,
            device=torch.device("cpu"),
            functional_learning_rate=1.0,
            inner_steps=2,
            config=_budget_config("probe"),
            fgd_config=_fgd_config(),
        )
    except ValueError as error:
        assert "probe" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError for a missing probe")


def test_candidate_reports_objective_reduction() -> None:
    loader = _budget_loader()
    candidate = train_nonlinear_candidate(
        base_model=_LinearModel(),
        train_loader=loader,
        device=torch.device("cpu"),
        functional_learning_rate=1.0,
        inner_steps=20,
        config=_budget_config("minibatch", inner_learning_rate=0.05),
        fgd_config=_fgd_config(),
    )
    assert candidate.initial_objective is not None
    assert candidate.final_objective is not None
    assert candidate.objective_reduction == (
        candidate.initial_objective - candidate.final_objective
    )
