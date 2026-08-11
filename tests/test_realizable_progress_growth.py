"""Opt-in growth by certified finite-step realizability."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import fgdlib.search.certify as certify
from fgdlib.search.certify import (
    CertifiedRealizableProgress,
    grow_until_certified,
)


class _DummyModel(torch.nn.Module):
    def __init__(self, name: str, parameter_count: int = 1) -> None:
        super().__init__()
        self.name = name
        self.weight = torch.nn.Parameter(torch.zeros(parameter_count))
        self._growable_layers = [object()]


def _config():
    return SimpleNamespace(
        fgd_approx=SimpleNamespace(
            rel_error_threshold=0.5,
            certify_growth_target=None,
            certify_growth_min_gain=0.1,
            certify_growth_min_gain_patience=1,
            max_total_parameters=None,
            eps=1.0e-12,
            certify_family_in_target_band=False,
            certify_adaptive_growth=False,
            certify_exact_block_schur_where=False,
        )
    )


def _install_controlled_geometry(monkeypatch, current_eps: float, candidate_eps: float):
    current = _DummyModel("current", 1)
    candidate = _DummyModel("candidate", 2)
    systems = {"current": object(), "candidate": object()}

    def relative_error_and_system(model, _x, _y, _config):
        epsilon = current_eps if model.name == "current" else candidate_eps
        return epsilon, systems[model.name]

    monkeypatch.setattr(certify, "_relative_error_and_system", relative_error_and_system)
    monkeypatch.setattr(certify, "_grow_clone", lambda *args, **kwargs: candidate)
    return current, candidate


def test_growth_can_win_when_epsilon_is_already_tiny(monkeypatch) -> None:
    """Finite realizability can break the eps~=0 deadlock without forcing."""
    current, candidate = _install_controlled_geometry(monkeypatch, 0.0034, 0.0036)

    def score(model, _system, _epsilon):
        if model.name == "current":
            return CertifiedRealizableProgress(0.0034, None, 0.0)
        return CertifiedRealizableProgress(0.0036, 0.02, 1.25)

    selected, result = grow_until_certified(
        model=current,
        x=torch.zeros(1, 1),
        y=torch.zeros(1, 1),
        train_loader=(),
        device=torch.device("cpu"),
        config=_config(),
        function_preserving=True,
        realizable_progress=score,
        layer_bottlenecks=lambda _model: [2.0],
    )

    assert selected is candidate
    assert result.growths == 1
    assert result.relative_error == pytest.approx(0.0036)
    assert result.stop_reason == "realizable_growth_turn_taken"


def test_growth_is_refused_without_realizable_progress_gain(monkeypatch) -> None:
    """A lower eps alone cannot buy growth under the new opt-in value gate."""
    current, _ = _install_controlled_geometry(monkeypatch, 0.0034, 0.0030)

    def score(model, _system, _epsilon):
        if model.name == "current":
            return CertifiedRealizableProgress(0.0034, 0.01, 1.0)
        return CertifiedRealizableProgress(0.0030, 0.01, 1.0)

    selected, result = grow_until_certified(
        model=current,
        x=torch.zeros(1, 1),
        y=torch.zeros(1, 1),
        train_loader=(),
        device=torch.device("cpu"),
        config=_config(),
        function_preserving=True,
        realizable_progress=score,
        layer_bottlenecks=lambda _model: [2.0],
    )

    assert selected is current
    assert result.growths == 0
    assert result.stop_reason == "realizable_progress_not_improved"


def test_disabled_path_retains_epsilon_selection(monkeypatch) -> None:
    """Without the callback, the historical exact-eps selection is unchanged."""
    current, candidate = _install_controlled_geometry(monkeypatch, 0.6, 0.4)

    selected, result = grow_until_certified(
        model=current,
        x=torch.zeros(1, 1),
        y=torch.zeros(1, 1),
        train_loader=(),
        device=torch.device("cpu"),
        config=_config(),
        function_preserving=True,
    )

    assert selected is candidate
    assert result.trajectory == pytest.approx((0.6, 0.4))
    assert result.stop_reason == "certified"


def test_probe_diagnostic_is_observational(monkeypatch) -> None:
    """Enabling scalar diagnostics changes neither tensors nor loop outcome."""
    current, _ = _install_controlled_geometry(monkeypatch, 0.1, 0.2)
    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[3.0]])
    x_before, y_before = x.clone(), y.clone()

    _, baseline = grow_until_certified(
        model=current,
        x=x,
        y=y,
        train_loader=(),
        device=torch.device("cpu"),
        config=_config(),
    )
    observed = []
    _, diagnosed = grow_until_certified(
        model=current,
        x=x,
        y=y,
        train_loader=(),
        device=torch.device("cpu"),
        config=_config(),
        probe_diagnostic=lambda model, system, epsilon: observed.append(
            (model, system, epsilon)
        ),
    )

    assert diagnosed == baseline
    assert observed and observed[0][0] is current
    assert torch.equal(x, x_before)
    assert torch.equal(y, y_before)
