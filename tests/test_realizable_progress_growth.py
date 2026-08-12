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


def test_two_zeros_abstain_and_the_epsilon_argmin_still_decides(monkeypatch) -> None:
    """The measured freeze: neither endpoint realizes, so the comparison is void.

    MEASURED on run 2kbo8rf4, full MNIST: current_eta=none,
    current_certified_progress=0, candidate_eta=none,
    candidate_certified_progress=0, so delta_progress printed as a signed -0.0
    and the search RETURNED. eps 0.288 certified, rel_err 0.356 was admissible,
    every eta was rejected by the endpoint transaction, and train_loss stayed
    identical to the last digit for 35 consecutive epochs.

    Refusing growth while no step can certify is the deadlock the ordinary path
    asserts against; comparing two zeros is not a refusal, it is no opinion.
    """
    current, candidate = _install_controlled_geometry(monkeypatch, 0.0034, 0.0030)

    def score(_model, _system, _epsilon):
        # No eta anywhere: the state the freeze was measured in.
        return CertifiedRealizableProgress(0.0034, None, 0.0)

    messages: list[str] = []
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
        progress=messages.append,
    )

    assert selected is candidate, "the abstention must not end the search"
    assert result.growths == 1
    assert result.stop_reason != "realizable_progress_not_improved"
    assert any("abstained" in message for message in messages)


def test_a_realizable_base_still_refuses_an_unrealizable_candidate(
    monkeypatch,
) -> None:
    """Abstention is confined to two zeros; a real comparison still refuses.

    When the current model DOES realize a step, "growing does not help" is an
    informative verdict and must keep its terminal refusal -- otherwise the
    criterion would buy a neuron on every outer step.
    """
    current, _ = _install_controlled_geometry(monkeypatch, 0.0034, 0.0030)

    def score(model, _system, _epsilon):
        if model.name == "current":
            return CertifiedRealizableProgress(0.0034, 0.01, 1.0)
        return CertifiedRealizableProgress(0.0030, None, 0.0)

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


def test_the_abstention_buys_one_increment_and_hands_the_turn_back(
    monkeypatch,
) -> None:
    """It unfreezes; it does not licence an unbounded run of growths."""
    current, candidate = _install_controlled_geometry(monkeypatch, 0.0034, 0.0030)

    _, result = grow_until_certified(
        model=current,
        x=torch.zeros(1, 1),
        y=torch.zeros(1, 1),
        train_loader=(),
        device=torch.device("cpu"),
        config=_config(),
        function_preserving=True,
        realizable_progress=lambda *_: CertifiedRealizableProgress(0.0034, None, 0.0),
        layer_bottlenecks=lambda _model: [2.0],
    )

    assert result.growths == 1


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
