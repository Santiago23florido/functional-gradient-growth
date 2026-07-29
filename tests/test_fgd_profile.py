"""The opt-in FGD profiler must be inert unless explicitly enabled."""

from __future__ import annotations

from fgdlib.profile import increment, profiled, reset, snapshot, timed


def test_disabled_profiling_preserves_return_values(monkeypatch) -> None:
    monkeypatch.delenv("FGD_PROFILE", raising=False)
    reset()

    @profiled("minimal_relative_error_seconds")
    def measured(value: object) -> object:
        with timed("damping_factorization_seconds"):
            increment("exact_tangent_system_calls")
        return value

    marker = object()
    assert measured(marker) is marker
    assert snapshot()["exact_tangent_system_calls"] == 0
    assert snapshot()["minimal_relative_error_seconds"] == 0.0
    assert snapshot()["damping_factorization_seconds"] == 0.0


def test_enabled_profiling_records_calls_and_time(monkeypatch) -> None:
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    with timed("exact_tangent_system_seconds"):
        increment("exact_tangent_system_calls")

    values = snapshot()
    assert values["exact_tangent_system_calls"] == 1
    assert values["exact_tangent_system_seconds"] >= 0.0
