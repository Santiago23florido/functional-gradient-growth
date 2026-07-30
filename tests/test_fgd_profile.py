"""The opt-in FGD profiler must be inert unless explicitly enabled."""

from __future__ import annotations

import pytest
import torch

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.profile import (
    _COUNTER_FIELDS,
    _REASONS,
    PROFILE_FIELDS,
    fallback,
    increment,
    profiled,
    reset,
    set_max,
    set_value,
    snapshot,
    timed,
)
from fgdlib.tangent import FGDApproxConfig, exact_tangent_system

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP


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
    assert snapshot()["where_scans"] == 0
    assert snapshot()["where_total_seconds"] == 0.0


def test_enabled_profiling_records_calls_and_time(monkeypatch) -> None:
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    with timed("exact_tangent_system_seconds"):
        increment("exact_tangent_system_calls")

    values = snapshot()
    assert values["exact_tangent_system_calls"] == 1
    assert values["exact_tangent_system_seconds"] >= 0.0
    assert values["where_full_candidate_system_calls"] == 0
    assert values["where_winner_mismatch_fallbacks"] == 0


# --------------------------------------------------------------------------
# Phase A tangent instrumentation: setters, classification, and the
# parent/child timer contract.
# --------------------------------------------------------------------------


TANGENT_FIELDS = tuple(
    field for field in PROFILE_FIELDS if field.startswith("tangent_")
)

#: The EXCLUSIVE, mutually disjoint children of tangent_system_total_seconds.
#: streamed_jacobian_seconds is deliberately absent: it is a nested PARENT of
#: several of these, so adding it would double-count (see the contract in
#: _compute_exact_tangent_projection_step's docstring).
TANGENT_CHILD_TIMERS = (
    "tangent_forward_target_seconds",
    "tangent_jacrev_seconds",
    "tangent_jacobian_flatten_seconds",
    "tangent_jacobian_cast_seconds",
    "tangent_analytic_jacobian_seconds",
    "tangent_jtr_seconds",
    "tangent_qr_seconds",
    "tangent_surrogate_seconds",
    "tangent_final_factorization_seconds",
    "tangent_projection_solve_seconds",
    "tangent_sensor_seconds",
)


def test_disabled_profiling_leaves_new_tangent_fields_zero(monkeypatch) -> None:
    monkeypatch.delenv("FGD_PROFILE", raising=False)
    reset()
    for field in TANGENT_FIELDS:
        increment(field, 3)
        set_value(field, 5)
        set_max(field, 7)
        fallback(field, "should-not-be-recorded")
    values = snapshot()
    assert all(values[field] == 0 for field in TANGENT_FIELDS)
    assert _REASONS == {}


def test_set_max_keeps_the_peak_not_the_last_value(monkeypatch) -> None:
    """The three setters have three different jobs and are not
    interchangeable: a peak-shape gauge written with ``set_value`` would
    report the LAST block's shape and one written with ``increment`` would
    report the SUM of every block's."""
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    for value in (10, 40, 25):
        set_max("tangent_peak_jacobian_block_rows", value)
        set_value("tangent_output_row_count", value)
        increment("tangent_sample_chunk_count", value)
    values = snapshot()
    assert values["tangent_peak_jacobian_block_rows"] == 40  # peak
    assert values["tangent_output_row_count"] == 25  # last
    assert values["tangent_sample_chunk_count"] == 75  # sum
    reset()


def test_set_max_never_decreases(monkeypatch) -> None:
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    set_max("tangent_qr_input_rows", 100)
    set_max("tangent_qr_input_rows", 1)
    assert snapshot()["tangent_qr_input_rows"] == 100
    reset()


def test_fallback_records_counter_and_reason(monkeypatch) -> None:
    """A fallback that is counted but not explained tells you a run degraded
    without telling you why, which is the same as not knowing."""
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    fallback("tangent_unsupported_structure_fallbacks", "model_has_buffers")
    fallback("tangent_unsupported_structure_fallbacks", "model_not_in_eval_mode")
    fallback("tangent_unsupported_structure_fallbacks", "model_has_buffers")
    assert snapshot()["tangent_unsupported_structure_fallbacks"] == 3
    assert _REASONS["tangent_unsupported_structure_fallbacks"] == {
        "model_has_buffers",
        "model_not_in_eval_mode",
    }
    reset()
    assert _REASONS == {}
    assert snapshot()["tangent_unsupported_structure_fallbacks"] == 0


def test_new_fields_are_classified_consistently() -> None:
    """Integer-shape gauges belong in ``_COUNTER_FIELDS`` so ``snapshot``
    renders them as ints even though they are written with
    ``set_max``/``set_value``; the two genuinely fractional gauges must stay
    out of it."""
    integer_gauges = {
        "tangent_qr_input_rows",
        "tangent_qr_input_columns",
        "tangent_peak_jacobian_block_rows",
        "tangent_peak_jacobian_block_columns",
        "tangent_parameter_count",
        "tangent_output_row_count",
    }
    assert integer_gauges <= _COUNTER_FIELDS
    assert "tangent_condition_estimate" not in _COUNTER_FIELDS
    assert "tangent_oracle_max_relative_error" not in _COUNTER_FIELDS

    seconds = {field for field in TANGENT_FIELDS if field.endswith("_seconds")}
    assert seconds and not (seconds & _COUNTER_FIELDS)

    counters = {
        field
        for field in TANGENT_FIELDS
        if field.endswith(
            ("_calls", "_count", "_fallbacks", "_surrogates", "_verifications")
        )
    }
    assert counters <= _COUNTER_FIELDS


def test_snapshot_renders_counter_fields_as_ints(monkeypatch) -> None:
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    set_max("tangent_qr_input_rows", 12.0)
    set_value("tangent_condition_estimate", 1234.5)
    values = snapshot()
    assert isinstance(values["tangent_qr_input_rows"], int)
    assert isinstance(values["tangent_condition_estimate"], float)
    assert values["tangent_condition_estimate"] == 1234.5
    reset()


def _streamed_profile(monkeypatch, backend: str) -> dict[str, float]:
    monkeypatch.setenv("FGD_PROFILE", "1")
    monkeypatch.setenv("FGD_TANGENT_BACKEND", backend)
    reset()
    torch.manual_seed(0)
    model = GrowingMLP(
        in_features=6,
        out_features=3,
        hidden_size=4,
        number_hidden_layers=2,
        device=torch.device("cpu"),
    )
    model.eval()
    torch.manual_seed(5)
    x = torch.randn(32, 6)
    y = torch.randn(32, 3)
    config = FGDApproxConfig(
        projection_solver="exact",
        functional_loss="mse",
        certify_stream_gram=True,
        certify_stream_chunk=8,
    )
    assert exact_tangent_system(model, x, y, config) is not None
    values = snapshot()
    reset()
    return values


@pytest.mark.parametrize("backend", ["legacy", "optimized"])
def test_tangent_subtimers_do_not_exceed_the_total(monkeypatch, backend: str) -> None:
    values = _streamed_profile(monkeypatch, backend)
    total = values["tangent_system_total_seconds"]
    assert total > 0.0
    for field in TANGENT_CHILD_TIMERS:
        assert values[field] <= total, field
    assert values["streamed_jacobian_seconds"] <= total
    assert sum(values[field] for field in TANGENT_CHILD_TIMERS) <= total * 1.05


@pytest.mark.parametrize("backend", ["legacy", "optimized"])
def test_streamed_jacobian_timer_contains_its_children(
    monkeypatch, backend: str
) -> None:
    """``streamed_jacobian_seconds`` is an INCLUSIVE parent of the jacobian /
    jtr / qr / surrogate timers -- never a sibling. Summing it with them is
    the double count the contract warns about, so the relation pinned here is
    containment, not addition."""
    values = _streamed_profile(monkeypatch, backend)
    nested = (
        "tangent_jacrev_seconds",
        "tangent_jacobian_flatten_seconds",
        "tangent_jacobian_cast_seconds",
        "tangent_analytic_jacobian_seconds",
        "tangent_jtr_seconds",
        "tangent_qr_seconds",
        "tangent_surrogate_seconds",
        "tangent_final_factorization_seconds",
    )
    parent = values["streamed_jacobian_seconds"]
    assert parent > 0.0
    assert sum(values[field] for field in nested) <= parent * 1.05


@pytest.mark.parametrize("backend", ["legacy", "optimized"])
def test_the_two_jacobian_timer_groups_are_mutually_exclusive(
    monkeypatch, backend: str
) -> None:
    """A zero in one group is how a profile says which backend ran; if both
    were non-zero the profile could not answer that question at all."""
    values = _streamed_profile(monkeypatch, backend)
    jacrev_group = (
        values["tangent_jacrev_seconds"]
        + values["tangent_jacobian_flatten_seconds"]
        + values["tangent_jacobian_cast_seconds"]
    )
    analytic_group = values["tangent_analytic_jacobian_seconds"]
    if backend == "legacy":
        assert jacrev_group > 0.0 and analytic_group == 0.0
        assert values["tangent_analytic_jacobian_calls"] == 0
    else:
        assert analytic_group > 0.0 and jacrev_group == 0.0
        assert values["tangent_analytic_jacobian_calls"] > 0


@pytest.mark.parametrize("backend", ["legacy", "optimized"])
def test_backend_call_counters_do_not_double_count(monkeypatch, backend: str) -> None:
    values = _streamed_profile(monkeypatch, backend)
    assert values["tangent_system_calls"] == 1
    assert (
        values["tangent_backend_legacy_calls"]
        + values["tangent_backend_optimized_calls"]
        == values["tangent_system_calls"]
    )
    assert values["tangent_qr_calls"] == values["tangent_sample_chunk_count"] == 4
