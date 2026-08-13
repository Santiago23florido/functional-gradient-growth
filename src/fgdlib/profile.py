"""Opt-in timing and counters for the expensive FGD tangent stages.

Classification rule for the tangent_* fields added for the exact
tangent-system instrumentation: counters and gauges that hold integer
shapes or counts belong in ``_COUNTER_FIELDS`` so ``snapshot()`` renders
them as ints -- this includes fields written with ``set_max``/``set_value``
(``tangent_qr_input_rows``, ``tangent_qr_input_columns``,
``tangent_peak_jacobian_block_rows``, ``tangent_peak_jacobian_block_columns``,
``tangent_parameter_count``, ``tangent_output_row_count``), not only fields
written with ``increment``. ``tangent_condition_estimate`` and
``tangent_oracle_max_relative_error`` are genuinely fractional and stay
float.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

import torch

PROFILE_FIELDS = (
    "exact_tangent_system_calls",
    "exact_tangent_system_seconds",
    "streamed_jacobian_seconds",
    "minimal_relative_error_seconds",
    "select_projection_damping_seconds",
    "select_projection_damping_factored_seconds",
    "damping_factorization_seconds",
    "realize_initial_system_seconds",
    "realize_vjp_calls",
    "realize_vjp_seconds",
    "realize_factorization_seconds",
    "realize_solve_seconds",
    "realize_line_search_seconds",
    "realize_iterations",
    "line_search_trials",
    "current_parameter_count",
    "realize_cholesky_fallbacks",
    # Opt-in endpoint transaction. Existing tangent/realization counters show
    # the expensive solves; these three expose the bounded retry decisions.
    "transactional_realization_trials",
    "transactional_realization_accepted",
    "transactional_realization_rejected",
    # The same transaction, applied to the FAMILY step, which certified a
    # direction on the probe and then committed a full-length displacement --
    # MEASURED on MNIST, three runs, the full-train functional went x3.97,
    # x6.21 and x15.44 on the one family step each of them took. The third
    # counter is the risk of the guard rather than of the bug: if no alpha
    # ever lowers the real objective the ladder is off in practice, and the
    # ladder is what cuts function-preserving growths from 57 to 6.
    "family_transaction_accepted",
    "family_transaction_backtracks",
    "family_transaction_rejected",
    # The adaptive certification probe. A base held fixed for the whole run is
    # interpolated by the very steps it certifies -- MEASURED at 14.2x easier
    # per image than the population by transaction ~50 -- so these say whether
    # the base is actually being redrawn and how much counterexample memory the
    # row bound is evicting to stay affordable.
    "probe_base_resamples",
    "probe_counterexample_evictions",
    # Certification probes observed BELOW the interpolation floor NK > P, where
    # rank(J) <= min(NK, P) lets eps read a spurious zero. MEASURED on run
    # unguzxkq: NK/P fell to 0.45 and eps collapsed 0.44 -> 0.009 while the
    # validation relative error stayed at 1.02. Non-zero here invalidates every
    # eps in the run, so it must be countable rather than eyeballed.
    "probe_below_parameter_floor",
    # Releases of the caching allocator's pool at the points where the probe
    # changes shape. MEASURED at the cluster's NK: 0.232 GB live against
    # 2.359 GB reserved, and three jobs died of a pool that ratcheted while the
    # live footprint stayed small.
    "probe_allocator_cache_releases",
    # The growth lookahead declining to answer because the certificate on its
    # throwaway clone could not be computed. The lookahead is ADVISORY, so the
    # honest answer is "no opinion" -- but a high count means the dense float32
    # SVD it runs is failing often, which is a real defect to attack rather than
    # tolerate. MEASURED: run 1g0895r3 (job 457944) died at epoch 6 after
    # 4h01m because this path raised instead of abstaining, with the GPU at 41%
    # and MaxRSS at 2.6 GB of 64 G -- nothing exhausted, a hint killed the run.
    "growth_lookahead_non_finite_abstentions",
    # Validation certificates answered from the matrix-free factors instead of
    # a dense NK x P SVD. MEASURED: that SVD was 99.95% of run 1g0895r3 and the
    # line it died on. The fallback counter is the one to watch -- if it is
    # high the factors are not usable and the dense cost is back.
    "validation_factored_certificates",
    "validation_factored_fallbacks",
    # The realizable-progress criterion abstaining rather than refusing, which
    # is the only thing standing between "no eta realizes a step" and a frozen
    # run: MEASURED at 35 epochs of exact no-op before it existed.
    "certify_realizable_abstentions",
    "where_scans",
    "where_candidates",
    "where_base_system_reuses",
    "where_full_candidate_system_calls",
    "where_fast_candidate_scores",
    "where_new_column_seconds",
    "where_cross_statistics_seconds",
    "where_candidate_spectrum_seconds",
    "where_schur_factorization_seconds",
    "where_schur_solve_seconds",
    "where_sensor_seconds",
    "where_verification_seconds",
    "where_full_fallbacks",
    "where_unsupported_structure_fallbacks",
    "where_ambiguous_fallbacks",
    "where_numerical_fallbacks",
    "where_winner_mismatch_fallbacks",
    "where_final_winner_full_validations",
    "where_total_seconds",
    # certify_force_growth_on_finite_step_failure discriminator: how often
    # a failed-to-commit step FORCED grow_until_certified's growth versus
    # how often forcing was SUPPRESSED because the measurement behind the
    # failure was non-finite (the over-firing symptom, not the MNIST
    # deadlock -- see fgdlib/tangent.py's field comment).
    "certify_forced_growths",
    "certify_force_suppressed_nonfinite",
    "certify_family_deferral_refused",
    "certify_growth_not_warranted",
    # Its counterpart, and the pair is the whole diagnostic: the lookahead is
    # only useful if it says BOTH things. A run where _warranted_entries climbs
    # monotonically while _not_warranted stays at 0 is a lookahead that is not
    # discriminating -- it would grow one neuron per outer step until the
    # parameter budget, which is the failure this trigger exists to avoid.
    "certify_growth_warranted_entries",
    "growth_where_base_unavailable",
    # Declared late because it almost never fired: "every layer expresses what
    # is asked of it, so buy nothing" was unreachable in practice until a
    # stopping criterion started producing all-zero bottlenecks. It has been
    # passed to fallback() since fe4c88d and was never registered, so
    # FGD_PROFILE=1 raised KeyError the first time growth actually stopped --
    # the one event the counter exists to record.
    "growth_where_no_bottleneck",
    # certify_growth_target discriminators: how often the best available growth
    # failed the certify_growth_min_gain floor (the loop was chasing a target it
    # cannot reach at a price worth paying), how often max_total_parameters
    # ended the loop, and how many candidate clones were dropped as unaffordable
    # before being scored.
    "certify_growth_target_stalls",
    "certify_budget_stops",
    "certify_budget_rejected_candidates",
    # Nonlinear primary-family instrumentation. These fields stay independent
    # of the tangent counters so a nonlinear smoke profile can prove both what
    # it did and that no tangent construction occurred.
    "nonlinear_total_seconds",
    "nonlinear_candidate_training_seconds",
    "nonlinear_certification_seconds",
    "nonlinear_growth_statistics_seconds",
    "nonlinear_growth_application_seconds",
    "nonlinear_ladder_attempts",
    "nonlinear_accepted_steps",
    "nonlinear_failed_ladders",
    "nonlinear_growth_events",
    # K-fold bottleneck significance. The cost is the whole risk of the
    # criterion -- 2K statistics passes per layer per event against 1 today --
    # so it is measured, never assumed. The two counters separate "the test
    # ran" from "the test said no", which is what distinguishes a criterion
    # that stops growth from one that was never consulted.
    "growth_crossfold_seconds",
    "growth_crossfold_layers_tested",
    "growth_crossfold_layers_rejected",
    # --- exact tangent-system construction instrumentation (Phase A) ---
    # Counters.
    "tangent_system_calls",
    "tangent_qr_calls",
    "tangent_sample_chunk_count",
    "tangent_backend_optimized_calls",
    "tangent_backend_legacy_calls",
    "tangent_analytic_jacobian_calls",
    "tangent_unsupported_structure_fallbacks",
    "tangent_numerical_fallbacks",
    "tangent_spectrum_fallbacks",
    "tangent_rank_deficient_surrogates",
    "tangent_surrogate_svd_fallbacks",
    "tangent_oracle_verifications",
    "tangent_capture_suspensions",
    # Coarse net that cannot go silent: incremented whenever backend !=
    # legacy but a Jacobian block was built WITHOUT the analytic path, for
    # ANY reason, in ANY of the three construction branches (streamed-Gram,
    # row-chunked, full). A specific, named reason is ALSO recorded via
    # ``tangent_unsupported_structure_fallbacks`` (or another reasoned
    # counter) at the same time -- this field exists so "optimized/auto ran
    # with zero analytic calls and zero specific fallback counted" can never
    # again happen without being caught by an invariant check.
    "tangent_backend_inapplicable_paths",
    # Seconds accumulators. tangent_system_total_seconds is INCLUSIVE of the
    # whole exact-tangent-system construction; the rest below it are
    # EXCLUSIVE, mutually-disjoint children (see fgdlib/tangent.py for the
    # parent/child contract -- never sum streamed_jacobian_seconds into
    # tangent_system_total_seconds, it is a nested parent, not a sibling).
    "tangent_system_total_seconds",
    # Matrix-free family. These were being passed to timed()/increment()
    # without ever being declared, so FGD_PROFILE=1 raised KeyError on the
    # first matrix-free call -- the one path that most needed measuring could
    # not be measured at all.
    "matrix_free_tangent_seconds",
    "matrix_free_tangent_calls",
    "matrix_free_vmap_fallbacks",
    # How often the dual route was unreachable because the operators were the
    # vmapped kind, which carries no analytic factors for dual_gram to read.
    # Distinct from matrix_free_vmap_fallbacks: that counts "the analytic
    # structure was refused", this counts "and the ratio gate would have sent
    # it down a route that cannot serve it".
    "matrix_free_dual_unavailable",
    "matrix_free_capture_pause_missing",
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
    "tangent_projection_solve_calls",
    "tangent_sensor_seconds",
    # Max-gauges (written with set_max).
    "tangent_qr_input_rows",
    "tangent_qr_input_columns",
    "tangent_peak_jacobian_block_rows",
    "tangent_peak_jacobian_block_columns",
    "tangent_oracle_max_relative_error",
    # Last-value gauges (written with set_value, overwrite semantics).
    "tangent_parameter_count",
    "tangent_output_row_count",
    "tangent_condition_estimate",
)

_COUNTER_FIELDS = {
    "exact_tangent_system_calls",
    "realize_vjp_calls",
    "realize_iterations",
    "line_search_trials",
    "current_parameter_count",
    "realize_cholesky_fallbacks",
    "transactional_realization_trials",
    "transactional_realization_accepted",
    "transactional_realization_rejected",
    "family_transaction_accepted",
    "family_transaction_backtracks",
    "family_transaction_rejected",
    "probe_base_resamples",
    "probe_counterexample_evictions",
    "probe_below_parameter_floor",
    "probe_allocator_cache_releases",
    "growth_lookahead_non_finite_abstentions",
    "validation_factored_certificates",
    "validation_factored_fallbacks",
    "certify_realizable_abstentions",
    "where_scans",
    "where_candidates",
    "where_base_system_reuses",
    "where_full_candidate_system_calls",
    "where_fast_candidate_scores",
    "where_full_fallbacks",
    "where_unsupported_structure_fallbacks",
    "where_ambiguous_fallbacks",
    "where_numerical_fallbacks",
    "where_winner_mismatch_fallbacks",
    "where_final_winner_full_validations",
    "certify_forced_growths",
    "certify_force_suppressed_nonfinite",
    "certify_family_deferral_refused",
    "certify_growth_not_warranted",
    "certify_growth_warranted_entries",
    "growth_where_base_unavailable",
    "growth_where_no_bottleneck",
    "certify_growth_target_stalls",
    "certify_budget_stops",
    "certify_budget_rejected_candidates",
    "nonlinear_ladder_attempts",
    "nonlinear_accepted_steps",
    "nonlinear_failed_ladders",
    "nonlinear_growth_events",
    "growth_crossfold_layers_tested",
    "growth_crossfold_layers_rejected",
    "matrix_free_dual_unavailable",
    "matrix_free_capture_pause_missing",
    "tangent_system_calls",
    "tangent_projection_solve_calls",
    "tangent_qr_calls",
    "tangent_sample_chunk_count",
    "tangent_backend_optimized_calls",
    "tangent_backend_legacy_calls",
    "tangent_analytic_jacobian_calls",
    "tangent_unsupported_structure_fallbacks",
    "tangent_numerical_fallbacks",
    "tangent_spectrum_fallbacks",
    "tangent_rank_deficient_surrogates",
    "tangent_surrogate_svd_fallbacks",
    "tangent_oracle_verifications",
    "tangent_capture_suspensions",
    "tangent_backend_inapplicable_paths",
    # Integer-shape gauges, per the classification rule in the module
    # docstring: written with set_max/set_value but still rendered as ints.
    "tangent_qr_input_rows",
    "tangent_qr_input_columns",
    "tangent_peak_jacobian_block_rows",
    "tangent_peak_jacobian_block_columns",
    "tangent_parameter_count",
    "tangent_output_row_count",
    "matrix_free_tangent_calls",
    "matrix_free_vmap_fallbacks",
}
_VALUES: dict[str, float] = {field: 0.0 for field in PROFILE_FIELDS}
_REASONS: dict[str, set[str]] = {}
_LOCK = threading.Lock()
_P = ParamSpec("_P")
_R = TypeVar("_R")


def profile_enabled() -> bool:
    """Whether profiling was explicitly enabled for this process."""
    return os.environ.get("FGD_PROFILE", "").lower() in {"1", "true", "yes", "on"}


def _synchronize_cuda() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def increment(field: str, amount: int | float = 1) -> None:
    """Increment one profiler field when profiling is enabled."""
    if not profile_enabled():
        return
    with _LOCK:
        _VALUES[field] += float(amount)


def set_value(field: str, value: int | float) -> None:
    """Set one profiler field when profiling is enabled."""
    if not profile_enabled():
        return
    with _LOCK:
        _VALUES[field] = float(value)


def set_max(field: str, value: float) -> None:
    """Monotone gauge: keep the largest value seen. `set_value` OVERWRITES and
    would report the LAST block's shape, not the peak; `increment` would report
    the SUM. Peaks need their own setter."""
    if not profile_enabled():
        return
    with _LOCK:
        _VALUES[field] = max(_VALUES[field], float(value))


def fallback(field: str, reason: str) -> None:
    """Increment a fallback counter AND record why. No silent fallback."""
    increment(field)
    if not profile_enabled():
        return
    with _LOCK:
        _REASONS.setdefault(field, set()).add(reason)


@contextmanager
def timed(field: str) -> Iterator[None]:
    """Accumulate wall time, synchronizing CUDA on both timer boundaries."""
    if not profile_enabled():
        yield
        return
    _synchronize_cuda()
    started = time.perf_counter()
    try:
        yield
    finally:
        _synchronize_cuda()
        increment(field, time.perf_counter() - started)


def profiled(field: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Time a function without changing its arguments or return value."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with timed(field):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def snapshot() -> dict[str, int | float]:
    """Return a stable copy, primarily for tests and structured reporting."""
    with _LOCK:
        values = dict(_VALUES)
    return {
        field: int(value) if field in _COUNTER_FIELDS else value
        for field, value in values.items()
    }


def reset() -> None:
    """Reset process-local measurements. Intended for isolated tests."""
    with _LOCK:
        for field in _VALUES:
            _VALUES[field] = 0.0
        _REASONS.clear()


def report() -> None:
    """Print one machine-readable profiler block at normal process exit."""
    if not profile_enabled():
        return
    for field, value in snapshot().items():
        if field in _COUNTER_FIELDS:
            rendered = str(value)
        else:
            rendered = f"{float(value):.6f}"
        print(f"[FGD-PROFILE] {field}={rendered}", flush=True)
    with _LOCK:
        reasons_snapshot = {
            field: sorted(reasons) for field, reasons in _REASONS.items()
        }
    for field in sorted(reasons_snapshot):
        for reason in reasons_snapshot[field]:
            print(f"[FGD-PROFILE] reason {field}={reason}", flush=True)


atexit.register(report)
