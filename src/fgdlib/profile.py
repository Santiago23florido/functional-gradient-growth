"""Opt-in timing and counters for the expensive FGD tangent stages."""

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
)

_COUNTER_FIELDS = {
    "exact_tangent_system_calls",
    "realize_vjp_calls",
    "realize_iterations",
    "line_search_trials",
    "current_parameter_count",
    "realize_cholesky_fallbacks",
}
_VALUES: dict[str, float] = {field: 0.0 for field in PROFILE_FIELDS}
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


atexit.register(report)
