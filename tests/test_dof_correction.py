"""The degrees-of-freedom correction makes a subsample certificate honest.

The certificate ``eps`` is an R^2-like ratio, so a least-squares projection
over-fits a finite probe and biases ``eps`` DOWN toward the spurious 0. The
adjusted-R^2 correction removes that bias and blows up at interpolation. These
pin the three properties the correction must have.
"""

from __future__ import annotations

import math

from fgdlib.search.damping import dof_corrected_relative_error


def _rho(relative_error: float) -> float:
    return 1.0 / (1.0 + relative_error**2)


def test_correction_matches_adjusted_r2_formula():
    raw = 0.8355
    nk, m = 192, 19
    rho = _rho(raw)
    rho_adj = 1.0 - (1.0 - rho) * (nk - 1) / (nk - m - 1)
    expected = math.sqrt((1.0 - rho_adj) / rho_adj)
    got = dof_corrected_relative_error(raw, nk, m)
    assert abs(got - expected) < 1e-9


def test_correction_raises_eps_over_the_raw_subsample_value():
    # Debiasing an over-fit (downward-biased) subsample eps can only INCREASE it.
    raw = 0.60
    corrected = dof_corrected_relative_error(raw, n_observations=128, effective_rank=19)
    assert corrected > raw


def test_interpolation_regime_is_infinite():
    # NK <= m + 1: the probe can be interpolated, the certificate is vacuous.
    assert dof_corrected_relative_error(0.0, n_observations=20, effective_rank=19) == float("inf")
    assert dof_corrected_relative_error(0.0, n_observations=19, effective_rank=19) == float("inf")
    assert dof_corrected_relative_error(0.1, n_observations=15, effective_rank=30) == float("inf")


def test_is_a_noop_on_the_whole_dataset():
    # NK >> m: the (NK-1)/(NK-m-1) factor is ~1, so the correction barely moves eps.
    raw = 0.90
    corrected = dof_corrected_relative_error(raw, n_observations=100_000, effective_rank=30)
    assert abs(corrected - raw) < 1e-3


def test_monotonic_in_sample_size():
    # For a fixed raw eps, a larger probe needs a smaller correction.
    raw = 0.70
    small = dof_corrected_relative_error(raw, n_observations=100, effective_rank=19)
    large = dof_corrected_relative_error(raw, n_observations=1000, effective_rank=19)
    assert small > large >= raw
