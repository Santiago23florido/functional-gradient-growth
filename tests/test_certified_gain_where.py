"""The certified-gain where-rule: non-uniform shapes, without stalling.

The shipped rule levels the widths -- the caller relieves the narrowest
location before scoring anything, the filter keeps only bottleneck candidates
while the ceiling binds, and the fallback takes the cheapest of those. Every
run lands on h, h, h+1. That costs parameter efficiency: MEASURED, the best
non-uniform architecture of three layers (4-19-10-13-1, 452 params) matches
the uniform 588-param one at 0.876 under conventional training.

These pin the replacement rule and, above all, the two properties whose
absence broke the earlier attempt: it must never return empty while the
certificate binds, and cheap-and-useless must not win on price.
"""

from __future__ import annotations

import pytest

from fgdlib.search.unified import (
    Candidate,
    rank_candidates_by_certified_gain,
)
from fgdlib.tangent import FGDApproxConfig


def _width(index: int, cost: int, eps_after: float) -> Candidate:
    return Candidate(
        kind="width", index=index, cost=cost, relative_error_after=eps_after
    )


def test_defaults_keep_the_shipped_rule() -> None:
    config = FGDApproxConfig()
    assert config.growth_where == "rank_ceiling"
    assert config.growth_where_cost_exponent == pytest.approx(1.0)
    assert config.growth_where_min_gain_fraction == pytest.approx(0.25)
    assert config.growth_where_allow_depth is False
    assert config.growth_where_burst is True


def test_never_returns_empty_while_the_certificate_binds() -> None:
    """The property whose absence stalled growth at 53 parameters.

    Every candidate is flat or worse, so nothing "gains". The rule must still
    hand back exactly one -- the lowest measured eps -- because above the
    certificate no step exists at any damping and growth is the only move.
    """
    before = 0.60
    candidates = [
        _width(0, cost=20, eps_after=0.62),
        _width(1, cost=32, eps_after=0.61),
        _width(2, cost=17, eps_after=0.65),
    ]
    ranked = rank_candidates_by_certified_gain(
        candidates,
        relative_error_before=before,
        gradient_sq_norm=1.0,
        certificate_binds=True,
    )
    assert len(ranked) == 1
    assert ranked[0].index == 1          # lowest eps, not lowest cost


def test_no_gain_and_a_satisfied_certificate_buys_nothing() -> None:
    ranked = rank_candidates_by_certified_gain(
        [_width(0, cost=20, eps_after=0.62)],
        relative_error_before=0.60,
        gradient_sq_norm=1.0,
        certificate_binds=False,
    )
    assert ranked == []


def test_cheap_and_useless_cannot_outbid_a_real_gain() -> None:
    """The C2 degeneracy, pinned.

    MEASURED without the raw-gain floor: 11 of 12 consecutive purchases went
    to the last growable location, whose cost does not grow with its own
    width and which adds a single tangent direction. Per-parameter value
    alone rewards exactly that.
    """
    before = 0.60
    useless_but_cheap = _width(2, cost=6, eps_after=before - 1e-4)
    real_but_dear = _width(0, cost=40, eps_after=before - 0.20)
    ranked = rank_candidates_by_certified_gain(
        [useless_but_cheap, real_but_dear],
        relative_error_before=before,
        gradient_sq_norm=1.0,
        certificate_binds=True,
    )
    assert ranked[0].index == 0
    assert useless_but_cheap not in ranked   # eliminated by the floor


def test_a_comparable_cheap_buy_still_wins() -> None:
    """The floor is a guard, not a cost blindfold."""
    before = 0.60
    cheap = _width(2, cost=6, eps_after=before - 0.19)
    dear = _width(0, cost=40, eps_after=before - 0.20)
    ranked = rank_candidates_by_certified_gain(
        [cheap, dear],
        relative_error_before=before,
        gradient_sq_norm=1.0,
        certificate_binds=True,
    )
    assert ranked[0].index == 2


def test_no_uniformity_mandate_remains() -> None:
    """With widths [20, 3, 20] the shipped rule would force index 1.

    Here the widest location wins if it measures the larger gain, because
    nothing filters on the width minimum any more.
    """
    before = 0.60
    wide_pays = _width(0, cost=20, eps_after=before - 0.30)
    narrow_does_not = _width(1, cost=20, eps_after=before - 0.01)
    ranked = rank_candidates_by_certified_gain(
        [wide_pays, narrow_does_not],
        relative_error_before=before,
        gradient_sq_norm=1.0,
        certificate_binds=True,
    )
    assert ranked[0].index == 0


def test_cost_exponent_zero_is_a_pure_argmin() -> None:
    """The documented rollback for cost degeneracy: certify.py's own rule."""
    before = 0.60
    cheap_small_gain = _width(2, cost=2, eps_after=before - 0.10)
    dear_big_gain = _width(0, cost=100, eps_after=before - 0.30)
    with_cost = rank_candidates_by_certified_gain(
        [cheap_small_gain, dear_big_gain],
        relative_error_before=before,
        gradient_sq_norm=1.0,
        cost_exponent=1.0,
        min_gain_fraction=0.0,
        certificate_binds=True,
    )
    without_cost = rank_candidates_by_certified_gain(
        [cheap_small_gain, dear_big_gain],
        relative_error_before=before,
        gradient_sq_norm=1.0,
        cost_exponent=0.0,
        min_gain_fraction=0.0,
        certificate_binds=True,
    )
    assert with_cost[0].index == 2       # value per parameter favours cheap
    assert without_cost[0].index == 0    # raw value favours the big gain


def test_non_finite_candidates_are_dropped() -> None:
    before = 0.60
    good = _width(0, cost=20, eps_after=0.40)
    ranked = rank_candidates_by_certified_gain(
        [
            good,
            Candidate("width", 1, 20, float("inf")),
            Candidate("width", 2, 20, None),
        ],
        relative_error_before=before,
        gradient_sq_norm=1.0,
        certificate_binds=True,
    )
    assert ranked == [good]


def test_all_candidates_unmeasurable_returns_empty() -> None:
    ranked = rank_candidates_by_certified_gain(
        [Candidate("width", 0, 20, None)],
        relative_error_before=0.60,
        gradient_sq_norm=1.0,
        certificate_binds=True,
    )
    assert ranked == []
