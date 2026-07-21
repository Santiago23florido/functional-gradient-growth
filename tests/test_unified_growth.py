"""Width and depth judged by one certified quantity.

The requirement these pin: every structural step, including a change in the
number of layers, is decided by Lemma 3.5's relative error and nothing else.
"""

from __future__ import annotations

import pytest

from fgdlib.unified_growth import Candidate, expansion_value, rank_candidates


def _candidate(kind: str, index: int, cost: int, after: float | None) -> Candidate:
    return Candidate(
        kind=kind, index=index, cost=cost, relative_error_after=after
    )


def test_value_is_the_gain_in_expressible_gradient_energy() -> None:
    """Delta(N eta) = ||r||^2 [1/(1+eps_a^2) - 1/(1+eps_b^2)]."""
    gradient_sq_norm = 10.0
    value = expansion_value(
        relative_error_before=1.0,
        relative_error_after=0.5,
        gradient_sq_norm=gradient_sq_norm,
    )
    expected = gradient_sq_norm * (1 / 1.25 - 1 / 2.0)
    assert value == pytest.approx(expected)


def test_a_candidate_that_does_not_help_scores_zero() -> None:
    """Growth must never be rewarded for enlarging nothing."""
    assert (
        expansion_value(
            relative_error_before=0.4,
            relative_error_after=0.9,      # worse
            gradient_sq_norm=5.0,
        )
        == 0.0
    )
    # And an unmeasurable candidate cannot win by accident.
    assert (
        expansion_value(
            relative_error_before=0.4,
            relative_error_after=None,
            gradient_sq_norm=5.0,
        )
        == 0.0
    )


def test_depth_and_width_compete_in_one_ranking() -> None:
    """The whole point: no separate policy decides layers.

    A depth insertion at width 8 costs 72 parameters; a neuron on a 784-wide
    input projection costs 793. If the layer buys nearly as much, it must
    win on value per parameter -- and a policy that decided depth separately
    could not have expressed that trade at all.
    """
    candidates = [
        _candidate("width", 0, 793, 0.62),   # expensive, buys a lot
        _candidate("depth", 1, 72, 0.66),    # cheap, buys nearly as much
        _candidate("width", 2, 19, 0.79),    # cheapest, buys little
    ]
    ranked = rank_candidates(
        candidates,
        relative_error_before=0.80,
        gradient_sq_norm=100.0,
        statistical_threshold=0.0,          # admit everything, test the ORDER
    )
    assert ranked[0].kind == "depth"


def test_an_expensive_candidate_still_wins_when_it_earns_it() -> None:
    """Guards against re-creating R2's starvation of the input projection."""
    candidates = [
        _candidate("width", 0, 793, 0.10),   # expensive but transformative
        _candidate("width", 2, 19, 0.79),    # cheap, marginal
    ]
    ranked = rank_candidates(
        candidates,
        relative_error_before=0.80,
        gradient_sq_norm=100.0,
        statistical_threshold=0.0,
    )
    assert ranked[0].index == 0


def test_no_improving_candidate_terminates_the_search() -> None:
    """R3: nothing enlarges the reachable set -> nothing is bought."""
    candidates = [
        _candidate("width", 0, 793, 0.95),
        _candidate("depth", 1, 72, 0.90),
    ]
    assert (
        rank_candidates(
            candidates, relative_error_before=0.80, gradient_sq_norm=100.0
        )
        == []
    )


def test_the_best_candidate_is_always_admitted() -> None:
    """GroMo's own rule: keep at least the best, whatever the threshold."""
    candidates = [_candidate("depth", 1, 10_000, 0.799)]
    ranked = rank_candidates(
        candidates,
        relative_error_before=0.80,
        gradient_sq_norm=1.0,
        statistical_threshold=1e9,          # absurdly strict
    )
    assert len(ranked) == 1
