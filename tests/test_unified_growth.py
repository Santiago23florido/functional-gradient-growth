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


def test_rank_ceiling_identifies_the_bottleneck() -> None:
    from fgdlib.unified_growth import rank_limiting_locations

    assert rank_limiting_locations([2, 3, 4]) == [0]
    assert rank_limiting_locations([8, 8, 10]) == [0, 1]
    assert rank_limiting_locations([]) == []


def test_a_cheap_purchase_cannot_outbid_bottleneck_relief() -> None:
    """rank J <= min_l w_l, so refining a wide layer cannot lift eps's cap.

    This is the failure that stalled the search at 784->2->3->4: value per
    parameter kept buying the cheap late layers for eps gains of ~0.002
    while every purchase refined inside a rank-2 image.
    """
    candidates = [
        _candidate("width", 2, 13, 0.79),         # cheap, wide layer
        Candidate(
            kind="width",
            index=0,
            cost=793,
            relative_error_after=0.78,
            relieves_rank_ceiling=True,           # the narrow one
        ),
    ]
    ranked = rank_candidates(
        candidates,
        relative_error_before=0.80,
        gradient_sq_norm=100.0,
        statistical_threshold=0.0,
        rank_ceiling_binds=True,
    )
    assert ranked and ranked[0].index == 0

    # With the ceiling NOT binding, value per parameter governs again.
    unconstrained = rank_candidates(
        candidates,
        relative_error_before=0.80,
        gradient_sq_norm=100.0,
        statistical_threshold=0.0,
        rank_ceiling_binds=False,
    )
    assert unconstrained[0].index == 2


def test_blind_immediate_eps_falls_back_to_relieving_the_bottleneck() -> None:
    """Immediate eps is blind to capacity that needs training to pay off.

    Terminating on that reading is what froze the structure. When the rank
    ceiling binds, the cheapest relief is bought anyway.
    """
    candidates = [
        _candidate("width", 2, 13, 0.95),          # immediately worse
        Candidate(
            kind="width",
            index=0,
            cost=793,
            relative_error_after=0.99,             # also immediately worse
            relieves_rank_ceiling=True,
        ),
    ]
    ranked = rank_candidates(
        candidates,
        relative_error_before=0.80,
        gradient_sq_norm=100.0,
        rank_ceiling_binds=True,
    )
    assert ranked and ranked[0].index == 0
    # Without a bottleneck to relieve, R3 termination still applies.
    assert (
        rank_candidates(
            [_candidate("width", 2, 13, 0.95)],
            relative_error_before=0.80,
            gradient_sq_norm=100.0,
        )
        == []
    )
