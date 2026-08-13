"""Widening against inserting a layer, arbitrated in one measured quantity.

The criterion exists because GroMo's topology pins some widths outright: a
conv whose consumer sits behind a pool can never be widened, however badly it
is the bottleneck, and the only relief is a new layer. A width-only rule
cannot express that trade and so spends elsewhere forever.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgdlib.search.unified import (
    Candidate,
    rank_candidates_by_bottleneck_per_parameter,
)


def width(index: int, cost: int) -> Candidate:
    return Candidate(kind="width", index=index, cost=cost, relative_error_after=None)


def depth(index: int, cost: int) -> Candidate:
    return Candidate(kind="depth", index=index, cost=cost, relative_error_after=None)


def rank(candidates, scores, **kwargs):
    return rank_candidates_by_bottleneck_per_parameter(
        candidates, bottlenecks=scores, **kwargs
    )


def test_a_layer_wins_when_it_buys_more_per_parameter() -> None:
    candidates = [width(0, 28), depth(2, 39)]
    scores = {("width", 0): 1.0e-3, ("depth", 2): 1.5e-3}
    # 1.5/39 = 3.85e-5 against 1.0/28 = 3.57e-5
    assert rank(candidates, scores)[0].kind == "depth"


def test_a_layer_loses_when_it_does_not() -> None:
    candidates = [width(0, 28), depth(2, 39)]
    scores = {("width", 0): 1.0e-3, ("depth", 2): 1.1e-3}
    # 1.1/39 = 2.82e-5 against 1.0/28 = 3.57e-5
    assert rank(candidates, scores)[0].kind == "width"


def test_the_raw_floor_eliminates_cheap_and_useless_before_the_division() -> None:
    """The guard whose absence sent 11 of 12 purchases to the cheapest spot."""
    candidates = [width(0, 400), width(1, 2)]
    scores = {("width", 0): 1.0, ("width", 1): 1.0e-4}
    # Per parameter the cheap one wins by 20x, but its raw value is 1e-4 of
    # the best, far under the 0.25 floor, so it never reaches the division.
    ranked = rank(candidates, scores, min_gain_fraction=0.25)
    assert [c.index for c in ranked] == [0]


def test_a_zero_cost_exponent_is_a_pure_argmax_over_the_admitted() -> None:
    candidates = [width(0, 400), width(1, 2)]
    scores = {("width", 0): 1.0, ("width", 1): 0.9}
    assert [c.index for c in rank(candidates, scores, cost_exponent=0.0)] == [0, 1]
    assert [c.index for c in rank(candidates, scores, cost_exponent=1.0)] == [1, 0]


def test_nothing_is_bought_when_no_location_is_a_bottleneck() -> None:
    candidates = [width(0, 28), depth(2, 39)]
    assert rank(candidates, {("width", 0): 0.0, ("depth", 2): 0.0}) == []
    assert rank(candidates, {}) == []


def test_an_unmeasurable_candidate_can_never_win() -> None:
    candidates = [width(0, 28), depth(2, 39)]
    scores = {("width", 0): 1.0e-3, ("depth", 2): float("inf")}
    assert [c.kind for c in rank(candidates, scores)] == ["width"]


def test_joint_moves_are_not_eligible() -> None:
    """They carry two indices, so a single bottleneck cannot price them."""
    joint = Candidate(
        kind="joint", index=0, cost=60, relative_error_after=None, indices=(0, 1)
    )
    scores = {("width", 0): 1.0e-3}
    assert [c.kind for c in rank([width(0, 28), joint], scores)] == ["width"]


# --- against the real measurement -------------------------------------


CONV_STACK = [
    {"conv": [2, 3]},
    {"conv": [2, 3]},
    "maxpool",
    {"conv": [2, 3]},
    {"conv": [2, 3]},
    "maxpool",
    {"avgpool": [3, 3]},
    "flatten",
    {"mlp": [2, 1]},
]


def build_model():
    from fgdlib.models.convstack import build_conv_stack_model

    torch.manual_seed(0)
    return build_conv_stack_model(
        CONV_STACK,
        in_features=784,
        out_features=10,
        device=torch.device("cpu"),
        input_shape=(1, 28, 28),
    )


def build_loader(n: int = 128, batch: int = 64):
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(n, 1, 28, 28, generator=generator)
    y = torch.zeros(n, 10)
    y[torch.arange(n), torch.randint(0, 10, (n,), generator=generator)] = 1.0
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=batch
    )


def measured_scores():
    """The scores the pipeline branch would build, via the same calls."""
    import copy

    from fgdlib.search.depth import insert_identity_layer
    from fgdlib.tangent import compute_expressivity_bottlenecks
    from stable_tiny.pipeline import load_pipeline_config

    device = torch.device("cpu")
    config = load_pipeline_config(
        "configs/fgd/family_ladder_matrix_free_N1024.yaml"
    ).fgd_approx
    base = build_model()
    loader = build_loader()
    parameters = sum(p.numel() for p in base.parameters() if p.requires_grad)

    scores: dict[tuple[str, int], float] = {}
    candidates: list[Candidate] = []
    from fgdlib.search.growth import growable_neuron_costs

    widths = compute_expressivity_bottlenecks(base, loader, device, config)
    costs = growable_neuron_costs(base, 784)
    for index, (value, cost) in enumerate(zip(widths, costs)):
        scores[("width", index)] = value
        candidates.append(width(index, cost))

    for position in range(1, len(base.layers)):
        trial = copy.deepcopy(base)
        inserted = insert_identity_layer(trial, position=position, device=device)
        where = next(
            (
                i
                for i, layer in enumerate(trial._growable_layers)
                if layer is inserted
            ),
            None,
        )
        if where is None:
            continue
        trial_scores = compute_expressivity_bottlenecks(
            trial, loader, device, config
        )
        cost = (
            sum(p.numel() for p in trial.parameters() if p.requires_grad)
            - parameters
        )
        scores[("depth", position)] = trial_scores[where]
        candidates.append(depth(position, cost))
    return candidates, scores


def test_every_candidate_the_branch_builds_is_measurable() -> None:
    """R5: if the depth scores were all zero the criterion would be inert."""
    candidates, scores = measured_scores()
    assert len([k for k in scores if k[0] == "width"]) == 3
    assert len([k for k in scores if k[0] == "depth"]) >= 4
    assert all(value > 0.0 for value in scores.values())
    # Four orders of magnitude apart: a ranking here is a measurement, not a
    # tie-break.
    assert max(scores.values()) / min(scores.values()) > 1000.0


def test_the_criterion_picks_a_real_winner_on_measured_numbers() -> None:
    candidates, scores = measured_scores()
    ranked = rank_candidates_by_bottleneck_per_parameter(
        candidates, bottlenecks=scores, min_gain_fraction=0.25, cost_exponent=1.0
    )
    assert ranked
    best = ranked[0]
    rate = scores[(best.kind, best.index)] / best.cost
    for candidate in candidates:
        key = (candidate.kind, candidate.index)
        if key in scores and scores[key] >= 0.25 * max(scores.values()):
            assert rate >= scores[key] / candidate.cost - 1e-12


def test_the_default_is_off_so_the_width_only_branch_is_what_runs() -> None:
    from stable_tiny.pipeline import load_pipeline_config

    for name in (
        "family_ladder_N1024",
        "family_ladder_matrix_free_N1024",
        "mnist_matrix_free",
    ):
        config = load_pipeline_config(f"configs/fgd/{name}.yaml")
        assert config.fgd_approx.growth_where_balance is False
