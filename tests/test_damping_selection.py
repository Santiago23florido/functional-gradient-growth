"""The projection's regularisation, chosen by measurement instead of tuned.

The damping arbitrates between the two conditions the method needs at once,
and they pull in opposite directions: lowering it makes ``eps`` small while
``||u||`` explodes; raising it does the reverse. These tests pin that the
selection never trades the certificate away, that it ranks by the decrease
the lemma itself guarantees, and that it reports honestly when no
regularisation satisfies both.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fgdlib.search.damping import (
    DAMPING_FAN_RATIO,
    DAMPING_FAN_STEPS,
    select_projection_damping,
)
from stable_tiny.pipeline import build_model, load_pipeline_config


@pytest.fixture
def setup():
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        # Wide enough that the tangent space can certify at all: a fresh 3x2
        # sits at eps ~ 1.9, where no damping helps and selection correctly
        # declines. The trade-off these tests are about only exists once the
        # certificate is reachable.
        model=replace(config.model, hidden_size=32, number_hidden_layers=2),
        fgd_approx=replace(
            config.fgd_approx,
            projection_solver="exact",
            certify_linearization_tolerance=0.1,
        ),
    )
    device = torch.device("cpu")
    model = build_model(config, device)
    torch.manual_seed(0)
    x = torch.randn(8, config.data.in_features)
    y = torch.randn(8, config.data.out_features)
    return model, x, y, config.fgd_approx


def test_every_candidate_satisfies_the_certificate_or_scores_zero(setup) -> None:
    """The certificate is never traded for a bigger step."""
    model, x, y, fa = setup
    choice = select_projection_damping(model, x, y, fa)
    if choice is None:
        pytest.skip("no damping both certifies and realises a step here")
    threshold = min(fa.rel_error_threshold, 0.5)
    for candidate in choice.candidates:
        if candidate.guaranteed_decrease > 0.0:
            assert candidate.relative_error < threshold


def test_the_chosen_rung_maximises_the_guaranteed_decrease(setup) -> None:
    """Ranking is by eta * ||g||^2 -- the lemma's own quantity, not by eps."""
    model, x, y, fa = setup
    choice = select_projection_damping(model, x, y, fa)
    if choice is None:
        pytest.skip("no damping both certifies and realises a step here")
    best = max(c.guaranteed_decrease for c in choice.candidates)
    assert choice.candidate.guaranteed_decrease == pytest.approx(best)
    assert choice.candidate.guaranteed_decrease > 0.0


def test_a_smaller_eps_does_not_win_on_its_own(setup) -> None:
    """A tiny eps bought with an unrealisable step must score zero.

    This is the failure the whole module exists for: the lowest eps is
    reached at the lowest damping, where ||u|| is largest and no rate in the
    certified interval keeps the step inside the regime the lemma describes.
    """
    model, x, y, fa = setup
    choice = select_projection_damping(model, x, y, fa)
    if choice is None:
        pytest.skip("no damping both certifies and realises a step here")
    for candidate in choice.candidates:
        if candidate.learning_rate is None:
            assert candidate.guaranteed_decrease == 0.0


def test_update_norm_falls_as_damping_rises(setup) -> None:
    """The trade-off the module rests on, as a measured monotonicity."""
    model, x, y, fa = setup
    choice = select_projection_damping(model, x, y, fa)
    if choice is None:
        pytest.skip("no damping both certifies and realises a step here")
    # The fan is emitted from the boundary downwards, so damping decreases
    # along the list and ||u|| must increase.
    dampings = [c.relative_damping for c in choice.candidates]
    norms = [c.update_norm for c in choice.candidates]
    assert dampings == sorted(dampings, reverse=True)
    assert norms == sorted(norms)


def test_relative_error_rises_with_damping(setup) -> None:
    """More regularisation is a worse approximation -- what bisection needs.

    Compared end to end rather than rung by rung: once ``eps`` reaches the
    floor set by the numerical rank, successive rungs differ only by
    round-off and their order carries no information.
    """
    model, x, y, fa = setup
    choice = select_projection_damping(model, x, y, fa)
    if choice is None:
        pytest.skip("no damping both certifies and realises a step here")
    errors = [c.relative_error for c in choice.candidates]
    assert errors[0] >= errors[-1]        # damping decreases along the fan


def test_the_fan_spans_the_configured_range(setup) -> None:
    """Coarse spacing was measured to straddle the optimum, so pin the range."""
    model, x, y, fa = setup
    choice = select_projection_damping(model, x, y, fa)
    if choice is None:
        pytest.skip("no damping both certifies and realises a step here")
    assert len(choice.candidates) <= DAMPING_FAN_STEPS + 1
    span = choice.candidates[0].relative_damping / choice.candidates[-1].relative_damping
    assert span == pytest.approx(
        DAMPING_FAN_RATIO ** -(len(choice.candidates) - 1), rel=1e-6
    )


def test_an_inadequate_tangent_space_yields_no_choice(setup) -> None:
    """Then growth is the answer, and saying so is the useful outcome.

    Inadequacy has to come from the STRUCTURE, not from a strict threshold:
    the fixture's model has ~1440 parameters against 24 output dimensions, so
    its Jacobian has full row rank and ``eps`` reaches zero at any threshold.
    A width-1 network cannot, which is the case that matters -- it is the
    state a run actually starts in.
    """
    _, x, y, fa = setup
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(config.model, hidden_size=1, number_hidden_layers=1),
        fgd_approx=fa,
    )
    starved = build_model(config, torch.device("cpu"))
    assert select_projection_damping(starved, x, y, fa) is None


def test_the_model_is_left_untouched(setup) -> None:
    """Selection is measurement: it must not move a single parameter."""
    model, x, y, fa = setup
    before = [p.detach().clone() for p in model.parameters()]
    select_projection_damping(model, x, y, fa)
    for parameter, original in zip(model.parameters(), before):
        assert torch.equal(parameter.detach(), original)
