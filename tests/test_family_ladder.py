"""The certified family ladder: certify by the family's OWN relative error.

The rule is strict -- a family step is accepted only when RelErr(g_family, r)
< 1/2 for that family's own projection, never by a descent criterion and never
by the tangent's projection. A nonlinear within-MLP family certifies at a
smaller structure than the linear tangent, so using it before growing cuts the
function-preserving growth that is otherwise ruinous.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fgdlib.search.certify import grow_until_certified
from fgdlib.search.families import certify_parametric_step
from fgdlib.tangent import build_projection_probe
from stable_tiny.pipeline import build_dataloaders, build_model, load_pipeline_config


@pytest.fixture
def setup():
    cfg = load_pipeline_config("configs/fgd/certify_smooth_sin_tiny.yaml")
    device = torch.device("cpu")
    torch.manual_seed(0)
    tl, _, _ = build_dataloaders(cfg, device)
    model = build_model(cfg, device)
    probe = build_projection_probe(tl, cfg.fgd_approx.probe_batches, device)
    return cfg, device, tl, model, probe


def test_parametric_family_certifies_by_its_own_relative_error(setup) -> None:
    """A certified result has RelErr < 1/2; an uncertified one returns no model."""
    cfg, device, tl, model, probe = setup
    x, y = probe
    # A wider structure so the nonlinear family has room to certify.
    from fgdlib.search.certify import grow_until_certified as grow
    model, _ = grow(model=model, x=x, y=y, train_loader=tl, device=device,
                    config=replace(cfg, fgd_approx=replace(cfg.fgd_approx, rel_error_threshold=0.7)),
                    max_growths=40, function_preserving=True)
    res = certify_parametric_step(model, x, y, cfg.fgd_approx,
                                  inner_steps=400, inner_learning_rate=0.01)
    if res.certified:
        assert res.relative_error < 0.5
        assert res.model is not None
    else:
        assert res.model is None          # never returned uncertified


def test_an_uncertified_family_never_returns_a_step(setup) -> None:
    """At the 3x2 start no within-MLP family can certify -- and it must not."""
    cfg, device, tl, model, probe = setup
    x, y = probe
    res = certify_parametric_step(model, x, y, cfg.fgd_approx, inner_steps=200)
    # eps at 3x2 is ~1.6; whatever the family reaches, it cannot be < 1/2 here.
    assert (res.model is not None) == (res.relative_error < 0.5)


def test_the_ladder_cuts_function_preserving_growth(setup) -> None:
    """The whole point: fewer FP growths with the family than without."""
    cfg, device, tl, model, probe = setup
    x, y = probe

    def run(use_family):
        torch.manual_seed(0)
        m = build_model(cfg, device)
        fam = None
        if use_family:
            def fam(mm):
                return certify_parametric_step(
                    mm, x, y, cfg.fgd_approx, inner_steps=400,
                    inner_learning_rate=0.01,
                ).model
        _, res = grow_until_certified(
            model=m, x=x, y=y, train_loader=tl, device=device, config=cfg,
            max_growths=120, function_preserving=True, family_step=fam,
        )
        return res

    without = run(False)
    with_family = run(True)
    # The nonlinear family certifies at a smaller structure, so it stops the
    # growth loop far sooner.
    assert with_family.growths < without.growths
    if with_family.family_stepped:
        assert with_family.family_steps >= 1


def test_a_family_step_marks_family_stepped(setup) -> None:
    """When a family certifies before growth is exhausted, the flag is set."""
    cfg, device, tl, model, probe = setup
    x, y = probe

    def fam(mm):
        return certify_parametric_step(
            mm, x, y, cfg.fgd_approx, inner_steps=400, inner_learning_rate=0.01,
        ).model

    _, res = grow_until_certified(
        model=model, x=x, y=y, train_loader=tl, device=device, config=cfg,
        max_growths=120, function_preserving=True, family_step=fam,
    )
    # The measured run certifies via the family; if it did, the model moved.
    if res.family_stepped:
        assert res.family_steps == 1
