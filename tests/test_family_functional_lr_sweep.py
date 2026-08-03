"""Sweeping the family's functional step size: more search, the same bar.

``certify_parametric_step`` asks whether a clone can realise ONE target,
``f - eta_f r``. A failure there says the clone could not reach that distance,
not that the family is unavailable -- and the ladder was recording it as the
latter, because its call site passed a single ``eta_f`` where ``parametric_gd``
has always walked a list.

What these tests pin is that the sweep is search and nothing else: every
candidate faces the same ``RelErr(Delta, r) < min(rel_error_threshold, 1/2)``
on the same probe, an empty list is the old single-eta call unchanged, and the
walk stops at the first certificate so a run that certifies at the head costs
what it costs today.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fgdlib.search.families import (
    certify_parametric_step,
    certify_parametric_step_swept,
)
from fgdlib.tangent import build_projection_probe
from stable_tiny.pipeline import build_dataloaders, build_model, load_pipeline_config


@pytest.fixture
def setup():
    cfg = load_pipeline_config("configs/experiments/certify_smooth_sin_tiny.yaml")
    device = torch.device("cpu")
    torch.manual_seed(0)
    tl, _, _ = build_dataloaders(cfg, device)
    model = build_model(cfg, device)
    probe = build_projection_probe(tl, cfg.fgd_approx.probe_batches, device)
    return cfg, device, tl, model, probe


def test_an_empty_sweep_is_the_single_eta_call(setup) -> None:
    """() must leave the configured single eta_f in charge, unchanged."""
    cfg, _, _, model, probe = setup
    x, y = probe
    fa = replace(cfg.fgd_approx, certify_family_functional_lr=0.7)

    torch.manual_seed(0)
    single = certify_parametric_step(
        model, x, y, fa, functional_learning_rate=0.7, inner_steps=60,
        inner_learning_rate=0.01,
    )
    torch.manual_seed(0)
    swept = certify_parametric_step_swept(
        model, x, y, fa, (), inner_steps=60, inner_learning_rate=0.01,
    )

    assert swept.certified == single.certified
    assert swept.relative_error == pytest.approx(single.relative_error)
    assert swept.functional_learning_rate == pytest.approx(0.7)


def test_the_sweep_stops_at_the_first_certificate(setup) -> None:
    """First certificate wins -- later candidates are never even trained."""
    cfg, _, _, model, probe = setup
    x, y = probe
    seen: list[float] = []

    def spy(*args, **kwargs):
        seen.append(kwargs["functional_learning_rate"])
        # Certify only the second candidate, so the third must never be tried.
        certified = len(seen) == 2
        return type(
            "R", (), {
                "certified": certified,
                "relative_error": 0.4 if certified else 0.9,
                "cosine": 0.9 if certified else 0.4,
                "model": object() if certified else None,
                "functional_learning_rate": kwargs["functional_learning_rate"],
            },
        )()

    import fgdlib.search.families as families

    original = families.certify_parametric_step
    families.certify_parametric_step = spy
    try:
        result = families.certify_parametric_step_swept(
            model, x, y, cfg.fgd_approx, (1.0, 0.5, 0.25),
        )
    finally:
        families.certify_parametric_step = original

    assert seen == [1.0, 0.5]
    assert result.certified
    assert result.functional_learning_rate == pytest.approx(0.5)


def test_every_swept_candidate_faces_the_same_bar(setup) -> None:
    """No candidate is accepted below the threshold the single call enforces.

    The sweep widens the SEARCH, never the criterion: a swept acceptance is an
    acceptance ``certify_parametric_step`` would also have returned at that
    same eta_f.
    """
    cfg, device, tl, model, probe = setup
    x, y = probe
    from fgdlib.search.certify import grow_until_certified

    # Widen first, so the nonlinear family has room to certify at all.
    model, _ = grow_until_certified(
        model=model, x=x, y=y, train_loader=tl, device=device,
        config=replace(
            cfg, fgd_approx=replace(cfg.fgd_approx, rel_error_threshold=0.7)
        ),
        max_growths=40, function_preserving=True,
    )

    rates = (1.0, 0.5, 0.25, 0.125)
    threshold = min(cfg.fgd_approx.rel_error_threshold, 0.5)

    torch.manual_seed(0)
    swept = certify_parametric_step_swept(
        model, x, y, cfg.fgd_approx, rates, inner_steps=200,
        inner_learning_rate=0.01,
    )

    if swept.certified:
        assert swept.relative_error < threshold
        assert swept.model is not None
        assert swept.functional_learning_rate in rates
        # And the very same eta_f certifies on its own, with no sweep involved.
        torch.manual_seed(0)
        alone = certify_parametric_step(
            model, x, y, cfg.fgd_approx,
            functional_learning_rate=swept.functional_learning_rate,
            inner_steps=200, inner_learning_rate=0.01,
        )
        assert alone.certified
    else:
        # A failed sweep returns a real attempt, never a step.
        assert swept.model is None
        assert swept.relative_error >= threshold


def test_a_sweep_certifies_at_least_as_often_as_its_head(setup) -> None:
    """Search cannot lose: the head is candidate one, so it is a lower bound."""
    cfg, device, tl, model, probe = setup
    x, y = probe
    from fgdlib.search.certify import grow_until_certified

    model, _ = grow_until_certified(
        model=model, x=x, y=y, train_loader=tl, device=device,
        config=replace(
            cfg, fgd_approx=replace(cfg.fgd_approx, rel_error_threshold=0.7)
        ),
        max_growths=40, function_preserving=True,
    )

    torch.manual_seed(0)
    head = certify_parametric_step(
        model, x, y, cfg.fgd_approx, functional_learning_rate=1.0,
        inner_steps=200, inner_learning_rate=0.01,
    )
    torch.manual_seed(0)
    swept = certify_parametric_step_swept(
        model, x, y, cfg.fgd_approx, (1.0, 0.5, 0.25),
        inner_steps=200, inner_learning_rate=0.01,
    )

    if head.certified:
        assert swept.certified
        assert swept.functional_learning_rate == pytest.approx(1.0)


def test_the_config_parses_a_list_of_functional_rates(tmp_path) -> None:
    """YAML list -> tuple of floats, like parametric_gd's own rates."""
    import yaml

    raw = yaml.safe_load(
        open("configs/experiments/certify_smooth_sin_tiny.yaml", encoding="utf-8")
    )
    raw["fgd_approx"]["certify_family_functional_lrs"] = [1.0, 0.5, 0.25]
    path = tmp_path / "swept.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = load_pipeline_config(path)
    assert cfg.fgd_approx.certify_family_functional_lrs == (1.0, 0.5, 0.25)


def test_the_default_config_sweeps_nothing(setup) -> None:
    """Off by default, so every existing result is untouched."""
    cfg, *_ = setup
    assert cfg.fgd_approx.certify_family_functional_lrs == ()
