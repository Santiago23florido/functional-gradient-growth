"""The second ladder must BE the ladder, with only J approximated.

Two claims, and the second matters more than the first: the approximated
system reproduces the exact one closely enough to drive the same decisions,
and choosing it changes nothing about how the default config runs.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from fgdlib.search.certify import exact_relative_error, exact_tangent_system
from fgdlib.tangent import build_projection_probe
from stable_tiny.pipeline import (
    _is_isolated_primary,
    build_dataloaders,
    build_model,
    load_pipeline_config,
)

LADDER = "configs/fgd/family_ladder_N1024.yaml"
APPROX = "configs/fgd/mftangent_ladder_N1024.yaml"


def _probe(path):
    config = load_pipeline_config(path)
    device = torch.device("cpu")
    train_loader, _, _ = build_dataloaders(config, device)
    model = build_model(config, device)
    x, y = build_projection_probe(train_loader, config.fgd_approx.probe_batches)
    return config, model, x, y


def test_the_two_configs_differ_only_in_the_family() -> None:
    """Everything the ladder decides with must be identical."""
    ladder = load_pipeline_config(LADDER).fgd_approx
    approx = load_pipeline_config(APPROX).fgd_approx
    differing = {
        field.name
        for field in dataclasses.fields(ladder)
        if getattr(ladder, field.name) != getattr(approx, field.name)
    }
    assert differing == {"family_order"}


def test_both_configs_run_the_ladder_not_a_standalone_family() -> None:
    """The approximated one must NOT fall into the isolated-family engine.

    MEASURED when it did: 0.1919 test accuracy against the ladder's 0.9487,
    because a standalone family has none of the rungs, no realisation path
    and no nonlinear fallback. The solver was never what that gap was about.
    """
    assert _is_isolated_primary(load_pipeline_config(LADDER)) is False
    assert _is_isolated_primary(load_pipeline_config(APPROX)) is False


def test_approximated_system_matches_the_exact_one() -> None:
    """Same shape, and an eps the ladder would act on identically."""
    _c_exact, model, x, y = _probe(LADDER)
    config_exact = load_pipeline_config(LADDER).fgd_approx
    config_approx = load_pipeline_config(APPROX).fgd_approx

    system_exact = exact_tangent_system(model, x, y, config_exact)
    system_approx = exact_tangent_system(model, x, y, config_approx)
    assert system_approx is not None
    assert system_approx.jacobian.shape == system_exact.jacobian.shape

    eps_exact = exact_relative_error(model, x, y, config_exact, system=system_exact)
    eps_approx = exact_relative_error(model, x, y, config_approx, system=system_approx)
    assert eps_approx == pytest.approx(eps_exact, rel=1e-3)


def test_the_default_config_never_takes_the_approximate_branch() -> None:
    """The seam is gated on the family name and nothing else."""
    config = load_pipeline_config(LADDER).fgd_approx
    assert tuple(config.family_order) == ("tangent",)
    _c, model, x, y = _probe(LADDER)
    first = exact_tangent_system(model, x, y, config)
    second = exact_tangent_system(model, x, y, config)
    assert torch.equal(first.jacobian, second.jacobian)
