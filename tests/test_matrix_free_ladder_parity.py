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
    # 5e-3, from measurement rather than taste. The Krylov subspace converges
    # to a (P-1)-dimensional invariant subspace, so one direction of J is
    # unreachable and eps comes out slightly LOW -- 1.5e-03 at P=25. Forcing
    # k = P does not close it, it triples it (8.8e-03), because the extra
    # direction is numerical noise. The bias is one-sided, so the guard that
    # matters is elsewhere: the step is certified on the displacement actually
    # applied, never on this eps.
    assert eps_approx == pytest.approx(eps_exact, rel=5e-3)
    assert eps_approx < eps_exact, "the truncation bias is optimistic, by construction"


def test_the_default_config_never_takes_the_approximate_branch() -> None:
    """The seam is gated on the family name and nothing else."""
    config = load_pipeline_config(LADDER).fgd_approx
    assert tuple(config.family_order) == ("tangent",)
    _c, model, x, y = _probe(LADDER)
    first = exact_tangent_system(model, x, y, config)
    second = exact_tangent_system(model, x, y, config)
    assert torch.equal(first.jacobian, second.jacobian)


def test_the_factored_spectrum_is_the_spectrum_of_j() -> None:
    """``svd(J) = (A, S, VB)`` -- the identity the whole path rests on.

    If this ever stops holding, every downstream consumer reading the factors
    is silently deciding on a different operator than the one it thinks.
    """
    from fgdlib.search.damping import DAMPING_BRACKET
    from fgdlib.search.mffactored import (
        factored_minimal_relative_error,
        factored_spectrum,
    )

    config = load_pipeline_config(APPROX).fgd_approx
    _c, model, x, y = _probe(APPROX)
    system = exact_tangent_system(model, x, y, config)
    assert system.factors is not None

    spectrum = factored_spectrum(system)
    assert spectrum is not None
    _left, singular_values, right = spectrum
    _a, reference, _b = torch.linalg.svd(
        system.jacobian.double(), full_matrices=False
    )
    count = singular_values.numel()
    assert torch.allclose(reference[:count], singular_values, rtol=1e-5, atol=1e-8)

    # Never an (N K, P) object anywhere in the factored route.
    assert right.shape[1] == count < system.jacobian.shape[1]

    factored = factored_minimal_relative_error(system, config, DAMPING_BRACKET[0])
    assert factored == pytest.approx(
        exact_relative_error(model, x, y, config, system=system), rel=1e-3
    )


def test_the_exact_system_carries_no_factors() -> None:
    """The default config must not even have the option of this path."""
    config = load_pipeline_config(LADDER).fgd_approx
    _c, model, x, y = _probe(LADDER)
    assert exact_tangent_system(model, x, y, config).factors is None
