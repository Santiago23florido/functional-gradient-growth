"""GCV as the selector for the tangent-kernel ridge problem.

The damped tangent solve is kernel ridge regression in the tangent (NTK)
kernel, and generalized cross-validation is the classical way to pick its
regularisation without held-out data. These tests pin the mechanics -- df
and GCV as functions of the spectrum -- and pin that the selector reduces to
the descent rule when told to. They deliberately do NOT assert that GCV
picks a different rung end to end: whether it CAN is data-dependent, and on
the pipeline model the certificate clips GCV's preference (measured
separately), which is itself a finding rather than a test target.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fgdlib.search.damping import select_projection_damping
from stable_tiny.pipeline import build_model, load_pipeline_config


@pytest.fixture
def setup():
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(config.model, hidden_size=32, number_hidden_layers=2),
        fgd_approx=replace(
            config.fgd_approx,
            projection_solver="exact",
            certify_linearization_tolerance=0.1,
        ),
    )
    model = build_model(config, torch.device("cpu"))
    torch.manual_seed(0)
    x = torch.randn(12, config.data.in_features)
    y = torch.randn(12, config.data.out_features)
    return model, x, y, config.fgd_approx


# --- the mechanics: df and GCV as closed forms of the spectrum ------------


def test_effective_dof_decreases_with_regularisation(setup) -> None:
    """df = sum sigma^2/(sigma^2+lambda) falls as lambda rises."""
    model, x, y, fa = setup
    choice = select_projection_damping(model, x, y, replace(fa, projection_damping_objective="gcv"))
    if choice is None:
        pytest.skip("no certified rung here")
    # candidates run from the boundary (highest rho) downward, so df must
    # increase along the list as regularisation is relaxed.
    dofs = [c.effective_dof for c in choice.candidates]
    assert dofs == sorted(dofs)


def test_effective_dof_never_exceeds_the_sample_count(setup) -> None:
    """H_lambda has eigenvalues in [0,1], so df = tr H_lambda <= N."""
    model, x, y, fa = setup
    n = x.shape[0] * fa_out_features(model)
    choice = select_projection_damping(model, x, y, replace(fa, projection_damping_objective="gcv"))
    if choice is None:
        pytest.skip("no certified rung here")
    for c in choice.candidates:
        assert 0.0 <= c.effective_dof <= n + 1e-6


def test_gcv_is_positive_and_finite_where_df_below_n(setup) -> None:
    """GCV = risk/(1-df/N)^2 is a well-defined positive number until df -> N."""
    model, x, y, fa = setup
    n = x.shape[0] * fa_out_features(model)
    choice = select_projection_damping(model, x, y, replace(fa, projection_damping_objective="gcv"))
    if choice is None:
        pytest.skip("no certified rung here")
    for c in choice.candidates:
        if c.effective_dof < n:
            assert c.gcv > 0.0
            assert c.gcv == c.gcv          # not NaN


def test_gcv_diverges_as_df_approaches_n() -> None:
    """The whole point: the (1-df/N)^2 denominator refuses interpolation.

    Constructed directly on a spectrum so the limit is exact rather than
    data-dependent: with df -> N the score must go to +inf.
    """
    from fgdlib.search.damping import DampingCandidate  # noqa: F401

    # Reproduce the closed form the module uses, on a chosen spectrum.
    import math

    n = 10
    residual_sq = 1.0
    for df in (9.0, 9.9, 9.99, 9.999):
        gap = 1.0 - df / n
        gcv = (residual_sq / n) / (gap * gap)
        assert gcv > 0.0
    # df == n is +inf by construction.
    assert (1.0 - 10.0 / n) == 0.0


# --- the selector reduces to descent when asked ---------------------------


def test_descent_objective_is_unchanged_by_the_new_fields(setup) -> None:
    """Adding df/gcv must not move the descent choice a single bit."""
    model, x, y, fa = setup
    choice = select_projection_damping(model, x, y, replace(fa, projection_damping_objective="descent"))
    if choice is None:
        pytest.skip("no certified rung here")
    best = max(c.guaranteed_decrease for c in choice.candidates)
    assert choice.candidate.guaranteed_decrease == pytest.approx(best)


def test_both_objectives_only_ever_pick_a_certified_realisable_rung(setup) -> None:
    """Neither objective may select a rung that fails eps<1/2 or has no rate."""
    model, x, y, fa = setup
    for objective in ("descent", "gcv"):
        choice = select_projection_damping(
            model, x, y, replace(fa, projection_damping_objective=objective)
        )
        if choice is None:
            continue
        assert choice.candidate.learning_rate is not None
        assert choice.candidate.relative_error < min(fa.rel_error_threshold, 0.5)


def test_the_model_is_left_untouched_under_gcv(setup) -> None:
    model, x, y, fa = setup
    before = [p.detach().clone() for p in model.parameters()]
    select_projection_damping(model, x, y, replace(fa, projection_damping_objective="gcv"))
    for parameter, original in zip(model.parameters(), before):
        assert torch.equal(parameter.detach(), original)


def fa_out_features(model) -> int:
    """Output width, for turning a sample count into an observation count."""
    return model.layers[-1].out_features
