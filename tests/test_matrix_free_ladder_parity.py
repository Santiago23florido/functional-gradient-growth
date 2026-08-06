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


def _dense(system):
    """Rebuild ``J = W V^T`` for comparison.

    Production never does this -- that product is 4.8 GB at MNIST width, which
    is the entire reason it is not built. At test scale it is the only way to
    check the factored route against the operator it claims to represent.
    """
    left_factor, right_factor = system.factors
    return left_factor.double() @ right_factor.double().t()


def test_matrix_free_only_overrides_are_explicit_and_exact_defaults_stay_off() -> None:
    """Pin every intentional matrix-free override and every safety default."""
    ladder = load_pipeline_config(LADDER).fgd_approx
    approx = load_pipeline_config(APPROX).fgd_approx
    differing = {
        field.name
        for field in dataclasses.fields(ladder)
        if getattr(ladder, field.name) != getattr(approx, field.name)
    }
    assert differing == {
        "family_order",
        "theory_lr_search_steps",
        "tangent_measured_max_lr",
        "transactional_realized_descent",
        "transactional_max_retries",
        "transactional_descent_atol",
        "certify_functional_lr_cap",
        "certify_family_lemma35_rate",
        "max_total_parameters",
    }
    assert ladder.matrixfree_rank == 0, "the exact ladder must never truncate"
    assert ladder.transactional_realized_descent is False
    assert ladder.transactional_max_retries == 0
    assert ladder.certify_functional_lr_cap is None
    assert ladder.certify_family_lemma35_rate is False
    assert ladder.max_total_parameters is None
    assert approx.transactional_realized_descent is True
    assert approx.transactional_max_retries == 3
    assert approx.certify_functional_lr_cap == pytest.approx(0.1)
    assert approx.certify_family_lemma35_rate is True
    assert approx.max_total_parameters == 600


def test_truncation_is_conservative_not_optimistic() -> None:
    """A lower rank must push ``eps`` UP, never down.

    ``J_k = J V V^T`` has a smaller range than ``J``, so less of ``r`` is
    representable. That makes truncation under-certify, which is the safe
    direction -- and it is worth pinning, because the intuitive guess is the
    opposite and I held it for a while. If this ever flips, a truncated run is
    certifying directions the exact tangent would refuse.
    """
    import dataclasses as dc

    config, model, x, y = _wide(16)
    errors = {}
    for rank in (0, 256, 128, 64):
        approx = dc.replace(config.fgd_approx, matrixfree_rank=rank)
        errors[rank] = exact_relative_error(model, x, y, approx)
    assert errors[0] <= errors[256] <= errors[128] <= errors[64], errors


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
    # J itself is never built on the factored route; only its width survives,
    # and the operator it stands for must have the exact one's shape.
    assert system_approx.jacobian.numel() == 0
    assert _dense(system_approx).shape == system_exact.jacobian.shape

    eps_exact = exact_relative_error(model, x, y, config_exact, system=system_exact)
    eps_approx = exact_relative_error(model, x, y, config_approx, system=system_approx)
    # 5e-3, from measurement rather than taste, and it covers the NEAR-FULL
    # rank regime only: here P=25 so the cap never binds and k = P - 1. The
    # Krylov subspace converges to a (P-1)-dimensional invariant subspace, so
    # one direction is unreachable and eps lands 1.5e-03 LOW. Forcing k = P
    # triples that (8.8e-03) because the extra direction is numerical noise.
    #
    # This is a different regime from real truncation. Once k << rank the bias
    # reverses and grows: J_k = J V V^T has a smaller RANGE, less of r is
    # representable, and eps comes out HIGH -- +44% at k=127, P=641. That
    # direction is the safe one and is pinned by
    # test_truncation_is_conservative_not_optimistic.
    assert eps_approx == pytest.approx(eps_exact, rel=5e-3)


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
    dense = _dense(system)
    _a, reference, _b = torch.linalg.svd(_dense(system), full_matrices=False)
    count = singular_values.numel()
    assert torch.allclose(reference[:count], singular_values, rtol=1e-5, atol=1e-8)

    # Never an (N K, P) object anywhere in the factored route.
    assert right.shape[1] == count < dense.shape[1]

    factored = factored_minimal_relative_error(system, config, DAMPING_BRACKET[0])
    assert factored == pytest.approx(
        exact_relative_error(model, x, y, config, system=system), rel=1e-3
    )


def test_the_exact_system_carries_no_factors() -> None:
    """The default config must not even have the option of this path."""
    config = load_pipeline_config(LADDER).fgd_approx
    _c, model, x, y = _probe(LADDER)
    assert exact_tangent_system(model, x, y, config).factors is None


def _wide(width: int):
    import dataclasses

    base = load_pipeline_config(APPROX)
    config = dataclasses.replace(
        base, model=dataclasses.replace(base.model, hidden_size=width)
    )
    device = torch.device("cpu")
    train_loader, _, _ = build_dataloaders(config, device)
    model = build_model(config, device)
    x, y = build_projection_probe(train_loader, config.fgd_approx.probe_batches)
    return config, model, x, y


def test_factored_gram_is_the_gram() -> None:
    """``J^T J = V (W^T W) V^T`` -- O(P^2 k) compute instead of O(NK P^2)."""
    from fgdlib.search.mffactored import factored_gram

    config, model, x, y = _wide(16)
    system = exact_tangent_system(model, x, y, config.fgd_approx)
    jacobian = _dense(system)
    reference = jacobian.t() @ jacobian
    produced = factored_gram(system)
    gap = float(
        torch.linalg.matrix_norm(produced - reference)
        / torch.linalg.matrix_norm(reference)
    )
    assert gap < 1e-12, gap


def test_factored_solve_matches_the_dense_normal_equations() -> None:
    """The k x k solve returns the same direction as the P x P one.

    ``J = W V^T`` puts the damped solution in ``span(V)``, so ``u = V c`` with
    ``(W^T W + lam I) c = W^T t`` -- a k x k system. Neither the O(P^2) Gram
    nor its factorisation is built, which at MNIST width is the difference
    between running and not running at all.
    """
    from fgdlib.search.mffactored import factored_projection_solve

    config, model, x, y = _wide(16)
    system = exact_tangent_system(model, x, y, config.fgd_approx)
    jacobian = _dense(system)
    target = system.target.reshape(-1).double()
    size = jacobian.shape[1]
    damping = 1e-6 * float(torch.linalg.matrix_norm(jacobian, 2)) ** 2

    produced, _predicted = factored_projection_solve(system, target, damping)
    reference = torch.linalg.solve(
        jacobian.t() @ jacobian + damping * torch.eye(size, dtype=torch.float64),
        jacobian.t() @ target,
    )
    cosine = float(
        torch.dot(produced, reference) / (produced.norm() * reference.norm())
    )
    assert cosine == pytest.approx(1.0, abs=1e-6), cosine
    residual = lambda v: float((jacobian @ v - target).norm())
    assert residual(produced) == pytest.approx(residual(reference), rel=1e-6)


def test_factored_damping_selection_agrees_on_eps_and_lambda() -> None:
    """The factored selector must pick the SAME rung as the dense one.

    eps and lambda agree tightly; the update does not, and that is recorded
    rather than hidden. MEASURED at P=641: cos 0.979 with a 20% norm gap,
    because the ladder's chosen damping is ~1e-12 relative and near-zero
    damping puts the weight on the smallest singular values -- exactly where
    a rank-k subspace is blind. The step is re-certified on what it applies.
    """
    from fgdlib.search.damping import (
        select_projection_damping,
        select_projection_damping_factored,
    )

    import dataclasses as dc

    config, model, x, y = _wide(16)
    # FULL rank on both sides. This test is about the SELECTOR agreeing, not
    # about truncation -- that is what the monotonicity test above measures,
    # and at rank 128 the two would legitimately sit 44% apart.
    factored_config = dc.replace(config.fgd_approx, matrixfree_rank=0)
    system = exact_tangent_system(model, x, y, factored_config)
    exact_config = dc.replace(factored_config, family_order=("tangent",))
    dense = select_projection_damping(
        model, x, y, exact_config,
        system=exact_tangent_system(model, x, y, exact_config),
    )
    factored = select_projection_damping_factored(
        model, x, y, factored_config, system=system
    )
    assert dense is not None and factored is not None
    assert factored.candidate.relative_damping == pytest.approx(
        dense.candidate.relative_damping, rel=1e-9
    )
    assert factored.candidate.relative_error == pytest.approx(
        dense.candidate.relative_error, rel=1e-3
    )


def test_functional_rate_cap_applies_only_to_the_factored_selector() -> None:
    """Keep theorem proposal separate from attempted matrix-free rate."""
    from fgdlib.search.damping import (
        select_projection_damping,
        select_projection_damping_factored,
    )

    import dataclasses as dc

    config, model, x, y = _wide(16)
    cap = 1e-4
    factored_config = dc.replace(
        config.fgd_approx,
        certify_functional_lr_cap=cap,
    )
    factored_system = exact_tangent_system(model, x, y, factored_config)
    factored = select_projection_damping_factored(
        model,
        x,
        y,
        factored_config,
        system=factored_system,
    )
    assert factored is not None
    assert factored.candidate.certified_learning_rate is not None
    assert factored.candidate.certified_learning_rate > cap
    assert factored.candidate.learning_rate == pytest.approx(cap)

    exact_config = dc.replace(factored_config, family_order=("tangent",))
    exact = select_projection_damping(
        model,
        x,
        y,
        exact_config,
        system=exact_tangent_system(model, x, y, exact_config),
    )
    assert exact is not None
    assert exact.candidate.learning_rate == pytest.approx(
        exact.candidate.certified_learning_rate
    )
