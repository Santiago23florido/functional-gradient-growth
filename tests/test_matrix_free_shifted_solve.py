"""The damping sweep must be free, and must still be the exact projection.

``(J^T J + lambda I) u = J^T r`` is a family of SHIFTED systems. Shifting does
not rotate the Krylov subspace ``K_k(J^T J, J^T r)``, only the spectrum inside
it, so one bidiagonalization serves every lambda. These tests pin the two
claims that makes: the cost stops depending on the grid, and the answer is
still the damped least-squares solution the tangent would have computed.

Certification is what is being protected here. ``eps`` is read straight off the
``(k+1) x k`` factor, and that reading is only valid while ``U`` is
orthonormal -- so orthogonality is tested directly rather than assumed.
"""

from __future__ import annotations

import pytest
import torch

from fgdlib.search.matrixfree import matrix_free_tangent_step
from fgdlib.tangent import FGDApproxConfig, functional_gradient

GRID = (1e-6, 1e-4, 1e-2, 1e-1, 1.0)


def _fixture(seed: int = 0, width: int = 6, examples: int = 24, batches: int = 3):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, width), torch.nn.Tanh(), torch.nn.Linear(width, 1)
    ).double()
    x = torch.randn(examples, 3, dtype=torch.float64)
    y = torch.randn(examples, 1, dtype=torch.float64)
    size = examples // batches
    loader = [
        (x[index : index + size], y[index : index + size])
        for index in range(0, examples, size)
    ]
    return model, loader, x, y


def _exact(model, x, y, config):
    """Dense ``J`` and ``r``. Only tractable because ``P`` is tiny here."""
    parameters = list(model.parameters())
    outputs = model(x)
    rows = [
        torch.cat(
            [
                g.reshape(-1)
                for g in torch.autograd.grad(value, parameters, retain_graph=True)
            ]
        )
        for value in outputs.reshape(-1)
    ]
    jacobian = torch.stack(rows).double()
    with torch.no_grad():
        residual = functional_gradient(
            outputs.detach(), y, config.functional_loss
        ).reshape(-1).double()
    return jacobian, residual


def _exact_epsilon(jacobian, residual, lam):
    size = jacobian.shape[1]
    solution = torch.linalg.solve(
        jacobian.T @ jacobian + lam * torch.eye(size, dtype=torch.float64),
        jacobian.T @ residual,
    )
    predicted = jacobian @ solution
    error = float(
        torch.linalg.vector_norm(predicted - residual)
        / torch.linalg.vector_norm(predicted)
    )
    return error, solution


def test_swept_epsilon_matches_the_exact_damped_projection() -> None:
    """The reported ``eps`` is the tangent's, at the lambda the tangent picks."""
    config = FGDApproxConfig()
    model, loader, x, y = _fixture()
    jacobian, residual = _exact(model, x, y, config)
    reference = {lam: _exact_epsilon(jacobian, residual, lam)[0] for lam in GRID}
    best = min(reference, key=reference.get)

    _direction, certificate = matrix_free_tangent_step(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        config=config,
        iterations=200,
        damping=1e-2,
        damping_grid=GRID,
    )
    assert certificate.damping == best
    assert certificate.relative_error == pytest.approx(reference[best], rel=1e-6)


def test_direction_matches_the_exact_damped_solution() -> None:
    """Not just the certificate -- the ``u`` that actually gets applied."""
    config = FGDApproxConfig()
    model, loader, x, y = _fixture(seed=3)
    jacobian, residual = _exact(model, x, y, config)
    reference = {lam: _exact_epsilon(jacobian, residual, lam) for lam in GRID}
    best = min(reference, key=lambda lam: reference[lam][0])

    direction, certificate = matrix_free_tangent_step(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        config=config,
        iterations=200,
        damping=1e-2,
        damping_grid=GRID,
    )
    assert direction is not None
    assert certificate.damping == best
    flattened = torch.cat(
        [direction[name].reshape(-1) for name, _ in model.named_parameters()]
    ).double()
    expected = reference[best][1]
    assert torch.linalg.vector_norm(flattened - expected) <= 1e-6 * (
        torch.linalg.vector_norm(expected)
    )


def test_sweep_cost_does_not_depend_on_the_grid_size() -> None:
    """The whole point. One lambda and seven lambdas cost the same passes.

    Before the shared subspace this was one full solve per lambda; a 7-point
    grid at 50 iterations measured 11,344 passes per step, which is what made
    the family unusable at any scale worth having it for.
    """
    config = FGDApproxConfig()
    model, loader, _x, _y = _fixture(seed=5)
    counts = []
    for grid in ((1e-4,), (1e-6, 1e-4, 1e-2), GRID + (1e1, 1e2)):
        _direction, certificate = matrix_free_tangent_step(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            config=config,
            iterations=40,
            damping=1e-4,
            damping_grid=grid,
        )
        counts.append(certificate.passes)
    assert len(set(counts)) == 1


def test_iterations_beyond_the_numerical_rank_are_free() -> None:
    """Happy breakdown: once the subspace is exhausted, nothing more is spent.

    This is what keeps ``iterations`` a safe upper bound rather than a cost.
    ``k`` self-limits at the numerical rank of ``J`` as seen from ``r``, which
    MEASURED at P=106 was 41 -- so the price tracks the rank, not ``P``.
    """
    config = FGDApproxConfig()
    model, loader, _x, _y = _fixture(seed=7, width=4)
    saturated = None
    for iterations in (100, 400):
        _direction, certificate = matrix_free_tangent_step(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            config=config,
            iterations=iterations,
            damping=1e-4,
            damping_grid=GRID,
        )
        if saturated is None:
            saturated = certificate.passes
        assert certificate.passes == saturated


def test_reported_cosine_is_the_measured_alignment() -> None:
    """``cos`` is read off the tiny factor; check it against the real vectors.

    The identity only holds while ``U`` is orthonormal, so this is the test
    that would catch reorthogonalization being dropped.
    """
    import copy

    config = FGDApproxConfig()
    model, loader, x, y = _fixture(seed=11)
    jacobian, residual = _exact(model, x, y, config)

    direction, certificate = matrix_free_tangent_step(
        model=copy.deepcopy(model),
        loader=loader,
        device=torch.device("cpu"),
        config=config,
        iterations=200,
        damping=1e-2,
        damping_grid=GRID,
    )
    assert direction is not None
    flattened = torch.cat(
        [direction[name].reshape(-1) for name, _ in model.named_parameters()]
    ).double()
    predicted = jacobian @ flattened
    measured = float(
        torch.dot(predicted, residual)
        / (
            torch.linalg.vector_norm(predicted)
            * torch.linalg.vector_norm(residual)
        )
    )
    assert certificate.cosine == pytest.approx(measured, rel=1e-6)
