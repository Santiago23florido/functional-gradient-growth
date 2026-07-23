"""The grow-to-certify loop: Lemma 3.5 satisfied by construction."""

from __future__ import annotations

from dataclasses import replace

import torch

from fgdlib.search.certify import exact_relative_error, grow_until_certified
from stable_tiny.pipeline import build_model, load_pipeline_config


def _fixture(samples: int, hidden_size: int = 3):
    device = torch.device("cpu")
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(
            config.model, hidden_size=hidden_size, number_hidden_layers=2
        ),
        fgd_approx=replace(config.fgd_approx, projection_solver="exact"),
    )
    model = build_model(config, device)
    torch.manual_seed(0)
    x = torch.randn(samples, config.data.in_features)
    y = torch.randn(samples, config.data.out_features)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=min(samples, 25)
    )
    return config, model, x, y, loader, device


def test_the_loop_reaches_the_certificate() -> None:
    """The point of the method: eps < 1/2 holds when the loop returns."""
    config, model, x, y, loader, device = _fixture(samples=50)
    grown, result = grow_until_certified(
        model=model, x=x, y=y, train_loader=loader, device=device,
        config=config, max_growths=40,
    )
    assert result.certified
    assert result.relative_error < config.fgd_approx.rel_error_threshold
    # And the returned model really is the certified one.
    assert exact_relative_error(grown, x, y, config.fgd_approx) == (
        result.relative_error
    )


def test_epsilon_is_monotone_along_the_loop() -> None:
    """What makes termination a theorem rather than a hope.

    f never moves (every growth is function-preserving), so r is fixed and
    each addition strictly increases ||P_T(r)||.
    """
    config, model, x, y, loader, device = _fixture(samples=50)
    _, result = grow_until_certified(
        model=model, x=x, y=y, train_loader=loader, device=device,
        config=config, max_growths=40,
    )
    trajectory = result.trajectory
    assert len(trajectory) >= 2
    assert all(
        later < earlier for earlier, later in zip(trajectory, trajectory[1:])
    ), trajectory


def test_the_loop_never_moves_the_represented_function() -> None:
    """Growth refines the representation; only a certified step moves f."""
    config, model, x, y, loader, device = _fixture(samples=50)
    model.eval()
    with torch.no_grad():
        before = model(x).clone()

    grown, result = grow_until_certified(
        model=model, x=x, y=y, train_loader=loader, device=device,
        config=config, max_growths=40,
    )
    assert result.growths > 0          # the test would be vacuous otherwise
    grown.eval()
    with torch.no_grad():
        after = grown(x)
    assert torch.allclose(before, after, atol=1e-4)


def test_an_already_certified_structure_is_left_alone() -> None:
    """No growth is spent when the condition already holds.

    With few samples the tangent space is already saturated, so the loop must
    return immediately -- capacity is only ever added to satisfy the
    certificate.
    """
    config, model, x, y, loader, device = _fixture(samples=10)
    parameters_before = sum(p.numel() for p in model.parameters())
    _, result = grow_until_certified(
        model=model, x=x, y=y, train_loader=loader, device=device,
        config=config, max_growths=40,
    )
    assert result.certified
    assert result.growths == 0
    assert sum(p.numel() for p in model.parameters()) == parameters_before


def test_certification_cost_grows_with_the_probe() -> None:
    """The measured tension worth knowing about.

    eps < 1/2 asks the tangent space to capture 80 % of the gradient energy
    over N*K output dimensions, so a bigger probe -- a STRONGER certificate --
    needs strictly more structure. Measured: 25 samples certify in 6 growths,
    100 samples need 24.
    """
    growths = {}
    for samples in (25, 100):
        config, model, x, y, loader, device = _fixture(samples=samples)
        _, result = grow_until_certified(
            model=model, x=x, y=y, train_loader=loader, device=device,
            config=config, max_growths=60,
        )
        assert result.certified
        growths[samples] = result.growths
    assert growths[100] > growths[25], growths
