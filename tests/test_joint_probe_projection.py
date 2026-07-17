"""The certificate solves ONE joint projection over a fixed probe."""

from __future__ import annotations

import math

import pytest
import torch

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.tangent import (
    FGDApproxConfig,
    _compute_tangent_projection_step,
    _output_relative_error_from_stats,
    build_projection_probe,
    evaluate_fgd_validation_certificate,
    measure_direction_projection,
)

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP  # noqa: E402


def _toy_problem() -> tuple[GrowingMLP, list[tuple[torch.Tensor, torch.Tensor]]]:
    torch.manual_seed(0)
    model = GrowingMLP(
        in_features=3,
        out_features=2,
        hidden_size=4,
        number_hidden_layers=1,
        device=torch.device("cpu"),
    )
    batches = [
        (torch.randn(5, 3), torch.randn(5, 2) * 0.3)
        for _ in range(4)
    ]
    return model, batches


def test_probe_is_the_concatenation_of_the_first_batches() -> None:
    _, batches = _toy_problem()
    x, y = build_projection_probe(batches, probe_batches=3)
    assert torch.equal(x, torch.cat([batches[0][0], batches[1][0], batches[2][0]]))
    assert torch.equal(y, torch.cat([batches[0][1], batches[1][1], batches[2][1]]))
    # Fewer batches than requested is allowed: the probe is what exists.
    x_small, _ = build_projection_probe(batches[:1], probe_batches=4)
    assert torch.equal(x_small, batches[0][0])
    with pytest.raises(ValueError):
        build_projection_probe(batches, probe_batches=0)
    with pytest.raises(ValueError):
        build_projection_probe([], probe_batches=2)


def test_certificate_equals_single_joint_projection_over_the_probe() -> None:
    """The certificate must be the joint solve, not an aggregate of solves."""
    model, batches = _toy_problem()
    config = FGDApproxConfig(
        projection_solver="exact_kernel_eigh",
        probe_batches=4,
    )
    certificate = evaluate_fgd_validation_certificate(
        model=model,
        data_loader=batches,
        device=torch.device("cpu"),
        config=config,
        learning_rate=None,
    )
    assert certificate.sensor_valid
    assert certificate.relative_error is not None

    x = torch.cat([x for x, _ in batches])
    y = torch.cat([y for _, y in batches])
    joint_step = _compute_tangent_projection_step(
        model=model,
        x=x,
        y=y,
        config=config,
    )
    assert certificate.relative_error == pytest.approx(
        joint_step.output_error.relative_error,
        rel=1e-12,
    )
    assert certificate.output_relative_error is not None
    assert certificate.output_relative_error.approximation_norm == pytest.approx(
        joint_step.output_error.approximation_norm,
        rel=1e-12,
    )


def test_certificate_does_not_fall_back_to_independent_batch_projections() -> None:
    """Guard: aggregated independent per-batch solves are a DIFFERENT number."""
    model, batches = _toy_problem()
    config = FGDApproxConfig(
        projection_solver="exact_kernel_eigh",
        probe_batches=4,
    )
    certificate = evaluate_fgd_validation_certificate(
        model=model,
        data_loader=batches,
        device=torch.device("cpu"),
        config=config,
        learning_rate=None,
    )

    dot = 0.0
    approx_sq = 0.0
    target_sq = 0.0
    per_batch_updates = []
    for x, y in batches:
        step = _compute_tangent_projection_step(
            model=model,
            x=x,
            y=y,
            config=config,
        )
        per_batch_updates.append(step.parameter_updates)
        dot += step.dot_product
        approx_sq += step.approximation_sq_norm
        target_sq += step.target_sq_norm
    aggregated = _output_relative_error_from_stats(
        dot_product=dot,
        approximation_sq_norm=approx_sq,
        target_sq_norm=target_sq,
        eps=config.eps,
    )

    # The per-batch parameter directions genuinely differ from one another,
    # so an aggregate of them cannot equal one shared direction.
    first = torch.cat([u.reshape(-1) for u in per_batch_updates[0]])
    second = torch.cat([u.reshape(-1) for u in per_batch_updates[1]])
    assert not torch.allclose(first, second)
    assert certificate.relative_error != pytest.approx(
        aggregated.relative_error,
        rel=1e-6,
    )
    # A shared direction cannot beat each batch's OWN optimal projection, so
    # the joint (honest) relative error is at least the aggregated one.
    assert certificate.relative_error >= aggregated.relative_error - 1e-9


def test_all_probe_batches_share_one_parameter_direction() -> None:
    """g on each probe batch is J_batch u* for the SAME u*."""
    model, batches = _toy_problem()
    config = FGDApproxConfig(
        projection_solver="exact_kernel_eigh",
        probe_batches=4,
    )
    x = torch.cat([x for x, _ in batches])
    y = torch.cat([y for _, y in batches])
    joint_step = _compute_tangent_projection_step(
        model=model,
        x=x,
        y=y,
        config=config,
    )
    shared_direction = joint_step.parameter_updates

    # Evaluating the shared direction jointly reproduces the joint stats...
    joint_stats = measure_direction_projection(
        model,
        shared_direction,
        x,
        y,
        config,
    )
    assert joint_stats.approximation_sq_norm == pytest.approx(
        joint_step.approximation_sq_norm,
        rel=1e-4,
    )
    # ...and the per-batch images of the SAME direction add up exactly to the
    # joint measurement (linearity of J u over the stacked probe).
    dot = 0.0
    approx_sq = 0.0
    target_sq = 0.0
    for batch_x, batch_y in batches:
        stats = measure_direction_projection(
            model,
            shared_direction,
            batch_x,
            batch_y,
            config,
        )
        dot += stats.dot_product
        approx_sq += stats.approximation_sq_norm
        target_sq += stats.target_sq_norm
    assert dot == pytest.approx(joint_stats.dot_product, rel=1e-6)
    assert approx_sq == pytest.approx(joint_stats.approximation_sq_norm, rel=1e-6)
    assert target_sq == pytest.approx(joint_stats.target_sq_norm, rel=1e-6)
    assert math.isfinite(joint_stats.output_error.relative_error)
