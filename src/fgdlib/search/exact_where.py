"""Exact shared-base sufficient statistics for exhaustive ``where`` scans.

Function-preserving width growth embeds every old parameter coordinate in the
same tensor prefix and appends coordinates.  This module verifies that concrete
layout and streams only the appended tangent columns.  It deliberately does
not choose a winner: candidate scoring and certified fallback live in
``certify.py`` so these statistics can be tested directly against materialised
full-Jacobian oracles.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch.func import functional_call, jacrev

from fgdlib.tangent import (
    ExactTangentSystem,
    FGDApproxConfig,
    _clear_inaccessible_tensor_caches,
    _flatten_jacobian,
    _output_relative_error_from_stats,
    validate_exact_tangent_system,
)


class UnsupportedGrowthStructure(RuntimeError):
    """The candidate does not have the verified prefix-preserving layout."""


@dataclass(frozen=True)
class GrowthColumnLayout:
    """Coordinate correspondence between one base and grown model."""

    candidate: torch.nn.Module
    old_candidate_coordinates: tuple[int, ...]
    new_candidate_coordinates: tuple[int, ...]
    parameter_names: tuple[str, ...]
    new_local_coordinates: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CandidateStatistics:
    """Exact float64 sufficient statistics for ``[J_base | C]``."""

    layout: GrowthColumnLayout
    new_candidate_coordinates: tuple[int, ...]
    cross_gram: torch.Tensor
    new_gram: torch.Tensor
    new_rhs: torch.Tensor
    # R from a streamed QR of [J_base, C, r].  It preserves every Gram and
    # target statistic while allowing a cancellation-free Schur complement.
    joint_factor: torch.Tensor


@dataclass(frozen=True)
class SharedBaseStatistics:
    """Base statistics reused by every candidate in one unchanged scan."""

    gram: torch.Tensor
    rhs: torch.Tensor
    target_sq_norm: torch.Tensor
    candidates: tuple[CandidateStatistics, ...]


class CandidateScoringError(RuntimeError):
    """A numerical invariant prevented certified fast scoring."""


@dataclass(frozen=True)
class BaseSpectrum:
    """One deterministic float64 SVD shared by all candidate ridge solves."""

    singular_values: torch.Tensor
    right_vectors: torch.Tensor


@dataclass(frozen=True)
class ExactCandidateScore:
    """Finite-Tikhonov score and diagnostics for one candidate."""

    relative_error: float
    absolute_damping: float
    old_update: torch.Tensor
    new_update: torch.Tensor
    normal_residual: float
    schur_residual: float
    sensor_residual: float
    error_bound: float


def _flat_prefix_coordinates(
    old_shape: torch.Size,
    new_shape: torch.Size,
) -> tuple[int, ...]:
    if len(old_shape) != len(new_shape) or any(
        old > new for old, new in zip(old_shape, new_shape)
    ):
        raise UnsupportedGrowthStructure(
            f"parameter shape is not prefix-preserving: {old_shape} -> {new_shape}"
        )
    coordinates: list[int] = []
    for old_flat in range(int(torch.tensor(old_shape).prod())):
        old_coordinate = torch.unravel_index(torch.tensor(old_flat), old_shape)
        new_flat = 0
        for dimension, coordinate in enumerate(old_coordinate):
            stride = (
                int(torch.tensor(new_shape[dimension + 1 :]).prod())
                if dimension + 1 < len(new_shape)
                else 1
            )
            new_flat += int(coordinate) * stride
        coordinates.append(new_flat)
    return tuple(coordinates)


def identify_growth_columns(
    base: torch.nn.Module,
    candidate: torch.nn.Module,
    x: torch.Tensor,
    *,
    preservation_tolerance: float,
) -> GrowthColumnLayout:
    """Validate and return the exact old/new flattened-coordinate partition."""
    base_parameters = tuple(base.named_parameters())
    candidate_parameters = tuple(candidate.named_parameters())
    if tuple(name for name, _ in base_parameters) != tuple(
        name for name, _ in candidate_parameters
    ):
        raise UnsupportedGrowthStructure("trainable parameter ordering changed")

    old_global: list[int] = []
    new_global: list[int] = []
    new_local_by_parameter: list[tuple[int, ...]] = []
    candidate_offset = 0
    grew = False
    for (name, old), (_, new) in zip(base_parameters, candidate_parameters):
        old_local = _flat_prefix_coordinates(old.shape, new.shape)
        old_local_set = set(old_local)
        new_local = tuple(
            coordinate
            for coordinate in range(new.numel())
            if coordinate not in old_local_set
        )
        old_view = new.reshape(-1)[list(old_local)]
        if not torch.equal(old.detach().reshape(-1), old_view.detach()):
            raise UnsupportedGrowthStructure(
                f"old parameter values changed while growing {name}"
            )
        grew = grew or bool(new_local)
        old_global.extend(candidate_offset + coordinate for coordinate in old_local)
        new_global.extend(candidate_offset + coordinate for coordinate in new_local)
        new_local_by_parameter.append(new_local)
        candidate_offset += new.numel()

    if not grew:
        raise UnsupportedGrowthStructure("candidate did not add parameter coordinates")

    base_was_training = base.training
    candidate_was_training = candidate.training
    base.eval()
    candidate.eval()
    try:
        with torch.no_grad():
            base_output = base(x)
            candidate_output = candidate(x)
        scale = max(1.0, float(base_output.detach().abs().max()))
        tolerance = float(preservation_tolerance) * scale
        drift = float((candidate_output - base_output).detach().abs().max())
        if not torch.isfinite(candidate_output).all() or drift > tolerance:
            raise UnsupportedGrowthStructure(
                "candidate is not function-preserving "
                f"({drift:.3e} > {tolerance:.3e})"
            )
    finally:
        base.train(base_was_training)
        candidate.train(candidate_was_training)

    return GrowthColumnLayout(
        candidate=candidate,
        old_candidate_coordinates=tuple(old_global),
        new_candidate_coordinates=tuple(new_global),
        parameter_names=tuple(name for name, _ in candidate_parameters),
        new_local_coordinates=tuple(new_local_by_parameter),
    )


def _candidate_new_jacobian_block(
    layout: GrowthColumnLayout,
    x_block: torch.Tensor,
) -> torch.Tensor:
    candidate = layout.candidate
    named_parameters = tuple(candidate.named_parameters())
    parameters = tuple(parameter for _, parameter in named_parameters)
    buffers = OrderedDict(candidate.named_buffers())
    new_values = torch.cat(
        [
            parameter.detach().reshape(-1)[list(local)]
            for parameter, local in zip(parameters, layout.new_local_coordinates)
            if local
        ]
    )

    def call_new(values: torch.Tensor) -> torch.Tensor:
        state: OrderedDict[str, torch.Tensor] = OrderedDict()
        offset = 0
        for (name, parameter), local in zip(
            named_parameters, layout.new_local_coordinates
        ):
            flat = parameter.detach().reshape(-1)
            if local:
                count = len(local)
                indices = torch.tensor(local, device=flat.device)
                flat = flat.scatter(0, indices, values[offset : offset + count])
                offset += count
            state[name] = flat.reshape(parameter.shape)
        state.update(buffers)
        return functional_call(candidate, state, (x_block,)).reshape(-1)

    rows = int(candidate(x_block).numel())
    return jacrev(call_new)(new_values).reshape(rows, -1)


def _base_jacobian_block(
    base: torch.nn.Module,
    x_block: torch.Tensor,
) -> torch.Tensor:
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in base.named_parameters()
        if parameter.requires_grad
    )
    names = tuple(name for name, _ in named_parameters)
    parameters = tuple(parameter for _, parameter in named_parameters)
    buffers = OrderedDict(base.named_buffers())

    def call_base(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        state = OrderedDict(zip(names, values))
        state.update(buffers)
        return functional_call(base, state, (x_block,)).reshape(-1)

    rows = int(base(x_block).numel())
    return _flatten_jacobian(jacrev(call_base)(parameters), rows)


def stream_shared_candidate_statistics(
    base: torch.nn.Module,
    system: ExactTangentSystem,
    layouts: tuple[GrowthColumnLayout, ...],
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> SharedBaseStatistics:
    """Accumulate exact full-probe ``B``, ``D`` and ``c`` for all candidates."""
    validate_exact_tangent_system(system, base, x, y, config)
    full_target = system.full_target
    if full_target is None:
        if system.target.numel() != int(base(x).numel()):
            raise RuntimeError("streamed tangent system has no row-aligned target")
        full_target = system.target

    base_jacobian = system.jacobian
    base_is_materialized = base_jacobian.shape[0] == full_target.numel()
    gram = base_jacobian.to(torch.float64).t() @ base_jacobian.to(torch.float64)
    rhs = base_jacobian.to(torch.float64).t() @ system.target.to(torch.float64)
    target_sq_norm = (system.target.to(torch.float64) ** 2).sum()

    cross = [
        torch.zeros(
            gram.shape[0],
            len(layout.new_candidate_coordinates),
            dtype=torch.float64,
            device=gram.device,
        )
        for layout in layouts
    ]
    new_grams = [
        torch.zeros(
            len(layout.new_candidate_coordinates),
            len(layout.new_candidate_coordinates),
            dtype=torch.float64,
            device=gram.device,
        )
        for layout in layouts
    ]
    new_rhs = [
        torch.zeros(
            len(layout.new_candidate_coordinates),
            dtype=torch.float64,
            device=gram.device,
        )
        for layout in layouts
    ]
    joint_factors: list[torch.Tensor | None] = [None for _ in layouts]

    base_was_training = base.training
    candidate_states = [layout.candidate.training for layout in layouts]
    base.eval()
    for layout in layouts:
        layout.candidate.eval()
    try:
        sample_count = int(x.shape[0])
        output_per_sample = full_target.numel() // max(sample_count, 1)
        sample_chunk = int(getattr(config, "certify_stream_chunk", 0) or 0)
        if sample_chunk <= 0:
            sample_chunk = sample_count
        for start in range(0, sample_count, sample_chunk):
            stop = min(start + sample_chunk, sample_count)
            row_start = start * output_per_sample
            row_stop = stop * output_per_sample
            x_block = x[start:stop]
            if base_is_materialized:
                j_block = base_jacobian[row_start:row_stop]
            else:
                j_block = _base_jacobian_block(base, x_block)
            j64 = j_block.to(torch.float64)
            r64 = full_target[row_start:row_stop].to(torch.float64)
            for index, layout in enumerate(layouts):
                c_block = _candidate_new_jacobian_block(layout, x_block).to(
                    torch.float64
                )
                cross[index].add_(j64.t() @ c_block)
                new_grams[index].add_(c_block.t() @ c_block)
                new_rhs[index].add_(c_block.t() @ r64)
                joint_block = torch.cat(
                    [j64, c_block, r64.unsqueeze(1)], dim=1
                )
                stacked = (
                    joint_block
                    if joint_factors[index] is None
                    else torch.cat([joint_factors[index], joint_block], dim=0)
                )
                joint_factors[index] = torch.linalg.qr(stacked, mode="reduced").R
            _clear_inaccessible_tensor_caches(base)
            for layout in layouts:
                _clear_inaccessible_tensor_caches(layout.candidate)
    finally:
        base.train(base_was_training)
        for layout, was_training in zip(layouts, candidate_states):
            layout.candidate.train(was_training)

    candidate_statistics: list[CandidateStatistics] = []
    for layout, candidate_cross, candidate_gram, candidate_rhs, joint_factor in zip(
        layouts, cross, new_grams, new_rhs, joint_factors
    ):
        if joint_factor is None:
            raise RuntimeError("candidate statistics received an empty probe")
        nonzero = torch.diagonal(candidate_gram) != 0
        indices = tuple(
            coordinate
            for coordinate, keep in zip(
                layout.new_candidate_coordinates, nonzero.tolist()
            )
            if keep
        )
        candidate_statistics.append(
            CandidateStatistics(
                layout=layout,
                new_candidate_coordinates=indices,
                cross_gram=candidate_cross[:, nonzero],
                new_gram=candidate_gram[nonzero][:, nonzero],
                new_rhs=candidate_rhs[nonzero],
                joint_factor=torch.cat(
                    [
                        joint_factor[:, : gram.shape[0]],
                        joint_factor[
                            :,
                            gram.shape[0] : gram.shape[0]
                            + len(layout.new_candidate_coordinates),
                        ][:, nonzero],
                        joint_factor[:, -1:],
                    ],
                    dim=1,
                ),
            )
        )
    return SharedBaseStatistics(
        gram=gram,
        rhs=rhs,
        target_sq_norm=target_sq_norm,
        candidates=tuple(candidate_statistics),
    )


def factor_shared_base(system: ExactTangentSystem) -> BaseSpectrum:
    """Factor the Phase 1 base representation once in float64."""
    _, singular_values, right_vectors = torch.linalg.svd(
        system.jacobian.to(torch.float64),
        full_matrices=False,
    )
    if (
        singular_values.numel() == 0
        or not torch.isfinite(singular_values).all()
        or not torch.isfinite(right_vectors).all()
    ):
        raise CandidateScoringError("nonfinite base spectrum")
    return BaseSpectrum(
        singular_values=singular_values,
        right_vectors=right_vectors,
    )


def _spectral_solve(
    spectrum: BaseSpectrum,
    rhs: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """Apply ``(G + damping I)^-1`` without forming an inverse."""
    right = spectrum.right_vectors
    coordinates = right @ rhs
    represented = right.t() @ coordinates
    denominator = spectrum.singular_values.square() + damping
    if coordinates.ndim > 1:
        denominator = denominator.unsqueeze(1)
    range_solution = right.t() @ (
        coordinates / denominator
    )
    return range_solution + (rhs - represented) / damping


def exact_block_schur_score(
    shared: SharedBaseStatistics,
    candidate: CandidateStatistics,
    spectrum: BaseSpectrum,
    *,
    eps: float,
) -> ExactCandidateScore:
    """Score one exact augmented ridge system through its block Schur solve."""
    gram = shared.gram
    rhs = shared.rhs
    cross = candidate.cross_gram
    new_gram = candidate.new_gram
    new_rhs = candidate.new_rhs
    tensors = (gram, rhs, cross, new_gram, new_rhs, shared.target_sq_norm)
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise CandidateScoringError("nonfinite sufficient statistics")
    if new_rhs.numel() == 0:
        raise CandidateScoringError("candidate has no nonzero new tangent columns")

    gram_scale = max(1.0, float(torch.linalg.norm(gram)))
    symmetry_residual = float(torch.linalg.norm(gram - gram.t())) / gram_scale
    new_scale = max(1.0, float(torch.linalg.norm(new_gram)))
    new_symmetry_residual = (
        float(torch.linalg.norm(new_gram - new_gram.t())) / new_scale
    )
    if symmetry_residual > 1e-12 or new_symmetry_residual > 1e-12:
        raise CandidateScoringError("nonsymmetric augmented Gram")

    parameter_count = gram.shape[0]
    new_count = new_rhs.numel()
    joint = candidate.joint_factor
    reduced_jacobian = joint[:, : parameter_count + new_count]
    reduced_target = joint[:, -1]
    reduced_base = joint[:, :parameter_count]
    reduced_new = joint[:, parameter_count : parameter_count + new_count]
    consistency_scale = max(
        1.0,
        float(torch.linalg.norm(cross)),
        float(torch.linalg.norm(new_gram)),
        float(torch.linalg.norm(new_rhs)),
    )
    consistency_residual = max(
        float(torch.linalg.norm(reduced_base.t() @ reduced_new - cross)),
        float(torch.linalg.norm(reduced_new.t() @ reduced_new - new_gram)),
        float(torch.linalg.norm(reduced_new.t() @ reduced_target - new_rhs)),
    ) / consistency_scale
    if consistency_residual > 2e-8:
        raise CandidateScoringError("joint factor disagrees with B, D, or c")
    candidate_singular = torch.linalg.svdvals(reduced_jacobian)
    largest = float(candidate_singular.max()) ** 2
    if not largest > 0.0 or not torch.isfinite(candidate_singular).all():
        raise CandidateScoringError("candidate tangent system is degenerate")
    damping = 1e-16 * largest

    left, base_singular, right = torch.linalg.svd(
        reduced_base, full_matrices=False
    )
    if not torch.allclose(
        base_singular,
        spectrum.singular_values,
        atol=1e-10,
        rtol=1e-8,
    ):
        raise CandidateScoringError("candidate scan changed the base spectrum")
    base_coordinates = left.t() @ reduced_new
    target_coordinates = left.t() @ reduced_target
    new_residual = reduced_new - left @ base_coordinates
    target_residual = reduced_target - left @ target_coordinates
    weights = damping / (base_singular.square() + damping)
    schur = (
        new_residual.t() @ new_residual
        + (base_coordinates * weights.unsqueeze(1)).t() @ base_coordinates
        + damping
        * torch.eye(
            new_count,
            dtype=new_gram.dtype,
            device=new_gram.device,
        )
    )
    schur = 0.5 * (schur + schur.t())
    schur_rhs = (
        new_residual.t() @ target_residual
        + base_coordinates.t() @ (weights * target_coordinates)
    )
    factor, info = torch.linalg.cholesky_ex(schur)
    if int(info.max()) != 0:
        raise CandidateScoringError("Schur Cholesky factorization failed")
    new_update = torch.cholesky_solve(
        schur_rhs.unsqueeze(1), factor
    ).squeeze(1)
    old_update = right.t() @ (
        base_singular
        / (base_singular.square() + damping)
        * (target_coordinates - base_coordinates @ new_update)
    )
    if not torch.isfinite(old_update).all() or not torch.isfinite(new_update).all():
        raise CandidateScoringError("nonfinite block-Schur solution")

    old_normal = (
        gram @ old_update
        + cross @ new_update
        + damping * old_update
        - rhs
    )
    new_normal = (
        cross.t() @ old_update
        + new_gram @ new_update
        + damping * new_update
        - new_rhs
    )
    normal_rhs_norm = max(
        1.0,
        float(torch.linalg.norm(torch.cat([rhs, new_rhs]))),
    )
    normal_residual = float(
        torch.linalg.norm(torch.cat([old_normal, new_normal]))
    ) / normal_rhs_norm
    schur_residual = float(
        torch.linalg.norm(schur @ new_update - schur_rhs)
    ) / max(1.0, float(torch.linalg.norm(schur_rhs)))
    if normal_residual > 2e-6 or schur_residual > 2e-8:
        raise CandidateScoringError("ridge or Schur residual is excessive")

    gram_dot_product = float(
        torch.dot(rhs, old_update) + torch.dot(new_rhs, new_update)
    )
    gram_approximation_sq = float(
        torch.dot(old_update, gram @ old_update)
        + 2.0 * torch.dot(old_update, cross @ new_update)
        + torch.dot(new_update, new_gram @ new_update)
    )
    flat_update = torch.cat([old_update, new_update])
    reduced_approximation = reduced_jacobian @ flat_update
    dot_product = float(torch.dot(reduced_approximation, reduced_target))
    approximation_sq = float(
        torch.dot(reduced_approximation, reduced_approximation)
    )
    solution_sq = float(
        torch.dot(old_update, old_update) + torch.dot(new_update, new_update)
    )
    alternative_sq = dot_product - damping * solution_sq
    sensor_residual = max(
        abs(approximation_sq - alternative_sq),
        abs(gram_dot_product - dot_product),
        abs(gram_approximation_sq - approximation_sq),
    ) / max(
        1.0,
        abs(approximation_sq),
        abs(alternative_sq),
        abs(dot_product),
    )
    if sensor_residual > 2e-5:
        raise CandidateScoringError(
            "reconstructed projection sensor is inconsistent "
            f"({sensor_residual:.3e})"
        )
    sensor = _output_relative_error_from_stats(
        dot_product=dot_product,
        approximation_sq_norm=approximation_sq,
        target_sq_norm=float(torch.dot(reduced_target, reduced_target)),
        eps=eps,
    )
    relative_error = float(sensor.relative_error)
    if not torch.isfinite(torch.tensor(relative_error)):
        raise CandidateScoringError("nonfinite relative error")
    error_bound = max(
        1e-11,
        10.0 * (normal_residual + schur_residual + sensor_residual),
    )
    return ExactCandidateScore(
        relative_error=relative_error,
        absolute_damping=damping,
        old_update=old_update,
        new_update=new_update,
        normal_residual=normal_residual,
        schur_residual=schur_residual,
        sensor_residual=sensor_residual,
        error_bound=error_bound,
    )
