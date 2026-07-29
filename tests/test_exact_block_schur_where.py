"""Finite-ridge and exhaustive-winner parity for exact block-Schur ``where``."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.search.certify import _grow_clone, _select_growth_candidate
from fgdlib.search.damping import DAMPING_BRACKET, minimal_relative_error_from_system
from fgdlib.search.exact_where import (
    CandidateStatistics,
    GrowthColumnLayout,
    SharedBaseStatistics,
    exact_block_schur_score,
    factor_shared_base,
    identify_growth_columns,
    stream_shared_candidate_statistics,
)
from fgdlib.tangent import (
    FGDApproxConfig,
    ExactTangentSystem,
    _output_relative_error_from_tensors,
    _solve_tangent_projection,
    exact_tangent_system,
)
from stable_tiny.pipeline import build_model, load_pipeline_config


ensure_gromo_importable()


def _problem(seed: int, *, width: int = 2, out_features: int = 2):
    torch.manual_seed(seed)
    config = load_pipeline_config("configs/experiments/default.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            in_features=3,
            out_features=out_features,
        ),
        model=replace(
            config.model,
            hidden_size=width,
            number_hidden_layers=2,
        ),
        fgd_approx=replace(
            config.fgd_approx,
            projection_fast_factorization=False,
            certify_stream_gram=False,
            certify_stream_chunk=4,
            certify_exact_block_schur_where=True,
            growth_preservation_tolerance=2e-6,
        ),
    )
    model = build_model(config, torch.device("cpu"))
    x = torch.randn(32, 3)
    y = torch.randn(32, out_features)
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)
    return config, model, x, y, loader


@pytest.mark.parametrize(
    ("seed", "width", "out_features"),
    [(0, 2, 1)],
)
def test_candidate_damping_ridge_and_relative_error_match_full_oracle(
    seed: int,
    width: int,
    out_features: int,
) -> None:
    config, model, x, y, loader = _problem(
        seed, width=width, out_features=out_features
    )
    base_system = exact_tangent_system(model, x, y, config.fgd_approx)
    assert base_system is not None
    candidates = tuple(
        candidate
        for location in range(len(model._growable_layers))
        if (
            candidate := _grow_clone(
                model,
                loader,
                location,
                torch.device("cpu"),
                config,
                True,
            )
        )
        is not None
    )
    layouts = tuple(
        identify_growth_columns(
            model,
            candidate,
            x,
            preservation_tolerance=(
                config.fgd_approx.growth_preservation_tolerance
            ),
        )
        for candidate in candidates
    )
    shared = stream_shared_candidate_statistics(
        model,
        base_system,
        layouts,
        x,
        y,
        config.fgd_approx,
    )
    spectrum = factor_shared_base(base_system)

    for candidate, statistics in zip(candidates, shared.candidates):
        score = exact_block_schur_score(
            shared,
            statistics,
            spectrum,
            eps=config.fgd_approx.eps,
        )
        full_system = exact_tangent_system(candidate, x, y, config.fgd_approx)
        assert full_system is not None
        sigma_max = float(torch.linalg.svdvals(full_system.jacobian.double()).max())
        full_damping = DAMPING_BRACKET[0] * sigma_max**2
        assert score.absolute_damping == pytest.approx(
            full_damping, rel=1e-5, abs=1e-12
        )

        update, approximation = _solve_tangent_projection(
            full_system.jacobian,
            full_system.target,
            full_damping,
            work_dtype=torch.float64,
        )
        assert torch.isfinite(update).all()
        sensor = _output_relative_error_from_tensors(
            approximation=approximation,
            target=full_system.target,
            eps=config.fgd_approx.eps,
        )
        assert score.relative_error == pytest.approx(
            sensor.output_error.relative_error,
            rel=1e-5,
            abs=1e-7,
        )


@pytest.mark.parametrize(("seed", "width"), [(2, 2), (7, 3), (11, 4)])
def test_exact_where_winner_matches_original_exhaustive_oracle(
    seed: int,
    width: int,
) -> None:
    config, model, x, y, loader = _problem(seed, width=width)
    base_system = exact_tangent_system(model, x, y, config.fgd_approx)
    assert base_system is not None
    current = minimal_relative_error_from_system(base_system, config.fgd_approx)

    fast = _select_growth_candidate(
        model,
        base_system,
        x,
        y,
        loader,
        torch.device("cpu"),
        config,
        True,
        current,
    )
    oracle = _select_growth_candidate(
        model,
        base_system,
        x,
        y,
        loader,
        torch.device("cpu"),
        replace(
            config,
            fgd_approx=replace(
                config.fgd_approx,
                certify_exact_block_schur_where=False,
            ),
        ),
        True,
        current,
    )
    assert fast[2] == oracle[2]
    assert fast[1] == pytest.approx(oracle[1], rel=1e-9, abs=1e-11)
    assert fast[3] is not None


@pytest.mark.parametrize(("rows", "base_columns", "new_columns"), [(40, 5, 2), (64, 9, 4)])
def test_block_schur_solution_matches_direct_augmented_ridge(
    rows: int,
    base_columns: int,
    new_columns: int,
) -> None:
    torch.manual_seed(rows + base_columns)
    j_base = torch.randn(rows, base_columns, dtype=torch.float64)
    new = torch.randn(rows, new_columns, dtype=torch.float64)
    target = torch.randn(rows, dtype=torch.float64)
    joint = torch.cat([j_base, new, target.unsqueeze(1)], dim=1)
    joint_factor = torch.linalg.qr(joint, mode="reduced").R
    layout = GrowthColumnLayout(
        candidate=torch.nn.Identity(),
        old_candidate_coordinates=tuple(range(base_columns)),
        new_candidate_coordinates=tuple(
            range(base_columns, base_columns + new_columns)
        ),
        parameter_names=(),
        new_local_coordinates=(),
    )
    statistics = CandidateStatistics(
        layout=layout,
        new_candidate_coordinates=layout.new_candidate_coordinates,
        cross_gram=j_base.t() @ new,
        new_gram=new.t() @ new,
        new_rhs=new.t() @ target,
        joint_factor=joint_factor,
    )
    shared = SharedBaseStatistics(
        gram=j_base.t() @ j_base,
        rhs=j_base.t() @ target,
        target_sq_norm=torch.dot(target, target),
        candidates=(statistics,),
    )
    system = ExactTangentSystem(
        jacobian=j_base,
        target=target,
        parameters=(),
        loss=0.0,
        full_target=target,
    )
    score = exact_block_schur_score(
        shared,
        statistics,
        factor_shared_base(system),
        eps=1e-12,
    )
    augmented = torch.cat([j_base, new], dim=1)
    damping = 1e-16 * float(torch.linalg.svdvals(augmented).max()) ** 2
    direct = torch.linalg.solve(
        augmented.t() @ augmented
        + damping
        * torch.eye(base_columns + new_columns, dtype=torch.float64),
        augmented.t() @ target,
    )
    actual = torch.cat([score.old_update, score.new_update])
    assert torch.allclose(actual, direct, atol=1e-10, rtol=1e-9)


def test_nearly_duplicate_columns_remain_finite_and_exact() -> None:
    """The finite ridge filter, not a hard-rank projector, handles duplicates."""
    torch.manual_seed(41)
    j_base = torch.randn(24, 5, dtype=torch.float64)
    duplicate = j_base[:, :1] + 1e-10 * torch.randn(24, 1, dtype=torch.float64)
    extra = torch.cat([duplicate, torch.randn(24, 1, dtype=torch.float64)], dim=1)
    target = torch.randn(24, dtype=torch.float64)
    augmented = torch.cat([j_base, extra], dim=1)
    singular = torch.linalg.svdvals(augmented)
    damping = 1e-16 * float(singular.max()) ** 2
    gram = augmented.t() @ augmented
    rhs = augmented.t() @ target
    direct = torch.linalg.solve(
        gram + damping * torch.eye(gram.shape[0], dtype=torch.float64),
        rhs,
    )
    assert torch.isfinite(direct).all()
    normal_residual = torch.linalg.norm(
        (gram + damping * torch.eye(gram.shape[0], dtype=torch.float64)) @ direct
        - rhs
    )
    assert float(normal_residual) < 1e-9


def test_environment_flag_is_benchmark_only(monkeypatch) -> None:
    from fgdlib.search.certify import _exact_block_schur_where_enabled

    config = FGDApproxConfig()
    assert config.certify_exact_block_schur_where is False
    monkeypatch.delenv("FGD_EXACT_BLOCK_SCHUR_WHERE", raising=False)
    assert not _exact_block_schur_where_enabled(config)
    monkeypatch.setenv("FGD_EXACT_BLOCK_SCHUR_WHERE", "1")
    assert _exact_block_schur_where_enabled(config)
