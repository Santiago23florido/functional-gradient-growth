"""The projection, its sensor, and the certificate boolean, across backends.

Section 8.E asks for parity of everything the outer step reads off the
tangent projection -- the parameter update, the functional approximation, the
dot product, both squared norms, the relative error and the strict
certificate boolean -- and 8.F asks that the degenerate systems each take the
fallback they are documented to take, with the counter and reason to prove
it.

8.F is deliberately built from SYNTHETIC factors with prescribed spectra
rather than from models that happen to be ill-conditioned. Rank deficiency,
the [1e-13, 1e-11] * sigma_max ambiguity band and a singular value sitting on
the damping scale are all properties of the spectrum, and a model-based
fixture would only reach them by accident -- and would stop reaching them the
day the fixture's seed changed.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import replace

import pytest
import torch
from torch.func import functional_call, jacrev

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.profile import _REASONS, reset, snapshot
from fgdlib.search.damping import minimal_relative_error_from_system
from fgdlib.tangent import (
    ExactTangentSystem,
    FGDApproxConfig,
    _compute_exact_tangent_projection_step,
    _flatten_jacobian,
    _stream_gram_surrogate,
    _surrogate_from_factor,
    _trainable_named_parameters,
)

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP

IN_FEATURES = 5
OUT_FEATURES = 3
SAMPLES = 40
OUT_OF_RANGE = 0.3

#: Synthetic factors: enough rows that R is square and full-height.
SYNTHETIC_ROWS = 90
SYNTHETIC_COLUMNS = 30


@pytest.fixture
def profiling(monkeypatch):
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    yield
    reset()


# --------------------------------------------------------------------------
# 8.E -- the projection step itself
# --------------------------------------------------------------------------


def _model(dtype: torch.dtype, seed: int, hidden: int, depth: int) -> GrowingMLP:
    torch.manual_seed(seed)
    model = GrowingMLP(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_size=hidden,
        number_hidden_layers=depth,
        device=torch.device("cpu"),
    )
    model.eval()
    return model.to(dtype)


def _full_jacobian(model: GrowingMLP, x: torch.Tensor) -> torch.Tensor:
    named = _trainable_named_parameters(model)
    names = tuple(named)
    parameters = tuple(named.values())
    buffers = OrderedDict(model.named_buffers())

    def call(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        state = OrderedDict(zip(names, values))
        state.update(buffers)
        return functional_call(model, state, (x,)).reshape(-1)

    with torch.no_grad():
        rows = model(x).numel()
    return _flatten_jacobian(jacrev(call)(parameters), rows).to(torch.float64)


def _certifying_probe(
    dtype: torch.dtype = torch.float64,
    seed: int = 3,
    probe_seed: int = 11,
    hidden: int = 4,
    depth: int = 2,
):
    """A probe whose residual mostly lies in the tangent range, so the
    certificate is a live decision rather than a foregone rejection."""
    model = _model(dtype, seed, hidden, depth)
    torch.manual_seed(probe_seed)
    x = torch.randn(SAMPLES, IN_FEATURES, dtype=dtype)
    jacobian = _full_jacobian(model, x)
    torch.manual_seed(probe_seed + 100)
    coefficients = torch.randn(jacobian.shape[1], dtype=torch.float64)
    direction = (jacobian @ coefficients).reshape(SAMPLES, OUT_FEATURES)
    direction = direction / direction.norm()
    generic = torch.randn(SAMPLES, OUT_FEATURES, dtype=torch.float64)
    generic = generic / generic.norm()
    with torch.no_grad():
        prediction = model(x).to(torch.float64)
    y = (prediction - (direction + OUT_OF_RANGE * generic)).to(dtype)
    return model, x, y


def _config(**overrides) -> FGDApproxConfig:
    return replace(
        FGDApproxConfig(
            projection_solver="exact",
            functional_loss="mse",
            certify_stream_gram=True,
            certify_stream_chunk=8,
        ),
        **overrides,
    )


def _step(monkeypatch, backend: str, **probe_kwargs):
    monkeypatch.setenv("FGD_TANGENT_BACKEND", backend)
    model, x, y = _certifying_probe(**probe_kwargs)
    step = _compute_exact_tangent_projection_step(
        model=model, x=x, y=y, config=_config(), return_system=False
    )
    assert step is not None
    return model, step


@pytest.mark.parametrize("probe_seed", [11, 12, 13])
@pytest.mark.parametrize("seed", [0, 3])
def test_projection_update_and_relative_error_match_between_backends(
    monkeypatch, seed: int, probe_seed: int
) -> None:
    """Every field the outer step reads, not just the headline eps.

    ``dot_product``, ``approximation_sq_norm`` and ``target_sq_norm`` are
    the three numbers the sensor is built from; a backend that matched only
    the ratio would pass a RelErr comparison and still hand the acceptance
    logic different evidence.
    """
    _, legacy = _step(monkeypatch, "legacy", seed=seed, probe_seed=probe_seed)
    _, optimized = _step(monkeypatch, "optimized", seed=seed, probe_seed=probe_seed)

    assert optimized.dot_product == pytest.approx(legacy.dot_product, rel=1e-9)
    assert optimized.approximation_sq_norm == pytest.approx(
        legacy.approximation_sq_norm, rel=1e-9
    )
    assert optimized.target_sq_norm == pytest.approx(legacy.target_sq_norm, rel=1e-9)

    left = legacy.output_error
    right = optimized.output_error
    assert right.relative_error == pytest.approx(left.relative_error, abs=1e-9)
    assert right.approximation_norm == pytest.approx(left.approximation_norm, rel=1e-9)
    assert right.target_norm == pytest.approx(left.target_norm, rel=1e-9)
    assert right.directional_cosine == pytest.approx(left.directional_cosine, abs=1e-9)

    assert len(optimized.parameter_updates) == len(legacy.parameter_updates)
    for update, reference in zip(optimized.parameter_updates, legacy.parameter_updates):
        assert update.shape == reference.shape
        assert update.dtype == reference.dtype
        assert torch.allclose(update, reference, rtol=1e-7, atol=1e-9)


def test_parameter_update_lands_on_the_named_parameters_in_order(
    monkeypatch,
) -> None:
    """A projection whose update tensors matched numerically but were
    unflattened in a different order would corrupt the model silently."""
    model, step = _step(monkeypatch, "optimized")
    named = _trainable_named_parameters(model)
    assert len(step.parameter_updates) == len(named)
    for update, parameter in zip(step.parameter_updates, named.values()):
        assert update.shape == parameter.shape


@pytest.mark.parametrize("probe_seed", [11, 12, 13])
@pytest.mark.parametrize(
    "hidden,depth", [(4, 2), (6, 1)], ids=["hidden4-depth2", "hidden6-depth1"]
)
def test_certificate_boolean_is_identical_between_backends(
    monkeypatch, hidden: int, depth: int, probe_seed: int
) -> None:
    """The certificate is a STRICT ``eps < 1/2``.

    A 1e-7 drift can only flip it on an exact tie, which is why the test
    checks the boolean AND the margin: if the two backends ever land on
    opposite sides of the threshold, the margin assertion says how close the
    call was, instead of leaving a bare ``False != True``.
    """
    threshold = 0.5
    _, legacy = _step(
        monkeypatch, "legacy", hidden=hidden, depth=depth, probe_seed=probe_seed
    )
    _, optimized = _step(
        monkeypatch, "optimized", hidden=hidden, depth=depth, probe_seed=probe_seed
    )
    left = legacy.output_error.relative_error
    right = optimized.output_error.relative_error
    assert left is not None and right is not None
    margin = abs(left - threshold)
    assert abs(right - left) <= 1e-9
    assert abs(right - left) < margin, "the two backends straddle the threshold"
    assert (left < threshold) == (right < threshold)


# --------------------------------------------------------------------------
# 8.F -- degenerate systems take the documented branch
# --------------------------------------------------------------------------


def _jacobian_with_spectrum(values: torch.Tensor, seed: int) -> torch.Tensor:
    """A rows x columns matrix whose singular values are exactly ``values``."""
    columns = values.numel()
    torch.manual_seed(seed)
    left = torch.linalg.qr(torch.randn(SYNTHETIC_ROWS, columns, dtype=torch.float64)).Q
    right = torch.linalg.qr(torch.randn(columns, columns, dtype=torch.float64)).Q
    return (left * values) @ right.t()


def _streamed(jacobian: torch.Tensor, residual: torch.Tensor):
    r_factor = torch.linalg.qr(jacobian, mode="reduced").R
    return r_factor, jacobian.t() @ residual, float((residual * residual).sum())


def _residual_for(jacobian: torch.Tensor, out_of_range: float, seed: int):
    """``a + out_of_range * n`` with ``a`` in range(J) and ``n`` generic."""
    torch.manual_seed(seed)
    coefficients = torch.randn(jacobian.shape[1], dtype=torch.float64)
    in_range = jacobian @ coefficients
    in_range = in_range / in_range.norm()
    generic = torch.randn(jacobian.shape[0], dtype=torch.float64)
    generic = generic / generic.norm()
    return in_range + out_of_range * generic


def _min_relative_error(jacobian: torch.Tensor, target: torch.Tensor) -> float:
    return minimal_relative_error_from_system(
        ExactTangentSystem(jacobian=jacobian, target=target, parameters=(), loss=0.0),
        FGDApproxConfig(),
    )


def _both_surrogates(r_factor, b_acc, r_sq):
    optimized = _surrogate_from_factor(
        r_factor, b_acc, r_sq, torch.float64, strict=True
    )
    legacy = _stream_gram_surrogate(r_factor, b_acc, r_sq, torch.float64)
    return optimized, legacy


def _assert_equivalent(optimized, legacy, *, rtol=1e-9, atol=1e-11) -> None:
    """Same shape, same sufficient statistics, same minimum relative error."""
    j_opt, r_opt = optimized
    j_leg, r_leg = legacy
    assert j_opt.shape == j_leg.shape
    assert r_opt.shape == r_leg.shape
    assert torch.allclose(j_opt.t() @ j_opt, j_leg.t() @ j_leg, rtol=rtol, atol=atol)
    assert torch.allclose(j_opt.t() @ r_opt, j_leg.t() @ r_leg, rtol=rtol, atol=atol)
    assert float((r_opt * r_opt).sum()) == pytest.approx(
        float((r_leg * r_leg).sum()), rel=rtol, abs=atol
    )
    assert _min_relative_error(j_opt, r_opt) == pytest.approx(
        _min_relative_error(j_leg, r_leg), abs=1e-9
    )


def test_duplicate_columns_take_the_svd_surrogate_and_still_match(
    profiling,
) -> None:
    """Exactly repeated columns make R singular, so ``z = R^{-T} b`` does not
    exist and the SVD-free route MUST NOT be taken: reason
    ``rank_deficient``, plus the rank-deficiency counter."""
    torch.manual_seed(21)
    base = torch.randn(SYNTHETIC_ROWS, SYNTHETIC_COLUMNS, dtype=torch.float64)
    base[:, 5] = base[:, 2]
    base[:, 9] = base[:, 2]
    residual = _residual_for(base, 0.4, seed=22)
    r_factor, b_acc, r_sq = _streamed(base, residual)

    optimized, legacy = _both_surrogates(r_factor, b_acc, r_sq)
    values = snapshot()
    assert values["tangent_surrogate_svd_fallbacks"] == 1
    assert values["tangent_rank_deficient_surrogates"] == 1
    assert _REASONS["tangent_surrogate_svd_fallbacks"] == {"rank_deficient"}
    assert optimized[0].shape[0] < SYNTHETIC_COLUMNS + 1  # rank+1 rows, rank < P
    _assert_equivalent(optimized, legacy, rtol=1e-7, atol=1e-9)


def test_zero_columns_take_the_svd_surrogate_and_still_match(profiling) -> None:
    """A dead parameter (an entire zero column) is the rank deficiency a
    grown network actually produces: the new neuron's outgoing weight is
    zero, so its column of J is zero until the first step moves it."""
    torch.manual_seed(23)
    base = torch.randn(SYNTHETIC_ROWS, SYNTHETIC_COLUMNS, dtype=torch.float64)
    base[:, 7] = 0.0
    base[:, 11] = 0.0
    residual = _residual_for(base, 0.4, seed=24)
    r_factor, b_acc, r_sq = _streamed(base, residual)

    optimized, legacy = _both_surrogates(r_factor, b_acc, r_sq)
    assert snapshot()["tangent_rank_deficient_surrogates"] == 1
    assert _REASONS["tangent_surrogate_svd_fallbacks"] == {"rank_deficient"}
    _assert_equivalent(optimized, legacy, rtol=1e-7, atol=1e-9)


def test_nearly_duplicate_columns_fall_back_as_threshold_ambiguous(
    profiling,
) -> None:
    """A singular value inside [1e-13, 1e-11] * sigma_max is one CPU-vs-CUDA
    LAPACK last-digit away from changing the rank decision, so the fast route
    refuses it EVEN THOUGH the rank test says full rank -- reason
    ``threshold_ambiguous``, and no rank-deficiency counter."""
    values = torch.ones(SYNTHETIC_COLUMNS, dtype=torch.float64)
    values[0] = 4.0
    values[-1] = 4.0 * 5e-12  # inside the band, above the 1e-12 rank cut
    jacobian = _jacobian_with_spectrum(values, seed=25)
    residual = _residual_for(jacobian, 0.4, seed=26)
    r_factor, b_acc, r_sq = _streamed(jacobian, residual)

    spectrum = torch.linalg.svdvals(r_factor)
    ratio = float(spectrum[-1]) / float(spectrum[0])
    assert 1e-13 <= ratio <= 1e-11, ratio

    optimized, legacy = _both_surrogates(r_factor, b_acc, r_sq)
    snap = snapshot()
    assert snap["tangent_surrogate_svd_fallbacks"] == 1
    assert snap["tangent_rank_deficient_surrogates"] == 0
    assert _REASONS["tangent_surrogate_svd_fallbacks"] == {"threshold_ambiguous"}
    assert optimized[0].shape == legacy[0].shape


def test_singular_values_at_the_damping_scale_take_the_fast_route(
    profiling,
) -> None:
    """``minimal_relative_error_from_system`` damps at 1e-16 * sigma_max^2,
    i.e. sigma ~ 1e-8 * sigma_max. A spectrum reaching exactly there is
    fully in range for the rank test (1e-8 >> 1e-12) and must take the
    SVD-free route -- no fallback at all -- while still reproducing the
    legacy answer through the filtered tail."""
    values = torch.logspace(0, -8, SYNTHETIC_COLUMNS, dtype=torch.float64)
    jacobian = _jacobian_with_spectrum(values, seed=27)
    residual = _residual_for(jacobian, 0.4, seed=28)
    r_factor, b_acc, r_sq = _streamed(jacobian, residual)

    optimized, legacy = _both_surrogates(r_factor, b_acc, r_sq)
    snap = snapshot()
    assert snap["tangent_surrogate_svd_fallbacks"] == 0
    assert snap["tangent_numerical_fallbacks"] == 0
    assert snap["tangent_rank_deficient_surrogates"] == 0
    assert _REASONS == {}
    assert optimized[0].shape == (SYNTHETIC_COLUMNS + 1, SYNTHETIC_COLUMNS)
    _assert_equivalent(optimized, legacy, rtol=1e-6, atol=1e-9)


#: eps floor for an almost-perfectly-explained residual. ``res = ||r||^2 -
#: in_range`` is then a total cancellation of two float64 numbers, so its
#: relative error is O(1) and eps = sqrt(res/in_range) bottoms out around
#: sqrt(2^-52) ~ 1.5e-8 whichever way it is computed. MEASURED on the fixture
#: below: 0.0 for the SVD-free route (the max(0, .) clamp wins) against
#: 2.1e-8 for the legacy SVD route. Both mean "indistinguishable from an
#: exact fit"; neither is more correct than the other, and no tolerance below
#: this floor is meaningful.
CANCELLATION_EPS_FLOOR = 1e-7


def test_very_small_residual_outside_the_range_stays_non_negative(
    profiling,
) -> None:
    """``res = ||r||^2 - ||z||^2`` is a DIFFERENCE of two nearly equal
    numbers when the residual is almost entirely explainable. The clamp at
    zero is what keeps ``sqrt(res)`` defined; both surrogates must agree that
    eps is indistinguishable from zero rather than one of them producing a
    NaN or a negative under the square root.

    The agreement band here is ``CANCELLATION_EPS_FLOOR``, not the float64
    band used everywhere else in this file, and the constant's comment says
    why: below it the two routes are comparing rounding noise.
    """
    torch.manual_seed(29)
    jacobian = torch.randn(SYNTHETIC_ROWS, SYNTHETIC_COLUMNS, dtype=torch.float64)
    residual = _residual_for(jacobian, 1e-9, seed=30)
    r_factor, b_acc, r_sq = _streamed(jacobian, residual)

    (j_opt, r_opt), (j_leg, r_leg) = _both_surrogates(r_factor, b_acc, r_sq)
    assert torch.isfinite(r_opt).all()
    assert float(r_opt[-1]) >= 0.0
    assert float(r_leg[-1]) >= 0.0
    optimized = _min_relative_error(j_opt, r_opt)
    legacy = _min_relative_error(j_leg, r_leg)
    assert 0.0 <= optimized <= CANCELLATION_EPS_FLOOR
    assert 0.0 <= legacy <= CANCELLATION_EPS_FLOOR
    assert abs(optimized - legacy) <= CANCELLATION_EPS_FLOOR
    assert snapshot()["tangent_numerical_fallbacks"] == 0


def test_exactly_zero_residual_is_handled_by_both_surrogates(profiling) -> None:
    """``r`` entirely inside the range: ``res`` comes out at or below zero
    from rounding alone, and the clamp is the only thing between the
    certificate and a NaN eps."""
    torch.manual_seed(31)
    jacobian = torch.randn(SYNTHETIC_ROWS, SYNTHETIC_COLUMNS, dtype=torch.float64)
    coefficients = torch.randn(SYNTHETIC_COLUMNS, dtype=torch.float64)
    residual = jacobian @ coefficients
    r_factor, b_acc, r_sq = _streamed(jacobian, residual)

    (j_opt, r_opt), (j_leg, r_leg) = _both_surrogates(r_factor, b_acc, r_sq)
    assert torch.isfinite(r_opt).all() and torch.isfinite(r_leg).all()
    assert float(r_opt[-1]) >= 0.0
    assert float(r_leg[-1]) >= 0.0
    assert _min_relative_error(j_opt, r_opt) <= CANCELLATION_EPS_FLOOR
    assert _min_relative_error(j_leg, r_leg) <= CANCELLATION_EPS_FLOOR


def test_nearly_tied_certificate_values_do_not_flip_between_backends(
    profiling,
) -> None:
    """Drive eps onto the threshold and check the tie does not break.

    Risk #4 of the plan: a 1e-7 analytic-vs-jacrev drift perturbing a
    certificate sitting exactly on 1/2. Here the out-of-range residual is
    bisected until the legacy eps is within 1e-9 of 0.5, and the two
    surrogates are then required to agree by MORE than that margin -- so the
    strict ``<`` cannot land on opposite sides.
    """
    torch.manual_seed(33)
    jacobian = torch.randn(SYNTHETIC_ROWS, SYNTHETIC_COLUMNS, dtype=torch.float64)

    def legacy_eps(scale: float) -> float:
        residual = _residual_for(jacobian, scale, seed=34)
        r_factor, b_acc, r_sq = _streamed(jacobian, residual)
        j_leg, r_leg = _stream_gram_surrogate(r_factor, b_acc, r_sq, torch.float64)
        return _min_relative_error(j_leg, r_leg)

    low, high = 0.0, 4.0
    margin = float("inf")
    for _ in range(80):
        middle = 0.5 * (low + high)
        if legacy_eps(middle) < 0.5:
            low = middle
            margin = 0.5 - legacy_eps(low)
            # Stop at a REALISTIC tie. Bisecting to convergence drives the
            # margin to ~4e-16, below the ~9e-16 at which the two float64
            # surrogates differ, and at that point the strict ``<`` really
            # can land on opposite sides -- the measure-zero tie the plan's
            # risk #4 names and the reason a certificate is never claimed to
            # be exact at 2^-52. What has to hold is that a tie a MILLION
            # times wider than the disagreement does not flip.
            if margin <= 1e-9:
                break
        else:
            high = middle
    residual = _residual_for(jacobian, low, seed=34)
    r_factor, b_acc, r_sq = _streamed(jacobian, residual)
    (j_opt, r_opt), (j_leg, r_leg) = _both_surrogates(r_factor, b_acc, r_sq)

    eps_legacy = _min_relative_error(j_leg, r_leg)
    eps_optimized = _min_relative_error(j_opt, r_opt)
    margin = abs(eps_legacy - 0.5)
    assert 1e-13 < margin <= 1e-9, (
        f"the fixture is not tied in the intended window: margin {margin:.3e}"
    )
    assert abs(eps_optimized - eps_legacy) <= 1e-12
    assert abs(eps_optimized - eps_legacy) < margin
    assert (eps_legacy < 0.5) == (eps_optimized < 0.5)


def test_ill_conditioned_factor_matches_legacy(profiling) -> None:
    """cond(J) = 1e10: full rank by the 1e-12 rule, nowhere near the
    ambiguity band, so the SVD-free route runs -- on the kind of spectrum
    where the triangular solve is the part most likely to lose accuracy."""
    values = torch.logspace(0, -10, SYNTHETIC_COLUMNS, dtype=torch.float64)
    jacobian = _jacobian_with_spectrum(values, seed=35)
    residual = _residual_for(jacobian, 0.4, seed=36)
    r_factor, b_acc, r_sq = _streamed(jacobian, residual)

    optimized, legacy = _both_surrogates(r_factor, b_acc, r_sq)
    assert snapshot()["tangent_surrogate_svd_fallbacks"] == 0
    _assert_equivalent(optimized, legacy, rtol=1e-5, atol=1e-8)


def test_svd_free_surrogate_reproduces_the_exact_in_range_energy() -> None:
    """The arbiter for ``[R; 0], z = R^{-T} b`` against
    ``[diag(sigma) V^T; 0]``: both are just two ways of computing
    ``in_range = b^T (R^T R)^{-1} b``, and the certificate is
    ``sqrt((||r||^2 - in_range)/in_range)``.

    MEASURED on the production CIFAR system (P=2092, cond 4.78e3) against an
    80-bit reference: the triangular solve is exact to 0.0 relative (CPU) /
    1.7e-16 (CUDA), the CPU LAPACK SVD to 0.0, and the CUDA default SVD
    driver to 3.9e-13. The SVD-free route is the MORE accurate of the two,
    not a looser approximation of it.
    """
    torch.manual_seed(37)
    jacobian = torch.randn(SYNTHETIC_ROWS, SYNTHETIC_COLUMNS, dtype=torch.float64)
    residual = _residual_for(jacobian, 0.4, seed=38)
    r_factor, b_acc, r_sq = _streamed(jacobian, residual)

    solved = torch.linalg.solve_triangular(
        r_factor.t(), b_acc.unsqueeze(1), upper=False
    ).squeeze(1)
    in_range = float((solved * solved).sum())
    expected = math.sqrt((r_sq - in_range) / in_range)

    (j_opt, r_opt), (j_leg, r_leg) = _both_surrogates(r_factor, b_acc, r_sq)
    assert _min_relative_error(j_opt, r_opt) == pytest.approx(expected, rel=1e-6)
    assert _min_relative_error(j_leg, r_leg) == pytest.approx(expected, rel=1e-6)
