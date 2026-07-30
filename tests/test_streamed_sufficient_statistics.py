"""The streamed surrogate is only allowed to change the REPRESENTATION.

``certify_stream_gram`` never materialises the NK x P Jacobian. It returns a
tiny stand-in ``(J_s, r_s)`` and every downstream consumer -- the damping
search, the certificate, the realise path, the candidate cross-statistics --
reads the data only through four quantities:

    G = J^T J,   b = J^T r,   ||r||^2,   and the ROW COUNT.

The row count is not decoration: ``select_projection_damping`` uses
``target.numel()`` as GCV's ``n_observations``, so a surrogate with a
different number of rows would silently change the leave-one-out risk and
therefore which damping wins. Any two surrogates agreeing on those four are
related by an orthogonal left factor and are interchangeable; any two that
do not are different problems.

These tests pin all four against the FULL Jacobian, for both backends, and
pin that the optimized surrogate is orthogonally equivalent to the legacy
one by explicitly constructing the orthogonal matrix relating them.

Tolerances: the model is float64 here, so the surrogate is built and stored
in float64 and the float64 band (rtol=1e-9, atol=1e-11) applies. The
float32-model case is covered separately at its own MEASURED band, because
``out_dtype`` follows the model's own dtype.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import replace

import pytest
import torch
from torch.func import functional_call, jacrev

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.tangent import (
    FGDApproxConfig,
    _flatten_jacobian,
    _trainable_named_parameters,
    exact_tangent_system,
)

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP

IN_FEATURES = 5
OUT_FEATURES = 3
HIDDEN = 4
#: N*K = 120 > P = 59, so the streamed R is square and full rank -- the
#: regime the SVD-free surrogate is designed for.
SAMPLES = 40

RTOL64 = 1e-9
ATOL64 = 1e-11


def _model(dtype: torch.dtype = torch.float64) -> GrowingMLP:
    torch.manual_seed(3)
    model = GrowingMLP(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_size=HIDDEN,
        number_hidden_layers=2,
        device=torch.device("cpu"),
    )
    model.eval()
    return model.to(dtype)


def _probe(dtype: torch.dtype = torch.float64):
    torch.manual_seed(11)
    return (
        torch.randn(SAMPLES, IN_FEATURES, dtype=dtype),
        torch.randn(SAMPLES, OUT_FEATURES, dtype=dtype),
    )


def _config(chunk: int = 8, **overrides) -> FGDApproxConfig:
    return replace(
        FGDApproxConfig(
            projection_solver="exact",
            functional_loss="mse",
            certify_stream_gram=True,
            certify_stream_chunk=chunk,
        ),
        **overrides,
    )


def _full_jacobian(model: GrowingMLP, x: torch.Tensor) -> torch.Tensor:
    """The NK x P Jacobian the surrogate stands in for, in float64."""
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


def _statistics(jacobian: torch.Tensor, target: torch.Tensor):
    jacobian64 = jacobian.to(torch.float64)
    target64 = target.reshape(-1).to(torch.float64)
    return (
        jacobian64.t() @ jacobian64,
        jacobian64.t() @ target64,
        float((target64 * target64).sum()),
        int(jacobian64.shape[0]),
    )


def _system(backend: str, monkeypatch, *, chunk: int = 8, dtype=torch.float64):
    monkeypatch.setenv("FGD_TANGENT_BACKEND", backend)
    model = _model(dtype)
    x, y = _probe(dtype)
    system = exact_tangent_system(model, x, y, _config(chunk))
    assert system is not None
    return model, x, y, system


# --------------------------------------------------------------------------
# 8.C -- G, b, ||r||^2 against the full Jacobian
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["legacy", "optimized"])
def test_streamed_gram_rhs_and_residual_match_the_full_jacobian(
    monkeypatch, backend: str
) -> None:
    model, x, _, system = _system(backend, monkeypatch)
    assert system.full_target is not None

    jacobian = _full_jacobian(model, x)
    target = system.full_target.reshape(-1).to(torch.float64)
    gram, rhs, residual_sq, rows = _statistics(jacobian, target)
    assert rows == SAMPLES * OUT_FEATURES

    gram_s, rhs_s, residual_sq_s, _ = _statistics(system.jacobian, system.target)
    assert torch.allclose(gram_s, gram, rtol=RTOL64, atol=ATOL64)
    assert torch.allclose(rhs_s, rhs, rtol=RTOL64, atol=ATOL64)
    assert residual_sq_s == pytest.approx(residual_sq, rel=RTOL64, abs=ATOL64)


@pytest.mark.parametrize("backend", ["legacy", "optimized"])
def test_streamed_surrogate_is_far_smaller_than_the_jacobian_it_replaces(
    monkeypatch, backend: str
) -> None:
    """The surrogate exists to be O(P^2), not O(NKP). If it ever came back
    with NK rows the memory argument would be gone and nothing else in the
    suite would notice."""
    model, _, _, system = _system(backend, monkeypatch)
    parameter_numel = sum(
        parameter.numel() for parameter in _trainable_named_parameters(model).values()
    )
    assert system.jacobian.shape[1] == parameter_numel
    assert system.jacobian.shape[0] <= parameter_numel + 1
    assert system.jacobian.shape[0] < SAMPLES * OUT_FEATURES


def test_surrogate_row_count_is_preserved_between_backends(monkeypatch) -> None:
    """GCV reads ``target.numel()`` as ``n_observations``; a different row
    count is a different risk estimate and therefore a different selected
    damping, even with identical G, b and ||r||^2."""
    _, _, _, legacy = _system("legacy", monkeypatch)
    _, _, _, optimized = _system("optimized", monkeypatch)
    assert legacy.jacobian.shape == optimized.jacobian.shape
    assert legacy.target.numel() == optimized.target.numel()
    assert legacy.target.shape == optimized.target.shape


def test_surrogate_is_orthogonally_equivalent_to_the_legacy_one(
    monkeypatch,
) -> None:
    """Construct the orthogonal matrix, do not merely assert the invariants.

    ``[R; 0]`` and ``[diag(sigma) V^T; 0]`` are claimed to differ by exactly
    one orthogonal left factor. Equal Grams already imply SOME such factor
    exists; building it and checking ``Q^T Q = I`` and ``Q [J_leg | r_leg] =
    [J_opt | r_opt]`` proves the SAME factor carries the residual too, which
    is what makes the pair -- not just the Jacobian -- interchangeable.
    """
    _, _, _, legacy = _system("legacy", monkeypatch)
    _, _, _, optimized = _system("optimized", monkeypatch)

    left = torch.cat(
        [legacy.jacobian.to(torch.float64), legacy.target.reshape(-1, 1).double()],
        dim=1,
    )
    right = torch.cat(
        [
            optimized.jacobian.to(torch.float64),
            optimized.target.reshape(-1, 1).double(),
        ],
        dim=1,
    )
    assert left.shape == right.shape and left.shape[0] == left.shape[1]

    rotation = torch.linalg.solve(left.t(), right.t()).t()
    identity = torch.eye(rotation.shape[0], dtype=torch.float64)
    assert torch.allclose(rotation.t() @ rotation, identity, rtol=1e-8, atol=1e-8)
    assert torch.allclose(rotation @ left, right, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("chunk", [1, 3, 7, SAMPLES])
@pytest.mark.parametrize("backend", ["legacy", "optimized"])
def test_chunk_count_does_not_change_the_statistics(
    monkeypatch, backend: str, chunk: int
) -> None:
    """``certify_stream_chunk`` is a memory knob, not a modelling choice: the
    incremental QR must accumulate the same G, b and ||r||^2 whatever the
    grouping."""
    _, _, _, reference = _system(backend, monkeypatch, chunk=SAMPLES)
    _, _, _, system = _system(backend, monkeypatch, chunk=chunk)

    gram_ref, rhs_ref, rsq_ref, _ = _statistics(reference.jacobian, reference.target)
    gram, rhs, rsq, _ = _statistics(system.jacobian, system.target)
    assert torch.allclose(gram, gram_ref, rtol=1e-8, atol=1e-9)
    assert torch.allclose(rhs, rhs_ref, rtol=1e-8, atol=1e-9)
    assert rsq == pytest.approx(rsq_ref, rel=1e-9)
    assert system.jacobian.shape == reference.jacobian.shape


def test_backends_agree_on_the_sufficient_statistics(monkeypatch) -> None:
    _, _, _, legacy = _system("legacy", monkeypatch)
    _, _, _, optimized = _system("optimized", monkeypatch)
    gram_l, rhs_l, rsq_l, rows_l = _statistics(legacy.jacobian, legacy.target)
    gram_o, rhs_o, rsq_o, rows_o = _statistics(optimized.jacobian, optimized.target)
    assert rows_l == rows_o
    assert torch.allclose(gram_o, gram_l, rtol=RTOL64, atol=ATOL64)
    assert torch.allclose(rhs_o, rhs_l, rtol=RTOL64, atol=ATOL64)
    assert rsq_o == pytest.approx(rsq_l, rel=RTOL64, abs=ATOL64)


def test_residual_energy_outside_the_row_space_is_carried_by_the_last_row(
    monkeypatch,
) -> None:
    """``||r_s||^2 = ||r||^2`` only because the extra zero row carries the
    out-of-range energy. Without it the certificate would see only the part
    of the residual the tangent space can already explain, and eps would be
    optimistic by construction."""
    model, x, _, system = _system("optimized", monkeypatch)
    assert system.full_target is not None
    jacobian = _full_jacobian(model, x)
    target = system.full_target.reshape(-1).to(torch.float64)

    projection = jacobian @ torch.linalg.lstsq(jacobian, target).solution
    out_of_range_sq = float(((target - projection) ** 2).sum())
    assert out_of_range_sq > 0.0

    surrogate_target = system.target.reshape(-1).to(torch.float64)
    assert float(surrogate_target[-1]) == pytest.approx(
        math.sqrt(out_of_range_sq), rel=1e-6
    )
    assert torch.all(system.jacobian[-1] == 0)


# --------------------------------------------------------------------------
# float32 model: the same identities, at the band float32 actually supports
# --------------------------------------------------------------------------


def test_float32_model_statistics_match_within_the_float32_band(
    monkeypatch,
) -> None:
    """``out_dtype`` follows the model's dtype, so a float32 model gets a
    float32 surrogate and both backends carry float32 error.

    MEASURED at production scale (CIFAR-10, N=10 000, K=10, P=2092): the
    optimized and legacy Grams differ by 7.96e-9 relative and their
    right-hand sides by 4.30e-9 -- the size of float32 model arithmetic
    (jacrev in float32 is itself 7.81e-8 from the float64-model Jacobian),
    not of the surrogate construction. The band below is set from that, not
    from what happens to pass here.
    """
    _, _, _, legacy = _system("legacy", monkeypatch, dtype=torch.float32)
    _, _, _, optimized = _system("optimized", monkeypatch, dtype=torch.float32)
    gram_l, rhs_l, rsq_l, rows_l = _statistics(legacy.jacobian, legacy.target)
    gram_o, rhs_o, rsq_o, rows_o = _statistics(optimized.jacobian, optimized.target)

    assert rows_l == rows_o
    assert float((gram_o - gram_l).norm() / gram_l.norm()) < 1e-6
    assert float((rhs_o - rhs_l).norm() / rhs_l.norm()) < 1e-6
    assert rsq_o == pytest.approx(rsq_l, rel=1e-6)
