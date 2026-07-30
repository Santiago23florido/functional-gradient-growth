"""The spectrum and both damping decisions must survive the change of backend.

Everything the certificate says is a statement about the singular values of
the tangent Jacobian: ``minimal_relative_error_from_system`` filters with
``sigma^2/(sigma^2 + 1e-16 sigma_max^2)``, and ``select_projection_damping``
bisects for the largest rho whose eps still certifies and then ranks a fan of
rungs by Lemma 3.5's guaranteed decrease. Both read the data only through the
surrogate, so a surrogate that changed the spectrum -- even in the tail --
would change the growth signal without changing anything a coarser test looks
at.

WHAT THE BACKEND DISAGREEMENT ACTUALLY IS (measured, not assumed).
At production scale (CIFAR-10 grayscale, N=10 000, K=10, P=2092,
certify_stream_gram, float32 model, cond(J)=4.78e3, rank = P so the SVD-free
route fires on every call):

  * with the DOWNSTREAM factorization in float64
    (projection_fast_factorization=False) the optimized and legacy backends
    give BITWISE EQUAL minimum relative error (|dRelErr| = 0.0) and
    |dlambda|/lambda = 6.1e-7;
  * with ``projection_fast_factorization: true`` -- what
    configs/fgd/cifar_streaming.yaml actually sets -- both backends run a
    FLOAT32 SVD of the (P+1) x P surrogate, and THAT stage is itself
    +2.4e-4 away from the closed-form RelErr. The optimized-vs-legacy gap
    under it grows to |dRelErr| = 2.43e-5 and |dlambda|/lambda = 7.6e-4.
    For scale: running the SAME legacy backend with only
    ``certify_stream_chunk`` changed from 1024 to 2500 -- a permitted,
    mathematically neutral implementation choice -- moves it by 3.25e-5 and
    1.1e-3, i.e. MORE. The 4th-digit lambda difference is the float32
    downstream factorization, not the tangent backend.

So the parity claim is pinned where it is a claim about this phase -- the
exact factorization -- and the float32 stage is pinned separately as the
larger, pre-existing error term it is.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import replace

import pytest
import torch
from torch.func import functional_call, jacrev

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.profile import reset, snapshot
from fgdlib.search.damping import (
    minimal_relative_error_from_system,
    select_projection_damping,
)
from fgdlib.tangent import (
    FGDApproxConfig,
    _flatten_jacobian,
    _tangent_spectrum,
    _trainable_named_parameters,
    exact_tangent_system,
)

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP

IN_FEATURES = 5
OUT_FEATURES = 3
HIDDEN = 4
SAMPLES = 40
#: Residual energy placed OUTSIDE the tangent range, as a fraction of the
#: in-range part. 0.3 puts eps around 0.2 -- comfortably certifying, so
#: select_projection_damping returns a rung instead of None, which is what
#: makes "the selected damping is identical" a testable statement at all.
OUT_OF_RANGE = 0.3

RTOL64 = 1e-9
ATOL64 = 1e-11


def _model(dtype: torch.dtype, seed: int) -> GrowingMLP:
    torch.manual_seed(seed)
    model = GrowingMLP(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_size=HIDDEN,
        number_hidden_layers=2,
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


def _certifying_probe(dtype: torch.dtype, seed: int = 3, probe_seed: int = 11):
    """A probe whose residual mostly LIES IN the tangent range.

    A random target leaves eps > 1 on this architecture, ``select_projection_
    damping`` returns None, and there is no selected damping to compare. Here
    the target is built as ``f - (d + OUT_OF_RANGE * n)`` with ``d`` in the
    range of J and ``n`` generic, which is the eps < 1/2 regime the whole
    certificate is about.
    """
    model = _model(dtype, seed)
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


def _config(*, fast: bool = False, chunk: int = 8) -> FGDApproxConfig:
    return replace(
        FGDApproxConfig(
            projection_solver="exact",
            functional_loss="mse",
            certify_stream_gram=True,
            certify_stream_chunk=chunk,
        ),
        projection_fast_factorization=fast,
    )


def _decisions(
    monkeypatch, backend: str, *, dtype=torch.float64, fast=False, seed=3, probe_seed=11
):
    monkeypatch.setenv("FGD_TANGENT_BACKEND", backend)
    config = _config(fast=fast)
    model, x, y = _certifying_probe(dtype, seed, probe_seed)
    system = exact_tangent_system(model, x, y, config)
    assert system is not None
    minimum = minimal_relative_error_from_system(system, config)
    choice = select_projection_damping(model, x, y, config, system=system)
    return system, minimum, choice


# --------------------------------------------------------------------------
# 8.D -- the spectrum itself
# --------------------------------------------------------------------------


def _random_factor(rows: int, columns: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    jacobian = torch.randn(rows, columns, dtype=torch.float64)
    return torch.linalg.qr(jacobian, mode="r").R


def test_spectrum_equals_the_singular_values_of_the_jacobian_it_came_from() -> None:
    """``R^T R = J^T J`` exactly, so ``svdvals(R)`` IS ``svdvals(J)``. That
    identity is the only reason the streamed factor is allowed to stand in
    for the Jacobian at all."""
    torch.manual_seed(5)
    jacobian = torch.randn(300, 40, dtype=torch.float64)
    r_factor = torch.linalg.qr(jacobian, mode="r").R
    spectrum = _tangent_spectrum(r_factor)
    expected = torch.linalg.svdvals(jacobian)
    assert spectrum.shape == expected.shape
    assert torch.allclose(spectrum, expected, rtol=RTOL64, atol=ATOL64)


def test_spectrum_squares_are_the_gram_eigenvalues() -> None:
    r_factor = _random_factor(200, 30, seed=6)
    spectrum = _tangent_spectrum(r_factor)
    eigenvalues = torch.linalg.eigvalsh(r_factor.t() @ r_factor)
    assert torch.allclose(spectrum.square(), eigenvalues.flip(0), rtol=1e-8, atol=1e-10)


def test_spectrum_is_returned_sorted_descending_on_the_factor_device() -> None:
    r_factor = _random_factor(120, 25, seed=7)
    spectrum = _tangent_spectrum(r_factor)
    assert spectrum.device == r_factor.device
    assert torch.all(spectrum[:-1] >= spectrum[1:])


def test_spectrum_frobenius_identity_holds() -> None:
    """``sum(sigma_i^2) == ||R||_F^2`` is an algebraic identity, true at any
    conditioning; the implementation raises above 1e-12 relative because a
    violation means the spectrum is simply wrong. Pinned here at the
    tolerance the implementation uses."""
    for seed in range(4):
        r_factor = _random_factor(150, 35, seed=seed)
        spectrum = _tangent_spectrum(r_factor)
        frobenius_sq = float((r_factor * r_factor).sum())
        spectrum_sq = float(spectrum.square().sum())
        assert abs(spectrum_sq - frobenius_sq) / frobenius_sq <= 1e-12


def test_power_and_inverse_power_estimates_bracket_the_spectrum(
    monkeypatch,
) -> None:
    """The two iterative diagnostics must corroborate LAPACK to 5 % on a
    well-conditioned factor -- and must therefore record NO fallback."""
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    r_factor = _random_factor(400, 30, seed=8)
    spectrum = _tangent_spectrum(r_factor)
    sigma_max = float(spectrum[0])
    sigma_min = float(spectrum[-1])

    generator = torch.Generator()
    generator.manual_seed(0)
    vector = torch.randn(r_factor.shape[1], dtype=torch.float64, generator=generator)
    vector = vector / vector.norm()
    for _ in range(50):
        vector = r_factor.t() @ (r_factor @ vector)
        vector = vector / vector.norm()
    power = math.sqrt(float(vector @ (r_factor.t() @ (r_factor @ vector))))
    assert abs(power - sigma_max) <= 0.05 * sigma_max

    other = torch.randn(r_factor.shape[1], dtype=torch.float64, generator=generator)
    other = other / other.norm()
    for _ in range(50):
        forward = torch.linalg.solve_triangular(
            r_factor.t(), other.unsqueeze(1), upper=False
        ).squeeze(1)
        other = torch.linalg.solve_triangular(
            r_factor, forward.unsqueeze(1), upper=True
        ).squeeze(1)
        other = other / other.norm()
    forward = torch.linalg.solve_triangular(
        r_factor.t(), other.unsqueeze(1), upper=False
    ).squeeze(1)
    backward = torch.linalg.solve_triangular(
        r_factor, forward.unsqueeze(1), upper=True
    ).squeeze(1)
    inverse_power = 1.0 / math.sqrt(float(other @ backward))
    assert abs(inverse_power - sigma_min) <= 0.05 * sigma_min

    assert snapshot()["tangent_spectrum_fallbacks"] == 0
    reset()


def test_condition_estimate_gauge_records_sigma_max_over_sigma_min(
    monkeypatch,
) -> None:
    """cond(J) is the number that says whether a normal-equation Gram path
    could ever have been viable -- cond(J^T J) is its square -- so it is
    recorded on every call rather than reconstructed afterwards."""
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    r_factor = _random_factor(200, 28, seed=9)
    spectrum = _tangent_spectrum(r_factor)
    expected = float(spectrum[0]) / float(spectrum[-1])
    assert snapshot()["tangent_condition_estimate"] == pytest.approx(
        expected, rel=1e-12
    )
    reset()


@pytest.mark.parametrize("backend", ["legacy", "optimized"])
def test_surrogate_spectrum_matches_the_full_jacobian(
    monkeypatch, backend: str
) -> None:
    """8.D sigma_max AND the whole spectrum: the surrogate must not merely
    preserve the leading singular value, since the damping filter reads the
    tail too."""
    system, _, _ = _decisions(monkeypatch, backend)
    monkeypatch.setenv("FGD_TANGENT_BACKEND", backend)
    model, x, _ = _certifying_probe(torch.float64)
    jacobian = _full_jacobian(model, x)

    expected = torch.linalg.svdvals(jacobian)
    actual = torch.linalg.svdvals(system.jacobian.to(torch.float64))
    assert actual.shape == expected.shape
    assert float(actual[0]) == pytest.approx(float(expected[0]), rel=1e-9)
    assert torch.allclose(actual, expected, rtol=1e-8, atol=1e-10)


# --------------------------------------------------------------------------
# 8.D -- the two damping decisions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("probe_seed", [11, 12, 13])
@pytest.mark.parametrize("seed", [0, 3])
def test_minimal_relative_error_matches_between_backends(
    monkeypatch, seed: int, probe_seed: int
) -> None:
    """The minimum-damping certificate is what growth reads. MEASURED worst
    case over this sweep: 3.7e-14 absolute; over the production CIFAR
    system: 0.0 (bitwise). The 1e-9 bar in the plan holds with five orders
    of margin once the downstream factorization is float64."""
    _, legacy, _ = _decisions(monkeypatch, "legacy", seed=seed, probe_seed=probe_seed)
    _, optimized, _ = _decisions(
        monkeypatch, "optimized", seed=seed, probe_seed=probe_seed
    )
    assert legacy < 0.5 and optimized < 0.5
    assert abs(optimized - legacy) <= 1e-9


@pytest.mark.parametrize("probe_seed", [11, 12, 13])
@pytest.mark.parametrize("seed", [0, 3])
def test_selected_damping_is_identical_between_backends(
    monkeypatch, seed: int, probe_seed: int
) -> None:
    """rho, lambda, eps, the certified rate and the guaranteed decrease.

    The selected rung is an ARGMAX over a geometric fan (ratio 10^-0.25)
    whose objective is nearly flat at its maximum, so this is the decision
    most exposed to a perturbation: MEASURED at production scale, running
    the legacy backend against ITSELF with only ``certify_stream_chunk``
    changed from 1024 to 512 flips the winning rung outright
    (|dlambda|/lambda = 0.78) while eps, eta and the decrease all move by
    less than 3.4e-6. Backend parity has to be tight enough that no such
    flip is possible: MEASURED worst |dlambda|/lambda over this sweep is
    2.5e-15, against a rung spacing of 10^0.25 - 1 = 0.78.
    """
    _, _, legacy = _decisions(monkeypatch, "legacy", seed=seed, probe_seed=probe_seed)
    _, _, optimized = _decisions(
        monkeypatch, "optimized", seed=seed, probe_seed=probe_seed
    )
    assert legacy is not None and optimized is not None
    left, right = legacy.candidate, optimized.candidate

    assert right.relative_damping == pytest.approx(left.relative_damping, rel=1e-9)
    assert right.absolute_damping == pytest.approx(left.absolute_damping, rel=1e-9)
    assert right.relative_error == pytest.approx(left.relative_error, rel=1e-9)
    assert right.update_norm == pytest.approx(left.update_norm, rel=1e-8)
    assert right.approximation_norm == pytest.approx(left.approximation_norm, rel=1e-8)
    assert right.effective_dof == pytest.approx(left.effective_dof, rel=1e-8)
    assert right.gcv == pytest.approx(left.gcv, rel=1e-8)
    assert right.guaranteed_decrease == pytest.approx(
        left.guaranteed_decrease, rel=1e-8
    )
    assert (right.learning_rate is None) == (left.learning_rate is None)
    if left.learning_rate is not None:
        assert right.learning_rate == pytest.approx(left.learning_rate, rel=1e-9)


def test_relative_error_agreement_meets_the_one_part_per_million_bar(
    monkeypatch,
) -> None:
    """The acceptance bar |RelErr_opt - RelErr_leg| <= 1e-6, stated as such.

    Holds with the exact (float64) downstream factorization: MEASURED 3.7e-14
    here and 0.0 on the production CIFAR system. It does NOT hold under
    ``projection_fast_factorization: true`` at production scale (2.43e-5),
    but neither does legacy-against-legacy with a different stream chunk
    (3.25e-5) -- see the companion test below and the module docstring.
    """
    worst = 0.0
    for seed in (0, 1, 2, 3):
        _, legacy, _ = _decisions(monkeypatch, "legacy", seed=seed)
        _, optimized, _ = _decisions(monkeypatch, "optimized", seed=seed)
        worst = max(worst, abs(optimized - legacy))
    assert worst <= 1e-6


def test_float32_fast_factorization_dominates_the_backend_disagreement(
    monkeypatch,
) -> None:
    """``projection_fast_factorization`` is the larger error term, by far.

    It replaces the float64 SVD of the surrogate with a FLOAT32 one. That is
    a deliberate, pre-existing speed/accuracy trade in ``damping.py`` and it
    has nothing to do with which tangent backend produced the surrogate --
    but it is what a run's 4th-digit ``lambda`` difference is actually made
    of. MEASURED at production scale it puts the reported eps +2.4e-4 away
    from the closed-form value; MEASURED over the fixtures below it reaches
    1.9e-1, while the optimized-vs-legacy disagreement under the exact
    factorization stays at 3.0e-7. Asserting the ORDERING (rather than
    either magnitude) is what stays true across machines.
    """
    fast_error = 0.0
    backend_gap = 0.0
    for seed in (0, 1, 2):
        monkeypatch.setenv("FGD_TANGENT_BACKEND", "legacy")
        model, x, y = _certifying_probe(torch.float32, seed=seed)
        exact_config = _config(fast=False)
        fast_config = _config(fast=True)
        system = exact_tangent_system(model, x, y, exact_config)
        assert system is not None
        exact_value = minimal_relative_error_from_system(system, exact_config)
        fast_value = minimal_relative_error_from_system(
            replace(system, config_signature=repr(fast_config)), fast_config
        )
        fast_error = max(fast_error, abs(fast_value - exact_value))

        monkeypatch.setenv("FGD_TANGENT_BACKEND", "optimized")
        model, x, y = _certifying_probe(torch.float32, seed=seed)
        optimized_system = exact_tangent_system(model, x, y, exact_config)
        assert optimized_system is not None
        optimized_value = minimal_relative_error_from_system(
            optimized_system, exact_config
        )
        backend_gap = max(backend_gap, abs(optimized_value - exact_value))

    assert fast_error > 10.0 * backend_gap, (
        f"float32 factorization error {fast_error:.3e} no longer dominates the "
        f"backend disagreement {backend_gap:.3e}"
    )
