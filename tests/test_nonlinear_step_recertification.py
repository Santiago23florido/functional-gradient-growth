"""The APPLIED nonlinear displacement must be the CERTIFIED one.

``certify_parametric_step`` and the original nonlinear primary both earned a
certificate for the FULL candidate ``theta'`` and then committed the
parameter-space interpolation ``theta + alpha (theta' - theta)``. For a
nonlinear network

    f_theta - f_{theta + alpha d}  !=  alpha (f_theta - f_{theta + d}),

so that certificate does not describe the step that was taken -- and as
``alpha -> 0`` the interpolation collapses back onto the tangent direction
``alpha J_theta d``, which is exactly the approximation the nonlinear family
exists to avoid.

These tests pin the corrected contract: every interpolation is measured again
from scratch on the same probe against the same threshold.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from fgdlib.search.nonlinear import (
    scale_parameter_displacement,
    search_interpolated_step,
    stream_nonlinear_certificate,
)
from fgdlib.tangent import FGDApproxConfig


class _CurvedModel(torch.nn.Module):
    """Outputs ``[w1, w2**2]`` -- linear in ``w1``, quadratic in ``w2``.

    The quadratic coordinate is what breaks ``Delta_alpha = alpha Delta``:
    moving ``w2`` from ``1`` to ``-1`` leaves ``w2**2`` unchanged at the
    endpoints while sweeping it to ``0`` at the midpoint, so the halfway
    interpolation points somewhere the full displacement never does.
    """

    def __init__(self, w1: float, w2: float) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([w1, w2], dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.w[0], self.w[1] ** 2]).reshape(2, 1)


def _curved_probe_loader() -> DataLoader:
    """Two probe rows with ``y`` chosen so the residual is ``r = (1, 0)``.

    Base model outputs ``f = (0, 1)``; ``r = 2 (f - y) = (1, 0)`` requires
    ``y = (-0.5, 1)``.
    """
    x = torch.zeros((2, 1), dtype=torch.float64)
    y = torch.tensor([-0.5, 1.0], dtype=torch.float64).reshape(2, 1)
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)


def _fgd_config() -> FGDApproxConfig:
    return FGDApproxConfig(functional_loss="mse", rel_error_threshold=0.5)


def test_full_candidate_certifies_but_half_interpolation_does_not() -> None:
    """The construction itself: alpha=1 certifies, alpha=0.5 must not.

    base ``w = (0, 1)``  ->  ``f = (0, 1)``
    cand ``w = (-0.5, -1)`` -> ``f = (-0.5, 1)``, ``Delta = (0.5, 0)``,
    ``cos(Delta, r) = 1``.
    alpha=0.5 ``w = (-0.25, 0)`` -> ``f = (-0.25, 0)``,
    ``Delta = (0.25, 1)``, ``cos = 0.2425``.
    """
    base = _CurvedModel(0.0, 1.0)
    candidate = _CurvedModel(-0.5, -1.0)
    config = _fgd_config()

    full = stream_nonlinear_certificate(
        base_model=base,
        candidate_model=candidate,
        certification_loader=_curved_probe_loader(),
        device=torch.device("cpu"),
        config=config,
    )
    assert full.certified
    assert math.isclose(full.cosine, 1.0, abs_tol=1e-9)

    half_model = scale_parameter_displacement(
        base_model=base,
        candidate_model=candidate,
        rate=0.5,
    )
    half = stream_nonlinear_certificate(
        base_model=base,
        candidate_model=half_model,
        certification_loader=_curved_probe_loader(),
        device=torch.device("cpu"),
        config=config,
    )
    assert not half.certified
    assert math.isclose(half.cosine, 0.25 / math.sqrt(1.0625), rel_tol=1e-9)
    # The whole point: the scaled cosine is NOT the full cosine.
    assert abs(half.cosine - full.cosine) > 0.5


def test_interpolation_certificate_is_never_inherited_from_full_candidate() -> None:
    """Searching only alpha=0.5 must return NO step, not the full certificate."""
    base = _CurvedModel(0.0, 1.0)
    candidate = _CurvedModel(-0.5, -1.0)

    selected, trials = search_interpolated_step(
        base_model=base,
        candidate_model=candidate,
        certification_loader=_curved_probe_loader(),
        device=torch.device("cpu"),
        config=_fgd_config(),
        alpha_grid=(0.5,),
    )
    assert selected is None
    assert len(trials) == 1
    assert trials[0].alpha == 0.5
    assert not trials[0].certified
    assert trials[0].rejection_reason == "relative_error_above_threshold"


def test_largest_certified_alpha_is_selected_over_uncertified_shorter_one() -> None:
    base = _CurvedModel(0.0, 1.0)
    candidate = _CurvedModel(-0.5, -1.0)

    selected, trials = search_interpolated_step(
        base_model=base,
        candidate_model=candidate,
        certification_loader=_curved_probe_loader(),
        device=torch.device("cpu"),
        config=_fgd_config(),
        alpha_grid=(0.25, 0.5, 1.0),
        policy="largest_certified",
    )
    assert selected is not None
    assert selected.alpha == 1.0
    assert selected.certified
    # Every alpha was measured on its own, not derived from the full candidate.
    assert {trial.alpha for trial in trials} == {0.25, 0.5, 1.0}
    assert all(
        trial.rejection_reason is not None for trial in trials if trial.alpha != 1.0
    )


def test_selected_step_carries_its_own_measured_certificate() -> None:
    """``selected.stats`` must describe ``selected.model``, not the candidate."""
    base = _CurvedModel(0.0, 1.0)
    candidate = _CurvedModel(-0.5, -1.0)

    selected, _ = search_interpolated_step(
        base_model=base,
        candidate_model=candidate,
        certification_loader=_curved_probe_loader(),
        device=torch.device("cpu"),
        config=_fgd_config(),
        alpha_grid=(1.0,),
    )
    assert selected is not None
    remeasured = stream_nonlinear_certificate(
        base_model=base,
        candidate_model=selected.model,
        certification_loader=_curved_probe_loader(),
        device=torch.device("cpu"),
        config=_fgd_config(),
    )
    assert math.isclose(remeasured.cosine, selected.stats.cosine, rel_tol=1e-12)
    assert math.isclose(
        remeasured.relative_error,
        selected.stats.relative_error,
        rel_tol=1e-12,
    )


def test_accepted_step_always_shows_measured_functional_descent() -> None:
    """A certified DIRECTION with no measured descent is still rejected."""
    base = _CurvedModel(0.0, 1.0)
    # Moves along +r instead of -r: aligned but the loss increases.
    away = _CurvedModel(1.0, 1.0)

    selected, trials = search_interpolated_step(
        base_model=base,
        candidate_model=away,
        certification_loader=_curved_probe_loader(),
        device=torch.device("cpu"),
        config=_fgd_config(),
        alpha_grid=(1.0,),
    )
    assert selected is None
    assert trials[0].rejection_reason in {
        "relative_error_above_threshold",
        "no_functional_descent",
    }


def test_effective_secant_rate_is_distinct_from_eta_f_and_alpha() -> None:
    """eta* = <Delta, r>/|r|^2 is its own quantity, not eta_f and not alpha."""
    base = _CurvedModel(0.0, 1.0)
    candidate = _CurvedModel(-0.5, -1.0)
    stats = stream_nonlinear_certificate(
        base_model=base,
        candidate_model=candidate,
        certification_loader=_curved_probe_loader(),
        device=torch.device("cpu"),
        config=_fgd_config(),
    )
    # Delta = (0.5, 0), r = (1, 0)  ->  eta* = 0.5 / 1 = 0.5.
    assert math.isclose(stats.effective_secant_rate, 0.5, rel_tol=1e-12)
    assert math.isclose(stats.dot_product, 0.5, rel_tol=1e-12)
    assert math.isclose(stats.gradient_sq_norm, 1.0, rel_tol=1e-12)
