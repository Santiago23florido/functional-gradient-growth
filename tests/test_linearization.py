"""Lemma 3.5's hypothesis, enforced: the step must BE the function-space step.

The lemma licenses ``eta`` for ``f <- f - eta g``. The optimiser performs
``theta <- theta - eta u``, and those coincide only while the second-order
remainder stays small. These tests pin that the defect measures exactly
that, that narrowing only ever moves inward from the certified rate, and
that the check never consults the loss -- it is not the descent gate
returning under another name.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from fgdlib.search.linearization import (
    certified_linear_learning_rate,
    linearization_defect,
    predicted_displacement,
)
from stable_tiny.pipeline import load_pipeline_config


@pytest.fixture
def config():
    base = load_pipeline_config("configs/fgd/default.yaml").fgd_approx
    return replace(base, certify_linearization_tolerance=0.1)


# Double precision throughout: the defect divides a residual by eta, which
# amplifies round-off, and these tests are about the MATHEMATICS of the
# remainder rather than about float32's floor. In float32 the linear model
# already shows a defect of 1.2e-04 at eta = 1e-3 from rounding alone.
_DTYPE = torch.float64


class _Linear(nn.Module):
    """f(theta) = W x -- exactly linear in theta, so the defect must vanish."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.weight = nn.Parameter(torch.randn(3, 4, dtype=_DTYPE))

    def forward(self, x):
        return x @ self.weight.T


class _Quadratic(nn.Module):
    """f(theta) = (a . theta)^2 -- curvature the certificate cannot see."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.theta = nn.Parameter(torch.randn(4, dtype=_DTYPE) * scale)

    def forward(self, x):
        inner = (x * self.theta).sum(dim=-1, keepdim=True)
        return inner**2


def _updates(model):
    torch.manual_seed(1)
    return tuple(torch.randn_like(p) for p in model.parameters())


def test_predicted_displacement_is_the_exact_directional_derivative() -> None:
    """For a model linear in theta, J u is the whole displacement."""
    model = _Linear()
    x = torch.randn(6, 4, dtype=_DTYPE)
    updates = _updates(model)
    jacobian_vector = predicted_displacement(model, x, updates)
    # f(theta - u) - f(theta) = -J u exactly here.
    with torch.no_grad():
        base = model(x).clone()
        for parameter, update in zip(model.parameters(), updates):
            parameter -= update
        moved = model(x).clone()
    assert torch.allclose(base - moved, jacobian_vector, atol=1e-10)


def test_defect_vanishes_when_the_step_is_exactly_function_space() -> None:
    """No second-order remainder means the lemma applies verbatim."""
    model = _Linear()
    x = torch.randn(6, 4, dtype=_DTYPE)
    updates = _updates(model)
    for eta in (1e-3, 0.1, 1.0, 5.0):
        assert linearization_defect(model, x, updates, eta) < 1e-10


def test_defect_grows_with_the_step_size_under_curvature() -> None:
    """The remainder is O(eta^2 |u|^2): larger steps leave the regime."""
    model = _Quadratic()
    x = torch.randn(6, 4, dtype=_DTYPE)
    updates = _updates(model)
    defects = [
        linearization_defect(model, x, updates, eta)
        for eta in (1e-4, 1e-2, 1e-1, 1.0)
    ]
    assert defects == sorted(defects)
    assert defects[0] < 1e-3 < defects[-1]


def test_narrowing_only_ever_moves_inward(config) -> None:
    """Every rate returned was already certified -- the search never grows eta."""
    model = _Quadratic()
    x = torch.randn(6, 4, dtype=_DTYPE)
    updates = _updates(model)
    result = certified_linear_learning_rate(model, x, updates, 2.0, config)
    assert result.learning_rate is not None
    assert result.learning_rate <= 2.0
    assert result.certified_learning_rate == 2.0
    assert result.defect <= config.certify_linearization_tolerance


def test_an_admissible_rate_is_returned_unchanged(config) -> None:
    """A step already inside the regime is not shrunk for no reason."""
    model = _Linear()
    x = torch.randn(6, 4, dtype=_DTYPE)
    updates = _updates(model)
    result = certified_linear_learning_rate(model, x, updates, 0.5, config)
    assert result.learning_rate == pytest.approx(0.5)
    assert result.backtracks == 0


def test_curvature_that_outlasts_the_interval_yields_no_rate(config) -> None:
    """Then the STRUCTURE, not the step size, is what has to change.

    For a smooth model the defect always vanishes as eta -> 0, so a rate
    becomes unavailable by running out of INTERVAL, not out of smoothness:
    the search stops at theory_lr_min. That floor is the meaningful
    boundary, and reaching it is a statement about the direction, not a
    numerical failure.
    """
    model = _Quadratic(scale=1e3)
    x = torch.randn(6, 4, dtype=_DTYPE) * 1e2
    updates = _updates(model)
    strict = replace(
        config, certify_linearization_tolerance=1e-9, theory_lr_min=0.1
    )
    result = certified_linear_learning_rate(model, x, updates, 1.0, strict)
    assert result.learning_rate is None
    assert result.certified_learning_rate == 1.0   # what was given up


def test_the_floor_is_respected(config) -> None:
    """No returned rate may sit below theory_lr_min -- that is the interval."""
    model = _Quadratic(scale=1e2)
    x = torch.randn(6, 4, dtype=_DTYPE) * 10
    updates = _updates(model)
    strict = replace(config, certify_linearization_tolerance=1e-8)
    result = certified_linear_learning_rate(model, x, updates, 1.0, strict)
    if result.learning_rate is not None:
        assert result.learning_rate > strict.theory_lr_min


def test_the_check_never_consults_the_loss() -> None:
    """It is not the descent gate under another name.

    The defect depends only on inputs, parameters and the update -- no
    targets participate, so a step that worsens the loss and a step that
    improves it are measured identically.
    """
    model = _Linear()
    x = torch.randn(6, 4, dtype=_DTYPE)
    updates = _updates(model)
    reference = linearization_defect(model, x, updates, 0.3)
    # Flipping the update's sign reverses descent into ascent; the defect,
    # being a statement about linearity alone, is unchanged.
    reversed_updates = tuple(-update for update in updates)
    assert linearization_defect(model, x, reversed_updates, 0.3) == pytest.approx(
        reference, abs=1e-12
    )


def test_a_degenerate_measurement_is_not_a_pass() -> None:
    """Zero predicted displacement means "cannot tell", which must not admit."""
    model = _Linear()
    x = torch.randn(6, 4, dtype=_DTYPE)
    zero = tuple(torch.zeros_like(p) for p in model.parameters())
    assert linearization_defect(model, x, zero, 0.1) == float("inf")
    assert linearization_defect(model, x, _updates(model), 0.0) == float("inf")


def test_the_model_is_left_untouched(config) -> None:
    """Probing is measurement, not a step: parameters must be restored."""
    model = _Quadratic()
    x = torch.randn(6, 4, dtype=_DTYPE)
    updates = _updates(model)
    before = [p.detach().clone() for p in model.parameters()]
    certified_linear_learning_rate(model, x, updates, 2.0, config)
    for parameter, original in zip(model.parameters(), before):
        assert torch.equal(parameter.detach(), original)
