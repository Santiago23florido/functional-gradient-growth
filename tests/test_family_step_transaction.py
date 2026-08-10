"""The family step certified a direction and committed a distance unmeasured.

MEASURED on MNIST, three independent runs, one family step each and one
unexplained jump each -- a one-to-one correspondence::

    cost_probe            x3.97
    mnist_lookahead_entry x6.21
    cf_mnist_mf_on_full   x15.44

The sequence, verbatim from the log::

    [TRANSACTION] full_train=2.332345e+04 -> 2.331797e+04  accepted=True
    [FAMILY] eta_f=1 certified (cos 0.9242, eps 0.3819)
    [TRANSACTION] full_train=9.252774e+04 -> 8.014143e+04  accepted=True

The family's certificate is a cosine on the PROBE: it licenses which way, never
how far, and the ladder then replaces the model with the trained clone whole --
a functional step of size 1. The tangent path is guarded against exactly this
and rejects it with ``full_train_functional_increase``; the family step never
reached that guard. Worse, the next transaction re-based its ``full_train``
baseline on the already-degraded model, so a 4x regression became the new zero
and every certificate went on reporting health.

No probe-side bound can catch this -- not Lemma 3.5's rate, not the PL constant
-- because the defect is that the probe improves (-32 %) while the objective
worsens (+34 %). Only measuring the objective can.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from fgdlib.gromo_setup import ensure_gromo_importable
from stable_tiny.pipeline import (
    _interpolate_toward,
    _transactionally_accept_family_step,
    load_pipeline_config,
)

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP  # noqa: E402


CPU = torch.device("cpu")


def _model(seed: int = 0) -> GrowingMLP:
    torch.manual_seed(seed)
    return GrowingMLP(
        in_features=4,
        out_features=2,
        hidden_size=6,
        number_hidden_layers=2,
        activation=torch.nn.SELU(),
        device=CPU,
    )


def _fixture():
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(32, 4, generator=generator)
    y = torch.randn(32, 2, generator=generator)
    probe = (x[:8], y[:8])
    full_train = [(x[i : i + 8], y[i : i + 8]) for i in range(0, 32, 8)]
    config = load_pipeline_config("configs/fgd/mnist_matrix_free.yaml")
    return _model(), probe, full_train, config


def _perturbed(base: GrowingMLP, scale: float) -> GrowingMLP:
    """A 'clone' displaced by a fixed direction, so its effect is controllable."""
    stepped = copy.deepcopy(base)
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for parameter in stepped.parameters():
            parameter.add_(
                torch.randn(parameter.shape, generator=generator) * scale
            )
    return stepped


# ---------------------------------------------------------------------------
# The bug itself.
# ---------------------------------------------------------------------------


def test_a_step_that_wrecks_the_real_objective_is_refused() -> None:
    """The measured failure: return None so the ladder grows instead."""
    base, probe, full_train, config = _fixture()
    # Large enough that no admissible alpha exists at this scale.
    stepped = _perturbed(base, scale=8.0)

    accepted = _transactionally_accept_family_step(
        base_model=base, stepped=stepped, probe=probe,
        full_train_batches=full_train, device=CPU, config=config, progress=None,
    )
    assert accepted is None


def test_refusing_leaves_the_base_model_untouched() -> None:
    """A rejected transaction must not have moved the model it was given."""
    base, probe, full_train, config = _fixture()
    before = [p.detach().clone() for p in base.parameters()]

    _transactionally_accept_family_step(
        base_model=base, stepped=_perturbed(base, scale=8.0), probe=probe,
        full_train_batches=full_train, device=CPU, config=config, progress=None,
    )

    for original, current in zip(before, base.parameters()):
        assert torch.equal(original, current.detach())


# ---------------------------------------------------------------------------
# The interpolation the backtrack is built on.
# ---------------------------------------------------------------------------


def test_alpha_one_is_the_stepped_model_and_alpha_zero_the_base() -> None:
    base = _model()
    stepped = _perturbed(base, scale=1.0)

    at_one = _interpolate_toward(base, stepped, 1.0)
    at_zero = _interpolate_toward(base, stepped, 0.0)
    for parameter, goal in zip(at_one.parameters(), stepped.parameters()):
        assert torch.allclose(parameter, goal, atol=1e-7)
    for parameter, start in zip(at_zero.parameters(), base.parameters()):
        assert torch.allclose(parameter, start, atol=1e-7)


def test_interpolation_does_not_mutate_either_endpoint() -> None:
    base = _model()
    stepped = _perturbed(base, scale=1.0)
    base_before = [p.detach().clone() for p in base.parameters()]
    stepped_before = [p.detach().clone() for p in stepped.parameters()]

    _interpolate_toward(base, stepped, 0.5)

    for original, current in zip(base_before, base.parameters()):
        assert torch.equal(original, current.detach())
    for original, current in zip(stepped_before, stepped.parameters()):
        assert torch.equal(original, current.detach())


# ---------------------------------------------------------------------------
# The certificate is re-earned at each alpha, never inherited.
# ---------------------------------------------------------------------------


def test_the_certificate_is_remeasured_on_the_committed_model() -> None:
    """Inheriting it is the defect the nonlinear branch already documented.

    MEASURED there: ``cos 0.969 -> 0.640 -> 0.523 -> 0.476`` across
    ``alpha`` 1 / .5 / .25 / .125, so a certificate earned at alpha=1 says
    nothing about a shortened commit.
    """
    from fgdlib.search.families import measure_family_displacement

    base, probe, _full, config = _fixture()
    stepped = _perturbed(base, scale=2.0)

    full = measure_family_displacement(
        base, stepped, probe[0], probe[1], config.fgd_approx
    )
    short = measure_family_displacement(
        base, _interpolate_toward(base, stepped, 0.125),
        probe[0], probe[1], config.fgd_approx,
    )
    # The point is only that they are independently measured quantities, not
    # one number reused; a nonlinear model has no reason to keep them equal.
    assert full != short


def test_measuring_a_displacement_leaves_both_models_in_their_modes() -> None:
    from fgdlib.search.families import measure_family_displacement

    base, probe, _full, config = _fixture()
    stepped = _perturbed(base, scale=1.0)
    base.train()
    stepped.eval()

    measure_family_displacement(
        base, stepped, probe[0], probe[1], config.fgd_approx
    )

    assert base.training is True
    assert stepped.training is False


# ---------------------------------------------------------------------------
# Blast radius: the tangent route cannot reach any of this.
# ---------------------------------------------------------------------------


def test_the_guard_is_bound_to_the_matrix_free_route() -> None:
    """N1024 is `family_order: [tangent]` with the guard off, and the
    validator forbids turning it on there -- so the family step keeps its
    current behaviour on every existing result."""
    from fgdlib.tangent import validate_transactional_realized_descent

    n1024 = load_pipeline_config("configs/fgd/family_ladder_N1024.yaml").fgd_approx
    assert n1024.family_order == ("tangent",)
    assert n1024.transactional_realized_descent is False

    with pytest.raises(ValueError, match="matrix_free_tangent"):
        validate_transactional_realized_descent(
            replace(n1024, transactional_realized_descent=True)
        )


def test_mnist_has_the_guard_on_so_the_family_step_is_covered() -> None:
    mnist = load_pipeline_config("configs/fgd/mnist_matrix_free.yaml").fgd_approx
    assert mnist.family_order == ("matrix_free_tangent",)
    assert mnist.transactional_realized_descent is True
    assert mnist.certify_family_ladder is True


def test_the_three_counters_are_registered() -> None:
    from fgdlib.profile import PROFILE_FIELDS

    for field in (
        "family_transaction_accepted",
        "family_transaction_backtracks",
        "family_transaction_rejected",
    ):
        assert field in PROFILE_FIELDS
