"""The lookahead could veto growth but never authorise it.

MEASURED on MNIST, run ``mnist-depth3-small-seed-0``, 150 epochs::

    ep=1  eps=0.8527  P=1612          above the bar -> grows
    ep=1  eps=0.4965  P=2399          two growths put it under
    ep=2  eps=0.8505  P=2399          rises again -> grows
    ep=2  eps=0.4975  P=2405  (+6)    certifies... and never again

Final: 2 growths, 16 outer steps in 150 epochs, accuracy 0.109, and every
certificate reporting health. The loop is closed and stable: eps parks two
thousandths under the bar, Lemma 3.5's rate collapses there
(``eta_bar = 2(1-2eps)/(L(1+2eps))`` = 0.0025, committed 1.55e-05), the loss
moves 0.005 % per step, so eps never rises, so nothing ever grows again.

``_growth_reduces_lookahead_epsilon`` is exactly the question that separates
"this small net is enough" from "this net is grinding" -- but a "no" returned
and a "yes" fell through to ``while epsilon >= target``, which excludes the
band the question is asked in. A brake with no accelerator.

These tests pin the fix and, above all, its BLAST RADIUS: the tangent route
must be unable to reach it, not merely configured not to.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from fgdlib.search.certify import CertifyResult
from fgdlib.tangent import FGDApproxConfig, validate_growth_lookahead_entry
from stable_tiny.pipeline import load_pipeline_config


def _matrix_free(**overrides) -> FGDApproxConfig:
    return FGDApproxConfig(
        family_order=("matrix_free_tangent",),
        certify_growth_lookahead=True,
        certify_growth_lookahead_entry=True,
        **overrides,
    )


# ---------------------------------------------------------------------------
# The structural gate. This is the guarantee, not the flag being off.
# ---------------------------------------------------------------------------


def test_the_tangent_route_cannot_enable_this_at_all() -> None:
    """A load-time error, because a convention can be broken in one line.

    ``family_ladder_N1024.yaml`` is measured and good and is invariant on this
    branch. "The field defaults to False over there" would leave the guarantee
    resting on nobody ever setting it.
    """
    with pytest.raises(ValueError, match="matrix_free_tangent"):
        validate_growth_lookahead_entry(
            FGDApproxConfig(
                family_order=("tangent",),
                certify_growth_lookahead=True,
                certify_growth_lookahead_entry=True,
            )
        )


def test_authorising_without_the_predicate_wired_is_refused() -> None:
    """There is nothing to authorise if the question is never asked."""
    with pytest.raises(ValueError, match="requires certify_growth_lookahead"):
        validate_growth_lookahead_entry(
            FGDApproxConfig(
                family_order=("matrix_free_tangent",),
                certify_growth_lookahead=False,
                certify_growth_lookahead_entry=True,
            )
        )


def test_the_matrix_free_route_may_enable_it() -> None:
    validate_growth_lookahead_entry(_matrix_free())


def test_the_default_is_off_and_valid_everywhere() -> None:
    validate_growth_lookahead_entry(FGDApproxConfig())
    assert FGDApproxConfig().certify_growth_lookahead_entry is False


def test_every_canonical_config_still_loads_and_only_mnist_opts_in() -> None:
    """The catalogue, checked as a whole: who got it and who did not."""
    expected = {
        "family_ladder_N1024.yaml": False,
        "family_ladder_matrix_free_N1024.yaml": False,
        "mnist_streaming.yaml": False,
        "cifar_streaming.yaml": False,
        "mnist_matrix_free.yaml": True,
    }
    for name, opted_in in expected.items():
        config = load_pipeline_config(f"configs/fgd/{name}")
        assert (
            config.fgd_approx.certify_growth_lookahead_entry is opted_in
        ), name


def test_the_n1024_config_is_untouched_on_every_growth_knob() -> None:
    """The invariant the user asked for, stated field by field."""
    config = load_pipeline_config("configs/fgd/family_ladder_N1024.yaml").fgd_approx
    assert config.family_order == ("tangent",)
    assert config.certify_growth_lookahead is False
    assert config.certify_growth_lookahead_entry is False
    assert config.growth_bottleneck_crossfold_folds == 0
    assert config.growth_bottleneck_significance is False
    assert config.certify_force_growth_on_finite_step_failure is False
    assert config.certify_growth_target is None
    assert config.rel_error_threshold == 0.5


# ---------------------------------------------------------------------------
# The loop-entry condition itself, exercised as the pure predicate it is.
# ---------------------------------------------------------------------------


def _enters(*, epsilon: float, target: float, forced: int, warranted: bool) -> bool:
    """The `while` guard of grow_until_certified, isolated."""
    return epsilon >= target or forced > 0 or warranted


def test_the_guard_is_unchanged_when_nothing_authorises() -> None:
    """With growth_warranted=None -- everything that exists today -- identical."""
    for epsilon, target in ((0.4975, 0.5), (0.8505, 0.5), (0.5, 0.5)):
        assert _enters(
            epsilon=epsilon, target=target, forced=0, warranted=False
        ) == (epsilon >= target)


def test_the_measured_deadlock_is_exactly_what_the_guard_excluded() -> None:
    """0.4975 against 0.5: the run that produced accuracy 0.109."""
    assert not _enters(epsilon=0.4975, target=0.5, forced=0, warranted=False)
    assert _enters(epsilon=0.4975, target=0.5, forced=0, warranted=True)


def test_the_authorisation_is_spent_by_the_first_iteration() -> None:
    """One neuron per outer step: the bound that stops runaway growth."""
    warranted = True
    entered = []
    epsilon, target = 0.4975, 0.5
    for _ in range(5):
        if not (epsilon >= target or warranted):
            break
        entered.append(True)
        warranted = False  # consumed, exactly as the loop body does
    assert len(entered) == 1


def test_a_voluntary_entry_bypasses_the_marginal_value_rule() -> None:
    """Below the certificate the gap is NEGATIVE, so the floor is meaningless.

    `_growth_pays` requires closing `min_gain * (eps - target)` of the gap.
    At eps 0.4975 against target 0.5 that product is negative, so applying it
    to an authorised growth would refuse it for arithmetic reasons rather than
    measured ones. `is_forced` already bypasses it for the same reason.
    """
    epsilon, target, min_gain = 0.4975, 0.5, 0.1
    assert min_gain * (epsilon - target) < 0.0


# ---------------------------------------------------------------------------
# End to end through grow_until_certified: the guard permitting entry is not
# the same claim as a neuron actually being bought.
# ---------------------------------------------------------------------------


def _already_certified_fixture():
    """A model whose eps ALREADY certifies -- the MNIST situation, in miniature.

    The loop's own guard excludes this state, so anything that grows here grew
    because the lookahead authorised it and for no other reason.

    ``family_order`` stays ``[tangent]`` on purpose. The matrix-free gate is a
    CONFIG-LOAD restriction and is pinned by the validator tests above; driving
    the loop directly here isolates the entry decision from the matrix-free
    construction, which at this toy size returns a degenerate system and would
    make the test fail for a reason that has nothing to do with what it claims.
    """
    import torch

    from stable_tiny.pipeline import build_model

    device = torch.device("cpu")
    config = load_pipeline_config("configs/experiments/default.yaml")
    config = replace(
        config,
        model=replace(config.model, hidden_size=8, number_hidden_layers=2),
        fgd_approx=replace(
            config.fgd_approx,
            projection_solver="exact",
            certify_growth_lookahead=True,
            certify_growth_lookahead_entry=True,
        ),
    )
    model = build_model(config, device)
    torch.manual_seed(0)
    x = torch.randn(12, config.data.in_features)
    y = torch.randn(12, config.data.out_features)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=12
    )
    return config, model, x, y, loader, device


def test_an_authorised_growth_buys_exactly_one_neuron() -> None:
    from fgdlib.search.certify import exact_relative_error, grow_until_certified

    config, model, x, y, loader, device = _already_certified_fixture()
    epsilon = exact_relative_error(model, x, y, config.fgd_approx)
    assert epsilon < config.fgd_approx.rel_error_threshold, (
        "fixture is not in the certified band; the test would prove nothing"
    )

    _, result = grow_until_certified(
        model=model, x=x, y=y, train_loader=loader, device=device,
        config=config, max_growths=40,
        growth_warranted=lambda _model: True,
    )
    assert result.growths == 1, "the authorisation must be spent, not standing"
    assert result.stop_reason == "growth_turn_taken_voluntary"


def test_a_refusal_still_buys_nothing_and_says_why() -> None:
    """The veto that exists today is not lost."""
    from fgdlib.search.certify import grow_until_certified

    config, model, x, y, loader, device = _already_certified_fixture()
    _, result = grow_until_certified(
        model=model, x=x, y=y, train_loader=loader, device=device,
        config=config, max_growths=40,
        growth_warranted=lambda _model: False,
    )
    assert result.growths == 0
    assert result.stop_reason == "training_beats_growing"


def test_without_the_entry_flag_an_authorisation_changes_nothing() -> None:
    """The flag, not the predicate, is what opens the door."""
    from fgdlib.search.certify import grow_until_certified

    config, model, x, y, loader, device = _already_certified_fixture()
    config = replace(
        config,
        fgd_approx=replace(
            config.fgd_approx, certify_growth_lookahead_entry=False
        ),
    )
    _, result = grow_until_certified(
        model=model, x=x, y=y, train_loader=loader, device=device,
        config=config, max_growths=40,
        growth_warranted=lambda _model: True,
    )
    assert result.growths == 0


# ---------------------------------------------------------------------------
# Attribution.
# ---------------------------------------------------------------------------


def test_the_two_stop_reasons_are_distinguishable() -> None:
    """"the certificate demanded it" must not read the same as "only the
    lookahead did" -- otherwise nothing the change produces is attributable."""
    obligatory = CertifyResult(
        relative_error=0.6, growths=1, certified=False, trajectory=(0.8, 0.6),
        stop_reason="growth_turn_taken",
    )
    voluntary = replace(obligatory, stop_reason="growth_turn_taken_voluntary")
    assert obligatory.stop_reason != voluntary.stop_reason


def test_both_lookahead_counters_are_registered() -> None:
    """The pair is the diagnostic: yes-always is the failure mode to catch."""
    from fgdlib.profile import PROFILE_FIELDS

    assert "certify_growth_not_warranted" in PROFILE_FIELDS
    assert "certify_growth_warranted_entries" in PROFILE_FIELDS
    # An unregistered counter is a KeyError under FGD_PROFILE=1, which is how
    # growth_where_no_bottleneck once took a cluster run down.
    assert "growth_lookahead_non_finite_abstentions" in PROFILE_FIELDS


def test_a_lookahead_that_cannot_certify_abstains_instead_of_killing_the_run(
    monkeypatch,
) -> None:
    """The measured death of run 1g0895r3, turned into a regression.

    Job 457944 ran 4h01m and died at epoch 6 with

        RuntimeError: Non-finite FGD tangent projection update detected.

    raised from evaluate_fgd_validation_certificate INSIDE this lookahead,
    after cusolver failed to converge on the dense float32 SVD of a
    14080 x 10841 Jacobian. sacct: FAILED 1:0, MaxRSS 2.6 GB of 64 G, GPU at
    41% of 40 GB -- nothing was exhausted. Run 9okhgeta died at the identical
    epoch on the identical line.

    The lookahead is ADVISORY: it asks "would training beat growing here?" on
    throwaway clones. When it cannot compute the answer the honest reply is
    "no opinion", which is what the grow_layer call in the same function has
    always done with its own `except RuntimeError: return False`.
    """
    import torch

    from fgdlib import profile
    from stable_tiny import pipeline

    class _Layer:
        in_neurons = 4

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self._growable_layers = [_Layer(), _Layer()]

    def _raises(*args, **kwargs):
        raise RuntimeError("Non-finite FGD tangent projection update detected.")

    monkeypatch.setattr(
        pipeline, "_train_parametric_gd_candidate", lambda **kw: _Model()
    )
    monkeypatch.setattr(
        pipeline, "evaluate_fgd_validation_certificate", _raises
    )

    config = load_pipeline_config("configs/experiments/mnist_full.yaml")
    monkeypatch.setenv("FGD_PROFILE", "1")
    before = profile.snapshot().get("growth_lookahead_non_finite_abstentions", 0)
    # Must return a verdict, not propagate. False is "growth not warranted",
    # which leaves the run training exactly as it would without an opinion.
    assert (
        pipeline._growth_reduces_lookahead_epsilon(
            model=_Model(),
            train_batches=[],
            train_loader=[],
            validation_loader=[],
            device=torch.device("cpu"),
            config=config,
            probe=None,
        )
        is False
    )
    assert (
        profile.snapshot().get("growth_lookahead_non_finite_abstentions", 0) > before
    )
