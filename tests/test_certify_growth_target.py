"""The growth TARGET, separated from the step CERTIFICATE.

``rel_error_threshold`` was doing two incompatible jobs. It is the eps below
which Lemma 3.5 admits a STEP -- every step site reads it as
``eps < min(rel_error_threshold, 1/2)`` -- and it was also the eps the
grow-to-certify loop CHASED. Lowering it to make the loop aspire higher
therefore disabled the step it was trying to earn: MEASURED on MNIST at 0.3,
eps sat at 0.457, no damping and no rate certified, lr was 0 in every epoch,
and the only thing still moving the model was the family ladder at
functional_lr 1.0 with no size control (accuracy oscillating 0.33 <-> 0.74).

These tests pin the split, and above all pin that it is INERT by default: with
``certify_growth_target`` unset the value gate is true by the loop invariant
alone, not by a flag.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from fgdlib.profile import reset, snapshot
from fgdlib.search.certify import (
    _growth_pays,
    certify_growth_target,
    grow_until_certified,
)
from stable_tiny.pipeline import (
    build_model,
    lemma35_learning_rate,
    load_pipeline_config,
)

# Captured on the fixture below at the commit that introduced this file, with
# no target configured. Exact float equality, deliberately: the highest risk in
# the diff is that budget filtering or the value gate perturbs the argmin of
# `_select_growth_candidate` or its tie-break, and allclose would hide that.
BASELINE_TRAJECTORY_50 = (
    1.186815699899564,
    1.0577568130039785,
    1.030305823166083,
    1.0088216530804581,
    0.896136596375565,
    0.7998091816513789,
    0.7125855218563423,
    0.6646540384628474,
    0.5920657306661673,
    0.5362022902870865,
    0.49632628441223814,
)
BASELINE_TRAJECTORY_100 = (
    1.4255240620608867,
    1.3505467859837765,
    1.2901353022141333,
    1.2485116442667343,
    1.2139233875421807,
    1.1868036532674808,
    1.1403740036413785,
    1.1023095998661727,
    0.9888444693416303,
    0.971902551964388,
    0.9545355863781892,
    0.9259445836252135,
    0.8697393965943009,
    0.8514478419995422,
    0.8104349214631728,
    0.7877423836020038,
    0.765535674483758,
    0.7482051105454521,
    0.7294257523396631,
    0.7121398214840123,
    0.7000125126075349,
    0.6772371048118412,
    0.6373815101806314,
    0.6225846302112811,
    0.6116457517514786,
    0.5928080007899355,
    0.5844524893207976,
    0.5649518232936152,
    0.5564410543155777,
    0.5334041332697185,
    0.524556634665944,
    0.5076289893270066,
    0.49497297408867713,
)


def _fixture(samples: int, hidden_size: int = 3, **fgd_approx):
    """The fixture of ``tests/test_grow_to_certify_loop.py``, verbatim."""
    device = torch.device("cpu")
    config = load_pipeline_config("configs/experiments/default.yaml")
    config = replace(
        config,
        model=replace(
            config.model, hidden_size=hidden_size, number_hidden_layers=2
        ),
        fgd_approx=replace(
            config.fgd_approx, projection_solver="exact", **fgd_approx
        ),
    )
    model = build_model(config, device)
    torch.manual_seed(0)
    x = torch.randn(samples, config.data.in_features)
    y = torch.randn(samples, config.data.out_features)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=min(samples, 25)
    )
    return config, model, x, y, loader, device


def _approx_config(**fields):
    config = load_pipeline_config("configs/experiments/default.yaml")
    return replace(config.fgd_approx, **fields)


# --- the two pure predicates ---------------------------------------------


def test_the_default_target_is_the_configured_threshold() -> None:
    """Unset means "chase the certificate", at every threshold."""
    for threshold in (0.3, 0.5, 0.7):
        config = _approx_config(
            rel_error_threshold=threshold, certify_growth_target=None
        )
        assert certify_growth_target(config) == threshold


def test_the_target_is_clipped_to_the_threshold_and_never_to_a_half() -> None:
    """It may only ask for MORE structure, and the clamp is the THRESHOLD.

    ``min(target, rel_error_threshold)`` -- not ``min(target, 0.5)``. A
    threshold above 1/2 is a deliberate looseness for the LOOP (see
    ``tests/test_fast_factorization.py``, which grows at 0.7 on purpose to
    reach an ill-conditioned structure); the STEP clamps itself at 1/2
    independently, at every site that reads Lemma 3.5. Clamping here as well
    would silently tighten those runs.
    """
    laxer = _approx_config(rel_error_threshold=0.3, certify_growth_target=0.9)
    assert certify_growth_target(laxer) == 0.3

    stricter = _approx_config(
        rel_error_threshold=0.5, certify_growth_target=0.3
    )
    assert certify_growth_target(stricter) == 0.3

    above_a_half = _approx_config(
        rel_error_threshold=0.7, certify_growth_target=0.6
    )
    assert certify_growth_target(above_a_half) == 0.6


def test_with_no_target_the_gate_is_true_by_the_loop_invariant() -> None:
    """The default identity is STRUCTURAL, not a flag check.

    With ``target == rel_error_threshold`` the certificate
    ``min(rel_error_threshold, 1/2)`` is never above the target, so the loop's
    own invariant ``eps >= target`` already implies ``eps >= certificate`` --
    the regime in which ``_growth_pays`` returns True unconditionally. Nothing
    in the loop can reach the value criterion, whatever the candidate does.
    """
    for threshold in (0.3, 0.5, 0.7):
        target = threshold
        certificate = min(threshold, 0.5)
        for epsilon in (target, target + 1e-12, threshold + 0.5, 3.0, float("inf")):
            for candidate_epsilon in (
                epsilon,                 # no gain at all
                epsilon + 1.0,           # strictly worse
                epsilon - 1e-15,         # a gain far below any floor
                float("inf"),
            ):
                assert _growth_pays(
                    epsilon=epsilon,
                    candidate_epsilon=candidate_epsilon,
                    growth_target=target,
                    certificate=certificate,
                    min_gain=0.1,
                ), (threshold, epsilon, candidate_epsilon)


def test_the_gate_cannot_deadlock_the_loop() -> None:
    """It never refuses while no step certifies -- the anti-deadlock claim.

    Refusing above the certificate would stop the loop in a state where no
    damping and no learning rate exist, which is exactly the freeze this whole
    design is meant to prevent. So even a candidate that gains essentially
    nothing, judged by an essentially zero floor, must be accepted there.
    """
    assert _growth_pays(
        epsilon=0.55,
        candidate_epsilon=0.55 - 1e-9,
        growth_target=0.3,
        certificate=0.5,
        min_gain=1e-9,
    )
    # And with no gain whatsoever, and an enormous floor.
    assert _growth_pays(
        epsilon=0.55,
        candidate_epsilon=0.55,
        growth_target=0.3,
        certificate=0.5,
        min_gain=1e6,
    )


def test_the_measured_mnist_trajectory_stops_chasing() -> None:
    """MEASURED on mnist_streaming.yaml: eps 0.457, gain 0.0007 per growth.

    Gap to the 0.3 target is 0.157, so the floor at min_gain 0.1 is 0.0157 and
    the growth is 22x below it. The ratio that decides -- gain/gap = 0.0045 --
    is dimensionless, so nothing about MNIST enters the predicate.
    """
    assert not _growth_pays(
        epsilon=0.457,
        candidate_epsilon=0.457 - 0.0007,
        growth_target=0.3,
        certificate=0.5,
        min_gain=0.1,
    )
    # min_gain 0.0 is the refutation experiment: it re-licenses the growth
    # this criterion refused, so a trajectory suspected of being cut short can
    # be re-run without it.
    assert _growth_pays(
        epsilon=0.457,
        candidate_epsilon=0.457 - 0.0007,
        growth_target=0.3,
        certificate=0.5,
        min_gain=0.0,
    )


def test_the_measured_n1024_trajectory_keeps_growing() -> None:
    """MEASURED on family_ladder_N1024.yaml: eps 0.3267, gain 0.00707.

    The same predicate, the same target and the same floor, two orders of
    magnitude the other way: gain/gap is 0.265 here against MNIST's 0.0045, so
    min_gain 0.1 sits between them with 2.6x and 22x of margin. Every growth
    along the measured descent pays, including the one that crosses 0.3.
    """
    gain = 0.00707
    epsilons = [0.3267 - index * gain for index in range(5)]
    for epsilon in epsilons[:-1]:
        assert _growth_pays(
            epsilon=epsilon,
            candidate_epsilon=epsilon - gain,
            growth_target=0.3,
            certificate=0.5,
            min_gain=0.1,
        ), epsilon


# --- the loop ------------------------------------------------------------


def test_the_loop_stops_where_a_step_can_still_be_taken() -> None:
    """The entire point of the design, end to end.

    Chasing an unreachable 0.05 with a demanding floor, the loop grows while no
    step exists (eps >= 1/2, growth mandatory) and stops at the FIRST voluntary
    growth that does not pay. What it returns is not a failure: eps is below
    the certificate, so Lemma 3.5 yields a rate and the outer step commits --
    which is precisely what a threshold of 0.05 would have forbidden for ever.

    The trajectory is the untargeted one, unchanged: the refused growth is
    evaluated BEFORE the model is replaced, so it costs neither a parameter nor
    an entry.
    """
    config, model, x, y, loader, device = _fixture(
        samples=50, certify_growth_target=0.05, certify_growth_min_gain=0.5
    )
    grown, result = grow_until_certified(
        model=model, x=x, y=y, train_loader=loader, device=device,
        config=config, max_growths=60,
    )
    assert result.stop_reason == "growth_target_unreachable"
    assert result.certified is False
    assert result.growth_target == 0.05
    assert result.relative_error < 0.5
    assert lemma35_learning_rate(result.relative_error, config.fgd_approx) is not None
    assert result.trajectory == BASELINE_TRAJECTORY_50
    assert result.growths == len(BASELINE_TRAJECTORY_50) - 1
    assert sum(p.numel() for p in grown.parameters()) == 503


def test_the_untargeted_trajectory_is_byte_identical() -> None:
    """The harness for the highest risk in the diff.

    Captured at HEAD before the change and compared with ``==`` on floats: if
    the parameter-budget filter or the value gate perturbed the argmin of
    ``_select_growth_candidate`` or its enumeration-order tie-break, the
    trajectory would move in the last bits and nothing else would notice.
    """
    for samples, expected in (
        (50, BASELINE_TRAJECTORY_50),
        (100, BASELINE_TRAJECTORY_100),
    ):
        config, model, x, y, loader, device = _fixture(
            samples=samples, max_total_parameters=None
        )
        _, result = grow_until_certified(
            model=model, x=x, y=y, train_loader=loader, device=device,
            config=config, max_growths=60,
        )
        assert result.trajectory == expected, samples
        assert result.stop_reason == "certified"
        assert result.certified
        assert result.growth_target == config.fgd_approx.rel_error_threshold


def test_an_exhausted_parameter_budget_spends_nothing() -> None:
    """``max_total_parameters`` existed but was invisible to this loop.

    Two ways to be out of budget, both reported as the budget and not as an
    inadequate structure. The fixture starts at 57 parameters and its cheapest
    growth costs 14, so a budget of 60 has headroom that nothing fits into: the
    candidates are priced by their POST-growth count and dropped before the
    O(P^3) scoring, exactly as the end-of-epoch path prices them.
    """
    for budget in (57, 60):
        config, model, x, y, loader, device = _fixture(
            samples=50, max_total_parameters=budget
        )
        before = sum(p.numel() for p in model.parameters())
        grown, result = grow_until_certified(
            model=model, x=x, y=y, train_loader=loader, device=device,
            config=config, max_growths=60,
        )
        assert result.stop_reason == "parameter_budget", budget
        assert result.growths == 0, budget
        assert len(result.trajectory) == 1, budget
        assert sum(p.numel() for p in grown.parameters()) == before


def test_the_profiler_counts_stalls_and_budget_stops(monkeypatch) -> None:
    """No silent stop: every cut leaves a counter behind.

    The stall counter is what refutes risk 1 -- a target cut short shows up as
    stalls without the run ever reaching the certificate -- and the rejected
    counter separates "could not pay" from "could not represent".
    """
    monkeypatch.setenv("FGD_PROFILE", "1")

    reset()
    config, model, x, y, loader, device = _fixture(
        samples=50, certify_growth_target=0.05, certify_growth_min_gain=0.5
    )
    _, stalled = grow_until_certified(
        model=model, x=x, y=y, train_loader=loader, device=device,
        config=config, max_growths=60,
    )
    values = snapshot()
    assert stalled.stop_reason == "growth_target_unreachable"
    assert values["certify_growth_target_stalls"] == 1
    assert values["certify_budget_stops"] == 0
    assert values["certify_budget_rejected_candidates"] == 0

    reset()
    config, model, x, y, loader, device = _fixture(
        samples=50, max_total_parameters=60
    )
    _, budgeted = grow_until_certified(
        model=model, x=x, y=y, train_loader=loader, device=device,
        config=config, max_growths=60,
    )
    values = snapshot()
    assert budgeted.stop_reason == "parameter_budget"
    assert values["certify_growth_target_stalls"] == 0
    assert values["certify_budget_stops"] == 1
    # Both growable locations were cloned, priced and dropped: 57 + 42 and
    # 57 + 14 both exceed 60.
    assert values["certify_budget_rejected_candidates"] == 2
    assert values["where_candidates"] == 0
