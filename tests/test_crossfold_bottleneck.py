"""Stopping without a parameter cap: is the bottleneck real or fitted to noise?

Every N=1024 run ends by exhausting its parameter allowance, never by deciding
it is done -- and a cap is a number nobody knows in advance. MEASURED, that is
not a cosmetic defect: on the easiest constructible target the method reaches
test 1.000 at 74 parameters and grows to 872 anyway, and three of four seeds
land on 602 parameters against a cap of 600, so the shape is where the ribbon
was cut rather than anything the data asked for.

Marchenko-Pastur was the first answer and is REFUTED (see
``FGDApproxConfig.growth_bottleneck_significance``): it needs ``gamma = r / n``
to be appreciable and here it is ~0.015. The cross-validated test asks a
question that has no asymptotic regime in it -- does the extension direction
still point at what data it was NOT fitted on wants? -- and these tests pin the
three things that makes it usable: it is off by default and inert when off, it
separates a real direction from a fitted one, and it leaves the model alone.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.tangent import (
    FGDApproxConfig,
    _held_out_bottleneck_cosine,
    _stride_fold_masks,
    compute_expressivity_bottlenecks,
    crossfold_bottleneck_significance,
    validate_bottleneck_stopping,
)

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP  # noqa: E402


CPU = torch.device("cpu")


def _model(seed: int = 0) -> GrowingMLP:
    torch.manual_seed(seed)
    return GrowingMLP(
        in_features=4,
        out_features=1,
        hidden_size=3,
        number_hidden_layers=3,
        activation=torch.nn.SELU(),
        device=CPU,
    )


def _loader(*, structured: bool, rows: int = 400, seed: int = 1) -> DataLoader:
    """Two targets of the same shape, one reachable and one not.

    ``structured`` is a smooth function OF the inputs, so a layer really is
    short of the width to express it. The noise target is drawn independently
    of ``x``, so any extension direction fitted to it is fitted to the sample
    and to nothing else -- which is exactly the case the criterion has to
    refuse, and the case a magnitude threshold cannot see because the fitted
    residual is large either way.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, 4, generator=generator)
    if structured:
        y = torch.sin(2.0 * x[:, :1]) + 0.5 * x[:, 1:2] * x[:, 2:3]
    else:
        y = torch.randn(rows, 1, generator=generator)
    return DataLoader(TensorDataset(x, y), batch_size=100, shuffle=False)


def _config(folds: int = 0, bar: float = 1.0) -> FGDApproxConfig:
    return FGDApproxConfig(
        growth_where="expressivity_bottleneck",
        tiny_maximum_added_neurons=1,
        growth_bottleneck_crossfold_folds=folds,
        growth_bottleneck_crossfold_t=bar,
    )


# ---------------------------------------------------------------------------
# Off is off: nothing above the second pass may differ.
# ---------------------------------------------------------------------------


def test_disabled_leaves_every_bottleneck_untouched() -> None:
    model, loader = _model(), _loader(structured=True)
    baseline = compute_expressivity_bottlenecks(model, loader, CPU, _config())
    again = compute_expressivity_bottlenecks(model, loader, CPU, _config(folds=0))
    assert baseline == again
    assert any(value > 0.0 for value in baseline)


def test_enabled_only_ever_zeroes_a_bottleneck_never_moves_one() -> None:
    """The ranking among the survivors is the in-sample one, untouched.

    Not the same as "it never changes WHERE": zeroing is per layer, so
    zeroing the argmax hands the turn to the runner-up, and MEASURED on seed 1
    that relocated the shape from 7-11-12 to 4-9-20 without refusing a single
    growth. What this pins is narrower and is what makes the shape
    attributable -- no surviving value is rescaled, so any change in where
    growth went is the eligibility gate and never a silent re-ranking.
    """
    model, loader = _model(), _loader(structured=True)
    baseline = compute_expressivity_bottlenecks(model, loader, CPU, _config())
    gated = compute_expressivity_bottlenecks(model, loader, CPU, _config(folds=5))

    assert len(gated) == len(baseline)
    for before, after in zip(baseline, gated):
        assert after in (0.0, before)


# ---------------------------------------------------------------------------
# It discriminates, which is the entire claim.
# ---------------------------------------------------------------------------


def test_a_real_direction_scores_positive_on_every_held_out_fold() -> None:
    model, loader = _model(), _loader(structured=True)
    bottlenecks = compute_expressivity_bottlenecks(model, loader, CPU, _config())
    layer = max(range(len(bottlenecks)), key=bottlenecks.__getitem__)

    inputs = torch.cat([x for x, _ in loader])
    targets = torch.cat([y for _, y in loader])
    statistic, samples = crossfold_bottleneck_significance(
        model=model,
        layer_index=layer,
        inputs=inputs,
        targets=targets,
        batch_size=100,
        device=CPU,
        config=_config(folds=5),
    )

    assert len(samples) == 5
    assert all(value > 0.0 for value in samples)
    assert statistic > 1.0


def test_a_direction_fitted_to_noise_does_not_survive_its_own_folds() -> None:
    """The bottleneck is nonzero and the criterion still refuses it.

    This is the case that separates the test from every magnitude threshold
    tried before: the in-sample value is a perfectly ordinary positive number,
    and only asking data the direction never saw exposes it.
    """
    model, loader = _model(), _loader(structured=False)
    bottlenecks = compute_expressivity_bottlenecks(model, loader, CPU, _config())
    assert all(value > 0.0 for value in bottlenecks)

    gated = compute_expressivity_bottlenecks(model, loader, CPU, _config(folds=5))
    assert any(value == 0.0 for value in gated)
    assert sum(value > 0.0 for value in gated) < len(bottlenecks)


def test_the_structured_layer_beats_the_noise_layers_by_an_order_of_magnitude() -> None:
    """The separation, not the bar, is what the criterion rests on."""
    inputs_structured = torch.cat([x for x, _ in _loader(structured=True)])
    targets_structured = torch.cat([y for _, y in _loader(structured=True)])
    inputs_noise = torch.cat([x for x, _ in _loader(structured=False)])
    targets_noise = torch.cat([y for _, y in _loader(structured=False)])

    def best(inputs: torch.Tensor, targets: torch.Tensor) -> float:
        model = _model()
        return max(
            crossfold_bottleneck_significance(
                model=model,
                layer_index=index,
                inputs=inputs,
                targets=targets,
                batch_size=100,
                device=CPU,
                config=_config(folds=5),
            )[0]
            for index in range(len(model._growable_layers))
        )

    assert best(inputs_structured, targets_structured) > 10.0 * best(
        inputs_noise, targets_noise
    )


# ---------------------------------------------------------------------------
# Determinism and non-interference.
# ---------------------------------------------------------------------------


def test_the_statistic_is_bit_identical_across_calls() -> None:
    """The pipeline is deterministic; the fold split must not be what breaks it."""
    model, loader = _model(), _loader(structured=True)
    inputs = torch.cat([x for x, _ in loader])
    targets = torch.cat([y for _, y in loader])
    kwargs = {
        "model": model,
        "layer_index": 0,
        "inputs": inputs,
        "targets": targets,
        "batch_size": 100,
        "device": CPU,
        "config": _config(folds=5),
    }
    first, first_samples = crossfold_bottleneck_significance(**kwargs)
    second, second_samples = crossfold_bottleneck_significance(**kwargs)
    assert first == second
    assert first_samples == second_samples


def test_the_model_is_returned_exactly_as_it_arrived() -> None:
    model, loader = _model(), _loader(structured=True)
    probe = torch.randn(8, 4, generator=torch.Generator().manual_seed(7))
    before_output = model(probe).detach().clone()
    before_parameters = [p.detach().clone() for p in model.parameters()]

    compute_expressivity_bottlenecks(model, loader, CPU, _config(folds=5))

    assert torch.equal(model(probe).detach(), before_output)
    for before, after in zip(before_parameters, model.parameters()):
        assert torch.equal(before, after.detach())


def test_folds_are_a_disjoint_cover_taken_by_stride() -> None:
    masks = _stride_fold_masks(10, 3)
    stacked = torch.stack(masks)
    assert torch.equal(stacked.sum(dim=0), torch.ones(10, dtype=torch.long))
    assert torch.equal(masks[0], torch.tensor([True, False, False] * 3 + [True]))


# ---------------------------------------------------------------------------
# Guard rails: never stop because the measurement failed.
# ---------------------------------------------------------------------------


def test_too_few_rows_reports_not_measured_rather_than_not_significant() -> None:
    model = _model()
    statistic, samples = crossfold_bottleneck_significance(
        model=model,
        layer_index=0,
        inputs=torch.randn(4, 4),
        targets=torch.randn(4, 1),
        batch_size=4,
        device=CPU,
        config=_config(folds=5),
    )
    assert statistic == float("inf")
    assert samples == []


def test_mismatched_shapes_are_not_measured_rather_than_scored_zero() -> None:
    assert _held_out_bottleneck_cosine(
        torch.ones(2, 3), torch.ones(3, 2)
    ) != _held_out_bottleneck_cosine(torch.ones(2, 3), torch.zeros(2, 3))


def test_a_zero_held_out_target_scores_zero_not_nan() -> None:
    assert _held_out_bottleneck_cosine(torch.ones(2, 2), torch.zeros(2, 2)) == 0.0


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


def test_the_two_stopping_rules_may_not_run_together() -> None:
    with pytest.raises(ValueError, match="alternative stopping rules"):
        validate_bottleneck_stopping(
            replace(_config(folds=5), growth_bottleneck_significance=True)
        )


def test_a_single_fold_is_refused() -> None:
    with pytest.raises(ValueError, match="a single fold holds nothing out"):
        validate_bottleneck_stopping(_config(folds=1))


def test_the_criterion_requires_the_bottleneck_growth_rule() -> None:
    with pytest.raises(ValueError, match="expressivity_bottleneck"):
        validate_bottleneck_stopping(
            replace(_config(folds=5), growth_where="certified_gain")
        )


def test_the_default_configuration_is_valid_and_disabled() -> None:
    validate_bottleneck_stopping(FGDApproxConfig())
    assert FGDApproxConfig().growth_bottleneck_crossfold_folds == 0
