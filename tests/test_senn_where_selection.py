"""SENN's *where*: the expansion-score ranking used to place new capacity.

`test_senn_expansion_score.py` validates the theory (the bridge to Lemma 3.5
and Theorem 3.2). This file pins the ranking actually wired into the growth
loop, which reads GroMo's TINY spectrum.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import torch

from fgdlib.growth import grow_layer, rank_layer_expansion_score
from fgdlib.tangent import tiny_optimal_update_kwargs
from stable_tiny.pipeline import build_model, load_pipeline_config


def _fixture(hidden_size: int = 3, samples: int = 128):
    device = torch.device("cpu")
    config = load_pipeline_config("configs/fgd/default.yaml")
    config = replace(
        config,
        model=replace(
            config.model, hidden_size=hidden_size, number_hidden_layers=3
        ),
    )
    model = build_model(config, device)
    torch.manual_seed(0)
    inputs = torch.randn(samples, config.data.in_features)
    targets = torch.randn(samples, config.data.out_features)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets), batch_size=64
    )
    return config, model, loader, device


def _functional_loss(model, loader) -> float:
    with torch.no_grad():
        return float(
            sum(
                torch.nn.functional.mse_loss(model(x), y, reduction="sum")
                for x, y in loader
            )
        )


def test_score_ranks_layers_like_the_realized_descent() -> None:
    """The property that makes the score a legitimate *where* criterion.

    GroMo documents the extension's first-order effect as
    ``L(A+dA) = L(A) - t*sigma'(0)*(eigenvalues_extension**2).sum()``, so the
    score should order locations the same way actually growing them does.
    That is checked here against the realized functional descent rather than
    asserted from the docstring.
    """
    config, model, loader, device = _fixture()
    kwargs = tiny_optimal_update_kwargs(config.fgd_approx, compute_delta=True)
    base_loss = _functional_loss(model, loader)

    scores: list[float] = []
    descents: list[float] = []
    for layer_index in range(3):
        scores.append(
            rank_layer_expansion_score(model, loader, layer_index, device, kwargs)
        )
        trial = copy.deepcopy(model)
        grow_layer(
            model=trial,
            train_loader=loader,
            layer_index=layer_index,
            device=device,
            line_search_config=config.scaling_line_search,
            optimal_update_kwargs=kwargs,
            function_preserving=False,
        )
        descents.append(base_loss - _functional_loss(trial, loader))

    order_by_score = sorted(range(3), key=lambda i: -scores[i])
    order_by_descent = sorted(range(3), key=lambda i: -descents[i])
    assert order_by_score == order_by_descent


def test_the_score_favours_the_input_layer() -> None:
    """The controlled difference from R2, which divided by parameter cost.

    R2's per-parameter division always bought the cheap late layer and
    starved the 784-wide input projection (measured end-to-end: 784->2->2->14,
    64.4 %). The raw score must not have that bias.
    """
    config, model, loader, device = _fixture()
    kwargs = tiny_optimal_update_kwargs(config.fgd_approx, compute_delta=True)
    scores = [
        rank_layer_expansion_score(model, loader, index, device, kwargs)
        for index in range(3)
    ]
    assert scores[0] == max(scores)


def test_ranking_leaves_the_model_untouched() -> None:
    """Ranking is a measurement: it must add no capacity and keep no update.

    The whole cost argument for SENN's where rests on this -- the ranking
    stops after the statistics pass and the SVD, so the golden-section line
    search is paid once on the winner instead of once per candidate.
    """
    config, model, loader, device = _fixture()
    kwargs = tiny_optimal_update_kwargs(config.fgd_approx, compute_delta=True)

    before = [p.detach().clone() for p in model.parameters()]
    count_before = sum(p.numel() for p in model.parameters())

    for layer_index in range(3):
        rank_layer_expansion_score(model, loader, layer_index, device, kwargs)

    after = list(model.parameters())
    assert sum(p.numel() for p in after) == count_before
    for original, current in zip(before, after):
        assert torch.equal(original, current)


def test_ranking_never_runs_the_line_search(monkeypatch) -> None:
    """Pin the cost claim rather than trusting it."""
    import fgdlib.growth as growth_module

    config, model, loader, device = _fixture()
    kwargs = tiny_optimal_update_kwargs(config.fgd_approx, compute_delta=True)

    def _fail(*args, **kwargs_inner):
        raise AssertionError("the ranking must not invoke the line search")

    monkeypatch.setattr(growth_module, "_golden_section_line_search", _fail)
    rank_layer_expansion_score(model, loader, 0, device, kwargs)


def test_fisher_preconditioning_changes_the_score() -> None:
    """`tiny_use_fisher` selects the output metric the score is measured in.

    It is one decision, not two: the same SVD yields the singular values (the
    score) and the singular vectors alpha/omega (the new neurons' initial
    weights), so preconditioning moves both together. SENN's Ingredient 2 --
    "choose the initialization that maximises Delta eta" -- is automatic here
    for that reason, but so is the failure mode.

    MEASURED, and why the shipped config leaves this off: with the Fisher
    factor the score ranks layer 2 first (37.7 vs 11.0) while the scaling
    line search then rejects that extension outright (scaling 0.000, worst
    realised loss). End-to-end, 7 of 8 growth events took scaling 0 and the
    run stalled at 65.6 % accumulating dead capacity. SENN's own theory is
    stated in the Euclidean metric (their 3.4), which is also the metric the
    bridge identity is derived under; KFAC is their approximation to it, and
    at these widths it is not needed.

    This test only pins that the flag is genuinely wired through to the
    score, so the choice above stays a measured one rather than a no-op.
    """
    config, model, loader, device = _fixture()
    plain = tiny_optimal_update_kwargs(
        replace(config.fgd_approx, tiny_use_fisher=False), compute_delta=True
    )
    fisher = tiny_optimal_update_kwargs(
        replace(config.fgd_approx, tiny_use_fisher=True), compute_delta=True
    )
    plain_scores = [
        rank_layer_expansion_score(model, loader, i, device, plain)
        for i in range(3)
    ]
    fisher_scores = [
        rank_layer_expansion_score(model, loader, i, device, fisher)
        for i in range(3)
    ]
    assert plain_scores != fisher_scores
    assert all(score >= 0.0 for score in plain_scores + fisher_scores)
