"""Dropout inside the growing MLP: regularizes training, invisible to certs.

The load-bearing claim (report SENN/CE §): the certificate is measured in
eval, so dropout is the identity there and cannot change any certified
quantity. These tests pin that, plus that the default builds the plain MLP
unchanged (the MNIST-preservation bar) and that growth still works.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from fgdlib.search.growth import grow_layer, rank_layer_expansion_score
from fgdlib.models.regularized_mlp import (
    _hidden_norm,
    make_hidden_post_function,
    make_post_layer_function,
)
from fgdlib.tangent import tiny_optimal_update_kwargs
from stable_tiny.pipeline import (
    build_dataloaders,
    build_model,
    build_projection_probe,
    evaluate_fgd_validation_certificate,
    load_pipeline_config,
)


def _config(dropout_rate: float, hidden: int = 8):
    config = load_pipeline_config("configs/fgd/search_ce_unified.yaml")
    return replace(
        config,
        model=replace(
            config.model,
            hidden_size=hidden,
            number_hidden_layers=2,
            dropout_rate=dropout_rate,
        ),
        data=replace(
            config.data,
            mnist_train_samples=1000,
            mnist_validation_samples=500,
            mnist_test_samples=500,
        ),
    )


def test_zero_rate_builds_the_plain_mlp() -> None:
    """The MNIST-preservation bar: rate 0 is the activation itself."""
    activation = nn.SELU()
    assert make_post_layer_function(activation, 0.0) is activation
    model = build_model(_config(0.0), torch.device("cpu"))
    assert isinstance(model.layers[0].post_layer_function, nn.SELU)


def test_dropout_only_on_hidden_layers() -> None:
    model = build_model(_config(0.3), torch.device("cpu"))
    assert isinstance(model.layers[0].post_layer_function, nn.Sequential)
    # The output layer is never regularized.
    assert isinstance(model.layers[-1].post_layer_function, nn.Identity)


def test_certificate_is_identical_with_and_without_dropout() -> None:
    """Eval turns dropout into the identity, so no certificate can move."""
    device = torch.device("cpu")
    config = _config(0.0)
    _, validation, _ = build_dataloaders(config, device)
    probe = build_projection_probe(
        validation, config.fgd_approx.probe_batches, device
    )

    plain = build_model(config, device)
    dropped = build_model(_config(0.3), device)
    # Same linear weights (dropout carries no parameters).
    dropped.load_state_dict(plain.state_dict(), strict=False)

    def eps(model):
        return evaluate_fgd_validation_certificate(
            model=model,
            data_loader=validation,
            device=device,
            config=config.fgd_approx,
            learning_rate=None,
            probe=probe,
        ).relative_error

    assert abs(eps(plain) - eps(dropped)) < 1e-9


def test_growth_still_function_preserving_with_dropout() -> None:
    device = torch.device("cpu")
    config = _config(0.3)
    train, _, _ = build_dataloaders(config, device)
    model = build_model(config, device)
    kwargs = tiny_optimal_update_kwargs(config.fgd_approx, compute_delta=True)

    # The SENN expansion score is computable through the regularized layer.
    assert rank_layer_expansion_score(model, train, 0, device, kwargs) >= 0.0

    depth_before = len(model.layers)
    grow_layer(
        model=model,
        train_loader=train,
        layer_index=0,
        device=device,
        line_search_config=config.scaling_line_search,
        optimal_update_kwargs=kwargs,
        function_preserving=True,
        preservation_tolerance=config.fgd_approx.growth_preservation_tolerance,
    )
    assert len(model.layers) == depth_before          # width grew, not depth
    assert isinstance(model.layers[0].post_layer_function, nn.Sequential)


def test_pipeline_growth_path_works_with_batchnorm() -> None:
    """The path that crashed: select_tiny_growth_layer_index reads the
    activation gradient of the post-function. With batch-norm in a Sequential
    GroMo skips it and uses the activation's known derivative; with a custom
    module it fell back to a numerical grad on a 0-D scalar, which batch-norm
    cannot process. This test exercises that exact path."""
    from fgdlib.tangent import select_tiny_growth_layer_index

    device = torch.device("cpu")
    config = replace(
        _config(0.2), model=replace(_config(0.2).model, use_batchnorm=True)
    )
    train, _, _ = build_dataloaders(config, device)
    model = build_model(config, device)
    # Must not raise (previously: ValueError expected 2D/3D input, got 0D).
    index = select_tiny_growth_layer_index(
        model=model, train_loader=train, device=device, config=config.fgd_approx
    )
    assert index is None or isinstance(index, int)


def test_batchnorm_is_per_layer_and_off_the_output() -> None:
    config = replace(_config(0.0), model=replace(_config(0.0).model, use_batchnorm=True))
    model = build_model(config, torch.device("cpu"))
    for hidden in list(model.layers)[:-1]:
        post = hidden.post_layer_function
        assert isinstance(post, nn.Sequential)
        norm = _hidden_norm(post)
        # Each hidden batch-norm has its OWN running statistics.
        assert norm is not None and norm.num_features == int(hidden.out_features)
    assert isinstance(model.layers[-1].post_layer_function, nn.Identity)
    # Distinct instances, not a shared one.
    assert (
        _hidden_norm(model.layers[0].post_layer_function)
        is not _hidden_norm(model.layers[1].post_layer_function)
    )


def test_batchnorm_growth_preserves_function_and_syncs() -> None:
    """The load-bearing test: per-feature BN grown in sync stays exact.

    LayerNorm cannot pass this -- it couples features, so a new neuron shifts
    the statistics of the existing ones. That is why only batch-norm is
    offered.
    """
    device = torch.device("cpu")
    config = replace(
        _config(0.0), model=replace(_config(0.0).model, use_batchnorm=True)
    )
    train, _, _ = build_dataloaders(config, device)
    model = build_model(config, device)
    kwargs = tiny_optimal_update_kwargs(config.fgd_approx, compute_delta=True)

    norm_before = _hidden_norm(model.layers[0].post_layer_function).num_features
    grow_layer(
        model=model,
        train_loader=train,
        layer_index=0,
        device=device,
        line_search_config=config.scaling_line_search,
        optimal_update_kwargs=kwargs,
        function_preserving=True,          # drift check is the guard
        preservation_tolerance=config.fgd_approx.growth_preservation_tolerance,
    )
    # The paired norm grew with the layer, so a forward still runs ...
    norm_after = _hidden_norm(model.layers[0].post_layer_function).num_features
    assert norm_after == int(model.layers[0].out_features)
    assert norm_after > norm_before
    model(torch.randn(4, config.data.in_features))     # no dimension error


def test_batchnorm_default_off_builds_plain_mlp() -> None:
    """MNIST-preservation bar for the batch-norm feature."""
    model = build_model(_config(0.0), torch.device("cpu"))
    assert isinstance(model.layers[0].post_layer_function, nn.SELU)


def test_sync_normalization_is_a_noop_without_norm() -> None:
    """Safe to call from grow_layer on the plain MLP."""
    from fgdlib.models.regularized_mlp import sync_normalization

    model = build_model(_config(0.0), torch.device("cpu"))
    before = [p.detach().clone() for p in model.parameters()]
    sync_normalization(model)
    for original, current in zip(before, model.parameters()):
        assert torch.equal(original, current)
