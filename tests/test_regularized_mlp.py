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

from fgdlib.growth import grow_layer, rank_layer_expansion_score
from fgdlib.regularized_mlp import ActivationThenDropout, make_post_layer_function
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
    assert isinstance(model.layers[0].post_layer_function, ActivationThenDropout)
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
    assert isinstance(
        model.layers[0].post_layer_function, ActivationThenDropout
    )


def test_extended_forward_protocol_passes_extension_undropped() -> None:
    """A new neuron must not be randomly zeroed before it is measured."""
    torch.manual_seed(0)
    composed = ActivationThenDropout(nn.SELU(), dropout_rate=1.0)  # drop all
    composed.train()
    x = torch.randn(4, 5)
    x_ext = torch.randn(4, 2)
    main, extension = composed.extended_forward(x, x_ext)
    # Main path fully dropped (rate 1.0) ...
    assert torch.count_nonzero(main) == 0
    # ... but the extension passes through the activation only, undropped.
    assert torch.allclose(extension, nn.SELU()(x_ext))
