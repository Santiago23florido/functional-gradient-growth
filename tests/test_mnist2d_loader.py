"""The 2-D MNIST loader must be the flat one, reshaped -- nothing else.

The conv arms are compared against a LINEAR ladder run on "the same 1024
images". That sentence is only true if the two loaders draw the same
examples in the same order with the same labels, so these tests pin the
identity rather than trusting that two code paths were kept in step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stable_tiny.data import make_mnist_dataloaders
from stable_tiny.pipeline import DataConfig, PipelineConfig, build_dataloaders


def _mnist_available() -> bool:
    from stable_tiny.data import _mnist_root_candidates

    required = (
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    )
    return any(
        all((candidate / name).exists() for name in required)
        for candidate in _mnist_root_candidates(None)
    )


needs_mnist = pytest.mark.skipif(
    not _mnist_available(), reason="MNIST IDX files not present"
)

SAMPLES = dict(train_samples=128, validation_samples=64, test_samples=64)


@needs_mnist
def test_image_shape_reshapes_every_split() -> None:
    loaders = make_mnist_dataloaders(
        batch_size=32, seed=0, image_shape=(1, 28, 28), **SAMPLES
    )
    for loader in loaders:
        x, y = next(iter(loader))
        assert x.shape[1:] == (1, 28, 28)
        assert y.shape[1:] == (10,)


@needs_mnist
def test_the_2d_loader_is_the_flat_one_reshaped() -> None:
    """Same examples, same order, same labels -- to the bit."""
    flat = make_mnist_dataloaders(batch_size=32, seed=0, **SAMPLES)
    conv = make_mnist_dataloaders(
        batch_size=32, seed=0, image_shape=(1, 28, 28), **SAMPLES
    )
    for flat_loader, conv_loader in zip(flat, conv):
        flat_batches = list(flat_loader)
        conv_batches = list(conv_loader)
        assert len(flat_batches) == len(conv_batches)
        for (x_flat, y_flat), (x_conv, y_conv) in zip(flat_batches, conv_batches):
            assert torch.equal(x_conv.flatten(1), x_flat)
            assert torch.equal(y_conv, y_flat)


@needs_mnist
def test_flat_loader_is_untouched_when_image_shape_is_absent() -> None:
    a = make_mnist_dataloaders(batch_size=32, seed=1, **SAMPLES)
    b = make_mnist_dataloaders(batch_size=32, seed=1, image_shape=None, **SAMPLES)
    for loader_a, loader_b in zip(a, b):
        for (xa, ya), (xb, yb) in zip(loader_a, loader_b):
            assert torch.equal(xa, xb)
            assert torch.equal(ya, yb)
            assert xa.dim() == 2


def test_image_shape_must_hold_784_elements() -> None:
    with pytest.raises(ValueError, match="784"):
        make_mnist_dataloaders(batch_size=8, image_shape=(1, 27, 28), **SAMPLES)


def _config(**data_kwargs: object) -> PipelineConfig:
    base = PipelineConfig()
    return PipelineConfig(
        run=base.run,
        data=DataConfig(
            kind="mnist2d",
            in_features=784,
            out_features=10,
            batch_size=32,
            mnist_train_samples=128,
            mnist_validation_samples=64,
            mnist_test_samples=64,
            **data_kwargs,  # type: ignore[arg-type]
        ),
        model=base.model,
        training=base.training,
        optimizer=base.optimizer,
        lr_scheduler=base.lr_scheduler,
        fgd_approx=base.fgd_approx,
        secant_fgd=base.secant_fgd,
        parametric_gd=base.parametric_gd,
        parametric_descent=base.parametric_descent,
        fgd_rkhs=base.fgd_rkhs,
        scaling_line_search=base.scaling_line_search,
        growth_schedule=base.growth_schedule,
        wandb=base.wandb,
    )


def test_build_dataloaders_rejects_an_input_shape_that_does_not_match() -> None:
    config = _config(input_shape=(2, 28, 28))
    with pytest.raises(ValueError, match="input_shape"):
        build_dataloaders(config, torch.device("cpu"))


@needs_mnist
def test_build_dataloaders_defaults_mnist2d_to_one_channel() -> None:
    train, _, _ = build_dataloaders(_config(), torch.device("cpu"))
    x, _ = next(iter(train))
    assert x.shape[1:] == (1, 28, 28)


def test_config_loader_coerces_input_shape_to_a_tuple(tmp_path: Path) -> None:
    from stable_tiny.pipeline import load_pipeline_config

    path = tmp_path / "cfg.yaml"
    path.write_text(
        "data:\n"
        "  kind: mnist2d\n"
        "  in_features: 784\n"
        "  out_features: 10\n"
        "  input_shape: [1, 28, 28]\n"
    )
    config = load_pipeline_config(str(path))
    assert config.data.input_shape == (1, 28, 28)
    assert isinstance(config.data.input_shape, tuple)
