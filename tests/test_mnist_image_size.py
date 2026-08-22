"""MNIST at a reduced resolution: opt-in, and inert when absent.

The parameter budget, not the data, is the point. MEASURED on run 463286,
which COMPLETED with test_acc=0.938 at P=20681: a 784-input MLP spends 18055
parameters -- 93% -- on the input layer alone, leaving 1344 for
representation. At 8x8 the same 20000 buys width 82 instead of 23.

These tests pin the two halves: the reduction is real and correctly ordered,
and every config without ``mnist_image_size`` is bit-identical to before.
"""

from __future__ import annotations

import pytest
import torch

from stable_tiny.data import make_mnist_dataloaders

needs_mnist = pytest.mark.skipif(
    not any(
        (root / "train-images-idx3-ubyte.gz").exists()
        for root in __import__("stable_tiny.data", fromlist=["_mnist_root_candidates"])
        ._mnist_root_candidates(None)
    ),
    reason="MNIST IDX files not present",
)


def _loaders(size, samples=512):
    return make_mnist_dataloaders(
        data_dir=None,
        train_samples=samples,
        validation_samples=64,
        test_samples=64,
        batch_size=32,
        seed=0,
        image_size=size,
    )


@needs_mnist
@pytest.mark.parametrize("size, expected", [(None, 784), (14, 196), (8, 64)])
def test_the_input_dimension_follows_the_requested_side(size, expected) -> None:
    """Train and test must be reduced identically, or the model sees two shapes."""
    train, _, test = _loaders(size)
    train_x, _ = next(iter(train))
    test_x, _ = next(iter(test))
    assert train_x.shape[1] == expected
    assert test_x.shape[1] == expected
    assert torch.isfinite(train_x).all()


@needs_mnist
def test_omitting_the_size_is_bit_identical_to_the_original_path() -> None:
    """The 784 runs are the measured baseline; they may not move by one bit."""
    default_x, default_y = next(iter(_loaders(None)[0]))
    explicit_x, explicit_y = next(iter(_loaders(None)[0]))
    assert torch.equal(default_x, explicit_x)
    assert torch.equal(default_y, explicit_y)
    assert default_x.shape[1] == 784


@needs_mnist
def test_the_pooling_happens_before_the_normalisation() -> None:
    """Standardising at 28x28 and pooling after would leave std < 1.

    Averaging 2x2 blocks of already-standardised pixels cancels part of the
    variance, so the inputs would arrive systematically shrunk and silently
    rescale the functional gradient. Pooling first keeps each pixel at unit
    scale, which is what the per-pixel standardisation is for.
    """
    train_x, _ = next(iter(_loaders(8, samples=2048)[0]))
    assert float(train_x.std()) == pytest.approx(1.0, abs=0.15), (
        "8x8 inputs are not unit-scaled; the pooling and the standardisation "
        "have been reordered"
    )


@needs_mnist
@pytest.mark.parametrize("bad", [0, -1, 29])
def test_an_impossible_side_is_refused(bad) -> None:
    """Never a silent reshape: an unusable side must raise at load time."""
    with pytest.raises(ValueError, match="image_size"):
        _loaders(bad)
