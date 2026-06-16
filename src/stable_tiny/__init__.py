"""Stable-TINY growth experiments.

Minimal harness around the *local* `gromo` library to observe the loss/accuracy
dynamics of a self-growing MLP (`GrowingMLP`) on a small classification task.

The single goal of this package (for now) is *observation*: train a growing MLP,
grow it a few times, and record fine-grained train/test loss and accuracy so we
can check whether a loss **spike** appears right after each growth event.

The stable-growth theory (S-orthogonal, see the PDF guide) is intentionally
**not** implemented yet.
"""
