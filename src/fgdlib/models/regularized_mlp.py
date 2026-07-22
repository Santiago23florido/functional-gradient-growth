"""Regularization inside the growing MLP, built from GroMo modules.

This is a thin layer of *this* library on top of GroMo -- it adds no growth
mechanism of its own; it composes GroMo's growth-aware regularizers into the
``post_layer_function`` slot of a ``LinearGrowingModule`` so the certified
growth loop keeps working unchanged.

Why it is certification-safe (verified, ``tangent.py`` runs ``model.eval()``
before every certificate):

* **Dropout** is the identity in eval, so it never enters a certificate and
  the functional gradient ``r = grad_f L`` and the tangent projection are
  untouched. It only regularizes the family training steps, which is exactly
  where the measured 46%/30% train/test gap on CIFAR needs help. GroMo's
  ``GrowingDropout`` is growth-transparent: it masks the main pre-activation
  and passes the extension through unchanged, so it also does not perturb the
  TINY statistics that select where to grow.

Normalization (batch-norm) is realised as a plain ``nn.Sequential`` on
purpose: GroMo already threads a growth candidate's extension through a
Sequential (``_apply_extended_post_layer_function``) and reads its activation
gradient correctly (``activation_gradient`` skips ``_BatchNorm`` and uses the
activation's known derivative). A CUSTOM module gets neither -- GroMo falls
back to a numerical ``torch.func.grad`` of the whole thing on a 0-D scalar,
which batch-norm cannot process. See ``make_hidden_post_function``.
"""

from __future__ import annotations

import torch
from torch import nn

from fgdlib.gromo_setup import ensure_gromo_importable

ensure_gromo_importable()

from gromo.modules.growing_dropout import GrowingDropout

from gromo.modules.growing_normalisation import GrowingBatchNorm1d

__all__ = [
    "make_hidden_post_function",
    "make_post_layer_function",
    "sync_normalization",
]


def make_hidden_post_function(
    num_features: int,
    activation: nn.Module,
    dropout_rate: float,
    device: torch.device | None = None,
) -> nn.Sequential:
    """``Sequential(BatchNorm1d, activation[, Dropout])`` for a hidden layer.

    A Sequential rather than a custom module so GroMo handles both the
    extended-forward threading and the activation-gradient inspection (a custom
    module forces a numerical fallback that batch-norm cannot survive).
    Dropout is appended only when the rate is positive.
    """
    modules: list[nn.Module] = [
        GrowingBatchNorm1d(num_features, device=device),
        activation,
    ]
    if dropout_rate > 0.0:
        modules.append(GrowingDropout(dropout_rate=dropout_rate))
    return nn.Sequential(*modules)


def _hidden_norm(post_function: nn.Module | None) -> GrowingBatchNorm1d | None:
    """The growable batch-norm inside a hidden post-function, if any.

    Handles both a bare batch-norm and the ``Sequential`` post-function
    :func:`make_hidden_post_function` builds.
    """
    if isinstance(post_function, GrowingBatchNorm1d):
        return post_function
    if isinstance(post_function, nn.Sequential):
        for module in post_function:
            if isinstance(module, GrowingBatchNorm1d):
                return module
    return None


def sync_normalization(model: nn.Module) -> None:
    """Grow each hidden batch-norm to match the width it normalises.

    A no-op on the plain MLP (no batch-norm present), so it is safe to call
    unconditionally from the growth path. Where a hidden layer has widened,
    its paired batch-norm is grown by the deficit with GroMo's identity
    defaults (weight 1, bias 0, running mean 0, running var 1), which keeps
    the structural step function-preserving.
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        return
    for layer in layers:
        norm = _hidden_norm(getattr(layer, "post_layer_function", None))
        if norm is None:
            continue
        width = int(layer.out_features)
        current = int(norm.num_features)
        if width > current:
            norm.grow(width - current)


def make_post_layer_function(
    activation: nn.Module, dropout_rate: float
) -> nn.Module:
    """Return the hidden-layer post-function: activation, plus dropout if asked.

    With ``dropout_rate == 0`` this is the activation itself, so a model built
    with regularization off is byte-identical to the plain MLP -- the property
    that keeps the MNIST result untouched. With dropout it is a Sequential,
    for the same reason :func:`make_hidden_post_function` is: GroMo reads a
    Sequential's activation gradient correctly (the activation's known
    derivative), where a custom module forces a numerical fallback.
    """
    if dropout_rate <= 0.0:
        return activation
    return nn.Sequential(activation, GrowingDropout(dropout_rate=dropout_rate))
