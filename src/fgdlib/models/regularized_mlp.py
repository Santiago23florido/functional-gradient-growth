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

from gromo.modules.growing_dropout import (
    GrowingDropout,
    GrowingDropout1d,
    GrowingDropout2d,
)

from gromo.modules.growing_normalisation import GrowingBatchNorm1d, GrowingBatchNorm2d
from gromo.utils.utils import known_activations_zero_plus_gradient

__all__ = [
    "GrowingDropoutFlat",
    "make_conv_post_function",
    "make_hidden_post_function",
    "make_post_layer_function",
    "sync_normalization",
]


class GrowingDropoutFlat(GrowingDropout, nn.Dropout):
    """Element-wise growth-safe dropout for a flat ``(N, features)`` activation.

    GroMo's ``GrowingDropout`` is an ABSTRACT base: it derives from
    ``nn.modules.dropout._DropoutNd``, which supplies no ``forward``, and
    contributes only ``extended_forward`` (pass the extension through
    unchanged, so dropout never zeroes the growth). The concrete classes it
    ships are ``GrowingDropout1d`` and ``GrowingDropout2d``, both CHANNEL
    dropout: ``nn.Dropout1d`` reads a 2-D tensor as ``(C, L)``, not
    ``(N, features)``, so neither is the element-wise dropout a hidden MLP
    layer wants.

    This composes the mixin with ``nn.Dropout`` exactly the way GroMo composes
    its own two, which is the smallest thing that is both concrete and
    correct here.
    """


# Declare dropout's derivative at 0+ instead of letting GroMo estimate it.
#
# ``GrowingModule.activation_gradient`` walks the post-function Sequential,
# looks each entry up in this table, skips ``_BatchNorm``, and for anything
# else warns and computes ``torch.func.grad(module)(eps)`` on a 0-D scalar.
# For dropout that estimate is a COIN FLIP: measured on a conv stack, eval
# gives 1.0507 (dropout is the identity there) but train gives 0.0 whenever
# the single probed element is the one dropped -- and the result is MEMOISED
# in ``_activation_gradient_previous_module``. Since the ``where`` rule is
# ``activation_gradient * sum(s_i**2)``, a cached 0.0 silently removes a layer
# from consideration for the rest of the run.
#
# Dropout is the identity in expectation and exactly the identity in eval,
# which is where every certificate and every statistics pass runs, so 1.0 is
# the honest constant. Registering it also removes the warning, which matters:
# a warning that fires routinely is a warning nobody reads.
known_activations_zero_plus_gradient.setdefault(GrowingDropoutFlat, 1.0)
known_activations_zero_plus_gradient.setdefault(GrowingDropout2d, 1.0)
known_activations_zero_plus_gradient.setdefault(GrowingDropout1d, 1.0)



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
        modules.append(GrowingDropoutFlat(dropout_rate=dropout_rate))
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
    activation: nn.Module, dropout_rate: float, spatial: bool = False
) -> nn.Module:
    """Return the hidden-layer post-function: activation, plus dropout if asked.

    With ``dropout_rate == 0`` this is the activation itself, so a model built
    with regularization off is byte-identical to the plain MLP -- the property
    that keeps the MNIST result untouched. With dropout it is a Sequential,
    for the same reason :func:`make_hidden_post_function` is: GroMo reads a
    Sequential's activation gradient correctly (the activation's known
    derivative), where a custom module forces a numerical fallback.

    ``spatial`` picks channel dropout (``GrowingDropout2d``) for a conv layer,
    where dropping individual pixels is nearly a no-op because neighbours are
    correlated. It changes nothing when ``dropout_rate == 0``.
    """
    if dropout_rate <= 0.0:
        return activation
    dropout_type = GrowingDropout2d if spatial else GrowingDropoutFlat
    return nn.Sequential(activation, dropout_type(dropout_rate=dropout_rate))


def make_conv_post_function(
    num_features: int,
    activation: nn.Module,
    dropout_rate: float,
    device: torch.device | None = None,
) -> nn.Sequential:
    """``Sequential(BatchNorm2d, activation[, Dropout2d])`` for a conv layer.

    The 2-D counterpart of :func:`make_hidden_post_function`, and a Sequential
    for the same reason: GroMo walks a Sequential looking up each entry's
    derivative at 0+ and skipping ``_BatchNorm``, so the arrangement keeps the
    exact SELU constant instead of a numerical estimate.
    """
    modules: list[nn.Module] = [
        GrowingBatchNorm2d(num_features, device=device),
        activation,
    ]
    if dropout_rate > 0.0:
        modules.append(GrowingDropout2d(dropout_rate=dropout_rate))
    return nn.Sequential(*modules)
