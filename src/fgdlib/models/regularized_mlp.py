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

The composition module below implements GroMo's ``extended_post_layer_function``
protocol (``growing_module.py``): a ``post_layer_function`` may process both
the main pre-activation ``x`` and the extension ``x_ext``. The activation is
element-wise so it applies to both; dropout masks only the main path, matching
``GrowingDropout``'s own convention.
"""

from __future__ import annotations

import torch
from torch import nn

from fgdlib.gromo_setup import ensure_gromo_importable

ensure_gromo_importable()

from gromo.modules.growing_dropout import GrowingDropout

from gromo.modules.growing_normalisation import GrowingBatchNorm1d

__all__ = [
    "ActivationThenDropout",
    "HiddenPostFunction",
    "make_post_layer_function",
    "sync_normalization",
]


class ActivationThenDropout(nn.Module):
    """``dropout(activation(x))``, honouring the extended-forward protocol.

    In the plain forward this is just activation followed by dropout. In
    ``extended_forward`` -- used while a growth candidate's extension flows
    through the network -- the activation is applied to both the main and the
    extension pre-activations (it is element-wise), while dropout masks only
    the main path, so a new neuron's contribution is never randomly zeroed
    before it has been measured.
    """

    def __init__(self, activation: nn.Module, dropout_rate: float) -> None:
        super().__init__()
        self.activation = activation
        self.dropout = GrowingDropout(dropout_rate=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(inputs))

    def extended_forward(
        self, x: torch.Tensor | None, x_ext: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        main = None if x is None else self.activation(x)
        extension = None if x_ext is None else self.activation(x_ext)
        # Dropout on the main path only; the extension passes through so the
        # growth measurement sees the new capacity undropped.
        if main is not None:
            main, _ = self.dropout.extended_forward(main, None)
        return main, extension

    def extra_repr(self) -> str:
        return f"dropout_rate={float(self.dropout.p):.3g}"


class HiddenPostFunction(nn.Module):
    """A hidden layer's post-function: ``dropout(activation(batchnorm(x)))``.

    Ordering follows the modern convention Linear -> BatchNorm -> activation
    -> dropout. Every stage honours GroMo's extended-forward protocol, so a
    growth candidate's extension flows through correctly:

    * **BatchNorm** normalises the main path with its running statistics and
      passes the extension through unchanged (GroMo's ``GrowingBatchNorm1d``).
      It is *per-feature*, which is the property that makes it
      function-preservingly growable -- adding a feature with identity
      parameters leaves every existing feature's normalisation untouched.
      LayerNorm is deliberately NOT offered: it normalises ACROSS features, so
      a new feature changes the statistics of all the others and breaks
      function preservation (verified: existing features drift on insertion).
    * **Activation** is element-wise, applied to both paths.
    * **Dropout** masks the main path only, so the extension is measured
      undropped.

    Unlike dropout, batch-norm is *not* eval-transparent: it is part of the
    represented function ``f`` (it normalises with fixed running stats at
    eval). That is fine for certification -- the certificate is computed on
    whatever ``f`` is; the only requirement growth imposes is that the
    structural step be function-preserving, which per-feature batch-norm,
    grown in sync, satisfies.
    """

    def __init__(
        self,
        num_features: int,
        activation: nn.Module,
        dropout_rate: float,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.norm = GrowingBatchNorm1d(num_features, device=device)
        self.activation = activation
        self.dropout: nn.Module = (
            GrowingDropout(dropout_rate=dropout_rate)
            if dropout_rate > 0.0
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.norm(inputs)))

    def extended_forward(
        self, x: torch.Tensor | None, x_ext: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        main, extension = self.norm.extended_forward(x, x_ext)
        main = None if main is None else self.activation(main)
        extension = None if extension is None else self.activation(extension)
        if main is not None and isinstance(self.dropout, GrowingDropout):
            main, _ = self.dropout.extended_forward(main, None)
        return main, extension


def _hidden_norm(post_function: nn.Module) -> GrowingBatchNorm1d | None:
    """The growable batch-norm inside a hidden post-function, if any."""
    norm = getattr(post_function, "norm", None)
    return norm if isinstance(norm, GrowingBatchNorm1d) else None


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
    that keeps the MNIST result untouched.
    """
    if dropout_rate <= 0.0:
        return activation
    return ActivationThenDropout(activation, dropout_rate)
