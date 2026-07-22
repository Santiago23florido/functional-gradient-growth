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

__all__ = ["ActivationThenDropout", "make_post_layer_function"]


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
