"""Function-preserving layer insertion, so *depth* can enter the search.

Why this exists. Uniform widening has nothing to say about depth -- it is a
width policy, and as the number of hidden layers grows it also adds L times
more parameters per growth event, over-provisioning more the deeper the
network gets. SENN (arXiv:2307.04526) is the only expansion method that
answers depth with the same quantity it answers width with: both are "add
parameters theta_p, measure the increase in the natural expansion score
eta". Table 1 of that paper is exactly this claim.

For this codebase that unification is free, because `fgdlib/senn.py`
establishes ``N*eta = ||r||^2 / (1 + eps^2)``: a candidate layer and a
candidate neuron compete in the *same certified currency*, Lemma 3.5's
relative error, and the cost-normalised rule in `fgdlib/growth.py` already
ranks proposals by ``s_i^2 / cost_i`` without caring what kind of proposal
they are. A layer has a score and a cost like anything else.

The obstacle is that inserting a layer must not change the represented
function, or the certified trajectory is discarded at every depth increase.
SENN's Ingredient 1 handles this for MLPs by replacing a linear transform
``W_i`` with ``(W_i W_q^-1) . (sigma_q = I) . W_q``, which requires the
activation to be *parameterised* so that it can start as the identity; they
use rational activations for this.

This module takes the same idea with the network's existing activation and
a single extra parameter, a homotopy

    sigma_alpha(x) = x + alpha * (sigma(x) - x),

initialised at ``alpha = 0``. At insertion the new layer is an identity
weight followed by an identity activation, so the function is preserved
*exactly* -- not to a tolerance. As ``alpha`` trains toward 1 the layer
becomes a genuine nonlinear one. Compared with the Net2DeeperNet trick this
needs no idempotent activation (``sigma(sigma(x)) = sigma(x)``), which SELU
does not satisfy.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["IdentityHomotopyActivation", "insert_identity_layer"]


class IdentityHomotopyActivation(nn.Module):
    """``x + alpha * (sigma(x) - x)``: the identity at ``alpha = 0``.

    A one-parameter path from the identity to ``sigma``. Starting at the
    identity is what makes layer insertion exactly function-preserving;
    ``alpha`` is an ordinary trainable parameter afterwards, so the new
    layer earns its nonlinearity by descent rather than being handed it.
    """

    def __init__(self, activation: nn.Module, alpha: float = 0.0) -> None:
        super().__init__()
        self.activation = activation
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.alpha * (self.activation(inputs) - inputs)

    def extra_repr(self) -> str:
        return f"alpha={float(self.alpha):.4g}"


def insert_identity_layer(
    model: nn.Module,
    position: int,
    activation: nn.Module | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """Insert an identity layer into ``model.layers`` at ``position``.

    The new module is a square ``LinearGrowingModule`` whose weight is the
    identity and whose bias is zero, followed by an
    :class:`IdentityHomotopyActivation` at ``alpha = 0``. The represented
    function is therefore unchanged exactly.

    ``position`` indexes ``model.layers`` and must not be 0 (there is no
    hidden representation before the first layer to duplicate) nor past the
    output layer.

    Returns the inserted module. The caller is responsible for re-running
    whatever statistics the growth machinery caches, since the layer list
    has changed.
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError("model has no `layers` ModuleList to insert into")
    if not 1 <= position <= len(layers) - 1:
        raise ValueError(
            f"position must be in [1, {len(layers) - 1}], got {position}"
        )

    from gromo.modules.linear_growing_module import LinearGrowingModule

    previous = layers[position - 1]
    width = int(previous.out_features)
    if activation is None:
        activation = getattr(previous, "post_layer_function", nn.SELU())

    inserted = LinearGrowingModule(
        width,
        width,
        post_layer_function=IdentityHomotopyActivation(activation),
        previous_module=previous,
        use_bias=True,
        name=f"Inserted {position}",
        device=device,
    )
    with torch.no_grad():
        inserted.weight.copy_(torch.eye(width, device=inserted.weight.device))
        if inserted.bias is not None:
            inserted.bias.zero_()

    rebuilt = list(layers)
    rebuilt.insert(position, inserted)
    # Re-link the chain: every module must point at its new predecessor, or
    # the growth statistics would be accumulated against a stale graph.
    for index, layer in enumerate(rebuilt):
        if hasattr(layer, "previous_module"):
            layer.previous_module = rebuilt[index - 1] if index else None
    model.layers = nn.ModuleList(rebuilt)

    # Refresh GroMo's own bookkeeping. ``GrowingMLP.__init__`` sets
    # ``_growable_layers = list(self.layers[1:])`` ONCE
    # (``containers/growing_mlp.py``), so without this the container keeps a
    # stale list: the inserted layer would never be selectable for widening
    # and the growable indices would no longer correspond to positions in
    # ``layers``. Rebuilding it the same way the constructor does, and then
    # re-running ``set_growing_layers``, keeps every downstream GroMo call
    # -- ``compute_statistics``, ``compute_optimal_updates``,
    # ``reset_computation`` -- operating on the real graph.
    if hasattr(model, "_growable_layers"):
        model._growable_layers = list(model.layers[1:])
    if hasattr(model, "set_growing_layers"):
        model.set_growing_layers(scheduling_method="all")
    return inserted


def inserted_layer_cost(width: int) -> int:
    """Parameters a depth insertion at ``width`` costs: ``width^2 + width``.

    The counterpart of ``growable_neuron_costs`` for depth proposals, so the
    two can be ranked together by certified decrease per parameter.
    """
    return width * width + width
