"""Declarative model stacks: choose where each mlp / dropout / batchnorm goes.

Instead of a rigid "uniform hidden size with batch-norm everywhere", the
architecture is written component by component in the config, e.g.

    model:
      stack:
        - mlp: [2, 1]      # an MLP block: width 2, 1 layer  (growable)
        - batchnorm        # attaches to the block just above
        - mlp: [2, 1]
        - dropout: 0.2
        - mlp: [2, 1]

An ``mlp`` is a *block* of ``num_layers`` linear layers of the given
``width`` (``mlp: w`` is the one-layer shorthand). ``batchnorm`` and
``dropout`` are regularizers that attach to the **most recent linear layer**,
as its post-function; ordering inside a layer's post-function is always the
modern convention batch-norm -> activation -> dropout, regardless of the token
order. Every mlp layer is growable -- the certified search decides which to
widen, exactly as before; the stack only sets the starting structure and
where the regularizers live.

The model is built on top of GroMo's ``GrowingMLP`` by replacing its
``layers`` with the chain the stack describes (the same technique
``fgdlib.search.depth.insert_identity_layer`` uses), so every downstream
GroMo call -- ``update_information``, ``compute_optimal_updates``, the growth
statistics -- operates on the real graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.models.regularized_mlp import (
    make_hidden_post_function,
    make_post_layer_function,
)

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP
from gromo.modules.linear_growing_module import LinearGrowingModule

__all__ = ["LayerSpec", "parse_stack", "build_stack_model"]


@dataclass(frozen=True)
class LayerSpec:
    """One growable linear layer and the regularizers on its post-function."""

    width: int
    batchnorm: bool = False
    dropout_rate: float = 0.0


def _as_int_pair(value: Any) -> tuple[int, int]:
    """Read ``[width, num_layers]``, ``width``, or ``{width, layers}``."""
    if isinstance(value, int):
        return value, 1
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return int(value[0]), 1
        return int(value[0]), int(value[1])
    if isinstance(value, dict):
        return int(value["width"]), int(value.get("layers", 1))
    raise ValueError(f"cannot read an mlp spec from {value!r}")


def parse_stack(stack: list[Any]) -> list[LayerSpec]:
    """Flatten a stack of components into per-linear-layer specs.

    ``batchnorm`` / ``dropout`` attach to the last linear layer emitted, so
    they must follow an ``mlp``. An mlp block of ``n`` layers emits ``n``
    specs; a regularizer after it lands on the block's last layer, which is
    the layer whose output the regularizer sees.
    """
    specs: list[LayerSpec] = []
    for item in stack:
        key, value = _component(item)
        if key == "mlp":
            width, num_layers = _as_int_pair(value)
            if width < 1 or num_layers < 1:
                raise ValueError(f"mlp needs width>=1 and layers>=1, got {item!r}")
            specs.extend(LayerSpec(width=width) for _ in range(num_layers))
        elif key == "batchnorm":
            if not specs:
                raise ValueError("batchnorm must follow an mlp")
            specs[-1] = LayerSpec(
                specs[-1].width, True, specs[-1].dropout_rate
            )
        elif key == "dropout":
            if not specs:
                raise ValueError("dropout must follow an mlp")
            rate = float(value if value is not None else 0.0)
            specs[-1] = LayerSpec(specs[-1].width, specs[-1].batchnorm, rate)
        else:
            raise ValueError(f"unknown stack component {key!r}")
    if not specs:
        raise ValueError("model.stack has no mlp layers")
    return specs


def _component(item: Any) -> tuple[str, Any]:
    """Normalise a stack item to ``(component_name, value)``."""
    if isinstance(item, str):
        return item.strip().lower(), None
    if isinstance(item, dict):
        if len(item) != 1:
            # Allow {batchnorm: true} / {dropout: 0.2} style single entries.
            for flag in ("batchnorm", "dropout", "mlp"):
                if flag in item:
                    return flag, item[flag]
            raise ValueError(f"a stack item needs exactly one component: {item!r}")
        (name, value), = item.items()
        return str(name).strip().lower(), value
    raise ValueError(f"cannot read a stack component from {item!r}")


def build_stack_model(
    stack: list[Any],
    in_features: int,
    out_features: int,
    device: torch.device,
    activation_factory=nn.SELU,
) -> GrowingMLP:
    """Build a ``GrowingMLP`` whose hidden layers follow ``stack``.

    The mlp blocks must share a common starting width (the certified search
    widens them apart from there, so a uniform start is the natural one, and
    it is what every example uses). This builds through GroMo's own
    ``GrowingMLP`` constructor and then sets each hidden layer's post-function
    from its spec, so it reuses the exact construction path the uniform
    ``use_batchnorm`` shorthand uses -- no manual layer surgery, and no
    interaction with the functorch tangent projection. The output layer is
    never regularized.
    """
    specs = parse_stack(stack)
    widths = {spec.width for spec in specs}
    if len(widths) != 1:
        raise ValueError(
            "model.stack currently requires every mlp block to share the same "
            f"starting width (growth widens them from there); got widths "
            f"{sorted(widths)}. Use one width across the blocks."
        )
    width = specs[0].width

    model = GrowingMLP(
        in_features=in_features,
        out_features=out_features,
        hidden_size=width,
        number_hidden_layers=len(specs),
        device=device,
    )

    # Set each hidden layer's post-function from its spec. GroMo's GrowingMLP
    # puts the hidden layers at indices [0 .. len(specs)-1]; the last layer is
    # the output and is left untouched.
    for spec, layer in zip(specs, list(model.layers)[:-1]):
        if spec.batchnorm:
            layer.post_layer_function = make_hidden_post_function(
                num_features=width,
                activation=activation_factory(),
                dropout_rate=spec.dropout_rate,
                device=device,
            )
        else:
            layer.post_layer_function = make_post_layer_function(
                activation_factory(), spec.dropout_rate
            )
    return model
