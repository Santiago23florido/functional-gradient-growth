"""Declarative convolutional stacks, on the same certified growth machinery.

The MLP stack in :mod:`fgdlib.models.stack` folds every auxiliary component --
batch-norm, dropout -- into the preceding layer's ``post_layer_function``, so
``model.layers`` alone describes execution. A conv stack cannot do that:
pooling and flatten CHANGE THE TENSOR SHAPE, so they have to be real steps in
the chain. Hence a separate container with an explicit execution plan::

    model:
      stack:
        - {conv: [2, 3]}     # out_channels, kernel   (padding = k//2, stride 1)
        - {conv: [2, 3]}
        - maxpool            # 2x2, stride 2
        - {conv: [2, 3]}
        - {conv: [2, 3]}
        - maxpool
        - {avgpool: [3, 3]}  # AdaptiveAvgPool2d -- fixes the flatten width
        - flatten
        - {mlp: [2, 1]}      # the output layer is appended automatically

``batchnorm`` / ``dropout`` still attach to the module above and still live
INSIDE ``post_layer_function``, as ``Sequential(norm, activation, dropout)``.
Pooling and flatten must NOT: besides the shape argument, GroMo reads the
activation derivative by INSPECTING that Sequential
(``growing_module.py:1080``), and anything it does not recognise sends it to
``torch.func.grad(module)(0-D scalar)`` with a warning. For a ``MaxPool2d``
that is either a crash or a garbage number -- and the ``where`` rule this
repo measured best is ``activation_gradient * sum(s_i**2)``, so a garbage
factor would silently corrupt every growth decision. ``tests/
test_conv_stack.py`` pins the derivative under ``warnings.simplefilter
("error")`` so the arrangement cannot rot.

Which layers are growable is not a policy here, it is GroMo's topology:

* ``LinearGrowingModule`` raises ``NotImplementedError`` when its
  ``previous_module`` is a conv (``linear_growing_module.py:585,632,1037``),
  so the first FC after the flatten can never be grown;
* ``RestrictedConv2dGrowingModule.bordered_unfolded_extended_prev_input``
  (``:1500``) combines the predecessor's unfolded input with this layer's
  border effect on a SHARED spatial grid, so any pooling in between
  invalidates it.

A layer is therefore growable iff its immediate predecessor is a module of
the same kind with no shape change in between -- exactly what ``VGG`` encodes
by hand (``vgg.py:57-62``, ``:302-306``). Far from being a limitation to work
around, it is what gives the width/depth balance criterion something real to
arbitrate: a pinned width can only be relieved by inserting a layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.models.regularized_mlp import (
    make_conv_post_function,
    make_hidden_post_function,
    make_post_layer_function,
)

ensure_gromo_importable()

from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.modules.conv2d_growing_module import (
    Conv2dGrowingModule,
    FullConv2dGrowingModule,
    RestrictedConv2dGrowingModule,
)
from gromo.modules.linear_growing_module import LinearGrowingModule

__all__ = [
    "CONV_GROWTH_SCHEMES",
    "ConvSpec",
    "ConvStackModel",
    "MlpSpec",
    "OpSpec",
    "build_conv_stack_model",
    "freeze_growth_scratch",
    "is_conv_stack",
    "parse_conv_stack",
]

CONV_GROWTH_SCHEMES: dict[str, type[Conv2dGrowingModule]] = {
    "restricted": RestrictedConv2dGrowingModule,
    "full": FullConv2dGrowingModule,
}

_CONV_TOKENS = frozenset({"conv", "maxpool", "avgpool", "flatten"})


@dataclass(frozen=True)
class ConvSpec:
    """One growable convolution and the regularizers on its post-function."""

    out_channels: int
    kernel_size: int = 3
    batchnorm: bool = False
    dropout_rate: float = 0.0


@dataclass(frozen=True)
class MlpSpec:
    """One growable linear layer and the regularizers on its post-function."""

    width: int
    batchnorm: bool = False
    dropout_rate: float = 0.0


@dataclass(frozen=True)
class OpSpec:
    """A shape-changing step: pooling or flatten. Never growable."""

    kind: str
    value: tuple[int, ...] = ()


def is_conv_stack(stack: list[Any] | None) -> bool:
    """True when the stack names any convolutional component.

    Used by :func:`fgdlib.models.stack.build_stack_model` to dispatch. The
    five canonical configs contain no such token, so the MLP path stays
    provably unreachable from them.
    """
    if not stack:
        return False
    from fgdlib.models.stack import _component

    for item in stack:
        try:
            key, _ = _component(item)
        except ValueError:
            continue
        if key in _CONV_TOKENS:
            return True
    return False


def _as_int_tuple(value: Any, length: int, default: int | None = None) -> tuple[int, ...]:
    if value is None:
        if default is None:
            raise ValueError("a value is required")
        return (default,) * length
    if isinstance(value, int):
        return (int(value),) * length
    if isinstance(value, (list, tuple)):
        items = [int(item) for item in value]
        if len(items) == 1:
            return (items[0],) * length
        if len(items) != length:
            raise ValueError(f"expected {length} values, got {value!r}")
        return tuple(items)
    raise ValueError(f"cannot read {length} ints from {value!r}")


def parse_conv_stack(stack: list[Any]) -> list[ConvSpec | MlpSpec | OpSpec]:
    """Flatten a conv stack into an ordered list of specs.

    ``batchnorm`` / ``dropout`` attach to the last GROWING spec emitted (a
    conv or an mlp), never to a pooling step.
    """
    from fgdlib.models.stack import _as_int_pair, _component

    specs: list[ConvSpec | MlpSpec | OpSpec] = []

    def last_growing() -> int:
        for index in range(len(specs) - 1, -1, -1):
            if not isinstance(specs[index], OpSpec):
                return index
        raise ValueError("a regularizer must follow a conv or an mlp")

    for item in stack:
        key, value = _component(item)
        if key == "conv":
            if isinstance(value, int):
                out_channels, kernel = value, 3
            else:
                pair = _as_int_tuple(value, 2) if value is not None else (0, 0)
                out_channels, kernel = pair
            if out_channels < 1:
                raise ValueError(f"conv needs out_channels>=1, got {item!r}")
            if kernel < 1 or kernel % 2 == 0:
                raise ValueError(
                    f"conv needs an odd kernel>=1 so padding=k//2 preserves the "
                    f"spatial size, got {item!r}"
                )
            specs.append(ConvSpec(out_channels=out_channels, kernel_size=kernel))
        elif key == "mlp":
            width, num_layers = _as_int_pair(value)
            if width < 1 or num_layers < 1:
                raise ValueError(f"mlp needs width>=1 and layers>=1, got {item!r}")
            specs.extend(MlpSpec(width=width) for _ in range(num_layers))
        elif key == "maxpool":
            specs.append(OpSpec("maxpool", _as_int_tuple(value, 2, default=2)))
        elif key == "avgpool":
            specs.append(OpSpec("avgpool", _as_int_tuple(value, 2, default=1)))
        elif key == "flatten":
            specs.append(OpSpec("flatten"))
        elif key in ("batchnorm", "dropout"):
            index = last_growing()
            spec = specs[index]
            if key == "batchnorm":
                specs[index] = type(spec)(
                    **{**vars(spec), "batchnorm": True}
                )
            else:
                rate = float(value if value is not None else 0.0)
                specs[index] = type(spec)(
                    **{**vars(spec), "dropout_rate": rate}
                )
        else:
            raise ValueError(f"unknown conv stack component {key!r}")

    if not any(isinstance(spec, ConvSpec) for spec in specs):
        raise ValueError("model.stack has no conv layers; use the MLP stack")
    if not any(isinstance(spec, OpSpec) and spec.kind == "flatten" for spec in specs):
        raise ValueError("a conv stack must flatten before its linear head")
    return specs


def freeze_growth_scratch(model: nn.Module) -> None:
    """Mark GroMo's lazily-allocated conv growth scratch as non-trainable.

    ``RestrictedConv2dGrowingModule`` caches a ``bordering_convolution`` the
    first time it needs the bordered unfolded input
    (``conv2d_growing_module.py:1508``). It holds a CONSTANT depthwise delta
    kernel -- scratch for simulating the border effect, never trained -- but
    ``create_bordering_effect_convolution`` builds a plain ``nn.Conv2d``, so
    it arrives as a trainable parameter.

    Everything that matters in this repo keys off ``requires_grad``:
    ``count_parameters`` (which prices growth candidates and enforces
    ``max_total_parameters``), ``_trainable_named_parameters`` and the
    matrix-free path's parameter tuple -- the Jacobian's COLUMNS. Left
    trainable it reports one added conv channel as 118 parameters instead of
    28, spends the budget on scratch, and puts scratch directions in the
    tangent space the certificate is computed over.

    Called from :meth:`ConvStackModel.update_computation`, which is where the
    allocation happens.
    """
    for module in model.modules():
        scratch = getattr(module, "bordering_convolution", None)
        if scratch is not None:
            for parameter in scratch.parameters():
                parameter.requires_grad_(False)


class ConvStackModel(SequentialGrowingModel):
    """A conv/pool/flatten/fc chain that the certified growth flow can drive.

    Exposes the two members the flow reads off a ``GrowingMLP`` -- ``layers``
    (the growing modules, in execution order) and ``flatten`` -- plus an
    explicit ``_plan`` interleaving them with the shape-changing ops.
    """

    def __init__(
        self,
        layers: list[nn.Module],
        ops: list[nn.Module],
        plan: list[tuple[str, int]],
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            in_features=in_features, out_features=out_features, device=device
        )
        self.layers = nn.ModuleList(layers)
        self.ops = nn.ModuleList(ops)
        self._plan = list(plan)
        # ``tangent.py`` reads ``model.flatten`` when it inspects structure.
        # Point it at the real one so it never sees a missing attribute.
        self.flatten = next(
            (op for op in ops if isinstance(op, nn.Flatten)), nn.Identity()
        )
        self.rebuild_links()

    # -- structure ------------------------------------------------------
    def rebuild_links(self) -> None:
        """Re-link ``previous_module`` and recompute which layers may grow.

        The single place that owns both, so an insertion cannot leave the two
        disagreeing. ``previous_module`` is ``None`` whenever an op intervenes:
        GroMo's growth statistics assume the predecessor's output IS this
        layer's input on the same grid, which a pool breaks.
        """
        previous: nn.Module | None = None
        separated = True
        links: list[nn.Module | None] = [None] * len(self.layers)
        for kind, payload in self._plan:
            if kind == "op":
                separated = True
                continue
            links[payload] = None if separated else previous
            previous = self.layers[payload]
            separated = False

        growable: list[nn.Module] = []
        for layer, link in zip(self.layers, links):
            layer.previous_module = link
            # Reset GroMo's cached activation derivative: it is memoised
            # against the PREVIOUS module's post-function, which just moved.
            layer._activation_gradient_previous_module = None
            if link is None:
                continue
            same_kind = isinstance(link, Conv2dGrowingModule) == isinstance(
                layer, Conv2dGrowingModule
            )
            if same_kind:
                growable.append(layer)
        self._growable_layers = growable
        self.set_growing_layers(scheduling_method="all")

    # -- execution ------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for kind, payload in self._plan:
            module = self.layers[payload] if kind == "layer" else self.ops[payload]
            x = module(x)
        return x

    def extended_forward(
        self, x: torch.Tensor, mask: dict | None = None  # noqa: ARG002
    ) -> torch.Tensor:
        """Forward with the growth extension branch threaded alongside.

        The ops are applied to BOTH branches, which is what makes pooling and
        flatten compatible with an extension (GroMo's ``VGG.extended_forward``,
        ``vgg.py:335``, does exactly this).
        """
        x_ext = None
        for kind, payload in self._plan:
            if kind == "layer":
                x, x_ext = self.layers[payload].extended_forward(x, x_ext)
            else:
                op = self.ops[payload]
                x = op(x)
                if x_ext is not None:
                    x_ext = op(x_ext)
        return x

    def update_computation(self) -> None:
        super().update_computation()
        # GroMo allocates the bordering convolution during this pass.
        freeze_growth_scratch(self)


def _spatial_after(op: OpSpec, height: int, width: int) -> tuple[int, int]:
    if op.kind == "maxpool":
        return height // op.value[0], width // op.value[1]
    if op.kind == "avgpool":
        return op.value[0], op.value[1]
    return height, width


def build_conv_stack_model(
    stack: list[Any],
    in_features: int,
    out_features: int,
    device: torch.device,
    activation_factory=nn.SELU,
    input_shape: tuple[int, ...] | None = None,
    conv_growth_scheme: str = "restricted",
) -> ConvStackModel:
    """Build the container described by ``stack``.

    ``in_features`` is the flat example size (784 for MNIST) and
    ``input_shape`` its image shape; the flatten width is derived from the
    spatial arithmetic rather than configured, so a pooling change cannot
    silently disagree with the head.
    """
    if conv_growth_scheme not in CONV_GROWTH_SCHEMES:
        raise ValueError(
            f"unknown conv_growth_scheme {conv_growth_scheme!r}; "
            f"expected one of {sorted(CONV_GROWTH_SCHEMES)}"
        )
    conv_type = CONV_GROWTH_SCHEMES[conv_growth_scheme]

    shape = tuple(int(extent) for extent in (input_shape or ()))
    if len(shape) != 3:
        raise ValueError(
            "a conv stack needs data.input_shape as (channels, height, width), "
            f"got {input_shape!r}"
        )
    channels, height, width = shape
    if channels * height * width != in_features:
        raise ValueError(
            f"data.input_shape {shape} has {channels * height * width} elements "
            f"but data.in_features is {in_features}."
        )

    specs = parse_conv_stack(stack)
    layers: list[nn.Module] = []
    ops: list[nn.Module] = []
    plan: list[tuple[str, int]] = []
    flattened: int | None = None

    for spec in specs:
        if isinstance(spec, OpSpec):
            if spec.kind == "maxpool":
                ops.append(nn.MaxPool2d(spec.value[0], spec.value[1]))
            elif spec.kind == "avgpool":
                ops.append(nn.AdaptiveAvgPool2d((spec.value[0], spec.value[1])))
            else:
                ops.append(nn.Flatten(start_dim=1, end_dim=-1))
                flattened = channels * height * width
            height, width = _spatial_after(spec, height, width)
            plan.append(("op", len(ops) - 1))
            continue

        index = len(layers)
        if isinstance(spec, ConvSpec):
            if flattened is not None:
                raise ValueError("a conv cannot follow the flatten")
            post = (
                make_conv_post_function(
                    num_features=spec.out_channels,
                    activation=activation_factory(),
                    dropout_rate=spec.dropout_rate,
                    device=device,
                )
                if spec.batchnorm
                else make_post_layer_function(
                    activation_factory(), spec.dropout_rate, spatial=True
                )
            )
            layers.append(
                conv_type(
                    in_channels=channels,
                    out_channels=spec.out_channels,
                    kernel_size=spec.kernel_size,
                    padding=spec.kernel_size // 2,
                    stride=1,
                    use_bias=True,
                    post_layer_function=post,
                    previous_module=None,  # rebuild_links owns this
                    name=f"Layer {index}",
                    device=device,
                )
            )
            channels = spec.out_channels
        else:
            if flattened is None:
                raise ValueError("an mlp head must follow the flatten")
            post = (
                make_hidden_post_function(
                    num_features=spec.width,
                    activation=activation_factory(),
                    dropout_rate=spec.dropout_rate,
                    device=device,
                )
                if spec.batchnorm
                else make_post_layer_function(
                    activation_factory(), spec.dropout_rate
                )
            )
            layers.append(
                LinearGrowingModule(
                    flattened,
                    spec.width,
                    post_layer_function=post,
                    previous_module=None,
                    use_bias=True,
                    name=f"Layer {index}",
                    device=device,
                )
            )
            flattened = spec.width
        plan.append(("layer", index))

    if flattened is None:
        raise ValueError("a conv stack must flatten before its linear head")

    # The output layer is appended rather than configured, so a stack can
    # never disagree with data.out_features.
    layers.append(
        LinearGrowingModule(
            flattened,
            out_features,
            post_layer_function=nn.Identity(),
            previous_module=None,
            use_bias=True,
            name=f"Layer {len(layers)}",
            device=device,
        )
    )
    plan.append(("layer", len(layers) - 1))

    return ConvStackModel(
        layers=layers,
        ops=ops,
        plan=plan,
        in_features=shape[0],
        out_features=out_features,
        device=device,
    ).to(device)
