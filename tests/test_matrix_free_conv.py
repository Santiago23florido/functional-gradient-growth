"""The matrix-free tangent on a conv model: an inference turned into a fact.

The whole conv effort rests on config 1's ``family_order:
[matrix_free_tangent]``, because the ANALYTIC Jacobian is Linear-only by
construction (``tangent.py``'s P1/P2/P3 predicates) and refuses a conv
container on purpose. The matrix-free path is supposed to be architecture-
agnostic -- ``functional_call`` plus ``jvp``/``vjp`` over ``named_parameters``
-- but "supposed to be" is not a measurement, and these tests make it one.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgdlib import profile
from fgdlib.models.convstack import build_conv_stack_model
from fgdlib.search.matrixfree import vmap_operators
from fgdlib.tangent import (
    _supported_analytic_structure,
    exact_tangent_system,
)
from stable_tiny.pipeline import load_pipeline_config

DEVICE = torch.device("cpu")

BASE_STACK = [
    {"conv": [2, 3]},
    {"conv": [2, 3]},
    "maxpool",
    {"conv": [2, 3]},
    {"conv": [2, 3]},
    "maxpool",
    {"avgpool": [3, 3]},
    "flatten",
    {"mlp": [2, 1]},
]


def model():
    torch.manual_seed(0)
    return build_conv_stack_model(
        BASE_STACK,
        in_features=784,
        out_features=10,
        device=DEVICE,
        input_shape=(1, 28, 28),
    )


def probe(rows: int = 16):
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(rows, 1, 28, 28, generator=generator)
    y = torch.zeros(rows, 10)
    y[torch.arange(rows), torch.randint(0, 10, (rows,), generator=generator)] = 1.0
    return x, y


def fgd_config():
    return load_pipeline_config(
        "configs/fgd/family_ladder_matrix_free_N1024.yaml"
    ).fgd_approx


def test_the_analytic_structure_refuses_a_conv_container() -> None:
    """It SHOULD refuse: the analytic form was never derived for conv.

    Refusing is what routes the flow to ``vmap_operators``, so this is a
    load-bearing negative, not an unsupported case.
    """
    x, _ = probe(4)
    assert _supported_analytic_structure(model(), x) is None


def test_apply_j_agrees_with_an_explicit_jacobian() -> None:
    """The operator IS the Jacobian, to float64 precision."""
    from torch.func import functional_call

    net = model()
    x, _ = probe(4)
    names = [n for n, p in net.named_parameters() if p.requires_grad]
    parameters = [p for p in net.parameters() if p.requires_grad]
    operators = vmap_operators(net, x, parameters, names)

    def flat_forward(packed: torch.Tensor) -> torch.Tensor:
        pieces, offset = {}, 0
        for name, parameter in zip(names, parameters):
            size = parameter.numel()
            pieces[name] = packed[offset : offset + size].reshape(parameter.shape)
            offset += size
        return functional_call(net, pieces, (x.double(),)).reshape(-1)

    flat = torch.cat([p.detach().reshape(-1) for p in parameters]).double()
    reference = torch.autograd.functional.jacobian(flat_forward, flat)

    generator = torch.Generator().manual_seed(3)
    block = torch.randn(5, operators.columns, generator=generator).double()
    assert torch.allclose(
        operators.apply_j(block), block @ reference.T, atol=1e-10, rtol=0.0
    )
    rows_block = torch.randn(5, operators.rows, generator=generator).double()
    assert torch.allclose(
        operators.apply_jt(rows_block), rows_block @ reference, atol=1e-10, rtol=0.0
    )


def test_the_scratch_is_not_a_column_of_the_jacobian() -> None:
    """A trainable ``bordering_convolution`` would add scratch directions."""
    from torch import nn

    from gromo.utils.training_utils import compute_statistics

    net = model()
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(64, 1, 28, 28, generator=generator)
    y = torch.zeros(64, 10)
    y[torch.arange(64), torch.randint(0, 10, (64,), generator=generator)] = 1.0
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=32
    )
    net.set_growing_layers(index=0)
    compute_statistics(
        net, loader, loss_function=nn.MSELoss(reduction="sum"), device=DEVICE
    )
    net.reset_computation()
    for layer in net._growing_layers:
        layer.delete_update(include_previous=True)
    net.currently_updated_layer_index = None

    assert any(
        getattr(module, "bordering_convolution", None) is not None
        for module in net.modules()
    ), "the scratch was not allocated, so this proves nothing"
    names = [n for n, p in net.named_parameters() if p.requires_grad]
    assert not any("bordering_convolution" in name for name in names)


def test_conv_takes_the_range_finder_branch_and_says_so() -> None:
    """Pins the routing, so a regression into ``dual_gram`` is caught.

    ``dual_gram`` needs analytic factors; reaching it with vmap operators is
    the ``AttributeError`` this repo already documented. The two counters are
    how a future change that silently re-routes conv gets noticed.
    """
    import os

    net = model()
    x, y = probe(16)
    previous = os.environ.get("FGD_PROFILE")
    os.environ["FGD_PROFILE"] = "1"
    try:
        profile.reset()
        system = exact_tangent_system(net, x, y, fgd_config())
        counters = profile.snapshot()
    finally:
        if previous is None:
            os.environ.pop("FGD_PROFILE", None)
        else:
            os.environ["FGD_PROFILE"] = previous
        profile.reset()

    assert system is not None
    assert getattr(system, "factors", None) is not None
    assert system.jacobian.numel() == 0
    assert counters.get("matrix_free_vmap_fallbacks", 0) >= 1
    assert counters.get("matrix_free_dual_unavailable", 0) >= 1


def test_the_solve_leaves_no_functorch_wrapper_behind() -> None:
    """The deepcopy that would raise 'Cannot access storage of TensorWrapper'."""
    net = model()
    x, y = probe(16)
    system = exact_tangent_system(net, x, y, fgd_config())
    assert system is not None
    clone = copy.deepcopy(net)
    with torch.no_grad():
        assert torch.equal(clone(x), net(x))


def test_the_spectrum_is_readable_and_the_probe_is_above_the_floor() -> None:
    """``NK`` must stay well above ``rank(J)`` or eps collapses to a spurious 0."""
    net = model()
    rows = 16
    x, y = probe(rows)
    system = exact_tangent_system(net, x, y, fgd_config())
    assert system is not None
    left, right = system.factors
    values = torch.linalg.svdvals(left.to(torch.float32))
    rank = int((values > float(values.max()) * right.shape[0] * 1e-6).sum())
    assert 0 < rank
    # NK = rows * out_features
    assert rows * 10 / rank > 2.0
