"""The analytic MLP Jacobian block must BE the jacrev Jacobian.

``_analytic_jacobian_block`` replaces ``jacrev`` + ``_flatten_jacobian`` +
``.to(float64)`` with a closed form. Nothing downstream re-derives the
Jacobian, so a layout slip -- a transposed weight block, a bias column in
the wrong place, rows ordered output-major instead of sample-major -- would
not raise anywhere; it would silently certify a different linear model.
These tests therefore pin the block against jacrev ELEMENTWISE, and pin the
column offsets and row order against the two orderings the rest of the
module assumes (``_trainable_named_parameters`` order, row = n*K + k).

Tolerance: the model runs in float32, so both paths carry float32 error and
agree only to float32 precision. MEASURED over 300 configurations
(depth x width x outputs x bias x seed) the worst relative Frobenius
disagreement is 1.13e-7 -- one float32 eps -- and ``rtol=1e-5, atol=1e-7``
never fails.

That band is the MODEL's dtype, not either path's algebra: with the model in
float64 the two agree to 1e-14 or better
(``test_float64_model_makes_the_analytic_block_exact``). Neither float32 path
is the exact Jacobian, and neither is systematically the better of the two --
MEASURED against the identical model upcast to float64 at production scale
they straddle it (see
``test_analytic_block_is_not_less_accurate_than_jacrev`` for the numbers).
"""

from __future__ import annotations

import copy
from collections import OrderedDict

import pytest
import torch
from torch.func import functional_call, jacrev
from torch.utils.data import DataLoader, TensorDataset

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.search.growth import ScalingLineSearchConfig, grow_layer
from fgdlib.tangent import (
    FGDApproxConfig,
    _analytic_jacobian_block,
    _flatten_jacobian,
    _supported_analytic_structure,
    _trainable_named_parameters,
    tiny_optimal_update_kwargs,
)

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP

#: float32-model band, justified in the module docstring.
RTOL = 1e-5
ATOL = 1e-7


def _model(
    *,
    in_features: int = 11,
    out_features: int = 3,
    hidden_size: int = 4,
    number_hidden_layers: int = 2,
    seed: int = 0,
    use_bias: bool = True,
    activation: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> GrowingMLP:
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    kwargs: dict[str, object] = {}
    if activation is not None:
        kwargs["activation"] = activation
    model = GrowingMLP(
        in_features=in_features,
        out_features=out_features,
        hidden_size=hidden_size,
        number_hidden_layers=number_hidden_layers,
        use_bias=use_bias,
        device=device,
        **kwargs,
    )
    model.eval()
    return model


def _probe(model: GrowingMLP, samples: int, in_features: int, seed: int = 50):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    return torch.randn(samples, in_features, device=device)


def _jacrev_block(model: GrowingMLP, x: torch.Tensor) -> torch.Tensor:
    """The legacy block, built exactly as ``_compute_exact_tangent_projection_step``
    builds it: jacrev over the trainable parameters, flattened, cast to float64."""
    named = _trainable_named_parameters(model)
    names = tuple(named)
    parameters = tuple(named.values())
    buffers = OrderedDict(model.named_buffers())

    def call_batch(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        state = OrderedDict(zip(names, values))
        state.update(buffers)
        return functional_call(model, state, (x,)).reshape(-1)

    with torch.no_grad():
        rows = model(x).numel()
    return _flatten_jacobian(jacrev(call_batch)(parameters), rows).to(torch.float64)


def _analytic_block(model: GrowingMLP, x: torch.Tensor) -> torch.Tensor:
    structure = _supported_analytic_structure(model, x)
    assert structure is not None, "the analytic structure must be supported here"
    return _analytic_jacobian_block(structure, x)


def _assert_parity(model: GrowingMLP, x: torch.Tensor) -> None:
    analytic = _analytic_block(model, x)
    legacy = _jacrev_block(model, x)
    assert analytic.shape == legacy.shape
    assert analytic.dtype == torch.float64
    assert torch.allclose(analytic, legacy, rtol=RTOL, atol=ATOL)
    relative = float((analytic - legacy).norm() / legacy.norm())
    assert relative <= 1e-6, f"relative Frobenius disagreement {relative:.3e}"


def _clear_growth_caching(model: GrowingMLP) -> None:
    """Drop the ``store_input`` counter GroMo's growth leaves incremented.

    ``store_input`` is a REFERENCE COUNT in gromo, and a function-preserving
    growth leaves it at 1 on the grown layer (see
    ``test_tangent_backend_fallback.test_growth_leaves_store_input_set_and_disables_the_fast_path``,
    which pins that leak). Predicate P8 correctly rejects such a model, so
    these tests clear the counter first in order to exercise the analytic
    block on the GROWN shapes -- which is the point of the test -- rather
    than silently testing the fallback again.
    """
    for layer in model.layers:
        while layer.store_input:
            layer.store_input = False
        while layer.store_pre_activity:
            layer.store_pre_activity = False


def _grow(model: GrowingMLP, x: torch.Tensor, y: torch.Tensor, location: int) -> None:
    loader = DataLoader(TensorDataset(x, y), batch_size=max(1, x.shape[0] // 2))
    grow_layer(
        model=model,
        train_loader=loader,
        layer_index=location,
        device=torch.device("cpu"),
        line_search_config=ScalingLineSearchConfig(),
        optimal_update_kwargs=tiny_optimal_update_kwargs(
            FGDApproxConfig(), compute_delta=False
        ),
        function_preserving=True,
        preservation_tolerance=1e-6,
    )
    model.eval()
    _clear_growth_caching(model)


# --------------------------------------------------------------------------
# 8.A -- block parity across the whole supported shape space
# --------------------------------------------------------------------------


@pytest.mark.parametrize("number_hidden_layers", [1, 2, 3])
def test_analytic_block_matches_jacrev_across_depths(number_hidden_layers: int) -> None:
    model = _model(number_hidden_layers=number_hidden_layers)
    _assert_parity(model, _probe(model, 9, 11))


@pytest.mark.parametrize("hidden_size", [1, 2, 3, 7, 16])
def test_analytic_block_matches_jacrev_across_widths(hidden_size: int) -> None:
    model = _model(hidden_size=hidden_size)
    _assert_parity(model, _probe(model, 9, 11))


@pytest.mark.parametrize("out_features", [1, 10])
def test_analytic_block_matches_jacrev_for_single_and_multiple_outputs(
    out_features: int,
) -> None:
    """K = 1 collapses the (N, K, out) sensitivity to a degenerate middle
    axis -- the broadcast that builds the outer product must not silently
    drop it."""
    model = _model(out_features=out_features)
    _assert_parity(model, _probe(model, 9, 11))


@pytest.mark.parametrize("use_bias", [True, False])
def test_analytic_block_matches_jacrev_with_and_without_bias(use_bias: bool) -> None:
    """Without bias the column offsets shift by out_features per layer; the
    block writer must follow ``use_bias``, not assume it."""
    model = _model(use_bias=use_bias)
    x = _probe(model, 9, 11)
    expected = 3 if use_bias else 0
    biases = sum(1 for name in _trainable_named_parameters(model) if "bias" in name)
    assert biases == expected
    _assert_parity(model, x)


@pytest.mark.parametrize("model_seed", [0, 1, 2, 3, 17])
def test_analytic_block_matches_jacrev_across_seeds(model_seed: int) -> None:
    model = _model(seed=model_seed)
    _assert_parity(model, _probe(model, 9, 11, seed=model_seed + 50))


@pytest.mark.parametrize("samples", [1, 2, 13])
def test_analytic_block_matches_jacrev_across_block_sizes(samples: int) -> None:
    """A single-sample chunk is the degenerate case the streaming loop hits
    when certify_stream_chunk does not divide N."""
    model = _model()
    _assert_parity(model, _probe(model, samples, 11))


def test_analytic_block_matches_jacrev_on_cpu() -> None:
    model = _model(device=torch.device("cpu"))
    _assert_parity(model, _probe(model, 9, 11))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_analytic_block_matches_jacrev_on_cuda() -> None:
    model = _model(device=torch.device("cuda"))
    x = _probe(model, 9, 11)
    assert x.is_cuda
    analytic = _analytic_block(model, x)
    assert analytic.is_cuda
    _assert_parity(model, x)


# --------------------------------------------------------------------------
# 8.A -- parity survives growth (the state the CIFAR run spends its life in)
# --------------------------------------------------------------------------


def test_analytic_block_matches_jacrev_after_one_function_preserving_growth() -> None:
    model = _model(in_features=6, out_features=3, hidden_size=2, seed=5)
    x = _probe(model, 12, 6)
    y = torch.randn(12, 3)
    widths_before = [layer.layer.out_features for layer in model.layers]
    _grow(model, x, y, 0)
    assert [layer.layer.out_features for layer in model.layers] != widths_before
    _assert_parity(model, x)


def test_analytic_block_matches_jacrev_after_multiple_growths() -> None:
    model = _model(in_features=6, out_features=3, hidden_size=2, seed=5)
    x = _probe(model, 12, 6)
    y = torch.randn(12, 3)
    locations = len(model._growable_layers)
    for step in range(3):
        _grow(model, x, y, step % locations)
        _assert_parity(model, x)


@pytest.mark.parametrize("location", [0, 1])
def test_analytic_block_matches_jacrev_for_each_growable_location(
    location: int,
) -> None:
    model = _model(in_features=6, out_features=3, hidden_size=2, seed=5)
    assert location < len(model._growable_layers)
    x = _probe(model, 12, 6)
    y = torch.randn(12, 3)
    _grow(model, x, y, location)
    _assert_parity(model, x)


# --------------------------------------------------------------------------
# 8.B -- ordering
# --------------------------------------------------------------------------


def test_analytic_column_order_matches_trainable_parameter_order() -> None:
    """Differentiate ONE parameter at a time and check the resulting slice
    lands at the offset ``_trainable_named_parameters`` implies.

    This is the check that a whole-block comparison cannot make: if two
    parameters of equal numel were swapped in the layout AND the values
    happened to be close, an elementwise comparison could still be within
    tolerance. Here each parameter's Jacobian is built independently.
    """
    model = _model(hidden_size=4, number_hidden_layers=2)
    x = _probe(model, 7, 11)
    named = _trainable_named_parameters(model)
    names = tuple(named)
    parameters = tuple(named.values())
    buffers = OrderedDict(model.named_buffers())
    with torch.no_grad():
        rows = model(x).numel()
    analytic = _analytic_block(model, x)

    offset = 0
    for index, name in enumerate(names):

        def call_one(value: torch.Tensor, _index: int = index) -> torch.Tensor:
            values = list(parameters)
            values[_index] = value
            state = OrderedDict(zip(names, values))
            state.update(buffers)
            return functional_call(model, state, (x,)).reshape(-1)

        single = jacrev(call_one)(parameters[index]).reshape(rows, -1).to(torch.float64)
        width = parameters[index].numel()
        assert single.shape[1] == width, name
        block = analytic[:, offset : offset + width]
        assert torch.allclose(block, single, rtol=RTOL, atol=ATOL), (
            f"{name} does not sit at columns [{offset}, {offset + width})"
        )
        offset += width
    assert offset == analytic.shape[1]


def test_analytic_weight_columns_are_row_major_over_output_then_input() -> None:
    """``df/dW[o, i]`` must land at column ``o * in_features + i`` within the
    layer's slice -- a transpose here is invisible to a Frobenius norm on a
    square weight, so it is checked against jacrev's own reshape."""
    model = _model(in_features=5, out_features=2, hidden_size=3, number_hidden_layers=1)
    x = _probe(model, 6, 5)
    analytic = _analytic_block(model, x)
    named = _trainable_named_parameters(model)
    names = tuple(named)
    parameters = tuple(named.values())
    buffers = OrderedDict(model.named_buffers())
    with torch.no_grad():
        rows = model(x).numel()

    def call_first_weight(value: torch.Tensor) -> torch.Tensor:
        values = list(parameters)
        values[0] = value
        state = OrderedDict(zip(names, values))
        state.update(buffers)
        return functional_call(model, state, (x,)).reshape(-1)

    weight = parameters[0]
    out_features, in_features = weight.shape
    per_entry = jacrev(call_first_weight)(weight).to(torch.float64)
    assert per_entry.shape == (rows, out_features, in_features)
    for o in range(out_features):
        for i in range(in_features):
            column = analytic[:, o * in_features + i]
            assert torch.allclose(column, per_entry[:, o, i], rtol=RTOL, atol=ATOL), (
                f"dW[{o}, {i}] is not at column {o * in_features + i}"
            )


def test_analytic_row_order_is_sample_major() -> None:
    """Row index must be ``n * K + k``. Perturbing sample ``n`` may only move
    rows ``n*K .. n*K+K-1``; an output-major layout would smear the change
    across the whole block."""
    model = _model(in_features=5, out_features=3, hidden_size=3)
    x = _probe(model, 6, 5)
    k = model.layers[-1].layer.out_features
    baseline = _analytic_block(model, x)

    for n in (0, 2, 5):
        perturbed = x.clone()
        perturbed[n] = perturbed[n] + 1.0
        moved = _analytic_block(model, perturbed)
        difference = (moved - baseline).abs().sum(dim=1)
        touched = torch.zeros_like(difference, dtype=torch.bool)
        touched[n * k : (n + 1) * k] = True
        assert float(difference[~touched].max()) == 0.0, (
            f"perturbing sample {n} moved rows outside [{n * k}, {(n + 1) * k})"
        )
        assert float(difference[touched].max()) > 0.0


# --------------------------------------------------------------------------
# 8.A -- the activation derivative is PyTorch's, not a hard-coded formula
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "activation",
    [
        torch.nn.Tanh(),
        torch.nn.ReLU(),
        torch.nn.GELU(),
        torch.nn.Sigmoid(),
        torch.nn.ELU(),
        torch.nn.SiLU(),
        torch.nn.Softplus(),
        torch.nn.LeakyReLU(),
        torch.nn.Identity(),
    ],
    ids=lambda phi: type(phi).__name__,
)
def test_activation_derivative_comes_from_autograd(
    activation: torch.nn.Module,
) -> None:
    """Every allow-listed activation must reproduce jacrev without the block
    builder knowing anything about it. ReLU is the sharp case: its
    subgradient convention at exactly 0 is PyTorch's choice, and the analytic
    path inherits it because it asks autograd rather than writing ``z > 0``."""
    model = _model(hidden_size=5, activation=activation)
    _assert_parity(model, _probe(model, 9, 11))


def test_relu_kink_is_hit_and_still_matches_jacrev() -> None:
    """Drive a pre-activation to exactly 0.0 so the subgradient convention is
    actually exercised rather than assumed unreachable."""
    model = _model(
        in_features=4,
        out_features=2,
        hidden_size=3,
        number_hidden_layers=1,
        activation=torch.nn.ReLU(),
    )
    with torch.no_grad():
        first = model.layers[0].layer
        first.weight.zero_()
        first.bias.zero_()
    x = _probe(model, 5, 4)
    with torch.no_grad():
        pre_activation = model.layers[0].layer(x)
    assert torch.all(pre_activation == 0.0)
    _assert_parity(model, x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_analytic_block_matches_jacrev_on_cuda_after_growth() -> None:
    """Growth changes the widths the block writer indexes with; CUDA changes
    the reduction order the products are formed in. Neither is exercised by
    the other test, so the combination gets its own."""
    model = _model(in_features=6, out_features=3, hidden_size=2, seed=5)
    x = _probe(model, 12, 6)
    y = torch.randn(12, 3)
    _grow(model, x, y, 0)
    model = model.to(torch.device("cuda"))
    model.eval()
    _assert_parity(model, x.to(torch.device("cuda")))


def test_analytic_block_is_not_less_accurate_than_jacrev() -> None:
    """``Agrees with jacrev`` and ``is accurate`` are different goals.

    Both paths run the forward and the sensitivity recursion in the MODEL's
    dtype, so on a float32 model NEITHER is the exact Jacobian: jacrev's own
    float32 result sits about 1e-7 relative away from the same model
    evaluated in float64. The analytic block differs from jacrev by the same
    order, and the question that matters is not which of the two they agree
    with but which is closer to the truth.

    MEASURED on the production CIFAR system (N=10 000, K=10, P=2092), against
    the identical model upcast to float64:

        Gram(jacrev f32)    vs float64 model  7.81e-8
        Gram(analytic f32)  vs float64 model  8.44e-8
        b(jacrev f32)       vs float64 model  5.02e-8
        b(analytic f32)     vs float64 model  4.86e-8

    i.e. they straddle: neither is systematically better, and the gap between
    them IS float32 model arithmetic. (Running the recursion itself in
    float64 would make the analytic block exact -- MEASURED 4.41 ms against
    5.20 ms per 10240 x 2092 block, so it would cost nothing -- but it would
    then agree with jacrev LESS well, not more. The certificate wants
    accuracy, not agreement with a particular float32 implementation.)

    This test pins the ordering claim at test scale: the analytic block is
    within a small factor of jacrev's own distance to the float64 truth.
    """
    model = _model(in_features=11, out_features=3, hidden_size=4, seed=4)
    x = _probe(model, 16, 11)

    reference_model = copy.deepcopy(model).to(torch.float64)
    reference_model.eval()
    reference = _analytic_block(reference_model, x.to(torch.float64))

    analytic = _analytic_block(model, x)
    legacy = _jacrev_block(model, x)
    scale = float(reference.norm())
    analytic_error = float((analytic - reference).norm()) / scale
    jacrev_error = float((legacy - reference).norm()) / scale

    assert analytic_error < 1e-6
    assert jacrev_error < 1e-6
    assert analytic_error <= 4.0 * jacrev_error, (
        f"analytic {analytic_error:.3e} vs jacrev {jacrev_error:.3e}"
    )


def test_float64_model_makes_the_analytic_block_exact() -> None:
    """With the model in float64 the two paths must agree to float64
    roundoff, which is what shows the 1e-7 band above is the MODEL's dtype
    and not the block builder's algebra."""
    model = _model(in_features=9, out_features=4, hidden_size=5, seed=6)
    model = model.to(torch.float64)
    model.eval()
    x = _probe(model, 11, 9).to(torch.float64)
    analytic = _analytic_block(model, x)
    legacy = _jacrev_block(model, x)
    relative = float((analytic - legacy).norm() / legacy.norm())
    assert relative <= 1e-14, relative
