"""The preservation check must measure the algebra, not the BLAS backend.

Function-preserving growth adds neurons whose outgoing weights are EXACTLY
zero, so the represented function is unchanged and the drift the check reads
is pure rounding. Widening a layer changes the GEMM's shape, the backend picks
a different tile/split-K reduction, and float addition is not associative --
so the same mathematical sum returns different low bits.

MEASURED on the real path (784 inputs, 3-5 hidden layers, widths 16-128):

    drift float32   1.2e-07 .. 3.0e-07      drift float64   0.0 .. 4.4e-16

a ratio of 2.7e8 to 8.1e8 -- the ratio of the two machine epsilons, and
nothing else. On the cluster's A100 the float32 figure reached 1.93e-5, 65x
the CPU one, because cuBLAS splits the reduction far harder than a sequential
CPU sum. That is what killed job 461736 at epoch 4, P=19481:

    RuntimeError: Function-preserving growth exceeded its output tolerance:
    1.932e-05 > 1.634e-06 (relative tolerance 1.000e-06 x output scale 1.634)

with MaxRSS at 2.6 GB of 64 G and the GPU freed -- nothing was exhausted, a
rounding comparison killed the run.

These tests pin the two halves of the fix: the check runs in float64, and the
1e-6 tolerance is UNCHANGED, so it now has nine orders of margin and gets
STRICTER rather than looser.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.search import growth as growth_module
from fgdlib.search.growth import ScalingLineSearchConfig, grow_layer
from fgdlib.tangent import FGDApproxConfig, tiny_optimal_update_kwargs

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP

DEVICE = torch.device("cpu")
#: The tolerance the shipped configs use. Never relaxed by this fix.
TOLERANCE = 1.0e-6


def _kwargs():
    return tiny_optimal_update_kwargs(FGDApproxConfig(), compute_delta=False)


def _problem(hidden: int, layers: int, samples: int = 1024, seed: int = 0):
    torch.manual_seed(seed)
    model = GrowingMLP(
        in_features=784,
        out_features=10,
        hidden_size=hidden,
        number_hidden_layers=layers,
        use_bias=True,
        device=DEVICE,
    )
    x = torch.randn(samples, 784)
    y = torch.zeros(samples, 10)
    y[torch.arange(samples), torch.randint(0, 10, (samples,))] = 1.0
    return model, DataLoader(TensorDataset(x, y), batch_size=64)


def _grow(model, loader, tolerance=TOLERANCE):
    return grow_layer(
        model=model,
        train_loader=loader,
        layer_index=0,
        device=DEVICE,
        line_search_config=ScalingLineSearchConfig(),
        optimal_update_kwargs=_kwargs(),
        function_preserving=True,
        preservation_tolerance=tolerance,
    )


@pytest.mark.parametrize(
    "hidden, layers", [(16, 3), (23, 3), (64, 3), (128, 3), (64, 5)]
)
def test_an_arithmetically_correct_growth_passes_the_unrelaxed_tolerance(
    hidden, layers
) -> None:
    """The regression itself: these widths must not raise at 1e-6.

    ``hidden=23`` is the cluster's own width at the crash -- P=19481 for
    784-h-h-h-10 solves to h~23.
    """
    model, loader = _problem(hidden, layers)
    before = sum(parameter.numel() for parameter in model.parameters())
    _grow(model, loader)
    after = sum(parameter.numel() for parameter in model.parameters())
    assert after > before, "the layer must actually have grown"


def test_the_drift_is_measured_in_float64() -> None:
    """Pinned directly, because the whole fix is this one dtype.

    Asserting on the drift value alone would pass for the wrong reason on a
    CPU, where float32 noise happens to sit under the bar anyway. The A100
    that killed the run is not available to the suite, so the mechanism is
    what gets pinned.
    """
    model, loader = _problem(64, 3)
    seen: list[torch.dtype] = []
    original = growth_module._preservation_forward

    def record(module, batch_x):
        result = original(module, batch_x)
        seen.append(result.dtype)
        return result

    growth_module._preservation_forward = record
    try:
        _grow(model, loader)
    finally:
        growth_module._preservation_forward = original

    assert seen, "the preservation check must forward at least once"
    assert set(seen) == {torch.float64}, (
        f"the drift check forwarded in {set(seen)}; in float32 it measures the "
        "BLAS reduction order rather than the growth algebra"
    )


def test_the_float64_drift_has_orders_of_margin() -> None:
    """The margin is the claim: 1e-6 must be far above the noise, not near it."""
    model, loader = _problem(64, 3)
    drifts: list[float] = []
    original = growth_module._preservation_forward
    reference: dict[int, torch.Tensor] = {}

    def record(module, batch_x):
        result = original(module, batch_x)
        key = int(batch_x.data_ptr())
        if key in reference:
            drifts.append(float((result - reference[key]).abs().max()))
        else:
            reference[key] = result.detach().clone()
        return result

    growth_module._preservation_forward = record
    try:
        _grow(model, loader)
    finally:
        growth_module._preservation_forward = original

    assert drifts, "no post-growth comparison was observed"
    worst = max(drifts)
    assert worst < TOLERANCE / 1.0e6, (
        f"float64 drift {worst:.3e} left less than six orders of margin under "
        f"{TOLERANCE:.0e}; the check would still be reading arithmetic noise"
    )


def test_a_genuinely_non_preserving_growth_still_raises() -> None:
    """The guard must get STRICTER, not looser.

    A 1e-5 displacement is what the A100's float32 noise was indistinguishable
    from. In float64 it is nine orders outside the noise and must be refused.
    """
    model, loader = _problem(64, 3)
    original = growth_module._preservation_forward
    calls = {"count": 0}
    # The reference loop runs one call per batch before growth; only the
    # post-growth pass is perturbed, since perturbing both would cancel.
    reference_calls = len(loader)

    def perturb(module, batch_x):
        calls["count"] += 1
        result = original(module, batch_x)
        if calls["count"] > reference_calls:
            return result + 1.0e-5
        return result

    growth_module._preservation_forward = perturb
    try:
        with pytest.raises(RuntimeError, match="exceeded its output tolerance"):
            _grow(model, loader)
    finally:
        growth_module._preservation_forward = original


def test_the_model_is_untouched_by_the_float64_check() -> None:
    """``functional_call``, not ``model.double()``: no parameter may move.

    Casting in place would mutate the live parameters mid-growth, and a
    deepcopy of a GroMo model holding update state is what has crashed this
    pipeline before with "Cannot access storage of TensorWrapper".
    """
    model, loader = _problem(64, 3)
    _grow(model, loader)

    assert all(
        parameter.dtype == torch.float32 for parameter in model.parameters()
    ), "the live model must still be float32 after the check"
    assert all(
        buffer.dtype.is_floating_point is False or buffer.dtype == torch.float32
        for buffer in model.buffers()
    )
    model.eval()
    batch_x = next(iter(loader))[0]
    with torch.no_grad():
        output = model(batch_x)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
