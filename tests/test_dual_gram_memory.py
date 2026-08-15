"""The dual Gram must be the same matrix, built without five copies of itself.

``dual_gram`` is the single largest allocation in the whole run. MEASURED on an
A100 at P=16750 / NK=20930 (``scripts/probe_dual_memory.py``): the Gram it
returns is 3.55 GB and building it peaked at **17.92 GB**, five times its own
result, while ``eigh`` and the ``keep``/``flip`` copies downstream added
**+0.00** to that peak. The parameter budget is capped at 16000 because of it.

The five came from spelling ``sum_l S_l * expand(A_l) [+ S_l]`` with a fresh
``(NK, NK)`` tensor at every step: the sensitivity Gram, the materialised
expansion, the product, the bias sum and the accumulation. Two of those are
avoidable outright -- the expansion is an ``(n, n)`` matrix broadcast over two
``K`` axes, and ``S * A + S`` factorises to ``S * (A + 1)``, moving the extra
term onto the small matrix -- and the rest fold into in-place ops.

These tests pin the only thing that licenses that rewrite: it is the same
matrix. The bias case is NOT bit-identical, because ``x * a + x`` and
``x * (a + 1)`` associate differently in floating point; it is pinned at the
measured 1e-13, and the certificate that consumes it is pinned as unchanged.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from fgdlib.search.matrixfree import dual_gram


class _Operators:
    """The three fields ``dual_gram`` reads, with nothing else attached."""

    def __init__(self, sensitivities, activations, use_bias, rows) -> None:
        self.sensitivities = sensitivities
        self.activations = activations
        self.use_bias = use_bias
        self.rows = rows


def _previous_implementation(operators) -> torch.Tensor:
    """The spelling that peaked at 5x its own result, kept as the oracle."""
    rows = operators.rows
    outputs = operators.sensitivities[0].shape[1]
    total = None
    for sensitivity, inputs, uses_bias in zip(
        operators.sensitivities, operators.activations, operators.use_bias
    ):
        flat = sensitivity.reshape(rows, -1)
        sensitivity_gram = flat @ flat.t()
        activation_gram = (
            (inputs @ inputs.t())
            .repeat_interleave(outputs, dim=0)
            .repeat_interleave(outputs, dim=1)
        )
        block = sensitivity_gram * activation_gram
        if uses_bias:
            block = block + sensitivity_gram
        total = block if total is None else total + block
    return total


def _operators(samples: int, outputs: int, layers: int, use_bias, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    rows = samples * outputs
    sensitivities = [
        torch.randn(rows, outputs, 6, generator=generator, dtype=torch.float64)
        for _ in range(layers)
    ]
    activations = [
        torch.randn(samples, 5, generator=generator, dtype=torch.float64)
        for _ in range(layers)
    ]
    return _Operators(sensitivities, activations, list(use_bias), rows)


@pytest.mark.parametrize("samples, outputs, layers", [(7, 3, 2), (11, 4, 3), (9, 5, 4)])
def test_the_bias_free_gram_is_bit_identical(samples, outputs, layers) -> None:
    """Without bias nothing is refactorised, so nothing may move at all."""
    operators = _operators(samples, outputs, layers, [False] * layers)
    assert torch.equal(dual_gram(operators), _previous_implementation(operators))


@pytest.mark.parametrize(
    "use_bias",
    [(True, True), (True, False), (False, True)],
    ids=["both", "first", "second"],
)
def test_the_bias_gram_matches_to_the_measured_tolerance(use_bias) -> None:
    """``S * A + S`` against ``S * (A + 1)``: same matrix, different rounding.

    MEASURED at 1.1e-13 absolute on float64 entries of order 10-100, i.e. about
    1e-15 relative -- machine epsilon. Pinned so a real regression cannot hide
    behind "it is just floating point".
    """
    operators = _operators(7, 3, len(use_bias), use_bias)
    produced = dual_gram(operators)
    expected = _previous_implementation(operators)
    assert produced.shape == expected.shape
    assert torch.allclose(produced, expected, rtol=0.0, atol=1e-11)


def test_the_certificate_does_not_move(monkeypatch) -> None:
    """The only deviation that would matter, measured where it is consumed.

    MEASURED on this config at P=3230 and P=4864: eps is identical to twelve
    decimals both ways, so the 1e-13 in the Gram does not survive ``eigh``.
    """
    from fgdlib import tangent as T
    from fgdlib.search import matrixfree
    from fgdlib.search.certify import exact_relative_error
    from stable_tiny.pipeline import build_model, load_pipeline_config

    base = load_pipeline_config("configs/experiments/mnist_full.yaml")
    config = dataclasses.replace(
        base, model=dataclasses.replace(base.model, hidden_size=4)
    )
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = build_model(config, device)
    model.eval()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    samples = int(1.25 * parameters / 10)
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(samples, 784, generator=generator)
    y = torch.zeros(samples, 10)
    y[torch.arange(samples), torch.randint(0, 10, (samples,), generator=generator)] = 1.0

    errors = {}
    for label, implementation in (
        ("previous", _previous_implementation),
        ("current", dual_gram),
    ):
        monkeypatch.setattr(matrixfree, "dual_gram", implementation)
        system = T.exact_tangent_system(model, x, y, config.fgd_approx)
        errors[label] = exact_relative_error(
            model, x, y, config.fgd_approx, system=system
        )

    assert errors["previous"] == pytest.approx(errors["current"], abs=1e-10)


def test_the_expansion_is_never_materialised() -> None:
    """The (NK, NK) repeat of an (n, n) matrix is what made it 5x.

    Reading the source is the only way to pin an allocation that is now absent:
    a broadcast leaves no tensor to assert on.
    """
    import inspect

    source = inspect.getsource(dual_gram)
    assert "repeat_interleave" not in source, (
        "materialising expand(a a^T) rebuilds an (NK, NK) tensor that holds "
        "only (n, n) worth of information"
    )
    assert "mul_(" in source, "the block multiply must stay in place"
