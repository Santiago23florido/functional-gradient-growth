"""The vmapped path must not leave functorch wrappers on the model.

MEASURED on Margaret, MNIST: the analytic structure was refused
(``forward_has_caching_side_effects``), the builder fell back to the vmapped
operators, and the run died SEVEN CALLS LATER in
``_transactional_realize_functional_step``::

    base_model = copy.deepcopy(model)
    ...
    NotImplementedError: Cannot access storage of TensorWrapper

The cause names itself once stated: gromo's growing modules store their input
and pre-activity on every forward -- which is precisely why the analytic
structure was refused -- so a vmapped forward caches functorch
``TensorWrapper`` values onto the modules, and they outlive the call that made
them. ``paused_computation`` clears those two flags, but it was only held open
while the ANALYTIC structure was probed; the fallback ran with the cache live.

The regression this file pins is deliberately end-state, not mechanism: after
building a matrix-free system on a model whose analytic structure is refused,
the model must still be deep-copyable. That is the property the pipeline
actually depends on, and the one whose absence produced a stack trace naming
none of this.
"""

from __future__ import annotations

import copy

import pytest
import torch

from fgdlib import tangent
from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.tangent import FGDApproxConfig

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP  # noqa: E402


CPU = torch.device("cpu")


def _model() -> GrowingMLP:
    torch.manual_seed(0)
    return GrowingMLP(
        in_features=4,
        out_features=2,
        hidden_size=3,
        number_hidden_layers=2,
        activation=torch.nn.SELU(),
        device=CPU,
    )


def _config() -> FGDApproxConfig:
    return FGDApproxConfig(family_order=("matrix_free_tangent",))


def _probe() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return (
        torch.randn(24, 4, generator=generator),
        torch.randn(24, 2, generator=generator),
    )


def _force_vmap_path(monkeypatch) -> None:
    """Refuse the analytic structure, exactly as the cluster's gromo did."""
    monkeypatch.setattr(
        tangent, "_supported_analytic_structure", lambda *a, **k: None
    )


def test_the_model_survives_a_deepcopy_after_the_vmap_path(monkeypatch) -> None:
    """The exact failure: build the system, then deep-copy the model."""
    _force_vmap_path(monkeypatch)
    model, (x, y) = _model(), _probe()

    system = tangent._matrix_free_tangent_system(model=model, x=x, y=y, config=_config())
    assert system is not None, "the vmapped fallback produced no system"

    clone = copy.deepcopy(model)  # this is what raised on Margaret
    assert torch.equal(clone(x), model(x))


def test_the_capture_flags_are_restored_afterwards(monkeypatch) -> None:
    """Pausing must be a loan, not a permanent change to the model."""
    _force_vmap_path(monkeypatch)
    model, (x, y) = _model(), _probe()

    before = [
        (module.store_input, module.store_pre_activity)
        for module in model.modules()
        if hasattr(module, "store_input")
    ]
    assert before, "no growing modules found; the test would prove nothing"

    tangent._matrix_free_tangent_system(model=model, x=x, y=y, config=_config())

    after = [
        (module.store_input, module.store_pre_activity)
        for module in model.modules()
        if hasattr(module, "store_input")
    ]
    assert after == before


def test_no_stored_tensor_is_a_functorch_wrapper(monkeypatch) -> None:
    """State the property directly, so a future regression names itself.

    ``deepcopy`` is the symptom; the defect is a wrapper reachable from the
    module state at all.
    """
    _force_vmap_path(monkeypatch)
    model, (x, y) = _model(), _probe()

    tangent._matrix_free_tangent_system(model=model, x=x, y=y, config=_config())

    for module in model.modules():
        for attribute in ("input", "pre_activity"):
            try:
                stored = getattr(module, attribute, None)
            except ValueError:
                # gromo raises "The input is not stored" from the property
                # itself. That is the BEST outcome, not a missing check: with
                # capture suspended there is nothing cached to be a wrapper.
                continue
            if isinstance(stored, torch.Tensor):
                # Reaching for the storage is what deepcopy does and what a
                # functorch wrapper refuses; if this raises, the leak is back.
                stored.untyped_storage()


def test_the_analytic_path_is_not_paused_by_this(monkeypatch) -> None:
    """Untouched control: with the structure accepted, nothing above changes."""
    model, (x, y) = _model(), _probe()
    accepted = tangent._matrix_free_tangent_system(model=model, x=x, y=y, config=_config())
    if accepted is None:
        pytest.skip("this model has no analytic structure to serve as control")
    assert copy.deepcopy(model) is not None
