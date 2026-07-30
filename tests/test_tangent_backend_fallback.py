"""The optimized backend must never degrade silently.

``FGD_TANGENT_BACKEND=auto`` is allowed to fall back to jacrev whenever the
model or probe does not satisfy the P1-P14 predicates the analytic Jacobian
was derived under (TANGENT_PLAN.md A.4). The danger is not the fallback -- it
is a fallback nobody can see: a run that quietly stops using the fast path,
or worse, one that keeps using it on a model it is wrong for. So every
predicate must (a) actually fire, (b) increment
``tangent_unsupported_structure_fallbacks`` and (c) record its OWN reason
string, distinct from every other predicate's, so a profile line names the
cause.

``optimized`` is the strict twin of ``auto``: same construction, but it
RAISES where ``auto`` falls back, which is what lets a test or a benchmark
prove the fast path really ran rather than having quietly degraded into the
path it was being compared against.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import pytest
import torch
from torch.func import functional_call, jacrev

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.profile import _REASONS, reset, snapshot
from fgdlib.tangent import (
    FGDApproxConfig,
    _activation_rejection_reason,
    _flatten_jacobian,
    _stream_gram_surrogate,
    _supported_analytic_structure,
    _trainable_named_parameters,
    exact_tangent_system,
    tangent_backend,
)

ensure_gromo_importable()

from gromo.containers.growing_mlp import GrowingMLP

IN_FEATURES = 6
OUT_FEATURES = 3
HIDDEN = 4
#: N*K must be at least P (= 63 here) so the streamed factor R comes out
#: SQUARE. A shorter probe leaves R at (N*K) x P, which the optimized
#: surrogate path does not support -- see the report accompanying this
#: change; nothing here relies on that regime.
SAMPLES = 32


def _model(**kwargs) -> GrowingMLP:
    torch.manual_seed(0)
    model = GrowingMLP(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_size=HIDDEN,
        number_hidden_layers=2,
        device=torch.device("cpu"),
        **kwargs,
    )
    model.eval()
    return model


def _probe() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    return (
        torch.randn(SAMPLES, IN_FEATURES),
        torch.randn(SAMPLES, OUT_FEATURES),
    )


@pytest.fixture
def profiling(monkeypatch):
    """Reasons are only recorded when FGD_PROFILE is on -- turn it on."""
    monkeypatch.setenv("FGD_PROFILE", "1")
    reset()
    yield
    reset()


# --------------------------------------------------------------------------
# P1-P14: one construction per predicate, one reason per construction
# --------------------------------------------------------------------------


class _CouplingIdentity(torch.nn.Identity):
    """Passes the P12 type allow-list (it IS an nn.Identity) but couples
    neighbouring features, which is exactly what the numeric probe exists
    to catch: the allow-list alone is not a sufficient guarantee."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1e-3 * value.roll(1, dims=1)


class _ShrinkingIdentity(torch.nn.Identity):
    """Allow-listed type that does not preserve shape."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :1]


class _StatefulIdentity(torch.nn.Identity):
    """Allow-listed type carrying a learned parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.gain


class _WeirdLinear(torch.nn.Linear):
    """A SUBCLASS of nn.Linear -- P3 rejects it because a subclass may
    override forward with arithmetic the analytic formulas never saw."""


class _UnsqueezingMLP(GrowingMLP):
    """Structurally a GrowingMLP but returns a 3-D output, breaking the
    row = n*K + k layout the analytic block writes."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(value).unsqueeze(-1)


def _case_not_a_growing_mlp():
    model = torch.nn.Sequential(torch.nn.Linear(IN_FEATURES, OUT_FEATURES))
    model.eval()
    return model, _probe()[0], "not_a_growing_mlp_sequential_container"


def _case_layer_not_growing_module():
    model = _model()
    model.layers[0] = torch.nn.Linear(IN_FEATURES, HIDDEN)
    return model, _probe()[0], "layer_not_linear_growing_module"


def _case_inner_layer_not_plain_linear():
    model = _model()
    model.layers[0].layer = _WeirdLinear(IN_FEATURES, HIDDEN)
    return model, _probe()[0], "inner_layer_not_plain_linear"


def _case_degenerate_fan_in():
    model = _model()
    model.layers[0].layer.in_features = 0
    return model, _probe()[0], "degenerate_fan_in_or_out"


def _case_bias_flag_inconsistent():
    model = _model()
    assert model.layers[0].layer.bias is not None
    model.layers[0].use_bias = False
    return model, _probe()[0], "bias_flag_inconsistent"


def _case_flatten_not_batch_preserving():
    model = _model()
    model.flatten = torch.nn.Flatten(start_dim=0, end_dim=-1)
    return model, _probe()[0], "flatten_not_batch_preserving"


def _case_pending_growth_extension():
    model = _model()
    model.layers[1].extended_input_layer = torch.nn.Linear(HIDDEN, HIDDEN)
    return model, _probe()[0], "pending_growth_extension_state"


def _case_forward_caching():
    model = _model()
    model.layers[0].store_input = True
    return model, _probe()[0], "forward_has_caching_side_effects"


def _case_frozen_parameter():
    model = _model()
    model.layers[0].layer.weight.requires_grad_(False)
    return model, _probe()[0], "unexpected_trainable_parameter_set"


def _case_model_has_buffers():
    model = _model()
    model.register_buffer("running_probe", torch.zeros(2))
    return model, _probe()[0], "model_has_buffers"


def _case_training_mode():
    model = _model()
    model.train()
    return model, _probe()[0], "model_not_in_eval_mode"


def _case_activation_not_allowlisted():
    model = _model(activation=torch.nn.Softmax(dim=-1))
    return model, _probe()[0], "activation_type_not_allowlisted"


def _case_activation_changes_shape():
    model = _model(activation=_ShrinkingIdentity())
    return model, _probe()[0], "activation_changes_shape"


def _case_activation_couples():
    model = _model(activation=_CouplingIdentity())
    return model, _probe()[0], "activation_has_cross_entry_coupling"


def _case_flatten_not_2d():
    model = _model()
    model.flatten = torch.nn.Identity()
    x = torch.randn(SAMPLES, 2, IN_FEATURES)
    return model, x, "flatten_not_2d"


def _case_probe_dtype_mismatch():
    model = _model()
    return model, _probe()[0].to(torch.float64), "probe_dtype_or_device_mismatch"


def _case_output_not_2d():
    torch.manual_seed(0)
    model = _UnsqueezingMLP(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_size=HIDDEN,
        number_hidden_layers=2,
        device=torch.device("cpu"),
    )
    model.eval()
    return model, _probe()[0], "output_not_2d_row_aligned"


#: (predicate label, builder). Ordered like P1-P14 so a failure names the
#: predicate that regressed.
PREDICATE_CASES = [
    ("P1_not_growing_mlp", _case_not_a_growing_mlp),
    ("P2_layer_type", _case_layer_not_growing_module),
    ("P3_inner_linear_subclass", _case_inner_layer_not_plain_linear),
    ("P4_degenerate_fan", _case_degenerate_fan_in),
    ("P5_bias_flag", _case_bias_flag_inconsistent),
    ("P6_flatten_shape", _case_flatten_not_batch_preserving),
    ("P7_pending_growth", _case_pending_growth_extension),
    ("P8_forward_caching", _case_forward_caching),
    ("P9_frozen_parameter", _case_frozen_parameter),
    ("P10_buffers", _case_model_has_buffers),
    ("P11_training_mode", _case_training_mode),
    ("P12a_activation_type", _case_activation_not_allowlisted),
    ("P12c_activation_shape", _case_activation_changes_shape),
    ("P12d_activation_coupling", _case_activation_couples),
    ("P13a_flatten_not_2d", _case_flatten_not_2d),
    ("P13b_probe_dtype", _case_probe_dtype_mismatch),
    ("P14_output_rank", _case_output_not_2d),
]


@pytest.mark.parametrize(
    "label,builder", PREDICATE_CASES, ids=[label for label, _ in PREDICATE_CASES]
)
def test_predicate_falls_back_and_records_its_own_reason(
    profiling, label: str, builder
) -> None:
    model, x, expected = builder()
    reset()
    assert _supported_analytic_structure(model, x) is None, label
    values = snapshot()
    assert values["tangent_unsupported_structure_fallbacks"] == 1, label
    assert _REASONS["tangent_unsupported_structure_fallbacks"] == {expected}, label


def test_every_predicate_records_a_distinct_reason(profiling) -> None:
    """A shared reason string would make the profile line useless: two
    different structural defects would be indistinguishable in a run's
    output. Distinctness is the whole point of recording a reason."""
    reasons = []
    for _, builder in PREDICATE_CASES:
        model, x, expected = builder()
        reset()
        assert _supported_analytic_structure(model, x) is None
        (recorded,) = _REASONS["tangent_unsupported_structure_fallbacks"]
        assert recorded == expected
        reasons.append(recorded)
    assert len(set(reasons)) == len(reasons), sorted(reasons)


def test_supported_model_records_no_fallback_at_all(profiling) -> None:
    model = _model()
    x, _ = _probe()
    structure = _supported_analytic_structure(model, x)
    assert structure is not None
    assert structure.parameter_names == tuple(_trainable_named_parameters(model))
    assert structure.output_features == OUT_FEATURES
    assert snapshot()["tangent_unsupported_structure_fallbacks"] == 0
    assert _REASONS == {}


def test_activation_with_parameters_is_rejected_by_the_activation_probe() -> None:
    """P12.2 in isolation.

    An activation carrying an nn.Parameter cannot reach P12 through a whole
    model: the parameter shows up in ``model.parameters()`` and P9 rejects
    the model first with ``unexpected_trainable_parameter_set``. That
    ordering is deliberate (the cheap set comparison runs before the numeric
    probes), so the stateless check is pinned directly on the predicate.
    """
    phi = _StatefulIdentity()
    reason = _activation_rejection_reason(
        phi, HIDDEN, torch.device("cpu"), torch.float32
    )
    assert reason == "activation_has_parameters_or_buffers"


def test_activation_probe_accepts_every_allowlisted_activation() -> None:
    for phi in (
        torch.nn.Identity(),
        torch.nn.SELU(),
        torch.nn.ReLU(),
        torch.nn.LeakyReLU(),
        torch.nn.ELU(),
        torch.nn.GELU(),
        torch.nn.SiLU(),
        torch.nn.Tanh(),
        torch.nn.Sigmoid(),
        torch.nn.Softplus(),
        torch.nn.Sequential(torch.nn.Tanh(), torch.nn.Identity()),
    ):
        assert (
            _activation_rejection_reason(
                phi, HIDDEN, torch.device("cpu"), torch.float32
            )
            is None
        ), type(phi).__name__


def test_growth_leaves_store_input_set_and_disables_the_fast_path(
    profiling,
) -> None:
    """GroMo's growth leaves ``store_input`` incremented on the grown layer.

    ``store_input`` is a reference count, not a boolean, and
    ``LinearGrowingModule.__setattr__`` propagates it to
    ``_internal_store_input``. A model left in that state caches activations
    on every forward, so the analytic path -- which calls ``nn.Linear``
    directly and produces no such cache -- would leave the two backends'
    MODEL STATE different even though their Jacobians agreed. P8 refuses,
    and this test pins that it refuses rather than silently diverging.
    """
    model = _model()
    model.layers[1].store_input = True
    assert model.layers[1]._internal_store_input
    x, _ = _probe()
    assert _supported_analytic_structure(model, x) is None
    assert _REASONS["tangent_unsupported_structure_fallbacks"] == {
        "forward_has_caching_side_effects"
    }


# --------------------------------------------------------------------------
# Backend resolution: legacy / auto / optimized
# --------------------------------------------------------------------------


def _streaming_config(**overrides) -> FGDApproxConfig:
    return replace(
        FGDApproxConfig(
            projection_solver="exact",
            certify_stream_gram=True,
            certify_stream_chunk=8,
        ),
        **overrides,
    )


def test_invalid_backend_value_raises(monkeypatch) -> None:
    monkeypatch.setenv("FGD_TANGENT_BACKEND", "fast")
    with pytest.raises(ValueError, match="Unsupported FGD_TANGENT_BACKEND"):
        tangent_backend()


def test_unset_backend_resolves_to_auto(monkeypatch) -> None:
    """The default is ``auto``: the optimized construction with counted,
    reasoned fallback. ``auto`` rather than ``optimized`` because strict
    mode RAISES on any unsupported structure, and gromo legitimately
    enables input caching during growth certification -- a production run
    must degrade there, not die. Rollback stays one env var away."""
    monkeypatch.delenv("FGD_TANGENT_BACKEND", raising=False)
    assert tangent_backend() == "auto"


@pytest.mark.parametrize("value", ["legacy", "auto", "optimized"])
def test_backend_values_round_trip(monkeypatch, value: str) -> None:
    monkeypatch.setenv("FGD_TANGENT_BACKEND", value)
    assert tangent_backend() == value


def test_strict_optimized_backend_raises_instead_of_falling_back(
    monkeypatch, profiling
) -> None:
    monkeypatch.setenv("FGD_TANGENT_BACKEND", "optimized")
    model = _model()
    model.register_buffer("running_probe", torch.zeros(2))
    x, y = _probe()
    with pytest.raises(RuntimeError, match="requires the analytic MLP Jacobian"):
        exact_tangent_system(model, x, y, _streaming_config())
    assert snapshot()["tangent_unsupported_structure_fallbacks"] == 1
    assert _REASONS["tangent_unsupported_structure_fallbacks"] == {"model_has_buffers"}


def test_auto_backend_falls_back_and_still_builds_the_system(
    monkeypatch, profiling
) -> None:
    """The whole point of ``auto``: an unsupported model costs the fast path,
    never the run."""
    monkeypatch.setenv("FGD_TANGENT_BACKEND", "auto")
    model = _model()
    model.register_buffer("running_probe", torch.zeros(2))
    x, y = _probe()
    system = exact_tangent_system(model, x, y, _streaming_config())
    assert system is not None
    values = snapshot()
    assert values["tangent_unsupported_structure_fallbacks"] == 1
    assert values["tangent_analytic_jacobian_calls"] == 0
    assert values["tangent_jacrev_seconds"] > 0.0
    assert values["tangent_backend_optimized_calls"] == 1
    assert values["tangent_backend_legacy_calls"] == 0


def test_optimized_backend_runs_the_analytic_path_on_a_supported_model(
    monkeypatch, profiling
) -> None:
    monkeypatch.setenv("FGD_TANGENT_BACKEND", "optimized")
    model = _model()
    x, y = _probe()
    system = exact_tangent_system(model, x, y, _streaming_config())
    assert system is not None
    values = snapshot()
    assert values["tangent_unsupported_structure_fallbacks"] == 0
    assert values["tangent_analytic_jacobian_calls"] == 4  # 32 samples / 8
    assert values["tangent_sample_chunk_count"] == 4
    assert values["tangent_jacrev_seconds"] == 0.0
    assert values["tangent_analytic_jacobian_seconds"] > 0.0
    assert values["tangent_qr_calls"] == 4
    assert values["tangent_parameter_count"] == sum(
        parameter.numel() for parameter in _trainable_named_parameters(model).values()
    )
    assert values["tangent_output_row_count"] == SAMPLES * OUT_FEATURES


def _reference_legacy_system(
    model: GrowingMLP, x: torch.Tensor, y: torch.Tensor, config: FGDApproxConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pre-backend streaming construction, transcribed.

    jacrev per sample chunk, ``_flatten_jacobian``, cast to float64,
    ``torch.linalg.qr(..., mode="reduced")``, then ``_stream_gram_surrogate``
    -- the exact sequence ``_compute_exact_tangent_projection_step`` ran
    before ``FGD_TANGENT_BACKEND`` existed. ``legacy`` must reproduce this
    BITWISE, otherwise "legacy is today's path" is only a claim.
    """
    named = _trainable_named_parameters(model)
    names = tuple(named)
    parameters = tuple(named.values())
    buffers = OrderedDict(model.named_buffers())
    was_training = model.training
    model.eval()
    try:
        output = model(x)
        loss = torch.sum((output - y) ** 2)
        target = torch.autograd.grad(loss, output)[0].detach().reshape(-1)
        n_samples = int(x.shape[0])
        out_per_sample = target.numel() // n_samples
        chunk = int(config.certify_stream_chunk)
        r_factor = None
        b_acc = None
        r_sq = 0.0
        for start in range(0, n_samples, chunk):
            stop = min(start + chunk, n_samples)
            x_batch = x[start:stop]

            def call_batch(values, _xb=x_batch):
                state = OrderedDict(zip(names, values))
                state.update(buffers)
                return functional_call(model, state, (_xb,)).reshape(-1)

            rows = (stop - start) * out_per_sample
            block = _flatten_jacobian(jacrev(call_batch)(parameters), rows).to(
                torch.float64
            )
            r_block = target[start * out_per_sample : stop * out_per_sample].to(
                torch.float64
            )
            contribution = block.t() @ r_block
            b_acc = contribution if b_acc is None else b_acc + contribution
            r_sq += float((r_block * r_block).sum())
            stacked = block if r_factor is None else torch.cat([r_factor, block], dim=0)
            r_factor = torch.linalg.qr(stacked, mode="reduced").R
        return _stream_gram_surrogate(r_factor, b_acc, r_sq, target.dtype)
    finally:
        model.train(was_training)


def test_legacy_backend_is_bit_identical_to_the_pre_backend_path(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FGD_TANGENT_BACKEND", "legacy")
    model = _model()
    x, y = _probe()
    config = _streaming_config(functional_loss="mse")
    system = exact_tangent_system(model, x, y, config)
    assert system is not None
    jacobian, target = _reference_legacy_system(model, x, y, config)
    assert torch.equal(system.jacobian, jacobian)
    assert torch.equal(system.target, target)


def test_legacy_backend_never_touches_the_analytic_path(monkeypatch, profiling) -> None:
    monkeypatch.setenv("FGD_TANGENT_BACKEND", "legacy")
    model = _model()
    x, y = _probe()
    assert exact_tangent_system(model, x, y, _streaming_config()) is not None
    values = snapshot()
    assert values["tangent_backend_legacy_calls"] == 1
    assert values["tangent_backend_optimized_calls"] == 0
    assert values["tangent_analytic_jacobian_calls"] == 0
    assert values["tangent_analytic_jacobian_seconds"] == 0.0
    # legacy must not even pay for the structural probe, so no reason is
    # recorded either way.
    assert values["tangent_unsupported_structure_fallbacks"] == 0
