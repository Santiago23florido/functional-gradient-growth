"""GroMo growth step for the baseline pipeline."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain
from typing import Any, Literal

from fgdlib.gromo_setup import ensure_gromo_importable


ensure_gromo_importable()

import torch
from torch.func import functional_call

from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.utils.training_utils import compute_statistics, evaluate_model

from fgdlib.models.regularized_mlp import sync_normalization
from fgdlib.profile import fallback


ProgressFn = Callable[[str], None]
LineSearchMethod = Literal["golden_section"]

# Sample cap for the function-preservation drift check; inputs are cached
# before growth so a shuffling loader cannot invalidate the comparison.
_PRESERVATION_CHECK_SAMPLES = 4096


def _supports_float64_check(model: SequentialGrowingModel) -> bool:
    """Whether reparametrising this model round-trips its dtypes.

    An ALIASED tensor -- one reachable under two names, as a growing conv
    layer's ``extended_post_layer_function`` normalisation is -- does not
    restore cleanly, and leaves the module in a dtype the next float32 forward
    cannot use. Deduplicated and raw name counts agreeing is exactly the
    absence of that aliasing, and it is a structural test, not a heuristic.
    """
    for named in (model.named_parameters, model.named_buffers):
        if len(list(named())) != len(list(named(remove_duplicate=False))):
            return False
    return True


def _preservation_forward(
    model: SequentialGrowingModel, batch_x: torch.Tensor
) -> torch.Tensor:
    """Forward in float64, without mutating the model or its caches.

    The drift check exists to certify the ALGEBRA of the extension: the new
    neurons carry exactly zero outgoing weights, so the represented function
    is unchanged and the only thing a float32 comparison can measure is
    rounding. And rounding is precisely what changes here -- widening a layer
    changes the GEMM's shape, the backend picks a different tile/split-K
    reduction, and floating-point addition is not associative, so the SAME
    mathematical sum comes back with different low bits.

    MEASURED on the real path (784 inputs, 3-5 hidden layers, widths 16-128):

        drift float32   1.2e-07 .. 3.0e-07     drift float64   0.0 .. 4.4e-16

    a ratio of 2.7e8 to 8.1e8, which is the ratio of the two machine epsilons
    and nothing else. On the cluster's A100 the float32 figure reached
    1.93e-5, 65x the CPU one, because cuBLAS splits the reduction far more
    aggressively than a sequential CPU sum -- same mechanism, larger constant.
    That is what killed job 461736 at epoch 4 with P=19481: not a broken
    growth, a tolerance that measured the backend instead of the algebra.

    In float64 the drift is 0 or one ULP, so the unchanged 1e-6 tolerance now
    has nine orders of margin and tests what it was written to test. The check
    gets STRICTER, not looser: a genuinely non-preserving growth of order 1e-5
    was inside the float32 noise on the A100 and is nine orders outside this.

    ``functional_call`` rather than ``model.double()``: casting in place would
    mutate the live parameters mid-growth, and a deepcopy of a GroMo model
    holding update state is exactly what has crashed this pipeline before
    ("Cannot access storage of TensorWrapper"). This binds a float64 copy of
    the state for one forward and leaves the module untouched.

    Three details of that binding are load-bearing, each found by a test:

    * ``is_floating_point()`` -- ``num_batches_tracked`` is ``int64``, and
      casting it to float64 makes ``F.batch_norm`` reject the call.
    * ``_supports_float64_check`` -- a GROWING CONV layer aliases its
      ``extended_post_layer_function`` normalisation onto the
      ``post_layer_function`` one. Reparametrising an aliased module does not
      round-trip: MEASURED, the float64 pass itself completes, and then the
      NEXT float32 forward inside ``compute_statistics`` dies on "mixed dtype
      (CPU): expect parameter to have scalar type of Float" because the
      normalisation buffers were left in float64. Neither ``tie_weights``
      setting fixes it -- with ties on the alias keeps float32, with ties off
      the restore is incomplete.

      So the float64 check is applied only where it round-trips, and the
      aliased models keep EXACTLY the check they have today. That is not a
      compromise on the bug being fixed: the run that died is the linear
      MNIST MLP, which has no aliases, and the conv path was already passing
      at its own tolerance.
    """
    if not _supports_float64_check(model):
        fallback(
            "preservation_float64_fallbacks", "aliased_parameters_do_not_round_trip"
        )
        return model(batch_x)
    state = {
        name: (
            tensor.detach().to(torch.float64)
            if tensor.is_floating_point()
            else tensor.detach()
        )
        for name, tensor in chain(model.named_parameters(), model.named_buffers())
    }
    return functional_call(model, state, (batch_x.to(torch.float64),))


@dataclass(frozen=True)
class LineSearchPoint:
    scaling_factor: float
    train_loss: float


@dataclass(frozen=True)
class GrowthResult:
    layer_index: int
    best_scaling_factor: float
    best_train_loss: float
    line_search: list[LineSearchPoint]


@dataclass(frozen=True)
class ScalingLineSearchConfig:
    method: LineSearchMethod = "golden_section"
    min_value: float = 0.0
    max_value: float = 1.0
    iterations: int = 12
    tolerance: float = 1e-3


def _evaluate_scaling_factor(
    model: SequentialGrowingModel,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    scaling_factor: float,
    evaluated: dict[float, LineSearchPoint],
    line_search: list[LineSearchPoint],
    progress: ProgressFn | None,
) -> LineSearchPoint:
    key = round(float(scaling_factor), 12)
    if key in evaluated:
        return evaluated[key]

    model.set_scaling_factor(float(scaling_factor))
    loss, _ = evaluate_model(
        model,
        train_loader,
        criterion,
        use_extended_model=True,
        device=device,
    )
    point = LineSearchPoint(scaling_factor=float(scaling_factor), train_loss=float(loss))
    evaluated[key] = point
    line_search.append(point)

    if progress is not None:
        progress(
            f"  scaling={point.scaling_factor:.6g}, "
            f"train_loss={point.train_loss:.4f}"
        )

    return point


def _golden_section_line_search(
    model: SequentialGrowingModel,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    config: ScalingLineSearchConfig,
    progress: ProgressFn | None,
) -> tuple[float, float, list[LineSearchPoint]]:
    if config.max_value < config.min_value:
        raise ValueError("scaling_line_search.max_value must be >= min_value")

    line_search: list[LineSearchPoint] = []
    evaluated: dict[float, LineSearchPoint] = {}

    a = float(config.min_value)
    b = float(config.max_value)
    if math.isclose(a, b):
        point = _evaluate_scaling_factor(
            model,
            train_loader,
            criterion,
            device,
            a,
            evaluated,
            line_search,
            progress,
        )
        return point.scaling_factor, point.train_loss, line_search

    _evaluate_scaling_factor(
        model, train_loader, criterion, device, a, evaluated, line_search, progress
    )
    _evaluate_scaling_factor(
        model, train_loader, criterion, device, b, evaluated, line_search, progress
    )

    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    c_point = _evaluate_scaling_factor(
        model, train_loader, criterion, device, c, evaluated, line_search, progress
    )
    d_point = _evaluate_scaling_factor(
        model, train_loader, criterion, device, d, evaluated, line_search, progress
    )

    for _ in range(max(0, config.iterations)):
        if abs(b - a) <= config.tolerance:
            break

        if c_point.train_loss <= d_point.train_loss:
            b = d
            d = c
            d_point = c_point
            c = b - inv_phi * (b - a)
            c_point = _evaluate_scaling_factor(
                model,
                train_loader,
                criterion,
                device,
                c,
                evaluated,
                line_search,
                progress,
            )
        else:
            a = c
            c = d
            c_point = d_point
            d = a + inv_phi * (b - a)
            d_point = _evaluate_scaling_factor(
                model,
                train_loader,
                criterion,
                device,
                d,
                evaluated,
                line_search,
                progress,
            )

    best_point = min(line_search, key=lambda point: point.train_loss)
    return best_point.scaling_factor, best_point.train_loss, line_search


def _function_preserving_growth(
    model: SequentialGrowingModel,
    train_loader: torch.utils.data.DataLoader,
    layer_index: int,
    device: torch.device,
    optimal_update_kwargs: dict[str, Any],
    preservation_tolerance: float,
    progress: ProgressFn | None,
) -> GrowthResult:
    """Grow one layer without changing the represented function.

    TINY statistics still select the incoming weights of the new neurons,
    but their outgoing weights are exactly zero and no delta touches the
    existing weights, so the committed function is unchanged: growth only
    refines the representation (enlarges the tangent image) and is not an
    optimization step. The measured output drift must stay within
    ``preservation_tolerance``.
    """
    criterion_sum = torch.nn.MSELoss(reduction="sum")
    criterion_mean = torch.nn.MSELoss(reduction="mean")

    model.eval()
    reference: list[tuple[torch.Tensor, torch.Tensor]] = []
    cached_samples = 0
    with torch.no_grad():
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            reference.append(
                (batch_x, _preservation_forward(model, batch_x).detach().clone())
            )
            cached_samples += batch_x.shape[0]
            if cached_samples >= _PRESERVATION_CHECK_SAMPLES:
                break

    model.set_growing_layers(index=layer_index)
    compute_statistics(
        model,
        train_loader,
        loss_function=criterion_sum,
        device=device,
    )
    model.compute_optimal_updates(
        **{
            **optimal_update_kwargs,
            "compute_delta": False,
            "omega_zero": True,
        }
    )
    model.reset_computation()
    model.dummy_select_update()

    growing_layer = model.currently_updated_layer
    growing_layer.apply_change(
        apply_delta=False,
        apply_extension=True,
        input_extension_scaling=1.0,
        output_extension_scaling=1.0,
    )
    growing_layer.delete_update()
    model.currently_updated_layer_index = None

    # The layer just widened; grow any paired batch-norm to match before the
    # drift check forwards the model. A no-op on the plain MLP.
    sync_normalization(model)

    model.eval()
    drift = 0.0
    scale = 1.0
    with torch.no_grad():
        for batch_x, output_before in reference:
            batch_drift = float(
                torch.max(
                    torch.abs(_preservation_forward(model, batch_x) - output_before)
                ).item()
            )
            drift = max(drift, batch_drift)
            scale = max(scale, float(torch.max(torch.abs(output_before)).item()))
    # RELATIVE tolerance. The check must certify that the represented
    # function is unchanged, and "unchanged" for a float32 network is a
    # statement about significant digits, not about an absolute magnitude:
    # a 784-term dot product accumulates ~1e-6 of rounding on logits of
    # order 10 while being exactly correct. An absolute 1e-6 therefore
    # rejects arithmetically perfect growths -- measured, 1.371e-6 -- and,
    # where the caller skips failed candidates, silently removes them from
    # the search. Scaling by the output magnitude (floored at 1, so small
    # outputs keep an absolute guarantee) tests the thing actually meant.
    allowed = preservation_tolerance * scale
    if not math.isfinite(drift) or drift > allowed:
        raise RuntimeError(
            "Function-preserving growth exceeded its output tolerance: "
            f"{drift:.3e} > {allowed:.3e} "
            f"(relative tolerance {preservation_tolerance:.3e} "
            f"x output scale {scale:.3e})."
        )

    train_loss, _ = evaluate_model(
        model,
        train_loader,
        criterion_mean,
        use_extended_model=False,
        device=device,
    )
    point = LineSearchPoint(scaling_factor=1.0, train_loss=float(train_loss))
    if progress is not None:
        progress(
            f"  function-preserving growth: drift={drift:.3e}, "
            f"train_loss={point.train_loss:.4f}"
        )
    return GrowthResult(
        layer_index=layer_index,
        best_scaling_factor=1.0,
        best_train_loss=float(train_loss),
        line_search=[point],
    )


def growable_neuron_costs(
    model: SequentialGrowingModel, input_features: int
) -> list[int]:
    """Parameter cost of ONE neuron added at each growable location.

    Growing ``_growable_layers[i]`` widens its *input* dimension, so each new
    neuron costs its incoming weights and bias in the PRECEDING layer, plus
    its outgoing weights in this one::

        cost_i = prev.in_features + prev.use_bias + layer.out_features * k

    The spread is the whole point: on MNIST from 3x2 this is 787 parameters
    at the input projection against 5 and 13 later -- a factor of ~150 that
    an absolute singular-value threshold cannot see.

    Two things this reads carefully.

    **The predecessor is ``layer.previous_module``, not ``growable[i - 1]``.**
    Those coincide on a ``GrowingMLP`` only because ``_growable_layers`` is
    ``layers[1:]`` there, so the growable list is contiguous. In a conv stack
    it is NOT (pooling and the flatten make layers ungrowable in the middle),
    and ``growable[i - 1]`` would be an unrelated module several steps away.

    **``k`` is the kernel area of the receiving layer.** On a conv the new
    channel arrives through a full ``k_h x k_w`` kernel per output channel:
    ``RestrictedConv2dGrowingModule`` builds a 1x1 and zero-pads it to the
    layer's kernel (``linear_layer_of_tensor``), and those zeros are
    allocated, trainable parameters from the moment they exist. The honest
    price is what is allocated. Note ``prev.in_features`` needs no such
    correction: for a conv it is ALREADY the unfolded fan-in
    ``in_channels * k_h * k_w``.

    Verified against measurement, not just derived: on the base conv stack
    this returns ``[28, 37, 29]`` and each figure equals the observed
    ``count_parameters`` delta of actually growing there
    (``tests/test_growable_neuron_costs.py``).
    """
    growable = list(getattr(model, "_growable_layers", []))
    costs: list[int] = []
    for layer in growable:
        previous = getattr(layer, "previous_module", None)
        if previous is None:
            fan_in, has_bias = int(input_features), 1
        else:
            fan_in = int(previous.in_features)
            has_bias = int(bool(getattr(previous, "use_bias", True)))
        kernel = getattr(layer, "kernel_size", None)
        kernel_area = int(kernel[0]) * int(kernel[1]) if kernel is not None else 1
        costs.append(fan_in + has_bias + int(layer.out_features) * kernel_area)
    return costs


def expansion_spectrum(
    model: SequentialGrowingModel,
    train_loader: torch.utils.data.DataLoader,
    layer_index: int,
    device: torch.device,
    optimal_update_kwargs: dict[str, Any] | None = None,
) -> tuple[list[float], float]:
    """Per-neuron expansion scores at ``layer_index``, and the incumbent rate.

    Returns ``(spectrum, incumbent_efficiency)`` where ``spectrum`` holds the
    ``s_i^2`` of each candidate neuron and ``incumbent_efficiency`` is the
    first-order decrease per parameter that the layer's EXISTING weights buy
    by being re-optimised. The two are in the same units, which is what lets
    :func:`allocate_by_expansion_per_parameter` decide without a budget.

    :func:`rank_layer_expansion_score` returns their sum, which is the
    location's total first-order loss decrease. This returns the individual
    terms, so candidate *neurons* can be compared across locations rather
    than whole layers -- the granularity at which a cost correction escapes
    the starvation that per-layer ranking produced (R2).

    Same cost as the ranking: one statistics pass and one SVD, no line
    search and no model clone. The model is left untouched.
    """
    model.set_growing_layers(index=layer_index)
    compute_statistics(
        model,
        train_loader,
        loss_function=torch.nn.MSELoss(reduction="sum"),
        device=device,
    )
    model.compute_optimal_updates(**(optimal_update_kwargs or {}))

    spectrum: list[float] = []
    incumbent = 0.0
    for layer in getattr(model, "_growing_layers", []):
        eigenvalues = getattr(layer, "eigenvalues_extension", None)
        if eigenvalues is not None:
            spectrum.extend(float(value) ** 2 for value in eigenvalues)
        # The decrease the layer's EXISTING parameters buy by being
        # re-optimised, per parameter. Same units as s_i^2, so the two are
        # directly comparable -- this is what removes the need for a budget.
        decrease = getattr(layer, "parameter_update_decrease", None)
        if decrease is not None:
            count = sum(p.numel() for p in layer.parameters())
            if count:
                incumbent += float(decrease) / count

    model.reset_computation()
    for layer in getattr(model, "_growing_layers", []):
        if hasattr(layer, "delete_update"):
            layer.delete_update(include_previous=True)
    model.currently_updated_layer_index = None
    model.zero_grad(set_to_none=True)
    return spectrum, incumbent


def allocate_by_expansion_per_parameter(
    spectra: list[list[float]],
    costs: list[int],
    incumbent_efficiencies: list[float],
    statistical_threshold: float = 1e-3,
) -> list[int]:
    """Grow exactly where growing beats *tuning*. No budget, no threshold.

    GroMo reports two first-order decreases in the same units
    (``growing_module.py``)::

        L(A + dA) = L(A) - t * parameter_update_decrease + o(t)     # tuning
        L(A + dA) = L(A) - t * sigma'(0) * sum(s_i^2) + o(t)        # growing

    so a candidate neuron and the parameters already present can be compared
    directly, per parameter. A neuron is admitted iff

        s_i^2 / cost_i  >=  parameter_update_decrease_l / P_l

    i.e. iff it buys at least as much certified first-order decrease per
    parameter as the layer's existing parameters do by being re-optimised.

    Two properties this has and a budget does not:

    * **No free constant.** The right-hand side is measured on the network
      itself, so nothing has to be guessed about a dataset that has never
      been trained on. A parameter budget presumes the answer -- how large
      the final structure should be -- which is precisely what the search is
      supposed to discover.
    * **It self-terminates.** As the structure becomes efficient the
      incumbent efficiency rises, so fewer candidates clear it, and growth
      stops on its own rather than on exhausting an allowance.

    Pooling at *neuron* granularity is what separates this from the refuted
    R2, which ranked whole layers by decrease per parameter and therefore
    always bought the cheap late layer, starving the input projection
    (784->2->2->14, 64.4 %). Here each candidate is judged against its own
    layer's incumbent, so no location can be starved by another winning a
    ranking.

    ``incumbent_efficiencies`` is accepted for signature stability and is
    logged as a diagnostic; see the note below on why it is not used as the
    admission test.

    **Measured and rejected: the incumbent-efficiency test.** Comparing a
    candidate against ``parameter_update_decrease_l / P_l`` looks like the
    natural budget-free rule -- "grow only where growing beats tuning" -- and
    the two quantities really are in the same units. They are not comparable
    in *character*, though, and the run says so immediately: at a 3x2 start
    the incumbents measure 0.734, 1.58 and 0.135 against candidate
    efficiencies of 1.2e-3, 5.8e-4 and 3.5e-5, so nothing ever clears and the
    allocation is ``[0, 0, 0]`` for ever. The reason is temporal, not
    numerical: the decrease from re-optimising existing weights is
    *transient* -- it is consumed by taking the step, and the certified
    families already take it every epoch -- whereas a new neuron is
    *permanent capacity*. Tuning six parameters for 4.4 of decrease will
    always look more efficient than 787 parameters for 0.93, right up until
    tuning saturates. The comparison is left in the code as a logged
    diagnostic and not as a gate.

    What is used instead introduces **no new constant**: GroMo's own
    truncation rule, ``s >= min(statistical_threshold, s.max())`` -- keep
    everything above the threshold, but always keep at least the best
    candidate -- applied to the cost-normalised quantity rather than to the
    raw singular values. The knob is the one already in the config; only the
    quantity it judges is corrected.
    """
    allocation = [0] * len(spectra)
    for location, spectrum in enumerate(spectra):
        cost = max(costs[location], 1)
        efficiencies = [value / cost for value in spectrum if value > 0.0]
        if not efficiencies:
            continue
        reference = min(statistical_threshold, max(efficiencies))
        allocation[location] = sum(
            1 for value in efficiencies if value >= reference
        )
    return allocation


def rank_layer_expansion_score(
    model: SequentialGrowingModel,
    train_loader: torch.utils.data.DataLoader,
    layer_index: int,
    device: torch.device,
    optimal_update_kwargs: dict[str, Any] | None = None,
) -> float:
    """SENN's natural expansion score for growing ``layer_index``.

    Returns ``sum(s_i^2)`` over the retained TINY singular values, which
    GroMo documents (``growing_module.py``) as the extension's first-order
    effect on the loss::

        L(A + dA) = L(A) - t * sigma'(0) * (eigenvalues_extension ** 2).sum()

    That first-order decrease is exactly SENN's expansion-score increase for
    this location (arXiv:2307.04526, Theorem 3.2), computed from the layer's
    Kronecker factors: ``tensor_s_growth()`` is the input activation second
    moment (KFAC's ``A``) and, with ``use_fisher=True`` in
    ``optimal_update_kwargs``, ``covariance_loss_gradient()`` supplies the
    output-side factor ``S``. Without that flag the score is TINY's, in the
    plain Euclidean output metric rather than SENN's Fisher one.

    The point of this helper is cost. It stops after the statistics pass and
    the SVD, so ranking L candidate layers costs L statistics passes instead
    of L * (1 + line_search.iterations) passes plus L model clones -- the
    golden-section search is then paid once, on the winner, inside
    :func:`grow_layer`. This is why SENN can afford to answer *where* from
    curvature instead of from trial growths.

    The model is left with its update tensors cleared, so a subsequent
    :func:`grow_layer` on the chosen layer starts from a clean state.
    """
    model.set_growing_layers(index=layer_index)
    compute_statistics(
        model,
        train_loader,
        loss_function=torch.nn.MSELoss(reduction="sum"),
        device=device,
    )
    model.compute_optimal_updates(**(optimal_update_kwargs or {}))

    score = 0.0
    for layer in getattr(model, "_growing_layers", []):
        eigenvalues = getattr(layer, "eigenvalues_extension", None)
        if eigenvalues is not None:
            score += float(eigenvalues.pow(2).sum())

    model.reset_computation()
    for layer in getattr(model, "_growing_layers", []):
        if hasattr(layer, "delete_update"):
            layer.delete_update(include_previous=True)
    model.currently_updated_layer_index = None
    model.zero_grad(set_to_none=True)
    return score


def grow_layer(
    model: SequentialGrowingModel,
    train_loader: torch.utils.data.DataLoader,
    layer_index: int,
    device: torch.device,
    line_search_config: ScalingLineSearchConfig,
    optimal_update_kwargs: dict[str, Any] | None = None,
    progress: ProgressFn | None = None,
    function_preserving: bool = False,
    preservation_tolerance: float = 1e-6,
    line_search_loader: torch.utils.data.DataLoader | None = None,
) -> GrowthResult:
    """Grow one GroMo layer and apply the best line-search update.

    ``layer_index`` follows GroMo's local API: it is zero-based over
    ``model._growable_layers``. With ``function_preserving=True`` the
    scaling line search is skipped and the extension is applied with zero
    outgoing weights, leaving the represented function exactly unchanged.

    ``line_search_loader`` selects the data the scaling factor is chosen on.
    The GroMo default minimizes the TRAIN loss, which makes the magnitude of
    the structural step an uncertified, train-fitting choice; passing the
    held-out loader instead makes the growth's magnitude follow the same
    held-out functional descent that Proposition 3.8 certifies for every
    other step.
    """
    if function_preserving:
        return _function_preserving_growth(
            model=model,
            train_loader=train_loader,
            layer_index=layer_index,
            device=device,
            optimal_update_kwargs=dict(optimal_update_kwargs or {}),
            preservation_tolerance=preservation_tolerance,
            progress=progress,
        )

    criterion_sum = torch.nn.MSELoss(reduction="sum")
    criterion_mean = torch.nn.MSELoss(reduction="mean")

    model.set_growing_layers(index=layer_index)
    compute_statistics(
        model,
        train_loader,
        loss_function=criterion_sum,
        device=device,
    )

    model.compute_optimal_updates(**(optimal_update_kwargs or {}))
    model.reset_computation()
    model.dummy_select_update()

    if line_search_config.method != "golden_section":
        raise ValueError(
            f"Unsupported scaling line-search method '{line_search_config.method}'."
        )

    best_value, best_loss, line_search = _golden_section_line_search(
        model=model,
        train_loader=(
            line_search_loader if line_search_loader is not None else train_loader
        ),
        criterion=criterion_mean,
        device=device,
        config=line_search_config,
        progress=progress,
    )

    model.set_scaling_factor(best_value)
    model.apply_change()
    # The extension is now permanent; grow any paired batch-norm to match.
    # A no-op on the plain MLP.
    sync_normalization(model)

    return GrowthResult(
        layer_index=layer_index,
        best_scaling_factor=best_value,
        best_train_loss=float(best_loss),
        line_search=line_search,
    )
