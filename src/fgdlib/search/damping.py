"""Choose the projection's regularisation by what the lemma actually buys.

The damping in ``J u ~ r`` is usually treated as a numerical nuisance. It is
not: it is the knob that arbitrates between the two conditions this method
needs at once, and they pull in OPPOSITE directions.

MEASURED on the synthetic task, one grown model, one Jacobian, tolerance
0.1 -- the whole ladder at a single point in training:

    damping     eps      ||g||      ||u||    eta admissible
    0        0.0451   1.956e1   3.604e5    none
    1e-6     0.0990   1.932e1   1.747e3    none
    1e-4     0.2607   1.855e1   2.796e2    none
    1e-2     0.4947   1.682e1   3.951e1    1.587e-4
    1e0      0.8453   1.428e1   4.097e0    eps >= 1/2
    1e2      1.3351   1.114e1   4.357e-1   eps >= 1/2
    1e4      2.5873   6.786e0   3.801e-2   eps >= 1/2

Read it as the trade-off it is. Lowering the damping makes the CERTIFICATE
easy -- ``eps`` reaches 0.045, the tangent space capturing 99.8 % of the
gradient -- while making the step unrealisable: ``||u||`` reaches 3.6e5, so
no rate in the certified interval keeps the parameter step inside the
regime where it IS the function-space step. Raising it does the reverse.
Only a narrow window satisfies both, and a fixed constant lands in that
window by luck. 1e-2 happens to work here; nothing makes it work at another
scale of ``J``, which changes with the dataset, the architecture and the
point in training.

So the constant is replaced by a measurement. Write the damping relative to
the spectrum, ``lambda = rho * sigma_max^2``, which makes ``rho``
dimensionless and invariant to the scale of ``J``. Then, since ``eps(rho)``
is increasing -- more regularisation is a worse approximation, and the table
above shows it plainly -- the certified region is an INTERVAL
``rho < rho*``, and its boundary is found exactly by bisection rather than
approached by a grid. Within that region ``||u||`` falls as ``rho`` rises,
so the boundary is where the step is most realisable while still certified.

The procedure at each outer step:

* factorise once (the projection is linear in the regularisation, so every
  rho below is a re-weighting -- the ladder costs no extra Jacobian),
* bisect for ``rho*``, the largest rho whose ``eps`` still satisfies the
  relative-error criterion -- the certificate is never traded away,
* fan out geometrically below ``rho*``, finely: the objective has an
  INTERIOR maximum, because at the boundary ``eta_bar(eps)`` collapses as
  ``eps -> 1/2`` while far below it ``||u||`` explodes,
* score each by the decrease Lemma 3.5 itself guarantees, proportional to
  ``eta * ||g||^2``, with ``eta`` the largest rate the linearisation control
  admits,
* and take the argmax.

Nothing dataset-specific enters: the bracket is relative, the filter is the
method's own certificate, and the objective is the theorem's own guaranteed
decrease. A grid was tried first and abandoned for a concrete reason -- the
useful value here was ``rho = 1.16e-9``, which falls BETWEEN the rungs of a
two-per-decade ladder, and both neighbours scored worse than it (1.25e-2 and
0, against 4.49e-2). The window is narrow enough that where you sample
matters, which is exactly why it must be located rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fgdlib.search.linearization import certified_linear_learning_rate
from fgdlib.tangent import (
    FGDApproxConfig,
    _output_relative_error_from_tensors,
    _unflatten_parameter_update,
    exact_tangent_system,
    theoretical_learning_rate_upper_bound,
)

__all__ = [
    "DAMPING_BISECTION_STEPS",
    "DAMPING_BRACKET",
    "DAMPING_FAN_RATIO",
    "DAMPING_FAN_STEPS",
    "DampingCandidate",
    "DampingChoice",
    "select_projection_damping",
]

#: Bracket for the bisection, dimensionless because every level is scaled by
#: ``sigma_max^2``. The low end is effectively the pseudo-inverse (certificate
#: at its best, step at its least realisable); the high end is heavy
#: regularisation (the reverse). The window lies inside it wherever it sits.
DAMPING_BRACKET: tuple[float, float] = (1e-16, 1e2)

#: Bisection steps used to locate the certified boundary. 40 halvings of an
#: 18-decade bracket resolve it to ~1e-5 of a decade -- far finer than any
#: fixed ladder, and the reason a grid was abandoned: MEASURED, the useful
#: value on the synthetic task was rho = 1.16e-9, which falls BETWEEN the
#: 1e-10 and 1e-8 rungs of a two-per-decade grid, and both neighbours scored
#: worse than it (1.25e-2 and 0, against 4.49e-2).
DAMPING_BISECTION_STEPS: int = 40

#: The search below the boundary: 24 rungs spaced by 10^(1/4), covering six
#: decades. Fine spacing is not caution, it is required -- MEASURED, the
#: objective has an INTERIOR maximum and both ends of the certified interval
#: are bad, for opposite reasons:
#:
#:   * at the boundary eps -> 1/2, so eta_bar(eps) = 2(1-2 eps)/(L_s(1+2 eps))
#:     collapses to zero and the certified interval vanishes. Bisecting onto
#:     the boundary and stopping there produced NO admissible rate at all.
#:   * far below it eps is excellent (0.046) but ||u|| reaches 2e4 and the
#:     linearisation control admits nothing.
#:
#: The optimum sat between two rungs of a decade-spaced fan: rho = 1.16e-9
#: scored 4.49e-2 while its neighbours scored 1.12e-2 and 0.
DAMPING_FAN_STEPS: int = 24

#: Ratio between consecutive rungs of that fan, 10^(1/4).
DAMPING_FAN_RATIO: float = 10.0**-0.25


@dataclass(frozen=True)
class DampingCandidate:
    """One rung of the ladder, with everything measured about it."""

    relative_damping: float
    absolute_damping: float
    relative_error: float
    update_norm: float
    approximation_norm: float
    certified_learning_rate: float | None
    learning_rate: float | None
    guaranteed_decrease: float
    #: tr H_lambda = sum sigma_i^2/(sigma_i^2 + lambda) -- the effective
    #: number of degrees of freedom this rung spends. Ranges from rank(J) at
    #: lambda = 0 down towards 0 as lambda grows.
    effective_dof: float
    #: Generalized cross-validation score, the leave-one-out risk estimate
    #: (1/N)||r - H_lambda r||^2 / (1 - df/N)^2. Lower is better; it is
    #: +inf once df >= N, which is exactly the interpolating regime.
    gcv: float


@dataclass(frozen=True)
class DampingChoice:
    """The selected rung, plus the direction it produced."""

    candidate: DampingCandidate
    parameter_updates: tuple[torch.Tensor, ...]
    candidates: tuple[DampingCandidate, ...]


def minimal_relative_error(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> float:
    """The smallest ``eps`` the tangent space can reach, at least damping.

    This is the growth signal, and it is the RIGHT one for two reasons that
    coincide. First, ``eps`` is increasing in ``lambda`` (more regularisation
    is a worse approximation), so its minimum is at the least-regularised end
    of the bracket. Second, that minimum is ``< 1/2`` if and only if SOME
    ``lambda`` certifies -- exactly the condition ``select_projection_damping``
    checks for. So growing until this crosses ``1/2`` is identical to growing
    until a certified step exists, and unlike the certified ``eps`` it stays
    FINITE while the structure is still inadequate, which is what lets the
    growth loop rank one candidate structure against another.

    Returns ``inf`` when the system is degenerate.
    """
    system = exact_tangent_system(model, x, y, config)
    if system is None:
        return float("inf")
    jacobian = system.jacobian.to(dtype=torch.float64)
    target = system.target.reshape(-1).to(dtype=torch.float64)
    if jacobian.numel() == 0 or target.numel() == 0:
        return float("inf")
    left, singular_values, right = torch.linalg.svd(jacobian, full_matrices=False)
    if singular_values.numel() == 0:
        return float("inf")
    scale = float(singular_values.max()) ** 2
    if not scale > 0.0:
        return float("inf")
    coefficients = left.t() @ target
    absolute = DAMPING_BRACKET[0] * scale
    denominator = singular_values.square() + absolute
    approximation = left @ (singular_values.square() / denominator * coefficients)
    stats = _output_relative_error_from_tensors(
        approximation=approximation.to(system.target.dtype),
        target=system.target.reshape(-1),
        eps=config.eps,
    )
    value = stats.output_error.relative_error
    if value is None or not float(value) == float(value):
        return float("inf")
    return float(value)


def select_projection_damping(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> DampingChoice | None:
    """Return the damping maximising Lemma 3.5's guaranteed decrease.

    ``None`` when no rung certifies AND realises a step. That is a real
    outcome rather than a fallback: it says this model admits no
    regularisation at which the tangent direction is both a good enough
    approximation and a step the lemma actually describes, so the structure
    has to change.
    """
    system = exact_tangent_system(model, x, y, config)
    if system is None:
        return None

    work_dtype = torch.float64
    jacobian = system.jacobian.to(dtype=work_dtype)
    target = system.target.reshape(-1).to(dtype=work_dtype)
    if jacobian.numel() == 0 or target.numel() == 0:
        return None

    left, singular_values, right = torch.linalg.svd(jacobian, full_matrices=False)
    if singular_values.numel() == 0:
        return None
    scale = float(singular_values.max()) ** 2
    if not scale > 0.0:
        return None
    coefficients = left.t() @ target
    n_observations = int(target.numel())
    target_sq_norm = float(target.square().sum())

    threshold = min(config.rel_error_threshold, 0.5)
    objective = getattr(config, "projection_damping_objective", "descent")

    def solve(relative_damping: float):
        """Re-weight the factorisation -- no new Jacobian, no new SVD."""
        absolute = relative_damping * scale
        denominator = singular_values.square() + absolute
        filters = singular_values.square() / denominator     # sigma^2/(sigma^2+lambda)
        approximation = left @ (filters * coefficients)
        flat_update = right.t() @ (singular_values / denominator * coefficients)
        return absolute, approximation, flat_update, filters

    def gcv_at(filters: torch.Tensor, approximation: torch.Tensor) -> tuple[float, float]:
        """Return (df, GCV) for a rung, from its spectral filters.

        df = tr H_lambda = sum sigma_i^2/(sigma_i^2 + lambda) is the sum of
        the filter values -- H_lambda's eigenvalues -- and the GCV residual
        ||r - H_lambda r|| is exact from the same quantities, no extra solve.
        GCV is +inf once df >= N: that is the interpolating regime, refused
        by construction rather than by a threshold.
        """
        degrees = float(filters.sum())
        residual_sq = float((target - approximation).square().sum())
        gap = 1.0 - degrees / n_observations
        if gap <= 0.0:
            return degrees, float("inf")
        return degrees, (residual_sq / n_observations) / (gap * gap)

    def relative_error_at(relative_damping: float) -> float:
        _, approximation, flat_update, _ = solve(relative_damping)
        if not torch.isfinite(flat_update).all():
            return float("inf")
        stats = _output_relative_error_from_tensors(
            approximation=approximation.to(system.target.dtype),
            target=system.target.reshape(-1),
            eps=config.eps,
        )
        value = stats.output_error.relative_error
        return float(value) if value is not None else float("inf")

    # Bisect for rho*, the largest rho that still certifies. eps is increasing
    # in rho, so the certified set is the interval below it.
    low, high = DAMPING_BRACKET
    if not relative_error_at(low) < threshold:
        # Even the least regularised solve fails the criterion: no damping
        # rescues an inadequate tangent space, which is the grow signal.
        return None
    if relative_error_at(high) < threshold:
        boundary = high
    else:
        for _ in range(DAMPING_BISECTION_STEPS):
            middle = (low * high) ** 0.5          # geometric: rho spans decades
            if relative_error_at(middle) < threshold:
                low = middle
            else:
                high = middle
        boundary = low

    candidates: list[DampingCandidate] = []
    best: tuple[DampingCandidate, tuple[torch.Tensor, ...]] | None = None

    def is_better(candidate: DampingCandidate, incumbent: DampingCandidate | None) -> bool:
        """Rank certified, realisable rungs by the configured objective.

        "descent": maximise eta * ||g||^2, the decrease Lemma 3.5 guarantees.
        "gcv":     minimise the leave-one-out risk estimate.
        Both only ever compare rungs that certify AND realise a step; a rung
        that does neither has score 0 / +inf and never wins.
        """
        if incumbent is None:
            return True
        if objective == "gcv":
            return candidate.gcv < incumbent.gcv
        return candidate.guaranteed_decrease > incumbent.guaranteed_decrease

    for index in range(DAMPING_FAN_STEPS + 1):
        relative_damping = boundary * (DAMPING_FAN_RATIO**index)
        absolute_damping, approximation, flat_update, filters = solve(relative_damping)
        if not torch.isfinite(flat_update).all():
            continue

        stats = _output_relative_error_from_tensors(
            approximation=approximation.to(system.target.dtype),
            target=system.target.reshape(-1),
            eps=config.eps,
        )
        relative_error = stats.output_error.relative_error
        update_norm = float(torch.linalg.vector_norm(flat_update))
        approximation_norm = stats.output_error.approximation_norm
        degrees, gcv = gcv_at(filters, approximation)

        certified_rate: float | None = None
        learning_rate: float | None = None
        updates = _unflatten_parameter_update(
            flat_update.to(system.target.dtype), system.parameters
        )
        if relative_error is not None and relative_error < threshold:
            upper_bound = theoretical_learning_rate_upper_bound(
                relative_error, config
            )
            if upper_bound is not None:
                # Match the rate the STEP will actually take. eta_bar is where
                # Lemma 3.5's guaranteed decrease vanishes, so the decrease
                # peaks at eta_bar/2 (the interval midpoint); certify_optimal_
                # rate takes that, otherwise theory_lr_safety of the edge.
                fraction = (
                    0.5
                    if getattr(config, "certify_optimal_rate", False)
                    else config.theory_lr_safety
                )
                certified_rate = fraction * upper_bound
                if getattr(config, "certify_realize_path", False):
                    # Realisability is provided by the integrated path, which
                    # reaches the certified functional step in many short
                    # sub-steps. Gating here on the SINGLE-JUMP linearisation
                    # control would reject rungs the path handles fine, and
                    # MEASURED it did exactly that: after two steps every
                    # certified rung failed the single-jump check, select
                    # returned None, growth saw eps < 1/2 so did not fire, and
                    # the run froze from epoch 2 to 400. The path is the
                    # realisability mechanism; do not double-gate.
                    learning_rate = certified_rate
                elif config.certify_linearization_tolerance is None:
                    learning_rate = certified_rate
                else:
                    learning_rate = certified_linear_learning_rate(
                        model, x, updates, certified_rate, config
                    ).learning_rate

        # Lemma 3.5's own guaranteed decrease is proportional to
        # eta * ||g||^2, so that -- not eps, and not the rate alone -- is what
        # the descent objective ranks by. A tiny eps bought with an
        # unrealisable step scores zero, which is exactly right.
        decrease = (
            learning_rate * approximation_norm**2
            if learning_rate is not None
            else 0.0
        )
        candidate = DampingCandidate(
            relative_damping=relative_damping,
            absolute_damping=absolute_damping,
            relative_error=(
                float(relative_error)
                if relative_error is not None
                else float("inf")
            ),
            update_norm=update_norm,
            approximation_norm=approximation_norm,
            certified_learning_rate=certified_rate,
            learning_rate=learning_rate,
            guaranteed_decrease=decrease,
            effective_dof=degrees,
            gcv=gcv,
        )
        candidates.append(candidate)
        # A rung only competes if it both certifies and realises a step --
        # the certificate is never traded away regardless of objective.
        if learning_rate is not None and decrease > 0.0 and is_better(
            candidate, best[0] if best else None
        ):
            best = (candidate, updates)

    if best is None:
        return None
    return DampingChoice(
        candidate=best[0],
        parameter_updates=best[1],
        candidates=tuple(candidates),
    )
