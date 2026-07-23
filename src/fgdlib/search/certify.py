"""Grow until Lemma 3.5 holds -- by construction, not by luck.

The ordinary flow grows *where it is cheapest* and steps *when it can*. This
module inverts that: it grows until the structure **provably satisfies the
certificate**, and only then is a step taken. The research question it serves
is whether enforcing the FGD conditions exactly -- never approximated, never
bypassed -- reaches a global optimum in loss and maximum accuracy.

Why this terminates (the theorem, pinned in
``tests/test_grow_to_certify_theorem.py``):

* From the bridge identity ``||r||^2 = ||g||^2 (1 + eps^2)`` with
  ``g = P_T(r)``, the condition is exactly

      eps < 1/2   <=>   ||P_T(r)||^2 > 0.8 ||r||^2

  "the tangent space captures more than 80 % of the gradient energy".
* Function-preserving growth leaves ``f`` **identical** yet strictly enlarges
  ``T = range(J)``: a new neuron enters with outgoing weight ``omega = 0`` so
  it contributes nothing to ``f``, but ``df/domega != 0`` is a genuinely new
  direction. Measured: ``f`` unchanged to 1.8e-07 while ``rank(J)`` rose
  57 -> 66.
* Hence ``r`` is fixed while ``T`` grows, so ``||P_T(r)||`` increases
  strictly and **eps falls with no training step at all** -- measured
  1.883 -> 1.732 -> 1.713 -> 1.674 -> 1.615.
* The residual ``rho = r - P_T(r)`` lives in a finite-dimensional space
  (``N*K``), so finitely many added directions drive it to zero.

Therefore the loop below crosses ``1/2`` in finitely many growths. The
``max_growths`` argument is a **safety valve against numerical pathology**,
not a budget: the theory says the loop terminates on its own.

Exactness over cost, deliberately: the location to grow is chosen by
measuring the resulting ``eps`` EXACTLY on each candidate (a clone grown
function-preservingly, scored with the full-Jacobian solver), rather than
ranked by a cheaper surrogate score. That is the globally best growth the
architecture can make at this point, established by measurement.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from fgdlib.search.growth import grow_layer
from fgdlib.tangent import (
    FGDApproxConfig,
    _compute_exact_tangent_projection_step,
    tiny_optimal_update_kwargs,
)

__all__ = [
    "CertifyResult",
    "exact_relative_error",
    "grow_until_certified",
]


@dataclass(frozen=True)
class CertifyResult:
    """Outcome of the grow-to-certify loop."""

    relative_error: float
    growths: int
    certified: bool
    trajectory: tuple[float, ...]


def exact_relative_error(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> float:
    """``eps`` from the FULL Jacobian -- no CG, no surrogate.

    Returns ``inf`` when the projection is degenerate, which the caller must
    read as "nothing of ``r`` is representable yet".
    """
    step = _compute_exact_tangent_projection_step(
        model=model, x=x, y=y, config=config
    )
    epsilon = step.output_error.relative_error
    if epsilon is None or not float(epsilon) == float(epsilon):  # NaN guard
        return float("inf")
    return float(epsilon)


def _grow_clone(
    model,
    train_loader,
    layer_index: int,
    device: torch.device,
    config,
    function_preserving: bool,
):
    """A copy of ``model`` grown at ``layer_index``.

    ``function_preserving`` is the central trade-off of this method, and it is
    a measured one:

    | growth          | tangent directions gained | parameters added |
    |-----------------|---------------------------|------------------|
    | preserving      | +9                        | 42               |
    | non-preserving  | +42                       | 42               |

    With ``omega = 0`` the incoming weights contribute nothing to the
    Jacobian (``df/dalpha = 0``), so only the outgoing weights add directions
    -- about a fifth of the parameters spent. Releasing ``omega`` makes every
    added parameter contribute an independent direction, the theoretical
    maximum, so the rank needed to certify is reached far sooner.

    What is given up is the monotonicity proof: a non-preserving growth moves
    ``f``, so ``r`` moves with it and ``eps`` may rise after a growth. The
    loop then relies on measurement rather than a theorem, which is why the
    preserving route stays the default.

    Returns ``None`` when the growth cannot be applied, so a failed candidate
    drops out of the comparison instead of aborting the search.
    """
    clone = copy.deepcopy(model)
    try:
        grow_layer(
            model=clone,
            train_loader=train_loader,
            layer_index=layer_index,
            device=device,
            line_search_config=config.scaling_line_search,
            optimal_update_kwargs=tiny_optimal_update_kwargs(
                config.fgd_approx,
                compute_delta=config.fgd_approx.growth_compute_delta,
            ),
            progress=None,
            function_preserving=function_preserving,
            preservation_tolerance=(
                config.fgd_approx.growth_preservation_tolerance
            ),
        )
    except RuntimeError:
        return None
    return clone


def grow_until_certified(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    train_loader,
    device: torch.device,
    config,
    max_growths: int = 64,
    function_preserving: bool = True,
    force: bool = False,
    progress=None,
):
    """Grow until ``eps < rel_error_threshold``; return the grown model.

    At each iteration every growable location is tried on a clone and scored
    by its EXACT resulting ``eps``; the location with the lowest ``eps`` is
    committed.

    ``force`` grows at least once even when ``eps`` is already below the
    threshold. It exists to close a real deadlock, measured on MNIST: after
    certification the flow sat at ``eps = 0.475`` for epoch after epoch with
    the loss frozen at 0.0619, because

    * ``eps < 1/2`` said the structure was adequate, so no growth fired, and
    * no admissible learning rate produced held-out descent, so no step
      committed.

    Neither mechanism could act and nothing changed, ever. The resolution is
    the distinction the method already rests on: ``eps < 1/2`` is Lemma 3.5's
    admissibility of a STEP -- it certifies that an admissible rate *exists*
    in the worst-case bound -- while the realised descent is a **separate**
    certificate condition. A step that fails the descent condition despite
    ``eps < 1/2`` has shown that the structure did not deliver, and growth,
    not more of the same step, is the answer. That is exactly R1's reasoning,
    applied here.

    Returns ``(model, CertifyResult)``.
    """
    threshold = config.fgd_approx.rel_error_threshold
    epsilon = exact_relative_error(model, x, y, config.fgd_approx)
    trajectory = [epsilon]
    growths = 0
    forced_remaining = 1 if force else 0

    while (
        epsilon >= threshold or forced_remaining > 0
    ) and growths < max_growths:
        forced_remaining = max(0, forced_remaining - 1)
        locations = range(len(getattr(model, "_growable_layers", [])))
        best_model = None
        # Preserving growth cannot make eps worse, so requiring an improvement
        # is free there. Non-preserving growth moves f, so the best available
        # candidate may still sit above the current eps; accepting it is what
        # lets the rank keep climbing, and the loop then relies on the measured
        # trajectory rather than the monotonicity theorem.
        best_epsilon = epsilon if function_preserving else float("inf")
        best_location = None
        for location in locations:
            candidate = _grow_clone(
                model, train_loader, location, device, config,
                function_preserving,
            )
            if candidate is None:
                continue
            candidate_epsilon = exact_relative_error(
                candidate, x, y, config.fgd_approx
            )
            if candidate_epsilon < best_epsilon:
                best_epsilon = candidate_epsilon
                best_model = candidate
                best_location = location

        if best_model is None:
            # No growable location reduces eps. The theorem says this cannot
            # persist while the residual is non-zero, so reaching here means
            # the architecture cannot add a direction along rho at all --
            # report honestly rather than loop.
            if progress is not None:
                progress(
                    f"[CERTIFY] no growth reduced eps ({epsilon:.4f}); "
                    "the structure cannot add a direction along the residual"
                )
            break

        model = best_model
        epsilon = best_epsilon
        growths += 1
        trajectory.append(epsilon)
        if progress is not None:
            progress(
                f"[CERTIFY] growth {growths} at location {best_location}: "
                f"eps -> {epsilon:.4f}"
                + ("  (certified)" if epsilon < threshold else "")
            )

    return model, CertifyResult(
        relative_error=epsilon,
        growths=growths,
        certified=epsilon < threshold,
        trajectory=tuple(trajectory),
    )
