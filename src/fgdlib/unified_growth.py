"""One criterion for width AND depth, in the certified currency of Lemma 3.5.

The design constraint is that every structural step -- including a change in
the NUMBER OF LAYERS -- must satisfy the theory the flow already certifies,
not a separate policy bolted on beside it.

Two properties make that possible.

**1. Growth is a pure representation refinement.** Both candidate kinds are
applied *function-preservingly*: widening uses GroMo's zero-outgoing-weight
extension, depth uses the identity-homotopy insertion in `fgdlib.depth`.
The represented ``f`` is therefore unchanged by the structural step, so

* the loss is unchanged at the step, hence Proposition 3.8's held-out
  descent certificate can never be violated *by growing*; descent remains
  entirely the business of the certified families, and
* the only thing that changes is ``range(J)``, the reachable set -- which is
  precisely the object Lemma 3.5 measures.

This is the clean division the theory wants: **training descends, growth
enlarges what can be descended along.**

**2. Both kinds are scored in the same certified quantity.** By the bridge
identity in `fgdlib.senn`,

    N * eta = ||P(r)||^2 = ||r||^2 / (1 + eps^2),

so a candidate's value is the increase in expressible gradient energy

    Delta(N eta) = ||r||^2 * [ 1/(1 + eps_after^2) - 1/(1 + eps_before^2) ]

read straight off the tangent certificate before and after. Because ``f`` is
unchanged, ``r`` and ``||r||^2`` are identical on both sides and the whole
difference is attributable to the enlarged reachable set. Dividing by the
candidate's parameter cost gives SENN's natural expansion score per
parameter, expressed in our own units -- and it is indifferent to whether
the proposal is a neuron or a layer, which is exactly what lets depth join
the search instead of needing a policy of its own.

Nothing here introduces a threshold or a budget: the ranking decides *what*
to buy, Lemma 3.5 (``eps < 1/2``) decides when to stop buying, and R1
decides when to start.
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = ["Candidate", "expansion_value", "rank_candidates"]


def rank_limiting_locations(widths: list[int]) -> list[int]:
    """Growable locations whose width caps the reachable set's dimension.

    For a composition ``W_L sigma ... sigma W_1`` the tangent image obeys

        rank J <= min_l w_l ,

    so while some layer sits at that minimum, **no purchase anywhere else
    can raise the dimension of what the structure can express**. Buying
    width in an already-wide layer refines within a subspace whose dimension
    is pinned by the narrow one.

    This is why a pure value-per-parameter ranking starves the input
    projection, and it is not a heuristic to be traded off: it is what
    Lemma 3.5's quantity is *about*. eps measures how well the reachable set
    expresses r; if the set's dimension is capped, eps is capped with it.
    Measured, the failure is stark -- ranking by value per parameter bought
    the three cheapest locations for eps reductions of 0.001 to 0.004 each
    and then stalled with eps at 1.87, structure 784->2->3->4, because every
    cheap purchase was refining inside a rank-2 image.

    Returns every index attaining the minimum, so the caller may still rank
    among them by value per parameter -- the constraint says *where the
    dimension can be lifted*, not which of those to buy.
    """
    if not widths:
        return []
    narrowest = min(widths)
    return [index for index, width in enumerate(widths) if width == narrowest]


def bottleneck_relief_target(widths: list[int]) -> tuple[int, int] | None:
    """The location the rank cap mandates buying, and how far.

    Returns ``(index, target_width)`` when exactly one location attains the
    minimum width, else ``None``.

    The *how far* is derived, not chosen. While ``l*`` is the **unique**
    minimum, ``rank J <= w_l*`` and that location alone caps the reachable
    set's dimension, so purchases there are mandated. The moment the minimum
    becomes **shared**, relieving ``l*`` on its own no longer lifts the cap,
    because the other location at the minimum still pins it. "Unique
    minimum" is therefore exactly the condition under which single-location
    buying is mandated, and levelling to the second-smallest width is where
    that mandate ends.

    Nothing here is a budget: the amount is read off the current widths, so
    nothing is presumed about a dataset that has never been trained on. It
    is the same rank inequality already used to decide *where*, read once
    more to decide *how much*.

    This is what fixes the pace. Buying one neuron per growth event made
    each purchase wait for R1 to fire again: measured on MNIST from 3x2, 20
    events across 44 epochs, with accuracy at 72.5 % by epoch 30 where the
    incumbent reached 88.7 % by epoch 14. Relieving to the level the mandate
    covers collapses that into a single event without loosening any
    criterion.
    """
    if len(widths) < 2:
        return None
    ordered = sorted(widths)
    if ordered[0] == ordered[1]:
        return None                      # the minimum is shared: no mandate
    return widths.index(ordered[0]), ordered[1]


@dataclass(frozen=True)
class Candidate:
    """A structural proposal, priced and scored.

    ``kind`` is ``"width"`` or ``"depth"``; ``index`` is the growable layer
    to widen or the position to insert at. ``cost`` is the parameter count
    the proposal adds, ``relative_error_after`` the Lemma-3.5 relative error
    the enlarged structure achieves on the same held-out probe.
    """

    kind: str
    index: int
    cost: int
    relative_error_after: float | None
    # True when this purchase raises min_l w_l, i.e. lifts the cap on
    # rank J and therefore on what eps can ever reach. See
    # rank_limiting_locations.
    relieves_rank_ceiling: bool = False


def expansion_value(
    *,
    relative_error_before: float | None,
    relative_error_after: float | None,
    gradient_sq_norm: float | None,
) -> float:
    """``Delta(N eta)``: the extra gradient energy the structure can express.

    Returns 0.0 when any input is missing or the candidate did not enlarge
    what is expressible, so an unmeasurable proposal can never win a
    ranking.
    """
    if (
        relative_error_before is None
        or relative_error_after is None
        or gradient_sq_norm is None
        or gradient_sq_norm <= 0.0
    ):
        return 0.0
    before = gradient_sq_norm / (1.0 + relative_error_before**2)
    after = gradient_sq_norm / (1.0 + relative_error_after**2)
    return max(after - before, 0.0)


def rank_candidates(
    candidates: list[Candidate],
    *,
    relative_error_before: float | None,
    gradient_sq_norm: float | None,
    statistical_threshold: float = 1e-3,
    rank_ceiling_binds: bool = False,
) -> list[Candidate]:
    """Return the candidates worth buying, best value-per-parameter first.

    Ranking is by ``Delta(N eta) / cost``. Admission reuses GroMo's own
    relative rule -- keep everything at or above
    ``min(statistical_threshold, best)``, which always keeps at least the
    best candidate -- so no new constant is introduced and the comparison is
    scale-free. A candidate that does not enlarge the reachable set scores
    zero and is never admitted; if none does, the returned list is empty and
    the structure is already minimal-adequate for this step (R3).
    """
    # While Lemma 3.5 is unsatisfied and some location caps the rank, only
    # purchases that lift the cap can change what eps is able to reach.
    # Ranking among the rest is refining inside a pinned subspace.
    binding = [c for c in candidates if c.relieves_rank_ceiling]
    if rank_ceiling_binds and binding:
        candidates = binding

    scored: list[tuple[float, Candidate]] = []
    for candidate in candidates:
        value = expansion_value(
            relative_error_before=relative_error_before,
            relative_error_after=candidate.relative_error_after,
            gradient_sq_norm=gradient_sq_norm,
        )
        if value <= 0.0:
            continue
        scored.append((value / max(candidate.cost, 1), candidate))

    if not scored:
        # Nothing measurably enlarged the reachable set *immediately*. That
        # is not the same as nothing being worth buying: a
        # function-preserving extension adds directions whose value only
        # materialises once they are trained, so the immediate eps is blind
        # to them -- measured earlier for width, where immediate eps ranked
        # every candidate as worse while look-ahead eps discriminated
        # cleanly. Falling straight through to "terminate" on that evidence
        # is what left the structure at 784->2->3->4 with eps at 1.87.
        #
        # When the rank ceiling is the binding constraint the theory still
        # dictates the move regardless of the immediate reading, so the
        # caller is handed the cheapest bottleneck relief instead of a
        # termination.
        fallback = [
            candidate
            for candidate in candidates
            if candidate.kind == "width" and candidate.relieves_rank_ceiling
        ]
        if fallback:
            return [min(fallback, key=lambda item: (item.cost, item.index))]
        return []

    best = max(value for value, _ in scored)
    reference = min(statistical_threshold, best)
    admitted = [item for item in scored if item[0] >= reference]
    admitted.sort(key=lambda item: (-item[0], item[1].kind, item[1].index))
    return [candidate for _, candidate in admitted]
