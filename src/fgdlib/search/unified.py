"""One criterion for width AND depth, in the certified currency of Lemma 3.5.

The design constraint is that every structural step -- including a change in
the NUMBER OF LAYERS -- must satisfy the theory the flow already certifies,
not a separate policy bolted on beside it.

Two properties make that possible.

**1. Growth is a pure representation refinement.** Both candidate kinds are
applied *function-preservingly*: widening uses GroMo's zero-outgoing-weight
extension, depth uses the identity-homotopy insertion in `fgdlib.search.depth`.
The represented ``f`` is therefore unchanged by the structural step, so

* the loss is unchanged at the step, hence Proposition 3.8's held-out
  descent certificate can never be violated *by growing*; descent remains
  entirely the business of the certified families, and
* the only thing that changes is ``range(J)``, the reachable set -- which is
  precisely the object Lemma 3.5 measures.

This is the clean division the theory wants: **training descends, growth
enlarges what can be descended along.**

**2. Both kinds are scored in the same certified quantity.** By the bridge
identity in `fgdlib.search.senn`,

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

import math
from dataclasses import dataclass

__all__ = [
    "Candidate",
    "expansion_value",
    "rank_candidates",
    "rank_candidates_by_certified_gain",
]


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
    # The layers this proposal widens, when it widens more than one. Empty
    # means the single layer named by ``index``, which is every proposal the
    # shipped rule makes. A JOINT proposal exists because the value of a new
    # feature upstream is conditional on the downstream being wide enough to
    # carry it: MEASURED at widths [5, 6, 2], widening layer 0 gains +0.0080
    # against +0.0220 downstream, four times worse, because nothing can reach
    # the output through a width-2 layer. By the time the downstream is wide
    # enough to use those features, widening layer 0 costs 5 + h2 and is no
    # longer cheap -- so a funnel is unreachable one layer at a time, however
    # good a funnel might be.
    indices: tuple[int, ...] = ()


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


def rank_candidates_by_certified_gain(
    candidates: list[Candidate],
    *,
    relative_error_before: float,
    gradient_sq_norm: float | None,
    cost_exponent: float = 1.0,
    min_gain_fraction: float = 0.25,
    certificate_binds: bool,
) -> list[Candidate]:
    """Rank by measured eps reduction per parameter -- no width mandate.

    ``rank_candidates`` above levels the widths: its caller relieves the
    narrowest location first, its filter keeps only bottleneck candidates
    while the ceiling binds, and its fallback picks the cheapest of those.
    Every run therefore lands on ``h, h, h+1``. That rests on
    ``rank J <= min_l w_l`` (:53), which is FALSE on the exact Jacobian --
    MEASURED at NK=200, widths (20,3,20) and (30,4,30) both reach rank 200
    and (2,2,2) reaches 25 = P. The expressivity argument behind it is real
    (``f`` factors through the narrowest activation), but that is a claim
    about what widening the narrow layer BUYS, and buying is what the exact
    scan measures. So the narrowest layer earns preference here rather than
    receiving it a priori.

    ``relative_error_*`` must come from the TRAIN probe, the same one the
    grow loop chases and Lemma 3.5's interval is built from. There
    function-preserving growth strictly enlarges ``range(J)``, so the gain is
    non-negative by the theorem in ``certify.py``; on the validation probe it
    is not, which is why the caller used to see it clamped to zero.

    Two guards, in this order, and the order matters:

    * the RAW gain must reach ``min_gain_fraction`` of the best gain on
      offer, checked BEFORE dividing by cost. Cheap-and-nearly-useless is
      eliminated before ``1/cost`` can rescue it. MEASURED without it, 11 of
      12 consecutive purchases went to the last growable location, whose cost
      ``h2 + 2`` does not grow with its own width and which contributes a
      single tangent direction.
    * only then does ``value / cost**cost_exponent`` decide, so the cheaper of
      two genuinely comparable buys still wins. ``cost_exponent = 0`` collapses
      this to a pure argmin over eps.

    When nothing gains, the fallback returns the single candidate with the
    LOWEST measured eps (cost only as a tie-break) provided the certificate
    still binds -- growth is mandatory there, no step exists at any damping.
    It cannot return empty while a finite candidate exists, which is the
    property whose absence stalled the earlier free-shape attempt at 53
    parameters.
    """
    priced = [
        candidate
        for candidate in candidates
        if candidate.relative_error_after is not None
        and math.isfinite(candidate.relative_error_after)
    ]
    if not priced:
        return []

    gains = {
        id(candidate): relative_error_before - candidate.relative_error_after
        for candidate in priced
    }
    best_gain = max(gains.values())

    if best_gain <= 0.0:
        if not certificate_binds:
            return []
        return [
            min(
                priced,
                key=lambda c: (c.relative_error_after, c.cost, c.index),
            )
        ]

    floor = min_gain_fraction * best_gain
    admitted = [c for c in priced if gains[id(c)] > 0.0 and gains[id(c)] >= floor]
    if not admitted:
        admitted = [c for c in priced if gains[id(c)] == best_gain]

    def _score(candidate: Candidate) -> float:
        value = expansion_value(
            relative_error_before=relative_error_before,
            relative_error_after=candidate.relative_error_after,
            gradient_sq_norm=gradient_sq_norm,
        )
        if value <= 0.0:
            # gradient_sq_norm is a common positive factor, so falling back to
            # the raw gain preserves the order when it is unavailable.
            value = gains[id(candidate)]
        return value / max(float(candidate.cost), 1.0) ** cost_exponent

    admitted.sort(key=lambda c: (-_score(c), c.kind, c.index))
    return admitted
