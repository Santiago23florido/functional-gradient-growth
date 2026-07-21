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
        return []

    best = max(value for value, _ in scored)
    reference = min(statistical_threshold, best)
    admitted = [item for item in scored if item[0] >= reference]
    admitted.sort(key=lambda item: (-item[0], item[1].kind, item[1].index))
    return [candidate for _, candidate in admitted]
