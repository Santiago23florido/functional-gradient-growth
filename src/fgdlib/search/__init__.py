"""Certified architecture search: the contribution built on top of GroMo.

Every decision here is governed by an FGD certificate, never a tuned knob:

- ``senn``      -- SENN's natural expansion score and its exact bridge to
                   Lemma 3.5's relative error (``N*eta = ||r||^2 / (1+eps^2)``),
                   the theory that grounds *where* to grow.
- ``growth``    -- the GroMo growth step (``grow_layer``), the per-neuron
                   expansion spectrum, parameter costs, and the
                   cost-normalised allocation.
- ``unified``   -- one criterion for width AND depth: rank candidates by
                   certified expansion per parameter, with the rank ceiling
                   ``rank J <= min_l w_l`` deciding where the dimension can be
                   lifted.
- ``depth``     -- function-preserving layer insertion, so depth enters the
                   search without discarding the certified trajectory.
- ``schedule``  -- the (legacy) epoch-based growth schedule, used only by
                   ``method: normal``.
"""
