"""Training-loop utilities around the certified growth flow.

- ``loop``         -- the parametric training helpers.
- ``optim``        -- optimizer construction.
- ``lr_scheduler`` -- learning-rate schedules (with growth-aware restarts).
- ``averaging``    -- Polyak-Ruppert averaging in FUNCTION space, licensed by
                      the convexity of L in f (Jensen), to damp trajectory
                      variance at fixed architecture.
"""
