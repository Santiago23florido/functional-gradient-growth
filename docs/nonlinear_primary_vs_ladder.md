# The nonlinear primary family vs. the ladder's nonlinear family

The goal of this branch is to replace the family ladder's expensive tangent
path — full Jacobians and tangent projection solves — with a nonlinear primary
family that builds neither, while reproducing the behaviour of the nonlinear
family the ladder already falls back to.

Under `family_order: [nonlinear]` the pipeline instead grew every epoch with
the loss frozen. This documents why, what was actually wrong, and what changed.

## Reference paths

| | Ladder nonlinear family | Nonlinear primary |
|---|---|---|
| generation | `certify_parametric_step` / `_swept` (`fgdlib/search/families.py`) | `train_nonlinear_candidate` (`fgdlib/search/nonlinear.py`) |
| certification | same function, inline | `stream_nonlinear_certificate` |
| step | `certify_parametric_step` tail | `search_interpolated_step` |
| driver | `grow_until_certified` (`fgdlib/search/certify.py`) | `_search_nonlinear_primary_candidate` (`stable_tiny/pipeline.py`) |

## Divergences found

Measured against `configs/fgd/family_ladder_N1024.yaml`
(`certify_family_*`) and `configs/fgd/nonlinear_family_ladder_N1024.yaml`.

| # | Aspect | Ladder | Nonlinear-only (before) | Impact |
|---|---|---|---|---|
| 1 | training data | `fgd_train_probe`, full batch, 1024 examples | `train_loader` minibatches of 64 | large |
| 2 | certification data | the **same** train probe | `validation_loader` | changes the guarantee |
| 3 | `inner_steps` unit | full-batch steps on the probe | minibatch optimizer steps | **64× budget gap** |
| 4 | examples per candidate | 400 × 1024 = 409,600 | 256 × 64 = 16,384 | **25× fewer** |
| 5 | objective | `((clone(x)-target)**2).sum()` | `torch.mean(...)` | rescales decay/penalty |
| 6 | clone mode | `clone.train()` | `candidate.eval()` | only with dropout/BN |
| 7 | AdamW lr | 0.01 | 0.003 | slower |
| 8 | weight decay | torch default 0.01 | 0.05 | — (measured negligible) |
| 9 | parameter penalty | none | 1e-6 | small |
| 10 | gradient clipping | none | 1.0 | small |
| 11 | `functional_learning_rates` | `[1, 2, 4, 8]` | `[0.5]` | different target |
| 12 | target at η_f=0.5 | — | exactly `y` (plain regression) | wrong family's semantics |
| 13 | plateau criterion | on `cos(Δ, r)`, cuts only doomed clones | on epoch mean objective | cuts climbing clones |
| 14 | committed step | rescale θ by the Lemma 3.5 rate | identical rescale | **certificate/step mismatch** |
| 15 | recertification of the applied step | none | none | **unsound** |
| 16 | growth gate | `_growth_reduces_lookahead_epsilon` | unconditional argmax bottleneck | grows more |

The source of items 7–12 is recorded in the config's own header: the AdamW
values were copied from the repository's **`parametric_descent`** family, which
accepts on measured descent with `min_cosine: 0.0`. Those values were never
tuned for a near-gradient certificate, and `η_f = 0.5` makes the functional
target exactly `y` rather than the ladder's `f − η_f r` overshoot.

## Root cause

Two independent defects, in this order of effect.

### 1. Budget starvation (the observed symptom)

`inner_steps` meant full-batch probe steps on one side and minibatch steps on
the other. The measured consequence, from `results/nonlinear_n1024_3seeds`:

```
[NONLINEAR] eta_f=0.5, inner_steps=16,  cos=0.3060
[NONLINEAR] eta_f=0.5, inner_steps=64,  cos=0.4804
[NONLINEAR] eta_f=0.5, inner_steps=256, cos=0.6527
```

The cosine rises monotonically with the budget and never plateaus — every
candidate was cut off mid-optimization, and certification needs
`cos > √3/2 ≈ 0.866`. Seeds 0 and 1 accepted **zero** steps in 25 epochs and
grew 25 times with the loss constant to four decimals; seed 2, which started
from a higher loss, accepted 24.

At ladder parity the same structures reach `cos 0.96–0.97`.

### 2. The certificate did not describe the applied step

The certificate was earned by the full candidate `θ'`, and the model committed
was `θ + α(θ' − θ)`. For a nonlinear network

```
f_θ − f_{θ+αd}  ≠  α (f_θ − f_{θ+d})
```

so the certificate belonged to a displacement that was never applied. As
`α → 0` the interpolation collapses onto `α J_θ d` — the tangent direction the
nonlinear family exists to avoid.

Re-measuring the interpolations makes the size of the error explicit:

```
alpha    1.0     0.5     0.25    0.125
cos      0.969   0.640   0.523   0.476
```

The direction is destroyed long before the distance becomes admissible. This
defect is inherited from `certify_parametric_step`, where it is masked because
the nonlinear family is only a fallback.

### 3. What parity actually revealed

With the budget fixed, every candidate certifies its **direction** and every
one is rejected on **distance**:

```
eta_f     1      2      4      8
cos     0.969  0.964  0.974  0.975
eps     0.272  0.266  0.227  0.224
eta*    0.929  1.861  3.815  7.494     bound ~0.32
```

A well-fitted clone realises `η* ≈ η_f`, while Lemma 3.5 admits only
`η ≤ 2(1−2ε)/(L_s(1+2ε))`. The ladder's own functional-rate sweep contains **no
admissible step size**; it only appeared to work because it shortened the step
in parameter space without re-measuring what that did to the direction.

## Why it certifies inside the ladder but not standalone

This is the single largest difference, and it is not about the candidate at
all — it is about the **bar**. The two paths were never asking the same
question.

**The ladder** (`certify_parametric_step`, families.py):

```python
certified = cosine > 0.0 and relative_error < threshold   # and nothing else
```

`certify_family_lemma35_rate` defaults to `False` and
`configs/fgd/family_ladder_N1024.yaml` does not set it, so no rescale happens.
`grow_until_certified` then does `model = stepped` directly — the family step
never passes through `_certify_fgd_candidate`. The ladder's only additional
gate is `certify_family_min_gain` on how far `eps` moved.

So the ladder accepts on **one condition: `eps < 1/2`**. It makes no statement
about the step's LENGTH.

**The nonlinear primary** required `eps < 1/2`, *plus* the full transactional
conditions on validation (this predates the branch), *plus* — after the fix —
the Lemma 3.5 interval on the realized `eta*`.

And the old code only *appeared* to enforce that last one:

```python
rate = family_lemma35_rate(stats.relative_error, config.fgd_approx)
last_certificate = _nonlinear_directional_certificate(learning_rate=rate, ...)
```

`family_lemma35_rate(eps)` returns a rate **admissible by construction**, so
`learning_rate_interval_valid` was always `True`. The check tested nothing,
while the committed displacement had a real `eta*` that was never measured.
Substituting the measured `eta*` turned a check that always passed into one
that rejects — which is why parity first looked *worse*, not better.

`acceptance_rule` now names the bar explicitly:

| value | direction | measured descent | realized distance |
|---|---|---|---|
| `direction_only` | yes | no | no | ← the ladder verbatim |
| `measured_descent` | yes | yes | no |
| `theory_interval` | yes | yes | yes |

The step-recertification fix is **independent of this choice** and stays on for
all three: with `alpha = 1` the committed model *is* the certified one, so the
ladder's bar is met without the unsound parameter rescale.

### The ladder's rule diverges as a PRIMARY family

Running `acceptance_rule: direction_only` — the ladder's rule verbatim — on
N=1024 (`results/ladderrule_divergence_seed0.log`):

```
epoch 7   eps 0.4744   train_loss 0.1709   accepted
epoch 8   eps 0.4274   train_loss 0.1532   accepted
epoch 9   eps 0.4053   train_loss 27.6955  accepted   <-- x180
epoch 13  eps 0.2383   train_loss 13.6936
epoch 18  eps 0.0964   train_loss 11.1519
```

At epoch 9 it accepted a step with a *good* direction (`cos ≈ 0.91`) and the
loss exploded by a factor of 180. It never recovers. Note the direction keeps
IMPROVING as the loss stays wrecked — `eps` falls to 0.096 — which is exactly
the signature of a well-aligned step that is far too long.

`tangent.py` already documents this failure for the in-band case:

> MEASURED without it, the unbounded family step diverged held-out loss
> 9.37 -> 44.53 in two epochs because it pre-empted the tangent path that
> produces the very bound it was missing.

That is the whole answer. **The ladder's direction-only rule is safe only
because the family is a FALLBACK**: the tangent path supplies the bounded steps
and the family fires occasionally between them. Promoted to primary, nothing
bounds the step length and it diverges. A nonlinear primary therefore needs a
distance criterion the ladder never had.

### `measured_descent` rejects exactly those steps — and then has nothing left

Running `acceptance_rule: measured_descent` with the ladder's `[1, 2, 4, 8]`
sweep accepts **nothing** in 25 epochs, and the reason is visible in the
measured descent of the candidates it refuses:

```
eta_f 2   cos 0.9639  eps 0.2661  eta* 1.861  descent  -1,477
eta_f 4   cos 0.9677  eps 0.2521  eta* 3.739  descent  -8,866
eta_f 8   cos 0.9573  eps 0.2892  eta* 7.253  descent -39,742
```

The direction is excellent (`eps` down to 0.224) and the step still *raises*
the loss by four orders of magnitude, because `eta*` overshoots the admissible
rate by 6–20x. These are precisely the steps `direction_only` accepted at epoch
9. So `measured_descent` is doing its job — but the ladder's sweep contains
nothing else for it to accept, which is the same conclusion the parity run
reached from the theoretical side.

The bar and the sweep have to change together: `measured_descent` needs the
descending sweep and the adaptive retries to have short candidates available.

## The correction

Four quantities that were previously conflated under `rate` are now distinct:

| symbol | meaning | field |
|---|---|---|
| `η_f` | rate generating the target `f − η_f r` | `functional_target_rate` |
| `η*` | `⟨Δ, r⟩ / ‖r‖²`, the scale the candidate **realized** | `effective_secant_rate` |
| `α` | parameter-space interpolation | `committed_alpha` |
| `η̄` | Lemma 3.5 admissible bound at the measured `ε` | certificate bound |

The step rule is now self-consistent:

1. Train a candidate at `η_f` with an explicitly named budget unit.
2. Measure its certificate on the configured split.
3. Re-measure **every** interpolation `α` from scratch — never inherit.
4. Check Lemma 3.5 using that interpolation's **own** `ε` and **own** `η*`.
5. Gate on the transactional conditions on the transactional split.

Because `η̄` depends on the `ε` the clone happens to reach, a fixed grid cannot
be placed in advance. `adaptive_rate_retries` runs the fixed-point iteration
`η_{k+1} = safety · η̄(ε_k)` seeded from the **most recent** measurement.
Seeding from the best `ε` overstates the bound, since the smallest `ε` belongs
to the largest rate.

## Growth

Function-preserving growth adds a neuron with zero outgoing weights, which was
a live suspect: with a zero outgoing weight no gradient reaches the unit's
incoming weights. It is **not** the cause. The gradient w.r.t. the outgoing
weight is proportional to the unit's *activation*, so the first step lifts it
off zero and the incoming weights move from the second step on
(`tests/test_nonlinear_growth_activation.py`). No degeneracy-breaking noise is
needed and none was added.

## Results (N = 1024, 2-2-2, seed 0, 25 epochs)

All four runs use the same data, seed, initial architecture and budget
(400 full-batch probe steps per candidate). They differ only in the
acceptance bar and the functional-rate sweep.

| config | rule | sweep | accepted | train loss | stable |
|---|---|---|---|---|---|
| `nonlinear_family_ladder` (before) | interval, but faked | `[0.5]` | 0 | 0.1935 → 0.1935 | frozen |
| `nonlinear_primary_parity` | theory_interval | `[1,2,4,8]` | 0 | 0.1935 → 0.1935 | frozen |
| `nonlinear_primary_ladderrule` | **direction_only** | `[1,2,4,8]` | 12 | 0.1935 → **27.70 → 11.15** | **diverges** |
| `nonlinear_primary_descent` | measured_descent | `[1,2,4,8]` | 0 | 0.1935 → 0.1935 | frozen |
| `nonlinear_primary_certified` | theory_interval | descending + adaptive | 1 | 0.1935 → 0.0691 | yes |
| `nonlinear_primary_practical` | measured_descent | descending + adaptive | **2** | 0.1935 → **0.0648** | yes |

Tangent calls are 0 in every one of them.

The two steps `practical` accepted, both at ADAPTIVELY derived rates rather
than any value in the fixed sweep:

```
epoch 15   eta_f 0.0524   cos 0.8924   eps 0.4513   eta* 0.0403   descent  +30.3
epoch 16   eta_f 0.2183   cos 0.9450   eps 0.3271   eta* 0.1927   descent +101.4
```

The ordering is the finding. Direction-only accepts the most and is the only
one that destroys the model; the two bars that examine the step's actual effect
accept far less and are the only ones that make progress. Test loss follows
train: 0.1930 → 0.0718 for `practical`, 0.1930 → 0.0758 for `certified`.

## Remaining limitations

* Progress is still sparse: 2 accepted steps in 25 epochs for `practical`, 1
  for `certified`. The binding constraint is the cosine the STRUCTURE can
  reach, not the step rule — certification needs `cos ≳ 0.95`, well above the
  `√3/2 ≈ 0.866` the direction test alone requires, because `η̄ → 0` as
  `ε → 1/2`.
* Immediately after an accepted step `eps` degrades sharply (1.0000 at epochs
  17–18 of the `practical` run) before recovering. Not investigated.
* `_growth_reduces_lookahead_epsilon` (the grow-vs-train lookahead the ladder
  uses) is still not wired into the nonlinear growth path; growth remains
  unconditional argmax-bottleneck.
* Only seed 0 was run for the new configurations. The "before" numbers come
  from the existing three-seed results, where seed 2 behaved differently
  (24 accepted steps) because it started from a higher loss.
* `direction_only` is available but should not be used as a primary family
  without the divergence caveat above.
* Three failures in `tests/test_regularized_mlp.py` predate this work and are
  unrelated (verified on a clean checkout of the branch point).
