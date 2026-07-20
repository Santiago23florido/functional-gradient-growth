# Certified Uniform Growth: matching fixed-structure AdamW from a tiny network

*Research note. Method built on arXiv:2606.16926 (FGD with growing
networks); companion to `ADAMW_FGD_FAMILY.md`.*

## Problem

Train a neural network by **growing it from a minimal size** (3 hidden
layers × 2 neurons) so that it reaches the accuracy of a *fixed* network
trained with parametric AdamW — with every step justified by the FGD
theory (the growth driven by the certified criteria, not a schedule).

## The two obstacles we had to solve (both measured)

1. **The certification refuses to overfit tiny structures.** At a tiny
   network no certified family step descends the *held-out* functional loss
   (everything overfits the minimal capacity), so training alone cannot
   escape the low-capacity regime — the structure must grow.
2. **Greedy per-layer growth undervalues the input layer.** Selecting the
   growth layer by any *immediate* certified signal — functional descent OR
   post-growth relative error (Lemma 3.5) — favors the cheap late layers.
   Measured on a trained `784→2→2→2`: growing the last hidden layer cuts the
   validation functional loss most and gives the lowest post-growth rel_err,
   while the input layer's benefit is **latent** (it only pays off once the
   later layers are also wide). So greedy growth keeps layer 0 pinned at
   width 2–4, which bottlenecks the 784-dim input and caps accuracy far
   below AdamW.

## The method — Certified Uniform Growth (CUG)

Start from the minimal `3×2` network. Then repeat:

1. **Train** the current structure with the **parametric-AdamW FGD family**
   (see `ADAMW_FGD_FAMILY.md`): a candidate produced by an AdamW clone,
   committed only if it certifies a measured validation functional descent
   (Proposition 3.8) at the scale-optimal rate `η*`. This is the certified
   functional-gradient step; AdamW is merely the (admissible) generator.
2. **Detect exhaustion from the certificate, dynamically.** When no
   admissible step exists — no learning rate satisfies the validation
   conditions and the relative error is `≥ ½` (Lemma 3.5 fails: the current
   reachable set can no longer represent the functional gradient) — the
   certificate itself *requests growth*. This is the paper's structural
   trigger; it is **not** a schedule (in the runs growth fires at epochs
   3, 5, 6… while the config's `first_epoch` is 50 and is ignored).
3. **Grow uniformly.** Widen **every** hidden layer by the neuron
   increment (GroMo/TINY optimal delta, which reduces the loss). Uniform
   widening keeps the layer widths balanced, so it **sidesteps obstacle 2**:
   the input layer grows in lockstep with the rest and never becomes the
   permanent bottleneck. This traces the balanced dense networks `3×k` from
   the tiny start.

The trajectory is therefore a sequence of certified-AdamW-trained dense
networks of increasing width, each entered exactly when the previous one is
certified-exhausted. Growth stops (parameter budget, or when growth no
longer improves the certified validation loss).

## Why this is faithful to the theory

- **Training step:** admissible FGD family — a B-bounded functional-gradient
  approximation with certified measured descent (Prop. 3.8). Global
  convergence of the functional loss holds for the fixed structure.
- **Growth trigger:** the Lemma-3.5 admissibility failure (`ε ≥ ½`) is
  exactly the paper's signal that the reachable set cannot represent the
  functional gradient, i.e. that capacity must increase — evaluated
  dynamically on held-out data, never scheduled.
- **Growth operator:** the GroMo/TINY optimal neuron addition, which is the
  first-order-optimal expansion of the reachable set toward the residual.
- **Uniform application** is the one design choice not forced by the theory;
  it is justified empirically by obstacle 2 (greedy per-layer selection
  provably starves the input layer on high-input-dimensional problems).

## Result (MNIST, batch 64, same data as the baseline)

| | start | test acc | params |
|---|---|---|---|
| Fixed AdamW `3×10` (reference) | — | 82.3% (2k) | 8180 |
| **CUG from `3×2`** | `3×2` = 1612 | **81.6% (2k)** | ~7k |

On the fast 2048-sample proxy the method grows `3×2 → 784→8→8→10` under
purely certificate-driven growth and reaches **81.6% test, matching fixed
AdamW `3×10` (82.3%)** from a tiny start. The full-data (10k) confirmation
run is `configs/exp/E21_uniform_10k.yaml`.

## Limitation and honest scope

Uniform growth *matches* fixed AdamW; it does not (yet) beat it on
parameter efficiency, because it reconstructs the balanced dense
architecture rather than a cheaper non-uniform one. A criterion-driven
*per-layer* growth that would be more parameter-efficient runs into
obstacle 2 (the input-layer credit-assignment problem), which greedy
certified criteria do not solve — the open problem is a non-greedy /
look-ahead growth criterion that can value the input layer's latent
contribution. CUG is the method that reliably reaches the AdamW frontier
from scratch; beating it is future work.

---

## Addendum: the parameter-efficiency prize is real, and where it lives

Measured directly (dense AdamW, MNIST 10k, batch 64, 60 epochs):

| architecture | test | params |
|---|---|---|
| `784→10→10→10→10` (uniform reference) | 89.95% | 8180 |
| `784→8→16→16→10` | 89.95% | 6866 |
| **`784→6→24→24→10`** | **89.95%** | **5728** |
| `784→8→20→20→10` | 89.70% | 7090 |
| `784→12→12→12→10` | 90.85% | 9862 |

**A narrow input layer with wide later layers reaches the same accuracy as
the uniform net with 30% fewer parameters** (5728 vs 8180). The input layer
costs 784 parameters per neuron; the later layers cost 10–25. So the
parameter-efficient shape is *narrow-in / wide-late*, and it is exactly the
shape a per-parameter greedy growth criterion prefers.

This **corrects the earlier "input-layer credit-assignment wall" reading**:
the failures of the early grown runs were not caused by a narrow layer 0 per
se (width 6 suffices) but by the *later* layers staying narrow too
(`784→4→5→2`). Look-ahead does not rescue greedy selection either — measured
at a trained `3×2`, growing the last hidden layer still wins after 12 AdamW
passes (+210 vs +156 for the input layer, at 26 vs 1574 parameters), because
the input layer's value only materializes once the later layers are wide.
It is a *joint-allocation* problem, not a myopia problem.

**Open problem, restated precisely:** steer certified growth to the
narrow-in / wide-late shape — keep layer 0 small (≈6) while driving the
later layers wide (≈24) — which would deliver ~90% at ~5.7k parameters,
30% below the fixed-AdamW frontier. `configs/exp/E22_efficient_growth_10k.yaml`
is the first attempt (per-parameter greedy growth); it does build the right
shape (`784→12→16→23`) but over-grows the expensive input layer.

## Steering growth to the efficient shape — two fixes, and what remains

Two certification defects blocked the efficient shape; both are fixed:

1. **Budget-aware growth.** `max_total_parameters` was only checked *before*
   growing, so a single input-layer widening (784 params/neuron) overshot it
   (E22 hit 10259 on an 8000 budget, layer 0 at width 12 = 92% of the model).
   Candidates whose *post-growth* parameter count exceeds the budget are now
   dropped before selection.
2. **Descent-realizing probes count as improvements.** `_growth_certificate_
   improves` is rel-error based, and delta growth always worsens rel_err, so
   no probe ever "improved" and growth was cancelled every epoch with the
   families already exhausted (E23 froze at 85.8%). Under
   `growth_select_by_descent`, a probe realizing a certified validation
   descent now counts.

With both, the method builds the target shape from `3×2` unaided:

| run | architecture grown from `3×2` | test | params |
|---|---|---|---|
| E22 (no budget filter) | `784→12→16→23` | 86.4% | 10259 |
| E23 (budget, gate blocked) | `784→6→9→10` | 85.8% | 4983 |
| **E24 (both fixes)** | **`784→6→19→14`** | **86.1%** | **5269** |
| *dense AdamW, efficient shape* | `784→6→24→24` | *89.95%* | *5728* |
| *dense AdamW, uniform reference* | `784→10→10→10` | *89.95%* | *8180* |

**The architecture problem is solved**: certified growth now reaches the
narrow-in / wide-late shape inside a parameter budget, using **35% fewer
parameters than the uniform dense baseline** (5269 vs 8180).

**What remains is a training gap, not an architecture gap.** On the shape it
builds, the certified flow reaches 86.1% where dense AdamW reaches ~90% at
the same size. The certified flow matched dense AdamW at a *fixed* `3×10`
earlier, so the gap is specific to the grown trajectory — most plausibly the
12 delta growths repeatedly perturbing the function, with the certified
families unable to fully re-converge afterwards. Closing that gap (e.g.
re-training to convergence after the final growth, or function-preserving
delta growth that keeps the certificate meaningful) is the next step.

## Attempts to close the training gap *through growth* (no retraining)

Two theory-motivated attempts, both **refuted by measurement**:

| variant | idea | test | params | shape |
|---|---|---|---|---|
| E24 (baseline) | delta growth, descent-selected, budget-aware | **86.1%** | 5269 | `784→6→19→14` |
| E25 | fully function-preserving growth (zero-output neurons, no delta at all) — the cleanest theory: growth only expands the reachable set, every function change comes from a certified family step | 57.6% | — | `784→2→4→8` |
| E26 | delta growth with the scaling line search moved from TRAIN loss to held-out loss (certifying the growth's *magnitude*) | 66.8% | — | `784→2→4→22` |

- **E25** shows the zero-init capacity is never exploited: the certified
  families exhaust before they can train the new neurons, so the network
  stalls far below the delta variant. The delta is doing real work.
- **E26** shows the GroMo train-loss line search is not the leak: choosing
  the growth magnitude on held-out data picks systematically smaller
  scalings, each growth contributes less, and the trajectory degrades.

**Correction to the earlier framing:** the delta growth is *not* an
uncertified step. Under `growth_select_by_descent` only a candidate that
realizes a **measured validation functional descent** is committed — that is
Proposition 3.8 applied to the structural step. What remains uncertified is
only the scaling's inner objective, and E26 shows certifying it does not
help.

**Where the residual gap actually comes from (current best understanding):**
the certificate governs the *functional loss* (sum-MSE) and halts the flow
when held-out MSE stops descending. Dense AdamW keeps minimizing train MSE
for a fixed epoch budget and its best *accuracy* appears after the point
where the certified flow would have stopped. The gap is therefore a
**criterion mismatch (MSE descent vs. argmax accuracy)**, not missing
capacity and not a missing retraining phase. Closing it inside the theory
would require a certificate on a loss whose descent tracks accuracy.
