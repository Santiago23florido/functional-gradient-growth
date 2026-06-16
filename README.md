# stable-tiny — growth triggers for tiny MLPs

Minimal harness around the **local `gromo`** library to watch what happens to a
self-growing MLP (`gromo.containers.growing_mlp.GrowingMLP`) when it grows.

**Goal (this stage): comparison only.** Train the same tiny MLP under the
regimes configured in YAML and compare the training dynamics. The main
functional comparison uses:

1. `gromo_tiny`: scheduled GroMo/TINY growth with AdamW.
2. `functional_triggered_tiny`: functional-gradient steps on the current MLP;
   when the empirical relative-error certificate fails, grow once using the
   normal GroMo/TINY growth machinery.

`baseline_mlp` still exists as a single-method option for controls, but it is
not part of `configs/compare_functional.yaml`.

The `gromo` library is not modified. The functional-gradient experiment lives
inside this harness so it can be tested as an alternative training/growth
policy.

## Theory-grounded certified growth (v2) — `functional_certified_tiny`

This is the main contribution. It tightens the link between the **expressivity
bottleneck** (GroMo/TINY) and **functional gradient descent** with adaptive
representations (arXiv:2606.16926), and it *verifies* the theory's conditions
instead of assuming them.

**The bridge.** The function space is the empirical output (logit) space
`ℝ^{B×C}`, where the Hilbert space equals the Banach space, so Assumptions 3.3
(`𝓗` descends in `𝓑`, α=1) and 3.4 (gradient compatibility, β=1) hold by
construction. The ideal functional gradient is `∇L(f) = dL/d(logits)` (computed
exactly). The achievable gradient is its projection onto the network's tangent
space, `g = P_T ∇L(f)`; the residual `r = ∇L(f) − g` is *exactly the expressivity
bottleneck* — the part of the desired functional update that no parameter move of
the current network can realize. Since `RelErr = ‖r‖/‖g‖`, the paper's
relative-error certificate failing is identical to the expressivity bottleneck
being too large.

**What v1 verified vs what v2 adds.** v1 (`functional_triggered_tiny`) only
checked the relative-error certificate. v2 additionally:

1. **Verifies the second constraint** — the sufficient-descent condition of
   Lemma 3.5 — with a function-space Armijo line search on the step size
   (`L(f − ηg) ≤ L(f) − c·η·⟨∇L,g⟩`), adapting η to the unknown smoothness K.
2. **Computes the projection exactly** (`functional_projection: exact`): for the
   tiny models here the full output-space Jacobian fits in memory, so `g` and the
   bottleneck `r` come from a least-squares solve instead of conjugate gradient.
   This removes the CG conditioning pathology (CG fails to converge on trained,
   over-parameterized nets) and makes the certificate exact.
3. **Uses the certificate as the growth policy** — grow while the probe batch
   fails to certify (bottleneck too large), and **stop once it certifies**. This
   is Algorithm 1's "refine the representation until certified" realized as
   network growth that targets the bottleneck residual (`tiny_best`).

The bottleneck is essentially zero once the parameter count exceeds the
output-space dimension `B·C` and the Jacobian is full rank — the genuine
expressivity threshold — and strictly positive below it.

**Result (`configs/compare_certified.yaml`, seed 0).** Both methods start from the
same under-capacity student (`hidden_size=4`) and train with AdamW; the only
difference is the growth policy (fixed schedule vs the functional certificate):

| method | test loss | test acc | params | growths |
|---|---|---|---|---|
| `gromo_tiny` (scheduled, "tiny simple") | 0.339 | 0.551 | 338 | 5 |
| `functional_certified_tiny` (certificate) | **0.333** | 0.541 | **269** | 7 |

Certificate-triggered growth reaches a **lower test loss at ~20% fewer
parameters**: it grows exactly to the size where the tangent space spans the
functional gradient, then stops and trains, instead of growing on a fixed
schedule. (`functional_train_optimizer: fgd` instead trains with the verified
functional steps themselves — a purer but weaker optimizer; see the config.)

```bash
../.venv/bin/python run.py --config configs/compare_certified.yaml
```

## Known-good benchmark: Gaussian-mixture ("blobs") — `configs/compare_blobs.yaml`

The random-teacher task has a large *unrealizable* teacher, so a small student
caps around 0.55 accuracy — a low ceiling that hides the benefit of good growth.
The `blobs` task (`stable_tiny.data.make_blobs_dataloaders`) is *realizable* with a
clear ceiling and an **optimal** model size: on the default settings a ~hidden-16
MLP reaches ~0.94, smaller nets underfit and much larger nets overfit. Growing to
the right size and stopping is the winning strategy.

Both methods start under-capacity (`hidden_size=4`) and train with AdamW; only the
growth policy differs:

| method | test acc | test loss | params | growths |
|---|---|---|---|---|
| `gromo_tiny` (scheduled) | 0.958 | 0.202 | 1611 | 6 |
| `functional_certified_tiny` | 0.958 | **0.162** | **760** | 3 |

Certificate-triggered growth matches the accuracy with **less than half the
parameters** and a lower test loss — it grows only until the tangent space spans
the functional gradient, then stops.

**Two practical findings (trial-and-error):**

1. *Lowering the similarity threshold (`functional_relative_error_tolerance`) does
   not help.* Sweeping it showed a sweet spot around 0.3; stricter values grow
   more and raise test loss. The threshold is not the knob that controls runaway
   growth.
2. *Variability in the optimization path comes from the learning rate.* At
   `lr=0.05` the path oscillates, the certificate flickers `certified ↔ not`, and
   that flicker keeps re-triggering growth until the budget is exhausted
   (3020 params, test loss ~0.8–1.6). At `lr=0.02` the path is stable. The
   principled fix is **growth hysteresis** (`functional_freeze_growth_after_certified`):
   the expressivity bottleneck is a capacity property, so once the tangent space
   has certified, later failures are optimization noise and must not trigger more
   growth.

```bash
../.venv/bin/python run.py --config configs/compare_blobs.yaml
```

## Function-space landscape visualization

`make_landscape.py` renders functional gradient descent *in function space*. A
network snapshot is its logit vector on a fixed probe set, `f = logits(probe) ∈
ℝ^{N·C}` — a space whose dimension is independent of network width, so a tiny net
and a grown net share one coordinate system. The runs are projected to 2D by PCA
over all snapshots, and the **exact** functional loss `L(f) = CE(softmax(f), y)`
is drawn as a topographic map (it depends only on `f`, not the parametrization).
Each descent is animated as a path over the fixed landscape, with growth events
marked as stars — growth is literally the trajectory gaining access to directions
the smaller tangent space could not reach.

```bash
../.venv/bin/python make_landscape.py --config configs/landscape.yaml
# outputs results/teacher_landscape_landscape.gif (+ .png final frame)
../.venv/bin/python make_landscape.py --static    # final-frame PNG only (fast)
../.venv/bin/python make_landscape.py --3d        # 3D relief: the surface as
#   terrain and each descent riding on it at its own loss height, with a rotating
#   view (outputs *_landscape_3d.gif + .png)
```

## What it does

`run.py` runs `stable_tiny.experiment.run_experiment`, which:

1. Builds a `GrowingMLP` (starts intentionally tiny: `hidden_size=8`).
2. Runs one of the configured methods.
3. Logs train/test loss, train/test accuracy, parameter count, and, for the
   functional method, the empirical relative-error certificate.
4. Saves a metric history JSON, prints a per-growth spike report, and writes a
   3-panel PNG (loss / accuracy / params) with dashed lines at growth events.

For `functional_triggered_tiny`, the functional space is the finite batch-logit
space. The ideal gradient is `dL/df` at the logits. The approximated gradient is
the projection induced by the current network tangent space, computed through
Jacobian-vector products. If the certificate
`(1 + eps) ||g - grad L|| < eps ||g||` fails, the representation is considered
insufficient and the harness grows the MLP with GroMo/TINY.

Two stabilizers are available for this trigger:

1. `functional_warmup_epochs`: AdamW epochs before the functional certificate is
   enforced.
2. `functional_failure_patience`: number of consecutive failed certificates
   required before triggering TINY growth.

## Task

Default is a **synthetic random-teacher classification** task (`task: teacher`):
labels come from a fixed random MLP. It needs no extra dependencies, lives on
the GPU, builds instantly, and is capacity-demanding so growth genuinely
matters. MNIST / FashionMNIST are available (`task: mnist`) but require
`torchvision` (imported lazily).

The default config trains in seconds on a GPU — far under the 25-minute budget,
leaving room to scale up.

## Run

```bash
# from this directory, using the project virtualenv that has the local gromo
../.venv/bin/python run.py

# compare scheduled TINY + AdamW against functional-triggered TINY
../.venv/bin/python run.py --config configs/compare_functional.yaml
./run_functional_compare.sh

# override config keys inline
../.venv/bin/python run.py --set growth_steps=12 --set epochs_per_step=8
../.venv/bin/python run.py --set task=fashion_mnist   # needs torchvision
./run_functional_compare.sh --set growth_steps=8 --set functional_cg_max_iter=16
./run_functional_compare.sh --set functional_warmup_epochs=3 --set functional_failure_patience=2
```

Outputs land in `results/` (git-ignored):
`*_history.json`, `*_comparison.json`, and, for multi-method configs, one
combined `*_comparison_curves.png`.

## Layout

```
stable-tiny/
├── run.py                  # CLI entry point
├── run_functional_compare.sh
├── configs/default.yaml    # single-method hyperparameters
├── configs/compare_functional.yaml
├── src/stable_tiny/
│   ├── data.py             # teacher / mnist / fashion_mnist loaders
│   ├── functional_descent.py
│   ├── growth.py           # gromo TINY growth step + TINY layer scoring
│   ├── experiment.py       # method loops, growth triggers, metric logging
│   └── plotting.py         # curves + spike report
└── results/                # outputs (git-ignored)
```
