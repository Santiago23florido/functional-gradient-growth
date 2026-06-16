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
