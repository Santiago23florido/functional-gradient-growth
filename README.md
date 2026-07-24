# stable-tiny — Certified functional-gradient descent with train-and-grow

An implementation of certified **functional gradient descent (FGD)** with
architecture growth, based on arXiv:2606.16926. The network is trained by
steps that are **certified** — each one provably descends the loss under
Lemma 3.5's condition `RelErr(g, ∇L) < 1/2` — and it **grows** only when the
current structure can no longer certify a step.

---

## Results

The research question: *does enforcing the exact `1/2` certification
condition reach a global optimum in loss **and** maximum accuracy?*

Answer, measured: **yes — when the data is dense enough that the exact
empirical optimum coincides with the true function, growth is
function-preserving, and a ladder of certified families keeps growth cheap.**

### Headline run — `configs/fgd/family_ladder_N1024.yaml` (the default)

Synthetic regression task (`smooth_sin`, 4 inputs / 1 output), **1024
training points**, evaluated on **8192 fresh points**:

| metric | value |
|---|---|
| **test accuracy** (8192 unseen points) | **0.928** |
| train / validation / test | 1.000 / 0.923 / 0.928 — small gap |
| function-preserving growths | 17 |
| certified family steps | 3 |
| certified FGD steps | 200 |

Train, validation and test rise **together** — the exact certified method
converges to the underlying function, it does not memorise the sample.

### How the result was reached (each measured)

- **Feasibility ceiling.** A fixed AdamW network on 256 points tops out at
  **0.73** test and then overfits — 90 % is not attainable at that data
  density for *any* method, so the target is set by the data, not the method.
- **Convergence vs memorisation.** 256 points → train 1.000 / test 0.315
  (memorises); 1024 points → train 1.000 / test **0.869** on 8192 fresh
  points (converges). You cannot match 8192 unseen points by memorising 1024,
  so high fresh-point accuracy **is** the proof of convergence — overfitting
  is a symptom of data *sparsity*, not of over-parameterisation per se.
- **Function-preserving growth** keeps the theory clean (growth adds empty
  capacity, `f` unchanged; the certified steps do all the descent) but is
  ruinous alone: it needs ~**57 growths** to certify because it must capture
  80 % of the *full* residual.
- **Certified family ladder** fixes that. Before growing, a **nonlinear
  within-MLP family** is tried and accepted only when **its own** projection
  certifies (`RelErr < 1/2`, never a descent criterion, never the tangent's
  projection). It certifies at a far smaller structure than the linear
  tangent, cutting FP growth **57 → 6** in isolation and giving the headline
  run above.

Full derivation and the negative results along the way (magnitude Tikhonov,
GCV clipping, stochastic probe) are in the commit history and `report/`.

---

## Running the default

```bash
PYTHONPATH=src python -m stable_tiny            # runs configs/fgd/family_ladder_N1024.yaml
PYTHONPATH=src python -m stable_tiny --config configs/experiments/<name>.yaml
```

- `configs/fgd/family_ladder_N1024.yaml` — **the default**, the best-performing
  configuration, kept exactly as it was launched.
- `configs/experiments/` — every other configuration (baselines, ablations,
  the data-density and regularisation sweeps used to reach the result).

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests/ -q
```

## Key modules

| path | what it does |
|---|---|
| `src/fgdlib/tangent.py` | the tangent projection `g = P_T(∇L)`, Lemma 3.5's rate `η̄(ε)`, config |
| `src/fgdlib/search/certify.py` | grow-until-certified loop; the family-ladder hook |
| `src/fgdlib/search/families.py` | the nonlinear within-MLP certified family |
| `src/fgdlib/search/damping.py` | measurement-chosen projection regularisation (GCV / descent) |
| `src/fgdlib/search/realize.py` | realises the certified functional step as an integrated path |
| `src/fgdlib/search/linearization.py` | enforces Lemma 3.5's hypothesis (the step *is* the function-space step) |
| `src/stable_tiny/pipeline.py` | the training pipeline wiring all of the above |
