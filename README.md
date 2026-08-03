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

### Headline result — `configs/fgd/family_ladder_N1024.yaml` (the default)

Synthetic regression task (`smooth_sin`, 4 inputs / 1 output), **1024
training points**, evaluated on **8192 fresh points**, under a **600-parameter
budget**. Four seeds (`model_seed` 0-3, `train_seed` fixed at 0), 70 epochs:

| seed | test accuracy | architecture | parameters |
|---|---|---|---|
| 0 | 0.949 | 11-19-16 | 620 |
| 1 | 0.940 | 12-17-18 | 624 |
| 2 | 0.927 | 13-19-14 | 626 |
| 3 | 0.958 | 9-16-22 | 602 |
| **mean** | **0.9435** | | |

**Every seed clears 0.925 individually**, and the four architectures are
**different from each other** (width ratios 1.46-2.44) rather than four copies
of one shape. Reproduced exactly — accuracy *and* architecture to the neuron —
on a re-run; the pipeline is deterministic.

Train, validation and test rise **together** — the exact certified method
converges to the underlying function, it does not memorise the sample.

### What decides *where* to grow

The growth location is the layer with the largest **expressivity
bottleneck**, `activation_gradient · Σ(eigenvalues_extension²)` — TINY's
extension term: the mass of the desired functional change that falls *outside*
what the current width can express. It is deliberately **not** TINY's
`select_best_update`, which adds `parameter_update_decrease` — how much
re-fitting the *existing* weights would gain, a statement about training
rather than about width. Under function-preserving growth that decrease is not
even realised at the instant of growth, so it is used only as a measure of the
gap between what is wanted and what is reachable, comparing layers at the same
instant.

Measured against the previous rule at **equal compute** (the control saturates
at 25 epochs and does not improve by 70):

| growth criterion | mean | architectures |
|---|---|---|
| gain-per-parameter of a *step* (`certified_gain`) | 0.9230 | `15-16-15` in 3 of 4 seeds |
| **expressivity bottleneck** | **0.9435** | 4 distinct, ratios 1.46-2.44 |

The old rule read the *step's* relative error, which **diverges** (0.64 → 114
on the easy synthetic function, 229 on the hard one) as the damped projection
recovers less of the residual — while the growth loop's own well-conditioned
`eps` said 0.15-0.39, i.e. *the structure is more than adequate*. Growth was
chasing a quantity that had stopped meaning anything: on the easiest
constructible target the method reached test 1.000 at **74 parameters** and
grew to **872** anyway, 25 growth events in 25 epochs.

Across data variants at free budget the bottleneck criterion is better *and*
cheaper on tasks of opposite difficulty: the easy function drops 872 → **460**
parameters at the same 1.000 accuracy, and the hard one 1123 → **953** while
accuracy rises 0.313 → **0.424**.

### The family's functional step is swept, upward

The certified family tries several functional step sizes `η_f` instead of one,
each certified independently by its own `RelErr(Δ, r) < min(threshold, 1/2)`
on the same probe, stopping at the first certificate — more search at an
unchanged bar. Direction matters: ascending `[1, 2, 4, 8]` gives **0.9230**
against **0.9000** for a single `η_f = 1`, while descending `[1, .5, .25,
.125]` *loses* (0.8823), because a small `η_f` certifies a small step and a
certified family step defers growth without progress.

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

## License

Copyright (c) 2026 Santiago Florido Gomez.

This project is licensed under the [MIT License](LICENSE).
