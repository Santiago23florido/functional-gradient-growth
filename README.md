# functional-gradient-growth

Certified functional gradient descent (FGD) for growing neural networks.

The repository has two layers:

- **`src/fgdlib/`** — the library. Everything that trains the network:
  - `fgdlib.rkhs` — certified RKHS FGD (Algorithm 1 of
    [arXiv:2606.16926](https://arxiv.org/abs/2606.16926)) with exact,
    measured constants and the global-optimality certificate for a fixed
    structure.
  - `fgdlib.tangent` — tangent-space FGD approximation with per-epoch
    validation certificates.
  - `fgdlib.empirical_pl` — empirical-PL certified training of **all**
    network weights (measured PL constant, validated steps, measured
    envelope).
  - `fgdlib.growth` — GroMo growth machinery (optimal extensions, scaling
    line search), plus schedules, optimizers and training utilities.
- **`src/stable_tiny/`** — the application layer: datasets, the experiment
  pipeline (which consumes the library), W&B logging and plotting.

## Training methods

| `training.method` | Trains | Certificate |
| --- | --- | --- |
| `normal` | all weights (SGD) | none |
| `fgd_approx` | all weights (tangent-space FGD) + RKHS head phase + growth | per-epoch tangent validation; certified head optimum per structure |
| `fgd_rkhs` | linear head over a fixed structure | global optimum of the fixed structure (exact constants) |
| `fgd_rkhs_grow` | certified head per structure + GroMo growth | per-structure global optimum; growth arbitrated by the closed-form ceiling `L*` |
| `fgd_pl` | **all weights** (full-batch descent) + growth on `mu` collapse | measured-PL envelope, validated each epoch |
| `fgd_adaptive_grow` | all weights through finite secants + representation-only growth | strict Algorithm 1 relative error and global envelope on the full empirical train set |

Run any config with:

```bash
PYTHONPATH=src python -m stable_tiny --config configs/fgd/<name>.yaml [--wandb]
```

---

## Mathematical background

### 1. FGD in a Hilbert space and the paper's guarantee

For a loss `L` over functions in a Hilbert space `H`, ideal FGD iterates
`f_{t+1} = f_t − η ∇L(f_t)` with the functional gradient (Riesz
representer). Under the assumptions of arXiv:2606.16926 — K-smoothness,
`H` descends (α), compatibility (β), Polyak–Łojasiewicz with constant μ —
and approximate gradients `g_t` with certified relative error
`‖g_t − ∇L‖/‖g_t‖ ≤ ε̄ < 1/2` (enforced by the acceptance test
`(1+ε)U_t < ε‖g_t‖` of Algorithm 1), Proposition 3.8 gives **global**
linear convergence:

```text
L(f_T) − L* ≤ (1 − 2 η β⁻² μ r)^T · (L(f_0) − L*),   r = α − Kη/2 − (β + 3Kη/2)·ε̄/(1−ε̄)
```

The guarantee is *conditional on the choice of H*, and `L*` is the optimum
**of H**. It says "you will reach the ceiling of H", never "the ceiling of
H is good".

`fgdlib.rkhs` instantiates this with `B = H` (so α = β = 1) and the
empirical regression functional `L(f) = (1/2n) Σ‖f(Xᵢ) − Yᵢ‖²`, for which
every constant is **computed, never assumed**: the functional gradient is
`Σᵢ Cᵢ k(Xᵢ,·)` with `C = R/n` (exactly representable in the dictionary),
`K_s` and `μ = λ_min(Gram)/n` come from eigendecompositions, `U` is the
exact projection error (valid for any solver output), and the top
dictionary level realizes `U = 0`, so Algorithm 1 terminates
unconditionally.

Two kernels are supported:

- **Gaussian** (`kernel: gaussian`): `H` is universal on a compact input
  domain, hence `L* = 0` — the certificate has full force (reaching the
  ceiling means interpolating the training data).
- **Linear over a frozen feature map** (`kernel: linear`): the trained
  model is *exactly* a fixed MLP (frozen hidden layers + trained output
  layer); `L*` is the exact least-squares optimum of that structure,
  computable in closed form. A fixed whitening reparametrization of the
  head (`feature_whitening`) makes the Gram near-isotropic, so the
  certified contraction is sharp (≈0.88/step on MNIST) instead of
  near-vacuous — same function class, same `L*`, better `H`-geometry.

### 2. Why a fixed nonconvex network cannot carry the global certificate

With trainable hidden weights the reachable set `{f_θ}` is the nonconvex
image of parameter space. Take a tanh network at `θ = 0`: the parametric
gradient vanishes, yet `L > L*` and the functional gradient is large.
This single point kills both required assumptions at once — PL
(`L − L* ≤ ‖∇L‖²/2μ` fails for every μ) and "H descends" (every tangent
direction has zero directional derivative, so α = 0). Such saddles exist
for every architecture with sign/permutation symmetries, regardless of
input-domain compactness (the failure lives in θ-space, not x-space).
Consequently **no method can certify a global optimum over all weights of
a fixed architecture**; the paper avoids this by working in `H` with an
*adaptive representation* that grows as needed — never a fixed network.

The tangent space at a fixed `θ₀` *is* a legitimate linear function space
(NTK regression = a frozen feature map), and freezing it recovers all
certificates for the linearized model. What breaks the theory is
*re-linearizing while moving*: the space changes at every step, the
applied parameter update matches the certified function step only to
first order, and PL dies at saddles.

### 3. Per-structure certificates and growth

For a growing MLP, "fixed structure" holds piecewise — between growth
events. The honest maximal guarantee is a **chain**:

1. *Within a structure*: certified local descent (tangent) plus the
   certified global optimum of the output layer (RKHS head phase, used in
   `fgd_approx` where the Hilbert-secant search used to run: when the
   tangent certificate stalls and the growth probe does not improve it,
   the head is driven to the structure's optimum; a rejected phase
   *certifies the architecture is exhausted*).
2. *Growth justified a priori*: the ceiling `L*(S)` of any candidate
   structure is a closed-form least-squares value; growth can be
   arbitrated and stopped by certified ceiling improvements (see
   `fgd_rkhs_grow`: `growth_min_ceiling_improvement`). Structures nest, so
   ceilings decrease monotonically.
3. *Optimality of a growth step is impossible for anyone*: choosing the
   best new neuron is nonconvex/NP-hard; even the paper only requires the
   refinement to be *sufficient* (pass the `U` test), not optimal. For
   dictionary (RKHS) growth, greedy approximation theory
   (Maurey–Jones–Barron) additionally provides convergence *rates* of the
   grown sequence to the global optimum — unavailable for tanh/ReLU
   neurons.

### 4. From loss certificates to accuracy

All certificates control the loss. With one-hot MSE there is a rigorous
bridge: if `‖f(x) − y‖² < 1/2` the argmax is necessarily correct, so by
Markov

```text
accuracy ≥ 1 − 4·L
```

The bound is vacuous until the certified loss drops below 1/4 and bites
near zero. This is exactly why small structures (ceilings ≈ 0.30–0.44,
barely below the know-nothing predictor's 0.45) fulfil every loss
certificate yet show poor accuracy, while the universal Gaussian `H`
(`L* = 0`) converts the same certificate into interpolation-level train
accuracy. Note also that on a restricted class the loss minimizer is
generally *not* the accuracy maximizer (squared loss penalizes magnitude,
accuracy only order); the two optima coincide as the loss approaches
zero.

### 5. Empirical-PL certified training (`fgd_pl`)

The definition that reconciles "satisfy the criteria" with "train the
whole network": replace assumed constants with **measured** ones.

For the MSE, `∇_θ L = Jᵀ r / n` exactly, so

```text
‖∇_θ L‖² = (1/n²) rᵀ (J Jᵀ) r ≥ (2 λ_min(K_t)/n) · L,    K_t = J_t J_tᵀ
```

is an algebraic identity: the loss satisfies PL at `θ_t` with the
*measured* constant `μ_t = λ_min(K_t)/n` of the empirical tangent (NTK)
Gram. Each full-batch step must realize a descent coefficient
`r_t = (L_t − L_{t+1})/(η‖∇L‖²) ≥ r_min` (backtracking otherwise), which
absorbs the second-order remainder of the parametrization a posteriori.
The run then carries the measured analogue of Proposition 3.8,

```text
L_T ≤ L_0 · Π_t (1 − 2 η_t μ_t r_t),
```

validated against the true loss every epoch. If the measured `μ_t` stays
bounded away from zero down to numerically zero loss, the run is
**certified globally optimal a posteriori** — the regime NTK theory
(Jacot; Du et al.; Allen-Zhu et al.) proves reachable for wide networks,
here *verified instead of assumed*. When the structure is too small, the
Gram is rank deficient (`P < n·m` certificate rows), `μ_t = 0`, the
certificate honestly switches off — and that collapse is the
certificate-driven trigger for growing the network. Growth must
additionally be *earned*: it fires only when `mu` is collapsed **and**
the relative per-epoch loss improvement fell below
`growth_min_progress` (the structure stopped paying off), with cooldown,
event cap and width cap — otherwise a permanently collapsed `mu` (e.g.
under a tight width cap) would grow the network on cadence rather than
by need. Certificates are
per structure and apply to the empirical loss on the certificate subset
(`certificate_points`; by eigenvalue interlacing, fewer certificate rows
can only raise the measured `λ_min`, at the honest price of certifying a
smaller subset).

**Strict theory mode** (`strict_certificates`, default on): a descent step
is executed **only while the measured PL property holds**. When `μ` is
collapsed the trainer refuses to step — the theory-prescribed responses
are growing the structure (immediately, certificate-driven) or, once
growth is exhausted, finishing with the exact RKHS head phase (where
every paper property holds) and stopping. No uncertified descent is ever
executed, so the trajectory never leaves the theory.

Practical notes with certified semantics:

- Losses for the acceptance test are measured in **float64**; when even
  full backtracking cannot realize a measurable descent, the trainer
  declares stationarity at the achievable precision instead of stalling.
- **Ridge** (`F = L + (ridge/2)‖θ‖²`) is available for output-magnitude
  control, with an honest downgrade: the global vs-zero envelope is
  exclusive to the pure data loss (zero is its universal lower bound and
  the Gram identity compares against it; the augmented residual's Gram is
  rank deficient in that comparison). With ridge on, the certified
  statements are per-step sufficient descent and stationarity of `F`.
- The certificate targets the **empirical** optimum, so driving the train
  loss toward zero on few samples overfits *by design*; the primary
  defenses are more data and the best-validation snapshot the pipeline
  returns (a deployment choice along the certified trajectory —
  certificates are unaffected). Generalization is not, and cannot be,
  covered by an optimization certificate.

### 6. Strict adaptive train-and-grow (`fgd_adaptive_grow`)

This method applies Algorithm 1 directly to the finite empirical output
space on the **complete training set**:

```text
H = B = R^(n x m),   <u,v> = (1/n) sum_i <u_i,v_i>,
L(f) = (1/2)||f-y||_B^2,   K = alpha = beta = mu = 1,   L* = 0.
```

The constants are consequences of this geometry and cannot be supplied as
hyperparameters. Parameter gradients, tangent projections and nonlinear
target fitting only generate disposable candidates. Before any candidate
can replace the live model, the trainer measures its actual finite secant

```text
g_t = (f_theta_t - f_theta_candidate) / eta
```

on every training point and requires the paper's strict acceptance test
`(1 + epsilon) U_t < epsilon ||g_t||_B`, where
`U_t = ||g_t - grad L(f_t)||_B` is exact in this finite space. It also checks
the general learning-rate ceiling, positive descent coefficient, sufficient
descent and the accumulated global envelope. Norms, relative error and the
directional cosine are recorded for accepted and rejected attempts.

All full-train quantities are computed as exact reductions of bounded
minibatches. The implementation never performs a full-train accelerator
forward or materializes a full Jacobian: losses, norms and inner products are
accumulated in `float64`; exact tangent proposals accumulate `J_b^T J_b` and
`J_b^T grad_b`; large tangent problems use matrix-free chunked CG.

The MNIST reference config uses a fast strict path: the affine head and one
64-iteration tangent solve are proposed on a fixed 256-point training screen.
Only screen-valid candidates are promoted to the complete-train certificate;
the screen can never commit a model. If no candidate certifies, one scheduled
function-preserving layer is grown and the same functional step is retried.

If the affine head, projected parameter gradient, nested tangent spaces and
nonlinear secants all fail, GroMo may add neurons with useful incoming
features and exactly zero outgoing weights. This changes the representation
without changing `f_t`; the same functional iteration is then retried. If no
such refinement remains, training stops with `representation_exhausted` and
does not execute an uncertified fallback. See
`configs/fgd/adaptive_grow_mnist.yaml` for the reference configuration.

The guarantee is strictly **empirical**. A finite training set is compact,
but this does not certify a continuous input domain, validation performance
or generalization.

### Honest limitations

- Global optimality over **all weights of a fixed architecture** is
  uncertifiable for any method (Section 2); everything here is either
  per-structure, in-`H`, or a-posteriori-verified.
- The `fgd_pl` envelope is conditional-but-verified: it certifies the run
  that happened; it cannot promise in advance that `μ_t` stays positive
  for a narrow network.
- Accuracy is never certified directly (0–1 loss is nonsmooth and
  combinatorial); it is inherited through `accuracy ≥ 1 − 4L` once the
  certified loss is small.
