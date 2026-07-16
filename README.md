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
|---|---|---|
| `normal` | all weights (SGD) | none |
| `fgd_approx` | all weights (tangent-space FGD) + RKHS head phase + growth | per-epoch tangent validation; certified head optimum per structure |
| `fgd_rkhs` | linear head over a fixed structure | global optimum of the fixed structure (exact constants) |
| `fgd_rkhs_grow` | certified head per structure + GroMo growth | per-structure global optimum; growth arbitrated by the closed-form ceiling `L*` |
| `fgd_pl` | **all weights** (full-batch descent) + growth on `mu` collapse | measured-PL envelope, validated each epoch |

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

```
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

```
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

```
‖∇_θ L‖² = (1/n²) rᵀ (J Jᵀ) r ≥ (2 λ_min(K_t)/n) · L,    K_t = J_t J_tᵀ
```

is an algebraic identity: the loss satisfies PL at `θ_t` with the
*measured* constant `μ_t = λ_min(K_t)/n` of the empirical tangent (NTK)
Gram. Each full-batch step must realize a descent coefficient
`r_t = (L_t − L_{t+1})/(η‖∇L‖²) ≥ r_min` (backtracking otherwise), which
absorbs the second-order remainder of the parametrization a posteriori.
The run then carries the measured analogue of Proposition 3.8,

```
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
