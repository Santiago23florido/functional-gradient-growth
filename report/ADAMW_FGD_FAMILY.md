# The Parametric-AdamW family as a certified FGD approximator

*Companion note to arXiv:2606.16926 (FGD with growing networks) and this
repo's `README_FGD_GROWTH_MODEL.md`.*

## 1. Why AdamW belongs in the FGD framework

The FGD framework does **not** prescribe how the functional-gradient step is
produced. Proposition 3.8 / Theorem 3.10 only require, at each outer step
`t`, an approximation `g_t` of the functional gradient
`r_t = ∇_F L(f_t)` that is **admissible**, i.e. that satisfies one of the
paper's certificates. Any generator that produces an admissible `g_t` is a
legitimate *family*.

For the empirical sum-MSE loss on `(X, Y)`,

```
L(f)   = ‖f(X) − Y‖²                    (sum over the sample)
r_t    = ∇_F L(f_t) = 2 (f_t(X) − Y).
```

An unconstrained parametric optimizer such as **AdamW**, run on a clone of
the current network, is an excellent *empirical* approximator of the
functional gradient. This note formalizes it as an FGD family and states
exactly the certificate under which it is admissible — which is what makes
it usable inside the growing-network algorithm, on the same footing as the
tangent projection (`g = P_T r`) and the RKHS head.

## 2. The family, formally

**Generator.** Fix a nominal functional rate `η₀ ∈ (0, ½]` and an inner
budget. Starting from the current parameters `θ_t`, train a clone `θ′` with
AdamW (decoupled weight decay `λ`) to reduce

```
J(θ′) = ‖ f_{θ′}(X) − ( f_t(X) − η₀ r_t ) ‖²          (functional target)
      = ‖ f_{θ′}(X) − ( (1−2η₀) f_t(X) + 2η₀ Y ) ‖².
```

`η₀ = ½` makes the target exactly `Y` (plain loss minimization); `η₀ < ½`
is the *damped* functional target `f_t − η₀ r_t`, i.e. Proposition 3.8's
iterate. The realized **output-space displacement** on the sample is

```
Δ_t = f_t(X) − f_{θ′}(X).
```

**Scale calibration.** Declare the functional learning rate that makes `Δ_t`
a scale-optimal estimate of a step along `r_t`:

```
η*_t = ⟨Δ_t, r_t⟩ / ‖r_t‖²      ⇒   g_t := Δ_t / η*_t.
```

At `η*_t` the secant relative error is exactly `ε_t = sin∠(Δ_t, r_t) =
√(1 − cos²∠(Δ_t, r_t))`; this is the fix for the historical scale mismatch
of parametric families (a fixed nominal `η` mis-declares the step size).

**Admissibility (the certificate).** `g_t` is an admissible functional
gradient — and the family fires — iff it certifies on **held-out
validation** under either paper route:

- **Lemma 3.5 (relative-error route).** `ε_t < ½` and the applied
  `η_t ∈ (0, η̄(ε_t))` with `η̄(ε) = 2(1−2ε) / (L_s(1+2ε))`. Then the step
  descends with the Lemma-3.5 coefficient.
- **Proposition 3.8 (measured-descent route).** The realized validation
  functional descent `D_t = L(f_t) − L(f_{t+1}) > 0` with the *measured*
  coefficient

  ```
  ρ_t = D_t / ( η*_t · ‖r_t‖² ) = D_t / ( η*_t · 4 L(f_t) ),
  ```

  using the exact sum-MSE identity `‖r_t‖² = ‖2(f_t−Y)‖² = 4 L(f_t)`
  (the configured `theory_mu = 2`, `theory_loss_star = 0`). The global
  contraction `∏_t (1 − 2 η*_t μ ρ_t / β²)` then equals the realized loss
  ratio exactly, so Proposition 3.8's global-convergence envelope holds
  with equality. `Crel` and the LR interval are diagnostics on this route,
  not gates; the binding gate is measured descent plus the progress floor
  `η*_t ρ_t ≥ min_progress`.

Either certificate keeps the paper's guarantees intact: descent implies
alignment exactly for sum-MSE (`L(f−Δ) = L(f) − ⟨Δ,r⟩ + ‖Δ‖²`, so `D>0 ⇒
⟨Δ,r⟩ > ‖Δ‖² > 0`), and function space is convex, so there is no false
progress — only genuine functional descent or a request to grow.

## 3. Where AdamW sits among the families

| Family | `g_t` generator | Admissibility route | Cost / step |
|---|---|---|---|
| tangent | `P_T r_t = J(JᵀJ+λ)⁻¹Jᵀr_t` (one linear solve) | Lemma 3.5, or Prop 3.8 via `tangent_measured_descent` | 1 solve |
| RKHS head | closed-form kernel-ridge optimum of the output layer | Lemma 3.5 on the head | 1 solve |
| **parametric-AdamW** | **AdamW clone toward `f_t − η₀ r_t`** | **Prop 3.8 (measured) or Lemma 3.5** | k inner passes |

The parametric-AdamW family is enabled by
`fgd_approx.family_order: [..., parametric_descent]` with
`parametric_descent.optimizer: adamw`. It is **fair by construction**: the
clone trains on the same data in the same batch size as any dense baseline;
the only privileged information is the acceptance test, which is a
theorem, not extra data.

## 4. Empirical standing (MNIST, batch 64, 10k/2k/2k)

- As a **single-step** functional-gradient generator, AdamW is the
  best-aligned family measured: validation `cos∠(Δ, r) ≈ 0.96`
  (`ε ≈ 0.29`), versus the linearized tangent's `0.90` on a large probe and
  `0.29` on a small one. One 24-pass AdamW candidate drops the validation
  functional `≈16×` and certifies.
- Consequently the certified flow **matches** dense AdamW at equal
  architecture (the "undertraining" gap of the pure tangent flow is
  closed). It does not out-optimize AdamW at fixed architecture, because on
  MNIST there is no parametric pathology for function-space convexity to
  exploit.

## 5. The intended use: parameter efficiency through growth

The family is the *training* engine; the **growing-network method** is
where FGD earns its keep. Starting from a minimal `3×2` network and adding
neurons only where the certified functional residual demands it
(`growth_prefer_lower_error` targets the layer that most reduces the
post-growth relative error, including the expensive input layer), the goal
is to reach — at **fewer parameters** than a fixed AdamW network — a
**comparable or better** test accuracy. That is the claim this family is
built to support and the experiment `configs/exp/E15_grow_from_3x2.yaml`
evaluates.
