# SENN's natural expansion score as a second answer to *where* to grow

*Companion to `CROSS_ENTROPY_FGD.md` §6. Reference: Mitchell, Menzenbach,
Kersting, Mundt, "Self-Expanding Neural Networks", arXiv:2307.04526v3.*

The certified flow answers **when** to grow with Lemma 3.5 (ε ≥ ½ proposes)
and R1 (a certified step that no longer reduces ε lets the proposal stand).
It answered **where** with uniform widening, which `CROSS_ENTROPY_FGD.md`
§6.7(d) shows overpays: TINY's truncation accepts a direction costing 784
parameters in $w_1$ and one costing 19 in $w_3$ on the same absolute
evidence, and the resulting network spends 96 % of its parameters on one
layer. This note validates SENN's criterion and adopts it as a second,
config-selectable *where*.

---

## 1. The two frameworks measure the same object

SENN scores an expansion by the **natural expansion score** (their eq. 3):

$$\eta \;=\; g^{\!\top} F^{-1} g \;=\; \frac{1}{N}\,\bigl\lVert P_\Theta(g_y)\bigr\rVert^2 ,$$

where $g_y$ is the gradient with respect to the concatenated network outputs,
$J$ the parameters-to-outputs Jacobian, $F=\tfrac1N J^{\!\top}\!J$ the Fisher
matrix under the Euclidean output metric (the choice SENN states in §3.4),
and $P_\Theta$ the orthogonal projection onto $\operatorname{range}(J)$.

**That projection is the object this codebase already certifies.** $g_y$ is
our functional gradient $r$; $\operatorname{range}(J)$ is our tangent space;
the shared-direction probe solves exactly $P(r)=J u^\star$ with
$u^\star=\arg\min\lVert Ju-r\rVert^2$. Lemma 3.5's relative error is
$\varepsilon=\lVert P(r)-r\rVert/\lVert P(r)\rVert$, so orthogonality gives
$\lVert r\rVert^2=\lVert P(r)\rVert^2+\lVert r-P(r)\rVert^2$ and hence

$$\boxed{\;N\eta \;=\; \lVert P(r)\rVert^2 \;=\; \frac{\lVert r\rVert^2}{1+\varepsilon^2}\;}\tag{$\ast$}$$

verified numerically to $10^{-9}$ over random Jacobians
(`tests/test_senn_expansion_score.py`). Two consequences:

1. **Lemma 3.5's admissibility is SENN's score threshold.** $\varepsilon<\tfrac12$
   is *identical* to $N\eta > 0.8\lVert r\rVert^2$. The two frameworks state
   the same condition in different coordinates, so adopting SENN's score
   introduces **no new assumption and weakens no certificate**.
2. **Maximising $\Delta\eta$ is minimising $\varepsilon$**, at fixed
   $\lVert r\rVert^2$.

### What is *not* adopted

SENN answers **when** with two tuned thresholds — relative $\tau$ and
absolute $\alpha$ (their Ingredient 4). A tuned threshold does not transfer
to a dataset that has never been trained on, which is exactly why R1 exists.
The *when* stays ours. Only the *where* is taken.

---

## 2. Why this is not R2 again

§6.2 refuted R2, which ranked layers by $\Delta\varepsilon$ **per added
parameter**. By $(\ast)$, SENN's ranking is a monotone function of the same
$\Delta\varepsilon$ — so the criteria differ in exactly one respect:

> **SENN maximises the raw score increase; R2 divided it by the parameter
> cost.**

That division was the defect. After a $784\to2$ projection every later layer
has a tiny fan-in and is therefore always the cheaper buy, so R2 grew layer 2
ten times consecutively and finished at $784\to2\to2\to14$, 64.4 %. Dropping
the division is the controlled change under test, and SENN supplies its
justification. SENN's appendix D independently notes that the GGN
underestimates curvature for earlier layers, biasing the method toward *more*
capacity there — the direction our bottleneck needs.

Measured on a fresh network, the raw score ranks the input layer first by a
factor of ~70 (`tests/test_senn_where_selection.py`), where the
per-parameter ranking put it last.

---

## 3. The criterion is already computed — KFAC is the mechanism

SENN does not rank locations by building trial models; it ranks them from the
**KFAC factors of the existing layer** (§3.4, appendix B). Reading GroMo, that
quantity is already there:

- `growing_module.py:876` documents the extension's first-order effect as
  $L(A+dA)=L(A)-t\,\sigma'(0)\,\bigl(\sum_i s_i^2\bigr)+o(t)$. So
  $\sum_i s_i^2$ **is** the first-order loss decrease the expansion buys —
  SENN's expansion-score increase for that location.
- `compute_optimal_added_parameters` forms $P=S^{-1/2}N$ and takes its SVD,
  where $S$ is the input activation second moment — KFAC's $A$ factor.
- The **output-side** factor SENN also needs is present as
  `covariance_loss_gradient()`, applied as
  `matrix_p = matrix_p @ matrix_e_inverse_sqrt`, gated by `tiny_use_fisher`.

so `sum(eigenvalues_extension**2)` reads the score off directly, and
`tiny_use_fisher` selects which output metric it is measured in.

### Score and initialization are one decision, which is why the metric matters

The same SVD supplies both the singular values (the score) and the singular
vectors $\alpha,\omega$ (the new neurons' initial weights). They cannot be
chosen independently: preconditioning the decomposition changes the ranking
*and* the initialization together. SENN's Ingredient 2 — "choose the
initialization that maximises $\Delta\eta$" — is therefore automatic here,
but it also means a bad metric choice corrupts both at once.

**Measured: enabling the Fisher factor does exactly that, and it must stay
off.** On MNIST from $3\times2$ under cross-entropy:

| `tiny_use_fisher` | layer | score | scaling chosen | train loss |
|---|---|---|---|---|
| false | 0 | 0.974 | 0.510 | 0.122 |
| false | 1 | 0.0030 | 0.376 | 0.121 |
| false | 2 | 0.0005 | 0.706 | **0.086** |
| **true** | 0 | 11.02 | 0.749 | 0.137 |
| **true** | 1 | 1.23 | 0.728 | 0.147 |
| **true** | 2 | **37.69** | **0.000** | **0.187** |

With the Fisher factor the score ranks layer 2 first by a wide margin, and
the scaling line search then rejects that very extension outright — zero
magnitude, worst realised loss of the three. The score is *anti-correlated*
with the outcome. End-to-end the consequence was unambiguous: **7 of 8 growth
events took scaling 0**, adding parameters that contribute nothing, so
$\varepsilon$ never improved, R1 re-fired every epoch, and the run stalled at
65.6 % while accumulating dead capacity.

This is not a departure from SENN. Their §3.4 states the theory in the
Euclidean output metric — "for the purposes of simplicity we choose the
euclidean metric, corresponding to $F:=\tfrac1N J^{\!\top}\!J$" — which is
also the metric the bridge identity $(\ast)$ is derived under. KFAC
$S\otimes A$ is their *practical approximation* to that matrix, adopted
because the full Fisher is intractable at their scale; at the widths here the
approximation is not needed and, measured, it hurts.

### A pre-existing mismatch this exposed

`grow_layer` and `rank_layer_expansion_score` both accumulate GroMo's
statistics with `torch.nn.MSELoss(reduction="sum")` **hard-coded**
(`src/fgdlib/growth.py`), regardless of `functional_loss`. So under
cross-entropy the growth machinery — the delta, the score and the line search
— is computed against a *different* objective from the one the certificates
govern. This predates SENN and affects uniform widening identically, so it
does not confound the A/B; but it is a real inconsistency and belongs on the
list in `CROSS_ENTROPY_FGD.md` §6.8.

### Cost

`rank_layer_expansion_score` (`src/fgdlib/growth.py`) stops after the
statistics pass and the SVD. Ranking $L$ layers therefore costs $L$
statistics passes, against the previous $L\times(1+12)$ passes plus $L$ model
clones plus, under the ε selectors, $L\times 10$ look-ahead AdamW steps. The
12-iteration golden-section line search is paid **once**, on the chosen
location. Tests pin that the ranking leaves the model bit-identical and never
enters the line search.

---

## 4. The allocation loop

Picking a single location per growth event is precisely what starved the
input layer under R2. SENN's Ingredient 4 does not do that either: it adds at
the best location and **repeats** until no worthwhile proposal remains.

The implementation follows that loop, with two adaptations that keep it
threshold-free:

- **Budget.** Additions per growth event are capped at the number of growable
  layers — exactly what uniform widening spends per event. The comparison
  against `search_ce_uniform.yaml` therefore isolates the *allocation*, not
  the amount.
- **Stopping.** SENN's $\tau$/$\alpha$ thresholds are not adopted; the loop
  stops when no location buys a first-order decrease.
- **Recomputation.** Scores are recomputed after every addition, which is
  what lets the criterion notice that the binding constraint has moved.

That recomputation turns out to matter. On MNIST from $3\times2$ under
cross-entropy, the allocation self-corrects:

| growth event | epoch | layers receiving capacity |
|---|---|---|
| 1 | 2 | `[2, 1, 1]` |
| 2 | 5 | `[0, 0, 0]` |
| 3 | 7 | `[0, 0, 0]` |

It first relieves the late layers, then discovers that the $784\to2$
projection has become the binding constraint and pours every subsequent
addition into it. R2, ranking the same quantity per parameter and without
recomputation, grew layer 2 ten times in a row and never touched layer 0.

---

## 5. The measured A/B — and what it actually showed

Three *where* criteria, identical in every other respect (same seeds, same
batch size, no budget, no schedule), from a $3\times2$ start on MNIST under
cross-entropy:

| *where* criterion | best test | params | growths | architecture |
|---|---|---|---|---|
| uniform widening | **90.25 %** | 6552 | 2 | $784\to8\to8\to10$ |
| SENN natural expansion | 88.70 % | 6552 | 2 | $784\to8\to8\to10$ |
| expansion per parameter | 88.85 % | 5774 | 5 | $784\to7\to10\to9$ |

**SENN and uniform widening found the identical architecture and still
differ by 1.55 points.** That single fact governs the reading of the whole
table: the spread between criteria is *training-trajectory variance at fixed
architecture*, not architecture quality. No single-run comparison here can
separate one criterion from another — and by the same token, the incumbent's
0.1-point margin over the dense AdamW baseline (90.15 % @ 8180) was never a
margin at all.

So the honest conclusion is not "SENN loses". It is:

> **The *where* is not the bottleneck.** The search converges to
> 5 800–6 600 parameters from a two-neuron start regardless of how capacity
> is allocated among layers. What determines the outcome is *when the search
> stops* — Lemma 3.5's $\varepsilon<\tfrac12$ — which is the part of the
> method that carries the theory.

That is a positive result about the method's stability and it sharpens where
further work belongs. It also means the comparison must be re-run across
seeds before any of these numbers is quoted as a result; that sweep is what
`scripts/seed_sweep.py` exists for.

### What each criterion is still worth

- **Expansion per parameter** reached comparable accuracy at **12 % fewer
  parameters** (5774 against 6552) and produced the narrow-in/wide-late shape
  the dense frontier rewards, without that shape being imposed. Whether the
  parameter saving survives seeding is exactly the open question.
- **SENN's score** is the cheapest of the three to evaluate (§3) and remains
  the one with an exact bridge to the certified quantity (§1). Its value here
  is that ranking locations costs $L$ statistics passes instead of $L$ model
  clones plus $13L$ passes.
- **Uniform widening** remains the default: nothing measured yet displaces it,
  and a criterion that is not measurably better should not become the default
  merely because it is better motivated.
