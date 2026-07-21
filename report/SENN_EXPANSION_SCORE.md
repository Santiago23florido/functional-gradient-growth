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

So **`tiny_use_fisher: true` turns TINY's computation into SENN's
KFAC-preconditioned score**, and `sum(eigenvalues_extension**2)` reads it off.

### Score and initialization cannot disagree

The same SVD supplies both the singular values (the score) and the singular
vectors $\alpha,\omega$ (the new neurons' initial weights). Ranking with the
Fisher-preconditioned metric while initialising from the unpreconditioned
decomposition would be incoherent; because both come from one factorisation,
enabling `tiny_use_fisher` makes the *initialization* SENN-optimal too —
their Ingredient 2, "choose the initialization that maximises $\Delta\eta$".

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

## 5. Status

The theory is validated and the criterion is implemented, tested and
config-selectable (`growth_selection: natural_expansion`, with
`tiny_use_fisher: true`). Whether it **beats** uniform widening end-to-end is
an empirical question that the A/B run against the incumbent
(90.25 % @ 6552, `784→8→8→10`) settles — and R2 is the standing reminder that
a criterion which looks correct in isolation can still fail in the loop. The
result is recorded here either way.
