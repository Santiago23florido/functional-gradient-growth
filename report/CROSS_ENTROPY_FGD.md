# Does the FGD theory apply to cross-entropy? — and why MSE stalls accuracy

*Mathematical companion to `README_FGD_GROWTH_MODEL.md` §13. Written before
implementing the substitution, so the scope of the guarantee is fixed in
advance.*

Notation: the network output on the sample is $f \in \mathbb{R}^{n\times K}$
($n$ points, $K$ classes), $Y$ the one-hot label matrix, $f_i\in\mathbb{R}^K$
the $i$-th row, $c_i$ its true class. The *functional* gradient is the
gradient with respect to $f$ itself, $r=\nabla_f L$, which is the object the
FGD framework approximates.

---

## 1. What the framework actually requires of the loss

Reading Sections 7–9 of the growth model, the certificates rest on exactly
three properties of $L$ as a function of $f$:

| # | property | used by |
|---|---|---|
| (P1) | **convexity in $f$** | the "no spurious local minima in function space" argument; the claim that descent implies progress toward the global functional optimum |
| (P2) | **$L_s$-smoothness in $f$** | Lemma 3.5: the descent lemma and the admissible step bound $\bar\eta(\epsilon)=\dfrac{2(1-2\epsilon)}{L_s(1+2\epsilon)}$ |
| (P3) | **a Polyak–Łojasiewicz constant**, $\ \|\nabla_f L\|^2 \ge 2\mu\,(L-L^\star)$ | Prop. 3.8's *linear* contraction $\prod_t\bigl(1-2\eta_t\mu\rho_t/\beta^2\bigr)$, i.e. $C_{\mathrm{glob}}$ |

Nothing else about $L$ is used. So the question "does FGD apply to
cross-entropy?" is the question of which of (P1)–(P3) survive.

---

## 2. Sum-MSE: the baseline case

$$
L(f)=\|f-Y\|_F^2,
\qquad
r=\nabla_f L = 2\,(f-Y),
\qquad
\nabla^2_f L = 2\,\mathrm{Id}.
$$

- **(P1)** convex — the Hessian is $2\,\mathrm{Id}\succ0$. ✓
- **(P2)** $L_s = 2$, matching the configured `theory_smoothness_constant: 2.0`. ✓
- **(P3)** holds with *equality*:

$$
\|r\|^2 = \|2(f-Y)\|^2 = 4\,\|f-Y\|^2 = 4L,
$$

so with $L^\star=0$ the PL inequality $\|\nabla_f L\|^2\ge 2\mu(L-L^\star)$ is
saturated at $\mu=2$ — exactly the configured `theory_mu: 2.0`,
`theory_loss_star: 0.0`. This identity is what makes the certified
contraction *equal* the realized loss ratio. ✓

All three hold, which is why the implementation is tight for MSE.

---

## 3. Softmax cross-entropy

With $p_i=\mathrm{softmax}(f_i)$,

$$
L(f)=\sum_{i=1}^{n}\Bigl[\log\textstyle\sum_k e^{f_{ik}} - f_{i c_i}\Bigr],
\qquad
r_i=\nabla_{f_i}L = p_i - y_i,
\qquad
\nabla^2_{f_i}L=\mathrm{diag}(p_i)-p_ip_i^{\!\top}.
$$

### 3.1 (P1) Convexity — holds

$\log\sum_k e^{f_{ik}}$ is convex (log-sum-exp) and $-f_{ic_i}$ is linear, so
$L$ is convex in $f$. Equivalently $\mathrm{diag}(p)-pp^{\!\top}\succeq 0$,
since for any $v$,

$$
v^{\!\top}\bigl(\mathrm{diag}(p)-pp^{\!\top}\bigr)v
=\sum_k p_k v_k^2-\Bigl(\sum_k p_kv_k\Bigr)^{\!2}
=\operatorname{Var}_{k\sim p}(v_k)\;\ge\;0 .
$$

**The structural pillar of the framework survives**: function space stays
convex, so there are no spurious local minima there and certified descent
still means progress toward the global functional optimum.

### 3.2 (P2) Smoothness — holds, with a *better* constant

The Hessian is a covariance under $p$, so for $\|v\|_\infty\le1$ its
quadratic form is $\operatorname{Var}_{k\sim p}(v_k)\le\tfrac14\cdot$ …; the
sharp bound on the spectral norm is

$$
\lambda_{\max}\bigl(\mathrm{diag}(p)-pp^{\!\top}\bigr)\;\le\;\tfrac12 ,
$$

attained in the limit of two classes carrying $p=\tfrac12$ each. Hence
$L_s=\tfrac12$ (against $L_s=2$ for MSE). Since the Hessian is block
diagonal across samples, the same constant holds for the sum.

**Consequence, concrete and testable.** Lemma 3.5's admissible interval

$$
\bar\eta(\epsilon)=\frac{2(1-2\epsilon)}{L_s\,(1+2\epsilon)}
$$

becomes **four times wider** under cross-entropy. The implementation
requirement is simply `theory_smoothness_constant: 0.5`.

### 3.3 (P3) Polyak–Łojasiewicz — **fails**, in both regimes

This is where the transfer is not free, and the honest statement matters.

*Near the optimum.* Let $p_c\to1$ on a sample. Then

$$
L=-\log p_c = (1-p_c)+O\bigl((1-p_c)^2\bigr),
\qquad
\|r\|^2=\|p-y\|^2=(1-p_c)^2+\sum_{j\ne c}p_j^2 = \Theta\bigl((1-p_c)^2\bigr),
$$

so $\|r\|^2=\Theta(L^2)$ and therefore

$$
\frac{\|\nabla_f L\|^2}{L-L^\star}=\Theta(L)\;\longrightarrow\;0 .
$$

No constant $\mu>0$ can lower-bound this as $L\to0$.

*Far from the optimum.* If $p_c\to0$ then $L=-\log p_c\to\infty$ while
$\|r\|^2=\|p-y\|^2\le 2$ stays bounded. Again no $\mu>0$ works.

**Conclusion.** Cross-entropy admits **no global PL constant in function
space**, so Proposition 3.8's *linear* contraction — and with it the
$C_{\mathrm{glob}}$ envelope as currently written — does **not** transfer.

### 3.4 What replaces it

Convexity (P1) plus smoothness (P2) still give a global convergence
guarantee, only a slower one. For the exact functional step
$f_{t+1}=f_t-\eta\,r_t$ with $\eta\le 1/L_s$ and a bounded distance
$R=\|f_0-f^\star\|$ to a functional minimizer,

$$
L(f_T)-L^\star \;\le\; \frac{R^2}{2\eta\,T}\;=\;O(1/T),
$$

the standard convex/smooth rate, and it degrades gracefully under the
$\epsilon$-approximation of Lemma 3.5 (the approximate step remains a
descent direction with the same $\bar\eta(\epsilon)$ interval).

So the transfer is:

| ingredient | MSE | cross-entropy |
|---|---|---|
| convex function space (P1) | ✓ | ✓ |
| Lemma 3.5 step bound (P2) | $L_s=2$ | $L_s=\tfrac12$ (interval $4\times$ wider) |
| per-step **measured** descent certificate (Prop. 3.8 route) | ✓ | ✓ — it is *measured*, so it never invoked PL |
| linear contraction / $C_{\mathrm{glob}}$ (P3) | ✓ ($\mu=2$, equality) | ✗ — replaced by the convex $O(1/T)$ bound |
| growth trigger (Lemma 3.5 admissibility, $\epsilon\ge\tfrac12$) | ✓ | ✓ |

Everything the *method* uses to decide — admissible families, measured
held-out descent, growth as a certified structural step — carries over
unchanged. What weakens is the *rate* of the global guarantee: linear
becomes sublinear. That is a real loss of strength and must be stated as
such rather than papered over.

---

## 4. Why MSE stalls accuracy — the mechanism

The two losses differ in *where their gradient vanishes*, and that is the
whole story.

$$
r^{\mathrm{MSE}} = 2(f-Y)\quad\text{vanishes at } f=Y \ \ (\text{a finite point}),
$$
$$
r^{\mathrm{CE}} = p-Y\quad\text{vanishes only as } f_{c}\to+\infty \ \ (\text{at infinity}).
$$

**MSE penalizes confidence.** If a sample is already classified correctly
and the model becomes more confident, $f_{ic}$ grows past the target value
$1$; then $\partial L/\partial f_{ic}=2(f_{ic}-1)>0$ and MSE *pushes the
logit back down*. Squared error to a one-hot target is a distance to a
specific finite point, so overshooting it is a loss increase.

**Accuracy does not see distance, only the argmax.** The set of outputs that
classify a sample correctly is the unbounded cone
$\{f_i: f_{ic_i}>f_{ij}\ \forall j\ne c_i\}$. The MSE level sets are balls
around $Y$. A trajectory can move deeper into the correct cone — improving
the decision on hard samples — while moving *away* from the centre of the
ball, because gaining margin on the hard samples changes the logits of the
easy ones.

Hence $L_{\mathrm{MSE}}$ and accuracy are not merely imperfectly aligned:
**in the high-confidence regime they are anti-correlated**. This is not a
conjecture; it is the measurement taken immediately after a growth
(`README_FGD_GROWTH_MODEL.md` §13.5):

| state | held-out $L_{\mathrm{MSE}}$ | val accuracy | test accuracy |
|---|---|---|---|
| just after growth | 3914 | 53.4 % | 53.6 % |
| after 40 unconstrained AdamW passes | **11900** | **68.6 %** | **68.5 %** |

The certificate evaluated that same step and reported
$\cos\angle(\Delta,r)=-0.017$, $D=-7986<0$: a functional **ascent**, so the
family rejected it — correctly, for the functional it was given.

Cross-entropy has no such conflict. Its gradient never opposes additional
confidence on the true class, and its descent monotonically improves the
margin, so a certificate enforcing CE descent does not halt while accuracy
is still improving. (Both losses are classification-calibrated in the
infinite-sample limit; the difference is the finite-sample geometry —
MSE's optimum $f=Y$ is *reachable*, and further accuracy gains require
leaving it.)

### 4.1 The downstream consequence that was observed

Because the flow requests growth when its families stop certifying steps, an
MSE plateau is misread as *insufficient capacity*. With no parameter budget
the run then grew 13 times, to $784\to26\to13\to14$ (21 107 parameters),
while the state relative error was $\epsilon=0.237\ll\tfrac12$ — i.e. while
Lemma 3.5 said the reachable set could still represent $r_t$ and the paper
therefore said **not** to grow. The corrected trigger
(`growth_requires_admissibility_failure`) removes the parameter waste, but
it cannot recover the accuracy: with the MSE certificate the flow simply
stops earlier. The loss substitution is the only fix that addresses the
cause.

---

## 5. Implementation checklist for the substitution

1. Functional gradient: `mse_functional_gradient` $\to$ $p-Y$ with
   $p=\mathrm{softmax}(f)$.
2. Functional loss: sum-MSE $\to$ summed cross-entropy; the metric reported
   as "functional loss" must be the *certified* loss.
3. `theory_smoothness_constant: 2.0 → 0.5` (§3.2).
4. `theory_mu` / `theory_loss_star`: the exact identity $\|r\|^2=4L$ is
   **MSE-only**. Under CE the $C_{\mathrm{glob}}$ product must be either
   disabled or replaced by the $O(1/T)$ bound of §3.4; keeping the linear
   envelope with an invented $\mu$ would be an unsupported claim.
5. The per-step measured-descent certificate and the family ladder need
   **no change**. The *growth trigger* does: the $C_{\mathrm{prog}}$ floor is
   not a valid limit criterion here, because an infimum that is not attained
   admits descent for ever. §6 replaces it.
6. The accuracy metric is already argmax-based and needs no change; targets
   stay one-hot.

The honest summary: **(P1) and (P2) transfer, so the method and every
per-step certificate transfer; (P3) does not, so the global guarantee drops
from linear to $O(1/T)$.** In exchange, the certified objective stops
fighting the metric the model is judged on.

---

## 6. The structural criterion: what "the limit of a structure" means

The substitution of §5 exposes a hole. The growth trigger used the certified
progress floor $C_{\mathrm{prog}}$: *grow when the current structure can no
longer produce certified progress*. Under sum-MSE that is well posed, because
$\inf L$ is attained at the finite point $f=Y$ and progress genuinely runs
out. Under cross-entropy it is not: $L\to 0$ only as $f_c\to\infty$, so
raising the confidence of already-correct samples always yields more descent.
Measured on MNIST from a $3\times2$: certified progress stayed pinned at
$2.3\times10^{-2}$ against a $10^{-4}$ floor for eighty epochs — **one growth,
50.25 % test**. The criterion never fires, so the structure never grows.

A criterion is needed that is defined for *either* certified functional.

### 6.1 R1 — the structural limit is stationarity of $\varepsilon$

The reinterpretation is of Lemma 3.5's own quantity,

$$\varepsilon_t \;=\; \frac{\lVert g_t - r_t\rVert}{\lVert g_t\rVert},$$

the relative error with which the reachable set of the current architecture
expresses the functional gradient. Lemma 3.5 uses $\varepsilon<\tfrac12$ as
the *admissibility* condition on a step. The reinterpretation uses its
*evolution* as the limit condition on a structure:

> **R1.** If a **certified** family step does not reduce the held-out
> $\varepsilon$, the descent being taken is going into directions the
> structure cannot follow. The structure — not the training — is the binding
> constraint, and growth is the correct response.

The step is still committed (it certified; nothing is discarded), it simply
no longer postpones the structural step. This is a monotonicity comparison
between two measured values: **no threshold, no window, no schedule, no
parameter budget**, so it transfers unchanged to a dataset never seen before.
This is the property that made it necessary — the tuned floor could not.

**The precise division of labour** (worth stating exactly, because R1 is
easily overread as a new trigger — it is not):

| stage | what decides it |
|---|---|
| growth is **proposed** | Lemma 3.5 admissibility failure, $\varepsilon\ge\tfrac12$, or an explicit request from the tangent certificate |
| the proposal is **withdrawn** | a family step certifies **and** reduces $\varepsilon$ — training is still enlarging what the structure expresses, so capacity is not the constraint |
| the proposal **stands** | R1: the family step certified but $\varepsilon$ did not fall |

So the *trigger* is the paper's own structural criterion, unmodified. R1
governs only whether training may render that trigger unnecessary. Nothing
in the chain reads the epoch number: the trigger discards it explicitly
(`should_trigger_fgd_growth` begins `del epoch, last_growth_epoch`), and no
parameter cap participates (`max_total_parameters` is empty). The
`growth_schedule.every` / `first_epoch` fields are read **only** by
`method: normal`; under `fgd_approx` the sole field consulted is `.enabled`,
a master on/off switch. The measured run is the proof: growth fired at
epochs 2 and 4 while `first_epoch: 50` — a schedule could not have produced
that.

Why $\varepsilon$ and not the loss: $\varepsilon$ is scale-free and is a
property of the *reachable set*, not of how far training has progressed
inside it. Measured on MNIST under CE, that distinction is visible: with the
structure **fixed**, $\varepsilon$ rises $1.93\to3.14$ as training proceeds
(training is exhausting what the structure can express); with the structure
**growing**, it falls $1.93\to0.28$.

### 6.2 R2 — proposed, and refuted by measurement

R1 answers *when* to grow. The natural companion was to answer *where* by the
same currency: spend the next parameter where it most reduces $\varepsilon$,

$$\ell^\star=\arg\max_\ell \frac{\Delta\varepsilon_\ell}{\Delta p_\ell}.$$

Isolated, the criterion looks right. On a trained $3\times2$ under CE
(base $\varepsilon=1.7732$) the look-ahead ranking is the only one that
discriminates at all, and it agrees with the ground truth:

| grown layer | immediate $\varepsilon$ | look-ahead $\varepsilon$ | test after 10 steps |
|---|---|---|---|
| 0 (input) | 1.876 | 2.025 | 17.6 % |
| 1 | 1.833 | 1.952 | 19.2 % |
| 2 | 1.810 | **1.458** | **48.4 %** |

End-to-end it fails. The run grew layer 2 ten consecutive times and finished
at $784\to2\to2\to14$, **64.4 %**, with $\varepsilon$ saturating near $0.75$.
The cause is structural and applies to *every* per-parameter criterion —
certified descent, immediate $\varepsilon$ and look-ahead $\varepsilon$ alike:
after a $784\to2$ projection, every later layer has a tiny fan-in and is
therefore always the cheaper buy, while the information destroyed by the
input projection is not recoverable downstream. Measured effective rank
confirms it is not a tuning artifact: width $4\to$ rank 4, $14\to11$,
$64\to19$, $256\to47$ — sublinear, so a wide late layer cannot restore what
a narrow early layer discarded.

**R2 is therefore not adopted.** Greedy per-layer ranking is myopic in
exactly the direction that matters, and no reweighting inside the greedy
frame repairs it. Growth direction falls back to **uniform widening**, which
spends parameters where the ranking provably cannot look. R1 decides *when*;
uniform widening decides *where*.

### 6.3 R3 — termination is Lemma 3.5 itself

Growth stops when $\varepsilon<\tfrac12$: the reachable set represents $r$,
Lemma 3.5 is satisfied, and the admissible-step machinery of the paper
applies without any structural change. This is what gives "optimal structure"
a precise meaning here — **the smallest structure reached whose reachable set
expresses the functional gradient** — and it is a criterion of the theory, not
an added stopping rule.

### 6.4 Measured result

From a $3\times2$ (2 hidden neurons per layer) on MNIST, batch 64, no budget
and no schedule, against the fixed-structure AdamW baseline it is compared
against:

| method | test | params | note |
|---|---|---|---|
| **certified grow, CE, R1 + uniform** | **90.25 %** | **6552** | $784\to8\to8\to10$, 2 growths |
| dense AdamW, fixed | 90.15 % | 8180 | $784\to10\to10\to10$ |
| dense AdamW, fixed | 90.50 % | 9862 | $784\to12\to12\to12$ |
| certified grow, CE, per-parameter (R2) | 64.4 % | — | $784\to2\to2\to14$ |
| certified grow, MSE | 86.1 % | 5269 | previous best under §4's conflict |

Both growth events fired for the R1 reason and are logged as such
("committed but eps did not decrease … the structure is at its
representation limit"), at epochs 2 and 4; the remaining 76 epochs added no
parameters, because $\varepsilon$ kept decreasing.

**Higher accuracy than the fixed-structure baseline with 20 % fewer
parameters**, discovered from a two-neuron start without being told anything
about the dataset.

### 6.5 What is and is not guaranteed

Claimed, and certified per step:

* every committed step realises held-out functional descent (Prop. 3.8);
* every accepted tangent step satisfies $\varepsilon<\tfrac12$ and a strict
  admissible learning-rate interval (Lemma 3.5, with $L_s=\tfrac12$ under CE
  giving a $4\times$ wider interval than MSE);
* growth fires only at a measured representation limit, and stops exactly
  when Lemma 3.5 is satisfied;
* convexity in $f$ (§3.1) means a functional-space stationary point is a
  global minimum of the *functional* problem.

**Not** claimed:

* global optimality of the **architecture**. The search is greedy in a
  one-dimensional family (uniform width) and there is no proof it reaches the
  parameter-optimal structure. A hand-picked $784\to6\to24\to24$ reaches
  89.95 % with 5728 parameters: better parameter efficiency than what the
  search found, at slightly lower accuracy. The search is *competitive with
  and cheaper than the baseline*, not provably optimal.
* the linear global contraction $C_{\mathrm{glob}}$ under CE — §3.3: no PL
  constant exists, and the $O(1/T)$ convex rate of §3.4 replaces it.
