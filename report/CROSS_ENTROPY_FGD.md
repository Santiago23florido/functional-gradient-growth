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
5. The per-step measured-descent certificate, the family ladder, the growth
   trigger and the descent-per-parameter growth criterion need **no change**.
6. The accuracy metric is already argmax-based and needs no change; targets
   stay one-hot.

The honest summary: **(P1) and (P2) transfer, so the method and every
per-step certificate transfer; (P3) does not, so the global guarantee drops
from linear to $O(1/T)$.** In exchange, the certified objective stops
fighting the metric the model is judged on.
