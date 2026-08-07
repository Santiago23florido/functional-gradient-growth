# Stopping growth without a parameter budget

## The problem, restated from the measurement

Every run ended by exhausting its parameter allowance, never by deciding it
was done. That is not a cosmetic defect:

- On the easiest constructible target the method reached test 1.000 at **74
  parameters** and grew to **872** anyway, 25 growth events in 25 epochs.
- Three of four N=1024 seeds landed on **602 parameters** against a cap of
  600, and all four landed on `15-16-15`. That shape was never a choice — it
  is the point on a nearly diagonal trajectory where the allowance ran out.
- A capped seed then sat frozen for 34 of its 35 remaining epochs, unable to
  grow and unable to certify.

So while the cap exists, neither "this is the minimum-parameter architecture
for its accuracy" nor "the method does not bias toward one architecture across
datasets" can be claimed: the cap fixes the answer to both.

## Two candidates that died, and why they are kept in the code

| candidate | why it died |
|---|---|
| absolute threshold on the extension's singular values (`tiny_statistical_threshold`) | it is a magnitude, so it has to be re-guessed per dataset |
| Marchenko-Pastur (`growth_bottleneck_significance`) | needs `gamma = r/n` appreciable; here `r` is a hidden width (2-32) against `n = 1024` probe rows, so `gamma ~ 0.015` puts the edge at **1.26x the bulk** and it filters nothing. Uncapped, the shape degenerated to `20-32-2` |
| first-order comparison against `parameter_update_decrease` | invalid under function-preserving growth: growing does not move `f`, so no decrease is realised to compare against |

Both are still in `tangent.py`, off, so the negative results are not
rediscovered.

## The criterion: does the direction survive data it was not fitted on?

`crossfold_bottleneck_significance` (`fgdlib/tangent.py`). For each of `K`
folds of the probe:

1. fit the extension on the other `K-1` folds with the shipped update kwargs;
2. freeze the fitted pair `(alpha, omega)`;
3. score it against the held-out fold's **own** desired-update matrix `N`.

Step 3 is the whole point, and it is exact rather than analogous. GroMo builds
`alpha = sqrt(s) S^{-1/2} U` and `omega = sqrt(s) V` from the SVD of
`P = S^{-1/2} N`, so

```
<alpha omega, N>_F  =  trace(P_k^T P)  =  sum(s_i^2)
```

which is the bottleneck itself. Evaluated against a fold the direction never
saw, the same bilinear form therefore returns *this layer's bottleneck,
measured where the direction was not chosen*. Normalised by both Frobenius
norms it is a cosine `b_k` in `[-1, 1]`, so **magnitude cancels** and the rule
transfers between datasets unchanged.

The bar is a t statistic, not a level:

```
t = mean(b) / (std(b) / sqrt(K))          grow only while t > 1
```

A direction the data really asks for scores positive on every fold. A
direction fitted to sampling noise scatters around zero — or goes decisively
negative, which is what a solved task produces.

`t = inf` is reserved for NOT MEASURED (too few rows, or a structure the
identity is not derived for). A stopping rule that fires because it could not
measure would stop for the wrong reason.

### Where it attaches

`compute_expressivity_bottlenecks` is the single place the quantity is
computed, and all three growth paths read it. A layer that fails the test is
returned as `0.0`, which the callers **already** treat as "no width is the
bottleneck here". Nothing in `pipeline.py` or `certify.py` needed to change.

The criterion only ever turns a value INTO a zero, never moves one, so the
ranking among the layers that pass is the in-sample one, untouched.

That is **not** the same as "it never changes where". Zeroing is per layer, so
zeroing the argmax hands the turn to the runner-up. MEASURED on N=1024 seed 1:
it refused no growth at all, and still landed on `4-9-20` instead of `7-11-12`
with 22 growth events against 19. Growth halts only when every layer fails at
once; short of that the criterion redirects.

The `grow_until_certified` argmin of eps is untouched. That loop has a theorem
and substituting the bottleneck for it was MEASURED at 0.7380 mean with one
seed paralysed at 46 parameters.

## Measured: does it size to the task?

Two tasks of opposite difficulty, free budget, everything else the canonical
`configs/fgd/family_ladder_N1024.yaml`. Only the data knobs differ, so any
difference between runs is causal.

| run | parameters | widths | test acc | growth refused |
|---|---|---|---|---|
| easy (1 feature, f 0.5, no interaction), off | 270 | `6-9-16` | 0.9996 | — |
| **easy, K=5** | **158** | `5-6-12` | **1.0000** | **8 epochs** |
| hard (4 features, f 3.0, inter 0.6), off | 252 | `11-8-10` | 0.1581 | — |
| **hard, K=5** | **252** | `11-8-10` | 0.1581 | **0 epochs** |

The hard run is **identical** with and without the criterion — the test ran on
every growth decision and refused none. The easy run drops **41 % of its
parameters and its accuracy goes up**.

That asymmetry is the result. The criterion is silent where structure is
genuinely needed and bites where it is not, which is exactly what a parameter
cap cannot do, because a cap does not know which task it is cutting.

The measured statistic says the same thing directly. On the hard task, on the
layers carrying structure:

```
[CROSSFOLD] L1 t=5.6212 b=+0.1655 +0.0869 +0.1182 +0.2272 +0.2402
[CROSSFOLD] L2 t=4.7575 b=+0.0079 +0.0233 +0.0192 +0.0097 +0.0110
```

every fold positive. On the easy task once it is solved:

```
[CROSSFOLD] L0 t=-9.1313 b=-0.5752 -0.7837 -0.4210 -0.7310 -0.8043
```

every fold negative — the direction the in-sample bottleneck proposes actively
disagrees with data it did not see.

## Measured: the four N=1024 seeds, free budget

`model_seed` 0-3, `train_seed` 0, canonical config, no cap. Paired: each seed
run twice, changing only `growth_bottleneck_crossfold_folds`.

| seed | off acc | off params | off widths | on acc | on params | on widths | refusals |
|---|---|---|---|---|---|---|---|
| 0 | 0.9352 | 529 | `10-18-14` | 0.9282 | **378** | `9-14-12` | 1 |
| 1 | 0.8101 | 280 | `7-11-12` | 0.7964 | 286 | `4-9-20` | 0 |
| 2 | 0.5691 | 158 | `8-7-6` | 0.5691 | 158 | `8-7-6` | 0 |
| 3 | 0.8237 | 283 | `8-11-11` | 0.8237 | 283 | `8-11-11` | 0 |
| **mean** | **0.7845** | **312.5** | | **0.7794** | **276.3** | | |

**-11.6 % parameters for -0.005 accuracy**, and two of the four seeds are
bit-identical: on those the criterion ran on every growth decision and refused
nothing.

⚠️ **Read the accuracy column narrowly.** The canonical config is 25 epochs,
and all four seeds reach their best accuracy at the LAST epoch, so these runs
are limited by compute rather than by structure. The column therefore measures
"did gating growth hurt within this compute budget", not converged accuracy,
and is NOT comparable to the 0.9435 headline, which was 70 epochs under a
600-parameter cap.

The seed-2 and seed-3 identity is the informative part: a stopping rule that
transfers has to be inert where nothing is wrong, and it is.

## Cost, and the honest limits

- **Cost: +9.7 % of wall clock at K=5**, MEASURED on N=1024 seed 0 (145 s off,
  159 s on), not the 10x the arithmetic suggests. The accounted bucket is
  larger than the net: `growth_crossfold_seconds` is 21.0 s while the run only
  got 14 s longer, because a criterion that refuses growth also SAVES work --
  `where_total_seconds` fell from 4.48 to 2.27 with the same
  `tangent_system_total_seconds` (12.37 vs 12.45). Read it with
  `growth_crossfold_layers_tested` / `_rejected`, which separate "the test ran"
  (69) from "the test said no" (16). Lower `K` before changing anything else:
  `K` moves the variance of the statistic, not its bias.
- **The MNIST cost is NOT this cost.** Here the probe is 1024 rows of 4
  inputs. `mnist_matrix_free` is 704 rows of 784, and `mnist_streaming` about
  10,000 — `compute_statistics` scales with both, and the share has to be
  re-measured there before the criterion is trusted at that width.
- **The folds overlap.** The `K` training sets share `K-2` of their `K` parts,
  so the sample standard deviation understates the true spread and `t` is
  optimistic. The bar of 1.0 is therefore read as a sign test with a scale
  attached, not as a p-value. The separation measured above (5.9 against -9.1)
  is far wider than that bias.
- **False positives keep growth alive.** Stopping requires *every* layer to
  fail, so one layer passing by chance is enough to buy another neuron. This is
  why the criterion thins growth rather than halting it abruptly.
- **Tiny `r`.** At width 2 the extension has a single direction and `b_k` is one
  number per fold. `t` is still defined — the variance is taken across folds,
  not across the spectrum, which is precisely the dependency that killed MP.
