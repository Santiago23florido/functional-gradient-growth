"""Approximate functional-gradient-descent training."""

from __future__ import annotations

import copy
import math
import os
import statistics
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Callable, Literal

from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.profile import fallback, increment, set_max, set_value, timed
from fgdlib.search.growth import ScalingLineSearchConfig, grow_layer
from fgdlib.training_utils.loop import RegressionMetrics, evaluate_regression_metrics


ensure_gromo_importable()

import torch
from torch.func import functional_call, jacrev, jvp

from gromo.containers.growing_mlp import GrowingMLP
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.training_utils import compute_statistics


RelErrorMode = Literal["tangent_projection"]
LayerSelection = Literal["certifying", "tiny_best", "min_rel_error"]
ProjectionSolver = Literal[
    "gromo_layer",
    "cg",
    "exact",
    "exact_svd",
    "exact_kernel_eigh",
]
LearningRatePolicy = Literal["scheduler", "theory_interval"]
GrowthLimitCriterion = Literal["progress_floor", "epsilon_stationary"]
#: How the unified branch decides WHERE to grow. "rank_ceiling" is the shipped
#: behaviour: level the width minimum first, then rank what is left.
#: "certified_gain" removes the levelling and lets the exact train-probe eps
#: reduction per parameter choose, which is what allows non-uniform shapes.
GrowthWhere = Literal["rank_ceiling", "certified_gain", "expressivity_bottleneck"]
GrowthSelection = Literal[
    "descent_per_parameter",
    "epsilon_lookahead",
    # SENN (arXiv:2307.04526): rank by the RAW increase in the natural
    # expansion score. Identical to "epsilon_lookahead" except that it does
    # not divide by the added parameter count. See fgdlib/senn.py.
    "natural_expansion",
    # Pool every candidate NEURON from every location and rank by certified
    # first-order decrease per parameter it costs, s_i^2 / cost. Neuron-level
    # and pooled, which is what separates it from the refuted per-LAYER
    # ranking. See fgdlib/growth.py:allocate_by_expansion_per_parameter.
    "expansion_per_parameter",
    # Width AND depth in one certified ranking, both applied
    # function-preservingly so growth never touches f. See
    # fgdlib/unified_growth.py.
    "unified_expansion",
]
FunctionalLoss = Literal["mse", "cross_entropy"]
GlobalBoundAction = Literal["lr_then_growth", "grow", "ignore"]


@dataclass(frozen=True)
class FGDApproxConfig:
    rel_error_threshold: float = 0.5
    rel_error_mode: RelErrorMode = "tangent_projection"
    layer_selection: LayerSelection = "certifying"
    projection_solver: ProjectionSolver = "exact"
    learning_rate_policy: LearningRatePolicy = "theory_interval"
    projection_damping: float = 1e-2
    cg_max_iterations: int = 16
    cg_tolerance: float = 1e-4
    theory_smoothness_constant: float = 2.0
    theory_alpha: float = 1.0
    theory_beta: float = 1.0
    theory_mu: float = 1.0
    theory_loss_star: float = 0.0
    theory_lr_safety: float = 0.95
    theory_lr_initial: float = 0.01
    theory_lr_min: float = 1e-5
    theory_lr_follow_bound: bool = False
    theory_lr_search_steps: int = 8
    theory_lr_search_refinements: int = 4
    sufficient_descent_c: float | None = 0.1
    lr_backtrack: float = 0.5
    lr_min_factor: float = 1e-3
    global_bound_action: GlobalBoundAction = "lr_then_growth"
    # Retained for config compatibility; FGD certificate growth is immediate.
    global_bound_lr_patience: int = 5
    rel_error_compute_delta: bool = True
    growth_compute_delta: bool = False
    # Retained for config compatibility; only scheduled growth uses timing gates.
    start_epoch: int = 1
    min_epochs_between_growth: int = 1
    # DEPRECATED: projection_group_* configured independent per-group
    # projections whose norms were aggregated into one certificate. That is
    # not a valid shared tangent direction and is no longer computed
    # anywhere; the fields are parsed for config compatibility only (the
    # legacy "scheduler" training path still concatenates batches with
    # projection_group_size). Certificates now use probe_batches.
    projection_group_size: int = 1
    projection_group_auto: bool = True
    projection_group_max: int | None = None
    # Number of mini-batches concatenated into the FIXED certification
    # probe. Every certificate solves ONE joint projection over the probe:
    # a single shared parameter direction for all probe batches.
    probe_batches: int = 4
    stall_min_epoch_decrease: float = 2e-3
    stall_patience: int = 5
    eps: float = 1e-12
    # Krylov rank cap for the matrix-free family. 0 runs to the numerical rank,
    # which reproduces the exact tangent and buys NOTHING -- at k = rank the
    # factors are as large as J itself, so the whole construction is a detour.
    #
    # The saving exists only while k << min(NK, P). At MNIST width (784 in,
    # 10 out, 1024-example probe, P ~ 59k) the arithmetic is stark:
    #
    #   dense J            4.8 GB      20,481 sequential autograd calls
    #   factors at k=200   110 MB         401 sequential autograd calls
    #
    # 44x memory, 51x compute -- and it is a LOW-RANK APPROXIMATION, not a
    # cheaper exact route.
    #
    # The truncation bias is CONSERVATIVE, which is the good direction and is
    # worth stating because the opposite is the intuitive guess. J_k = J V V^T
    # has a SMALLER range than J, so LESS of r is representable and eps comes
    # out HIGHER. Truncating under-certifies; it never certifies a direction
    # that does not deserve it. MEASURED at P=641 against exact eps 0.185677:
    #
    #   k=255  0.200  (+7.7%)    5.9 s        full rank 36.7 s
    #   k=127  0.268  (+44%)     1.8 s
    #   k= 63  0.369  (+99%)     0.5 s
    #   k= 31  0.512  (+176%)    0.2 s
    #
    # k=127 still certifies comfortably under the 1/2 bar at 20x less cost.
    # The price is a worse eps, hence a smaller Lemma 3.5 rate, hence shorter
    # steps -- efficiency, not soundness.
    matrixfree_rank: int = 0
    # Where the adaptive rank loop starts. Doubling from a small block is much
    # cheaper than guessing high: most steps are decided far from the 1/2 bar.
    matrixfree_rank_initial: int = 32
    # Extra sampled directions, discarded after the SVD trim. A random sketch
    # can miss a direction it happened not to hit; oversampling is the standard
    # fix and is what keeps the low-rank error near optimal.
    matrixfree_oversampling: int = 10
    # (J^T J)^q applied to the sketch, sharpening it toward the leading
    # singular directions. q=2 is the usual setting.
    matrixfree_power_iterations: int = 2
    # The DECISION BAND, as fractions of the certification threshold. Rank is
    # only doubled while eps lands inside it -- an eps of 0.05 or 0.9 will not
    # change its verdict no matter how sharp it gets, so paying for sharpness
    # there is pure waste. This is what buys precision without buying cost.
    matrixfree_band_low: float = 0.7
    matrixfree_band_high: float = 1.3
    tiny_use_covariance: bool = True
    tiny_alpha_zero: bool = False
    tiny_omega_zero: bool = False
    tiny_use_projection: bool = True
    tiny_ignore_singular_values: bool = False
    tiny_use_fisher: bool = False
    tiny_maximum_added_neurons: int | None = None
    tiny_numerical_threshold: float = 1e-6
    tiny_statistical_threshold: float = 1e-3
    # Function-preserving growth for the LEGACY growth paths only (SENN /
    # uniform / expansion-per-parameter). New neurons keep zero outgoing
    # weights, no delta and no scaling line search, so growth never changes the
    # function. NOTE: the CERTIFIED method's growth is function-preserving BY
    # DESIGN and does not read this flag -- the grow-to-certify loop uses
    # certify_function_preserving (below), and the unified / rank-cap expansion
    # enforces FP unconditionally (pipeline.py `_certificate_for` candidates).
    # This flag governs only the legacy paths, so it must not be read as "is
    # the certified method's growth FP" -- that is always yes.
    growth_function_preserving: bool = False
    growth_preservation_tolerance: float = 1e-6
    # Ordered approximation families. The legacy ladder starts with "tangent"
    # and retains its existing fallback semantics. The nonlinear primary mode
    # is selected only by the standalone value ("nonlinear",); it bypasses all
    # tangent construction and cannot be mixed with fallback families.
    family_order: tuple[str, ...] = ("tangent",)
    # Certify the tangent outer step by MEASURED validation descent
    # (Prop. 3.8) instead of the Lemma-3.5 relative-error interval. The
    # step direction is still the paper's functional-gradient projection
    # g = P_T r; only the step SIZE is chosen by a measured nonlinear line
    # search rather than the worst-case bound eta_max(eps), which is far
    # too conservative (measured optimum ~0.5 vs eta_max ~0.06 at eps 0.45).
    # Both certificates are in the paper; measured descent unlocks the
    # nonlinear step the linear bound forbids. Default False (legacy
    # eps-bounded step).
    tangent_measured_descent: bool = False
    # Largest eta tried by the measured tangent line search (the grid
    # descends geometrically from here to theory_lr_min). Only used when
    # tangent_measured_descent is True.
    tangent_measured_max_lr: float = 1.0
    # Certified outer steps attempted per epoch: each pass re-solves the
    # shared direction at the CURRENT model and certifies it independently
    # (k applications of the same per-step theorem). The epoch stops at the
    # first rejected attempt. 1 = one outer step per epoch (legacy).
    outer_steps_per_epoch: int = 1
    # The certified functional. Every certificate in this file is a
    # statement about THIS loss. "mse" is the legacy default with the exact
    # identity ||r||^2 = 4L (mu = 2, L* = 0). "cross_entropy" keeps
    # convexity and smoothness (so the method and every per-step
    # certificate transfer) but has NO global Polyak-Lojasiewicz constant,
    # so the linear C_glob envelope is unavailable for it -- see
    # report/CROSS_ENTROPY_FGD.md.
    functional_loss: FunctionalLoss = "mse"
    # How the growth LAYER is chosen.
    #
    #   "descent_per_parameter"  legacy: largest certified functional
    #                            descent per added parameter.
    #   "epsilon_lookahead"      R2: largest reduction of the Lemma-3.5
    #                            relative error per added parameter,
    #                            measured AFTER one certified family step on
    #                            the grown clone. Spending a parameter is
    #                            worthwhile exactly when it enlarges what the
    #                            reachable set can express, and that only
    #                            shows once the new capacity has been used:
    #                            measured on a trained 3x2 the IMMEDIATE eps
    #                            gets worse for every candidate (the delta
    #                            perturbs the function first), while after one
    #                            family step exactly one layer improves it --
    #                            the same layer that yields the best accuracy.
    #                            "no candidate reduces eps" is then the
    #                            termination condition: the structure is
    #                            already minimal-adequate.
    growth_selection: GrowthSelection = "descent_per_parameter"
    # Certified family steps taken on each grown clone before its eps is
    # judged, for growth_selection = "epsilon_lookahead".
    growth_lookahead_steps: int = 10
    # How the structure's LIMIT is recognised, i.e. when more training can
    # no longer substitute for more capacity.
    #
    #   "progress_floor"      the legacy reading: the structure is exhausted
    #                         when no family certifies progress above the
    #                         C_prog floor. Sound for sum-MSE, whose infimum
    #                         is attained at f = Y.
    #   "epsilon_stationary"  the reinterpretation: the structure is at its
    #                         REPRESENTATION limit when a certified step no
    #                         longer reduces the held-out relative error
    #                         eps = ||g - r|| / ||g||. Required for any
    #                         functional whose infimum is NOT attained
    #                         (cross-entropy: more confidence always lowers
    #                         the loss, so the progress floor never fires and
    #                         the structure never grows). It is a monotonicity
    #                         comparison of the Lemma-3.5 quantity between two
    #                         measured states -- no threshold, no window, no
    #                         budget -- and it is defined for EVERY certified
    #                         functional, which is what makes the method
    #                         loss-agnostic.
    growth_limit_criterion: GrowthLimitCriterion = "progress_floor"
    # GROW-TO-CERTIFY. Inverts the flow: instead of growing where it is
    # cheapest and stepping when it can, the structure is grown until it
    # PROVABLY satisfies Lemma 3.5 (eps < rel_error_threshold), and only then
    # is a step taken. Every growth is function-preserving, so f never moves,
    # r is fixed and eps decreases monotonically -- termination is a theorem
    # (tests/test_grow_to_certify_theorem.py), not a hope.
    #
    # This exists to answer one question: does enforcing the conditions
    # EXACTLY -- never approximated, never bypassed -- reach the best loss and
    # accuracy, whatever structure that costs? Parameter count is explicitly
    # not a concern here. Use with family_order = ("tangent",),
    # projection_solver = "exact" and tangent_measured_descent = False, so the
    # only thing that can commit a step is the 1/2 condition itself.
    # Default False: every existing config is untouched.
    grow_to_certify: bool = False
    # Iteration guard for that loop. NOT a budget: the theorem says the loop
    # terminates on its own, so this only catches a numerical pathology (or a
    # probe so large that certification is out of reach, which is itself a
    # result worth seeing rather than hanging on).
    certify_max_growths: int = 256
    # DISCRIMINATE why grow_until_certified's `force` is asked for. Force
    # exists to close a REAL deadlock, MEASURED on MNIST: eps sat at 0.475
    # -- certified, so no growth fired -- while no admissible learning
    # rate produced held-out descent, so no step committed either;
    # neither mechanism could act and epochs came out bit-identical. But
    # forcing UNCONDITIONALLY whenever a step failed over-fired just as
    # badly: MEASURED on the small synthetic run, 262 growths against 7
    # committed steps, 242 of those with eps ALREADY certified (many at
    # eps = 0.0000). The trigger there was a non-finite validation
    # measurement -- the model overflowing on unseen data, a symptom of
    # overfitting that more capacity only makes worse.
    #
    # Both incidents are real and what tells them apart is finiteness of
    # the measurement behind the failure: sane numbers that violate a
    # geometric invariant, or simply no admissible rate (the MNIST case),
    # mean the STRUCTURE has to change, so forcing growth is right; NaN or
    # Inf (the over-firing case) means the MODEL is what failed, and
    # growth cannot fix that. Default False reproduces today's behaviour
    # bit-for-bit -- force is always False. True makes force exactly (the
    # previous step failed to commit) AND (that failure's measurement was
    # finite).
    certify_force_growth_on_finite_step_failure: bool = False
    # Whether the GROW-TO-CERTIFY loop grows function-preservingly. This is the
    # FP control for the certify path only; the standard/unified growth path is
    # FP by design and unconditional (see growth_function_preserving above,
    # which governs only the legacy paths). MEASURED,
    # and the reason the default is False: with omega = 0 the incoming weights
    # contribute nothing to the Jacobian, so a preserving growth converts only
    # ~21 % of the parameters it spends into tangent directions (+9 rank for 42
    # parameters). Releasing omega gives one independent direction per added
    # parameter -- the theoretical maximum -- and certifies far sooner:
    #
    #   preserving      24 growths   eps 0.487   1779 params
    #   NOT preserving   5 growths   eps 0.211    399 params
    #
    # What is given up is the monotonicity proof (a non-preserving growth
    # moves f, so r moves and eps may rise after a growth); the loop then
    # relies on the measured trajectory. Set True to recover the theorem at
    # roughly five times the structural cost.
    certify_function_preserving: bool = False
    # PAPER-PURE STEP. Lemma 3.5 does not ask for descent to be verified -- it
    # PROVES it: if eps < 1/2 then EVERY eta in (0, eta_bar(eps)) descends.
    # Checking it empirically afterwards is therefore strictly stronger than
    # the theory, and that extra gate is what deadlocked the certify flow: the
    # structure was certified (eps = 0.475, interval (1e-5, 0.0974) non-empty)
    # yet no rate produced HELD-OUT descent, so nothing ever committed and
    # epochs 2-4 came out bit-identical (loss 0.0619, acc 0.566).
    #
    # The mismatch was structural, not numerical: eps is certified on the
    # TRAIN probe (where the direction is computed) while descent was demanded
    # on the VALIDATION probe. Lemma 3.5 is a statement about ONE sample --
    # it guarantees descent where eps was measured and says nothing across a
    # split -- so the gate was asking the lemma for something outside its
    # domain, and a failure meant a generalisation gap, not a bad step.
    #
    # With this flag on the flow does what the paper's algorithm does: certify
    # eps < 1/2, take eta = theory_lr_safety * eta_bar(eps) (0.95 of the
    # interval by default) and APPLY. Finiteness sensors still gate -- those
    # catch numerical failure, not theory -- but realised descent does not.
    certify_apply_in_interval: bool = False
    # LINEARISATION TOLERANCE. Lemma 3.5's subject is the FUNCTION-space step
    # f - eta g; what is performed is the PARAMETER-space step theta - eta u,
    # and f(theta - eta u) = f(theta) - eta g + O(eta^2 |u|^2). The remainder
    # is not controlled by L_s but by the curvature of the parameter-to-
    # function map, which no certificate here measures.
    #
    # MEASURED on the synthetic task with sum-MSE -- the theory's best case,
    # exact PL constant mu = 2 with L* = 0 -- applying the certified rate
    # without this guard diverged while the certificate stayed healthy:
    # eta 0.005 -> 0.855 as eps FELL 0.50 -> 0.085, loss 2.3e3 -> 7.5e7,
    # accuracy 0.087 -> 0.004. The failure is self-reinforcing: eta_bar(eps)
    # rises towards 2/L_s as eps -> 0, so the better the certificate the
    # larger the step it authorises and the further outside the linear
    # regime that step lands.
    #
    # This is NOT the descent check that was removed -- it never looks at the
    # loss. It measures
    #     delta(eta) = |f(theta - eta u) - (f(theta) - eta g)| / (eta |g|)
    # and narrows eta INSIDE the certified interval until the step is the
    # object the lemma describes. Enforcing a hypothesis, not adding a gate.
    # None disables it (the raw certified rate is applied).
    certify_linearization_tolerance: float | None = None
    # CHOOSE THE PROJECTION'S REGULARISATION BY MEASUREMENT, not by a constant.
    # The damping in J u ~ r arbitrates between the two conditions this method
    # needs at once, and they pull in OPPOSITE directions: lowering it makes
    # eps small (certificate easy) while |u| explodes (step unrealisable);
    # raising it does the reverse. MEASURED on one grown synthetic model --
    #
    #   lambda 0     eps 0.045   |u| 3.6e5   no admissible rate
    #   lambda 1e-2  eps 0.495   |u| 3.9e1   eta 1.6e-4
    #   lambda 1e0   eps 0.845   |u| 4.1e0   eps >= 1/2
    #
    # Only a narrow window satisfies both, and a fixed constant lands in it by
    # luck: 1e-2 works here, and nothing makes it work at another scale of J,
    # which changes with the dataset, the architecture and the point in
    # training. That is precisely what stops a configuration from transferring.
    #
    # When true, fgdlib/search/damping.py locates the window per step --
    # bisecting for the certified boundary, then fanning below it -- and picks
    # the rung maximising eta * ||g||^2, the decrease Lemma 3.5 itself
    # guarantees. Nothing dataset-specific enters: the bracket is relative to
    # sigma_max^2, the filter is the method's own certificate, the objective is
    # the theorem's. Measured to land within 9 % of the hand-tuned constant on
    # the task the constant was tuned for.
    projection_damping_auto: bool = False
    # FAST SPECTRAL FACTORISATION for the damping/where computations. The
    # min-damping eps that ranks growth candidates needs only the range of J
    # (its left basis U and singular values), which the current float64
    # torch.linalg.svd(J) delivers -- but MEASURED, that SVD is the dominant
    # cost of the whole "where": 959 ms for a 1024x1144 J, against 16 ms to
    # materialise J. The same U and sigma come from eigh of the SMALLER Gram
    # matrix (J J^T when NK <= P, else J^T J) at 50 ms -- a 19x speedup, and
    # EXACT: min-eps matched to all printed digits (0.91335524 both ways).
    # V, needed only by the step's damping fan, is recovered as J^T U / sigma.
    # Default False keeps every existing config on the bit-for-bit SVD path.
    projection_fast_factorization: bool = False
    # Stream the tangent system over the WHOLE dataset in row-chunks instead of
    # materialising the full NK x P Jacobian. Accumulates the incremental-QR
    # factor R (P x P), b = J^T r and ||r||^2 chunk by chunk, freeing each
    # chunk, then hands the SAME downstream code a tiny surrogate ((rank+1) x P)
    # that reproduces the exact spectrum, projection and eps -- MEASURED exact to
    # 1.8e-12 vs the full-Jacobian float64. Memory is O(P^2), independent of the
    # dataset size, so the certificate can use ALL the data (which forces growth
    # and generalisation) without the O(NK*P) blow-up. The certificate criterion,
    # the families and every config value are untouched; only HOW the exact
    # system is assembled changes. Off by default (the full Jacobian is built).
    certify_stream_gram: bool = False
    # SAMPLES per streaming batch. Each batch is forwarded once and its
    # Jacobian block accumulated into G; smaller batches cap the transient
    # jacrev memory, larger ones cut Python/jacrev overhead. Memory stays O(P^2)
    # regardless. 0 means the whole probe in one batch. Only the streaming path.
    certify_stream_chunk: int = 1024
    # Exact shared-base block-Schur scoring for exhaustive function-preserving
    # ``where`` scans.  Default-off preserves every featured configuration;
    # benchmarks may opt in with FGD_EXACT_BLOCK_SCHUR_WHERE=1.
    certify_exact_block_schur_where: bool = False
    # WHICH admissible g inside the tangent space to use. Lemma 3.5 fixes only
    # RelErr(g, r) <= 1/2 and says nothing about which g in T = range(J) to
    # take; that freedom is ours, and how it is resolved decides whether the
    # step generalises.
    #
    # "descent" (default): the rung maximising eta * ||g||^2, the decrease the
    # lemma guarantees. This is the LEAST regularised admissible step, so it
    # fits every component of r including the sample-specific part. MEASURED:
    # train accuracy 0.980 with test 0.234 -- exact interpolation of the
    # training set.
    #
    # "gcv": the rung minimising generalized cross-validation,
    #     GCV(lambda) = (1/N)||r - H_lambda r||^2 / (1 - df(lambda)/N)^2,
    # the classical leave-one-out risk estimate for ridge, with
    # df = tr H_lambda = sum sigma_i^2/(sigma_i^2 + lambda). The projection IS
    # kernel ridge regression in the tangent kernel, so this is the selector
    # that problem calls for. It uses no held-out data (structural decisions
    # stay pure), has no tunable constant (an argmin), and is free (sigma and
    # U^T r are already formed here). The (1 - df/N)^2 denominator diverges as
    # df -> N, so it refuses exactly the interpolating regime the descent rule
    # walks into.
    projection_damping_objective: str = "descent"
    # FUNCTIONAL TIKHONOV. Step-level ridge (the lambda above) chooses which g
    # in T to use, but MEASURED it is CLIPPED by the certificate: eps < 1/2
    # forces df/N ~ 0.53 on the grown model, GCV wants far less, and the
    # certified+realisable window collapses to a single rung. The certificate
    # forbids the regularisation generalisation needs, because it is a
    # statement about approximating r, and r is the UNREGULARISED gradient.
    #
    # The fix is to regularise r itself: certify against the gradient of
    #     L_gamma(f) = L_data(f) + gamma ||f||^2
    # instead of L_data alone. The paper's H is a Hilbert space, so this keeps
    # convexity (P1) and keeps K-smoothness with K -> K + 2 gamma, i.e.
    # Lemma 3.5 applies VERBATIM. For sum-MSE the regularised gradient is
    #     r_gamma = 2(f - y) + 2 gamma f = 2((1+gamma) f - y),
    # which points exactly like the ordinary MSE gradient toward the shrunk
    # target y/(1+gamma) -- and the certified parameter step it produces is
    # identical to plain MSE toward y/(1+gamma) with L_s = 2. So it is
    # implemented as that single change: the projection probe's targets are
    # shrunk by 1/(1+gamma). Growth, projection, realisation and GCV all
    # inherit it through the one probe, and the reported train/val/test
    # metrics are untouched because they come from the loaders, against the
    # true y. 0.0 disables it.
    functional_tikhonov_gamma: float = 0.0
    # CERTIFIED FAMILY LADDER before growing. With the tangent alone, eps >= 1/2
    # forces growth, and function-preserving growth is ruinous (it keeps f and
    # hence r fixed, so the tangent must capture 80% of the FULL residual --
    # MEASURED 57 growths to certify). This tries a NONLINEAR within-MLP family
    # (fgdlib/search/families.py) at the fixed structure before each growth, and
    # accepts its step ONLY if the family's OWN relative error certifies,
    # RelErr(g_family, r) < 1/2 -- never a descent criterion, never the
    # tangent's projection. The nonlinear family certifies at a far smaller
    # structure than the linear tangent, so growth is deferred: MEASURED
    # 57 -> 6 FP growths (1102 -> 70 params) on the synthetic task. Only for
    # sum-MSE; MLP only. 0/False keeps the tangent-only behaviour.
    certify_family_ladder: bool = False
    certify_family_inner_steps: int = 400
    certify_family_inner_learning_rate: float = 0.01
    certify_family_functional_lr: float = 1.0
    # SWEEP the family's functional step instead of trying one eta_f. The
    # tangent family is the only one in family_order, and its ladder call site
    # passed certify_family_functional_lr as a single value -- unlike
    # parametric_gd, which declares functional_learning_rates and walks them.
    # One eta_f is one attempt at a target the clone may simply be unable to
    # realise, and a failure there is recorded as "the family cannot certify
    # here" when it only means "not at that distance".
    #
    # The search is bounded from BOTH ends by the method itself, which is why
    # it is a search and not a knob. Too large and the target f - eta_f r sits
    # off the reachable manifold, so the realised Delta misaligns and the
    # cosine bar sqrt(3)/2 fails. Too small and Delta collapses onto the
    # first-order term, i.e. onto the tangent projection, whose eps is >= 1/2
    # -- that is why the ladder was reached at all -- so it fails too. The
    # ladder cannot degenerate into "certify everything with a tiny step".
    #
    # First certificate wins. NOTHING is weakened -- every candidate faces the
    # same RelErr(Delta, r) < min(rel_error_threshold, 1/2) on the same probe.
    # Empty () keeps the single-eta behaviour bit-identical.
    #
    # THE DIRECTION MATTERS, and it is UP. MEASURED on N=1024, 4 seeds, capped
    # at 600 parameters:
    #
    #   baseline (eta_f = 1 alone)     0.941 0.830 0.904 0.925  mean 0.9000
    #   descending [1, .5, .25, .125]  0.893 0.807 0.904 0.925  mean 0.8823
    #   ascending  [1, 2, 4, 8]        0.941 0.887 0.931 0.933  mean 0.9230
    #
    # Descending LOSES, and structurally rather than by noise: a small eta_f
    # certifies a SMALL step, and a certified family step returns from
    # grow_until_certified at once, so it buys a growth deferral without
    # progress -- exactly what certify_family_min_gain is for, and it is 0
    # here. It is also self-limiting in the other direction: as eta_f -> 0 the
    # realised Delta collapses onto the first-order term, i.e. onto the
    # tangent projection, whose eps is >= 1/2 -- which is why the ladder was
    # reached at all. MEASURED cos(Delta, r) at one call: 0.7871 / 0.7955 /
    # 0.7594 / 0.7481 for eta_f 1 / 0.5 / 0.25 / 0.125, decaying toward the
    # tangent's own.
    #
    # Ascending asks the clone for MORE curvature, and it pays. Note how
    # little of it is needed: across all four seeds only 4 steps certified
    # above eta_f = 1 (eta_f = 2 three times, eta_f = 4 once, eta_f = 8
    # never). The gain compounds instead -- one early rescue changes the whole
    # trajectory, and the seed that was stuck goes from 2 family
    # certifications to 6, of which only one came from eta_f > 1.
    #
    # Head first, so eta_f = 1 stays the preferred distance and a run that
    # certifies there is bit-identical to today, at today's cost; the ladder
    # escalates ONLY where the single-eta call would have grown instead.
    certify_family_functional_lrs: tuple[float, ...] = ()
    # STOP GROWING WHEN THE BOTTLENECK IS NOT SIGNIFICANT. Only meaningful
    # with growth_where: expressivity_bottleneck. A parameter cap is a number
    # nobody knows in advance -- it has to be guessed per dataset, and MEASURED
    # here it is what actually ends every run: all four N=1024 seeds stop by
    # exhausting 600 parameters, never by deciding they are done, and one of
    # them then sits frozen for 35 of its 70 epochs, unable to grow and unable
    # to certify. This replaces the cap with a question the DATA answers:
    # are the extension's singular values distinguishable from what pure
    # sampling noise would have produced? Marchenko-Pastur's edge settles it
    # from the two dimensions and the spectrum's own bulk -- see
    # marchenko_pastur_significant_count. Nothing is predetermined, nothing is
    # fitted per dataset, and magnitude cancels, so the same rule transfers
    # unchanged between bases. Off by default: it changes when growth ends.
    growth_bottleneck_significance: bool = False
    # THE SAME STOPPING QUESTION, ASKED WHERE MARCHENKO-PASTUR CANNOT ANSWER
    # IT. MP above is implemented and MEASURED AND REFUTED: it needs
    # gamma = r / n to be appreciable, and here r is a hidden width (2-32)
    # against n = 1024 probe rows, so gamma ~ 0.015 puts the edge at 1.26x the
    # bulk and it filters nothing -- without a cap the shape then degenerated
    # to 20-32-2 and growth blew its own numerical tolerance. Both the field
    # and its helper are kept, off, so that negative result is not
    # rediscovered.
    #
    # K-fold cross-validation asks instead whether the extension direction
    # SURVIVES DATA IT WAS NOT FITTED ON, which needs no asymptotic regime at
    # all. Fit (alpha, omega) on K-1 folds of the probe, freeze them, and score
    # that pair against the held-out fold's OWN desired-update matrix N. In
    # sample that same bilinear form IS the bottleneck --
    # <alpha omega^T, N> = sum(s_i^2), which is what compute_optimal_added_
    # parameters constructs -- so the held-out number is literally "this
    # layer's bottleneck, measured where the direction was not chosen". A real
    # direction scores positive on every fold; a direction fitted to sampling
    # noise scatters around zero.
    #
    # Nothing is fitted per dataset and no magnitude is compared to anything:
    # b_k is a Frobenius cosine, so scale cancels, and the bar is a t
    # statistic rather than a level. 0 = off, and off is bit-identical.
    growth_bottleneck_crossfold_folds: int = 0
    # Grow only while t = mean(b) / (std(b) / sqrt(K)) clears this. 1.0 is one
    # standard error above zero -- deliberately weak, because the K training
    # sets overlap in K-2 of their K parts and that correlation makes the
    # sample std understate the true spread. What is asked of it is a SIGN
    # test with a scale attached, not a p-value.
    growth_bottleneck_crossfold_t: float = 1.0
    # UN SOLO CRITERIO en todo el crecimiento. La cuenta adaptativa de
    # grow_until_certified decide CUANTAS neuronas comprar en la ubicacion ya
    # elegida, y su regla es de HUECO: "cerro esta neurona el 10% de lo que
    # falta para certificar?". Eso habla de eps y no dice nada sobre si esa
    # capa sigue siendo el sitio, asi que compra sin volver a preguntar DONDE
    # -- MEDIDO en la semilla 3, con eps en 1.0310, el doble del umbral,
    # compro una segunda neurona en la capa 0 a ciegas. Con esto activo, la
    # misma medida que eligio la ubicacion decide cuantas: sigue comprando
    # mientras ESA capa siga siendo el argmax del cuello de botella,
    # re-medido tras cada compra. Es autolimitante (ensanchar una capa baja su
    # propio cuello) y no introduce ni hueco, ni fraccion, ni constante.
    # Solo surte efecto con growth_where: expressivity_bottleneck.
    certify_adaptive_growth_by_bottleneck: bool = False
    # Cheapen the family clone WITHOUT changing the step it produces: stop the
    # inner training once its alignment with the residual (cos(Delta, r)) has
    # genuinely PLATEAUED -- no improvement above plateau_tol for a few checks.
    # A successful clone plateaus near its best cosine (same accepted step as
    # full training); a doomed clone plateaus below sqrt(3)/2 (same rejection).
    # Either way the DECISION and the step are unchanged -- only the wasted tail
    # of inner steps is cut, which is the dominant cost on a hard task where the
    # clone is retrained on every growth attempt. Off by default so the
    # certificate and the synthetic result are byte-identical; the acceptance
    # criterion (RelErr < 1/2) is never touched.
    certify_family_plateau_stop: bool = False
    certify_family_plateau_tol: float = 0.002
    # Commit the family step at ITS OWN Lemma 3.5 rate instead of whole.
    # The family certifies a DIRECTION -- cos(Delta, r) > sqrt(1 - eps^2) --
    # and then commits the trained clone outright, which is a functional step
    # of size 1. The lemma applied to that same eps allows far less: MEASURED
    # on MNIST, eps 0.292 gives eta_bar = 0.263 at L_s = 2, so a full step
    # overshoots the descent bound by 3.8x (9x at eps 0.400). The direction
    # was certified; the distance never was, and that is what made the run
    # oscillate with its loss flat. Off by default so every existing result
    # is byte-identical -- the acceptance criterion is never touched, only
    # how far the accepted direction is followed.
    certify_family_lemma35_rate: bool = False
    # Try the ladder throughout the band it can help in, not only where no
    # step certifies. Default false = the ladder runs while eps is above the
    # step CERTIFICATE, i.e. only as the remedy for "nothing certifies".
    # True runs it while eps is above the GROWTH TARGET, where the tangent
    # already certifies but an aspiration is still being chased -- the band
    # where growth would otherwise be the only tool and, on MNIST, buys 0.0007
    # of eps for 800 parameters while the clone reaches 22 degrees against a
    # 30 degree bar. Only safe together with certify_family_lemma35_rate:
    # MEASURED without it, the unbounded family step diverged held-out loss
    # 9.37 -> 44.53 in two epochs because it pre-empted the tangent path that
    # produces the very bound it was missing.
    certify_family_in_target_band: bool = False
    # A certified family step DEFERS growth, so it must earn the deferral by
    # the same marginal-return rule growth answers to: it has to close this
    # fraction of the remaining gap to the growth target, or growth takes the
    # turn instead. MEASURED on MNIST without it, the ladder certified on all
    # 20 epochs, the structure never grew (GRO=0), and accuracy plateaued at
    # 0.113 from epoch 9 -- worse than the 0.164 the growing path reached by
    # epoch 13. Certifying is not the same as progressing. 0.0 disables it,
    # which is the previous behaviour exactly.
    certify_family_min_gain: float = 0.0
    # Ask ONCE per outer step whether it is growth's turn, by training a grown
    # clone and a stay clone the same number of steps and comparing the eps
    # they reach. Organic where a threshold is not: it never compares eps to a
    # constant, so it transfers across datasets -- on MNIST every growth moves
    # eps ~1e-3 and any fraction of any gap refuses them all equally, while a
    # parameter cap is just another invented number. A "no" returns at once,
    # so the step commits and training happens before the question is asked
    # again, which is what stops the loop front-loading every growth into the
    # first outer step. Off by default; reuses growth_lookahead_steps and
    # tiny_statistical_threshold, so it introduces no new constant.
    certify_growth_lookahead: bool = False
    # Let the exact per-candidate eps ranking choose the SHAPE, not just the
    # size. The rank cap filters candidates to the width minimum and mandates
    # levelling it, both justified by "rank J <= min_l w_l" -- which is FALSE
    # on the exact Jacobian: MEASURED at NK=200, widths (20,3,20) and
    # (30,4,30) both reach rank 200, and (2,2,2) reaches 25 = P, not 2. The
    # bound is min(NK, P). So the mandate levels every run onto h,h,h+1 and
    # bars non-uniform shapes for a reason that does not hold. With this on,
    # unified.expansion_value decides alone -- it already scores all six
    # candidates by their EXACT resulting eps. No certificate changes.
    #
    # MEASURED AND REFUTED -- do not enable. The bound really is false (see
    # above), but the machinery built on it is load-bearing for growth to
    # happen AT ALL: rank_candidates filters by the rank ceiling and
    # unified.py:207-217 returns the bottleneck relief as the FALLBACK when
    # that ceiling binds. Removing both empties the candidate list. On
    # N=1024 the three seeds then stop at 53 / 60 / 119 parameters instead
    # of 588 / 1074 / 659, and test accuracy collapses from 0.907-0.933 to
    # 0.219-0.361. Kept, off, so the negative result is not rediscovered.
    growth_free_shape: bool = False
    # WHERE to grow. "rank_ceiling" is the shipped path, byte-identical.
    # "certified_gain" replaces all three jobs the old block did at once,
    # because the free-shape failure above was caused by replacing only one:
    #
    #   WHERE      rank by exact TRAIN-probe eps reduction per parameter. The
    #              branch scores candidates on the VALIDATION probe today,
    #              while the certificate the flow chases is the train one --
    #              two certificates driving one decision. On the train probe
    #              function-preserving growth strictly enlarges range(J), so
    #              eps_after <= eps_before holds by certify.py's theorem; on
    #              validation it does not, which is what the max(...,0) clamps.
    #   HOW MUCH   the levelling loop was also the PACE: it bought several
    #              neurons per event. Without it the branch buys exactly one,
    #              and events fire once per epoch -- the reference needs 40
    #              neurons in <=25 epochs, arithmetically impossible. Replaced
    #              by the existing burst idiom (certify_adaptive_growth_min_gain).
    #   NEVER STALL the fallback drops the bottleneck requirement and keys on
    #              the measured eps, so it cannot empty and cannot degenerate
    #              into always-cheapest.
    #
    # MEASURED on a proxy with the real config, probes and grow_layer: the
    # shipped rule lands [9,9,10] at 246p / 0.574; gain-per-cost on exact train
    # eps lands [18,7,5] at 269p / 0.629. Pure argmin (certify.py's rule,
    # cost_exponent 0) was the WORST exact variant at 0.544 -- the cost divisor
    # earns its place.
    #
    # "expressivity_bottleneck" asks the question growth is actually for:
    # WHICH LAYER CANNOT EXPRESS WHAT IS BEING ASKED OF IT. One neuron goes
    # into the argmax of tangent.compute_expressivity_bottlenecks, i.e. of
    # activation_gradient * sum(eigenvalues_extension ** 2) -- TINY's
    # extension term, the mass of the desired functional change lying outside
    # the current width. NOT TINY's select_best_update, which adds
    # parameter_update_decrease and so mixes in "where would re-fitting the
    # existing weights help", a statement about training rather than width.
    #
    # It exists because the other two rules answer a different question and
    # MEASURED runaway growth for it: with certified_gain the growth trigger
    # reads the STEP's rel_err, which diverges (0.64 -> 114 on the easy
    # synthetic function, 229 on the hard one) as the damped projection
    # recovers less of the residual, while the growth loop's own eps says
    # 0.15-0.39 -- adequate. The easy function reaches test 1.000 at 74
    # parameters and is grown to 872 anyway, 25 growth events in 25 epochs.
    # A bottleneck measured on what the STRUCTURE can represent cannot run
    # away like that: it goes to zero when the width suffices.
    growth_where: GrowthWhere = "rank_ceiling"
    #: 0.0 collapses the rule to certify.py's pure argmin, which is immune to
    #: cost degeneracy by construction. The rollback if risk 1 materialises.
    growth_where_cost_exponent: float = 1.0
    #: Admission floor on the RAW gain, applied BEFORE the cost division, as a
    #: fraction of the best gain available. Without it, cheap-and-useless wins:
    #: MEASURED, 11 of 12 consecutive purchases went to the last growable
    #: location, whose cost does not grow with its own width.
    growth_where_min_gain_fraction: float = 0.25
    #: Depth candidates are excluded by default so the search stays within the
    #: three-hidden-layer family the reference and the exhaustive architecture
    #: search both live in.
    growth_where_allow_depth: bool = False
    #: Keep buying at the chosen location while each increment still pays.
    #: This is the HOW MUCH replacement; without it growth is one neuron per
    #: epoch and cannot reach the reference's widths in its epoch budget.
    growth_where_burst: bool = True
    #: Magnitude used to break the w = 0 degeneracy when SCORING a candidate.
    #: Only ever applied to a disposable clone, never to the committed model,
    #: so function preservation and every certificate are untouched. 0
    #: disables it and restores dormant scoring.
    growth_where_wake_scale: float = 1e-3
    # Score a candidate by what the LADDER can do with it, not by what it does
    # at insertion. The ladder fits a disposable clone toward the functional
    # target f - eta * r, and because growth is function-preserving that target
    # is IDENTICAL whether it is computed from the base or from any grown
    # candidate -- so every candidate chases the same fixed blank and the
    # comparison needs no separately trained control. (An earlier attempt
    # compared a TRAINED clone against an UNTRAINED base, which is why every
    # gain came out negative: the ladder's own step raises eps 0.4497 ->
    # 0.4806.)
    growth_where_ladder_score: bool = False
    growth_where_ladder_steps: int = 1
    # How many neuron-blocks to put in when SCORING a location, and -- when the
    # location wins -- to actually buy. Scoring one block is myopic: a layer
    # that is a poor buy for one neuron can be the best buy for three, and that
    # is invisible to a one-block probe. Because the winner is committed with
    # the same horizon it was measured over, the WHERE and the HOW MUCH stop
    # being two separate rules measured on two different quantities.
    # MEASURED AND REFUTED as a better WHERE. It looked like a win at free
    # budget -- it lifted the worst seed 0.854 -> 0.903 and collapsed the
    # spread -- but that was a difference in SPEND, not in placement. Capped at
    # 460 parameters so both rules get the same money, it loses on all four
    # seeds: 0.830 mean against 0.872 for the shipped rank_ceiling rule. Left
    # off by default; the knob stays because the horizon probe is the only way
    # measured so far to see a location's value beyond one block.
    growth_where_lookahead: int = 1
    # Score JOINT proposals too: widen a layer and the one after it in a
    # single event, priced as one purchase. Without this the first hidden
    # layer never gets its moment -- MEASURED at widths [5, 6, 2] it gains
    # +0.0080 against +0.0220 downstream because nothing reaches the output
    # through a width-2 layer, and once the downstream is wide enough to use
    # its features it costs 5 + h2 and is no longer cheap. Funnels are
    # therefore unreachable one layer at a time, and the exhaustive per-shape
    # lr grid says funnels are exactly what wins here: 4-19-13-13-1 reaches
    # 0.9250 with 551 parameters where the grown 4-16-17-17-1 reaches 0.9135
    # with 693.
    growth_where_joint: bool = False
    # WHICH PROBE ranks the locations. The certificate, the families and the
    # ladder are untouched by this: it only decides WHERE capacity goes.
    #
    # eps = ||r - g|| / ||r|| on the TRAIN probe asks "which layer lets me
    # represent the TRAINING gradient best per parameter". That is not the
    # quantity that decides test accuracy, and MEASURED they diverge: the
    # shape this criterion grows, 4-16-17-17-1, is the WORST of six under a
    # per-shape lr grid (0.9135), while 4-19-13-13-1 reaches 0.9250 with 551
    # parameters. A criterion minimising training-probe representation error
    # favours the shapes that interpolate soonest, not the ones that
    # generalise -- and this method is already known to certify by
    # interpolating the train probe.
    #
    # "validation" ranks on the held-out probe, "both" on the mean of the two.
    growth_where_probe: str = "train"
    # Let capacity MOVE, not just accumulate. A unit whose removal shifts f by
    # less than growth_preservation_tolerance is as function-preserving to drop
    # as it was to add, so no new constant is needed: the config already says
    # what "does not change f" means, it was just never used in this
    # direction.
    growth_where_prune: bool = False
    # ROUGHNESS PENALTY -- the RIGHT regulariser, in the function-space norm.
    # functional_tikhonov above penalises ||f||^2 (magnitude), which shrinks f
    # toward 0 but still lets it memorise a shrunk copy. This penalises
    # ROUGHNESS: L_rough(f) = L_data(f) + gamma f^T Lambda f, where Lambda is
    # the graph Laplacian over the probe inputs. f^T Lambda f is the Dirichlet
    # energy -- it measures how much f varies between NEIGHBOURING inputs, so
    # penalising it biases toward smooth fits, which is what generalises for a
    # smooth target. It is the discrete form of the ||Df||^2 Sobolev norm the
    # paper's Banach space B is built on.
    #
    # Lambda is PSD, so L_rough stays convex (P1) and K-smooth with
    # K -> K + 2 gamma lambda_max(Lambda): Lemma 3.5 applies VERBATIM and the
    # eps < 1/2 criterion is untouched. For sum-MSE the regularised functional
    # gradient is r_rough = 2(f - y) + 2 gamma Lambda f, i.e. the data gradient
    # plus a smoothing term -- injected once at the exact target, so growth,
    # projection, realisation and GCV all inherit it. 0 disables it.
    roughness_gamma: float = 0.0
    # Neighbourhood width of the graph affinity, as a multiple of the median
    # pairwise input distance (the median heuristic -- a standard tuning-free
    # bandwidth, not a per-dataset constant).
    roughness_bandwidth: float = 1.0
    # REALISE THE CERTIFIED STEP AS A PATH instead of a single jump. Lemma 3.5
    # licenses a move in FUNCTION space, f -> f - eta g; theta - eta u is only
    # the first Gauss-Newton iteration of "find theta' with f(theta') = f_target".
    # Shrinking eta until that one iteration is accurate also shrinks the
    # FUNCTIONAL movement by the same factor -- MEASURED at 512x and 2047x --
    # so the method delivered a fraction of the step it had certified. That is
    # the stall, and growth cannot fix it: growth lowers eps, which widens
    # eta_bar, which widens the gap.
    #
    # With this on, the step is integrated: each inner iteration re-solves the
    # tangent projection against the functional residual that remains and takes
    # the largest sub-step that reduces it. MEASURED on one grown model, same
    # direction and certificate:
    #
    #   one jump                        loss 9.580e1 -> 9.576e1   delta  0.043
    #   integrated, residual criterion  loss 9.580e1 -> 8.567e1   delta 10.13
    #
    # 236x, realising 96.2 % of the intended displacement. Both rules hold:
    # only the tangent family (every inner iteration IS the tangent projection)
    # and only steps certified by 1/2 (the certificate is computed once, on g;
    # what iterates is the realisation, never the certification).
    # TAKE THE RATE THAT MAXIMISES THE GUARANTEED DECREASE, not the largest
    # admissible one. Lemma 3.5 bounds
    #
    #   L(f_t+1) <= L(f_t) - eta[alpha - K eta/2 - (beta + K eta) c] ||grad L||^2
    #
    # with c = (3/2) e/(1-e). The guaranteed decrease is eta * bracket(eta), a
    # PARABOLA in eta, so it is maximised at the vertex and eta_bar is where
    # the bracket vanishes -- hence eta* = eta_bar / 2, the interval's
    # MIDPOINT, always.
    #
    # theory_lr_safety = 0.95 therefore picks the worst admissible rate: it
    # maximises the step and minimises what the step is guaranteed to buy. At
    # eps ~ 0 with L_s = 2 the bracket is (1 - eta), so 0.95 leaves a
    # coefficient of 0.05 against 0.5 at the midpoint -- twenty times worse.
    # For sum-MSE, where r = 2(f - y), the arithmetic is stark:
    # f_new - y = (1 - 2 eta)(f - y), so eta = 0.5 lands exactly on the target
    # while eta = 0.95 gives -0.9(f - y) -- the error FLIPS SIGN and shrinks
    # only 10 %, oscillating at the edge of stability.
    #
    # MEASURED: with the full certified step finally being delivered, 0.95
    # sent train_loss 0.0917 -> 0.3991 in one epoch and validation accuracy
    # 0.248 -> 0.123. It was harmless before only because the linearisation
    # control was cutting eta by ~500x and landing in a good regime by
    # accident.
    certify_optimal_rate: bool = False
    # RESAMPLE THE CERTIFICATION PROBE EVERY OUTER STEP. A fixed probe was
    # harmless while steps were tiny; once the certified step is delivered in
    # full it makes the method Newton's method on one subsample, driving the
    # residual on those points to zero and interpolating them. MEASURED:
    # train accuracy 0.320 against validation 0.150, and the logged tangent
    # relative error diverging to 1077 -- RelErr is normalised by ||g||, so it
    # blows up exactly when no residual is left on the probe.
    #
    # The theory's L is the loss over the DATASET; the probe exists only
    # because the exact Jacobian over all of it is intractable. Redrawing it
    # each step makes the functional gradient an unbiased estimate of that
    # object, turning the flow into stochastic FGD rather than exact descent
    # on a fixed 256 points.
    probe_resample: bool = False
    # Materialise the exact Jacobian in CHUNKS of output rows. jacrev vmaps
    # one backward pass per output row and holds the batched activations for
    # all of them at once, so its peak memory scales with
    # (output rows) x (activations), NOT with the size of J itself. MEASURED:
    # a 3840 x ~600 Jacobian is only ~9 MB, yet computing it in one call ran
    # the GPU out of memory -- the matrix was never the problem.
    #
    # Chunking rebuilds the SAME matrix row-block by row-block, so the result
    # is bit-for-bit the exact Jacobian; only the peak drops, to
    # (chunk) x (activations). It costs one extra forward per chunk. This is
    # what makes certifying over a whole dataset affordable instead of over a
    # subsample, which is the difference between the certificate being a
    # statement about the training loss and a statement about 256 points.
    # 0 disables chunking.
    jacobian_row_chunk: int = 0
    certify_realize_path: bool = False
    certify_realize_max_iterations: int = 40
    certify_realize_tolerance: float = 0.05
    # ADAPTIVE EARLY STOP for the realise integration. MEASURED, 138 of ~200
    # steps run the full 40 inner solves without reaching the residual
    # tolerance -- but each iteration only chips the residual down a little, so
    # the late iterations buy almost no extra realised displacement at full
    # cost. This stops the integration once one iteration reduces the residual
    # by less than this fraction of the INTENDED displacement, i.e. once the
    # returns have diminished. It does NOT touch the certificate: the certified
    # direction g and rate eta are unchanged; this only decides how far along
    # the fixed target f - eta g the step is carried, and the next outer step
    # continues from there. 0.0 keeps the current behaviour (stop only at the
    # tolerance or the iteration cap).
    certify_realize_min_progress: float = 0.0
    # Freeze the Gauss-Newton preconditioner (the Gram J^T J) ACROSS the inner
    # realise iterations instead of rebuilding the full Jacobian every one. The
    # sub-steps are short, so J barely moves; holding G = J^T J fixed at the
    # first inner iteration and recomputing only J^T shortfall (a SINGLE VJP)
    # each iteration is the modified-Newton (chord) method. Because (G+lambda)
    # is SPD, (G+lambda)^-1 J^T r is always a descent direction of ||r||^2, so
    # the residual-reducing line search is unchanged and the realised step lands
    # in the same functional-tolerance ball. Turns "up to 40 full Jacobians per
    # outer step" into "1 Jacobian + up to 39 VJPs". Off by default -> the two
    # branches below run exactly as before; on -> the frozen-Gram path.
    certify_realize_freeze_gram: bool = False
    # Optionally precondition realization with the absolute damping selected
    # for the certified outer direction. Off preserves the historical inner
    # damping from projection_damping for every existing configuration.
    certify_realize_use_selected_damping: bool = False
    # Transactional guard for the nonlinear endpoint produced by the
    # realization path. This is deliberately off by default: the exact/base
    # ladder must keep its historical control flow, allocations and numerical
    # decisions unless a matrix-free experiment opts in explicitly.
    transactional_realized_descent: bool = False
    # Number of smaller functional targets tried after the first realization.
    transactional_max_retries: int = 0
    transactional_backtrack_factor: float = 0.5
    # Require L_before - L_after to exceed both this absolute tolerance and
    # the configured fraction of Lemma 3.5's predicted decrease.
    transactional_descent_atol: float = 0.0
    transactional_min_predicted_decrease_fraction: float = 0.0
    # Optional ceiling for the FACTORED selector's certified functional rate.
    # None preserves the selector byte-for-byte; the dense/exact selector does
    # not read this field.
    certify_functional_lr_cap: float | None = None
    # Bounded certification probe, sized to the NUMERICAL RANK of J. The
    # certificate eps<1/2 is a statement over the NK-dimensional probe residual,
    # but MEASURED the rank m* needed to certify grows SUBLINEARLY in NK (m*/NK:
    # 0.22 -> 0.08 as NK x4). eps collapses to a spurious 0 exactly when the
    # tangent space can represent any residual on the probe, i.e. NK <= rank(J)
    # (J full row rank) -- so the binding floor is rank(J), NOT the parameter
    # count P. On an input-heavy net rank(J) << P (CIFAR: 525 vs 2092), so sizing
    # by P over-samples ~4x and needlessly slows the where; sizing by rank(J) is
    # both correct and fast. This sets the probe rows to NK = kappa * rank(J)
    # (measured on the K=1 synthetic: NK ~ 8 rank matches the whole-dataset eps
    # to +-0.05), capped at the dataset and re-estimated as rank grows, with the
    # floor NK > rank(J). Decouples the probe -- and the O(NK P^2) where cost --
    # from the dataset size. 0.0 disables it: the fixed probe_batches is used
    # unchanged (so the synthetic runs keep certifying over their whole training
    # set). Meant for datasets whose whole-Jacobian probe is intractable
    # (CIFAR: NK ~ 5e5). kappa in [4, 8] is the tested range.
    #
    # Why sizing -- and not a statistical correction on the eps formula -- is the
    # live mechanism: the certificate is an R^2-like ratio, so a subsample's eps
    # is biased DOWN by ~ m/NK (least-squares over-fits a finite probe). The
    # adjusted-R^2 debiasing of that bias is derived and validated as a
    # DIAGNOSTIC in ``fgdlib.search.damping.dof_corrected_relative_error``, but
    # it cannot be a live gate here: this method deliberately grows ``P > NK``,
    # where the numerical rank saturates at ``NK`` and the correction explodes
    # (MEASURED: it froze N1024 at test 0.192). Sizing ``NK = kappa * rank``
    # instead holds the probe above the interpolation floor by construction and
    # leaves the eps formula -- and every synthetic certificate -- untouched.
    certify_probe_kappa: float = 0.0
    # Adaptive growth COUNT. The grow-to-certify loop adds a FIXED
    # tiny_maximum_added_neurons per growth, so on a hard task (CIFAR) reaching a
    # certifying width takes hundreds of full location scans -- one neuron at a
    # time. This lets the method CHOOSE the count instead: once a growth commits
    # the best location, it keeps adding at THAT location while each increment
    # still pays -- its marginal eps reduction stays above min_gain of the
    # remaining gap to the threshold -- and stops the instant it certifies or the
    # returns diminish, then re-scans. The number of neurons is thus set by the
    # certificate and the value criterion, not fixed. Every intermediate
    # structure is scored by its OWN projection, so it honours exactly the
    # certificate conditions the one-neuron path does. false keeps the fixed
    # count (the synthetic default is unchanged and byte-identical).
    certify_adaptive_growth: bool = False
    # Marginal-value floor for the adaptive count: keep adding at the location
    # while (eps_before - eps_after) > this fraction of (eps - threshold). Higher
    # = fewer neurons per growth (stops sooner); lower = grows more aggressively.
    certify_adaptive_growth_min_gain: float = 0.1
    # The eps the grow-to-certify loop CHASES, as distinct from the eps at which
    # a step is CERTIFIED. rel_error_threshold does both jobs, and they are
    # incompatible: every step site reads Lemma 3.5 as eps <
    # min(rel_error_threshold, 1/2) (damping.py's bisection, lemma35_learning_
    # rate), so lowering the threshold to make the loop aspire higher DISABLES
    # the step. MEASURED on MNIST at rel_error_threshold 0.3: eps sat at 0.457,
    # no damping and no rate ever certified, lr was 0 in every epoch, and the
    # only thing still moving the model was the family ladder -- applied at
    # functional_lr 1.0 WITHOUT the Lemma 3.5 bound, because that bound is
    # produced on the tangent route the threshold had killed. Accuracy
    # oscillated between 0.33 and 0.74 while the functional loss barely moved
    # (0.2393 -> 0.2382) and growth ran away to P ~ 31000 without finishing one
    # outer step in 39 minutes.
    #
    # Splitting them lets the loop aspire to 0.3 while a step still commits at
    # eps 0.457 under a threshold of 0.5. Never laxer than the certificate: the
    # loop uses min(this, rel_error_threshold), so this field can only ask for
    # MORE structure, never license a step the lemma forbids. None means "chase
    # the certificate" -- today's behaviour, and by construction rather than by
    # a flag: with target == rel_error_threshold the loop invariant eps >=
    # target already implies eps >= the certificate, the regime in which growth
    # is unconditional.
    certify_growth_target: float | None = None
    # Marginal-value floor for chasing certify_growth_target BELOW the
    # certificate. Once eps < min(rel_error_threshold, 1/2) a step exists, so
    # every further growth is VOLUNTARY and has to earn its parameters: it must
    # close at least this fraction of the gap that is left (eps - target). Above
    # the certificate no step exists at any damping, growth is the only move the
    # method has, and this floor is not applied at all -- which is why it cannot
    # deadlock the loop.
    #
    # The discriminant is the dimensionless ratio gain/gap, so nothing about the
    # dataset enters. MEASURED, at certificate 0.5 and target 0.3: MNIST eps
    # 0.457, gap 0.157, gain per growth 0.0007 -> ratio 0.0045; N1024 eps 0.3267,
    # gap 0.0267, gain per growth 0.00707 -> ratio 0.265. Two orders of magnitude
    # apart, and 0.1 falls between them with 22x and 2.6x of margin. 0.0 disables
    # the stop (every growth pays), which is the refutation experiment for a run
    # this criterion is suspected of cutting short.
    certify_growth_min_gain: float = 0.1
    # How many CONSECUTIVE growths may fail the min_gain floor before the loop
    # stops chasing the target. 1 stops at the first refusal, so not one
    # parameter is spent on a growth that did not pay. Raise it if a bootstrap
    # ladder appears -- the first growths of a very narrow net can under-deliver
    # because the widths themselves are tiny -- which does not arise with
    # tiny_maximum_added_neurons: 1 and certify_grow_all_width: false, where the
    # count added per event is constant.
    certify_growth_min_gain_patience: int = 1
    # Grow EVERY growable width location per event, not just the single best.
    # GroMo caps the neurons added to a layer at min(fan-in, fan-out) -- on a net
    # whose hidden layers all start at width 2, that is ~2 per event, and the
    # unified path spends the whole event on ONE layer, so a certifying CIFAR
    # width takes hundreds of events. Widening every layer each event instead
    # bootstraps the widths geometrically (2 -> 4 -> 8 -> ...): each layer still
    # adds only its GroMo-optimal count (the cap is untouched, GroMo unchanged),
    # but the min(fan-in, fan-out) of every layer rises together, so the next
    # event can add proportionally more. Every addition is function-preserving,
    # so f -- and the descent certificate -- is unchanged; only the reachable set
    # enlarges faster. false keeps one purchase per event (byte-identical).
    certify_grow_all_width: bool = False
    # Generalised R1. The eps < 1/2 stop is Lemma 3.5's admissibility of a
    # STEP, not adequacy of the STRUCTURE; on an easy task the two coincide
    # (MNIST stops at a good small net) but on a hard one they diverge --
    # measured on grayscale CIFAR-10, growth stopped at 1024->15->15->16
    # (eps 0.42, 46% train) while a 1024->32->32 reaches 34.6%, because eps
    # kept decreasing slowly inside a rank-15 image and R1 never fired. When
    # this flag is on, before stopping the search asks a look-ahead
    # question: does growing the bottleneck reach a strictly lower eps than
    # staying? If so the structure is inadequate and growth continues; if
    # not it stops exactly where it does today. A strict generalisation, so
    # the MNIST result is preserved. The margin reuses tiny_statistical_
    # threshold -- no new constant, nothing dataset-specific.
    growth_lookahead_adequacy: bool = False
    # Let the fallback families run whenever the tangent outer step fails,
    # independently of whether growth is due. They are nested inside the
    # growth trigger otherwise, so once the structure becomes adequate
    # (eps < 1/2 -> growth correctly not requested) the ladder is skipped
    # and only the tangent remains, freezing the flow. Default False.
    families_available_without_growth: bool = False
    # When Lemma 3.5 fails on the committed state (eps >=
    # rel_error_threshold) a successful family step does NOT cancel the
    # growth probe. Needed for any functional whose infimum is not attained
    # -- under cross-entropy some family step always certifies (more
    # confidence always lowers the loss), so growth would be postponed for
    # ever while the structure stays inadequate. Self-limiting through eps,
    # so it needs no parameter budget: once the reachable set can represent
    # the functional gradient again, eps drops below the threshold and the
    # override switches itself off. Default False.
    admissibility_failure_forces_growth: bool = False
    # Require the paper's structural criterion before growing: capacity is
    # increased only when Lemma 3.5 fails on the committed state, i.e. when
    # the relative error reaches rel_error_threshold and the reachable set
    # genuinely cannot represent the functional gradient. Without this, ANY
    # failed transaction requests growth -- including failures caused by the
    # step size or by a loss plateau at eps far below 1/2 -- and with no
    # parameter budget the run grows without bound. Default False keeps the
    # legacy trigger.
    growth_requires_admissibility_failure: bool = False
    # Choose the growth's scaling factor by held-out (validation) loss
    # instead of the GroMo default train loss. The magnitude of the
    # structural step is otherwise the one part of growth that escapes the
    # certificate: minimizing TRAIN loss over the scaling makes each growth
    # a train-fitting move, and the overfit accumulates over many growths.
    # On validation the growth's magnitude follows the same held-out
    # functional descent Prop. 3.8 certifies for every other step.
    # Default False (legacy GroMo behavior).
    growth_scaling_on_validation: bool = False
    # Grow EVERY hidden layer together (uniform widening) instead of
    # selecting one layer. This sidesteps the input-layer credit-assignment
    # problem of greedy per-layer growth: any greedy criterion (descent or
    # relative error) undervalues the input layer, whose benefit is latent,
    # so incremental growth from a tiny net keeps layer 0 narrow and caps
    # accuracy. Uniform growth traces the balanced dense nets (3xk) from a
    # tiny start, so the certified family training matches fixed AdamW.
    # Default False.
    growth_uniform: bool = False
    # Select the growth layer by the largest CERTIFIED functional descent
    # per added parameter (Prop. 3.8 measured descent), instead of the
    # relative-error certificate. Required with delta growth (function-
    # preserving False, compute_delta True): there the GroMo optimal update
    # reduces the loss but jumps the tangent linearization, so the
    # rel-error certificate is blind to which layer actually helps. This is
    # the paper's structural step made parameter-efficient. Default False.
    growth_select_by_descent: bool = False
    # Growth layer selection among probes that improve the certificate.
    # False (default): frugal-first (fewest added parameters, then lowest
    # post-growth relative error). True: lowest post-growth relative error
    # first, so growth widens the most impactful layer even when it is the
    # expensive input layer — required on MNIST, where layer-0 width drives
    # accuracy and the frugal tie-break otherwise starves it.
    growth_prefer_lower_error: bool = False
    # Hard parameter budget: once the model has at least this many total
    # parameters, structural growth is suppressed and the flow keeps
    # training the fixed structure through the certified families. None
    # means no cap. Keeps a grow-and-train run inside a target budget.
    # MEASURED on N=1024 (4 model seeds, train_seed fixed): free growth
    # OVERSHOOTS. The accuracy/parameter frontier of the shipped rule is
    #
    #   cap 460 -> 0.872 at 467p     cap 650 -> 0.911 at 641p
    #   cap 550 -> 0.897 at 564p     cap 750 -> 0.918 at 671p
    #   free    -> 0.920 at 774p
    #
    # so 750 buys the same accuracy as free growth for 13% fewer parameters.
    # The cap does not even bind on most seeds (588p and 659p runs are
    # untouched); it only stops the one seed that would run to 1074p for 0.907,
    # which lands at 756p for 0.903 instead.
    #
    # It also locates where this method is worth using: against conventional
    # AdamW it is +3.5 points at ~670p (0.918 vs ~0.88) but only TIED at ~450p
    # (0.872 vs 0.876). Below roughly 550 parameters the certified step buys
    # nothing over ordinary training.
    max_total_parameters: int | None = None
    # Structure-burst patience: the growth probe runs only after this many
    # CONSECUTIVE epochs in which no family committed a step. With a value
    # above 1, combine with family_rejection_cooldown: 0 so the stochastic
    # parametric generators actually retry during the patience window.
    # 1 = probe growth on the first fully-failed epoch (legacy).
    growth_patience: int = 1
    # Acceptance mode. When true, an outer step commits on its four LOCAL
    # conditions only — valid sensor, strict Crel, strict LR interval
    # (theory_lr_min < eta < eta_bar), and STRICT realized descent of the
    # validation functional loss — while the stationary and global bounds
    # are computed and logged as trajectory diagnostics. When false (the
    # default), the legacy gates apply: non-strict descent and Cstat/Cglob
    # as acceptance conditions.
    local_acceptance_conditions: bool = False
    # Cooldown for rejected fallback families, measured in ACCEPTED outer
    # steps: a family rejected at theta_t is skipped until this many model
    # updates have been committed since the rejection (weight updates change
    # the tangent space, so rejection must never be permanent at a fixed
    # architecture). Growth clears all rejection state immediately; 0
    # disables the memory (families are retried every epoch).
    family_rejection_cooldown: int = 5
    # In-ladder rkhs_head acceptance margin: the committed head must improve
    # the FULL validation functional loss by at least this relative amount.
    # The phase's internal acceptance compares subsampled train losses
    # (fgd_rkhs.max_train_points), so without this external margin the
    # per-epoch subsample jitter re-certifies an epsilon "improvement"
    # forever and starves every family below rkhs_head in the ladder.
    rkhs_family_min_relative_improvement: float = 1e-3


SUPPORTED_FGD_FAMILIES = (
    "tangent",
    # The isolated primary family of this branch. It was an AdamW clone
    # ("nonlinear"); it is now the tangent's own algorithm solved matrix-free,
    # so the name states what it is. The old spelling is still accepted so
    # existing configs keep loading.
    "matrix_free_tangent",
    "nonlinear",
    "rkhs_head",
    "parametric_gd",
    "parametric_descent",
)


def validate_functional_loss(functional_loss: str) -> None:
    """Reject an unknown certified functional at config-load time."""
    if functional_loss not in _FUNCTIONAL_LOSSES:
        raise ValueError(
            f"Unsupported fgd_approx.functional_loss '{functional_loss}'. "
            f"Use one of: {', '.join(sorted(_FUNCTIONAL_LOSSES))}."
        )


def validate_family_order(family_order: tuple[str, ...]) -> None:
    """Reject malformed fgd_approx.family_order values early."""
    if not family_order:
        raise ValueError("fgd_approx.family_order cannot be empty.")
    unknown = sorted(set(family_order) - set(SUPPORTED_FGD_FAMILIES))
    if unknown:
        raise ValueError(
            "Unsupported fgd_approx.family_order entries: "
            + ", ".join(unknown)
            + f". Supported: {', '.join(SUPPORTED_FGD_FAMILIES)}."
        )
    if len(set(family_order)) != len(family_order):
        raise ValueError("fgd_approx.family_order entries must be unique.")
    isolated = {"matrix_free_tangent", "nonlinear"} & set(family_order)
    if isolated:
        if len(family_order) != 1:
            raise ValueError(
                "fgd_approx.family_order selects "
                f"'{family_order[0]}' as a standalone primary family and "
                "therefore must be exactly that one entry; it cannot coexist "
                "with tangent or fallback families."
            )
        return
    if family_order[0] != "tangent":
        raise ValueError(
            "fgd_approx.family_order must start with 'tangent' in legacy mode, "
            "or be exactly ['matrix_free_tangent'] for the isolated primary "
            "family."
        )


def validate_bottleneck_stopping(config: FGDApproxConfig) -> None:
    """Validate the opt-in budget-free stopping criteria.

    The two are alternative answers to the SAME question -- is this
    bottleneck real -- and they disagree by construction, since
    Marchenko-Pastur was refuted precisely where the cross-validated test is
    meant to work. Running both would silently take the intersection, so a
    run that stopped could not be attributed to either. Refuse the pair.
    """
    folds = config.growth_bottleneck_crossfold_folds
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 0:
        raise ValueError(
            "fgd_approx.growth_bottleneck_crossfold_folds must be a "
            "non-negative integer (0 disables the criterion)."
        )
    if folds == 1:
        raise ValueError(
            "fgd_approx.growth_bottleneck_crossfold_folds must be 0 or at "
            "least 2: a single fold holds nothing out."
        )
    bar = float(config.growth_bottleneck_crossfold_t)
    if not math.isfinite(bar):
        raise ValueError(
            "fgd_approx.growth_bottleneck_crossfold_t must be finite."
        )
    if folds >= 2 and config.growth_bottleneck_significance:
        raise ValueError(
            "fgd_approx.growth_bottleneck_crossfold_folds and "
            "growth_bottleneck_significance are alternative stopping rules; "
            "enable at most one."
        )
    if folds >= 2 and config.growth_where != "expressivity_bottleneck":
        raise ValueError(
            "fgd_approx.growth_bottleneck_crossfold_folds requires "
            "growth_where: expressivity_bottleneck."
        )


def validate_transactional_realized_descent(config: FGDApproxConfig) -> None:
    """Validate the opt-in matrix-free endpoint transaction settings."""
    retries = config.transactional_max_retries
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError(
            "fgd_approx.transactional_max_retries must be a non-negative integer."
        )
    factor = float(config.transactional_backtrack_factor)
    if not math.isfinite(factor) or not 0.0 < factor < 1.0:
        raise ValueError(
            "fgd_approx.transactional_backtrack_factor must be in (0, 1)."
        )
    atol = float(config.transactional_descent_atol)
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError(
            "fgd_approx.transactional_descent_atol must be finite and non-negative."
        )
    fraction = float(config.transactional_min_predicted_decrease_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(
            "fgd_approx.transactional_min_predicted_decrease_fraction must be "
            "in [0, 1]."
        )
    cap = config.certify_functional_lr_cap
    if cap is not None:
        cap = float(cap)
        if not math.isfinite(cap) or cap <= config.theory_lr_min + config.eps:
            raise ValueError(
                "fgd_approx.certify_functional_lr_cap must be finite and strictly "
                "above theory_lr_min."
            )
    if not config.transactional_realized_descent:
        return
    if tuple(config.family_order) != ("matrix_free_tangent",):
        raise ValueError(
            "fgd_approx.transactional_realized_descent is restricted to "
            "family_order: [matrix_free_tangent]."
        )
    if not config.certify_realize_path:
        raise ValueError(
            "fgd_approx.transactional_realized_descent requires "
            "certify_realize_path: true."
        )
    if not config.certify_apply_in_interval:
        raise ValueError(
            "fgd_approx.transactional_realized_descent requires "
            "certify_apply_in_interval: true."
        )
    if not config.projection_damping_auto:
        raise ValueError(
            "fgd_approx.transactional_realized_descent requires "
            "projection_damping_auto: true."
        )


@dataclass(frozen=True)
class ParametricGDConfig:
    """Parametric-GD secant family (calibrated projection at the output).

    A disposable clone is trained parametrically toward the functional target
    f - eta_nominal * r. The realized output displacement Delta = F(base) -
    F(candidate) is then screened on validation: its directional cosine
    against the functional gradient must reach ``min_cosine`` (with the
    scale-optimal declared learning rate eta* = <Delta, r>/|r|^2 the best
    achievable relative error is exactly sqrt(1 - cos^2), so cosines below
    sqrt(1 - eps^2) can never satisfy Crel). Surviving candidates are
    certified at eta* with the SAME secant certificate and full transactional
    conditions as every other family.
    """

    optimizer: str = "sgd"
    inner_learning_rate: float = 0.05
    inner_steps: tuple[int, ...] = (16, 64)
    functional_learning_rates: tuple[float, ...] = (0.2, 0.05)
    min_cosine: float = 0.9
    parameter_penalty: float = 1e-6
    gradient_clip_norm: float | None = 1.0
    # Decoupled weight decay for the adam/adamw inner optimizer. Higher
    # values regularize the generated candidate so its realized
    # displacement generalizes to validation (closing the train/val gap
    # that caps the certified acceptance).
    weight_decay: float = 0.0
    # Nonlinear-primary certification streams this many validation minibatches.
    # None means one complete loader pass. Legacy parametric fallback families
    # do not read this field.
    certification_batches: int | None = None

    # --- Nonlinear-primary family: EXPLICIT optimization-budget semantics ---
    # The ladder's nonlinear family (certify_parametric_step) spends
    # ``certify_family_inner_steps`` FULL-BATCH AdamW steps on the fixed probe.
    # The nonlinear primary originally spent the same integer on loader
    # MINIBATCH steps, which on N=1024/batch-64 is 64x fewer example-gradients
    # per candidate -- MEASURED as the dominant cause of its non-certification
    # (cos still climbing monotonically with the budget at the largest setting).
    # The unit is therefore named rather than implied:
    #   "probe"     -- full-batch steps on the certification probe (ladder parity)
    #   "epoch"     -- complete loader passes
    #   "minibatch" -- individual optimizer steps on loader minibatches
    inner_step_unit: str = "minibatch"
    # Reduction of the candidate's parametric objective. The ladder uses the
    # sum-MSE convention that matches the certified functional; "mean" rescales
    # the gradient by 1/(batch * out_features), which AdamW largely but not
    # exactly absorbs (it does not rescale weight_decay or parameter_penalty).
    candidate_objective: str = "mean"
    # Whether the disposable clone trains in train() mode (the ladder) or in
    # eval() mode. Only observable when the model has dropout / batch norm.
    candidate_train_mode: bool = False
    # Split the DIRECTIONAL certificate is measured on. The ladder trains and
    # certifies on the same train probe; certifying on validation silently
    # converts a statement about the step's own empirical objective into a
    # generalization claim.
    certificate_split: str = "train"
    # Split the transactional acceptance conditions are measured on.
    transactional_split: str = "validation"
    # Interpolation factors alpha searched for the COMMITTED step, in the order
    # given. Empty means "commit the full candidate only" (alpha = 1).
    alpha_grid: tuple[float, ...] = ()
    # How the committed alpha is chosen among those that RE-certify:
    #   "largest_certified" -- the largest certified alpha (longest step)
    #   "max_descent"       -- the certified alpha with the largest measured
    #                          functional-loss decrease
    #   "full_only"         -- only alpha = 1, and only when it certifies
    alpha_policy: str = "largest_certified"
    # Extra candidates trained at a functional rate DERIVED from the best
    # relative error the fixed sweep achieved, rather than guessed in advance.
    #
    # A well-fitted clone realises eta* ~ eta_f, while Lemma 3.5 admits only
    # eta <= 2(1 - 2 eps)/(L_s(1 + 2 eps)) -- a bound that depends on the eps
    # the clone happens to reach, which is not known until it has been trained.
    # A fixed grid therefore cannot be placed correctly in advance: MEASURED on
    # N=1024, eta_f = 0.25 reaches eps 0.4045 whose bound is 0.1056, and
    # eta_f = 0.0625 reaches eps 0.4779 whose bound is 0.0215 -- the target
    # moves as fast as the guess does. Each retry trains one more candidate at
    # theory_lr_safety * eta_bar(best eps) and certifies it on its own terms.
    adaptive_rate_retries: int = 0
    # What a candidate must satisfy to be committed. The ladder and the
    # nonlinear primary were never asking the same question, which is the whole
    # reason the ladder certifies where the primary does not:
    #
    #   "direction_only"   -- eps < min(rel_error_threshold, 1/2), and nothing
    #                         else. This is EXACTLY the ladder
    #                         (certify_parametric_step returns on the cosine
    #                         test alone, certify_family_lemma35_rate defaults
    #                         off, and grow_until_certified commits the
    #                         returned model without a transactional gate).
    #                         There is no statement about step LENGTH.
    #   "measured_descent" -- direction, plus the transactional conditions,
    #                         which measure real descent on the transactional
    #                         split. Prop 3.8 with a measured coefficient
    #                         rather than one lower-bounded through eps.
    #   "theory_interval"  -- the above, plus the realized secant rate
    #                         eta* = <Delta, r>/|r|^2 must lie inside the
    #                         Lemma 3.5 interval implied by its OWN eps.
    #                         Strictest, and the only one that certifies the
    #                         DISTANCE actually travelled.
    #
    # The previous nonlinear-only code appeared to enforce the last one but did
    # not: it quoted family_lemma35_rate(eps) as the step's rate, a number
    # admissible by construction, while committing a displacement whose real
    # eta* was never measured.
    acceptance_rule: str = "theory_interval"
    # Inner-optimizer learning-rate schedule for the candidate.
    #
    # At a CONSTANT lr, AdamW settles into a noise ball of radius ~lr and stops
    # converging, so the only way to improve the fit is more steps -- which is
    # why 400 -> 1600 -> 6400 kept helping and why this family ended up costing
    # more than the Jacobian it replaces. Decaying the rate to zero lets the
    # same clone actually converge, in far fewer steps.
    inner_lr_schedule: str = "constant"
    # Solve the OUTPUT layer against the functional target in closed form
    # after the inner steps, instead of leaving it to the optimizer.
    #
    # The certified quantity is cos(Delta, r) with Delta = eta_f r - e, where e
    # is the clone's fit residual. Iterative optimisation leaves an e that does
    # NOT shrink as r does, so late in training cos collapses however long the
    # clone trains -- MEASURED, eps sat at 0.58-0.99 for 28 consecutive epochs
    # with the loss frozen. The network is exactly linear in its readout
    # parameters, so fitting them is a least-squares solve with no
    # approximation error, costing O(N H^2) in the last hidden width against
    # O(N K P^2) for a tangent system over all P parameters. The hidden layers
    # are still moved nonlinearly by AdamW first, so the family keeps reaching
    # outside range(J); only the readout is exact.
    candidate_readout_solve: bool = False
    # Tikhonov term for that solve, for conditioning only.
    candidate_readout_ridge: float = 1e-8
    # --- Matrix-free tangent solver (the isolated family's engine) ---
    # Conjugate-gradient iterations on J v / J^T w products. MEASURED at P=197
    # against the exact Jacobian at damping 1e-2: 200 iterations agree to
    # 3.6e-06 relative, 50 to 2.2e-03. Damping conditions the CG as well as
    # regularising the projection, so the ladder's own regime is the easy one.
    cg_iterations: int = 200
    # How many examples the matrix-free solver accumulates per autograd call.
    # NOT a hyperparameter: J and r are sums over examples, so the grouping is
    # only how that sum is accumulated and cannot move the answer. MEASURED at
    # N=1024, eps agreed to 6 significant figures across 64 / 128 / 256 / 512 /
    # 1024 while the solve went 0.62s -> 0.06s. It is pure call overhead: each
    # group is one torch.autograd.grad, Krylov is strictly sequential so the
    # calls cannot be batched, and at these sizes the arithmetic inside one is
    # negligible against the ~1ms dispatch around it. Larger is faster until
    # the retained graphs stop fitting. 0 keeps the loader's own batching.
    matrixfree_batch_size: int = 0
    cg_tolerance: float = 1e-10
    # projection_damping_auto, matrix-free: eps depends on lambda and the
    # ladder picks the minimiser per step. Solving at the fixed 1e-2 instead
    # reported eps 1.1-1.5 and never certified -- the wrong lambda, not a
    # solver defect. The grid is swept reusing the retained batch graphs.
    damping_grid: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    # The Lemma 3.5 rate is first order, so the displacement it produces is not
    # exactly eta J u. These scales are tried in order and each is certified on
    # the displacement it ACTUALLY applies.
    step_backoff: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)

    def validate(self) -> None:
        if self.optimizer not in ("sgd", "adam", "adamw"):
            raise ValueError(
                "parametric_gd.optimizer must be 'sgd', 'adam' or 'adamw'."
            )
        if self.weight_decay < 0.0:
            raise ValueError("parametric_gd.weight_decay must be non-negative.")
        if self.certification_batches is not None and self.certification_batches < 1:
            raise ValueError(
                "parametric_gd.certification_batches must be positive or null."
            )
        if self.inner_learning_rate <= 0.0:
            raise ValueError("parametric_gd.inner_learning_rate must be positive.")
        if not self.inner_steps or any(v < 1 for v in self.inner_steps):
            raise ValueError(
                "parametric_gd.inner_steps must contain positive integers."
            )
        if not self.functional_learning_rates or any(
            v <= 0.0 for v in self.functional_learning_rates
        ):
            raise ValueError(
                "parametric_gd.functional_learning_rates must be positive."
            )
        if not 0.0 < self.min_cosine <= 1.0:
            raise ValueError("parametric_gd.min_cosine must lie in (0, 1].")
        if self.parameter_penalty < 0.0:
            raise ValueError("parametric_gd.parameter_penalty must be non-negative.")
        if self.inner_step_unit not in ("minibatch", "epoch", "probe"):
            raise ValueError(
                "parametric_gd.inner_step_unit must be 'minibatch', 'epoch' "
                "or 'probe'."
            )
        if self.candidate_objective not in ("mean", "sum"):
            raise ValueError(
                "parametric_gd.candidate_objective must be 'mean' or 'sum'."
            )
        if self.certificate_split not in ("train", "validation"):
            raise ValueError(
                "parametric_gd.certificate_split must be 'train' or 'validation'."
            )
        if self.transactional_split not in ("train", "validation"):
            raise ValueError(
                "parametric_gd.transactional_split must be 'train' or 'validation'."
            )
        if self.cg_iterations < 1:
            raise ValueError("parametric_gd.cg_iterations must be positive.")
        if not self.step_backoff or any(v <= 0.0 for v in self.step_backoff):
            raise ValueError(
                "parametric_gd.step_backoff entries must be positive."
            )
        if self.candidate_readout_ridge < 0.0:
            raise ValueError(
                "parametric_gd.candidate_readout_ridge must be non-negative."
            )
        if self.inner_lr_schedule not in ("constant", "cosine", "linear"):
            raise ValueError(
                "parametric_gd.inner_lr_schedule must be 'constant', "
                "'cosine' or 'linear'."
            )
        if self.acceptance_rule not in (
            "theory_interval",
            "measured_descent",
            "direction_only",
        ):
            raise ValueError(
                "parametric_gd.acceptance_rule must be 'theory_interval', "
                "'measured_descent' or 'direction_only'."
            )
        if self.adaptive_rate_retries < 0:
            raise ValueError(
                "parametric_gd.adaptive_rate_retries must be non-negative."
            )
        if any(not 0.0 < value <= 1.0 for value in self.alpha_grid):
            raise ValueError("parametric_gd.alpha_grid entries must lie in (0, 1].")
        if self.alpha_policy not in (
            "largest_certified",
            "max_descent",
            "full_only",
        ):
            raise ValueError(
                "parametric_gd.alpha_policy must be 'largest_certified', "
                "'max_descent' or 'full_only'."
            )


@dataclass(frozen=True)
class ParametricDescentConfig:
    """Parametric family certified by MEASURED functional descent.

    Same candidate generation as parametric_gd (clone trained toward the
    functional target f - eta_nominal * r; note eta_nominal = 0.5 makes the
    target exactly y, i.e. plain parametric loss descent). Acceptance does
    NOT go through the relative-error route: for the empirical sum-MSE
    functional the function-space PL inequality is the exact identity
    |grad L|^2 = 4 L (mu = 2, L* = 0, the configured theory constants), so
    Proposition 3.8's contraction only needs the per-step descent
    inequality — which is MEASURED on validation instead of lower-bounded
    via epsilon. The measured descent coefficient
    r_t = (L_t - L_{t+1}) / (eta* |grad L_t|^2) plugs into the same Cprog,
    Cstat and Cglob accumulators; the contraction it certifies equals the
    realized loss ratio exactly.
    """

    optimizer: str = "sgd"
    inner_learning_rate: float = 0.05
    inner_steps: tuple[int, ...] = (16, 64)
    functional_learning_rates: tuple[float, ...] = (0.5, 0.2)
    # Optional direction screen. 0.0 only requires a descent direction in
    # function space (eta* > 0); raise it to demand tangent-like alignment.
    min_cosine: float = 0.0
    parameter_penalty: float = 1e-6
    gradient_clip_norm: float | None = 1.0
    # Decoupled weight decay for the adam/adamw inner optimizer. Higher
    # values regularize the generated candidate so its realized
    # displacement generalizes to validation (closing the train/val gap
    # that caps the certified acceptance).
    weight_decay: float = 0.0
    # Cprog floor on the measured progress eta* r_t = D_t / |grad L_t|^2
    # (= D/4L). Every accepted step must remove at least 4*min_progress of
    # the remaining loss; below that the structure is treated as exhausted
    # for this family, which is what lets growth take over instead of the
    # family certifying ever-smaller crumbs forever.
    min_progress: float = 1e-3

    def validate(self) -> None:
        if self.optimizer not in ("sgd", "adam", "adamw"):
            raise ValueError(
                "parametric_descent.optimizer must be 'sgd', 'adam' or 'adamw'."
            )
        if self.inner_learning_rate <= 0.0:
            raise ValueError("parametric_descent.inner_learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("parametric_descent.weight_decay must be non-negative.")
        if not self.inner_steps or any(v < 1 for v in self.inner_steps):
            raise ValueError(
                "parametric_descent.inner_steps must contain positive integers."
            )
        if not self.functional_learning_rates or any(
            v <= 0.0 for v in self.functional_learning_rates
        ):
            raise ValueError(
                "parametric_descent.functional_learning_rates must be positive."
            )
        if not 0.0 <= self.min_cosine <= 1.0:
            raise ValueError("parametric_descent.min_cosine must lie in [0, 1].")
        if self.parameter_penalty < 0.0:
            raise ValueError(
                "parametric_descent.parameter_penalty must be non-negative."
            )
        if self.min_progress <= 0.0:
            raise ValueError("parametric_descent.min_progress must be positive.")


@dataclass(frozen=True)
class SecantFGDConfig:
    enabled: bool = True
    inner_steps: int = 8
    inner_learning_rate: float = 1e-2
    parameter_penalty: float = 1e-6
    gradient_clip_norm: float | None = 1.0
    search_steps: int = 3
    max_learning_rate: float = 0.1
    min_learning_rate_factor: float = 1e-2
    growth_min_relative_error_improvement: float = 1e-3
    growth_min_learning_rate_improvement: float = 0.05


@dataclass(frozen=True)
class FGDLayerRelError:
    layer_index: int
    relative_error: float
    approximation_norm: float
    target_norm: float
    directional_cosine: float


@dataclass(frozen=True)
class FGDOutputRelError:
    relative_error: float
    approximation_norm: float
    target_norm: float
    directional_cosine: float


@dataclass(frozen=True)
class FGDApproxEpochResult:
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    learning_rate: float | None
    next_learning_rate: float | None
    learning_rate_upper_bound: float | None
    learning_rate_interval_valid: bool | None
    learning_rate_clipped_batches: int
    skipped_batches: int
    relative_error_condition_valid: bool | None
    loss_descent_valid: bool | None
    loss_non_descent_batches: int
    gradient_sq_norm: float | None
    theory_descent_coefficient: float | None
    min_positive_learning_rate: float | None
    relative_error: float | None
    selected_layer_index: int | None
    layer_relative_errors: list[FGDLayerRelError]
    output_relative_error: FGDOutputRelError | None
    sensor_valid: bool
    sensor_invalid_batches: int


@dataclass(frozen=True)
class FGDValidationCertificate:
    learning_rate_upper_bound: float | None
    max_valid_learning_rate: float | None
    learning_rate_interval_valid: bool | None
    skipped_batches: int
    relative_error_condition_valid: bool | None
    gradient_sq_norm: float | None
    theory_descent_coefficient: float | None
    relative_error: float | None
    output_relative_error: FGDOutputRelError | None
    sensor_valid: bool
    sensor_invalid_batches: int
    #: Names of the quantities that came out non-finite, when that is why the
    #: sensor rejected. Empty otherwise. Reported so a rejection says what
    #: actually overflowed instead of only that something did.
    non_finite_quantities: tuple[str, ...] = ()


@dataclass(frozen=True)
class _FunctionalStepStats:
    output_error: FGDOutputRelError
    dot_product: float
    approximation_sq_norm: float
    target_sq_norm: float


@dataclass(frozen=True)
class _TangentProjectionStep:
    output_error: FGDOutputRelError
    parameter_updates: tuple[torch.Tensor, ...]
    learning_rate_used: float
    loss_before: float
    loss_after: float
    descent_ok: bool
    dot_product: float
    approximation_sq_norm: float
    target_sq_norm: float


def graph_laplacian(x: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
    """Symmetric NORMALISED graph Laplacian over the probe inputs.

    ``Lambda_sym = I - D^{-1/2} W D^{-1/2}`` with Gaussian affinity
    ``W_ij = exp(-||x_i - x_j||^2 / (2 sigma^2))`` and ``sigma`` from the
    MEDIAN heuristic -- ``bandwidth`` times the median pairwise distance, the
    standard tuning-free bandwidth that adapts to the data scale.

    Normalised, not the raw ``D - W``, for a concrete reason: the unnormalised
    Laplacian's eigenvalues scale with the node degrees (~ N here), so the
    roughness term ``2 gamma Lambda f`` swamped the data gradient ``2(f - y)``
    by ~250x at gamma = 1 and collapsed f to a constant (MEASURED: train 0.000).
    ``Lambda_sym`` has eigenvalues in ``[0, 2]``, so gamma sits on the same
    scale as the data term and is dataset-independent.

    Still symmetric PSD (``f^T Lambda_sym f >= 0``), so ``L_data + gamma
    f^T Lambda_sym f`` stays convex and K-smooth and Lemma 3.5 is untouched.
    """
    distances = torch.cdist(x, x)
    off_diagonal = distances[~torch.eye(x.shape[0], dtype=torch.bool, device=x.device)]
    median = torch.median(off_diagonal)
    sigma = bandwidth * median.clamp_min(torch.finfo(x.dtype).eps)
    affinity = torch.exp(-(distances**2) / (2.0 * sigma**2))
    affinity.fill_diagonal_(0.0)
    degree = affinity.sum(dim=1)
    inverse_sqrt_degree = degree.clamp_min(torch.finfo(x.dtype).eps).rsqrt()
    normalised = (
        affinity * inverse_sqrt_degree.unsqueeze(0) * inverse_sqrt_degree.unsqueeze(1)
    )
    return torch.eye(x.shape[0], dtype=x.dtype, device=x.device) - normalised


def _roughness_target(
    output: torch.Tensor,
    x: torch.Tensor,
    config: FGDApproxConfig,
) -> torch.Tensor | None:
    """``2 gamma Lambda f`` -- the roughness term added to the data gradient.

    Returns ``None`` when the penalty is off, so callers add nothing. Only for
    sum-MSE, where the functional gradient is the simple ``r + 2 gamma Lambda f``;
    left off for cross-entropy, whose gradient does not add so cleanly.
    """
    if config.roughness_gamma <= 0.0 or config.functional_loss != "mse":
        return None
    laplacian = graph_laplacian(x.detach(), config.roughness_bandwidth)
    return 2.0 * config.roughness_gamma * (laplacian @ output.detach())


def mse_functional_gradient(
    y_pred: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return the output gradient for GroMo/TINY sum-MSE convention."""
    return 2.0 * (y_pred.detach() - y.detach())


def batch_functional_mse_loss(
    y_pred: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return the sum-MSE loss used by GroMo/TINY statistics."""
    return torch.sum((y_pred - y) ** 2)


def cross_entropy_functional_gradient(
    y_pred: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return the functional gradient of summed softmax cross-entropy.

    With logits ``f`` and one-hot targets ``Y``,
    ``L(f) = sum_i [logsumexp(f_i) - f_{i,c_i}]`` and

        r = grad_f L = softmax(f) - Y.

    Unlike the sum-MSE gradient ``2(f-Y)``, this vanishes only as the
    correct-class logit diverges, so it never opposes additional confidence
    on an already-correct sample. See report/CROSS_ENTROPY_FGD.md §4.
    """
    return torch.softmax(y_pred.detach(), dim=-1) - y.detach()


def batch_functional_cross_entropy_loss(
    y_pred: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return summed softmax cross-entropy against one-hot targets."""
    log_probabilities = torch.log_softmax(y_pred, dim=-1)
    return -torch.sum(y * log_probabilities)


# The certified functional. Every certificate in this module is a statement
# about THIS loss; see report/CROSS_ENTROPY_FGD.md for which parts of the
# theory transfer between the two (convexity and smoothness do, the
# Polyak-Lojasiewicz constant does not).

_FUNCTIONAL_GRADIENTS = {
    "mse": mse_functional_gradient,
    "cross_entropy": cross_entropy_functional_gradient,
}
_FUNCTIONAL_LOSSES = {
    "mse": batch_functional_mse_loss,
    "cross_entropy": batch_functional_cross_entropy_loss,
}
# Smoothness constant L_s of the functional in output space:
#   MSE:           Hessian = 2 Id                      -> L_s = 2
#   cross-entropy: Hessian = diag(p) - p p^T, PSD,     -> L_s = 1/2
#                  lambda_max <= 1/2 (verified numerically)
# Lemma 3.5's admissible interval is eta_bar = 2(1-2eps)/(L_s(1+2eps)), so
# the cross-entropy interval is four times wider.
FUNCTIONAL_SMOOTHNESS = {"mse": 2.0, "cross_entropy": 0.5}
# Whether the functional admits a global Polyak-Lojasiewicz constant. Only
# sum-MSE does (||r||^2 = 4L exactly, mu = 2, L* = 0). Cross-entropy has
# ||r||^2 = Theta(L^2) as p_c -> 1 and bounded ||r||^2 with unbounded L as
# p_c -> 0, so no mu > 0 exists and the LINEAR contraction of Prop. 3.8
# (the C_glob envelope) is not available for it.
FUNCTIONAL_HAS_PL_CONSTANT = {"mse": True, "cross_entropy": False}


def functional_gradient(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    functional_loss: FunctionalLoss = "mse",
) -> torch.Tensor:
    """Return r = grad_f L for the configured certified functional."""
    try:
        return _FUNCTIONAL_GRADIENTS[functional_loss](y_pred, y)
    except KeyError:
        raise ValueError(
            f"Unsupported fgd_approx.functional_loss '{functional_loss}'. "
            f"Use one of: {', '.join(_FUNCTIONAL_GRADIENTS)}."
        ) from None


def batch_functional_loss(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    functional_loss: FunctionalLoss = "mse",
) -> torch.Tensor:
    """Return L(f) on a batch for the configured certified functional."""
    try:
        return _FUNCTIONAL_LOSSES[functional_loss](y_pred, y)
    except KeyError:
        raise ValueError(
            f"Unsupported fgd_approx.functional_loss '{functional_loss}'. "
            f"Use one of: {', '.join(_FUNCTIONAL_LOSSES)}."
        ) from None


def relative_l2_error(
    approximation: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> float:
    """Compute ||approximation - target|| / ||approximation|| on a batch."""
    numerator = torch.sqrt(torch.mean((approximation - target) ** 2))
    denominator = torch.sqrt(torch.mean(approximation**2)).clamp_min(eps)
    return float((numerator / denominator).detach().item())


def _output_relative_error_from_tensors(
    approximation: torch.Tensor,
    target: torch.Tensor,
    eps: float,
) -> _FunctionalStepStats:
    dot_product = float(torch.sum(approximation * target).detach().item())
    approximation_sq_norm = float(torch.sum(approximation**2).detach().item())
    target_sq_norm = float(torch.sum(target**2).detach().item())
    output_error = _output_relative_error_from_stats(
        dot_product=dot_product,
        approximation_sq_norm=approximation_sq_norm,
        target_sq_norm=target_sq_norm,
        eps=eps,
    )
    return _FunctionalStepStats(
        output_error=output_error,
        dot_product=dot_product,
        approximation_sq_norm=approximation_sq_norm,
        target_sq_norm=target_sq_norm,
    )


def tiny_optimal_update_kwargs(
    config: FGDApproxConfig,
    *,
    compute_delta: bool,
) -> dict[str, object]:
    """Return GroMo options matching the TINY-style optimal layer update."""
    return {
        "compute_delta": compute_delta,
        "use_covariance": config.tiny_use_covariance,
        "alpha_zero": config.tiny_alpha_zero,
        "omega_zero": config.tiny_omega_zero,
        "use_projection": config.tiny_use_projection,
        "ignore_singular_values": config.tiny_ignore_singular_values,
        "use_fisher": config.tiny_use_fisher,
        "maximum_added_neurons": config.tiny_maximum_added_neurons,
        "numerical_threshold": config.tiny_numerical_threshold,
        "statistical_threshold": config.tiny_statistical_threshold,
    }


def _cleanup_tiny_update(model: GrowingMLP) -> None:
    model.reset_computation()
    for layer in getattr(model, "_growing_layers", []):
        if hasattr(layer, "delete_update"):
            layer.delete_update(include_previous=True)
    model.currently_updated_layer_index = None
    model.zero_grad(set_to_none=True)


def _forward_with_tiny_update(
    model: GrowingMLP,
    x: torch.Tensor,
) -> torch.Tensor:
    """Forward the temporary layer delta all the way to the network output."""
    x = model.flatten(x)
    x_ext = None
    for layer in model.layers:
        x, x_ext = layer.extended_forward(
            x,
            x_ext,
            use_optimal_delta=True,
            use_extended_input=False,
            use_extended_output=False,
        )
    return x


def _relative_error_from_stats(
    *,
    layer_index: int,
    dot_product: float,
    approximation_sq_norm: float,
    target_sq_norm: float,
    config: FGDApproxConfig,
) -> FGDLayerRelError:
    numerator_sq = approximation_sq_norm - 2.0 * dot_product + target_sq_norm
    numerator = math.sqrt(max(0.0, numerator_sq))

    denominator_sq = approximation_sq_norm
    denominator = max(math.sqrt(max(0.0, denominator_sq)), config.eps)
    approximation_norm = math.sqrt(max(0.0, approximation_sq_norm))
    target_norm = math.sqrt(max(0.0, target_sq_norm))

    cosine_denominator = max(
        math.sqrt(max(0.0, approximation_sq_norm * target_sq_norm)),
        config.eps,
    )
    directional_cosine = dot_product / cosine_denominator

    return FGDLayerRelError(
        layer_index=layer_index,
        relative_error=numerator / denominator,
        approximation_norm=approximation_norm,
        target_norm=target_norm,
        directional_cosine=directional_cosine,
    )


def _output_relative_error_from_stats(
    *,
    dot_product: float,
    approximation_sq_norm: float,
    target_sq_norm: float,
    eps: float,
) -> FGDOutputRelError:
    numerator_sq = approximation_sq_norm - 2.0 * dot_product + target_sq_norm
    numerator = math.sqrt(max(0.0, numerator_sq))
    denominator = max(math.sqrt(max(0.0, approximation_sq_norm)), eps)
    approximation_norm = math.sqrt(max(0.0, approximation_sq_norm))
    target_norm = math.sqrt(max(0.0, target_sq_norm))
    cosine_denominator = max(
        math.sqrt(max(0.0, approximation_sq_norm * target_sq_norm)),
        eps,
    )
    directional_cosine = dot_product / cosine_denominator
    return FGDOutputRelError(
        relative_error=numerator / denominator,
        approximation_norm=approximation_norm,
        target_norm=target_norm,
        directional_cosine=directional_cosine,
    )


def _projection_sensor_reason(
    *,
    dot_product: float,
    approximation_sq_norm: float,
    target_sq_norm: float,
    eps: float,
    relative_tolerance: float = 1e-4,
) -> str | None:
    """Return WHY the damped projection fails, or None when it is valid.

    Splits what the plain boolean sensor conflates into a NAMED cause,
    because the causes are not interchangeable for a caller deciding
    whether to force a growth (see FGDApproxConfig.certify_force_growth_
    on_finite_step_failure): "non_finite" means the model itself produced
    NaN/Inf -- an overfitting symptom that more capacity makes worse, so
    growth must NOT be forced on it. The other three are FINITE numbers
    that violate a geometric invariant of the exact projector (it should
    be non-expansive and non-adversarial) -- sane measurements the
    structure failed to satisfy, which growth can still fix.
    """
    values = (dot_product, approximation_sq_norm, target_sq_norm)
    if not all(math.isfinite(value) for value in values):
        return "non_finite"
    if approximation_sq_norm < -eps or target_sq_norm < -eps:
        return "negative_norm"

    approximation_sq_norm = max(0.0, approximation_sq_norm)
    target_sq_norm = max(0.0, target_sq_norm)
    norm_product = math.sqrt(approximation_sq_norm * target_sq_norm)
    dot_tolerance = relative_tolerance * max(norm_product, eps)
    if dot_product < -dot_tolerance:
        return "negative_dot_product"

    norm_tolerance = relative_tolerance * max(target_sq_norm, eps)
    if approximation_sq_norm > target_sq_norm + norm_tolerance:
        return "norm_overshoot"

    return None


def _projection_sensor_valid(
    *,
    dot_product: float,
    approximation_sq_norm: float,
    target_sq_norm: float,
    eps: float,
    relative_tolerance: float = 1e-4,
) -> bool:
    """Return whether the damped projection satisfies exact-operator invariants."""
    return (
        _projection_sensor_reason(
            dot_product=dot_product,
            approximation_sq_norm=approximation_sq_norm,
            target_sq_norm=target_sq_norm,
            eps=eps,
            relative_tolerance=relative_tolerance,
        )
        is None
    )


def _projection_step_sensor_reason(
    step: _TangentProjectionStep,
    config: FGDApproxConfig,
) -> str | None:
    return _projection_sensor_reason(
        dot_product=step.dot_product,
        approximation_sq_norm=step.approximation_sq_norm,
        target_sq_norm=step.target_sq_norm,
        eps=config.eps,
    )


def _projection_step_sensor_valid(
    step: _TangentProjectionStep,
    config: FGDApproxConfig,
) -> bool:
    return _projection_step_sensor_reason(step, config) is None


def certified_smoothness_constant(config: FGDApproxConfig) -> float:
    """Return the smoothness L_s of the CERTIFIED functional.

    Lemma 3.5's admissible interval is eta_bar = 2(1-2eps)/(L_s(1+2eps)),
    so L_s must describe the loss actually being certified: 2 for sum-MSE
    (Hessian 2*Id) and 1/2 for softmax cross-entropy (Hessian
    diag(p)-p p^T, lambda_max <= 1/2), which makes the cross-entropy
    interval four times wider. When theory_smoothness_constant is left at
    its MSE default the constant of the configured functional is used;
    an explicitly non-default value always wins, so the knob stays usable.
    """
    configured = float(config.theory_smoothness_constant)
    default_mse = FUNCTIONAL_SMOOTHNESS["mse"]
    if configured == default_mse:
        return FUNCTIONAL_SMOOTHNESS.get(config.functional_loss, configured)
    return configured


def theoretical_learning_rate_upper_bound(
    relative_error: float,
    config: FGDApproxConfig,
) -> float | None:
    """Return the FGD learning-rate upper bound from the current RelErr."""
    if relative_error < 0.0 or relative_error >= 0.5:
        return None

    smoothness = float(certified_smoothness_constant(config))
    if smoothness <= 0.0:
        raise ValueError("fgd_approx.theory_smoothness_constant must be positive.")

    return (
        2.0 * (1.0 - 2.0 * relative_error) / (smoothness * (2.0 * relative_error + 1.0))
    )


def theoretical_descent_coefficient(
    relative_error: float,
    learning_rate: float,
    config: FGDApproxConfig,
) -> float | None:
    """Return the proof coefficient r for the approximate FGD step."""
    if relative_error < 0.0 or relative_error >= 1.0:
        return None
    if learning_rate <= 0.0:
        return None

    smoothness = float(certified_smoothness_constant(config))
    alpha = float(config.theory_alpha)
    beta = float(config.theory_beta)
    if smoothness <= 0.0:
        raise ValueError("fgd_approx.theory_smoothness_constant must be positive.")
    if alpha <= 0.0:
        raise ValueError("fgd_approx.theory_alpha must be positive.")
    if beta <= 0.0:
        raise ValueError("fgd_approx.theory_beta must be positive.")

    error_ratio = relative_error / max(1.0 - relative_error, config.eps)
    return (
        alpha
        - 0.5 * smoothness * learning_rate
        - beta * error_ratio
        - 1.5 * smoothness * learning_rate * error_ratio
    )


def _clear_inaccessible_tensor_caches(model: torch.nn.Module) -> None:
    """Drop functorch-wrapped tensors cached by GroMo modules during jacrev."""
    for module in model.modules():
        for name, value in list(vars(module).items()):
            if torch.is_tensor(value):
                try:
                    value.untyped_storage()
                except (NotImplementedError, RuntimeError):
                    setattr(module, name, None)


def _trainable_named_parameters(
    model: torch.nn.Module,
) -> OrderedDict[str, torch.nn.Parameter]:
    return OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def _flatten_jacobian(
    jacobian: tuple[torch.Tensor, ...],
    output_numel: int,
) -> torch.Tensor:
    columns = [item.reshape(output_numel, -1) for item in jacobian]
    return torch.cat(columns, dim=1).detach()


def _unflatten_parameter_update(
    flat_update: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor, ...]:
    updates: list[torch.Tensor] = []
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        updates.append(flat_update[offset : offset + size].reshape_as(parameter))
        offset += size
    return tuple(updates)


def _flatten_parameter_tensors(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _zero_tangent_projection(
    jacobian_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_numel, parameter_numel = jacobian_matrix.shape
    original_dtype = jacobian_matrix.dtype
    return (
        torch.zeros(
            parameter_numel,
            device=jacobian_matrix.device,
            dtype=original_dtype,
        ),
        torch.zeros(
            output_numel,
            device=jacobian_matrix.device,
            dtype=original_dtype,
        ),
    )


def _solve_tangent_projection_svd(
    jacobian_matrix: torch.Tensor,
    target: torch.Tensor,
    damping: float,
    work_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the damped tangent projection with an SVD of J.

    ``work_dtype`` defaults to float64. float32 runs the SVD ~9x faster and is
    safe where the caller self-corrects -- the realise path re-solves against
    the measured residual each inner iteration, so a float32 sub-step's
    round-off is absorbed by the next one. It is the dominant cost of a run
    (MEASURED 26.7 s of 76.8 s in the projection SVD).
    """
    output_numel, parameter_numel = jacobian_matrix.shape
    damping = max(float(damping), 0.0)
    original_dtype = jacobian_matrix.dtype
    jacobian_work = jacobian_matrix.to(dtype=work_dtype)
    target_work = target.reshape(-1).to(dtype=work_dtype)
    if output_numel == 0 or parameter_numel == 0:
        return _zero_tangent_projection(jacobian_matrix)

    u, singular_values, vh = torch.linalg.svd(jacobian_work, full_matrices=False)
    if singular_values.numel() == 0:
        return _zero_tangent_projection(jacobian_matrix)

    coefficients = u.t() @ target_work
    eigenvalues = singular_values.square()

    if damping > 0.0:
        denominator = eigenvalues + damping
        output_factors = eigenvalues / denominator
        update_factors = singular_values / denominator
    else:
        max_singular = torch.max(singular_values)
        threshold = (
            torch.finfo(work_dtype).eps
            * max(output_numel, parameter_numel)
            * max(float(max_singular.detach().item()), 1.0)
        )
        nonzero = singular_values > threshold
        output_factors = torch.where(
            nonzero,
            torch.ones_like(singular_values),
            torch.zeros_like(singular_values),
        )
        update_factors = torch.where(
            nonzero,
            singular_values.reciprocal(),
            torch.zeros_like(singular_values),
        )

    approximation_work = u @ (output_factors * coefficients)
    flat_update_work = vh.t() @ (update_factors * coefficients)
    return (
        flat_update_work.to(dtype=original_dtype),
        approximation_work.to(dtype=original_dtype),
    )


def _solve_tangent_projection_kernel_eigh(
    jacobian_matrix: torch.Tensor,
    target: torch.Tensor,
    damping: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the damped tangent projection with a float64 eigendecomposition of K."""
    output_numel, parameter_numel = jacobian_matrix.shape
    damping = max(float(damping), 0.0)
    original_dtype = jacobian_matrix.dtype
    work_dtype = torch.float64
    jacobian_work = jacobian_matrix.to(dtype=work_dtype)
    target_work = target.reshape(-1).to(dtype=work_dtype)
    if output_numel == 0 or parameter_numel == 0:
        return _zero_tangent_projection(jacobian_matrix)

    kernel = jacobian_work @ jacobian_work.t()
    kernel = 0.5 * (kernel + kernel.t())
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    eigenvalues = eigenvalues.clamp_min(0.0)
    if eigenvalues.numel() == 0:
        return _zero_tangent_projection(jacobian_matrix)

    coefficients = eigenvectors.t() @ target_work
    if damping > 0.0:
        denominator = eigenvalues + damping
        output_factors = eigenvalues / denominator
        dual_factors = denominator.reciprocal()
    else:
        max_eigenvalue = torch.max(eigenvalues)
        threshold = (
            torch.finfo(work_dtype).eps
            * output_numel
            * max(float(max_eigenvalue.detach().item()), 1.0)
        )
        nonzero = eigenvalues > threshold
        output_factors = torch.where(
            nonzero,
            torch.ones_like(eigenvalues),
            torch.zeros_like(eigenvalues),
        )
        dual_factors = torch.where(
            nonzero,
            eigenvalues.reciprocal(),
            torch.zeros_like(eigenvalues),
        )

    dual_solution_work = eigenvectors @ (dual_factors * coefficients)
    approximation_work = eigenvectors @ (output_factors * coefficients)
    flat_update_work = jacobian_work.t() @ dual_solution_work
    return (
        flat_update_work.to(dtype=original_dtype),
        approximation_work.to(dtype=original_dtype),
    )


def _solve_tangent_projection(
    jacobian_matrix: torch.Tensor,
    target: torch.Tensor,
    damping: float,
    solver: ProjectionSolver = "exact",
    work_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return parameter update and projected output gradient."""
    increment("tangent_projection_solve_calls")
    if solver in {"exact", "exact_svd"}:
        return _solve_tangent_projection_svd(
            jacobian_matrix=jacobian_matrix,
            target=target,
            damping=damping,
            work_dtype=work_dtype,
        )
    if solver == "exact_kernel_eigh":
        return _solve_tangent_projection_kernel_eigh(
            jacobian_matrix=jacobian_matrix,
            target=target,
            damping=damping,
        )
    raise ValueError(
        f"Unsupported exact projection solver '{solver}'. "
        "Use one of: exact, exact_svd, exact_kernel_eigh."
    )


def _conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    max_iterations: int,
    tolerance: float,
    eps: float,
) -> torch.Tensor:
    """Solve A x = rhs using conjugate gradient with an implicit SPD matvec."""
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rhs_norm = torch.linalg.norm(rhs).clamp_min(eps)
    residual_sq = torch.dot(r, r)
    if torch.sqrt(residual_sq) <= tolerance * rhs_norm:
        return x

    for _ in range(max(1, max_iterations)):
        ap = matvec(p)
        denominator = torch.dot(p, ap)
        if torch.abs(denominator) <= eps:
            break

        alpha = residual_sq / denominator
        x = x + alpha * p
        r = r - alpha * ap
        next_residual_sq = torch.dot(r, r)
        if torch.sqrt(next_residual_sq) <= tolerance * rhs_norm:
            break

        beta = next_residual_sq / residual_sq.clamp_min(eps)
        p = r + beta * p
        residual_sq = next_residual_sq

    return x


@dataclass(frozen=True)
class ExactTangentSystem:
    """The materialised system ``J u ~ r`` at the current model.

    Exposed so a caller can solve it more than once without paying for the
    Jacobian again. The projection is linear in the regularisation, so a
    single factorisation answers a whole ladder of damping values; recomputing
    ``jacrev`` per value would multiply the dominant cost by the ladder size
    for no new information.
    """

    jacobian: torch.Tensor
    target: torch.Tensor
    parameters: tuple[torch.Tensor, ...]
    loss: float
    # The row-aligned full-probe target.  ``target`` is a compact sufficient-
    # statistics surrogate when ``certify_stream_gram`` is enabled; candidate
    # cross-statistics still need the original rows, but retaining this vector
    # is O(NK), not the forbidden O(NKP) second Jacobian.
    #: ``(W, V)`` with ``J = W V^T``, ``W = J V`` of shape ``(rows, k)`` and
    #: ``V`` of shape ``(P, k)`` with orthonormal columns. Present ONLY on the
    #: matrix-free family; ``None`` on every exact system, which is what keeps
    #: the default path constructing exactly what it constructs today.
    #:
    #: Carrying it is what lets the consumers skip ``J`` entirely: with
    #: ``W = A S B^T``, ``J = A S (V B)^T`` and ``V B`` is orthonormal, so
    #: ``svd(J) = (A, S, V B)`` EXACTLY -- from an SVD of a ``(rows, k)``
    #: matrix. Memory ``O(NK k + P k)``, linear in ``P``.
    factors: tuple[torch.Tensor, torch.Tensor] | None = field(
        default=None, repr=False, compare=False
    )
    full_target: torch.Tensor | None = field(default=None, repr=False, compare=False)
    owner_model: torch.nn.Module | None = field(default=None, repr=False, compare=False)
    parameter_names: tuple[str, ...] = field(default=(), repr=False)
    parameter_versions: tuple[int, ...] = field(default=(), repr=False)
    buffer_names: tuple[str, ...] = field(default=(), repr=False)
    buffers: tuple[torch.Tensor, ...] = field(default=(), repr=False, compare=False)
    buffer_versions: tuple[int, ...] = field(default=(), repr=False)
    probe_x: torch.Tensor | None = field(default=None, repr=False, compare=False)
    probe_y: torch.Tensor | None = field(default=None, repr=False, compare=False)
    probe_versions: tuple[int, int] = field(default=(-1, -1), repr=False)
    config_signature: str = field(default="", repr=False)
    evaluation_state: tuple[tuple[str, bool], ...] = field(default=(), repr=False)


def validate_exact_tangent_system(
    system: ExactTangentSystem,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> None:
    """Reject reuse unless every object and value defining ``J`` and ``r`` matches."""
    if system.owner_model is not model:
        raise ValueError("ExactTangentSystem belongs to a different model instance.")

    named_parameters = _trainable_named_parameters(model)
    names = tuple(named_parameters)
    parameters = tuple(named_parameters.values())
    if names != system.parameter_names or len(parameters) != len(system.parameters):
        raise ValueError("ExactTangentSystem trainable parameter ordering changed.")
    if any(
        current is not cached for current, cached in zip(parameters, system.parameters)
    ):
        raise ValueError("ExactTangentSystem trainable parameter objects changed.")
    versions = tuple(parameter._version for parameter in parameters)
    if versions != system.parameter_versions:
        raise ValueError("ExactTangentSystem parameter values changed.")

    named_buffers = tuple(model.named_buffers())
    buffer_names = tuple(name for name, _ in named_buffers)
    buffers = tuple(buffer for _, buffer in named_buffers)
    if buffer_names != system.buffer_names or len(buffers) != len(system.buffers):
        raise ValueError("ExactTangentSystem model buffers changed.")
    if any(current is not cached for current, cached in zip(buffers, system.buffers)):
        raise ValueError("ExactTangentSystem buffer objects changed.")
    if tuple(buffer._version for buffer in buffers) != system.buffer_versions:
        raise ValueError("ExactTangentSystem buffer values changed.")

    if system.probe_x is not x or system.probe_y is not y:
        raise ValueError("ExactTangentSystem probe or targets changed.")
    if (x._version, y._version) != system.probe_versions:
        raise ValueError("ExactTangentSystem probe or target values changed.")
    if repr(config) != system.config_signature:
        raise ValueError("ExactTangentSystem functional configuration changed.")
    state = tuple((name, module.training) for name, module in model.named_modules())
    if state != system.evaluation_state:
        raise ValueError("ExactTangentSystem model evaluation state changed.")


TangentBackend = Literal["legacy", "optimized", "auto"]
_TANGENT_BACKENDS: tuple[TangentBackend, ...] = ("legacy", "optimized", "auto")


def tangent_backend() -> TangentBackend:
    """Read FGD_TANGENT_BACKEND per call so tests can monkeypatch it.

    legacy    - today's path, bit-for-bit in every decision this module
                makes. DEFAULT while the optimized backend is validated
                (see NOTE below) -- flipping the default is a separate,
                later commit, not this one.
    auto      - the optimized construction (jacobian, QR, surrogate) with
                automatic, COUNTED, REASONED fallback to legacy whenever a
                structural predicate or a numerical check fails. Never
                raises for a reason the fallback machinery already covers.
    optimized - same as auto but RAISES instead of silently falling back,
                so a test (or a benchmark) can PROVE the fast path
                actually ran rather than having quietly degraded.

    NOTE: an unset FGD_TANGENT_BACKEND resolves to "auto". The one gate
    that "legacy" was held back for -- byte-identical selected damping --
    turned out not to be a property of this pipeline at all: legacy
    against ITSELF, changing only certify_stream_chunk from 512 to 1024,
    moves the chosen rung by 78% while eps, eta and the guaranteed
    decrease all move by <= 3.4e-6. The rung is an argmax over a
    geometric fan that is flat at its maximum, so ANY bit-level
    perturbation reshuffles it. What IS stable -- the strict certificate
    boolean, eps, the growth decision and location, the family choice --
    is identical between the backends on CIFAR, and with the downstream
    factorization in float64 the two agree BITWISE. Set
    FGD_TANGENT_BACKEND=legacy to roll back at any time.
    """
    value = os.environ.get("FGD_TANGENT_BACKEND", "auto")
    if value not in _TANGENT_BACKENDS:
        raise ValueError(
            f"Unsupported FGD_TANGENT_BACKEND '{value}'. "
            f"Use one of: {', '.join(_TANGENT_BACKENDS)}."
        )
    return value  # type: ignore[return-value]


def _tangent_verify_enabled() -> bool:
    """FGD_TANGENT_VERIFY=1: off-by-default oracle cross-checks (C.2/C.3).

    Recomputes the legacy CUDA/CPU-SVD surrogate alongside the optimized
    result purely to compare them -- real per-call cost, never enabled in
    production, only for CI/tests proving the fast path is exact.
    """
    return os.environ.get("FGD_TANGENT_VERIFY", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class AnalyticMLPStructure:
    """Everything the batched analytic Jacobian (below) needs, resolved ONCE
    per outer step rather than re-derived from the model on every sample
    chunk. ``linears``/``post_functions``/``use_bias`` are ordered exactly
    like ``model.layers``; ``parameter_names`` is the expected
    ``_trainable_named_parameters`` order the column layout matches.
    """

    linears: tuple[torch.nn.Linear, ...]
    post_functions: tuple[torch.nn.Module, ...]
    use_bias: tuple[bool, ...]
    parameter_names: tuple[str, ...]
    parameter_numel: int
    output_features: int


# P12.1: activations the analytic D_l recursion is derived for -- all
# pointwise, stateless maps. Deliberately excludes BatchNorm/LayerNorm/
# Softmax (regularized_mlp.make_hidden_post_function), which couple across
# the batch or feature dimension and would silently break the "D_l depends
# only on z_l[n, :]" assumption C.1 relies on.
_ANALYTIC_ACTIVATION_TYPES = (
    torch.nn.Identity,
    torch.nn.SELU,
    torch.nn.ReLU,
    torch.nn.LeakyReLU,
    torch.nn.ELU,
    torch.nn.GELU,
    torch.nn.SiLU,
    torch.nn.Tanh,
    torch.nn.Sigmoid,
    torch.nn.Softplus,
)


def _activation_type_allowed(phi: torch.nn.Module) -> bool:
    """P12.1: type allow-list, recursing into nn.Sequential children."""
    if isinstance(phi, torch.nn.Sequential):
        return len(phi) > 0 and all(_activation_type_allowed(child) for child in phi)
    return isinstance(phi, _ANALYTIC_ACTIVATION_TYPES)


def _activation_is_stateless(phi: torch.nn.Module) -> bool:
    """P12.2: nothing but the pointwise map -- no learned or running state."""
    return next(phi.parameters(), None) is None and next(phi.buffers(), None) is None


def _activation_preserves_shape(
    phi: torch.nn.Module, width: int, device: torch.device, dtype: torch.dtype
) -> bool:
    """P12.4: phi(z) has the same shape as z."""
    with torch.no_grad():
        probe = torch.randn(4, width, device=device, dtype=dtype)
        return phi(probe).shape == probe.shape


def _activation_has_no_coupling(
    phi: torch.nn.Module, width: int, device: torch.device, dtype: torch.dtype
) -> bool:
    """P12.3: numeric no-coupling probe (TANGENT_PLAN.md A.4).

    ``phi`` must act ENTRYWISE: build ``mixed`` equal to a fresh random
    ``u`` everywhere except at (row 0, column j), where it holds ``t``'s
    value; the output at (0, j) must then depend on nothing else, so it
    must equal ``phi(t)``'s output at (0, j) BITWISE. This is the actual
    behavioural guarantee the analytic ``D_l`` recursion relies on -- the
    type allow-list above already excludes BatchNorm/LayerNorm/Softmax, but
    this probe is what would catch a future stateless-looking activation
    that still couples across the batch or feature dimension.
    """
    with torch.no_grad():
        t = torch.randn(4, width, device=device, dtype=dtype)
        full = phi(t)
        u = torch.randn(4, width, device=device, dtype=dtype)
        for j in range(min(width, 4)):
            mixed = u.clone()
            mixed[0, j] = t[0, j]
            probed = phi(mixed)
            if not torch.equal(full[0, j], probed[0, j]):
                return False
    return True


def _activation_rejection_reason(
    phi: torch.nn.Module, width: int, device: torch.device, dtype: torch.dtype
) -> str | None:
    """P12: the specific reason ``phi`` is unsupported, or None if it qualifies."""
    if not _activation_type_allowed(phi):
        return "activation_type_not_allowlisted"
    if not _activation_is_stateless(phi):
        return "activation_has_parameters_or_buffers"
    if not _activation_preserves_shape(phi, width, device, dtype):
        return "activation_changes_shape"
    if not _activation_has_no_coupling(phi, width, device, dtype):
        return "activation_has_cross_entry_coupling"
    return None


def _supported_analytic_structure(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    capture_suspended: bool = False,
) -> AnalyticMLPStructure | None:
    """Return the structure when P1-P14 all hold, else None (reason recorded).

    Every predicate is a NECESSARY condition for the batched analytic
    Jacobian (C.1) to equal ``_flatten_jacobian(jacrev(...))`` exactly --
    see TANGENT_PLAN.md section A.4. A failing predicate always records a
    DISTINCT, counted reason via ``fallback`` before returning ``None``:
    the fast path never disappears silently.

    ``capture_suspended`` is set by a caller that has entered gromo's
    ``paused_computation()`` around this whole construction. P8 then holds
    BY CONSTRUCTION -- capture is off for the duration, so there is no
    caching side effect for the analytic path to skip -- and checking the
    flags again would just re-observe the caller's own suspension. It is
    never inferred here: only a caller that actually holds the context
    may pass it.
    """
    # P1: a gromo sequential MLP, at least one layer.
    if not (
        isinstance(model, GrowingMLP)
        and isinstance(model.layers, torch.nn.ModuleList)
        and len(model.layers) >= 1
    ):
        fallback(
            "tangent_unsupported_structure_fallbacks",
            "not_a_growing_mlp_sequential_container",
        )
        return None
    layers = list(model.layers)

    # P2: every layer is a LinearGrowingModule (rules out conv/attention
    # containers this analytic form was never derived for).
    if not all(isinstance(layer, LinearGrowingModule) for layer in layers):
        fallback(
            "tangent_unsupported_structure_fallbacks",
            "layer_not_linear_growing_module",
        )
        return None

    # P3: the inner layer is a PLAIN nn.Linear -- exact type, not a
    # subclass, since a subclass could override forward with extra math
    # the analytic formulas below know nothing about.
    if not all(type(layer.layer) is torch.nn.Linear for layer in layers):
        fallback(
            "tangent_unsupported_structure_fallbacks",
            "inner_layer_not_plain_linear",
        )
        return None

    # P4: no degenerate fan-in/out -- LinearGrowingModule's safe-forward
    # short-circuits to zeros when in_features == 0, which the analytic
    # forward below does not special-case.
    if not all(
        layer.layer.in_features > 0 and layer.layer.out_features > 0 for layer in layers
    ):
        fallback("tangent_unsupported_structure_fallbacks", "degenerate_fan_in_or_out")
        return None

    # P5: the bias flag and the actual nn.Linear.bias agree, so use_bias
    # can be trusted to decide the column layout below.
    if not all(layer.use_bias == (layer.layer.bias is not None) for layer in layers):
        fallback("tangent_unsupported_structure_fallbacks", "bias_flag_inconsistent")
        return None

    # P6: flatten is a no-op (already 2-D) or exactly the batch-preserving
    # nn.Flatten(start_dim=1, end_dim=-1) gromo builds -- either is the
    # plain reshape(x.shape[0], -1) the analytic forward applies.
    flatten = model.flatten
    flatten_ok = isinstance(flatten, torch.nn.Identity) or (
        isinstance(flatten, torch.nn.Flatten)
        and flatten.start_dim == 1
        and flatten.end_dim == -1
    )
    if not flatten_ok:
        fallback(
            "tangent_unsupported_structure_fallbacks", "flatten_not_batch_preserving"
        )
        return None

    # P7: no pending growth extension -- a mid-growth layer's forward adds
    # extended_input_layer / extended_output_layer / optimal_delta_layer
    # terms the analytic form does not know how to differentiate.
    if not all(
        layer.extended_input_layer is None
        and layer.extended_output_layer is None
        and layer.optimal_delta_layer is None
        for layer in layers
    ):
        fallback(
            "tangent_unsupported_structure_fallbacks", "pending_growth_extension_state"
        )
        return None

    # P8: forward has no caching side effects -- store_input/
    # store_pre_activity (and their internal counterparts) mutate module
    # state on every call, which the analytic path, calling nn.Linear
    # directly, would silently skip, leaving the two paths' model state to
    # diverge. Skipped only when the caller holds paused_computation(),
    # which turns capture off for the whole construction and so satisfies
    # the predicate rather than ignoring it.
    if not capture_suspended and not all(
        not layer.store_input
        and not layer.store_pre_activity
        and not layer._internal_store_input
        and not layer._internal_store_pre_activity
        for layer in layers
    ):
        fallback(
            "tangent_unsupported_structure_fallbacks",
            "forward_has_caching_side_effects",
        )
        return None

    # P9: the trainable parameter set is EXACTLY (weight, bias) per layer,
    # ascending, matching the column layout the analytic block writes into.
    trainable = _trainable_named_parameters(model)
    expected_names: list[str] = []
    for index, layer in enumerate(layers):
        expected_names.append(f"layers.{index}.layer.weight")
        if layer.use_bias:
            expected_names.append(f"layers.{index}.layer.bias")
    if tuple(trainable) != tuple(expected_names) or len(
        list(model.parameters())
    ) != len(trainable):
        fallback(
            "tangent_unsupported_structure_fallbacks",
            "unexpected_trainable_parameter_set",
        )
        return None

    # P10: no buffers to thread through functional_call.
    if next(model.named_buffers(), None) is not None:
        fallback("tangent_unsupported_structure_fallbacks", "model_has_buffers")
        return None

    # P11: eval mode (dropout identity, batch-norm frozen -- moot here
    # since P1/P2 already exclude batch-norm, but cheap and explicit).
    if model.training is not False:
        fallback("tangent_unsupported_structure_fallbacks", "model_not_in_eval_mode")
        return None

    # P12: every activation is pointwise and stateless. Memoised per
    # (post_layer_function identity, width) within this call, since gromo's
    # activation argument defaults to ONE shared nn.Module instance across
    # every hidden layer (TANGENT_PLAN.md A.1) -- probing it once avoids
    # repeating the numeric probe per layer for no new information.
    first_weight = layers[0].layer.weight
    probe_device = first_weight.device
    probe_dtype = first_weight.dtype
    verified_activations: dict[tuple[int, int], str | None] = {}
    for layer in layers:
        phi = layer.post_layer_function
        width = layer.layer.out_features
        key = (id(phi), width)
        if key not in verified_activations:
            verified_activations[key] = _activation_rejection_reason(
                phi, width, probe_device, probe_dtype
            )
        reason = verified_activations[key]
        if reason is not None:
            fallback("tangent_unsupported_structure_fallbacks", reason)
            return None

    # P13: the flattened probe is 2-D and matches the first layer's
    # dtype/device -- the analytic forward assumes both.
    probe = x[: min(4, x.shape[0])]
    with torch.no_grad():
        flattened_probe = flatten(probe)
    if flattened_probe.ndim != 2:
        fallback("tangent_unsupported_structure_fallbacks", "flatten_not_2d")
        return None
    if flattened_probe.dtype != probe_dtype or flattened_probe.device != probe_device:
        fallback(
            "tangent_unsupported_structure_fallbacks",
            "probe_dtype_or_device_mismatch",
        )
        return None

    # P14: the model maps (N, ...) -> (N, K) -- a 2-D, row-aligned output,
    # matching the "row n*K+k" layout the analytic block writes.
    with torch.no_grad():
        probe_output = model(probe)
    if probe_output.ndim != 2 or probe_output.shape[0] != probe.shape[0]:
        fallback("tangent_unsupported_structure_fallbacks", "output_not_2d_row_aligned")
        return None

    return AnalyticMLPStructure(
        linears=tuple(layer.layer for layer in layers),
        post_functions=tuple(layer.post_layer_function for layer in layers),
        use_bias=tuple(layer.use_bias for layer in layers),
        parameter_names=tuple(expected_names),
        parameter_numel=sum(parameter.numel() for parameter in trainable.values()),
        output_features=layers[-1].layer.out_features,
    )


def _analytic_jacobian_block(
    structure: AnalyticMLPStructure,
    x_block: torch.Tensor,
    out_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Rows n*K+k, columns in _trainable_named_parameters order. Section C.1.

    Replaces ``jacrev`` + ``_flatten_jacobian`` + ``.to(float64)`` with the
    closed-form MLP Jacobian: downstream sensitivities ``D_l`` propagate
    backward from the identity at the last layer, activation derivatives
    come from ``torch.autograd.grad`` on the REAL module (never hard-coded,
    so e.g. SELU's boundary convention at ``z == 0`` matches jacrev by
    construction), and the outer product ``D_l (x) A_l`` is written
    directly into the preallocated output slice.

    Precision rule: forward / D / gprime run in the model's OWN dtype,
    mirroring jacrev bit-for-bit; only the final outer product is formed in
    ``out_dtype``. The streamed branch passes float64 (strictly more
    accurate than legacy, which multiplies in float32 THEN casts, and
    fuses that cast for free); the row-chunked and full-Jacobian branches
    pass the model's own dtype instead, matching
    ``_flatten_jacobian(jacrev(...))``'s dtype exactly, since those
    branches do not cast to float64 either.
    """
    linears = structure.linears
    depth = len(linears)

    with torch.no_grad():
        a = x_block.reshape(x_block.shape[0], -1)
        activations: list[torch.Tensor] = [a]
        pre_activations: list[torch.Tensor] = []
        for layer, phi in zip(linears, structure.post_functions):
            z = layer(a)
            pre_activations.append(z)
            a = phi(z)
            activations.append(a)

    # Activation derivatives from autograd on the REAL module -- never
    # hard-coded. Each is a tiny, independent graph (one tensor in, one
    # tensor out), not the whole-network backward jacrev would build.
    gprimes: list[torch.Tensor] = []
    for z, phi in zip(pre_activations, structure.post_functions):
        z_ = z.detach().requires_grad_(True)
        a_ = phi(z_)
        (gprime,) = torch.autograd.grad(a_, z_, torch.ones_like(a_))
        gprimes.append(gprime.detach())

    n = x_block.shape[0]
    k = structure.output_features
    with torch.no_grad():
        eye_k = torch.eye(k, device=x_block.device, dtype=x_block.dtype)
        d_list: list[torch.Tensor] = [torch.empty(0)] * depth
        d_list[depth - 1] = eye_k.expand(n, k, k) * gprimes[-1].unsqueeze(1)
        for layer_index in range(depth - 2, -1, -1):
            d_list[layer_index] = (
                d_list[layer_index + 1] @ linears[layer_index + 1].weight
            ) * gprimes[layer_index].unsqueeze(1)

        out = torch.empty(
            n * k, structure.parameter_numel, device=x_block.device, dtype=out_dtype
        )
        off = 0
        for layer_index in range(depth):
            d64 = d_list[layer_index].to(out_dtype)
            a64 = activations[layer_index].to(out_dtype)
            out_features = linears[layer_index].out_features
            in_features = linears[layer_index].in_features
            width = out_features * in_features
            out[:, off : off + width] = (
                d64.unsqueeze(3) * a64.unsqueeze(1).unsqueeze(2)
            ).reshape(n * k, width)
            off += width
            if structure.use_bias[layer_index]:
                out[:, off : off + out_features] = d64.reshape(n * k, out_features)
                off += out_features
        assert off == structure.parameter_numel
    return out


def _tangent_spectrum(r_factor: torch.Tensor) -> torch.Tensor:
    """Singular values of the streamed factor R, via CPU LAPACK.

    ``R`` is P x P (upper triangular) once the streamed factor has
    accumulated at least P functional rows; when the probe has FEWER
    functional rows than parameters (N*K < P) it stays m x P with m < P
    for the whole call -- ``svdvals`` handles that shape natively (it
    just returns min(m, P) singular values), and ``R^T R = J^T J`` still
    holds exactly regardless of shape, so this is not an approximation
    either way, just a smaller and better-conditioned input to the same
    exact algorithm. CPU, because cuSOLVER's default float64 SVD driver is
    catastrophic at this size (MEASURED 25.7 s CUDA vs 1.43 s CPU
    ``svdvals``, TANGENT_PLAN.md C.4); the guard below retries on CUDA
    with the exact QR-based 'gesvd' driver ONLY if CPU LAPACK itself
    errors -- never to paper over a genuine inconsistency caught by the
    checks that follow.

    Three cheap, exact sanity checks accompany the spectrum:

    1. Frobenius identity ``sum(sigma_i^2) == ||R||_F^2`` -- an algebraic
       identity, true regardless of conditioning OR shape, MEASURED to
       agree to the last bit. A relative violation above 1e-12 means the
       returned spectrum cannot be trusted at all, so this RAISES (there
       is no narrower fallback for "the spectrum itself is wrong").
    2/3. Power / inverse-power iteration corroborate sigma_max / sigma_min
       independently of LAPACK. These are diagnostics, NOT gates: on a
       genuinely ill-conditioned R (MEASURED cond(J) = 5.9e14 on realistic
       input, TANGENT_PLAN.md C.0) the smallest singular value can sit too
       close to the iteration's noise floor for 50 steps to bracket it to
       5%, which is an expected property of a hard problem, not evidence
       LAPACK is wrong -- so a miss here is recorded via ``fallback`` and
       NOT raised, unlike the Frobenius identity above. The inverse-power
       (sigma_min) check additionally needs a SQUARE R -- solve_triangular
       requires it -- so on a rectangular R (m < P) it is SKIPPED, with
       the reason recorded, rather than forced into a square solve that
       does not exist; skipping a non-gating diagnostic must never raise.
    """
    p_dim = r_factor.shape[-1]
    r_cpu = r_factor.detach().to(device="cpu", dtype=torch.float64)
    try:
        sigma = torch.linalg.svdvals(r_cpu)
    except torch.linalg.LinAlgError:
        fallback("tangent_spectrum_fallbacks", "cpu_lapack_error")
        try:
            sigma = torch.linalg.svdvals(
                r_factor.detach().to(dtype=torch.float64), driver="gesvd"
            )
        except torch.linalg.LinAlgError as retry_error:
            fallback("tangent_spectrum_fallbacks", "cuda_gesvd_retry_failed")
            raise RuntimeError(
                "torch.linalg.svdvals failed on both the CPU default "
                "driver and the CUDA 'gesvd' driver for the streamed "
                "tangent factor R; cannot establish an exact spectrum."
            ) from retry_error
        sigma = sigma.to(device="cpu", dtype=torch.float64)

    if not torch.isfinite(sigma).all():
        fallback("tangent_spectrum_fallbacks", "non_finite_singular_values")
        raise RuntimeError(
            "torch.linalg.svdvals returned non-finite singular values for "
            "the streamed tangent factor R."
        )

    frobenius_sq = float((r_cpu * r_cpu).sum())
    spectrum_sq = float((sigma * sigma).sum())
    frobenius_relative = abs(spectrum_sq - frobenius_sq) / max(frobenius_sq, 1e-300)
    if frobenius_relative > 1e-12:
        fallback("tangent_spectrum_fallbacks", "frobenius_mismatch")
        raise RuntimeError(
            "Spectrum Frobenius identity failed for the streamed tangent "
            f"factor R: sum(sigma^2) relative mismatch "
            f"{frobenius_relative:.3e} against ||R||_F^2 (tolerance 1e-12)."
        )

    sigma_max = float(sigma[0])
    sigma_min = float(sigma[-1])

    # Deterministic starting vectors (a private generator, never the
    # global RNG the surrounding training loop depends on) so these
    # diagnostics are reproducible run to run.
    generator = torch.Generator()
    generator.manual_seed(0)

    # sigma_max sanity: 50 power iterations of R^T R against sigma_max^2.
    v = torch.randn(p_dim, dtype=torch.float64, generator=generator)
    v = v / v.norm()
    for _ in range(50):
        v = r_cpu.t() @ (r_cpu @ v)
        v = v / v.norm().clamp_min(1e-300)
    rayleigh_max = float(v @ (r_cpu.t() @ (r_cpu @ v)))
    power_sigma_max = math.sqrt(max(rayleigh_max, 0.0))
    if sigma_max > 0.0 and abs(power_sigma_max - sigma_max) > 0.05 * sigma_max:
        fallback("tangent_spectrum_fallbacks", "power_iteration_sigma_max_mismatch")

    # sigma_min sanity: 50 inverse power iterations, each ONE forward and
    # ONE backward triangular solve against R^T R -- R is never inverted.
    # This needs a SQUARE R: solve_triangular requires a square operand,
    # and R is only square when the streamed factor has accumulated at
    # least P rows. Whenever the probe has fewer functional rows than
    # parameters (N*K < P, e.g. a small probe against a wide model) R is
    # m x P with m < P for its entire life -- rank(R) <= m < P always, so
    # the SVD-free route can never fire for it anyway (see the rank check
    # in ``_surrogate_from_factor``), and there is no square triangular
    # system to invert here. This is a NON-GATING corroboration, so a
    # shape it cannot run on is skipped, with the reason recorded, rather
    # than forcing a square solve that does not exist.
    is_square = r_cpu.shape[0] == r_cpu.shape[1]
    if not is_square:
        fallback("tangent_spectrum_fallbacks", "rectangular_factor_skips_inverse_power")
    else:
        w = torch.randn(p_dim, dtype=torch.float64, generator=generator)
        w = w / w.norm()
        for _ in range(50):
            z = torch.linalg.solve_triangular(
                r_cpu.t(), w.unsqueeze(1), upper=False
            ).squeeze(1)
            w = torch.linalg.solve_triangular(
                r_cpu, z.unsqueeze(1), upper=True
            ).squeeze(1)
            w = w / w.norm().clamp_min(1e-300)
        z = torch.linalg.solve_triangular(
            r_cpu.t(), w.unsqueeze(1), upper=False
        ).squeeze(1)
        w_final = torch.linalg.solve_triangular(
            r_cpu, z.unsqueeze(1), upper=True
        ).squeeze(1)
        inverse_rayleigh = float(w @ w_final)
        power_sigma_min = (
            1.0 / math.sqrt(inverse_rayleigh) if inverse_rayleigh > 0.0 else 0.0
        )
        if sigma_min > 0.0 and abs(power_sigma_min - sigma_min) > 0.05 * sigma_min:
            fallback("tangent_spectrum_fallbacks", "inverse_power_sigma_min_mismatch")

    # cond(J) = sigma_max/sigma_min; cond(J^T J) = cond(J)^2 -- the
    # quantity that rules out a normal-equation Gram path at this scale
    # (C.0): MEASURED cond(J) ~ 5.9e14 makes cond(J^T J) ~ 3.4e29.
    condition = float("inf") if sigma_min <= 0.0 else sigma_max / sigma_min
    set_value("tangent_condition_estimate", condition)
    return sigma.to(device=r_factor.device)


def _verify_surrogate_oracle(
    r_factor: torch.Tensor,
    b_acc: torch.Tensor,
    r_sq: float,
    out_dtype: torch.dtype,
    j_s: torch.Tensor,
    r_s: torch.Tensor,
) -> None:
    """FGD_TANGENT_VERIFY oracle: the optimized surrogate against the legacy
    one, built from the SAME streamed ``(R, b, ||r||^2)``.

    Compares exactly the quantities every downstream consumer reads --
    ``J^T J``, ``J^T r``, ``||r||^2`` -- and, transitively, the one thing
    every consumer ultimately derives from them,
    ``damping.minimal_relative_error_from_system``, rather than
    re-deriving that formula here and risking the two drifting apart.
    Deferred import: ``fgdlib.search.damping`` imports ``fgdlib.tangent``
    at module scope, so importing it back HERE (never at module load) is
    what avoids the cycle.
    """
    from fgdlib.search import damping as _damping

    legacy_j_s, legacy_r_s = _stream_gram_surrogate(r_factor, b_acc, r_sq, out_dtype)

    def _gram_stats(
        jacobian: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        jacobian64 = jacobian.to(torch.float64)
        residual64 = residual.to(torch.float64)
        gram = jacobian64.t() @ jacobian64
        rhs = jacobian64.t() @ residual64
        return gram, rhs, float((residual64 * residual64).sum())

    gram_opt, rhs_opt, rsq_opt = _gram_stats(j_s, r_s)
    gram_leg, rhs_leg, rsq_leg = _gram_stats(legacy_j_s, legacy_r_s)

    gram_rel = float((gram_opt - gram_leg).norm()) / max(float(gram_leg.norm()), 1.0)
    rhs_rel = float((rhs_opt - rhs_leg).norm()) / max(float(rhs_leg.norm()), 1.0)
    rsq_rel = abs(rsq_opt - rsq_leg) / max(abs(rsq_leg), 1.0)

    config = FGDApproxConfig()
    error_opt = _damping.minimal_relative_error_from_system(
        ExactTangentSystem(jacobian=j_s, target=r_s, parameters=(), loss=0.0),
        config,
    )
    error_leg = _damping.minimal_relative_error_from_system(
        ExactTangentSystem(
            jacobian=legacy_j_s, target=legacy_r_s, parameters=(), loss=0.0
        ),
        config,
    )
    error_rel = abs(error_opt - error_leg) / max(abs(error_leg), 1.0)

    max_relative = max(gram_rel, rhs_rel, rsq_rel, error_rel)
    set_max("tangent_oracle_max_relative_error", max_relative)
    increment("tangent_oracle_verifications")
    if max_relative > 1e-9:
        raise RuntimeError(
            "FGD_TANGENT_VERIFY oracle disagreement between the optimized "
            f"and legacy tangent surrogates: max relative error "
            f"{max_relative:.3e} (gram={gram_rel:.3e}, rhs={rhs_rel:.3e}, "
            f"r_sq={rsq_rel:.3e}, min_rel_error={error_rel:.3e}) exceeds "
            "1e-9."
        )


def _surrogate_from_factor(
    r_factor: torch.Tensor,
    b_acc: torch.Tensor,
    r_sq: float,
    out_dtype: torch.dtype,
    *,
    strict: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SVD-free surrogate when rank == P; legacy CPU-SVD surrogate otherwise.

    For triangular R, ``R^T R = J^T J`` EXACTLY (the streamed incremental-QR
    factor), so whenever R has full column rank, R itself already IS a
    valid surrogate: ``z = R^{-T} b`` solves ``R^T z = b`` exactly, and
    ``[R; 0]``, ``[z; sqrt(residual)]`` reproduce ``(J_s^T J_s, J_s^T r_s,
    ||r_s||^2, row_count)`` with no SVD at all -- see TANGENT_PLAN.md C.2.
    ``[R; 0]`` and the legacy ``[diag(sigma) V^T; 0]`` differ by exactly one
    orthogonal left factor, invisible to every downstream consumer
    enumerated there.

    Rank -- and whether any singular value sits in the ambiguity band where
    CPU vs CUDA LAPACK could disagree about it -- is still decided from the
    EXACT spectrum (``_tangent_spectrum``), so the decision matches the
    legacy rule bit for bit; only the SVD used to BUILD the surrogate is
    skipped when it is safe to.

    ``strict`` mirrors optimized-vs-auto one level down: if the spectrum
    itself cannot be established (``_tangent_spectrum`` exhausts its LAPACK
    retries or fails its Frobenius identity -- both MEASURED to essentially
    never fire), ``strict=True`` (backend="optimized") re-raises so a test
    can prove the fast construction ran to completion; ``strict=False``
    (backend="auto") instead falls back to the legacy surrogate built from
    the SAME streamed ``(R, b, ||r||^2)`` -- the expensive streaming/QR
    work is already done, so "fall back" here means redoing only the cheap
    final assembly, never the Jacobian. Rank-deficiency and threshold
    ambiguity, in contrast, are NORMAL, expected outcomes on realistically
    conditioned data (TANGENT_PLAN.md risk #1) and are handled identically
    regardless of ``strict`` -- they are not failures of the optimized
    backend, just its other branch.
    """
    p_dim = r_factor.shape[-1]
    with timed("tangent_final_factorization_seconds"):
        try:
            sigma = _tangent_spectrum(r_factor)
        except RuntimeError:
            if strict:
                raise
            # _tangent_spectrum already recorded the SPECIFIC reason
            # (frobenius_mismatch / cpu_lapack_error / cuda_gesvd_retry_
            # failed / non_finite_singular_values) before raising; this is
            # the COARSER, call-site reason -- "the whole optimized
            # surrogate construction gave up on the spectrum and deferred
            # to the untouched legacy path" -- so no swallow-and-return
            # here is ever silent, regardless of which of those causes it.
            fallback(
                "tangent_spectrum_fallbacks",
                "spectrum_unavailable_legacy_surrogate_used",
            )
            # The condition estimate could not be established for this
            # call. Leaving tangent_condition_estimate at 0.0 (the
            # profiler's zero-init, or whatever a PREVIOUS successful call
            # left behind) would misreport an impossible reading --
            # condition numbers are always >= 1 -- as though it had been
            # measured now. NaN is the explicit "not measured this call"
            # sentinel: it prints as nan, never as a plausible number.
            set_value("tangent_condition_estimate", float("nan"))
            sigma = None

    if sigma is None:
        return _stream_gram_surrogate(r_factor, b_acc, r_sq, out_dtype)

    with timed("tangent_surrogate_seconds"):
        sigma_max = float(sigma[0]) if sigma.numel() else 0.0
        rank = int((sigma > sigma_max * 1e-12).sum().item()) if sigma.numel() else 0
        ambiguous = False
        if sigma.numel():
            lower, upper = 1e-13 * sigma_max, 1e-11 * sigma_max
            ambiguous = bool(((sigma >= lower) & (sigma <= upper)).any())

    if rank == p_dim and not ambiguous:
        with timed("tangent_surrogate_seconds"):
            r64 = r_factor.detach().to(torch.float64)
            b64 = b_acc.detach().to(torch.float64)
            z = torch.linalg.solve_triangular(
                r64.t(), b64.unsqueeze(1), upper=False
            ).squeeze(1)
            residual_norm = float((r64.t() @ z - b64).norm())
            b_norm = float(b64.norm())
            triangular_ok = residual_norm <= 1e-10 * b_norm

            j_s_out = r_s_out = None
            if triangular_ok:
                res = max(0.0, float(r_sq) - float((z * z).sum()))
                j_s = torch.cat([r64, r64.new_zeros(1, p_dim)], dim=0)
                r_s = torch.cat([z, z.new_tensor([math.sqrt(res)])])
                if not (
                    math.isfinite(r_sq)
                    and r_sq >= 0.0
                    and res >= 0.0
                    and torch.isfinite(r64).all()
                    and torch.isfinite(b64).all()
                    and torch.isfinite(z).all()
                    and torch.isfinite(j_s).all()
                    and torch.isfinite(r_s).all()
                ):
                    raise RuntimeError(
                        "Non-finite quantity while building the SVD-free "
                        "tangent surrogate."
                    )
                j_s_out = j_s.to(out_dtype)
                r_s_out = r_s.to(out_dtype)

        if triangular_ok:
            if _tangent_verify_enabled():
                _verify_surrogate_oracle(
                    r_factor, b_acc, r_sq, out_dtype, j_s_out, r_s_out
                )
            return j_s_out, r_s_out

        fallback("tangent_numerical_fallbacks", "triangular_residual")
    else:
        if rank != p_dim:
            fallback("tangent_surrogate_svd_fallbacks", "rank_deficient")
            increment("tangent_rank_deficient_surrogates")
        else:
            fallback("tangent_surrogate_svd_fallbacks", "threshold_ambiguous")

    # Legacy-shaped surrogate from a full CPU SVD (still ~4x faster than
    # the CUDA default driver, C.4), preserving the (rank+1) x P row-count
    # contract GCV reads as n_observations.
    with timed("tangent_final_factorization_seconds"):
        r_cpu = r_factor.detach().to(device="cpu", dtype=torch.float64)
        left, singular, right = torch.linalg.svd(r_cpu, full_matrices=False)
        del left

    with timed("tangent_surrogate_seconds"):
        keep = singular > float(singular.max()) * 1e-12
        sig = singular[keep].to(device=r_factor.device)
        v_rows = right[keep].to(device=r_factor.device)
        b_acc64 = b_acc.detach().to(torch.float64)
        coeff = v_rows @ b_acc64
        in_range = float((coeff * coeff / sig.square()).sum())
        residual = max(0.0, float(r_sq) - in_range)
        j_top = sig.unsqueeze(1) * v_rows
        j_surrogate = torch.cat([j_top, j_top.new_zeros(1, p_dim)], dim=0)
        r_surrogate = torch.cat([coeff / sig, coeff.new_tensor([math.sqrt(residual)])])
        j_s_out = j_surrogate.to(out_dtype)
        r_s_out = r_surrogate.to(out_dtype)

    if _tangent_verify_enabled():
        _verify_surrogate_oracle(r_factor, b_acc, r_sq, out_dtype, j_s_out, r_s_out)
    return j_s_out, r_s_out


def _stream_gram_surrogate(
    r_factor: torch.Tensor,
    b_acc: torch.Tensor,
    r_sq: float,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A tiny (rank+1) x P system reproducing the FULL-data tangent projection.

    From the streamed incremental-QR factor ``R`` (P x P, whose SVD is the exact
    SVD of the full Jacobian ``J``), ``b = J^T r`` and ``||r||^2`` it builds
    ``(J_s, r_s)`` with ``J_s^T J_s = J^T J``, ``J_s^T r_s = J^T r`` and
    ``||r_s||^2 = ||r||^2``. Every quantity the downstream projection needs --
    the spectrum, the damped update ``u``, and ``eps`` -- depends on the data
    ONLY through those three, so feeding ``(J_s, r_s)`` to the unchanged solver
    gives the exact full-data result at ``O(P^2)`` memory. The extra zero row
    carries the out-of-range residual energy so ``eps`` is exact, not just the
    in-range part.
    """
    # tangent_final_factorization_seconds and tangent_surrogate_seconds are
    # EXCLUSIVE siblings: two separate `with` blocks (not one enclosing both)
    # so that the SVD itself -- the dominant cost this profiler exists to
    # isolate -- is never folded into the cheap assembly time around it.
    with timed("tangent_final_factorization_seconds"):
        left, singular, right = torch.linalg.svd(r_factor, full_matrices=False)
    with timed("tangent_surrogate_seconds"):
        del left
        keep = singular > float(singular.max()) * 1e-12
        sig = singular[keep]
        v_rows = right[keep]  # (rank, P), rows are V^T
        coeff = v_rows @ b_acc  # V^T b, shape (rank,)
        in_range = float((coeff * coeff / sig.square()).sum())
        residual = max(0.0, r_sq - in_range)
        p_dim = r_factor.shape[1]
        j_top = sig.unsqueeze(1) * v_rows  # diag(sig) @ V^T = (rank, P)
        j_surrogate = torch.cat([j_top, j_top.new_zeros(1, p_dim)], dim=0)
        r_surrogate = torch.cat([coeff / sig, coeff.new_tensor([math.sqrt(residual)])])
    return j_surrogate.to(out_dtype), r_surrogate.to(out_dtype)


def _factored_relative_error_estimate(
    factors: tuple[torch.Tensor, torch.Tensor],
    target: torch.Tensor,
    eps: float,
) -> float:
    """``eps`` at least damping, straight off ``W``, for the rank loop only.

    ``J = W V^T`` and the least-damped projection of ``r`` onto range(J) is the
    projection onto range(W), so ``eps = ||P r - r|| / ||P r||`` needs nothing
    but a thin QR of the ``(rows, ell)`` factor. The parameter dimension never
    enters, so deciding whether to spend more rank costs almost nothing.
    """
    left, _right = factors
    flat = target.reshape(-1).to(dtype=left.dtype, device=left.device)
    basis, _ = torch.linalg.qr(left)
    projected = basis @ (basis.t() @ flat)
    norm = float(torch.linalg.vector_norm(projected))
    if not norm > eps:
        return float("inf")
    return float(torch.linalg.vector_norm(projected - flat)) / norm


@contextmanager
def _suspended_module_capture(model: GrowingMLP):
    """``paused_computation`` for a gromo that does not have it.

    The method landed on a branch, so whether the protection exists at all
    depends on which checkout is installed -- and the failure it prevents does
    not surface where it is caused: the modules cache functorch wrappers here,
    and the crash arrives later, inside an unrelated ``copy.deepcopy(model)``.
    A correctness property must not be contingent on a library branch, so the
    two flags gromo's own context manager sets are set here directly, over the
    same module set, and restored the same way.
    """
    flags = [
        (module, module.store_input, module.store_pre_activity)
        for module in model.modules()
        if hasattr(module, "store_input") and hasattr(module, "store_pre_activity")
    ]
    try:
        for module, _, _ in flags:
            module.store_input = False
            module.store_pre_activity = False
        yield
    finally:
        for module, store_input, store_pre_activity in reversed(flags):
            module.store_input = store_input
            module.store_pre_activity = store_pre_activity


def _matrix_free_tangent_system(
    *,
    model: GrowingMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> ExactTangentSystem | None:
    """``J = W V^T`` from a BLOCK range finder, never materialised.

    The exact route is fast because it is a handful of batched matmuls
    (``_analytic_jacobian_block``), not because of ``jacrev`` -- which is only
    its fallback. Its cost is the outer product at the end: the ``(NK, P)``
    matrix, 4.8 GB at MNIST width, plus the ``O(P^2)`` Gram downstream.

    So the same per-layer factors are kept and APPLIED instead of stored, and
    the subspace is found by a block range finder: ``2q+2`` products applied to
    the whole block at once, rather than the ``2k`` strictly sequential ones
    Golub-Kahan needed. That sequential depth was the entire reason the earlier
    version lost to the exact route even at k=64.

    The rank is ADAPTIVE. eps only has to be accurate where it decides, i.e.
    near the 1/2 bar; a cheap eps of 0.05 or 0.9 is not going to change its
    mind. So the rank starts small and doubles only while the answer sits in
    the decision band, which is how precision is bought without paying for it
    everywhere.

    Truncation biases eps UP: ``J_k = J V V^T`` has a smaller range, so less of
    ``r`` is representable. It under-certifies, never the reverse.
    """
    from fgdlib.search.matrixfree import (
        analytic_operators,
        dual_spectrum,
        randomized_factorization,
        vmap_operators,
    )

    was_training = model.training
    model.eval()
    # Held open across the whole vmap path and closed with the model's training
    # flag; see the fallback below for why it cannot be a local `with`.
    vmap_capture = ExitStack()
    try:
        parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        parameter_names = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        with torch.no_grad():
            outputs = model(x)
            target = functional_gradient(
                outputs, y, config.functional_loss
            ).reshape(-1)
            loss = float(
                torch.nn.functional.mse_loss(outputs, y).detach().item()
            )

        # Analytic operators when the structure allows, exactly as the exact
        # route decides it; vmapped autograd otherwise. Same two methods, so
        # nothing below can tell which one it got.
        operators = None
        pause = getattr(model, "paused_computation", None)
        with ExitStack() as capture_stack:
            structure = None
            if callable(pause):
                capture_stack.enter_context(pause())
                structure = _supported_analytic_structure(
                    model, x, capture_suspended=True
                )
                if structure is None:
                    capture_stack.close()
            else:
                structure = _supported_analytic_structure(model, x)
            if structure is not None:
                operators = analytic_operators(structure, x)
        if operators is None:
            increment("matrix_free_vmap_fallbacks")
            # THE CACHE STAYS OFF FOR THE WHOLE VMAP PATH, not merely while the
            # operators are built. gromo's growing modules store their input and
            # pre-activity on EVERY forward -- which is the very reason the
            # analytic structure was refused here
            # (forward_has_caching_side_effects) -- so a vmapped forward leaves
            # functorch TensorWrapper values sitting on the modules, and they
            # outlive this function. MEASURED on Margaret: the run died later,
            # in _transactional_realize_functional_step, on
            # `copy.deepcopy(model)` with "Cannot access storage of
            # TensorWrapper" -- a crash whose stack trace names none of this.
            #
            # The stack is closed in the `finally` below, not here, because the
            # applies that do the caching happen in the factorisation further
            # down, not at construction time.
            #
            # gromo's own paused_computation is preferred when present, but it
            # arrived on a BRANCH (feature/pausable-statistics-capture) and a
            # checkout without it silently loses the protection -- which is how
            # the cluster hit this and a laptop did not. Correctness here
            # cannot depend on which gromo is installed, so fall back to
            # toggling the two flags directly. Same two flags, same restore.
            if callable(pause):
                vmap_capture.enter_context(pause())
            else:
                fallback(
                    "matrix_free_capture_pause_missing",
                    "gromo_without_paused_computation",
                )
                vmap_capture.enter_context(_suspended_module_capture(model))
            operators = vmap_operators(model, x, parameters, parameter_names)

        columns = sum(p.numel() for p in parameters)

        # FACTOR THE SMALLER SIDE. rank(J) <= min(NK, P), so whichever of the
        # two dimensions is smaller bounds the whole spectrum and a basis of
        # that many columns captures the range EXACTLY. Nothing is truncated
        # either way; only the arithmetic changes.
        #
        # This matters because the two regimes really happen:
        #
        #   P < NK   the probe is sized above the interpolation floor, which
        #            certify_probe_kappa now enforces. MNIST: NK 7040, P 1612.
        #   NK < P   the over-parameterised regime, where the dual is the
        #            cheap side. MEASURED at P=19201, NK=1024: the primal Gram
        #            is 2949 MB against the dual's 8.4 MB.
        #
        # The gate is the RATIO, not min(NK, P), because the two routes have
        # different constants and the crossover is not at 1. MEASURED at the
        # real dimensions of both configs:
        #
        #   N1024  NK/P 1.6    dual 0.262 s   P-side 3.19 s   -> dual  12x
        #   MNIST  NK/P 4.37   dual 16.29 s   P-side 3.59 s   -> P-side 4.5x
        #
        # The dual's eigendecomposition is O(NK^3) and gets worse as kappa grows
        # the probe with the net; the P-side pays O(NK P^2) but moves large
        # blocks through the batched applies, which only pays off once NK is
        # several times P. 2x sits between the two measured points and keeps
        # N1024 on exactly the path it is on today.
        #
        # A low-rank SKETCH is a different thing and does not work here: at
        # P=1345 the exact eps is 1e-06 while rank 512 reported 0.21, because r
        # is spread across the whole spectrum. The rank below is the full
        # min(NK, P), so nothing is truncated on either branch.
        #
        # AND THE DUAL ROUTE IS NOT ALWAYS AVAILABLE. dual_gram reads the
        # ANALYTIC factors -- sensitivities, activations, use_bias -- while the
        # vmapped operators expose only the two applies, so on the fallback
        # path it cannot run at all. MEASURED on MNIST at the cluster's width:
        # the analytic structure was refused (forward_has_caching_side_effects)
        # and, with NK 3840 against P 2399, the ratio gate chose the dual
        # branch anyway and raised AttributeError on VmapOperators, killing the
        # run mid-flight. The range finder needs nothing but the applies, so it
        # serves both operator kinds; rank stays min(NK, P), which is the exact
        # bound on rank(J), so nothing is truncated and the analytic path keeps
        # the rank it has today (columns, since it only takes this branch when
        # rows > 2 * columns).
        dual_available = hasattr(operators, "sensitivities")
        if not dual_available:
            fallback("matrix_free_dual_unavailable", "vmap_operators_lack_factors")
        if operators.rows > 2 * columns or not dual_available:
            built = randomized_factorization(
                apply_j=operators.apply_j,
                apply_jt=operators.apply_jt,
                rows=operators.rows,
                columns=columns,
                rank=min(operators.rows, columns),
                oversampling=0,
                power_iterations=1,
                device=x.device,
            )
            if built is None:
                return None
            factors = built
        else:
            spectrum = dual_spectrum(operators, config.eps)
            if spectrum is None:
                return None
            left, singular_values = spectrum
            # The right factor, built by applying J^T to the retained left
            # directions -- one batched apply, no (NK, P) intermediate.
            right = (
                operators.apply_jt(left.t()) / singular_values.unsqueeze(1)
            ).t()
            factors = (left * singular_values, right)
        jacobian_matrix = factors[0].new_zeros((0, columns))
    finally:
        vmap_capture.close()
        model.train(was_training)

    if jacobian_matrix is None:
        return None

    named_buffers = tuple(model.named_buffers())
    return ExactTangentSystem(
        jacobian=jacobian_matrix,
        target=target,
        parameters=parameters,
        loss=loss,
        factors=factors,
        full_target=target,
        owner_model=model,
        parameter_names=parameter_names,
        parameter_versions=tuple(parameter._version for parameter in parameters),
        buffer_names=tuple(name for name, _ in named_buffers),
        buffers=tuple(buffer for _, buffer in named_buffers),
        buffer_versions=tuple(buffer._version for _, buffer in named_buffers),
        probe_x=x,
        probe_y=y,
        probe_versions=(x._version, y._version),
        config_signature=repr(config),
        evaluation_state=tuple(
            (name, module.training) for name, module in model.named_modules()
        ),
    )


def exact_tangent_system(
    model: GrowingMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> ExactTangentSystem | None:
    """Materialise ``J`` and ``r = grad_f L`` -- returns ``None`` if degenerate."""
    increment("exact_tangent_system_calls")
    set_value(
        "current_parameter_count",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )
    if tuple(config.family_order) == ("matrix_free_tangent",):
        # THE ONLY DIFFERENCE between the two ladder configurations. Every
        # rung, the realisation path, the fallback and the growth loop below
        # this point are the ladder's own code, reached by the ladder's own
        # route; only the construction of J is approximated. Gated on the
        # family so the default config takes the branch below untouched.
        with timed("exact_tangent_system_seconds"):
            return _matrix_free_tangent_system(
                model=model, x=x, y=y, config=config
            )
    with timed("exact_tangent_system_seconds"):
        step = _compute_exact_tangent_projection_step(
            model=model, x=x, y=y, config=config, return_system=True
        )
    return step if isinstance(step, ExactTangentSystem) else None


def _compute_exact_tangent_projection_step(
    model: GrowingMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
    return_system: bool = False,
) -> _TangentProjectionStep | ExactTangentSystem | None:
    """Compute g = P_T grad L by explicitly materializing the full Jacobian.

    Profiler parent/child contract (Phase A instrumentation, no algorithmic
    change): ``tangent_system_total_seconds`` is INCLUSIVE of this entire
    function body -- placed here rather than in ``exact_tangent_system`` so it
    also covers the ``_compute_tangent_projection_step`` -> exact path, which
    reaches this function without ever calling ``exact_tangent_system``.
    ``streamed_jacobian_seconds`` (below) is an existing, separate INCLUSIVE
    timer around the whole streaming/jacrev construction -- it is a NESTED
    PARENT of ``tangent_jacrev_seconds`` / ``tangent_jacobian_flatten_seconds``
    / ``tangent_jacobian_cast_seconds`` / ``tangent_jtr_seconds`` /
    ``tangent_qr_seconds`` / ``tangent_surrogate_seconds`` /
    ``tangent_final_factorization_seconds``, NOT a sibling of
    ``tangent_system_total_seconds``'s other children -- never add it to them,
    it would double-count. All other new sub-timers introduced below are
    EXCLUSIVE and mutually disjoint: summing them (plus
    ``tangent_forward_target_seconds``) reproduces
    ``tangent_system_total_seconds`` up to scheduling slack.
    """
    increment("tangent_system_calls")
    with timed("tangent_system_total_seconds"):
        # Resolved ONCE per call, read directly from the environment so
        # tests can monkeypatch it: legacy keeps every decision below
        # bit-for-bit identical to before this backend existed.
        backend = tangent_backend()
        if backend == "legacy":
            increment("tangent_backend_legacy_calls")
        else:
            increment("tangent_backend_optimized_calls")

        named_parameters = _trainable_named_parameters(model)
        if not named_parameters:
            raise RuntimeError("FGD tangent projection requires trainable parameters.")

        parameter_names = tuple(named_parameters.keys())
        parameters = tuple(named_parameters.values())
        buffers = OrderedDict(model.named_buffers())

        # Measure in eval. Besides the usual reason (the projection is a
        # statement about f, so dropout is the identity and batch-norm uses
        # its fixed running stats), there is a hard requirement: a batch-norm
        # in TRAIN mode updates its running buffers in place, and inside a
        # functorch transform (jacrev / jvp) that in-place update wraps the
        # buffers in functorch tensors that leak out -- the next transform
        # then dies with a level-escape. eval() removes it and is a no-op on
        # the plain MLP. Restored in the finally.
        was_training = model.training
        model.eval()

        with timed("tangent_forward_target_seconds"):
            output = model(x)
            loss = batch_functional_loss(output, y, config.functional_loss)
            if not torch.isfinite(loss).all():
                raise RuntimeError(
                    f"Non-finite FGD loss detected before projection: {loss}."
                )

            target_tensor = torch.autograd.grad(loss, output)[0].detach()
            # Roughness penalty: r_rough = r_data + 2 gamma Lambda f. Injected
            # here so growth (via minimal_relative_error), the projection, the
            # realise path and GCV all certify against the SAME regularised
            # gradient.
            roughness = _roughness_target(output, x, config)
            if roughness is not None:
                target_tensor = target_tensor + roughness.to(target_tensor.dtype)
            target = target_tensor.reshape(-1)
            full_target = target
            output_numel = target.numel()

        # Last-value gauges (overwrite semantics is fine here -- one call, one
        # shape): the parameter count and output row count that size every
        # matrix built below, so a profile can be read against P and N*K
        # without re-deriving them from the model.
        parameter_numel = sum(parameter.numel() for parameter in parameters)
        set_value("tangent_parameter_count", parameter_numel)
        set_value("tangent_output_row_count", output_numel)

        if torch.linalg.norm(target) <= torch.finfo(target.dtype).eps:
            zero_updates = tuple(
                torch.zeros_like(parameter) for parameter in parameters
            )
            output_error = _output_relative_error_from_stats(
                dot_product=0.0,
                approximation_sq_norm=0.0,
                target_sq_norm=0.0,
                eps=config.eps,
            )
            if return_system:
                return None
            return _TangentProjectionStep(
                output_error=output_error,
                parameter_updates=zero_updates,
                learning_rate_used=0.0,
                loss_before=float(loss.detach().item()),
                loss_after=float(loss.detach().item()),
                descent_ok=True,
                dot_product=0.0,
                approximation_sq_norm=0.0,
                target_sq_norm=0.0,
            )

        def call_with_parameters(
            parameter_values: tuple[torch.Tensor, ...],
        ) -> torch.Tensor:
            state = OrderedDict(zip(parameter_names, parameter_values))
            state.update(buffers)
            return functional_call(model, state, (x,)).reshape(-1)

        chunk = int(getattr(config, "jacobian_row_chunk", 0) or 0)
        stream_gram = bool(getattr(config, "certify_stream_gram", False))
        # Held open for the whole construction and released in the finally
        # below, alongside the eval/train restore, so a suspended capture can
        # never outlive the forward that needed it -- not even on an exception.
        capture_stack = ExitStack()
        try:
            with timed("streamed_jacobian_seconds"):
                # Structure resolved ONCE per call, SHARED by every branch
                # below (streamed-Gram, row-chunked, full): optimized/auto
                # try the analytic Jacobian block wherever the model/probe
                # qualify (P1-P14); optimized RAISES if they do not; auto
                # falls back to jacrev, per block, in whichever branch runs.
                # A model/probe either supports the analytic form or it does
                # not -- that has nothing to do with which of the three
                # branches a config happens to select, so this is resolved
                # exactly once and shared, never re-derived per branch.
                #
                # ``tangent_backend_inapplicable_paths`` is a second, COARSE
                # counter alongside the specific
                # ``tangent_unsupported_structure_fallbacks`` reason: it
                # exists so "backend != legacy but a block was built without
                # the analytic path" can never again go completely
                # uncounted, regardless of which branch or future reason
                # causes it (see the family_ladder_N1024 gap this closes).
                structure: AnalyticMLPStructure | None = None
                if backend != "legacy":
                    # gromo's paused_computation() suspends input/pre-activity
                    # capture WITHOUT discarding the accumulated statistics
                    # (reset_computation would discard them), which is exactly
                    # what an evaluation-only forward inside a growth-
                    # certification session needs. Entering it here satisfies
                    # P8 by construction instead of falling back: MEASURED, the
                    # capture flags were on for 499 of 501 constructions on
                    # family_ladder_N1024 and 5 of 14 on CIFAR, so this is the
                    # difference between the analytic path running everywhere
                    # and running almost nowhere.
                    #
                    # Only entered when the analytic path is what we are about
                    # to take. A legacy-jacrev fallback must keep running with
                    # the caller's own capture state, exactly as it does today.
                    pause = getattr(model, "paused_computation", None)
                    if callable(pause):
                        capture_stack.enter_context(pause())
                        increment("tangent_capture_suspensions")
                        structure = _supported_analytic_structure(
                            model, x, capture_suspended=True
                        )
                        if structure is None:
                            # Unsupported for some OTHER reason, so the
                            # suspension buys nothing and must not colour the
                            # legacy fallback: release it before falling back.
                            capture_stack.close()
                    else:
                        structure = _supported_analytic_structure(model, x)
                    if structure is None:
                        if backend == "optimized":
                            raise RuntimeError(
                                "FGD_TANGENT_BACKEND=optimized requires the "
                                "analytic MLP Jacobian structure, but the "
                                "model/probe did not satisfy it (see the "
                                "tangent_unsupported_structure_fallbacks "
                                "reason recorded above)."
                            )
                        fallback(
                            "tangent_backend_inapplicable_paths",
                            "unsupported_structure",
                        )

                if stream_gram:
                    # STREAMING over SAMPLES: accumulate the incremental-QR
                    # factor R (P x P), b = J^T r and ||r||^2 batch by batch --
                    # each sample forwarded EXACTLY ONCE (chunking output rows
                    # instead recomputes the whole forward per chunk, the
                    # dominant cost) -- freeing each block, so the full NK x P
                    # Jacobian is never held. The downstream then gets a tiny
                    # surrogate ((rank+1) x P) reproducing the exact spectrum
                    # and projection. Memory O(P^2), exact (1.8e-12 vs full
                    # float64).
                    n_samples = int(x.shape[0])
                    out_per_sample = output_numel // max(n_samples, 1)
                    sample_chunk = int(getattr(config, "certify_stream_chunk", 0) or 0)
                    if sample_chunk <= 0:
                        sample_chunk = n_samples

                    # qr(mode="r") runs the same geqrf as mode="reduced" and
                    # skips the discarded orgqr -- MEASURED bit-identical R
                    # (C.3). This is a property of "not legacy", independent
                    # of whether the analytic block above is available, so
                    # it applies even when auto fell back to jacrev blocks.
                    qr_mode = "reduced" if backend == "legacy" else "r"

                    r_factor = None
                    b_acc: torch.Tensor | None = None
                    r_sq = 0.0
                    for start in range(0, n_samples, sample_chunk):
                        stop = min(start + sample_chunk, n_samples)
                        x_batch = x[start:stop]

                        def call_batch(
                            parameter_values: tuple[torch.Tensor, ...],
                            _xb: torch.Tensor = x_batch,
                        ) -> torch.Tensor:
                            state = OrderedDict(zip(parameter_names, parameter_values))
                            state.update(buffers)
                            return functional_call(model, state, (_xb,)).reshape(-1)

                        rows = (stop - start) * out_per_sample
                        # Per-chunk bookkeeping BEFORE the block is built, so
                        # the peak-shape gauges reflect the block about to be
                        # materialized even if jacrev itself later raises.
                        increment("tangent_sample_chunk_count")
                        set_max("tangent_peak_jacobian_block_rows", rows)
                        set_max("tangent_peak_jacobian_block_columns", parameter_numel)
                        if structure is not None:
                            with timed("tangent_analytic_jacobian_seconds"):
                                increment("tangent_analytic_jacobian_calls")
                                block = _analytic_jacobian_block(structure, x_batch)
                        else:
                            with timed("tangent_jacrev_seconds"):
                                jac = jacrev(call_batch)(parameters)
                            with timed("tangent_jacobian_flatten_seconds"):
                                flat = _flatten_jacobian(jac, rows)
                            with timed("tangent_jacobian_cast_seconds"):
                                block = flat.to(torch.float64)
                        with timed("tangent_jtr_seconds"):
                            r_block = target[
                                start * out_per_sample : stop * out_per_sample
                            ].to(torch.float64)
                            contribution = block.t() @ r_block
                            b_acc = (
                                contribution if b_acc is None else b_acc + contribution
                            )
                            r_sq += float((r_block * r_block).sum())
                        with timed("tangent_qr_seconds"):
                            stacked = (
                                block
                                if r_factor is None
                                else torch.cat([r_factor, block], dim=0)
                            )
                            increment("tangent_qr_calls")
                            set_max("tangent_qr_input_rows", stacked.shape[0])
                            set_max("tangent_qr_input_columns", stacked.shape[1])
                            if qr_mode == "r" and _tangent_verify_enabled():
                                # Risk #3: mode="r" is only MEASURED bit-
                                # identical on this build. Never pay for a
                                # second QR in production -- only under the
                                # opt-in oracle.
                                r_reduced = torch.linalg.qr(stacked, mode="reduced").R
                                r_fast = torch.linalg.qr(stacked, mode="r").R
                                if torch.equal(r_reduced, r_fast):
                                    r_factor = r_fast
                                else:
                                    fallback(
                                        "tangent_numerical_fallbacks",
                                        "qr_mode_r_mismatch",
                                    )
                                    r_factor = r_reduced
                            else:
                                r_factor = torch.linalg.qr(stacked, mode=qr_mode).R
                        del block, r_block, stacked
                        _clear_inaccessible_tensor_caches(model)
                    if backend == "legacy":
                        jacobian_matrix, target = _stream_gram_surrogate(
                            r_factor, b_acc, r_sq, target.dtype
                        )
                    else:
                        jacobian_matrix, target = _surrogate_from_factor(
                            r_factor,
                            b_acc,
                            r_sq,
                            target.dtype,
                            strict=(backend == "optimized"),
                        )
                elif chunk > 0 and chunk < output_numel:
                    # This branch chunks over OUTPUT ROWS (the flattened
                    # n*K+k index), NOT samples -- a chunk boundary can cut
                    # a sample in half. The analytic block builder only
                    # knows how to build a block for a contiguous range of
                    # SAMPLES, so for each row range [start, stop) we build
                    # the analytic block for the smallest covering sample
                    # range [n_start, n_stop) and slice out exactly
                    # [start, stop) from it. Since row r within that
                    # covering block is (n - n_start)*K + k for sample n and
                    # output k, and the FULL flattened row is n*K + k, the
                    # two differ by exactly n_start*K for every row in the
                    # covering block -- so slicing
                    # [start - n_start*K : stop - n_start*K] out of the
                    # covering block reproduces precisely rows
                    # [start, stop) of the full Jacobian, bit for bit.
                    blocks = []
                    out_per_sample = output_numel // max(int(x.shape[0]), 1)
                    for start in range(0, output_numel, chunk):
                        stop = min(start + chunk, output_numel)
                        if structure is not None:
                            n_start = start // out_per_sample
                            n_stop = (stop - 1) // out_per_sample + 1
                            with timed("tangent_analytic_jacobian_seconds"):
                                increment("tangent_analytic_jacobian_calls")
                                sample_block = _analytic_jacobian_block(
                                    structure,
                                    x[n_start:n_stop],
                                    out_dtype=x.dtype,
                                )
                            row_offset = start - n_start * out_per_sample
                            blocks.append(
                                sample_block[row_offset : row_offset + (stop - start)]
                            )
                        else:

                            def call_rows(
                                parameter_values: tuple[torch.Tensor, ...],
                                _start: int = start,
                                _stop: int = stop,
                            ) -> torch.Tensor:
                                return call_with_parameters(parameter_values)[
                                    _start:_stop
                                ]

                            with timed("tangent_jacrev_seconds"):
                                jac = jacrev(call_rows)(parameters)
                            with timed("tangent_jacobian_flatten_seconds"):
                                blocks.append(_flatten_jacobian(jac, stop - start))
                        _clear_inaccessible_tensor_caches(model)
                    jacobian_matrix = torch.cat(blocks, dim=0)
                else:
                    if structure is not None:
                        with timed("tangent_analytic_jacobian_seconds"):
                            increment("tangent_analytic_jacobian_calls")
                            jacobian_matrix = _analytic_jacobian_block(
                                structure, x, out_dtype=x.dtype
                            )
                    else:
                        with timed("tangent_jacrev_seconds"):
                            jac = jacrev(call_with_parameters)(parameters)
                        with timed("tangent_jacobian_flatten_seconds"):
                            jacobian_matrix = _flatten_jacobian(jac, output_numel)
        finally:
            capture_stack.close()
            model.train(was_training)
            _clear_inaccessible_tensor_caches(model)
        if return_system:
            named_buffers = tuple(model.named_buffers())
            return ExactTangentSystem(
                jacobian=jacobian_matrix,
                target=target,
                parameters=parameters,
                loss=float(loss.detach().item()),
                full_target=full_target,
                owner_model=model,
                parameter_names=parameter_names,
                parameter_versions=tuple(
                    parameter._version for parameter in parameters
                ),
                buffer_names=tuple(name for name, _ in named_buffers),
                buffers=tuple(buffer for _, buffer in named_buffers),
                buffer_versions=tuple(buffer._version for _, buffer in named_buffers),
                probe_x=x,
                probe_y=y,
                probe_versions=(x._version, y._version),
                config_signature=repr(config),
                evaluation_state=tuple(
                    (name, module.training) for name, module in model.named_modules()
                ),
            )
        with timed("tangent_projection_solve_seconds"):
            flat_update, approximation = _solve_tangent_projection(
                jacobian_matrix=jacobian_matrix,
                target=target,
                damping=config.projection_damping,
                solver=config.projection_solver,
                work_dtype=(
                    torch.float32
                    if getattr(config, "projection_fast_factorization", False)
                    else torch.float64
                ),
            )
        with timed("tangent_sensor_seconds"):
            if (
                not torch.isfinite(flat_update).all()
                or not torch.isfinite(approximation).all()
            ):
                raise RuntimeError("Non-finite FGD tangent projection update detected.")

            stats = _output_relative_error_from_tensors(
                approximation=approximation,
                target=target,
                eps=config.eps,
            )
            parameter_updates = _unflatten_parameter_update(flat_update, parameters)
        return _TangentProjectionStep(
            output_error=stats.output_error,
            parameter_updates=parameter_updates,
            learning_rate_used=0.0,
            loss_before=float(loss.detach().item()),
            loss_after=float(loss.detach().item()),
            descent_ok=True,
            dot_product=stats.dot_product,
            approximation_sq_norm=stats.approximation_sq_norm,
            target_sq_norm=stats.target_sq_norm,
        )


def _compute_cg_tangent_projection_step(
    model: GrowingMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> _TangentProjectionStep:
    """Compute g = P_T grad L with implicit CG products through the Jacobian."""
    named_parameters = _trainable_named_parameters(model)
    if not named_parameters:
        raise RuntimeError("FGD tangent projection requires trainable parameters.")

    parameter_names = tuple(named_parameters.keys())
    parameters = tuple(named_parameters.values())
    buffers = OrderedDict(model.named_buffers())

    # Measure in eval. Besides the usual reason (the projection is a statement
    # about f, so dropout is the identity and batch-norm uses its fixed running
    # stats), there is a hard requirement: a batch-norm in TRAIN mode updates
    # its running buffers in place, and inside a functorch transform (jacrev /
    # jvp) that in-place update wraps the buffers in functorch tensors that
    # leak out -- the next transform then dies with a level-escape. eval()
    # removes it and is a no-op on the plain MLP. Restored in the finally.
    was_training = model.training
    model.eval()

    output = model(x)
    loss = batch_functional_loss(output, y, config.functional_loss)
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"Non-finite FGD loss detected before projection: {loss}.")

    target_tensor = torch.autograd.grad(loss, output, retain_graph=True)[0].detach()
    target = target_tensor.reshape(-1)
    output_numel = target.numel()
    parameter_numel = sum(parameter.numel() for parameter in parameters)

    if torch.linalg.norm(target) <= torch.finfo(target.dtype).eps:
        zero_updates = tuple(torch.zeros_like(parameter) for parameter in parameters)
        output_error = _output_relative_error_from_stats(
            dot_product=0.0,
            approximation_sq_norm=0.0,
            target_sq_norm=0.0,
            eps=config.eps,
        )
        return _TangentProjectionStep(
            output_error=output_error,
            parameter_updates=zero_updates,
            learning_rate_used=0.0,
            loss_before=float(loss.detach().item()),
            loss_after=float(loss.detach().item()),
            descent_ok=True,
            dot_product=0.0,
            approximation_sq_norm=0.0,
            target_sq_norm=0.0,
        )

    def call_with_parameters(
        parameter_values: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        state = OrderedDict(zip(parameter_names, parameter_values))
        state.update(buffers)
        return functional_call(model, state, (x,))

    def jvp_parameters_to_output(
        parameter_tangent: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        _, tangent_output = jvp(
            call_with_parameters,
            (parameters,),
            (parameter_tangent,),
        )
        return tangent_output.reshape(-1).detach()

    def vjp_output_to_parameters(vector: torch.Tensor) -> tuple[torch.Tensor, ...]:
        gradients = torch.autograd.grad(
            output,
            parameters,
            grad_outputs=vector.reshape_as(output),
            retain_graph=True,
            allow_unused=True,
        )
        return tuple(
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for parameter, gradient in zip(parameters, gradients)
        )

    def output_kernel_matvec(vector: torch.Tensor) -> torch.Tensor:
        parameter_tangent = vjp_output_to_parameters(vector)
        return jvp_parameters_to_output(parameter_tangent)

    damping = max(float(config.projection_damping), 0.0)

    def damped_output_kernel_matvec(vector: torch.Tensor) -> torch.Tensor:
        product = output_kernel_matvec(vector)
        if damping > 0.0:
            product = product + damping * vector
        return product

    try:
        if output_numel <= parameter_numel:
            dual_solution = _conjugate_gradient(
                damped_output_kernel_matvec,
                target,
                max_iterations=config.cg_max_iterations,
                tolerance=config.cg_tolerance,
                eps=config.eps,
            )
            approximation = output_kernel_matvec(dual_solution)
            parameter_updates = vjp_output_to_parameters(dual_solution)
        else:
            rhs = _flatten_parameter_tensors(vjp_output_to_parameters(target))

            def damped_parameter_gram_matvec(vector: torch.Tensor) -> torch.Tensor:
                parameter_tangent = _unflatten_parameter_update(vector, parameters)
                output_tangent = jvp_parameters_to_output(parameter_tangent)
                product = _flatten_parameter_tensors(
                    vjp_output_to_parameters(output_tangent)
                )
                if damping > 0.0:
                    product = product + damping * vector
                return product

            flat_update = _conjugate_gradient(
                damped_parameter_gram_matvec,
                rhs,
                max_iterations=config.cg_max_iterations,
                tolerance=config.cg_tolerance,
                eps=config.eps,
            )
            parameter_updates = _unflatten_parameter_update(flat_update, parameters)
            approximation = jvp_parameters_to_output(parameter_updates)
    finally:
        model.train(was_training)
        _clear_inaccessible_tensor_caches(model)

    if not torch.isfinite(approximation).all() or any(
        not torch.isfinite(update).all() for update in parameter_updates
    ):
        raise RuntimeError("Non-finite FGD CG tangent projection update detected.")

    stats = _output_relative_error_from_tensors(
        approximation=approximation,
        target=target,
        eps=config.eps,
    )
    return _TangentProjectionStep(
        output_error=stats.output_error,
        parameter_updates=parameter_updates,
        learning_rate_used=0.0,
        loss_before=float(loss.detach().item()),
        loss_after=float(loss.detach().item()),
        descent_ok=True,
        dot_product=stats.dot_product,
        approximation_sq_norm=stats.approximation_sq_norm,
        target_sq_norm=stats.target_sq_norm,
    )


def _grouped_batches(
    data_loader: torch.utils.data.DataLoader,
    group_size: int,
):
    """Yield concatenations of ``group_size`` consecutive mini-batches."""
    if group_size <= 1:
        yield from data_loader
        return

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in data_loader:
        xs.append(x)
        ys.append(y)
        if len(xs) == group_size:
            yield torch.cat(xs), torch.cat(ys)
            xs, ys = [], []
    if xs:
        yield torch.cat(xs), torch.cat(ys)


def build_projection_probe(
    data_loader: torch.utils.data.DataLoader,
    probe_batches: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate the first ``probe_batches`` mini-batches into one probe.

    The probe is the joint certification sample: every certificate solves a
    SINGLE shared-direction projection over it (conceptually the stacked
    system J_probe u = r_probe). Materialize the probe once per data source
    and pass the same tensors to every certificate evaluation, so family
    comparisons, growth-layer trials and consecutive epochs all measure on
    the same fixed probe.
    """
    if probe_batches < 1:
        raise ValueError("fgd_approx.probe_batches must be a positive integer.")
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in data_loader:
        xs.append(x)
        ys.append(y)
        if len(xs) == probe_batches:
            break
    if not xs:
        raise ValueError("Cannot build a projection probe from an empty data loader.")
    x = torch.cat(xs)
    y = torch.cat(ys)
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    return x, y


def _compute_tangent_projection_step(
    model: GrowingMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> _TangentProjectionStep:
    """Compute g = P_T grad L and the parameter update that realizes it."""
    if config.projection_solver in {"exact", "exact_svd", "exact_kernel_eigh"}:
        return _compute_exact_tangent_projection_step(
            model=model,
            x=x,
            y=y,
            config=config,
        )

    if config.projection_solver == "cg":
        return _compute_cg_tangent_projection_step(
            model=model,
            x=x,
            y=y,
            config=config,
        )

    raise ValueError(
        f"Unsupported fgd_approx.projection_solver '{config.projection_solver}'. "
        "Use one of: cg, exact, exact_svd, exact_kernel_eigh."
    )


def _apply_tangent_projection_step(
    model: GrowingMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    step: _TangentProjectionStep,
    learning_rate: float,
    config: FGDApproxConfig,
) -> _TangentProjectionStep:
    named_parameters = _trainable_named_parameters(model)
    parameters = tuple(named_parameters.values())
    base_parameters = tuple(parameter.detach().clone() for parameter in parameters)
    base_loss = step.loss_before
    directional_derivative = step.dot_product

    def apply(step_learning_rate: float) -> None:
        with torch.no_grad():
            for parameter, base, update in zip(
                parameters,
                base_parameters,
                step.parameter_updates,
            ):
                parameter.copy_(base)
                parameter.add_(update, alpha=-step_learning_rate)

    @torch.no_grad()
    def trial_loss() -> float:
        return float(
            batch_functional_loss(model(x), y, config.functional_loss).detach().item()
        )

    if learning_rate <= config.eps or directional_derivative <= config.eps:
        apply(0.0)
        return _TangentProjectionStep(
            output_error=step.output_error,
            parameter_updates=step.parameter_updates,
            learning_rate_used=0.0,
            loss_before=step.loss_before,
            loss_after=trial_loss(),
            descent_ok=directional_derivative <= config.eps,
            dot_product=step.dot_product,
            approximation_sq_norm=step.approximation_sq_norm,
            target_sq_norm=step.target_sq_norm,
        )

    if config.learning_rate_policy == "theory_interval":
        # The theoretical interval is certified on held-out validation data by
        # the pipeline. Training batches only realize that certified step.
        apply(learning_rate)
        accepted_loss = trial_loss()
        if not math.isfinite(accepted_loss):
            apply(0.0)
            accepted_loss = trial_loss()
            return _TangentProjectionStep(
                output_error=step.output_error,
                parameter_updates=step.parameter_updates,
                learning_rate_used=0.0,
                loss_before=step.loss_before,
                loss_after=accepted_loss,
                descent_ok=False,
                dot_product=step.dot_product,
                approximation_sq_norm=step.approximation_sq_norm,
                target_sq_norm=step.target_sq_norm,
            )
        return _TangentProjectionStep(
            output_error=step.output_error,
            parameter_updates=step.parameter_updates,
            learning_rate_used=learning_rate,
            loss_before=step.loss_before,
            loss_after=accepted_loss,
            descent_ok=True,
            dot_product=step.dot_product,
            approximation_sq_norm=step.approximation_sq_norm,
            target_sq_norm=step.target_sq_norm,
        )

    if config.sufficient_descent_c is None:
        apply(learning_rate)
        accepted_loss = trial_loss()
        return _TangentProjectionStep(
            output_error=step.output_error,
            parameter_updates=step.parameter_updates,
            learning_rate_used=learning_rate,
            loss_before=step.loss_before,
            loss_after=accepted_loss,
            descent_ok=accepted_loss <= step.loss_before + config.eps,
            dot_product=step.dot_product,
            approximation_sq_norm=step.approximation_sq_norm,
            target_sq_norm=step.target_sq_norm,
        )

    trial_learning_rate = learning_rate
    min_learning_rate = learning_rate * max(config.lr_min_factor, 0.0)
    accepted_learning_rate: float | None = None
    accepted_loss = base_loss
    while trial_learning_rate >= min_learning_rate:
        apply(trial_learning_rate)
        current_loss = trial_loss()
        sufficient_decrease = (
            base_loss
            - config.sufficient_descent_c * trial_learning_rate * directional_derivative
        )
        if current_loss <= sufficient_decrease:
            accepted_learning_rate = trial_learning_rate
            accepted_loss = current_loss
            break
        trial_learning_rate *= config.lr_backtrack

    descent_ok = accepted_learning_rate is not None
    if accepted_learning_rate is None:
        accepted_learning_rate = max(trial_learning_rate, min_learning_rate)
        apply(accepted_learning_rate)
        accepted_loss = trial_loss()

    return _TangentProjectionStep(
        output_error=step.output_error,
        parameter_updates=step.parameter_updates,
        learning_rate_used=accepted_learning_rate,
        loss_before=step.loss_before,
        loss_after=accepted_loss,
        descent_ok=descent_ok,
        dot_product=step.dot_product,
        approximation_sq_norm=step.approximation_sq_norm,
        target_sq_norm=step.target_sq_norm,
    )


@torch.no_grad()
def _layer_functional_error(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    layer_index: int,
    device: torch.device,
    config: FGDApproxConfig,
) -> FGDLayerRelError:
    model.set_scaling_factor(1.0)
    model.eval()

    dot_product = 0.0
    approximation_sq_norm = 0.0
    target_sq_norm = 0.0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        y_before = model(x)
        y_candidate = _forward_with_tiny_update(model, x)

        approximation = y_before - y_candidate
        target = functional_gradient(y_before, y, config.functional_loss)

        dot_product += float(torch.sum(approximation * target).detach().item())
        approximation_sq_norm += float(torch.sum(approximation**2).detach().item())
        target_sq_norm += float(torch.sum(target**2).detach().item())

    return _relative_error_from_stats(
        layer_index=layer_index,
        dot_product=dot_product,
        approximation_sq_norm=approximation_sq_norm,
        target_sq_norm=target_sq_norm,
        config=config,
    )


def marchenko_pastur_significant_count(
    singular_values: torch.Tensor,
    sample_count: int,
) -> int:
    """How many extension directions stand above the NOISE, by RMT alone.

    The stopping question -- "is this bottleneck real, or is it what an
    all-noise extension would have produced anyway?" -- answered without a
    threshold anyone has to choose. The extension's singular values come from
    ``S^{-1/2} N``, i.e. an ALREADY WHITENED matrix estimated from a finite
    probe, which is exactly the setting Marchenko-Pastur describes: if there
    is no signal, the squared singular values fill a bulk whose upper edge is
    ``sigma^2 (1 + sqrt(gamma))^2`` with ``gamma = r / n``, ``r`` the
    extension's dimension and ``n`` the probe's sample count. Anything above
    that edge cannot be explained by sampling noise.

    Nothing here is predetermined and nothing is fitted to a dataset: the
    edge's shape comes from random-matrix theory, ``gamma`` from the two
    dimensions, and ``sigma^2`` from the spectrum's OWN bulk -- the mean of
    the lower half of the squared values, which the top (signal) directions
    cannot contaminate. Magnitude cancels, because the edge scales with the
    same ``sigma^2`` the spectrum does. The identical rule therefore applies
    to smooth_sin and MNIST with no re-tuning.

    LIMIT, stated because it bites here: MP is an asymptotic statement, and a
    layer of width 3 gives r = 3, where "the lower half" is a single number
    and the bulk estimate is worth little. Read the count as a strong signal
    when ``r`` is large and as a weak one when it is tiny.
    """
    if singular_values is None or singular_values.numel() == 0:
        return 0
    squared = (singular_values.detach().double() ** 2).reshape(-1)
    rank = int(squared.numel())
    if rank == 0 or sample_count <= 0:
        return 0
    gamma = rank / float(sample_count)
    bulk_size = max(1, rank // 2)
    bulk = torch.sort(squared).values[:bulk_size]
    sigma_squared = float(bulk.mean())
    if not (sigma_squared > 0.0 and math.isfinite(sigma_squared)):
        return 0
    edge = sigma_squared * (1.0 + math.sqrt(gamma)) ** 2
    return int(torch.sum(squared > edge).item())


def _stride_fold_masks(count: int, folds: int) -> list[torch.Tensor]:
    """Fold membership by STRIDE (``i % K``), never by contiguous block.

    The probe arrives in whatever order its loader produced it, so a
    contiguous split hands each fold a different slice of that order -- on a
    class-grouped or sorted source that is not a resample of the same
    distribution, it is a different dataset per fold. Striding gives every
    fold the same distribution whatever the order is, and it is a pure
    function of the index, so the split is identical on every call. The
    pipeline is deterministic and this must not be the thing that breaks it.
    """
    index = torch.arange(count)
    return [index % folds == fold for fold in range(folds)]


def _materialize_loader(
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Every batch as two tensors, plus the batch size to re-cut them with.

    The folds have to be cut from the WHOLE probe, not from one batch at a
    time, so the source is drained once here. The original batch size is
    carried back out so each fold is fed to ``compute_statistics`` in the same
    sized pieces the caller was already using -- the accumulation is over
    batches either way, but the memory profile stays the one the config chose.
    """
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    batch_size = 0
    for x, y in data_loader:
        xs.append(x.to(device))
        ys.append(y.to(device))
        batch_size = max(batch_size, int(x.shape[0]))
    if not xs:
        raise ValueError("Cannot cross-validate a bottleneck on an empty loader.")
    return torch.cat(xs), torch.cat(ys), batch_size


def _fold_loader(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """A deterministic, unshuffled loader over one fold of the probe."""
    rows = int(inputs.shape[0])
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets),
        batch_size=max(1, min(int(batch_size), rows)) if rows else 1,
        shuffle=False,
    )


def _fitted_extension_matrix(layer: LinearGrowingModule) -> torch.Tensor | None:
    """``alpha omega``, the fitted extension direction, in ``N``'s own shape.

    ``compute_optimal_added_parameters`` builds ``alpha = sqrt(s) S^{-1/2} U``
    and ``omega = sqrt(s) V`` from the SVD of ``P = S^{-1/2} N``, and stores
    them transposed across the two Linear layers. Their outer product is
    ``S^{-1/2} P_k``, so its Frobenius inner product with ``N`` is
    ``trace(P_k^T P) = sum(s_i^2)`` -- the bottleneck, exactly and not by
    analogy. That identity is the entire reason this matrix is the right
    object to carry to a fold it was not fitted on: scored against the
    held-out fold's own ``N`` it returns the same quantity, measured where
    the direction was not chosen.

    ``None`` when the structure is not the two-Linear case the identity is
    derived for; the caller must read that as "not measured", never as "not
    significant".
    """
    extended_input = getattr(layer, "extended_input_layer", None)
    previous = getattr(layer, "previous_module", None)
    extended_output = getattr(previous, "extended_output_layer", None)
    if extended_input is None or extended_output is None:
        return None
    omega = extended_input.weight.detach().double()  # (out_features, k)
    alpha = extended_output.weight.detach().double()  # (k, prev_in_features)
    bias = getattr(extended_output, "bias", None)
    if bias is not None:
        # gromo split alpha as ``alpha[:, :-1]`` and ``alpha[:, -1]``; putting
        # the bias back as the LAST column is what restores N's row order.
        alpha = torch.cat([alpha, bias.detach().double()[:, None]], dim=1)
    if alpha.shape[0] != omega.shape[1]:
        return None
    return alpha.t() @ omega.t()


def _held_out_bottleneck_cosine(
    extension: torch.Tensor,
    held_out_n: torch.Tensor,
) -> float:
    """``cos(alpha omega, N_held_out)`` in the Frobenius geometry.

    The raw inner product is the bottleneck in the held-out fold's units, so
    it is not comparable between folds, between layers or between datasets.
    Dividing by both norms leaves a number in [-1, 1] whose SIGN says whether
    the fitted direction still points at what the held-out data wants, and
    whose scale is the problem's own. Magnitude cancels -- which is the
    property that lets one rule serve smooth_sin and MNIST without retuning.
    """
    if extension.shape != held_out_n.shape:
        return float("nan")
    left = float(torch.linalg.norm(extension))
    right = float(torch.linalg.norm(held_out_n))
    if not (left > 0.0 and right > 0.0):
        return 0.0
    value = float(torch.sum(extension * held_out_n)) / (left * right)
    return value if math.isfinite(value) else float("nan")


def crossfold_bottleneck_significance(
    model: GrowingMLP,
    layer_index: int,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
    config: FGDApproxConfig,
) -> tuple[float, list[float]]:
    """Is this layer's bottleneck REAL, or is it a direction fitted to noise?

    The stopping question, answered without a magnitude and without an
    asymptotic regime. For each of ``K`` folds: fit the extension on the other
    ``K-1``, freeze ``(alpha, omega)``, and score that frozen pair against the
    held-out fold's own desired-update matrix ``N``. A direction the data
    really asks for scores positive on every fold; a direction fitted to
    sampling noise scatters around zero.

    Returns ``(t, b)`` with ``t = mean(b) / (std(b) / sqrt(K))``. ``inf``
    means NOT MEASURED (too few rows, or a structure the identity behind
    :func:`_fitted_extension_matrix` is not derived for) and must never block
    growth -- a stopping rule that fires because it could not measure is a
    stopping rule that stops for the wrong reason.
    """
    folds = int(getattr(config, "growth_bottleneck_crossfold_folds", 0) or 0)
    rows = int(inputs.shape[0])
    if folds < 2 or rows < folds * 2:
        return float("inf"), []

    # The direction is what is being tested, so both of its factors have to
    # exist. alpha_zero / omega_zero say how the neuron is INITIALISED once
    # bought -- with either set there is no fitted direction to carry to a
    # held-out fold at all. compute_delta is forced for the same reason: with
    # use_projection the extension is fitted against tensor_n, which does not
    # exist until the optimal delta does.
    update_kwargs = dict(tiny_optimal_update_kwargs(config, compute_delta=True))
    update_kwargs["alpha_zero"] = False
    update_kwargs["omega_zero"] = False

    masks = _stride_fold_masks(rows, folds)
    values: list[float] = []

    for mask in masks:
        held_out = mask
        fitted = ~mask
        if int(fitted.sum()) == 0 or int(held_out.sum()) == 0:
            continue

        extension: torch.Tensor | None = None
        try:
            model.set_growing_layers(index=layer_index)
            compute_statistics(
                model,
                _fold_loader(inputs[fitted], targets[fitted], batch_size),
                loss_function=batch_functional_mse_loss,
                device=device,
            )
            model.compute_optimal_updates(**update_kwargs)
            model.reset_computation()
            model.dummy_select_update()
            extension = _fitted_extension_matrix(model.currently_updated_layer)
        finally:
            _cleanup_tiny_update(model)

        if extension is None:
            return float("inf"), []

        held_out_n: torch.Tensor | None = None
        try:
            model.set_growing_layers(index=layer_index)
            compute_statistics(
                model,
                _fold_loader(inputs[held_out], targets[held_out], batch_size),
                loss_function=batch_functional_mse_loss,
                device=device,
            )
            model.compute_optimal_updates(**update_kwargs)
            model.dummy_select_update()
            # tensor_n is a view over the accumulated statistics, not a stored
            # tensor, so it has to be read BEFORE reset_computation clears
            # them -- the fit above can reset first only because
            # eigenvalues_extension survives as a plain attribute.
            tensor_n = getattr(model.currently_updated_layer, "tensor_n", None)
            if tensor_n is not None:
                held_out_n = tensor_n.detach().double().clone()
        finally:
            _cleanup_tiny_update(model)

        if held_out_n is None:
            return float("inf"), []
        values.append(_held_out_bottleneck_cosine(extension, held_out_n))

    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        return float("inf"), values

    mean = statistics.fmean(values)
    spread = statistics.stdev(values)
    if spread <= 0.0:
        # Every fold agreed exactly. Positive is a unanimous yes; zero or
        # below is a unanimous no, and neither is a failure to measure.
        return (float("inf") if mean > 0.0 else float("-inf")), values
    return mean / (spread / math.sqrt(len(values))), values


def compute_expressivity_bottlenecks(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: FGDApproxConfig,
    progress: Callable[[str], None] | None = None,
) -> list[float]:
    """Per-layer EXPRESSIVITY BOTTLENECK: what the layer's width cannot reach.

    TINY's quantity, and only the part of it that is about expressivity. GroMo
    computes ``first_order_improvement = parameter_update_decrease +
    activation_gradient * sum(eigenvalues_extension ** 2)``, and its
    ``select_best_update`` takes the argmax of that SUM. The first term is the
    first-order gain from re-fitting the weights the layer ALREADY has -- a
    statement about training, not about width -- so including it answers
    "where would a step help most", which is not the question growth asks.

    The second term is the bottleneck itself: ``eigenvalues_extension`` are the
    singular values of the extension problem, i.e. the mass of the desired
    functional change that lies OUTSIDE what the current width can express and
    that new neurons would capture. Weighted by ``activation_gradient``, it is
    the first-order loss decrease attributable to ADDING capacity there.

    Returned per growable layer, in layer order, so the caller can put its
    neuron where the number is largest. ``0.0`` where GroMo declines to build
    an extension (no bottleneck it can name), which the caller must read as
    "nothing to gain here", not as a missing measurement.
    """
    bottlenecks: list[float] = []
    update_kwargs = tiny_optimal_update_kwargs(
        config,
        compute_delta=config.rel_error_compute_delta,
    )
    # The FULL spectrum, for the significance test only. The shipped kwargs
    # truncate to maximum_added_neurons (1 here) and to statistical_threshold,
    # both of which discard exactly the bulk that Marchenko-Pastur needs in
    # order to estimate the noise level from the data itself. Asking for the
    # untruncated SVD costs nothing extra -- it is the same decomposition,
    # simply not cut -- and the neuron actually added is still one.
    significance = bool(
        getattr(config, "growth_bottleneck_significance", False)
    )
    if significance:
        update_kwargs = dict(update_kwargs)
        update_kwargs["maximum_added_neurons"] = None
        update_kwargs["statistical_threshold"] = 0.0
    sample_count = 0
    if significance:
        for _batch in train_loader:
            _inputs = _batch[0] if isinstance(_batch, (tuple, list)) else _batch
            sample_count += int(_inputs.shape[0])

    for layer_index in range(len(model._growable_layers)):
        model.set_growing_layers(index=layer_index)
        try:
            compute_statistics(
                model,
                train_loader,
                loss_function=batch_functional_mse_loss,
                device=device,
            )
            model.compute_optimal_updates(**update_kwargs)
            model.reset_computation()
            model.dummy_select_update()
            layer = model.currently_updated_layer
            eigenvalues = getattr(layer, "eigenvalues_extension", None)
            if eigenvalues is None or eigenvalues.numel() == 0:
                bottlenecks.append(0.0)
                continue
            if significance:
                # NOT SIGNIFICANT -> NOT A BOTTLENECK. Zero here is a
                # measurement ("this layer's extension is indistinguishable
                # from noise"), which is what lets the caller stop without a
                # parameter cap: a cap is a number nobody knows in advance,
                # while this is a statement the data makes about itself.
                kept = marchenko_pastur_significant_count(
                    eigenvalues, sample_count
                )
                if kept == 0:
                    bottlenecks.append(0.0)
                    continue
                eigenvalues = eigenvalues[:kept]
            gradient = float(layer.activation_gradient)
            value = gradient * float(torch.sum(eigenvalues.double() ** 2))
            bottlenecks.append(value if math.isfinite(value) else 0.0)
        finally:
            _cleanup_tiny_update(model)

    # SECOND PASS, and separate on purpose: the loop above is left exactly as
    # it was so that with the criterion off nothing above this line can differ.
    # A layer already at zero is skipped -- it costs nothing and there is
    # nothing to test.
    #
    # A value is only ever turned INTO a zero, never moved, so the RANKING
    # among the layers that pass is the in-sample one, untouched. But zeroing
    # is per layer, and zeroing the argmax hands the turn to the runner-up --
    # MEASURED on seed 1, which refused no growth at all and still landed on
    # 4-9-20 instead of 7-11-12. So this gates ELIGIBILITY, and growth stops
    # only when every layer fails at once; it is not a whether-only switch.
    folds = int(getattr(config, "growth_bottleneck_crossfold_folds", 0) or 0)
    if folds >= 2 and any(value > 0.0 for value in bottlenecks):
        probe_x, probe_y, probe_batch = _materialize_loader(train_loader, device)
        with timed("growth_crossfold_seconds"):
            for layer_index, value in enumerate(bottlenecks):
                if value <= 0.0:
                    continue
                increment("growth_crossfold_layers_tested")
                statistic, samples = crossfold_bottleneck_significance(
                    model=model,
                    layer_index=layer_index,
                    inputs=probe_x,
                    targets=probe_y,
                    batch_size=probe_batch,
                    device=device,
                    config=config,
                )
                if progress is not None:
                    progress(
                        f"[CROSSFOLD] L{layer_index} t={statistic:.4f} b="
                        + " ".join(f"{sample:+.4f}" for sample in samples)
                    )
                bar = float(getattr(config, "growth_bottleneck_crossfold_t", 1.0))
                if not (statistic > bar):
                    increment("growth_crossfold_layers_rejected")
                    bottlenecks[layer_index] = 0.0

    return bottlenecks


def compute_tiny_layer_relative_errors(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: FGDApproxConfig,
) -> list[FGDLayerRelError]:
    """Compute TINY optimal update and functional RelErr for every growable layer."""
    layer_errors: list[FGDLayerRelError] = []
    update_kwargs = tiny_optimal_update_kwargs(
        config,
        compute_delta=config.rel_error_compute_delta,
    )

    for layer_index in range(len(model._growable_layers)):
        model.set_growing_layers(index=layer_index)
        try:
            compute_statistics(
                model,
                train_loader,
                loss_function=batch_functional_mse_loss,
                device=device,
            )
            model.compute_optimal_updates(**update_kwargs)
            model.reset_computation()
            model.dummy_select_update()
            layer_errors.append(
                _layer_functional_error(
                    model=model,
                    train_loader=train_loader,
                    layer_index=layer_index,
                    device=device,
                    config=config,
                )
            )
        finally:
            _cleanup_tiny_update(model)

    return layer_errors


def apply_fgd_parameter_update(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    selected_layer: FGDLayerRelError,
    device: torch.device,
    learning_rate: float,
    config: FGDApproxConfig,
) -> None:
    """Apply one FGD parameter update from GroMo's optimal delta approximation."""
    model.set_growing_layers(index=selected_layer.layer_index)
    try:
        compute_statistics(
            model,
            train_loader,
            loss_function=batch_functional_mse_loss,
            device=device,
        )
        model.compute_optimal_updates(
            **tiny_optimal_update_kwargs(config, compute_delta=True)
        )
        model.reset_computation()
        model.dummy_select_update()
        model.currently_updated_layer.apply_change(
            apply_delta=True,
            apply_extension=False,
            optimal_delta_scaling=learning_rate,
        )
    finally:
        _cleanup_tiny_update(model)


def select_fgd_growth_layer(
    layer_errors: list[FGDLayerRelError],
    config: FGDApproxConfig,
) -> FGDLayerRelError | None:
    if not layer_errors:
        return None
    if config.layer_selection == "min_rel_error":
        return min(layer_errors, key=lambda item: item.relative_error)
    raise ValueError(
        f"Unsupported fgd_approx.layer_selection '{config.layer_selection}'. "
        "Use one of: tiny_best, min_rel_error."
    )


def _output_error_from_layer_error(
    layer_error: FGDLayerRelError,
) -> FGDOutputRelError:
    return FGDOutputRelError(
        relative_error=layer_error.relative_error,
        approximation_norm=layer_error.approximation_norm,
        target_norm=layer_error.target_norm,
        directional_cosine=layer_error.directional_cosine,
    )


def train_one_epoch_gromo_layer_proxy(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    learning_rate: float,
    accuracy_tolerance: float,
    config: FGDApproxConfig,
    classification: bool = False,
) -> FGDApproxEpochResult:
    """Train with GroMo's algebraic per-layer optimal update as the FGD proxy."""
    model.train()
    layer_errors = compute_tiny_layer_relative_errors(
        model=model,
        train_loader=train_loader,
        device=device,
        config=config,
    )
    selected_layer = (
        min(layer_errors, key=lambda item: item.relative_error)
        if layer_errors
        else None
    )

    if selected_layer is not None:
        apply_fgd_parameter_update(
            model=model,
            train_loader=train_loader,
            selected_layer=selected_layer,
            device=device,
            learning_rate=learning_rate,
            config=config,
        )

    train_metrics: RegressionMetrics = evaluate_regression_metrics(
        model,
        train_loader,
        loss_function,
        device=device,
        accuracy_tolerance=accuracy_tolerance,
        classification=classification,
    )
    test_metrics: RegressionMetrics = evaluate_regression_metrics(
        model,
        test_loader,
        loss_function,
        device=device,
        accuracy_tolerance=accuracy_tolerance,
        classification=classification,
    )

    output_error = (
        _output_error_from_layer_error(selected_layer)
        if selected_layer is not None
        else None
    )
    return FGDApproxEpochResult(
        train_loss=train_metrics.loss,
        train_accuracy=train_metrics.accuracy,
        test_loss=test_metrics.loss,
        test_accuracy=test_metrics.accuracy,
        learning_rate=learning_rate,
        next_learning_rate=learning_rate,
        learning_rate_upper_bound=None,
        learning_rate_interval_valid=None,
        learning_rate_clipped_batches=0,
        skipped_batches=0,
        relative_error_condition_valid=None,
        loss_descent_valid=None,
        loss_non_descent_batches=0,
        gradient_sq_norm=None,
        theory_descent_coefficient=None,
        min_positive_learning_rate=learning_rate if learning_rate > 0.0 else None,
        relative_error=selected_layer.relative_error
        if selected_layer is not None
        else None,
        selected_layer_index=selected_layer.layer_index
        if selected_layer is not None
        else None,
        layer_relative_errors=layer_errors,
        output_relative_error=output_error,
        sensor_valid=True,
        sensor_invalid_batches=0,
    )


def select_tiny_growth_layer_index(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: FGDApproxConfig,
) -> int | None:
    """Read-only TINY scoring pass used to choose a growth layer."""
    if not getattr(model, "_growable_layers", None):
        return None

    model.set_growing_layers(scheduling_method="all")
    try:
        compute_statistics(
            model,
            train_loader,
            loss_function=batch_functional_mse_loss,
            device=device,
        )
        model.compute_optimal_updates(
            **tiny_optimal_update_kwargs(
                config,
                compute_delta=config.growth_compute_delta,
            )
        )
        scores = {
            int(index): float(info["update_value"].detach().cpu())
            for index, info in model.update_information().items()
        }
        if not scores:
            return None
        return max(scores, key=scores.get)
    finally:
        _cleanup_tiny_update(model)


def compute_tangent_projection_error(
    model: GrowingMLP,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: FGDApproxConfig,
) -> FGDOutputRelError:
    """Measure the global tangent-projection certificate without updating."""
    certificate = evaluate_fgd_validation_certificate(
        model=model,
        data_loader=data_loader,
        device=device,
        config=config,
        learning_rate=None,
    )
    if certificate.output_relative_error is None:
        raise RuntimeError("Invalid FGD tangent projection sensor measurement.")
    return certificate.output_relative_error


def _count_all_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def select_certifying_growth_layer_index(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: FGDApproxConfig,
    line_search_config: ScalingLineSearchConfig,
    probe: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> int | None:
    """Trial-grow with train data and choose by validation certificate."""
    if not getattr(model, "_growable_layers", None):
        return None

    if probe is None:
        # One fixed probe shared by every layer trial, so their certificates
        # are directly comparable.
        probe = build_projection_probe(validation_loader, config.probe_batches)
    _clear_inaccessible_tensor_caches(model)
    _cleanup_tiny_update(model)
    base_parameter_count = _count_all_parameters(model)
    trials: list[dict[str, float | int | bool]] = []
    optimal_update_kwargs = tiny_optimal_update_kwargs(
        config,
        compute_delta=config.growth_compute_delta,
    )

    for layer_index in range(len(model._growable_layers)):
        trial_model = copy.deepcopy(model)
        growth_result = grow_layer(
            model=trial_model,
            train_loader=train_loader,
            layer_index=layer_index,
            device=device,
            line_search_config=line_search_config,
            optimal_update_kwargs=optimal_update_kwargs,
            progress=None,
        )
        effective_growth = growth_result.best_scaling_factor > max(
            line_search_config.tolerance,
            config.eps,
        )
        certificate = evaluate_fgd_validation_certificate(
            model=trial_model,
            data_loader=validation_loader,
            device=device,
            config=config,
            learning_rate=None,
            probe=probe,
        )
        relative_error = certificate.relative_error
        certified = (
            effective_growth
            and certificate.sensor_valid
            and relative_error is not None
            and relative_error <= config.rel_error_threshold
        )
        trials.append(
            {
                "layer_index": layer_index,
                "relative_error": relative_error
                if relative_error is not None
                else float("inf"),
                "added_parameters": _count_all_parameters(trial_model)
                - base_parameter_count,
                "certified": certified,
                "sensor_valid": certificate.sensor_valid and effective_growth,
            }
        )
        del trial_model

    if not trials:
        return None

    certifying_trials = [trial for trial in trials if bool(trial["certified"])]
    if certifying_trials:
        # Explicit certified policy: fewest added parameters, then lowest
        # relative error, then layer index (deterministic).
        chosen = min(
            certifying_trials,
            key=lambda trial: (
                int(trial["added_parameters"]),
                float(trial["relative_error"]),
                int(trial["layer_index"]),
            ),
        )
    else:
        valid_trials = [trial for trial in trials if bool(trial["sensor_valid"])]
        if not valid_trials:
            return None
        # No candidate certifies: the lowest post-growth relative error
        # wins; parameter count is only a tie-breaker.
        chosen = min(
            valid_trials,
            key=lambda trial: (
                float(trial["relative_error"]),
                int(trial["added_parameters"]),
                int(trial["layer_index"]),
            ),
        )
    return int(chosen["layer_index"])


def should_trigger_fgd_growth(
    relative_error: float,
    epoch: int,
    last_growth_epoch: int | None,
    config: FGDApproxConfig,
) -> bool:
    """Return whether an FGD certificate requests immediate GroMo growth.

    FGD certificate failures are not delayed by epoch or dwell constraints.
    The pipeline-level ``growth_schedule.enabled`` flag remains the global
    switch that can disable architecture changes.
    """
    del epoch, last_growth_epoch
    return relative_error >= config.rel_error_threshold


def certificate_from_projection_stats(
    *,
    stats: _FunctionalStepStats,
    learning_rate: float | None,
    config: FGDApproxConfig,
    projection_sensor: bool = True,
) -> FGDValidationCertificate:
    """Build the FGD certificate from ONE joint probe measurement.

    ``stats`` describes a single approximation g of the functional gradient
    r over the whole probe (one shared direction). ``projection_sensor``
    additionally enforces the exact-projector invariants (<g, r> >= 0 and
    |g| <= |r|), which hold for tangent projections but not for general
    Hilbert secants.
    """
    values = (
        stats.dot_product,
        stats.approximation_sq_norm,
        stats.target_sq_norm,
    )
    # Name the quantity that failed. "sensor invalid" on its own reads as a
    # numerical defect in the certificate machinery, which sent this
    # investigation the wrong way for a while: with projection_sensor off the
    # ONLY test here is finiteness, so a failure means one of these three
    # overflowed. For sum-MSE target_sq_norm = 4*sum (f - y)^2, so what
    # overflows is the MODEL's output on unseen data, not our arithmetic --
    # the Jacobian is exact either way. It is a symptom of the fit, and the
    # right response is emphatically not to add capacity.
    non_finite = tuple(
        name
        for name, value in zip(
            ("dot_product", "approximation_sq_norm", "target_sq_norm"), values
        )
        if not math.isfinite(value)
    )
    finite = not non_finite
    sensor_valid = finite and (
        not projection_sensor
        or _projection_sensor_valid(
            dot_product=stats.dot_product,
            approximation_sq_norm=stats.approximation_sq_norm,
            target_sq_norm=stats.target_sq_norm,
            eps=config.eps,
        )
    )
    if not sensor_valid:
        return FGDValidationCertificate(
            learning_rate_upper_bound=None,
            max_valid_learning_rate=None,
            learning_rate_interval_valid=None,
            skipped_batches=0,
            relative_error_condition_valid=None,
            gradient_sq_norm=None,
            theory_descent_coefficient=None,
            relative_error=None,
            output_relative_error=None,
            sensor_valid=False,
            sensor_invalid_batches=1,
            non_finite_quantities=non_finite,
        )

    output_error = stats.output_error
    relative_error = output_error.relative_error
    relative_error_condition_valid = relative_error < min(
        config.rel_error_threshold,
        0.5,
    )

    learning_rate_upper_bound: float | None = None
    max_valid_learning_rate: float | None = None
    learning_rate_interval_valid: bool | None = None
    theory_descent_coefficient: float | None = None
    skipped_batches = 0
    if config.learning_rate_policy == "theory_interval":
        learning_rate_upper_bound = theoretical_learning_rate_upper_bound(
            relative_error,
            config,
        )
        if learning_rate_upper_bound is None:
            learning_rate_interval_valid = False
            skipped_batches = 1
        else:
            safe_upper_bound = config.theory_lr_safety * learning_rate_upper_bound
            interval_ok = safe_upper_bound > config.theory_lr_min + config.eps
            if interval_ok:
                max_valid_learning_rate = safe_upper_bound
            if learning_rate is not None and learning_rate > config.eps:
                if config.local_acceptance_conditions:
                    # Strict interval: theory_lr_min < eta < eta_bar, with
                    # eps as the only numerical tolerance.
                    upper_bound_ok = learning_rate < safe_upper_bound + config.eps
                else:
                    upper_bound_ok = learning_rate <= safe_upper_bound + config.eps
                interval_ok = (
                    interval_ok
                    and learning_rate > config.theory_lr_min
                    and upper_bound_ok
                )
            learning_rate_interval_valid = interval_ok
            if not interval_ok:
                skipped_batches = 1
            elif learning_rate is not None and learning_rate > config.eps:
                theory_descent_coefficient = theoretical_descent_coefficient(
                    relative_error,
                    learning_rate,
                    config,
                )

    return FGDValidationCertificate(
        learning_rate_upper_bound=learning_rate_upper_bound,
        max_valid_learning_rate=max_valid_learning_rate,
        learning_rate_interval_valid=learning_rate_interval_valid,
        skipped_batches=skipped_batches,
        relative_error_condition_valid=relative_error_condition_valid,
        gradient_sq_norm=output_error.target_norm**2,
        theory_descent_coefficient=theory_descent_coefficient,
        relative_error=relative_error,
        output_relative_error=output_error,
        sensor_valid=True,
        sensor_invalid_batches=0,
    )


def measure_direction_projection(
    model: GrowingMLP,
    parameter_updates: tuple[torch.Tensor, ...],
    x: torch.Tensor,
    y: torch.Tensor,
    config: FGDApproxConfig,
) -> _FunctionalStepStats:
    """Measure g = J u against r = grad L on a probe for one SHARED direction.

    This is the certificate measurement for an outer step: ``u`` is the
    direction that will actually move the model, evaluated at the CURRENT
    parameters through a Jacobian-vector product (the Jacobian is never
    materialized).
    """
    # The tangent direction u comes from a projection solve that may leave its
    # tensors attached to an autograd graph or a functorch level. jvp's
    # make_dual then fails with a functorch level-escape (seen with batch-norm
    # models). Detach the tangents to a clean, graph-free state -- they are a
    # fixed direction here, not something to differentiate through.
    parameter_updates = tuple(update.detach() for update in parameter_updates)
    named_parameters = _trainable_named_parameters(model)
    if not named_parameters:
        raise RuntimeError("FGD direction measurement requires trainable parameters.")
    parameter_names = tuple(named_parameters.keys())
    parameters = tuple(named_parameters.values())
    buffers = OrderedDict(model.named_buffers())

    # Measure in eval. The certificate is a statement about the represented
    # function f, so dropout must be the identity and batch-norm must use its
    # fixed running statistics -- and there is a hard requirement too: a
    # batch-norm in TRAIN mode updates its running buffers in place during the
    # forward, which is not valid inside forward-mode AD (jvp) and raises a
    # functorch level-escape. eval() removes both problems; it is a no-op on
    # the plain MLP.
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(x)
        target = functional_gradient(output, y, config.functional_loss).reshape(-1)

        def call_with_parameters(
            parameter_values: tuple[torch.Tensor, ...],
        ) -> torch.Tensor:
            state = OrderedDict(zip(parameter_names, parameter_values))
            state.update(buffers)
            return functional_call(model, state, (x,)).reshape(-1)

        _, approximation = jvp(
            call_with_parameters,
            (parameters,),
            (tuple(parameter_updates),),
        )
    finally:
        model.train(was_training)
        _clear_inaccessible_tensor_caches(model)
    return _output_relative_error_from_tensors(
        approximation=approximation.detach(),
        target=target,
        eps=config.eps,
    )


def evaluate_fgd_validation_certificate(
    model: GrowingMLP,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: FGDApproxConfig,
    learning_rate: float | None,
    probe: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> FGDValidationCertificate:
    """Evaluate FGD certificate conditions on a fixed held-out probe.

    The certificate solves ONE joint tangent projection over the probe (the
    concatenation of ``config.probe_batches`` mini-batches of
    ``data_loader``, unless an explicit ``probe`` is given): a single shared
    parameter direction u* = argmin_u |J_probe u - r_probe|^2, certified
    through g_probe = J_probe u*. Independent per-batch projections combined
    through their aggregated norms are no longer computed anywhere.
    """
    if config.rel_error_mode != "tangent_projection":
        raise ValueError(
            f"Unsupported fgd_approx.rel_error_mode '{config.rel_error_mode}'. "
            "Use tangent_projection."
        )
    if config.projection_solver == "gromo_layer":
        raise ValueError(
            "Validation certificates require projection_solver to be cg, exact, "
            "exact_svd, or exact_kernel_eigh."
        )

    model.eval()
    if probe is None:
        probe = build_projection_probe(data_loader, config.probe_batches)
    x, y = probe
    x = x.to(device)
    y = y.to(device)
    projection_step = _compute_tangent_projection_step(
        model=model,
        x=x,
        y=y,
        config=config,
    )
    stats = _FunctionalStepStats(
        output_error=projection_step.output_error,
        dot_product=projection_step.dot_product,
        approximation_sq_norm=projection_step.approximation_sq_norm,
        target_sq_norm=projection_step.target_sq_norm,
    )
    return certificate_from_projection_stats(
        stats=stats,
        learning_rate=learning_rate,
        config=config,
    )


def evaluate_secant_validation_certificate(
    base_model: GrowingMLP,
    candidate_model: GrowingMLP,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: FGDApproxConfig,
    learning_rate: float,
    probe: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> FGDValidationCertificate:
    """Certify a finite functional secant in the output Hilbert space.

    The realized displacement Delta = F(base) - F(candidate) is one shared
    direction by construction, so the certificate measures it jointly on the
    same fixed probe used by the tangent certificate. Secants are not
    projections, so the exact-projector sensor invariants are not enforced
    (only finiteness).
    """
    if learning_rate <= config.eps:
        raise ValueError("A secant FGD learning rate must be positive.")

    base_model.eval()
    candidate_model.eval()
    if probe is None:
        probe = build_projection_probe(data_loader, config.probe_batches)
    x, y = probe
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        base_output = base_model(x)
        candidate_output = candidate_model(x)
    target = functional_gradient(base_output, y, config.functional_loss)
    approximation = (base_output - candidate_output) / learning_rate
    if not (torch.isfinite(target).all() and torch.isfinite(approximation).all()):
        return FGDValidationCertificate(
            learning_rate_upper_bound=None,
            max_valid_learning_rate=None,
            learning_rate_interval_valid=None,
            skipped_batches=0,
            relative_error_condition_valid=None,
            gradient_sq_norm=None,
            theory_descent_coefficient=None,
            relative_error=None,
            output_relative_error=None,
            sensor_valid=False,
            sensor_invalid_batches=1,
        )

    stats = _output_relative_error_from_tensors(
        approximation,
        target,
        config.eps,
    )
    return certificate_from_projection_stats(
        stats=stats,
        learning_rate=learning_rate,
        config=config,
        projection_sensor=False,
    )


def train_one_epoch_fgd_approx(
    model: GrowingMLP,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    learning_rate: float,
    accuracy_tolerance: float,
    config: FGDApproxConfig,
    projection_group_size: int = 1,
    classification: bool = False,
    evaluate_test: bool = True,
) -> FGDApproxEpochResult:
    """Train one FGD epoch with the configured functional-gradient proxy.

    ``exact`` computes the global tangent projection with the full Jacobian.
    ``cg`` and ``gromo_layer`` remain available as cheaper proxy variants, but
    they are not the default training path.
    """
    if config.rel_error_mode != "tangent_projection":
        raise ValueError(
            f"Unsupported fgd_approx.rel_error_mode '{config.rel_error_mode}'. "
            "Use tangent_projection."
        )
    if config.learning_rate_policy not in {"scheduler", "theory_interval"}:
        raise ValueError(
            f"Unsupported fgd_approx.learning_rate_policy "
            f"'{config.learning_rate_policy}'. Use one of: scheduler, "
            "theory_interval."
        )

    if config.projection_solver == "gromo_layer":
        return train_one_epoch_gromo_layer_proxy(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_function=loss_function,
            device=device,
            learning_rate=learning_rate,
            accuracy_tolerance=accuracy_tolerance,
            config=config,
            classification=classification,
        )

    model.train()
    learning_rate_sum = 0.0
    batch_count = 0
    skipped_batches = 0
    min_positive_learning_rate: float | None = None
    sensor_invalid_batches = 0

    if config.learning_rate_policy == "theory_interval":
        if config.theory_lr_safety <= 0.0 or config.theory_lr_safety > 1.0:
            raise ValueError("fgd_approx.theory_lr_safety must be in (0, 1].")
        if config.theory_lr_min < 0.0:
            raise ValueError("fgd_approx.theory_lr_min must be non-negative.")
    if config.theory_mu <= 0.0:
        raise ValueError("fgd_approx.theory_mu must be positive.")

    for x, y in _grouped_batches(train_loader, projection_group_size):
        x = x.to(device)
        y = y.to(device)

        projection_step = _compute_tangent_projection_step(
            model=model,
            x=x,
            y=y,
            config=config,
        )
        if not _projection_step_sensor_valid(projection_step, config):
            sensor_invalid_batches += 1
            skipped_batches += 1
            batch_count += 1
            continue

        applied_step = _apply_tangent_projection_step(
            model=model,
            x=x,
            y=y,
            step=projection_step,
            learning_rate=learning_rate,
            config=config,
        )
        learning_rate_sum += applied_step.learning_rate_used
        batch_count += 1
        if applied_step.learning_rate_used <= config.eps and learning_rate > config.eps:
            skipped_batches += 1
        if applied_step.learning_rate_used > config.eps:
            min_positive_learning_rate = (
                applied_step.learning_rate_used
                if min_positive_learning_rate is None
                else min(min_positive_learning_rate, applied_step.learning_rate_used)
            )

    selected_layer_index = select_tiny_growth_layer_index(
        model=model,
        train_loader=train_loader,
        device=device,
        config=config,
    )

    train_metrics: RegressionMetrics = evaluate_regression_metrics(
        model,
        train_loader,
        loss_function,
        device=device,
        accuracy_tolerance=accuracy_tolerance,
        classification=classification,
    )
    test_metrics = (
        evaluate_regression_metrics(
            model,
            test_loader,
            loss_function,
            device=device,
            accuracy_tolerance=accuracy_tolerance,
            classification=classification,
        )
        if evaluate_test
        else RegressionMetrics(loss=float("nan"), accuracy=float("nan"))
    )

    return FGDApproxEpochResult(
        train_loss=train_metrics.loss,
        train_accuracy=train_metrics.accuracy,
        test_loss=test_metrics.loss,
        test_accuracy=test_metrics.accuracy,
        learning_rate=learning_rate_sum / max(1, batch_count),
        next_learning_rate=None,
        learning_rate_upper_bound=None,
        learning_rate_interval_valid=None,
        learning_rate_clipped_batches=0,
        skipped_batches=skipped_batches,
        relative_error_condition_valid=None,
        loss_descent_valid=None,
        loss_non_descent_batches=0,
        gradient_sq_norm=None,
        theory_descent_coefficient=None,
        min_positive_learning_rate=min_positive_learning_rate,
        relative_error=None,
        selected_layer_index=selected_layer_index,
        layer_relative_errors=[],
        output_relative_error=None,
        sensor_valid=sensor_invalid_batches == 0,
        sensor_invalid_batches=sensor_invalid_batches,
    )
