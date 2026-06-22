"""Train/grow loops for comparing post-growth training dynamics.

The harness supports three methods:

- ``baseline_mlp``: ordinary SGD training, no growth.
- ``gromo_tiny``: the original scheduled GroMo/TINY growth loop.
- ``functional_triggered_tiny``: certified empirical functional-gradient
  training on the current DAG; a failed relative-error certificate triggers a
  normal GroMo/TINY growth step.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch
import torch.utils.data
from torchmetrics.classification import MulticlassAccuracy

from gromo.containers.growing_mlp import GrowingMLP
from gromo.utils.training_utils import evaluate_model, gradient_descent

from .data import get_dataloaders
from .functional_descent import (
    FunctionalEpochInfo,
    certified_functional_step,
    exact_functional_step,
    functional_descent_epoch,
)
from .growth import grow_step, select_tiny_growth_layer


def _build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg.get("optimizer", "sgd").lower()
    lr = cfg["lr"]
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.get("momentum", 0.9))
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 0.0)
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 0.01)
        )
    raise ValueError(f"Unknown optimizer: {name!r}")


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_model(cfg: dict, meta: dict, device: torch.device) -> GrowingMLP:
    return GrowingMLP(
        in_features=meta["in_features"],
        out_features=meta["out_features"],
        hidden_size=cfg["hidden_size"],
        number_hidden_layers=cfg["number_hidden_layers"],
        device=device,
    )


def run_experiment(cfg: dict) -> dict:
    """Run one experiment method, returning the recorded history."""
    torch.manual_seed(cfg["seed"])
    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    method = cfg.get("method", "gromo_tiny")
    print(f"Using device: {device}")
    print(f"Method: {method}")

    train_loader, test_loader, meta = get_dataloaders(cfg)
    num_classes = meta["out_features"]
    criterion = torch.nn.CrossEntropyLoss()  # reduction="mean"

    def accuracy() -> MulticlassAccuracy:
        return MulticlassAccuracy(num_classes=num_classes)

    model = _build_model(cfg, meta, device)
    print(model)

    state = _ExperimentState(
        cfg=cfg,
        method=method,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        accuracy_factory=accuracy,
        device=device,
    )
    if cfg.get("record_trajectory"):
        n_probe = int(cfg.get("trajectory_probe_size", 256))
        probe_x, probe_y = next(iter(train_loader))
        # Gather a fixed probe set of the requested size from the train loader.
        xs, ys = [probe_x], [probe_y]
        gathered = probe_x.shape[0]
        for bx, by in train_loader:
            if gathered >= n_probe:
                break
            xs.append(bx)
            ys.append(by)
            gathered += bx.shape[0]
        state.probe_x = torch.cat(xs)[:n_probe].to(device)
        state.probe_y = torch.cat(ys)[:n_probe].to(device)
    state.log("init")

    t0 = time.time()
    if method == "baseline_mlp":
        _run_baseline_sgd(state)
    elif method == "gromo_tiny":
        _run_scheduled_gromo_tiny(state)
    elif method == "functional_triggered_tiny":
        _run_functional_triggered_tiny(state)
    elif method == "functional_certified_tiny":
        _run_functional_certified_tiny(state)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Expected baseline_mlp, gromo_tiny, "
            "functional_triggered_tiny, or functional_certified_tiny."
        )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    result = {
        "config": cfg,
        "method": method,
        "history": state.history,
        "growth_lines": state.growth_lines,
        "growth_info": state.growth_info,
        "functional_events": state.functional_events,
        "elapsed_sec": elapsed,
        "final_params": _count_params(model),
    }

    out_dir = Path(cfg.get("out_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg.get("run_name", "run")
    with open(out_dir / f"{run_name}_history.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved history to {out_dir / f'{run_name}_history.json'}")

    if cfg.get("record_trajectory") and state.trajectory_logits:
        import numpy as np

        traj_path = out_dir / f"{run_name}_trajectory.npz"
        np.savez_compressed(
            traj_path,
            logits=np.stack(state.trajectory_logits).astype("float32"),
            probe_labels=state.probe_y.detach().cpu().numpy(),
            eval_idx=np.array([m["eval_idx"] for m in state.trajectory_meta]),
            epoch=np.array([m["epoch"] for m in state.trajectory_meta]),
            params=np.array([m["params"] for m in state.trajectory_meta]),
            phase=np.array([m["phase"] for m in state.trajectory_meta]),
            method=np.array(method),
        )
        result["trajectory_path"] = str(traj_path)
        print(f"Saved function-space trajectory to {traj_path}")

    return result


class _ExperimentState:
    def __init__(
        self,
        *,
        cfg: dict,
        method: str,
        model: GrowingMLP,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        accuracy_factory,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.method = method
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.accuracy_factory = accuracy_factory
        self.device = device
        self.eval_idx = 0
        self.epoch = 0
        self.growth_count = 0
        self.history: dict[str, list] = {
            "eval_idx": [],
            "epoch": [],
            "phase": [],
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "params": [],
            "functional_relative_error": [],
            "functional_certified_batches": [],
            "functional_batches": [],
            "functional_bottleneck_fraction": [],
            "functional_descent_failures": [],
        }
        self.growth_lines: list[float] = []
        self.growth_info: list[dict] = []
        self.functional_events: list[dict] = []
        # Optional function-space trajectory recording (for the landscape viz):
        # a fixed probe set defines the empirical output (logit) space, which has
        # the same dimension across all network sizes.
        self.probe_x: torch.Tensor | None = None
        self.probe_y: torch.Tensor | None = None
        self.trajectory_logits: list = []
        self.trajectory_meta: list[dict] = []

    @torch.no_grad()
    def capture_trajectory(self, phase: str) -> None:
        if self.probe_x is None:
            return
        was_training = self.model.training
        self.model.eval()
        logits = self.model(self.probe_x).detach().cpu().numpy()
        if was_training:
            self.model.train()
        self.trajectory_logits.append(logits)
        self.trajectory_meta.append(
            {
                "eval_idx": self.eval_idx,
                "epoch": self.epoch,
                "phase": phase,
                "params": _count_params(self.model),
            }
        )

    def evaluate(self, loader: torch.utils.data.DataLoader) -> tuple[float, float]:
        loss, acc = evaluate_model(
            self.model,
            loader,
            self.criterion,
            metrics=self.accuracy_factory(),
            device=self.device,
        )
        return loss, acc

    def log(
        self,
        phase: str,
        *,
        functional_info: FunctionalEpochInfo | None = None,
    ) -> None:
        tr_loss, tr_acc = self.evaluate(self.train_loader)
        te_loss, te_acc = self.evaluate(self.test_loader)
        self.history["eval_idx"].append(self.eval_idx)
        self.history["epoch"].append(self.epoch)
        self.history["phase"].append(phase)
        self.history["train_loss"].append(tr_loss)
        self.history["train_acc"].append(tr_acc)
        self.history["test_loss"].append(te_loss)
        self.history["test_acc"].append(te_acc)
        self.history["params"].append(_count_params(self.model))

        if functional_info is None:
            rel = None
            certified = None
            batches = None
            bottleneck = None
            descent_failures = None
        else:
            rel = functional_info.max_relative_error
            certified = functional_info.certified_batches
            batches = functional_info.batches
            bottleneck = functional_info.max_bottleneck_fraction
            descent_failures = functional_info.descent_failures
        self.history["functional_relative_error"].append(rel)
        self.history["functional_certified_batches"].append(certified)
        self.history["functional_batches"].append(batches)
        self.history["functional_bottleneck_fraction"].append(bottleneck)
        self.history["functional_descent_failures"].append(descent_failures)

        extra = ""
        if functional_info is not None:
            extra = (
                f" | fgd_rel_err={functional_info.max_relative_error:.3g} "
                f"cert={functional_info.certified_batches}/{functional_info.batches}"
            )
        print(
            f"[{phase:13s}] eval={self.eval_idx:3d} epoch={self.epoch:3d} "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"test_loss={te_loss:.4f} acc={te_acc:.3f} | "
            f"params={self.history['params'][-1]}{extra}"
        )
        self.capture_trajectory(phase)
        self.eval_idx += 1


def _run_baseline_sgd(state: _ExperimentState) -> None:
    total_epochs = int(
        state.cfg.get(
            "baseline_epochs",
            state.cfg["growth_steps"] * state.cfg["epochs_per_step"]
            + state.cfg.get("final_epochs", 0),
        )
    )
    optimizer = _build_optimizer(state.model, state.cfg)
    for _ in range(total_epochs):
        state.epoch += 1
        gradient_descent(
            state.model,
            state.train_loader,
            optimizer,
            scheduler=None,
            loss_function=state.criterion,
            device=state.device,
        )
        state.log("sgd")


def _run_scheduled_gromo_tiny(state: _ExperimentState) -> None:
    n_growable = state.cfg["number_hidden_layers"]
    for step in range(state.cfg["growth_steps"]):
        optimizer = _build_optimizer(state.model, state.cfg)
        for _ in range(state.cfg["epochs_per_step"]):
            state.epoch += 1
            gradient_descent(
                state.model,
                state.train_loader,
                optimizer,
                scheduler=None,
                loss_function=state.criterion,
                device=state.device,
            )
            state.log("sgd")

        layer_to_grow = step % max(1, n_growable)
        _apply_growth(state, layer_to_grow, trigger="scheduled")

    _run_final_sgd(state)


def _run_functional_triggered_tiny(state: _ExperimentState) -> None:
    configured_max_epochs = state.cfg.get("functional_max_epochs")
    if configured_max_epochs is None:
        configured_max_epochs = (
            state.cfg["growth_steps"] * state.cfg["epochs_per_step"]
            + state.cfg.get("final_epochs", 0)
        )
    max_epochs = int(configured_max_epochs)

    configured_growth_steps = state.cfg.get("functional_growth_steps")
    if configured_growth_steps is None:
        configured_growth_steps = state.cfg["growth_steps"]
    max_growth_events = int(configured_growth_steps)
    stop_when_exhausted = bool(
        state.cfg.get("functional_stop_when_growth_budget_exhausted", False)
    )
    warmup_epochs = int(state.cfg.get("functional_warmup_epochs", 0))
    failure_patience = int(state.cfg.get("functional_failure_patience", 1))
    if warmup_epochs < 0:
        raise ValueError(f"functional_warmup_epochs must be >= 0, got {warmup_epochs}")
    if failure_patience < 1:
        raise ValueError(
            f"functional_failure_patience must be >= 1, got {failure_patience}"
        )

    warmup_optimizer = _build_optimizer(state.model, state.cfg)
    for _ in range(min(warmup_epochs, max_epochs)):
        state.epoch += 1
        gradient_descent(
            state.model,
            state.train_loader,
            warmup_optimizer,
            scheduler=None,
            loss_function=state.criterion,
            device=state.device,
        )
        state.log("warmup")

    consecutive_failures = 0

    for _ in range(max(0, max_epochs - warmup_epochs)):
        state.epoch += 1
        info = functional_descent_epoch(
            state.model,
            state.train_loader,
            state.criterion,
            device=state.device,
            lr=state.cfg.get("functional_lr", state.cfg["lr"]),
            damping=state.cfg.get("functional_damping", 1e-4),
            relative_error_tolerance=state.cfg.get(
                "functional_relative_error_tolerance", 0.5
            ),
            cg_min_iter=state.cfg.get("functional_cg_min_iter", 1),
            cg_max_iter=state.cfg.get("functional_cg_max_iter", 8),
            cg_tolerance=state.cfg.get("functional_cg_tolerance", 1e-3),
            weight_decay=state.cfg.get("functional_weight_decay", 0.0),
            grad_clip=state.cfg.get("functional_grad_clip"),
            batch_limit=state.cfg.get("functional_batch_limit"),
            progress_every=state.cfg.get("functional_progress_every"),
        )
        state.log("fgd", functional_info=info)
        state.functional_events.append(
            {
                "epoch": state.epoch,
                "failed": info.failed,
                "certified_batches": info.certified_batches,
                "batches": info.batches,
                "max_relative_error": info.max_relative_error,
                "failure_reason": info.failure.reason if info.failure else None,
                "failure_cg_iterations": info.failure.cg_iterations
                if info.failure
                else None,
                "failure_error_bound": info.failure.error_bound
                if info.failure
                else None,
                "failure_approx_norm": info.failure.approx_norm
                if info.failure
                else None,
                "consecutive_failures": consecutive_failures + 1
                if info.failed
                else 0,
                "failure_patience": failure_patience,
            }
        )
        if not info.failed:
            consecutive_failures = 0
            continue

        consecutive_failures += 1
        if consecutive_failures < failure_patience:
            print(
                "  >> functional certificate failed "
                f"({consecutive_failures}/{failure_patience}); waiting before growth"
            )
            continue

        if state.growth_count >= max_growth_events:
            print(
                "  >> functional certificate failed, but functional growth "
                "budget is exhausted"
            )
            if stop_when_exhausted:
                break
            continue

        layer_to_grow, selection_info = _choose_functional_growth_layer(state)
        _apply_growth(
            state,
            layer_to_grow,
            trigger="functional_certificate_failure",
            extra={
                **selection_info,
                "failed_relative_error": info.failure.relative_error
                if info.failure
                else info.max_relative_error,
                "certified_batches_before_growth": info.certified_batches,
                "consecutive_failures_before_growth": consecutive_failures,
            },
        )
        consecutive_failures = 0


def _functional_step_kwargs(cfg: dict) -> dict:
    """Common arguments shared by the functional step / epoch / refine loop."""
    return dict(
        lr=cfg.get("functional_lr", cfg["lr"]),
        damping=cfg.get("functional_damping", 1e-4),
        relative_error_tolerance=cfg.get("functional_relative_error_tolerance", 0.2),
        cg_min_iter=cfg.get("functional_cg_min_iter", 1),
        cg_max_iter=cfg.get("functional_cg_max_iter", 8),
        cg_tolerance=cfg.get("functional_cg_tolerance", 1e-3),
        weight_decay=cfg.get("functional_weight_decay", 0.0),
        grad_clip=cfg.get("functional_grad_clip"),
        metric=cfg.get("functional_certificate_metric", "euclidean"),
    )


def _run_functional_certified_tiny(state: _ExperimentState) -> None:
    """Theory-grounded functional descent (arXiv:2606.16926, Algorithm 1).

    Differences from ``functional_triggered_tiny`` (v1):

    1. Every batch takes a *verified-descent* step: g = P_T grad L(f) is always a
       descent direction (the empirical tangent kernel is PSD), so we always step
       and a function-space Armijo line search on the learning rate enforces the
       sufficient-descent condition of Lemma 3.5 (the second theoretical
       constraint, previously unchecked). v1 instead skipped uncertified steps,
       which stalls once the growth budget is gone.
    2. The relative-error certificate -- equivalently the expressivity bottleneck
       ``||grad L - g|| / ||g||`` -- is used only as the *growth trigger*: when too
       few batches certify, the representation is refined by growing the layer
       whose TINY update best captures the bottleneck residual. Refinement is
       incremental (grow, then keep stepping so the new neurons enter the tangent
       space); set ``functional_grow_until_certified`` to instead run Algorithm 1's
       tight inner loop on a frozen probe batch.
    """
    cfg = state.cfg
    configured_max_epochs = cfg.get("functional_max_epochs")
    if configured_max_epochs is None:
        configured_max_epochs = (
            cfg["growth_steps"] * cfg["epochs_per_step"] + cfg.get("final_epochs", 0)
        )
    max_epochs = int(configured_max_epochs)

    configured_growth_steps = cfg.get("functional_growth_steps")
    if configured_growth_steps is None:
        configured_growth_steps = cfg["growth_steps"]
    max_growth_events = int(configured_growth_steps)

    warmup_epochs = int(cfg.get("functional_warmup_epochs", 0))
    failure_patience = int(cfg.get("functional_failure_patience", 1))
    stop_when_exhausted = bool(
        cfg.get("functional_stop_when_growth_budget_exhausted", False)
    )
    if warmup_epochs < 0:
        raise ValueError(f"functional_warmup_epochs must be >= 0, got {warmup_epochs}")
    if failure_patience < 1:
        raise ValueError(
            f"functional_failure_patience must be >= 1, got {failure_patience}"
        )

    sufficient_descent_c = cfg.get("functional_sufficient_descent_c", 0.1)
    lr_backtrack = cfg.get("functional_lr_backtrack", 0.5)
    lr_min_factor = cfg.get("functional_lr_min_factor", 1e-3)
    max_refines = int(cfg.get("functional_max_refines", 4))
    certify_threshold = float(cfg.get("functional_certify_threshold", 0.5))
    grow_until_certified = bool(cfg.get("functional_grow_until_certified", False))
    # Growth hysteresis: the expressivity bottleneck is a *capacity* property, so
    # once the tangent space has certified, later certificate failures are
    # optimization noise (a flickering certificate under a high LR), not a lack of
    # capacity. Freezing growth after the first certification stops the runaway
    # over-growth that noise would otherwise trigger.
    freeze_after_certified = bool(cfg.get("functional_freeze_growth_after_certified", False))
    representation_sufficient = False
    # Certificate scope. "minibatch" (legacy): the certificate is evaluated on the
    # random training batches, which is noisy and therefore needs a fraction
    # threshold + growth hysteresis to absorb the noise. "fulldata": the
    # certificate is evaluated once per epoch on a fixed large probe, giving a
    # low-variance deterministic estimate of the expressivity bottleneck. A single
    # certificate then *is* the growth trigger -- no fraction threshold, no
    # hysteresis -- so the only remaining knob is the paper's tolerance eps, which
    # equals the maximum tolerated bottleneck fraction rho* = eps / (1 + eps).
    certificate_scope = cfg.get("functional_certificate_scope", "minibatch")
    certificate_probe_size = int(cfg.get("functional_certificate_probe_size", 512))
    # Marginal-utility stop (fulldata scope). Chasing the bottleneck to rho -> 0
    # over-capacitates and overfits, so instead of growing until the certificate
    # passes we grow only while growth keeps *paying off*: if a growth fails to
    # reduce the deterministic bottleneck by at least ``min_gain`` (relative), the
    # residual is treated as irreducible and growth is frozen. This is the
    # threshold-light stopping rule -- the default min_gain=0 means "stop as soon
    # as a growth stops helping at all".
    # Off by default: the growth objective is to *guarantee the certificate*
    # (Algorithm 1), not to minimize the bottleneck. The marginal-utility stop is
    # kept only as an ablation of a bottleneck-driven stopping rule.
    marginal_utility_stop = bool(cfg.get("functional_marginal_utility_stop", False))
    growth_min_gain = float(cfg.get("functional_growth_min_gain", 0.0))
    prev_bottleneck = float("inf")
    growth_stalled = False
    # Stopping rule (fulldata scope). The tolerance ``eps`` is a real knob under the
    # default "certificate" rule: it sets how small the bottleneck must get before
    # growth stops, and therefore the final model size. The "plateau" rule removes
    # that knob: it grows while each growth keeps reducing the *deterministic*
    # expressivity bottleneck and stops at the plateau (the elbow of the
    # bottleneck-vs-capacity curve, where the residual becomes irreducible). The
    # stopping point is then a property of the data+model, not of eps -- eps no
    # longer enters the stop decision at all (pair it with an eps-free growth
    # selection such as ``tiny_best`` for a fully eps-free pipeline).
    stop_rule = cfg.get("functional_stop_rule", "certificate")
    if stop_rule not in ("certificate", "plateau"):
        raise ValueError(
            f"Unknown functional_stop_rule={stop_rule!r}; expected "
            "'certificate' or 'plateau'."
        )
    plateau_mode = stop_rule == "plateau"
    # "fgd": train with verified-descent functional steps.
    # "adamw": train with AdamW between growths (apples-to-apples vs gromo_tiny),
    #          using the functional certificate *only* as the growth policy.
    train_optimizer = cfg.get("functional_train_optimizer", "fgd")

    warmup_optimizer = _build_optimizer(state.model, cfg)
    for _ in range(min(warmup_epochs, max_epochs)):
        state.epoch += 1
        gradient_descent(
            state.model,
            state.train_loader,
            warmup_optimizer,
            scheduler=None,
            loss_function=state.criterion,
            device=state.device,
        )
        state.log("warmup")

    # A fixed probe defines the empirical output space in which the certificate is
    # measured. "minibatch" scope keeps the legacy single-batch probe used by the
    # grow-until-certified refine loop; "fulldata" additionally keeps a fixed
    # multi-batch probe for the deterministic growth trigger.
    probe_x, probe_y = next(iter(state.train_loader))
    probe_x, probe_y = probe_x.to(state.device), probe_y.to(state.device)
    cert_batches = (
        _build_certificate_batches(state, certificate_probe_size)
        if certificate_scope == "fulldata"
        else None
    )

    consecutive_failures = 0
    for _ in range(max(0, max_epochs - warmup_epochs)):
        state.epoch += 1
        if train_optimizer == "fgd":
            info = functional_descent_epoch(
                state.model,
                state.train_loader,
                state.criterion,
                device=state.device,
                batch_limit=cfg.get("functional_batch_limit"),
                progress_every=cfg.get("functional_progress_every"),
                sufficient_descent_c=sufficient_descent_c,
                lr_backtrack=lr_backtrack,
                lr_min_factor=lr_min_factor,
                stop_on_failure=False,
                projection=cfg.get("functional_projection", "cg"),
                **_functional_step_kwargs(cfg),
            )
            phase = "fgd"
        else:
            # Train with AdamW; the functional certificate (measured on a probe
            # batch) is used only to decide whether the representation must grow.
            optimizer = _build_optimizer(state.model, cfg)
            gradient_descent(
                state.model,
                state.train_loader,
                optimizer,
                scheduler=None,
                loss_function=state.criterion,
                device=state.device,
            )
            info = _probe_certificate_info(state, probe_x, probe_y)
            phase = "adamw_certified"
        state.log(phase, functional_info=info)
        state.functional_events.append(
            {
                "epoch": state.epoch,
                "failed": info.failed,
                "certified_batches": info.certified_batches,
                "batches": info.batches,
                "max_relative_error": info.max_relative_error,
                "max_bottleneck_fraction": info.max_bottleneck_fraction,
                "mean_bottleneck_fraction": info.mean_bottleneck_fraction,
                "descent_failures": info.descent_failures,
                "failure_reason": info.failure.reason if info.failure else None,
            }
        )

        # The representation is insufficient when the expressivity bottleneck
        # dominates (certificate fails) or a step could not produce the Lemma-3.5
        # guaranteed descent.
        if certificate_scope == "fulldata":
            # Deterministic certificate on the fixed multi-batch probe is the sole
            # growth trigger (no fraction threshold, no hysteresis).
            cert = _fulldata_certificate_info(
                state, cert_batches,
                eps=cfg.get("functional_relative_error_tolerance", 0.2),
            )
            info.max_bottleneck_fraction = cert.max_bottleneck_fraction
            info.mean_bottleneck_fraction = cert.mean_bottleneck_fraction
            info.max_relative_error = cert.max_relative_error
            cur_bottleneck = cert.mean_bottleneck_fraction
            if plateau_mode:
                # eps-free: grow while the deterministic bottleneck keeps
                # decreasing; the certificate threshold (hence eps) plays no role
                # in the stop decision. The freeze block below sets growth_stalled
                # once a growth stops reducing the bottleneck.
                needs_refine = not growth_stalled
                representation_sufficient = growth_stalled
            else:
                # Stop iff the certificate holds (Algorithm 1): the representation
                # is guaranteed sufficient. ``growth_stalled`` only matters under
                # the opt-in marginal-utility ablation.
                needs_refine = cert.failed and not (marginal_utility_stop and growth_stalled)
                representation_sufficient = not cert.failed
            hysteresis_skip = False  # no hysteresis needed: the estimate is stable
        else:
            certify_fraction = info.certified_batches / max(1, info.batches)
            needs_refine = certify_fraction < certify_threshold or info.descent_failures > 0
            if certify_fraction >= certify_threshold:
                representation_sufficient = True
            hysteresis_skip = freeze_after_certified and representation_sufficient

        if not needs_refine:
            consecutive_failures = 0
            continue

        if hysteresis_skip:
            # Capacity already proven sufficient; treat this as optimization noise.
            consecutive_failures = 0
            continue

        consecutive_failures += 1
        if consecutive_failures < failure_patience:
            print(
                f"  >> representation insufficient (certified {info.certified_batches}"
                f"/{info.batches}, bottleneck~{info.mean_bottleneck_fraction:.2g}) "
                f"[{consecutive_failures}/{failure_patience}]; waiting before growth"
            )
            continue

        if state.growth_count >= max_growth_events:
            print("  >> representation insufficient, but growth budget is exhausted")
            if stop_when_exhausted:
                break
            consecutive_failures = 0
            continue

        if certificate_scope == "fulldata" and (marginal_utility_stop or plateau_mode):
            # Bottleneck-plateau stop. Only grow if the previous growth reduced the
            # deterministic bottleneck; once a growth stops paying off, the residual
            # is treated as irreducible and growth is frozen. Under the "plateau"
            # stop rule this is the *sole* stopping criterion (eps-free); under the
            # legacy marginal-utility ablation it gates the certificate rule.
            improved = cur_bottleneck < prev_bottleneck * (1.0 - growth_min_gain)
            if prev_bottleneck < float("inf") and not improved:
                growth_stalled = True
                print(
                    f"  >> growth stalled: bottleneck {cur_bottleneck:.3g} did not "
                    f"improve on {prev_bottleneck:.3g} (min_gain={growth_min_gain}); "
                    "freezing growth"
                )
                consecutive_failures = 0
                continue
            prev_bottleneck = cur_bottleneck

        if grow_until_certified:
            _grow_until_certified(
                state,
                probe_x,
                probe_y,
                eps=cfg.get("functional_relative_error_tolerance", 0.2),
                max_refines=max_refines,
                max_growth_events=max_growth_events,
            )
        else:
            # Incremental refinement: grow the layer chosen by the configured
            # policy (certifying selection only adds growths that make the
            # criterion hold); subsequent training trains the new neurons into the
            # tangent space before we re-evaluate.
            layer_to_grow, selection_info = _choose_functional_growth_layer(
                state,
                batches=cert_batches,
                eps=cfg.get("functional_relative_error_tolerance", 0.2),
            )
            _apply_growth(
                state,
                layer_to_grow,
                trigger="bottleneck_refine",
                extra={
                    **selection_info,
                    "certified_batches_before_growth": info.certified_batches,
                    "mean_bottleneck_before_growth": info.mean_bottleneck_fraction,
                },
            )
        consecutive_failures = 0


def _build_certificate_batches(
    state: _ExperimentState, size: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Collect a *fixed* set of train batches totalling (up to) ``size`` points.

    The expressivity bottleneck ``||grad L - P_T grad L|| / ||P_T grad L||`` is a
    property of the *function on the data distribution*, not of any one random
    mini-batch. Estimating the certificate on a fixed, larger probe makes the
    trigger low-variance and deterministic, which is what lets us drop the
    mini-batch-noise band-aids (fraction threshold + growth hysteresis) and use a
    tight tolerance. We keep the probe as a *list of mini-batches* rather than one
    giant batch so each exact Jacobian (jacrev) stays small in memory; the batches
    are combined in quadrature, which is exactly the certificate on their
    concatenation in output space.
    """
    batches, gathered = [], 0
    for bx, by in state.train_loader:
        batches.append((bx.to(state.device), by.to(state.device)))
        gathered += bx.shape[0]
        if gathered >= size:
            break
    return batches


def _fulldata_certificate_info(
    state: _ExperimentState,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    eps: float,
    model: torch.nn.Module | None = None,
) -> FunctionalEpochInfo:
    """Deterministic certificate over a fixed multi-batch probe.

    Each batch contributes ``||r_i||`` (residual / bottleneck) and ``||g_i||``
    (in-tangent projected gradient) and ``||grad L_i||``. Concatenating the
    batches in output space adds these norms in quadrature, so the certificate on
    the whole probe is ``(1+eps) sqrt(sum ||r_i||^2) < eps sqrt(sum ||g_i||^2)``.
    No parameters are modified (``apply_step=False``). ``model`` defaults to the
    live model but may be a (grown) trial model, which is what lets the certifying
    growth selector score each candidate growth by its post-growth certificate.
    """
    model = model if model is not None else state.model
    step_fn = (
        exact_functional_step
        if state.cfg.get("functional_projection", "cg") == "exact"
        else certified_functional_step
    )
    kwargs = _functional_step_kwargs(state.cfg)
    kwargs["relative_error_tolerance"] = eps
    sum_r2 = sum_g2 = sum_true2 = loss_sum = 0.0
    for x, y in batches:
        info = step_fn(model, x, y, state.criterion, apply_step=False, **kwargs)
        sum_r2 += info.residual_norm ** 2
        sum_g2 += info.in_tangent_norm ** 2
        sum_true2 += (info.residual_norm ** 2 + info.in_tangent_norm ** 2)
        loss_sum += info.loss
    r_norm = sum_r2 ** 0.5
    g_norm = sum_g2 ** 0.5
    true_norm = sum_true2 ** 0.5
    certified = bool((1.0 + eps) * r_norm < eps * g_norm)
    rel_err = r_norm / g_norm if g_norm > 0 else float("inf")
    bottleneck = r_norm / true_norm if true_norm > 0 else 0.0
    return FunctionalEpochInfo(
        loss=loss_sum / max(1, len(batches)),
        batches=len(batches),
        certified_batches=len(batches) if certified else 0,
        failed=not certified,
        failure=None,
        max_relative_error=rel_err,
        mean_relative_error=rel_err,
        descent_failures=0,
        mean_bottleneck_fraction=bottleneck,
        max_bottleneck_fraction=bottleneck,
    )


def _probe_certificate_info(
    state: _ExperimentState, x: torch.Tensor, y: torch.Tensor
) -> FunctionalEpochInfo:
    """Measure the functional certificate on a probe batch (no parameter change).

    Returns a ``FunctionalEpochInfo`` (one "batch") so the AdamW-trained variant
    can reuse the same growth-trigger logic as the FGD variant.
    """
    step_fn = (
        exact_functional_step
        if state.cfg.get("functional_projection", "cg") == "exact"
        else certified_functional_step
    )
    info = step_fn(
        state.model, x, y, state.criterion, apply_step=False,
        **_functional_step_kwargs(state.cfg),
    )
    return FunctionalEpochInfo(
        loss=info.loss,
        batches=1,
        certified_batches=1 if info.certified else 0,
        failed=not info.certified,
        failure=None if info.certified else info,
        max_relative_error=info.relative_error,
        mean_relative_error=info.relative_error,
        descent_failures=0,
        mean_bottleneck_fraction=info.bottleneck_fraction,
        max_bottleneck_fraction=info.bottleneck_fraction,
    )


def _grow_until_certified(
    state: _ExperimentState,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float,
    max_refines: int,
    max_growth_events: int,
):
    """Refine the representation (grow) until the certificate passes on ``x``.

    Faithful realization of the inner ``while`` loop of Algorithm 1: at each
    iteration we measure the relative-error certificate in the probe-batch output
    space without modifying parameters (``apply_step=False``); if it fails we add
    the neurons that best capture the expressivity-bottleneck residual.
    """
    step_kwargs = _functional_step_kwargs(state.cfg)
    step_fn = (
        exact_functional_step
        if state.cfg.get("functional_projection", "cg") == "exact"
        else certified_functional_step
    )
    last_info = None
    for refine in range(max_refines):
        info = step_fn(
            state.model, x, y, state.criterion, apply_step=False, **step_kwargs
        )
        last_info = info
        print(
            f"  refine {refine}: rel_err={info.relative_error:.3g} "
            f"bottleneck={info.bottleneck_fraction:.3g} "
            f"certified={info.certified}"
        )
        if info.certified:
            break
        if state.growth_count >= max_growth_events:
            print("  >> growth budget exhausted mid-refine")
            break
        layer_to_grow, selection_info = _choose_functional_growth_layer(state)
        _apply_growth(
            state,
            layer_to_grow,
            trigger="certificate_refine",
            extra={
                **selection_info,
                "refine_index": refine,
                "relative_error_before": info.relative_error,
                "bottleneck_fraction_before": info.bottleneck_fraction,
            },
        )
    return last_info


def _strip_cached_tensors(model: torch.nn.Module) -> None:
    """Drop gromo's cached forward tensors that are functorch-wrapped.

    gromo's growing layers cache the layer input (``_input``) during the forward
    pass for their growth statistics. When the forward happens inside the
    certificate's ``jacrev`` transform, that cache is a functorch ``TensorWrapper``
    that cannot be deep-copied (no accessible storage). The cache is recomputed by
    ``compute_statistics`` during the actual growth, so it is safe to clear before
    cloning a trial model.
    """
    for module in model.modules():
        for name, value in list(vars(module).items()):
            if torch.is_tensor(value):
                try:
                    value.untyped_storage()
                except (NotImplementedError, RuntimeError):
                    setattr(module, name, None)


def _select_certifying_growth_layer(
    state: _ExperimentState,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    eps: float,
) -> tuple[int, dict]:
    """Pick the growth option that makes the paper's certificate hold.

    For each growable layer we *trial-grow* a deep copy of the model with the TINY
    optimal update and measure the deterministic certificate on the fixed probe.
    Among options that certify (i.e. that make the criterion of Algorithm 1 hold),
    we add the one with the fewest new parameters -- the smallest representation
    that guarantees the algorithm. If none certifies at this tolerance, we add the
    option that *best advances toward* the criterion (lowest relative error), and
    the next epoch re-evaluates. The selection is driven by certificate
    satisfaction, not by the size of the bottleneck / first-order improvement.
    """
    base_params = _count_params(state.model)
    _strip_cached_tensors(state.model)
    trials = []
    for layer in range(state.cfg["number_hidden_layers"]):
        trial = copy.deepcopy(state.model)
        grow_step(
            trial,
            state.train_loader,
            layer_to_grow=layer,
            device=state.device,
            maximum_added_neurons=state.cfg.get("maximum_added_neurons"),
            line_search_factors=state.cfg.get(
                "line_search_factors", (0.0, 0.1, 0.5, 1.0)
            ),
        )
        cert = _fulldata_certificate_info(state, batches, eps=eps, model=trial)
        trials.append(
            {
                "layer": layer,
                "certified": not cert.failed,
                "relative_error": cert.max_relative_error,
                "bottleneck_fraction": cert.mean_bottleneck_fraction,
                "added_params": _count_params(trial) - base_params,
            }
        )
        del trial
    certifying = [t for t in trials if t["certified"]]
    if certifying:
        chosen = min(certifying, key=lambda t: t["added_params"])
    else:
        # No option certifies at eps: advance toward the criterion (Algorithm 1
        # keeps growing); pick the option whose post-growth certificate is closest.
        chosen = min(trials, key=lambda t: t["relative_error"])
    return chosen["layer"], {
        "selection": "certifying",
        "certifying_trials": trials,
        "chosen_certified": chosen["certified"],
        "chosen_relative_error": chosen["relative_error"],
    }


def _choose_functional_growth_layer(
    state: _ExperimentState,
    batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    eps: float | None = None,
) -> tuple[int, dict]:
    selection = state.cfg.get("functional_growth_layer_selection", "tiny_best")
    if selection == "sequential":
        layer = state.growth_count % max(1, state.cfg["number_hidden_layers"])
        return layer, {"selection": "sequential"}
    if selection == "tiny_best":
        return select_tiny_growth_layer(
            state.model,
            state.train_loader,
            state.device,
            maximum_added_neurons=state.cfg.get("maximum_added_neurons"),
        )
    if selection == "certifying":
        if batches is None or eps is None:
            raise ValueError(
                "certifying growth selection requires a probe and tolerance "
                "(only available under functional_certificate_scope=fulldata)."
            )
        return _select_certifying_growth_layer(state, batches, eps)
    raise ValueError(
        f"Unknown functional_growth_layer_selection={selection!r}. "
        "Expected 'tiny_best', 'sequential', or 'certifying'."
    )


def _apply_growth(
    state: _ExperimentState,
    layer_to_grow: int,
    *,
    trigger: str,
    extra: dict | None = None,
) -> None:
    info = grow_step(
        state.model,
        state.train_loader,
        layer_to_grow=layer_to_grow,
        device=state.device,
        maximum_added_neurons=state.cfg.get("maximum_added_neurons"),
        line_search_factors=state.cfg.get("line_search_factors", (0.0, 0.1, 0.5, 1.0)),
    )
    state.growth_count += 1
    growth_record = {
        "step": state.growth_count - 1,
        "epoch": state.epoch,
        "trigger": trigger,
        **info,
    }
    if extra:
        growth_record.update(extra)
    state.growth_info.append(growth_record)
    state.growth_lines.append(state.eval_idx - 0.5)
    print(
        f"  >> grew layer {layer_to_grow}: trigger={trigger} "
        f"scaling={info['scaling_factor']} line_search={info['line_search']}"
    )
    state.log("post_grow")


def _run_final_sgd(state: _ExperimentState) -> None:
    if state.cfg.get("final_epochs", 0) <= 0:
        return
    optimizer = _build_optimizer(state.model, state.cfg)
    for _ in range(state.cfg["final_epochs"]):
        state.epoch += 1
        gradient_descent(
            state.model,
            state.train_loader,
            optimizer,
            scheduler=None,
            loss_function=state.criterion,
            device=state.device,
        )
        state.log("sgd")
