"""Train/grow loops for comparing post-growth training dynamics.

The harness supports three methods:

- ``baseline_mlp``: ordinary SGD training, no growth.
- ``gromo_tiny``: the original scheduled GroMo/TINY growth loop.
- ``functional_triggered_tiny``: certified empirical functional-gradient
  training on the current DAG; a failed relative-error certificate triggers a
  normal GroMo/TINY growth step.
"""

from __future__ import annotations

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

    # A fixed probe batch defines the empirical output space in which the
    # certificate is measured during the grow-until-certified refine loop.
    probe_x, probe_y = next(iter(state.train_loader))
    probe_x, probe_y = probe_x.to(state.device), probe_y.to(state.device)

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

        # The representation is insufficient when too few batches certify (the
        # expressivity bottleneck dominates) or a step could not produce the
        # Lemma-3.5 guaranteed descent.
        certify_fraction = info.certified_batches / max(1, info.batches)
        needs_refine = certify_fraction < certify_threshold or info.descent_failures > 0
        if certify_fraction >= certify_threshold:
            representation_sufficient = True
        if not needs_refine:
            consecutive_failures = 0
            continue

        if freeze_after_certified and representation_sufficient:
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
            # Incremental refinement: grow the layer whose TINY optimal update
            # best captures the bottleneck residual; subsequent FGD steps train
            # the new neurons into the tangent space before we re-evaluate.
            layer_to_grow, selection_info = _choose_functional_growth_layer(state)
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


def _choose_functional_growth_layer(state: _ExperimentState) -> tuple[int, dict]:
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
    raise ValueError(
        f"Unknown functional_growth_layer_selection={selection!r}. "
        "Expected 'tiny_best' or 'sequential'."
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
