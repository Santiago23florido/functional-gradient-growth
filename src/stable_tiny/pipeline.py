"""Pipeline that joins config, data, train, grow, and outputs."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from stable_tiny.data import (
    MultiSinDataLoader,
    SmoothSinDataLoader,
    make_cifar10_dataloaders,
    make_mnist_dataloaders,
)
from fgdlib.tangent import (
    FGDApproxConfig,
    FGDApproxEpochResult,
    FGDLayerRelError,
    FGDOutputRelError,
    FGDValidationCertificate,
    ParametricDescentConfig,
    ParametricGDConfig,
    SecantFGDConfig,
    _FunctionalStepStats,
    _clear_inaccessible_tensor_caches,
    _compute_tangent_projection_step,
    _projection_step_sensor_valid,
    _trainable_named_parameters,
    batch_functional_loss,
    build_projection_probe,
    FUNCTIONAL_HAS_PL_CONSTANT,
    certificate_from_projection_stats,
    evaluate_fgd_validation_certificate,
    evaluate_secant_validation_certificate,
    measure_direction_projection,
    functional_gradient,
    select_tiny_growth_layer_index,
    should_trigger_fgd_growth,
    tiny_optimal_update_kwargs,
    train_one_epoch_fgd_approx,
    validate_family_order,
    validate_functional_loss,
)
from fgdlib.rkhs import (
    FGDRKHSConfig,
    FGDRKHSEpochResult,
    FGDRKHSStepRecord,
    FGDRKHSTrainer,
    FrozenAffineFeatureMap,
    KernelDictionaryModel,
)
from fgdlib.gromo_setup import ensure_gromo_importable
from fgdlib.depth import insert_identity_layer
from fgdlib.unified_growth import (
    Candidate,
    bottleneck_relief_target,
    rank_candidates,
    rank_limiting_locations,
)
from fgdlib.growth import (
    GrowthResult,
    ScalingLineSearchConfig,
    allocate_by_expansion_per_parameter,
    expansion_spectrum,
    grow_layer,
    growable_neuron_costs,
    rank_layer_expansion_score,
)
from fgdlib.growth_schedule import (
    GrowthScheduleConfig,
    layer_index_for_growth,
    should_grow,
)
from fgdlib.lr_scheduler import (
    LRSchedulerConfig,
    apply_learning_rate,
    learning_rate_for_epoch,
)
from fgdlib.optim import OptimizerConfig, build_optimizer, current_learning_rate
from fgdlib.training import (
    count_parameters,
    evaluate_regression_metrics,
    train_one_epoch,
)
from stable_tiny.wandb_logging import WandbConfig, build_wandb_logger


ensure_gromo_importable()

import torch

from gromo.containers.growing_mlp import GrowingMLP


ProgressFn = Callable[[str], None]
TrainingMethod = Literal["normal", "fgd_approx", "fgd_rkhs", "fgd_rkhs_grow"]
StepType = Literal["INIT", "SGD", "FGD", "SEC", "GRO", "RKHS"]
DataKind = Literal["multi_sin", "smooth_sin", "cifar10", "mnist"]


@dataclass(frozen=True)
class DataConfig:
    kind: DataKind = "smooth_sin"
    in_features: int = 10
    out_features: int = 3
    data_dir: str | None = None
    train_batches: int = 10
    validation_batches: int = 10
    test_batches: int = 1
    batch_size: int = 1_000
    train_seed: int = 0
    validation_seed: int = 2
    test_seed: int = 1
    active_features: int = 2
    frequency: float = 1.0
    phase_shift: float = 0.5
    interaction_strength: float = 0.25
    linear_strength: float = 0.1
    cifar_grayscale: bool = True
    cifar_train_samples: int | None = 5_000
    cifar_validation_samples: int | None = 1_000
    cifar_test_samples: int | None = 1_000
    mnist_train_samples: int | None = 10_000
    mnist_validation_samples: int | None = 2_000
    mnist_test_samples: int | None = 2_000


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 2
    number_hidden_layers: int = 2
    model_seed: int = 0


@dataclass(frozen=True)
class TrainingConfig:
    method: TrainingMethod = "normal"
    epochs: int = 200
    accuracy_tolerance: float = 1.0
    gradient_clip_norm: float | None = 1.0
    log_every: int = 10
    device: str = "auto"


@dataclass(frozen=True)
class RunConfig:
    name: str = "gromo_tutorial_baseline"
    results_dir: Path = Path("results")
    save_plot: bool = True
    show_plot: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    fgd_approx: FGDApproxConfig = field(default_factory=FGDApproxConfig)
    secant_fgd: SecantFGDConfig = field(default_factory=SecantFGDConfig)
    parametric_gd: ParametricGDConfig = field(default_factory=ParametricGDConfig)
    parametric_descent: ParametricDescentConfig = field(
        default_factory=ParametricDescentConfig
    )
    fgd_rkhs: FGDRKHSConfig = field(default_factory=FGDRKHSConfig)
    scaling_line_search: ScalingLineSearchConfig = field(
        default_factory=ScalingLineSearchConfig
    )
    growth_schedule: GrowthScheduleConfig = field(default_factory=GrowthScheduleConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    run: RunConfig = field(default_factory=RunConfig)


@dataclass(frozen=True)
class HistoryEntry:
    step: int
    step_type: StepType
    train_loss: float
    validation_loss: float
    test_loss: float
    train_accuracy: float
    validation_accuracy: float
    test_accuracy: float
    learning_rate: float
    num_params: int
    layer_index: int | None = None
    scaling_factor: float | None = None
    rel_error: float | None = None
    selected_layer_index: int | None = None
    fgd_layer_rel_errors: list[FGDLayerRelError] = field(default_factory=list)
    fgd_output_rel_error: FGDOutputRelError | None = None
    fgd_learning_rate_upper_bound: float | None = None
    fgd_max_valid_learning_rate: float | None = None
    fgd_learning_rate_interval_valid: bool | None = None
    fgd_learning_rate_clipped_batches: int = 0
    fgd_skipped_batches: int = 0
    fgd_relative_error_condition_valid: bool | None = None
    fgd_loss_descent_valid: bool | None = None
    fgd_loss_non_descent_batches: int = 0
    fgd_gradient_sq_norm: float | None = None
    fgd_min_gradient_sq_norm: float | None = None
    fgd_theory_descent_coefficient: float | None = None
    fgd_stationary_bound: float | None = None
    fgd_stationary_bound_valid: bool | None = None
    fgd_global_bound: float | None = None
    fgd_global_bound_valid: bool | None = None
    fgd_global_contraction: float | None = None
    fgd_theory_learning_rate_adjusted: bool = False
    fgd_sensor_valid: bool | None = None
    fgd_sensor_invalid_batches: int = 0
    fgd_update_norm: float | None = None
    fgd_candidate_accepted: bool | None = None
    fgd_lr_search_trials: int = 0
    fgd_approximation_kind: str | None = None
    fgd_rkhs_phase_attempted: bool = False
    fgd_rkhs_phase_accepted: bool | None = None
    fgd_rkhs_phase_steps: int = 0
    fgd_growth_probe_improved: bool | None = None
    fgd_rkhs_dictionary_size: int | None = None
    fgd_rkhs_functional_loss: float | None = None
    fgd_rkhs_loss_star: float | None = None


@dataclass
class PipelineResult:
    config: PipelineConfig
    history: list[HistoryEntry]
    growth_events: list[GrowthResult]
    model: GrowingMLP
    device: str


@dataclass(frozen=True)
class _FGDTheoryState:
    epoch_count: int
    min_gradient_sq_norm: float | None
    min_positive_learning_rate: float | None
    min_descent_coefficient: float | None
    global_contraction_product: float
    previous_validation_functional_loss: float


@dataclass(frozen=True)
class _FGDTrial:
    model: GrowingMLP
    epoch_result: FGDApproxEpochResult
    certificate: FGDValidationCertificate
    theory_state: _FGDTheoryState
    validation_functional_loss: float
    loss_descent_valid: bool
    stationary_bound: float | None
    stationary_bound_valid: bool | None
    global_bound: float | None
    global_bound_valid: bool | None
    global_contraction: float | None
    all_conditions_valid: bool


@dataclass(frozen=True)
class _FGDSearchResult:
    accepted: _FGDTrial | None
    last_trial: _FGDTrial | None
    trial_count: int
    sensor_failure: bool


@dataclass(frozen=True)
class _GrowthProbe:
    model: GrowingMLP
    result: GrowthResult
    certificate: FGDValidationCertificate
    improves_fgd: bool
    # Measured validation functional descent this growth realizes
    # (loss_before - loss_after) and the parameters it adds. Used by the
    # descent-per-parameter selection (Prop. 3.8-certified growth), which
    # is coherent with delta growth where the rel-error jumps.
    functional_descent: float = 0.0
    added_parameters: int = 0
    # Reduction of the Lemma-3.5 relative error this growth achieves once
    # the new capacity has been USED (one certified family step on the
    # grown clone). Positive means the reachable set genuinely expanded
    # toward r. See fgd_approx.growth_selection = "epsilon_lookahead".
    epsilon_reduction: float | None = None
    # The pre-growth relative error the reduction above is measured from.
    # SENN's expansion score is a nonlinear function of eps, so ranking by
    # score needs the base point, not only the delta.
    epsilon_before: float | None = None


def _section_dataclass(
    section_name: str,
    section_type: type,
    raw_config: Mapping[str, Any],
) -> Any:
    values = dict(raw_config.get(section_name, {}) or {})
    valid_keys = {field.name for field in fields(section_type)}
    unknown_keys = sorted(set(values) - valid_keys)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"Unknown keys in config section '{section_name}': {joined}")

    if section_type is OptimizerConfig and "betas" in values:
        betas = tuple(float(value) for value in values["betas"])
        if len(betas) != 2:
            raise ValueError("optimizer.betas must contain exactly two values")
        values["betas"] = betas

    if section_type is RunConfig and "results_dir" in values:
        values["results_dir"] = Path(values["results_dir"])

    if section_type is WandbConfig and "tags" in values:
        values["tags"] = tuple(str(value) for value in values["tags"] or ())

    if section_type is FGDRKHSConfig and "levels" in values:
        values["levels"] = (
            tuple(int(value) for value in values["levels"])
            if values["levels"] is not None
            else None
        )

    if section_type is FGDApproxConfig and "family_order" in values:
        values["family_order"] = tuple(
            str(value) for value in values["family_order"] or ()
        )

    if section_type in (ParametricGDConfig, ParametricDescentConfig):
        if "inner_steps" in values:
            values["inner_steps"] = tuple(
                int(value) for value in values["inner_steps"] or ()
            )
        if "functional_learning_rates" in values:
            values["functional_learning_rates"] = tuple(
                float(value) for value in values["functional_learning_rates"] or ()
            )

    return section_type(**values)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load pipeline hyperparameters from YAML."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected a YAML mapping in {config_path}")

    known_sections = {
        "data",
        "model",
        "training",
        "optimizer",
        "lr_scheduler",
        "fgd_approx",
        "secant_fgd",
        "parametric_gd",
        "parametric_descent",
        "fgd_rkhs",
        "scaling_line_search",
        "growth_schedule",
        "wandb",
        "run",
    }
    unknown_sections = sorted(set(raw) - known_sections)
    if unknown_sections:
        joined = ", ".join(unknown_sections)
        raise ValueError(f"Unknown config sections in {config_path}: {joined}")

    config = PipelineConfig(
        data=_section_dataclass("data", DataConfig, raw),
        model=_section_dataclass("model", ModelConfig, raw),
        training=_section_dataclass("training", TrainingConfig, raw),
        optimizer=_section_dataclass("optimizer", OptimizerConfig, raw),
        lr_scheduler=_section_dataclass("lr_scheduler", LRSchedulerConfig, raw),
        fgd_approx=_section_dataclass("fgd_approx", FGDApproxConfig, raw),
        secant_fgd=_section_dataclass("secant_fgd", SecantFGDConfig, raw),
        parametric_gd=_section_dataclass(
            "parametric_gd",
            ParametricGDConfig,
            raw,
        ),
        parametric_descent=_section_dataclass(
            "parametric_descent",
            ParametricDescentConfig,
            raw,
        ),
        fgd_rkhs=_section_dataclass("fgd_rkhs", FGDRKHSConfig, raw),
        scaling_line_search=_section_dataclass(
            "scaling_line_search",
            ScalingLineSearchConfig,
            raw,
        ),
        growth_schedule=_section_dataclass(
            "growth_schedule",
            GrowthScheduleConfig,
            raw,
        ),
        wandb=_section_dataclass("wandb", WandbConfig, raw),
        run=_section_dataclass("run", RunConfig, raw),
    )
    validate_family_order(config.fgd_approx.family_order)
    validate_functional_loss(config.fgd_approx.functional_loss)
    config.parametric_gd.validate()
    config.parametric_descent.validate()
    return config


def with_run_overrides(
    config: PipelineConfig,
    *,
    name: str | None = None,
    results_dir: Path | None = None,
    save_plot: bool | None = None,
    show_plot: bool | None = None,
) -> PipelineConfig:
    """Return a config with CLI run-output overrides applied."""
    run_config = config.run
    if name is not None:
        run_config = replace(run_config, name=name)
    if results_dir is not None:
        run_config = replace(run_config, results_dir=results_dir)
    if save_plot is not None:
        run_config = replace(run_config, save_plot=save_plot)
    if show_plot is not None:
        run_config = replace(run_config, show_plot=show_plot)
    return replace(config, run=run_config)


def with_wandb_overrides(
    config: PipelineConfig,
    *,
    enabled: bool | None = None,
    project: str | None = None,
    entity: str | None = None,
    group: str | None = None,
    mode: str | None = None,
    tags: list[str] | None = None,
) -> PipelineConfig:
    """Return a config with CLI W&B overrides applied."""
    wandb_config = config.wandb
    if enabled is not None:
        wandb_config = replace(wandb_config, enabled=enabled)
    if project is not None:
        wandb_config = replace(wandb_config, project=project)
    if entity is not None:
        wandb_config = replace(wandb_config, entity=entity)
    if group is not None:
        wandb_config = replace(wandb_config, group=group)
    if mode is not None:
        wandb_config = replace(wandb_config, mode=mode)
    if tags:
        wandb_config = replace(wandb_config, tags=wandb_config.tags + tuple(tags))
    return replace(config, wandb=wandb_config)


def with_growth_overrides(
    config: PipelineConfig,
    *,
    enabled: bool | None = None,
) -> PipelineConfig:
    """Return a config with CLI growth-schedule overrides applied."""
    growth_schedule = config.growth_schedule
    lr_scheduler = config.lr_scheduler
    if enabled is not None:
        growth_schedule = replace(growth_schedule, enabled=enabled)
        if not enabled:
            lr_scheduler = replace(
                lr_scheduler,
                restart_on_growth=False,
                t_max=config.training.epochs,
            )
    return replace(config, growth_schedule=growth_schedule, lr_scheduler=lr_scheduler)


def with_fgd_overrides(
    config: PipelineConfig,
    *,
    projection_solver: str | None = None,
    global_bound_action: str | None = None,
) -> PipelineConfig:
    """Return a config with FGD-specific CLI overrides applied."""
    fgd_config = config.fgd_approx
    if projection_solver is not None:
        fgd_config = replace(fgd_config, projection_solver=projection_solver)
    if global_bound_action is not None:
        fgd_config = replace(fgd_config, global_bound_action=global_bound_action)
    return replace(config, fgd_approx=fgd_config)


def select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_dataloaders(
    config: PipelineConfig,
    device: torch.device,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    data_config = config.data
    if data_config.kind == "cifar10":
        expected_features = 1024 if data_config.cifar_grayscale else 3072
        if data_config.in_features != expected_features:
            raise ValueError(
                "CIFAR-10 feature size mismatch: expected "
                f"data.in_features={expected_features} for "
                f"cifar_grayscale={data_config.cifar_grayscale}."
            )
        if data_config.out_features != 10:
            raise ValueError("CIFAR-10 requires data.out_features=10.")
        return make_cifar10_dataloaders(
            data_dir=data_config.data_dir,
            train_samples=data_config.cifar_train_samples,
            validation_samples=data_config.cifar_validation_samples,
            test_samples=data_config.cifar_test_samples,
            batch_size=data_config.batch_size,
            grayscale=data_config.cifar_grayscale,
            seed=data_config.train_seed,
            num_classes=data_config.out_features,
        )
    if data_config.kind == "mnist":
        if data_config.in_features != 784:
            raise ValueError("MNIST requires data.in_features=784.")
        if data_config.out_features != 10:
            raise ValueError("MNIST requires data.out_features=10.")
        return make_mnist_dataloaders(
            data_dir=data_config.data_dir,
            train_samples=data_config.mnist_train_samples,
            validation_samples=data_config.mnist_validation_samples,
            test_samples=data_config.mnist_test_samples,
            batch_size=data_config.batch_size,
            seed=data_config.train_seed,
            num_classes=data_config.out_features,
        )

    if data_config.kind == "multi_sin":
        loader_class = MultiSinDataLoader
        extra_kwargs: dict[str, Any] = {}
    elif data_config.kind == "smooth_sin":
        loader_class = SmoothSinDataLoader
        extra_kwargs = {
            "active_features": data_config.active_features,
            "frequency": data_config.frequency,
            "phase_shift": data_config.phase_shift,
            "interaction_strength": data_config.interaction_strength,
            "linear_strength": data_config.linear_strength,
        }
    else:
        raise ValueError(
            f"Unsupported data kind '{data_config.kind}'. "
            "Use one of: multi_sin, smooth_sin, cifar10, mnist."
        )

    train_loader = loader_class(
        nb_sample=data_config.train_batches,
        batch_size=data_config.batch_size,
        in_features=data_config.in_features,
        out_features=data_config.out_features,
        seed=data_config.train_seed,
        device=device,
        **extra_kwargs,
    )
    validation_loader = loader_class(
        nb_sample=data_config.validation_batches,
        batch_size=data_config.batch_size,
        in_features=data_config.in_features,
        out_features=data_config.out_features,
        seed=data_config.validation_seed,
        device=device,
        **extra_kwargs,
    )
    test_loader = loader_class(
        nb_sample=data_config.test_batches,
        batch_size=data_config.batch_size,
        in_features=data_config.in_features,
        out_features=data_config.out_features,
        seed=data_config.test_seed,
        device=device,
        **extra_kwargs,
    )
    return train_loader, validation_loader, test_loader


def is_classification_task(config: PipelineConfig) -> bool:
    return config.data.kind in {"cifar10", "mnist"}


def build_model(config: PipelineConfig, device: torch.device) -> GrowingMLP:
    data_config = config.data
    model_config = config.model
    torch.manual_seed(model_config.model_seed)
    return GrowingMLP(
        in_features=data_config.in_features,
        out_features=data_config.out_features,
        hidden_size=model_config.hidden_size,
        number_hidden_layers=model_config.number_hidden_layers,
        device=device,
    )


def should_log_epoch(epoch: int, config: PipelineConfig) -> bool:
    log_every = config.training.log_every
    return epoch == 1 or epoch == config.training.epochs or (
        log_every > 0 and epoch % log_every == 0
    )


def scheduled_learning_rate(
    config: PipelineConfig,
    epoch: int,
    cycle_start_epoch: int,
) -> float:
    return learning_rate_for_epoch(
        config.lr_scheduler,
        base_learning_rate=config.optimizer.learning_rate,
        epoch=epoch,
        total_epochs=config.training.epochs,
        growth_every=config.growth_schedule.every,
        first_growth_epoch=config.growth_schedule.first_epoch,
        cycle_start_epoch=cycle_start_epoch,
    )


@torch.no_grad()
def evaluate_functional_loss(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    functional_loss: str = "mse",
) -> float:
    """Total certified functional loss L(f) over ``data_loader``.

    This is the quantity every certificate in the flow is a statement
    about, so it must be the SAME functional the families and the growth
    trigger use (fgd_approx.functional_loss).
    """
    model.eval()
    total_loss = 0.0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        total_loss += float(
            batch_functional_loss(model(x), y, functional_loss).detach().item()
        )
    return total_loss


def certified_validation_learning_rate(
    certificate: FGDValidationCertificate,
    config: FGDApproxConfig,
) -> float | None:
    """Return the validation-certified LR, including the safety factor."""
    learning_rate = certificate.max_valid_learning_rate
    if (
        not certificate.sensor_valid
        or learning_rate is None
        or learning_rate <= config.theory_lr_min + config.eps
    ):
        return None
    return learning_rate


def _certified_pl_constant(config: PipelineConfig) -> float:
    """Return the PL constant mu that may legitimately be asserted.

    Proposition 3.8's LINEAR contraction rests on a global
    Polyak-Lojasiewicz constant, ||grad_f L||^2 >= 2 mu (L - L*). Only
    sum-MSE admits one: the identity ||r||^2 = 4L holds exactly, giving
    mu = 2 with L* = 0. Cross-entropy admits none -- ||r||^2 = Theta(L^2)
    as p_c -> 1 and ||r||^2 stays bounded while L diverges as p_c -> 0 --
    so the C_glob envelope is simply not available for it and returning 0
    leaves the bound undefined rather than asserting an invented rate.
    Convexity is untouched, so global optimality still holds; only the RATE
    drops to the convex O(1/T) bound. See report/CROSS_ENTROPY_FGD.md.
    """
    if not FUNCTIONAL_HAS_PL_CONSTANT.get(config.fgd_approx.functional_loss, False):
        return 0.0
    return config.fgd_approx.theory_mu


def _certify_fgd_candidate(
    *,
    candidate_model: GrowingMLP,
    epoch_result: FGDApproxEpochResult,
    certificate: FGDValidationCertificate,
    validation_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: PipelineConfig,
    theory_state: _FGDTheoryState,
    initial_functional_gap: float,
    theory_loss_star: float,
) -> _FGDTrial:
    """Evaluate the FGD acceptance conditions for a realizable candidate.

    With fgd_approx.local_acceptance_conditions enabled, acceptance is
    decided by the four LOCAL conditions of the current outer step (sensor,
    Crel, LR interval, strict realized descent) and the stationary/global
    convergence bounds are computed and logged only as diagnostics of the
    ACCUMULATED trajectory guarantees. With the flag off (default), the
    legacy gates apply: non-strict descent and Cstat/Cglob as acceptance
    conditions.
    """
    local_acceptance = config.fgd_approx.local_acceptance_conditions
    validation_functional_loss = evaluate_functional_loss(
        candidate_model,
        validation_loader,
        device,
        config.fgd_approx.functional_loss,
    )
    if local_acceptance:
        # Local condition 4: STRICT realized descent of the validation
        # functional loss, L(f_{t+1}) < L(f_t), up to the numerical eps
        # only.
        loss_descent_valid = (
            math.isfinite(validation_functional_loss)
            and theory_state.previous_validation_functional_loss
            - validation_functional_loss
            > config.fgd_approx.eps
        )
    else:
        # Legacy gate: non-strict descent (a no-progress step passes).
        loss_descent_valid = (
            validation_functional_loss
            <= theory_state.previous_validation_functional_loss
            + config.fgd_approx.eps
        )

    epoch_count = theory_state.epoch_count
    min_gradient_sq_norm = theory_state.min_gradient_sq_norm
    min_positive_learning_rate = theory_state.min_positive_learning_rate
    min_descent_coefficient = theory_state.min_descent_coefficient
    contraction_product = theory_state.global_contraction_product

    eta = epoch_result.min_positive_learning_rate
    gradient_sq_norm = certificate.gradient_sq_norm
    descent_coefficient = certificate.theory_descent_coefficient
    if gradient_sq_norm is not None and eta is not None:
        epoch_count += 1
        min_gradient_sq_norm = (
            gradient_sq_norm
            if min_gradient_sq_norm is None
            else min(min_gradient_sq_norm, gradient_sq_norm)
        )
        min_positive_learning_rate = (
            eta
            if min_positive_learning_rate is None
            else min(min_positive_learning_rate, eta)
        )
    if descent_coefficient is not None and descent_coefficient > 0.0:
        min_descent_coefficient = (
            descent_coefficient
            if min_descent_coefficient is None
            else min(min_descent_coefficient, descent_coefficient)
        )

    # Trajectory diagnostics: the stationary and global bounds monitor the
    # ACCUMULATED theoretical guarantees over the committed steps. They are
    # computed and logged for every trial but are NOT acceptance gates.
    stationary_bound: float | None = None
    stationary_bound_valid: bool | None = None
    global_bound: float | None = None
    global_bound_valid: bool | None = None
    global_contraction: float | None = None
    if (
        epoch_count > 0
        and min_positive_learning_rate is not None
        and min_descent_coefficient is not None
        and min_positive_learning_rate > 0.0
        and min_descent_coefficient > 0.0
    ):
        stationary_bound = initial_functional_gap / (
            epoch_count
            * min_positive_learning_rate
            * min_descent_coefficient
        )
        stationary_bound_valid = (
            min_gradient_sq_norm is not None
            and min_gradient_sq_norm
            <= stationary_bound + config.fgd_approx.eps
        )

        beta = config.fgd_approx.theory_beta
        mu = _certified_pl_constant(config)
        if (
            eta is not None
            and descent_coefficient is not None
            and beta > 0
            and mu > 0
        ):
            global_contraction = 1.0 - (
                2.0 * eta * mu * descent_coefficient / (beta**2)
            )
            contraction_product *= global_contraction
            global_bound = contraction_product * initial_functional_gap
            current_gap = max(
                validation_functional_loss - theory_loss_star,
                0.0,
            )
            global_bound_valid = (
                current_gap <= global_bound + config.fgd_approx.eps
            )

    updated_state = _FGDTheoryState(
        epoch_count=epoch_count,
        min_gradient_sq_norm=min_gradient_sq_norm,
        min_positive_learning_rate=min_positive_learning_rate,
        min_descent_coefficient=min_descent_coefficient,
        global_contraction_product=contraction_product,
        previous_validation_functional_loss=validation_functional_loss,
    )
    # LOCAL acceptance conditions — these four decide whether THIS outer
    # step commits:
    #   1. the projection and numerical sensor are valid;
    #   2. Crel: relative_error < min(rel_error_threshold, 0.5), strict;
    #   3. LR interval: theory_lr_min < eta < safe upper bound eta_bar,
    #      strict up to the configured eps;
    #   4. strict realized descent of the validation functional loss.
    all_conditions_valid = (
        epoch_result.sensor_valid
        and epoch_result.skipped_batches == 0
        and certificate.sensor_valid
        and certificate.relative_error_condition_valid is True
        and certificate.learning_rate_interval_valid is True
        and loss_descent_valid
    )
    if not local_acceptance:
        # Legacy mode: the accumulated stationary and global bounds also
        # gate acceptance. Under local_acceptance_conditions they are
        # intentionally ABSENT — they describe the trajectory, not this
        # step.
        all_conditions_valid = (
            all_conditions_valid
            and stationary_bound_valid is True
            and global_bound_valid is True
        )
    return _FGDTrial(
        model=candidate_model,
        epoch_result=epoch_result,
        certificate=certificate,
        theory_state=updated_state,
        validation_functional_loss=validation_functional_loss,
        loss_descent_valid=loss_descent_valid,
        stationary_bound=stationary_bound,
        stationary_bound_valid=stationary_bound_valid,
        global_bound=global_bound,
        global_bound_valid=global_bound_valid,
        global_contraction=global_contraction,
        all_conditions_valid=all_conditions_valid,
    )


def _apply_shared_direction_step(
    model: GrowingMLP,
    direction: tuple[torch.Tensor, ...],
    learning_rate: float,
) -> None:
    """Apply theta <- theta - eta * u for the shared probe direction u."""
    parameters = tuple(_trainable_named_parameters(model).values())
    if len(parameters) != len(direction):
        raise RuntimeError(
            "Shared-direction update does not match the trainable parameters."
        )
    with torch.no_grad():
        for parameter, update in zip(parameters, direction):
            parameter.add_(update.to(parameter.device), alpha=-learning_rate)


def _search_tangent_measured_descent(
    *,
    base_model: GrowingMLP,
    direction: tuple[torch.Tensor, ...],
    direction_stats: _FunctionalStepStats,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    accuracy_tolerance: float,
    config: PipelineConfig,
    theory_state: _FGDTheoryState,
    initial_functional_gap: float,
    theory_loss_star: float,
) -> _FGDSearchResult:
    """Certify the tangent functional-gradient step by MEASURED descent.

    The direction is the paper's g = P_T r (tangent projection); the step
    SIZE is chosen by a nonlinear line search that maximizes the certified
    measured validation descent (Prop. 3.8), instead of the worst-case
    Lemma-3.5 bound eta_max(eps). The line search sweeps a descending
    geometric grid of eta and commits the certified step with the largest
    measured progress. Every accepted step still satisfies the paper's
    per-step descent inequality — only the sufficient mechanism used to
    certify it changes (measured coefficient, not the epsilon interval).
    """
    relative_error = direction_stats.output_error.relative_error
    maximum_learning_rate = config.fgd_approx.tangent_measured_max_lr
    floor = config.fgd_approx.theory_lr_min + config.fgd_approx.eps
    steps = max(1, config.fgd_approx.theory_lr_search_steps)
    trial_count = 0
    best_trial: _FGDTrial | None = None
    last_trial: _FGDTrial | None = None
    best_progress = float("-inf")
    for index in range(steps):
        fraction = index / max(1, steps - 1)
        eta = maximum_learning_rate * (
            floor / maximum_learning_rate
        ) ** fraction
        if eta <= floor:
            continue
        candidate = copy.deepcopy(base_model)
        _apply_shared_direction_step(candidate, direction, eta)
        trial = _certify_measured_descent_candidate(
            candidate_model=candidate,
            base_model=base_model,
            train_batches=train_batches,
            validation_loader=validation_loader,
            loss_function=loss_function,
            device=device,
            eta_star=eta,
            relative_error=relative_error,
            accuracy_tolerance=accuracy_tolerance,
            config=config,
            classification=False,
            theory_state=theory_state,
            initial_functional_gap=initial_functional_gap,
            theory_loss_star=theory_loss_star,
        )
        trial_count += 1
        last_trial = trial
        if not trial.all_conditions_valid:
            continue
        progress = _certified_trial_progress(trial)
        if progress > best_progress:
            best_progress = progress
            best_trial = trial
    return _FGDSearchResult(
        best_trial,
        best_trial if best_trial is not None else last_trial,
        trial_count,
        False,
    )


def _evaluate_fgd_outer_trial(
    *,
    base_model: GrowingMLP,
    direction: tuple[torch.Tensor, ...],
    direction_stats: _FunctionalStepStats,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    learning_rate: float,
    accuracy_tolerance: float,
    config: PipelineConfig,
    classification: bool,
    theory_state: _FGDTheoryState,
    initial_functional_gap: float,
    theory_loss_star: float,
) -> _FGDTrial:
    """One genuine FGD outer step: theta - eta * u with the shared direction.

    The relative-error certificate comes from ``direction_stats``, measured
    at the CURRENT model f_t for the exact direction u that moves it — never
    at the endpoint of a training epoch, and never for a direction other
    than the one applied. The stepped clone is then checked transactionally
    (validation functional loss before vs after, Cstat, Cglob); rejection
    rolls back by discarding the clone.

    u is the projection on the TRAIN probe; on the validation probe it is a
    general Hilbert direction, so only finiteness is sensor-checked here
    (the exact-projector invariants do not apply) and Crel/the LR interval
    are the binding admissibility gates, as in Lemma 3.5.
    """
    certificate = certificate_from_projection_stats(
        stats=direction_stats,
        learning_rate=learning_rate,
        config=config.fgd_approx,
        projection_sensor=False,
    )
    trial_model = copy.deepcopy(base_model)
    _apply_shared_direction_step(trial_model, direction, learning_rate)
    train_metrics = evaluate_regression_metrics(
        trial_model,
        train_batches,
        loss_function,
        device=device,
        accuracy_tolerance=accuracy_tolerance,
        classification=classification,
    )
    epoch_result = FGDApproxEpochResult(
        train_loss=train_metrics.loss,
        train_accuracy=train_metrics.accuracy,
        test_loss=float("nan"),
        test_accuracy=float("nan"),
        learning_rate=learning_rate,
        next_learning_rate=None,
        learning_rate_upper_bound=certificate.learning_rate_upper_bound,
        learning_rate_interval_valid=certificate.learning_rate_interval_valid,
        learning_rate_clipped_batches=0,
        skipped_batches=0,
        relative_error_condition_valid=(
            certificate.relative_error_condition_valid
        ),
        loss_descent_valid=None,
        loss_non_descent_batches=0,
        gradient_sq_norm=certificate.gradient_sq_norm,
        theory_descent_coefficient=certificate.theory_descent_coefficient,
        min_positive_learning_rate=learning_rate,
        relative_error=certificate.relative_error,
        selected_layer_index=None,
        layer_relative_errors=[],
        output_relative_error=certificate.output_relative_error,
        sensor_valid=certificate.sensor_valid,
        sensor_invalid_batches=certificate.sensor_invalid_batches,
    )
    return _certify_fgd_candidate(
        candidate_model=trial_model,
        epoch_result=epoch_result,
        certificate=certificate,
        validation_loader=validation_loader,
        device=device,
        config=config,
        theory_state=theory_state,
        initial_functional_gap=initial_functional_gap,
        theory_loss_star=theory_loss_star,
    )


def _evaluate_secant_fgd_trial(
    *,
    base_model: GrowingMLP,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    learning_rate: float,
    accuracy_tolerance: float,
    config: PipelineConfig,
    probe: tuple[torch.Tensor, torch.Tensor] | None,
    classification: bool,
    theory_state: _FGDTheoryState,
    initial_functional_gap: float,
    theory_loss_star: float,
) -> _FGDTrial:
    """Fit a finite Hilbert secant with the current fixed architecture."""
    trial_model = copy.deepcopy(base_model)
    base_model.eval()
    trial_model.train()
    base_parameters = {
        name: parameter.detach().clone()
        for name, parameter in base_model.named_parameters()
    }
    optimizer = torch.optim.Adam(
        trial_model.parameters(),
        lr=config.secant_fgd.inner_learning_rate,
    )
    numerical_failure = False

    for _ in range(max(1, config.secant_fgd.inner_steps)):
        for x, y in train_batches:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                base_output = base_model(x)
                residual = functional_gradient(
                    base_output, y, config.fgd_approx.functional_loss
                )
                functional_target = base_output - learning_rate * residual

            optimizer.zero_grad(set_to_none=True)
            candidate_output = trial_model(x)
            objective = torch.mean((candidate_output - functional_target) ** 2)
            if config.secant_fgd.parameter_penalty > 0.0:
                penalty = torch.zeros((), device=device)
                for name, parameter in trial_model.named_parameters():
                    penalty = penalty + torch.mean(
                        (parameter - base_parameters[name]) ** 2
                    )
                objective = (
                    objective
                    + config.secant_fgd.parameter_penalty * penalty
                )
            if not torch.isfinite(objective):
                numerical_failure = True
                break
            objective.backward()
            gradient_clip_norm = config.secant_fgd.gradient_clip_norm
            if gradient_clip_norm is not None and gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    trial_model.parameters(),
                    gradient_clip_norm,
                )
            optimizer.step()
        if numerical_failure:
            break

    train_metrics = evaluate_regression_metrics(
        trial_model,
        train_batches,
        loss_function,
        device=device,
        accuracy_tolerance=accuracy_tolerance,
        classification=classification,
    )
    epoch_result = FGDApproxEpochResult(
        train_loss=train_metrics.loss,
        train_accuracy=train_metrics.accuracy,
        test_loss=float("nan"),
        test_accuracy=float("nan"),
        learning_rate=learning_rate,
        next_learning_rate=None,
        learning_rate_upper_bound=None,
        learning_rate_interval_valid=None,
        learning_rate_clipped_batches=0,
        skipped_batches=int(numerical_failure),
        relative_error_condition_valid=None,
        loss_descent_valid=None,
        loss_non_descent_batches=0,
        gradient_sq_norm=None,
        theory_descent_coefficient=None,
        min_positive_learning_rate=(
            None if numerical_failure else learning_rate
        ),
        relative_error=None,
        selected_layer_index=None,
        layer_relative_errors=[],
        output_relative_error=None,
        sensor_valid=not numerical_failure,
        sensor_invalid_batches=int(numerical_failure),
    )
    certificate = evaluate_secant_validation_certificate(
        base_model=base_model,
        candidate_model=trial_model,
        data_loader=validation_loader,
        device=device,
        config=config.fgd_approx,
        learning_rate=learning_rate,
        probe=probe,
    )
    return _certify_fgd_candidate(
        candidate_model=trial_model,
        epoch_result=epoch_result,
        certificate=certificate,
        validation_loader=validation_loader,
        device=device,
        config=config,
        theory_state=theory_state,
        initial_functional_gap=initial_functional_gap,
        theory_loss_star=theory_loss_star,
    )


def _search_fgd_certified_trial(
    *,
    maximum_learning_rate: float,
    evaluate_trial: Callable[[float], _FGDTrial],
    config: FGDApproxConfig,
) -> _FGDSearchResult:
    """Return the numerically largest LR found to satisfy every condition."""
    trial_count = 0
    last_trial: _FGDTrial | None = None
    sensor_failure = False

    def sensor_valid(trial: _FGDTrial) -> bool:
        return (
            trial.epoch_result.sensor_valid
            and trial.epoch_result.skipped_batches == 0
            and trial.certificate.sensor_valid
        )

    def run(learning_rate: float) -> _FGDTrial:
        nonlocal trial_count, last_trial, sensor_failure
        trial = evaluate_trial(learning_rate)
        trial_count += 1
        last_trial = trial
        sensor_failure = sensor_failure or not sensor_valid(trial)
        return trial

    lower_interval_bound = config.theory_lr_min + config.eps
    if maximum_learning_rate <= lower_interval_bound:
        return _FGDSearchResult(None, None, 0, False)
    floor_factor = min(max(config.lr_min_factor, 0.0), 1.0)
    minimum = max(
        lower_interval_bound,
        maximum_learning_rate * floor_factor,
    )
    steps = max(1, config.theory_lr_search_steps)
    if abs(maximum_learning_rate - minimum) <= config.eps:
        steps = 1

    failed_above: float | None = None
    for index in range(steps):
        if steps == 1:
            candidate = maximum_learning_rate
        else:
            fraction = index / (steps - 1)
            candidate = maximum_learning_rate * (
                minimum / maximum_learning_rate
            ) ** fraction
        trial = run(candidate)
        if not sensor_valid(trial):
            return _FGDSearchResult(None, last_trial, trial_count, True)
        if not trial.all_conditions_valid:
            failed_above = candidate
            continue

        best_trial = trial
        lower_passing = candidate
        if failed_above is not None and failed_above > lower_passing:
            upper_failing = failed_above
            for _ in range(max(0, config.theory_lr_search_refinements)):
                midpoint = 0.5 * (lower_passing + upper_failing)
                midpoint_trial = run(midpoint)
                if not sensor_valid(midpoint_trial):
                    return _FGDSearchResult(None, last_trial, trial_count, True)
                if midpoint_trial.all_conditions_valid:
                    lower_passing = midpoint
                    best_trial = midpoint_trial
                else:
                    upper_failing = midpoint
        return _FGDSearchResult(
            accepted=best_trial,
            last_trial=last_trial,
            trial_count=trial_count,
            sensor_failure=sensor_failure,
        )

    return _FGDSearchResult(
        accepted=None,
        last_trial=last_trial,
        trial_count=trial_count,
        sensor_failure=sensor_failure,
    )


def _growth_certificate_improves(
    before: FGDValidationCertificate,
    after: FGDValidationCertificate,
    config: PipelineConfig,
) -> bool:
    """Return whether a trial growth expands the usable FGD certificate."""
    if not after.sensor_valid or after.relative_error is None:
        return False
    if not before.sensor_valid or before.relative_error is None:
        return True
    if (
        after.relative_error_condition_valid is True
        and before.relative_error_condition_valid is not True
    ):
        return True
    if (
        before.relative_error - after.relative_error
        >= config.secant_fgd.growth_min_relative_error_improvement
    ):
        return True

    after_learning_rate = after.max_valid_learning_rate
    before_learning_rate = before.max_valid_learning_rate
    if after_learning_rate is None:
        return False
    if before_learning_rate is None:
        return True
    required_learning_rate = before_learning_rate * (
        1.0 + config.secant_fgd.growth_min_learning_rate_improvement
    )
    return after_learning_rate >= required_learning_rate


def _growth_reduces_lookahead_epsilon(
    *,
    model: GrowingMLP,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    probe: tuple[torch.Tensor, torch.Tensor] | None,
    current_relative_error: float,
    device: torch.device,
    config: PipelineConfig,
) -> bool:
    """Generalised R1: is the structure inadequate despite eps < 1/2?

    Asks the only question that separates "this small net is enough" (MNIST)
    from "this net is rank-limited and grinding" (CIFAR): does adding
    capacity at the bottleneck reach a strictly lower relative error than the
    current structure holds?

    The bottleneck is the width-minimum growable location -- ``rank J <=
    min_l w_l``, so that is where added capacity can raise what the reachable
    set expresses. The candidate is grown function-preservingly (so the
    comparison isolates capacity, not a lucky re-initialisation) and trained
    for ``growth_lookahead_steps`` certified steps; its held-out eps is then
    compared to ``current_relative_error``. Growth is warranted only when the
    grown structure beats the current one by more than ``tiny_statistical_
    threshold`` in relative terms -- the same margin the truncation uses, so
    no new constant enters and numerical jitter cannot force endless growth.

    Returns True when growth is warranted. False reproduces today's stop, so
    with the flag off (the default) this is never called and behaviour is
    unchanged.
    """
    widths = [int(layer.in_features) for layer in model._growable_layers]
    limiting = rank_limiting_locations(widths)
    if not limiting:
        return False
    bottleneck = limiting[0]

    trial = copy.deepcopy(model)
    optimal_update_kwargs = tiny_optimal_update_kwargs(
        config.fgd_approx, compute_delta=config.fgd_approx.growth_compute_delta
    )
    try:
        grow_layer(
            model=trial,
            train_loader=train_loader,
            layer_index=bottleneck,
            device=device,
            line_search_config=config.scaling_line_search,
            optimal_update_kwargs=optimal_update_kwargs,
            progress=None,
            function_preserving=True,
            preservation_tolerance=config.fgd_approx.growth_preservation_tolerance,
        )
    except RuntimeError:
        return False

    trained = _train_parametric_gd_candidate(
        base_model=trial,
        train_batches=train_batches,
        device=device,
        functional_learning_rate=(
            config.parametric_descent.functional_learning_rates[0]
        ),
        steps=config.fgd_approx.growth_lookahead_steps,
        config=config.parametric_descent,
        functional_loss=config.fgd_approx.functional_loss,
    )
    if trained is None:
        return False

    grown = evaluate_fgd_validation_certificate(
        model=trained,
        data_loader=validation_loader,
        device=device,
        config=config.fgd_approx,
        learning_rate=None,
        probe=probe,
    )
    if grown.relative_error is None:
        return False
    margin = config.fgd_approx.tiny_statistical_threshold
    return grown.relative_error < current_relative_error * (1.0 - margin)


def _probe_fgd_growth(
    *,
    model: GrowingMLP,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: torch.utils.data.DataLoader,
    base_certificate: FGDValidationCertificate,
    selected_layer_index: int | None,
    growth_count: int,
    device: torch.device,
    config: PipelineConfig,
    probe: tuple[torch.Tensor, torch.Tensor] | None,
) -> _GrowthProbe | None:
    """Trial growth on clones and retain the best FGD certificate change."""
    growable_layers = getattr(model, "_growable_layers", None)
    if not growable_layers:
        return None
    if config.fgd_approx.layer_selection == "certifying":
        layer_indices = list(range(len(growable_layers)))
    else:
        layer_indices = [
            selected_layer_index
            if selected_layer_index is not None
            else layer_index_for_growth(
                growth_count=growth_count,
                number_hidden_layers=config.model.number_hidden_layers,
                config=config.growth_schedule,
            )
        ]

    _clear_inaccessible_tensor_caches(model)
    probes: list[_GrowthProbe] = []
    optimal_update_kwargs = tiny_optimal_update_kwargs(
        config.fgd_approx,
        compute_delta=config.fgd_approx.growth_compute_delta,
    )
    base_parameter_count = count_parameters(model)
    # Base functional loss for the measured-descent growth criterion. Only
    # paid when that selection is active (delta growth reduces the loss, so
    # ranking candidates by their certified descent-per-parameter is the
    # coherent, Prop. 3.8-justified choice; the rel-error certificate is
    # blind to it because the delta step jumps the linearization).
    select_by_descent = config.fgd_approx.growth_select_by_descent
    select_by_epsilon = (
        config.fgd_approx.growth_selection == "epsilon_lookahead"
    )
    base_functional_loss = (
        evaluate_functional_loss(
            model, validation_loader, device, config.fgd_approx.functional_loss
        )
        if select_by_descent
        else 0.0
    )
    for layer_index in layer_indices:
        trial_model = copy.deepcopy(model)
        growth_result = grow_layer(
            model=trial_model,
            train_loader=train_batches,
            layer_index=layer_index,
            device=device,
            line_search_config=config.scaling_line_search,
            optimal_update_kwargs=optimal_update_kwargs,
            progress=None,
            function_preserving=config.fgd_approx.growth_function_preserving,
            preservation_tolerance=(
                config.fgd_approx.growth_preservation_tolerance
            ),
            line_search_loader=(
                validation_loader
                if config.fgd_approx.growth_scaling_on_validation
                else None
            ),
        )
        certificate = evaluate_fgd_validation_certificate(
            model=trial_model,
            data_loader=validation_loader,
            device=device,
            config=config.fgd_approx,
            learning_rate=None,
            probe=probe,
        )
        epsilon_reduction = None
        if select_by_epsilon:
            # Use the new capacity before judging it: the delta perturbs the
            # function, so the IMMEDIATE eps is worse for every candidate.
            trained_clone = _train_parametric_gd_candidate(
                base_model=trial_model,
                train_batches=train_batches,
                device=device,
                functional_learning_rate=(
                    config.parametric_descent.functional_learning_rates[0]
                ),
                steps=config.fgd_approx.growth_lookahead_steps,
                config=config.parametric_descent,
                functional_loss=config.fgd_approx.functional_loss,
            )
            if trained_clone is not None:
                after = evaluate_fgd_validation_certificate(
                    model=trained_clone,
                    data_loader=validation_loader,
                    device=device,
                    config=config.fgd_approx,
                    learning_rate=None,
                    probe=probe,
                )
                if (
                    base_certificate.relative_error is not None
                    and after.relative_error is not None
                ):
                    epsilon_reduction = (
                        base_certificate.relative_error - after.relative_error
                    )
        functional_descent = 0.0
        if select_by_descent:
            candidate_loss = evaluate_functional_loss(
                trial_model,
                validation_loader,
                device,
                config.fgd_approx.functional_loss,
            )
            functional_descent = base_functional_loss - candidate_loss
        probes.append(
            _GrowthProbe(
                model=trial_model,
                result=growth_result,
                certificate=certificate,
                improves_fgd=_growth_certificate_improves(
                    base_certificate,
                    certificate,
                    config,
                ),
                functional_descent=functional_descent,
                added_parameters=(
                    count_parameters(trial_model) - base_parameter_count
                ),
                epsilon_reduction=epsilon_reduction,
                epsilon_before=base_certificate.relative_error,
            )
        )

    # Budget-aware growth: a candidate that would push the model past the
    # parameter budget is not affordable and is dropped BEFORE selection.
    # Checking the budget only before growing lets one expensive
    # input-layer widening (784 params/neuron) blow through it; filtering
    # post-growth counts steers growth to the parameter-efficient
    # narrow-in / wide-late shape automatically, because the input layer
    # becomes unaffordable early while the late layers stay cheap.
    max_parameters = config.fgd_approx.max_total_parameters
    if max_parameters is not None:
        affordable = [
            probe
            for probe in probes
            if base_parameter_count + probe.added_parameters <= max_parameters
        ]
        if affordable:
            probes = affordable

    if select_by_epsilon:
        # R3 -- termination: None here means no candidate enlarges what the
        # structure can express, i.e. it is already minimal-adequate. The
        # caller leaves the structure unchanged; there is NO fallback, because
        # growing anyway would spend parameters that buy no representability.
        return _select_growth_probe_by_epsilon(probes, config.fgd_approx.eps)

    if select_by_descent:
        by_descent = _select_growth_probe_by_descent(
            probes, config.fgd_approx.eps
        )
        if by_descent is not None:
            return by_descent
        # No growth realizes an immediate validation descent (every
        # single-neuron delta overfits the tiny structure). Rather than
        # stall, still add the capacity the certificate ranks best so the
        # structure can escape the bottleneck and train into it — growth is
        # the paper's structural response to an exhausted family.
    return _select_growth_probe(
        probes,
        prefer_lower_error=config.fgd_approx.growth_prefer_lower_error,
    )


def _select_growth_probe_by_epsilon(
    probes: list[_GrowthProbe],
    eps: float,
) -> _GrowthProbe | None:
    """Grow where a parameter most enlarges what the structure can express.

    Ranks by the look-ahead reduction of the Lemma-3.5 relative error per
    added parameter. A candidate that does not reduce eps did not enlarge
    the reachable set toward r, so it is not a growth worth paying for; if
    NO candidate reduces eps the structure is already minimal-adequate and
    None is returned, which is the termination condition of the search.
    """
    improving = [
        probe
        for probe in probes
        if probe.epsilon_reduction is not None and probe.epsilon_reduction > eps
    ]
    if not improving:
        return None

    def efficiency(probe: _GrowthProbe) -> tuple[float, float, int]:
        added = max(probe.added_parameters, 1)
        return (
            -(probe.epsilon_reduction or 0.0) / added,
            -(probe.epsilon_reduction or 0.0),
            probe.result.layer_index,
        )

    return min(improving, key=efficiency)


def _select_growth_probe_by_descent(
    probes: list[_GrowthProbe],
    eps: float,
) -> _GrowthProbe | None:
    """Grow the layer with the largest certified descent PER PARAMETER.

    This is the paper's structural step made parameter-efficient: among
    growths that realize a genuine validation functional descent (Prop. 3.8
    measured descent > 0), pick the one that reduces the functional gap most
    per added parameter. Ties fall to the larger absolute descent, then the
    lower layer index. If no growth descends, fall back to None (the caller
    then leaves the structure unchanged and keeps training).
    """
    descending = [p for p in probes if p.functional_descent > eps]
    if not descending:
        return None

    def efficiency(probe: _GrowthProbe) -> tuple[float, float, int]:
        added = max(probe.added_parameters, 1)
        return (
            -probe.functional_descent / added,
            -probe.functional_descent,
            probe.result.layer_index,
        )

    return min(descending, key=efficiency)


def _probe_relative_error(probe: _GrowthProbe) -> float:
    relative_error = probe.certificate.relative_error
    return relative_error if relative_error is not None else float("inf")


def _probe_parameter_count(probe: _GrowthProbe) -> int:
    return sum(parameter.numel() for parameter in probe.model.parameters())


def _select_growth_probe(
    probes: list[_GrowthProbe],
    prefer_lower_error: bool = False,
) -> _GrowthProbe | None:
    """Deterministic growth-layer choice.

    Among probes that improve the FGD certificate the default policy is
    frugal-first: fewest total parameters, then lowest post-growth relative
    error, then layer index. With ``prefer_lower_error`` the improving case
    instead ranks by lowest post-growth relative error first (parameter
    count as a tie-break), so growth widens the most impactful layer even
    when it is the expensive input layer. When NO probe reaches the
    improvement threshold the priority is always lowest post-growth relative
    error first — growth exists to restore approximation capacity, so a
    smaller architecture must never outrank a better certificate.
    """
    if not probes:
        return None
    improving = [probe for probe in probes if probe.improves_fgd]
    if improving:
        if prefer_lower_error:
            improving_key = lambda probe: (
                _probe_relative_error(probe),
                _probe_parameter_count(probe),
                probe.result.layer_index,
            )
        else:
            improving_key = lambda probe: (
                _probe_parameter_count(probe),
                _probe_relative_error(probe),
                probe.result.layer_index,
            )
        return min(improving, key=improving_key)
    return min(
        probes,
        key=lambda probe: (
            _probe_relative_error(probe),
            _probe_parameter_count(probe),
            probe.result.layer_index,
        ),
    )


def _search_secant_fgd_candidate(
    *,
    model: GrowingMLP,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    accuracy_tolerance: float,
    config: PipelineConfig,
    probe: tuple[torch.Tensor, torch.Tensor] | None = None,
    classification: bool,
    theory_state: _FGDTheoryState,
    initial_functional_gap: float,
    theory_loss_star: float,
) -> _FGDSearchResult:
    """Search realizable non-tangent Hilbert secants at fixed architecture."""
    if (
        not config.secant_fgd.enabled
        or config.secant_fgd.max_learning_rate <= config.fgd_approx.eps
    ):
        return _FGDSearchResult(None, None, 0, False)

    search_config = replace(
        config.fgd_approx,
        theory_lr_search_steps=config.secant_fgd.search_steps,
        theory_lr_search_refinements=0,
        lr_min_factor=config.secant_fgd.min_learning_rate_factor,
    )

    def evaluate_trial(learning_rate: float) -> _FGDTrial:
        return _evaluate_secant_fgd_trial(
            base_model=model,
            train_batches=train_batches,
            validation_loader=validation_loader,
            loss_function=loss_function,
            device=device,
            learning_rate=learning_rate,
            accuracy_tolerance=accuracy_tolerance,
            config=config,
            probe=probe,
            classification=classification,
            theory_state=theory_state,
            initial_functional_gap=initial_functional_gap,
            theory_loss_star=theory_loss_star,
        )

    return _search_fgd_certified_trial(
        maximum_learning_rate=config.secant_fgd.max_learning_rate,
        evaluate_trial=evaluate_trial,
        config=search_config,
    )


def _measure_secant_projection(
    *,
    base_model: GrowingMLP,
    candidate_model: GrowingMLP,
    validation_loader: torch.utils.data.DataLoader,
    device: torch.device,
    eps: float,
    functional_loss: str = "mse",
) -> tuple[float, float] | None:
    """Aggregate (cosine, eta*) of the realized output displacement.

    Delta = F(base) - F(candidate) is compared against the functional
    gradient r = 2(F(base) - Y) on validation. eta* = <Delta, r> / |r|^2 is
    the declared functional learning rate that minimizes the secant relative
    error; at eta* that error equals sqrt(1 - cos^2) exactly, so the cosine
    is the scale-invariant admissibility measure of the family.
    """
    base_model.eval()
    candidate_model.eval()
    dot = 0.0
    delta_sq = 0.0
    target_sq = 0.0
    with torch.no_grad():
        for x, y in validation_loader:
            x = x.to(device)
            y = y.to(device)
            base_output = base_model(x).to(torch.float64)
            candidate_output = candidate_model(x).to(torch.float64)
            target = functional_gradient(
                base_output, y.to(torch.float64), functional_loss
            )
            delta = base_output - candidate_output
            if not (
                torch.isfinite(delta).all() and torch.isfinite(target).all()
            ):
                return None
            dot += float(torch.sum(delta * target).item())
            delta_sq += float(torch.sum(delta * delta).item())
            target_sq += float(torch.sum(target * target).item())
    if delta_sq <= eps or target_sq <= eps:
        return None
    cosine = dot / math.sqrt(delta_sq * target_sq)
    eta_star = dot / target_sq
    return cosine, eta_star


def _train_parametric_gd_candidate(
    *,
    base_model: GrowingMLP,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    functional_learning_rate: float,
    steps: int,
    config: ParametricGDConfig | ParametricDescentConfig,
    functional_loss: str = "mse",
) -> GrowingMLP | None:
    """Train a disposable clone toward the functional target f - eta * r.

    With eta = 0.5 the target f - 0.5 * 2(f - y) is exactly y, so that
    nominal rate reproduces plain parametric loss descent.
    """
    trial_model = copy.deepcopy(base_model)
    base_model.eval()
    trial_model.train()
    base_parameters = {
        name: parameter.detach().clone()
        for name, parameter in base_model.named_parameters()
    }
    trainable = [
        parameter
        for parameter in trial_model.parameters()
        if parameter.requires_grad
    ]
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(trainable, lr=config.inner_learning_rate)
    elif config.optimizer == "adamw":
        # The certificate only sees the realized displacement, so the
        # candidate generator may be as strong as the dense baseline's
        # optimizer (decoupled weight decay for generalization).
        optimizer = torch.optim.AdamW(
            trainable,
            lr=config.inner_learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            trainable,
            lr=config.inner_learning_rate,
            weight_decay=config.weight_decay,
        )

    for _ in range(max(1, steps)):
        for x, y in train_batches:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                base_output = base_model(x)
                functional_target = (
                    base_output
                    - functional_learning_rate
                    * functional_gradient(base_output, y, functional_loss)
                )
            optimizer.zero_grad(set_to_none=True)
            candidate_output = trial_model(x)
            objective = torch.mean((candidate_output - functional_target) ** 2)
            if config.parameter_penalty > 0.0:
                penalty = torch.zeros((), device=device)
                for name, parameter in trial_model.named_parameters():
                    penalty = penalty + torch.mean(
                        (parameter - base_parameters[name]) ** 2
                    )
                objective = objective + config.parameter_penalty * penalty
            if not torch.isfinite(objective):
                return None
            objective.backward()
            if (
                config.gradient_clip_norm is not None
                and config.gradient_clip_norm > 0.0
            ):
                torch.nn.utils.clip_grad_norm_(
                    trial_model.parameters(),
                    config.gradient_clip_norm,
                )
            optimizer.step()
    trial_model.eval()
    return trial_model


def _evaluate_parametric_gd_trial(
    *,
    base_model: GrowingMLP,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    functional_learning_rate: float,
    steps: int,
    accuracy_tolerance: float,
    config: PipelineConfig,
    probe: tuple[torch.Tensor, torch.Tensor] | None = None,
    classification: bool,
    theory_state: _FGDTheoryState,
    initial_functional_gap: float,
    theory_loss_star: float,
) -> _FGDTrial | None:
    """One calibrated parametric-GD secant; None if the cosine screen fails.

    The declared learning rate is NOT the nominal target rate: it is the
    scale-optimal eta* measured from the realized displacement, which is the
    fix for the historical scale mismatch of this family. Candidates whose
    projection cosine falls below parametric_gd.min_cosine are discarded
    before certification, so a misaligned family can never reach the
    growth path with a corrupted certificate.
    """
    candidate = _train_parametric_gd_candidate(
        base_model=base_model,
        train_batches=train_batches,
        device=device,
        functional_learning_rate=functional_learning_rate,
        steps=steps,
        config=config.parametric_gd,
        functional_loss=config.fgd_approx.functional_loss,
    )
    if candidate is None:
        return None
    projection = _measure_secant_projection(
        base_model=base_model,
        candidate_model=candidate,
        validation_loader=validation_loader,
        device=device,
        eps=config.fgd_approx.eps,
        functional_loss=config.fgd_approx.functional_loss,
    )
    if projection is None:
        return None
    cosine, eta_star = projection
    if cosine < config.parametric_gd.min_cosine:
        return None
    if eta_star <= config.fgd_approx.theory_lr_min + config.fgd_approx.eps:
        return None

    train_metrics = evaluate_regression_metrics(
        candidate,
        train_batches,
        loss_function,
        device=device,
        accuracy_tolerance=accuracy_tolerance,
        classification=classification,
    )
    epoch_result = FGDApproxEpochResult(
        train_loss=train_metrics.loss,
        train_accuracy=train_metrics.accuracy,
        test_loss=float("nan"),
        test_accuracy=float("nan"),
        learning_rate=eta_star,
        next_learning_rate=None,
        learning_rate_upper_bound=None,
        learning_rate_interval_valid=None,
        learning_rate_clipped_batches=0,
        skipped_batches=0,
        relative_error_condition_valid=None,
        loss_descent_valid=None,
        loss_non_descent_batches=0,
        gradient_sq_norm=None,
        theory_descent_coefficient=None,
        min_positive_learning_rate=eta_star,
        relative_error=None,
        selected_layer_index=None,
        layer_relative_errors=[],
        output_relative_error=None,
        sensor_valid=True,
        sensor_invalid_batches=0,
    )
    certificate = evaluate_secant_validation_certificate(
        base_model=base_model,
        candidate_model=candidate,
        data_loader=validation_loader,
        device=device,
        config=config.fgd_approx,
        learning_rate=eta_star,
        probe=probe,
    )
    return _certify_fgd_candidate(
        candidate_model=candidate,
        epoch_result=epoch_result,
        certificate=certificate,
        validation_loader=validation_loader,
        device=device,
        config=config,
        theory_state=theory_state,
        initial_functional_gap=initial_functional_gap,
        theory_loss_star=theory_loss_star,
    )


@torch.no_grad()
def _functional_gradient_sq_norm(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    functional_loss: str,
) -> float:
    """Measure ||r||^2 = ||grad_f L||^2 directly on held-out data.

    Only sum-MSE admits the closed identity ||r||^2 = 4L. For every other
    certified functional the quantity Prop. 3.8 needs is measured rather
    than assumed.
    """
    model.eval()
    total = 0.0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        residual = functional_gradient(model(x), y, functional_loss)
        total += float(torch.sum(residual * residual).item())
    return total


def _certify_measured_descent_candidate(
    *,
    candidate_model: GrowingMLP,
    base_model: GrowingMLP,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    eta_star: float,
    relative_error: float | None,
    accuracy_tolerance: float,
    config: PipelineConfig,
    classification: bool,
    theory_state: _FGDTheoryState,
    initial_functional_gap: float,
    theory_loss_star: float,
) -> _FGDTrial:
    """Certify a candidate by its MEASURED functional descent (Prop. 3.8).

    For the empirical sum-MSE functional the function-space PL inequality is
    the exact identity |grad L|^2 = 4 L (theory_mu = 2, L* = 0), so the
    global-contraction argument of Proposition 3.8 needs only the per-step
    descent inequality L_{t+1} <= L_t - eta_t r_t |grad L_t|^2. Here that
    inequality holds with equality by construction: the descent coefficient
    r_t = (L_t - L_{t+1}) / (eta* |grad L_t|^2) is measured on validation.
    The relative-error route (Lemma 3.5 / Crel / LR interval) is a
    sufficient mechanism this family does not use; the measured relative
    error is stored for diagnostics only. Cprog, Cstat and Cglob use the
    same accumulators and algebra as every other family.
    """
    validation_loss_before = evaluate_functional_loss(
        base_model,
        validation_loader,
        device,
        config.fgd_approx.functional_loss,
    )
    validation_loss_after = evaluate_functional_loss(
        candidate_model,
        validation_loader,
        device,
        config.fgd_approx.functional_loss,
    )
    descent = validation_loss_before - validation_loss_after
    # ||r||^2 at the base model. For sum-MSE this is the exact identity
    # |grad L|^2 = |2(F-Y)|^2 = 4L; for any other functional (cross-entropy)
    # no such identity exists, so it is measured directly on the same
    # held-out data.
    functional_name = config.fgd_approx.functional_loss
    if functional_name == "mse":
        gradient_sq_norm = 4.0 * validation_loss_before
    else:
        gradient_sq_norm = _functional_gradient_sq_norm(
            base_model, validation_loader, device, functional_name
        )

    eps = config.fgd_approx.eps
    loss_descent_valid = (
        math.isfinite(validation_loss_after)
        and validation_loss_after
        <= theory_state.previous_validation_functional_loss + eps
        and descent > eps
    )
    progress = descent / max(gradient_sq_norm, eps)
    progress_valid = progress >= config.parametric_descent.min_progress
    descent_coefficient = (
        descent / max(eta_star * gradient_sq_norm, eps)
        if eta_star > 0.0
        else None
    )

    epoch_count = theory_state.epoch_count
    min_gradient_sq_norm = theory_state.min_gradient_sq_norm
    min_positive_learning_rate = theory_state.min_positive_learning_rate
    min_descent_coefficient = theory_state.min_descent_coefficient
    contraction_product = theory_state.global_contraction_product

    if descent_coefficient is not None and descent_coefficient > 0.0:
        epoch_count += 1
        min_gradient_sq_norm = (
            gradient_sq_norm
            if min_gradient_sq_norm is None
            else min(min_gradient_sq_norm, gradient_sq_norm)
        )
        min_positive_learning_rate = (
            eta_star
            if min_positive_learning_rate is None
            else min(min_positive_learning_rate, eta_star)
        )
        min_descent_coefficient = (
            descent_coefficient
            if min_descent_coefficient is None
            else min(min_descent_coefficient, descent_coefficient)
        )

    stationary_bound: float | None = None
    stationary_bound_valid: bool | None = None
    global_bound: float | None = None
    global_bound_valid: bool | None = None
    global_contraction: float | None = None
    if (
        epoch_count > 0
        and min_positive_learning_rate is not None
        and min_descent_coefficient is not None
        and min_positive_learning_rate > 0.0
        and min_descent_coefficient > 0.0
    ):
        stationary_bound = initial_functional_gap / (
            epoch_count
            * min_positive_learning_rate
            * min_descent_coefficient
        )
        # Trajectory diagnostics only (never acceptance gates), as in
        # _certify_fgd_candidate.
        stationary_bound_valid = (
            min_gradient_sq_norm is not None
            and min_gradient_sq_norm <= stationary_bound + eps
        )

        beta = config.fgd_approx.theory_beta
        mu = _certified_pl_constant(config)
        if descent_coefficient is not None and beta > 0 and mu > 0:
            # With the measured coefficient and the exact MSE constants this
            # contraction equals the realized loss ratio L_{t+1} / L_t.
            global_contraction = 1.0 - (
                2.0 * eta_star * mu * descent_coefficient / (beta**2)
            )
            contraction_product *= global_contraction
            global_bound = contraction_product * initial_functional_gap
            current_gap = max(validation_loss_after - theory_loss_star, 0.0)
            global_bound_valid = current_gap <= global_bound + eps

    train_metrics = evaluate_regression_metrics(
        candidate_model,
        train_batches,
        loss_function,
        device=device,
        accuracy_tolerance=accuracy_tolerance,
        classification=classification,
    )
    epoch_result = FGDApproxEpochResult(
        train_loss=train_metrics.loss,
        train_accuracy=train_metrics.accuracy,
        test_loss=float("nan"),
        test_accuracy=float("nan"),
        learning_rate=eta_star,
        next_learning_rate=None,
        learning_rate_upper_bound=None,
        learning_rate_interval_valid=None,
        learning_rate_clipped_batches=0,
        skipped_batches=0,
        relative_error_condition_valid=None,
        loss_descent_valid=loss_descent_valid,
        loss_non_descent_batches=0,
        gradient_sq_norm=gradient_sq_norm,
        theory_descent_coefficient=descent_coefficient,
        min_positive_learning_rate=eta_star,
        relative_error=relative_error,
        selected_layer_index=None,
        layer_relative_errors=[],
        output_relative_error=None,
        sensor_valid=True,
        sensor_invalid_batches=0,
    )
    certificate = FGDValidationCertificate(
        learning_rate_upper_bound=None,
        max_valid_learning_rate=None,
        learning_rate_interval_valid=None,
        skipped_batches=0,
        # Diagnostic only for this family: acceptance never gates on it.
        relative_error_condition_valid=None,
        gradient_sq_norm=gradient_sq_norm,
        theory_descent_coefficient=descent_coefficient,
        relative_error=relative_error,
        output_relative_error=None,
        sensor_valid=True,
        sensor_invalid_batches=0,
    )
    updated_state = _FGDTheoryState(
        epoch_count=epoch_count,
        min_gradient_sq_norm=min_gradient_sq_norm,
        min_positive_learning_rate=min_positive_learning_rate,
        min_descent_coefficient=min_descent_coefficient,
        global_contraction_product=contraction_product,
        previous_validation_functional_loss=validation_loss_after,
    )
    # LOCAL acceptance conditions for the measured-descent family: strict
    # realized descent, the Cprog floor and a positive measured coefficient
    # decide whether THIS step commits.
    all_conditions_valid = (
        loss_descent_valid
        and progress_valid
        and descent_coefficient is not None
        and descent_coefficient > 0.0
    )
    if not config.fgd_approx.local_acceptance_conditions:
        # Legacy mode: Cstat/Cglob also gate. Under
        # local_acceptance_conditions they are trajectory diagnostics only.
        all_conditions_valid = (
            all_conditions_valid
            and stationary_bound_valid is True
            and global_bound_valid is True
        )
    return _FGDTrial(
        model=candidate_model,
        epoch_result=epoch_result,
        certificate=certificate,
        theory_state=updated_state,
        validation_functional_loss=validation_loss_after,
        loss_descent_valid=loss_descent_valid,
        stationary_bound=stationary_bound,
        stationary_bound_valid=stationary_bound_valid,
        global_bound=global_bound,
        global_bound_valid=global_bound_valid,
        global_contraction=global_contraction,
        all_conditions_valid=all_conditions_valid,
    )


def _evaluate_parametric_descent_trial(
    *,
    base_model: GrowingMLP,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    functional_learning_rate: float,
    steps: int,
    accuracy_tolerance: float,
    config: PipelineConfig,
    classification: bool,
    theory_state: _FGDTheoryState,
    initial_functional_gap: float,
    theory_loss_star: float,
) -> _FGDTrial | None:
    """One measured-descent candidate; None if the direction screen fails."""
    descent_config = config.parametric_descent
    candidate = _train_parametric_gd_candidate(
        base_model=base_model,
        train_batches=train_batches,
        device=device,
        functional_learning_rate=functional_learning_rate,
        steps=steps,
        config=descent_config,
        functional_loss=config.fgd_approx.functional_loss,
    )
    if candidate is None:
        return None
    projection = _measure_secant_projection(
        base_model=base_model,
        candidate_model=candidate,
        validation_loader=validation_loader,
        device=device,
        eps=config.fgd_approx.eps,
        functional_loss=config.fgd_approx.functional_loss,
    )
    if projection is None:
        return None
    cosine, eta_star = projection
    if eta_star <= config.fgd_approx.eps:
        return None
    if cosine < descent_config.min_cosine:
        return None
    # Diagnostic secant relative error at eta*: exactly sqrt(1 - cos^2).
    relative_error = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return _certify_measured_descent_candidate(
        candidate_model=candidate,
        base_model=base_model,
        train_batches=train_batches,
        validation_loader=validation_loader,
        loss_function=loss_function,
        device=device,
        eta_star=eta_star,
        relative_error=relative_error,
        accuracy_tolerance=accuracy_tolerance,
        config=config,
        classification=classification,
        theory_state=theory_state,
        initial_functional_gap=initial_functional_gap,
        theory_loss_star=theory_loss_star,
    )


def _search_parametric_descent_candidate(
    *,
    base_model: GrowingMLP,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    accuracy_tolerance: float,
    config: PipelineConfig,
    classification: bool,
    theory_state: _FGDTheoryState,
    initial_functional_gap: float,
    theory_loss_star: float,
) -> _FGDSearchResult:
    """Search measured-descent candidates over the configured budgets.

    Every configured candidate is evaluated and the certified one with the
    LARGEST measured progress eta* r_t commits — any certified candidate is
    valid under Prop. 3.8, so this selection is free in the theory and it
    tightens the envelope fastest. First-accept semantics previously
    committed the first crumb even when a bigger certified step existed,
    which slowed the march toward structure exhaustion.
    """
    trial_count = 0
    last_trial: _FGDTrial | None = None
    best_trial: _FGDTrial | None = None
    best_progress = float("-inf")
    for functional_learning_rate in (
        config.parametric_descent.functional_learning_rates
    ):
        for steps in config.parametric_descent.inner_steps:
            trial = _evaluate_parametric_descent_trial(
                base_model=base_model,
                train_batches=train_batches,
                validation_loader=validation_loader,
                loss_function=loss_function,
                device=device,
                functional_learning_rate=functional_learning_rate,
                steps=steps,
                accuracy_tolerance=accuracy_tolerance,
                config=config,
                classification=classification,
                theory_state=theory_state,
                initial_functional_gap=initial_functional_gap,
                theory_loss_star=theory_loss_star,
            )
            trial_count += 1
            if trial is None:
                continue
            last_trial = trial
            if not trial.all_conditions_valid:
                continue
            progress = _certified_trial_progress(trial)
            if progress > best_progress:
                best_progress = progress
                best_trial = trial
    return _FGDSearchResult(
        best_trial,
        best_trial if best_trial is not None else last_trial,
        trial_count,
        False,
    )


def _family_rejection_active(
    rejected_at_step: int | None,
    accepted_outer_steps: int,
    cooldown: int,
) -> bool:
    """Whether a family rejection is still in effect.

    The cooldown counts ACCEPTED outer steps committed since the rejection:
    ordinary weight updates change the tangent space and the behavior of
    every family, so a rejection at theta_t must never be permanent while
    the architecture stays fixed. Growth clears rejection state entirely;
    cooldown <= 0 disables the memory.
    """
    if rejected_at_step is None or cooldown <= 0:
        return False
    return accepted_outer_steps - rejected_at_step < cooldown


def _certified_trial_progress(trial: _FGDTrial) -> float:
    """Certified progress eta r of a trial (the Cprog mass of the step)."""
    coefficient = trial.certificate.theory_descent_coefficient
    learning_rate = trial.epoch_result.learning_rate
    if coefficient is None or learning_rate is None:
        return 0.0
    return learning_rate * coefficient


def _search_parametric_gd_candidate(
    *,
    base_model: GrowingMLP,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    accuracy_tolerance: float,
    config: PipelineConfig,
    probe: tuple[torch.Tensor, torch.Tensor] | None = None,
    classification: bool,
    theory_state: _FGDTheoryState,
    initial_functional_gap: float,
    theory_loss_star: float,
) -> _FGDSearchResult:
    """Search calibrated parametric-GD secants over the configured budgets.

    Like the measured-descent search, every candidate is evaluated and the
    certified one with the largest certified progress eta* r commits.
    """
    trial_count = 0
    last_trial: _FGDTrial | None = None
    best_trial: _FGDTrial | None = None
    best_progress = float("-inf")
    for functional_learning_rate in (
        config.parametric_gd.functional_learning_rates
    ):
        for steps in config.parametric_gd.inner_steps:
            trial = _evaluate_parametric_gd_trial(
                base_model=base_model,
                train_batches=train_batches,
                validation_loader=validation_loader,
                loss_function=loss_function,
                device=device,
                functional_learning_rate=functional_learning_rate,
                steps=steps,
                accuracy_tolerance=accuracy_tolerance,
                config=config,
                probe=probe,
                classification=classification,
                theory_state=theory_state,
                initial_functional_gap=initial_functional_gap,
                theory_loss_star=theory_loss_star,
            )
            trial_count += 1
            if trial is None:
                continue
            last_trial = trial
            if not trial.all_conditions_valid:
                continue
            progress = _certified_trial_progress(trial)
            if progress > best_progress:
                best_progress = progress
                best_trial = trial
    return _FGDSearchResult(
        best_trial,
        best_trial if best_trial is not None else last_trial,
        trial_count,
        False,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def config_payload(config: PipelineConfig) -> dict[str, Any]:
    return _json_safe(asdict(config))


def _materialize_dataset(
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate one full pass of a loader into fixed design tensors."""
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in data_loader:
        xs.append(x.to(device))
        ys.append(y.to(device))
    if not xs:
        raise ValueError("Cannot materialize an empty data loader.")
    return torch.cat(xs), torch.cat(ys)


def _run_fgd_rkhs_pipeline(
    *,
    config: PipelineConfig,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    classification: bool,
    wandb_logger: Any,
    progress: ProgressFn | None,
) -> PipelineResult:
    """Standalone certified RKHS FGD training loop (third training method).

    Implements Algorithm 1 of arXiv:2606.16926 over the fixed kernel
    dictionary structure; see ``stable_tiny.fgd_rkhs`` for the theory
    mapping. The network structure is fixed for the whole run: no GroMo
    growth, no optimizer, no learning-rate schedule -- the learning rate is
    the certified constant of Proposition 3.8.
    """
    loss_function = torch.nn.MSELoss()
    train_x, train_y = _materialize_dataset(train_loader, device)
    trainer = FGDRKHSTrainer(
        train_x=train_x,
        train_y=train_y,
        config=config.fgd_rkhs,
        device=device,
    )
    model = trainer.model
    theory = trainer.theory

    if progress is not None:
        progress(f"Using device: {device}")
        progress(f"Training method: {config.training.method}")
        if wandb_logger.enabled:
            progress(
                f"W&B logging enabled: project={config.wandb.project}, "
                f"run={config.wandb.run_name or config.run.name}"
            )
        progress("Model (fixed structure):")
        progress(str(model))
        kernel_note = (
            f"gamma={theory.kernel_gamma:.6g}"
            if theory.kernel_kind == "gaussian"
            else f"feature_dim={theory.feature_dimension}"
        )
        progress(
            "[RKHS] certified constants: "
            f"n={theory.train_points}, kernel={theory.kernel_kind}, "
            f"{kernel_note}, "
            f"K_s={theory.smoothness:.6g}, alpha={theory.alpha:.3g}, "
            f"beta={theory.beta:.3g}, lambda_min={theory.kernel_lambda_min:.3e}, "
            f"mu={theory.pl_mu:.3e}, L*={theory.loss_star:.6e}, "
            f"eps_bar={theory.epsilon_bar:.4f}, "
            f"lr={theory.learning_rate:.6g} "
            f"(< bound {theory.learning_rate_upper_bound:.6g}), "
            f"r={theory.descent_coefficient:.6g}, "
            f"contraction={theory.contraction:.12g}, "
            f"PL_certificate={'valid' if theory.pl_certificate_valid else 'vacuous'}"
        )
        if not theory.pl_certificate_valid:
            progress(
                "[RKHS] warning: the smallest Gram eigenvalue is numerically "
                "zero, so the global-optimality envelope of Prop. 3.8 is "
                "vacuous for this structure. Descent and stationary-point "
                "certificates (Lemma 3.5, Prop. 3.6) still hold. Increase "
                "fgd_rkhs.kernel_gamma (Gaussian kernel), change "
                "fgd_rkhs.feature_seed (linear kernel), or deduplicate "
                "inputs to obtain a non-trivial PL constant."
            )

    def metrics(loader: torch.utils.data.DataLoader):
        return evaluate_regression_metrics(
            model,
            loader,
            loss_function,
            device=device,
            accuracy_tolerance=config.training.accuracy_tolerance,
            classification=classification,
        )

    history: list[HistoryEntry] = []
    train_metrics = metrics(train_loader)
    validation_metrics = metrics(validation_loader)
    test_metrics = metrics(test_loader)
    init_entry = HistoryEntry(
        step=0,
        step_type="INIT",
        train_loss=train_metrics.loss,
        validation_loss=validation_metrics.loss,
        test_loss=test_metrics.loss,
        train_accuracy=train_metrics.accuracy,
        validation_accuracy=validation_metrics.accuracy,
        test_accuracy=test_metrics.accuracy,
        learning_rate=theory.learning_rate,
        num_params=count_parameters(model),
        fgd_approximation_kind="rkhs_dictionary",
        fgd_rkhs_functional_loss=theory.initial_loss,
    )
    history.append(init_entry)
    wandb_logger.log_history_entry(init_entry)
    if progress is not None:
        progress(
            f"[INIT] Epoch 0, train_loss={train_metrics.loss:.4f}, "
            f"validation_loss={validation_metrics.loss:.4f}, "
            f"test_loss={test_metrics.loss:.4f}, "
            f"train_acc={train_metrics.accuracy:.3f}, "
            f"validation_acc={validation_metrics.accuracy:.3f}, "
            f"test_acc={test_metrics.accuracy:.3f}"
        )

    last_test_loss = test_metrics.loss
    for epoch in range(1, config.training.epochs + 1):
        epoch_result = trainer.run_epoch()
        last_record = epoch_result.step_records[-1]
        train_metrics = metrics(train_loader)
        validation_metrics = metrics(validation_loader)
        test_metrics = metrics(test_loader)
        epoch_entry = HistoryEntry(
            step=epoch,
            step_type="RKHS",
            train_loss=train_metrics.loss,
            validation_loss=validation_metrics.loss,
            test_loss=test_metrics.loss,
            train_accuracy=train_metrics.accuracy,
            validation_accuracy=validation_metrics.accuracy,
            test_accuracy=test_metrics.accuracy,
            learning_rate=theory.learning_rate,
            num_params=count_parameters(model),
            rel_error=last_record.relative_error,
            fgd_learning_rate_upper_bound=theory.learning_rate_upper_bound,
            fgd_learning_rate_interval_valid=True,
            fgd_relative_error_condition_valid=(
                last_record.relative_error_condition_valid
            ),
            fgd_loss_descent_valid=all(
                record.descent_valid for record in epoch_result.step_records
            ),
            fgd_gradient_sq_norm=last_record.gradient_sq_norm,
            fgd_theory_descent_coefficient=theory.descent_coefficient,
            fgd_global_bound=epoch_result.global_bound,
            fgd_global_bound_valid=epoch_result.global_bound_valid,
            fgd_global_contraction=theory.contraction,
            fgd_approximation_kind="rkhs_dictionary",
            fgd_rkhs_dictionary_size=last_record.dictionary_size,
            fgd_rkhs_functional_loss=epoch_result.train_functional_loss,
        )
        history.append(epoch_entry)
        wandb_logger.log_history_entry(epoch_entry)

        if progress is not None and should_log_epoch(epoch, config):
            delta = test_metrics.loss - last_test_loss
            bound_msg = (
                f", global_bound={epoch_result.global_bound:.4e}"
                f" ({'ok' if epoch_result.global_bound_valid else 'VIOLATED'})"
                if epoch_result.global_bound is not None
                else ""
            )
            progress(
                f"[RKHS] Epoch {epoch}, "
                f"train_loss={train_metrics.loss:.4f}, "
                f"validation_loss={validation_metrics.loss:.4f}, "
                f"test_loss={test_metrics.loss:.4f} ({delta:+.4f}), "
                f"train_acc={train_metrics.accuracy:.3f}, "
                f"validation_acc={validation_metrics.accuracy:.3f}, "
                f"test_acc={test_metrics.accuracy:.3f}, "
                f"functional_loss={epoch_result.train_functional_loss:.4e}, "
                f"rel_err={last_record.relative_error:.4f}, "
                f"dict={last_record.dictionary_size}/{theory.train_points}"
                f"{bound_msg}"
            )
        last_test_loss = test_metrics.loss

        if epoch_result.global_bound_valid is False and progress is not None:
            progress(
                "[RKHS] warning: measured loss exceeded the Prop. 3.8 "
                "envelope; this indicates a numerical-precision issue."
            )
        if epoch_result.converged:
            if progress is not None:
                progress(
                    f"[RKHS] converged at epoch {epoch}: functional gradient "
                    "norm is numerically zero, so by the PL condition "
                    "(Assumption 3.7) the iterate is a global minimizer."
                )
            break

    return PipelineResult(
        config=config,
        history=history,
        growth_events=[],
        model=model,
        device=str(device),
    )


def _frozen_feature_map_from_grown_mlp(mlp: GrowingMLP) -> FrozenAffineFeatureMap:
    """Snapshot the hidden layers of a grown MLP as the frozen feature map.

    The constant-1 feature is appended so the certified head is exactly an
    affine output layer (weight + bias): the certified optimum is the true
    global optimum of the donor network's output layer given its current
    hidden weights.
    """
    weights: list[torch.Tensor] = []
    biases: list[torch.Tensor | None] = []
    activations: list[torch.nn.Module] = []
    for module in list(mlp.layers)[:-1]:
        linear = module.layer
        weights.append(linear.weight.detach().clone())
        biases.append(
            linear.bias.detach().clone() if linear.bias is not None else None
        )
        activations.append(copy.deepcopy(module.post_layer_function))
    return FrozenAffineFeatureMap(weights, biases, activations, append_one=True)


def _apply_certified_head(mlp: GrowingMLP, kernel_model: KernelDictionaryModel) -> None:
    """Write the certified-optimal head into the grown network's output layer."""
    head = kernel_model.linear_head_weight()
    output_layer = mlp.layers[-1].layer
    hidden_width = output_layer.weight.shape[1]
    with torch.no_grad():
        output_layer.weight.copy_(
            head[:hidden_width].T.to(
                dtype=output_layer.weight.dtype,
                device=output_layer.weight.device,
            )
        )
        if output_layer.bias is not None:
            if head.shape[0] > hidden_width:
                output_layer.bias.copy_(
                    head[hidden_width].to(
                        dtype=output_layer.bias.dtype,
                        device=output_layer.bias.device,
                    )
                )
            else:
                output_layer.bias.zero_()


def _select_rkhs_growth_layer(
    mlp: GrowingMLP,
    growth_count: int,
    config: PipelineConfig,
) -> int | None:
    """Next growable layer, skipping hidden blocks at the width cap."""
    growable = getattr(mlp, "_growable_layers", None)
    if not growable:
        return None
    preferred = layer_index_for_growth(
        growth_count=growth_count,
        number_hidden_layers=config.model.number_hidden_layers,
        config=config.growth_schedule,
    ) % len(growable)
    cap = config.fgd_rkhs.growth_max_hidden_size
    for offset in range(len(growable)):
        index = (preferred + offset) % len(growable)
        # Growing growable layer ``index`` widens hidden block ``index``,
        # whose current width is the output size of layers[index].
        width = mlp.layers[index].layer.out_features
        if cap is None or width < cap:
            return index
    return None


def _run_fgd_rkhs_grow_pipeline(
    *,
    config: PipelineConfig,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    classification: bool,
    wandb_logger: Any,
    progress: ProgressFn | None,
) -> PipelineResult:
    """Certified train-and-grow cycle (training.method: fgd_rkhs_grow).

    Each cycle freezes the grown network's hidden layers as the fixed
    structure, trains the output layer to the certified global optimum of
    that structure (Algorithm 1 of arXiv:2606.16926 with exact constants),
    writes the optimal head back into the network, and then grows one GroMo
    layer. The cycle stops when the closed-form ceiling ``L*`` of the newly
    grown structure stops improving (relative improvement below
    ``fgd_rkhs.growth_min_ceiling_improvement``), when every hidden block
    reached ``fgd_rkhs.growth_max_hidden_size``, or after
    ``fgd_rkhs.growth_max_cycles`` growth events. The certificate is
    conditional: it certifies the best possible output layer for the hidden
    weights the growth produced, never the nonconvex full-weight optimum.
    """
    rkhs_config = config.fgd_rkhs
    if rkhs_config.growth_max_cycles < 0:
        raise ValueError("fgd_rkhs.growth_max_cycles must be >= 0.")
    if rkhs_config.growth_epochs_per_cycle < 1:
        raise ValueError("fgd_rkhs.growth_epochs_per_cycle must be >= 1.")
    if rkhs_config.growth_min_ceiling_improvement < 0.0:
        raise ValueError(
            "fgd_rkhs.growth_min_ceiling_improvement must be >= 0."
        )
    if (
        rkhs_config.growth_max_hidden_size is not None
        and rkhs_config.growth_max_hidden_size < 1
    ):
        raise ValueError("fgd_rkhs.growth_max_hidden_size must be >= 1.")

    loss_function = torch.nn.MSELoss()
    torch.manual_seed(config.model.model_seed)
    mlp = GrowingMLP(
        in_features=config.data.in_features,
        out_features=config.data.out_features,
        hidden_size=config.model.hidden_size,
        number_hidden_layers=config.model.number_hidden_layers,
        device=device,
    )
    train_x, train_y = _materialize_dataset(train_loader, device)

    def metrics(loader: torch.utils.data.DataLoader):
        return evaluate_regression_metrics(
            mlp,
            loader,
            loss_function,
            device=device,
            accuracy_tolerance=config.training.accuracy_tolerance,
            classification=classification,
        )

    def hidden_widths() -> list[int]:
        return [module.layer.out_features for module in list(mlp.layers)[:-1]]

    if progress is not None:
        progress(f"Using device: {device}")
        progress(f"Training method: {config.training.method}")
        if wandb_logger.enabled:
            progress(
                f"W&B logging enabled: project={config.wandb.project}, "
                f"run={config.wandb.run_name or config.run.name}"
            )
        progress("Original model:")
        progress(str(mlp))

    history: list[HistoryEntry] = []
    growth_events: list[GrowthResult] = []
    train_metrics = metrics(train_loader)
    validation_metrics = metrics(validation_loader)
    test_metrics = metrics(test_loader)
    init_entry = HistoryEntry(
        step=0,
        step_type="INIT",
        train_loss=train_metrics.loss,
        validation_loss=validation_metrics.loss,
        test_loss=test_metrics.loss,
        train_accuracy=train_metrics.accuracy,
        validation_accuracy=validation_metrics.accuracy,
        test_accuracy=test_metrics.accuracy,
        learning_rate=0.0,
        num_params=count_parameters(mlp),
        fgd_approximation_kind="rkhs_grown_head",
    )
    history.append(init_entry)
    wandb_logger.log_history_entry(init_entry)

    epoch = 0
    growth_count = 0
    previous_ceiling: float | None = None
    stop_growing = False
    last_test_loss = test_metrics.loss

    while True:
        feature_map = _frozen_feature_map_from_grown_mlp(mlp)
        trainer = FGDRKHSTrainer(
            train_x=train_x,
            train_y=train_y,
            config=rkhs_config,
            device=device,
            feature_map=feature_map,
        )
        theory = trainer.theory
        ceiling = theory.loss_star
        if progress is not None:
            progress(
                f"[RKHS-GROW] cycle {growth_count}: structure "
                f"{config.data.in_features}->"
                f"{'->'.join(str(w) for w in hidden_widths())}->"
                f"{config.data.out_features} (hidden frozen), "
                f"certified ceiling L*={ceiling:.6e}, "
                f"K_s={theory.smoothness:.6g}, mu={theory.pl_mu:.3e}, "
                f"lr={theory.learning_rate:.6g}, "
                f"contraction={theory.contraction:.6g}, "
                f"PL_certificate="
                f"{'valid' if theory.pl_certificate_valid else 'vacuous'}"
            )
        if previous_ceiling is not None:
            improvement = (previous_ceiling - ceiling) / max(
                previous_ceiling,
                rkhs_config.eps,
            )
            if progress is not None:
                progress(
                    f"[RKHS-GROW] ceiling improvement after growth: "
                    f"{improvement:+.4%} "
                    f"(threshold {rkhs_config.growth_min_ceiling_improvement:.4%})"
                )
            if improvement < rkhs_config.growth_min_ceiling_improvement:
                stop_growing = True
                if progress is not None:
                    progress(
                        "[RKHS-GROW] growth no longer improves the certified "
                        "ceiling; this is the final structure."
                    )
        previous_ceiling = ceiling

        converged = False
        for _ in range(rkhs_config.growth_epochs_per_cycle):
            if epoch >= config.training.epochs:
                break
            epoch += 1
            epoch_result = trainer.run_epoch()
            last_record = epoch_result.step_records[-1]
            _apply_certified_head(mlp, trainer.model)
            train_metrics = metrics(train_loader)
            validation_metrics = metrics(validation_loader)
            test_metrics = metrics(test_loader)
            epoch_entry = HistoryEntry(
                step=epoch,
                step_type="RKHS",
                train_loss=train_metrics.loss,
                validation_loss=validation_metrics.loss,
                test_loss=test_metrics.loss,
                train_accuracy=train_metrics.accuracy,
                validation_accuracy=validation_metrics.accuracy,
                test_accuracy=test_metrics.accuracy,
                learning_rate=theory.learning_rate,
                num_params=count_parameters(mlp),
                rel_error=last_record.relative_error,
                fgd_learning_rate_upper_bound=theory.learning_rate_upper_bound,
                fgd_learning_rate_interval_valid=True,
                fgd_relative_error_condition_valid=(
                    last_record.relative_error_condition_valid
                ),
                fgd_loss_descent_valid=all(
                    record.descent_valid for record in epoch_result.step_records
                ),
                fgd_gradient_sq_norm=last_record.gradient_sq_norm,
                fgd_theory_descent_coefficient=theory.descent_coefficient,
                fgd_global_bound=epoch_result.global_bound,
                fgd_global_bound_valid=epoch_result.global_bound_valid,
                fgd_global_contraction=theory.contraction,
                fgd_approximation_kind="rkhs_grown_head",
                fgd_rkhs_dictionary_size=last_record.dictionary_size,
                fgd_rkhs_functional_loss=epoch_result.train_functional_loss,
                fgd_rkhs_loss_star=ceiling,
            )
            history.append(epoch_entry)
            wandb_logger.log_history_entry(epoch_entry)
            if progress is not None and should_log_epoch(epoch, config):
                delta = test_metrics.loss - last_test_loss
                progress(
                    f"[RKHS-GROW] Epoch {epoch}, "
                    f"train_loss={train_metrics.loss:.4f}, "
                    f"validation_loss={validation_metrics.loss:.4f}, "
                    f"test_loss={test_metrics.loss:.4f} ({delta:+.4f}), "
                    f"train_acc={train_metrics.accuracy:.3f}, "
                    f"validation_acc={validation_metrics.accuracy:.3f}, "
                    f"test_acc={test_metrics.accuracy:.3f}, "
                    f"functional_loss="
                    f"{epoch_result.train_functional_loss:.4e}, "
                    f"ceiling={ceiling:.4e}, "
                    f"rel_err={last_record.relative_error:.4f}"
                )
            last_test_loss = test_metrics.loss
            if epoch_result.converged:
                converged = True
                if progress is not None:
                    progress(
                        f"[RKHS-GROW] cycle {growth_count} reached the "
                        "certified global optimum of the current fixed "
                        "structure (functional gradient numerically zero)."
                    )
                break
        if not converged and progress is not None:
            progress(
                f"[RKHS-GROW] cycle {growth_count} epoch budget reached "
                "before certified convergence; growing anyway."
            )

        if (
            stop_growing
            or growth_count >= rkhs_config.growth_max_cycles
            or epoch >= config.training.epochs
        ):
            break
        layer_index = _select_rkhs_growth_layer(mlp, growth_count, config)
        if layer_index is None:
            if progress is not None:
                progress(
                    "[RKHS-GROW] every hidden block reached "
                    f"growth_max_hidden_size="
                    f"{rkhs_config.growth_max_hidden_size}; stopping growth."
                )
            break
        growth_result = grow_layer(
            model=mlp,
            train_loader=train_loader,
            layer_index=layer_index,
            device=device,
            line_search_config=config.scaling_line_search,
            optimal_update_kwargs=None,
            progress=None,
        )
        growth_count += 1
        growth_events.append(growth_result)
        wandb_logger.log_growth_event(
            event=growth_result,
            epoch=epoch,
            growth_count=growth_count,
        )
        train_metrics = metrics(train_loader)
        validation_metrics = metrics(validation_loader)
        test_metrics = metrics(test_loader)
        growth_entry = HistoryEntry(
            step=epoch,
            step_type="GRO",
            train_loss=train_metrics.loss,
            validation_loss=validation_metrics.loss,
            test_loss=test_metrics.loss,
            train_accuracy=train_metrics.accuracy,
            validation_accuracy=validation_metrics.accuracy,
            test_accuracy=test_metrics.accuracy,
            learning_rate=0.0,
            num_params=count_parameters(mlp),
            layer_index=layer_index,
            scaling_factor=growth_result.best_scaling_factor,
            fgd_approximation_kind="rkhs_grown_head",
            fgd_rkhs_loss_star=ceiling,
        )
        history.append(growth_entry)
        wandb_logger.log_history_entry(growth_entry)
        if progress is not None:
            progress(
                f"[RKHS-GROW] growth {growth_count}: layer {layer_index}, "
                f"widths={hidden_widths()}, "
                f"params={count_parameters(mlp)}"
            )

    return PipelineResult(
        config=config,
        history=history,
        growth_events=growth_events,
        model=mlp,
        device=str(device),
    )


@dataclass(frozen=True)
class _RKHSPhaseResult:
    """Outcome of one certified head-optimization phase (secant replacement)."""

    trainer: FGDRKHSTrainer
    steps: int
    accepted: bool
    converged: bool
    model_loss_before: float
    functional_loss_after: float
    last_record: FGDRKHSStepRecord | None
    descent_valid: bool
    global_bound: float | None
    global_bound_valid: bool | None


def _run_rkhs_head_phase(
    *,
    model: GrowingMLP,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    config: PipelineConfig,
    device: torch.device,
) -> _RKHSPhaseResult:
    """Certified head optimization of the current fixed structure.

    Replaces the Hilbert-secant search of the original flow: when the
    tangent-space approximation stops certifying and a growth probe does
    not improve the certificate, the network's hidden layers are frozen as
    the fixed structure and the output layer is driven to the certified
    global optimum of that structure (Algorithm 1 of arXiv:2606.16926 with
    exact constants; see ``stable_tiny.fgd_rkhs``). The phase is accepted
    iff it improves the model's functional train loss beyond the numerical
    certificate tolerance; a rejection therefore certifies that the output
    layer is already at the global optimum of the fixed structure, i.e.
    the architecture is exhausted and only growth can help. The model is
    NOT modified here; the caller applies the certified head only on
    acceptance.
    """
    train_x = torch.cat([x for x, _ in train_batches]).to(device)
    train_y = torch.cat([y for _, y in train_batches]).to(device)
    feature_map = _frozen_feature_map_from_grown_mlp(model)
    # The phase certifies the output layer of the fixed structure, so the
    # kernel is always the linear one over the frozen hidden activations
    # regardless of how fgd_rkhs is configured for the standalone methods.
    rkhs_config = replace(
        config.fgd_rkhs,
        kernel="linear",
        feature_hidden_layers=0,
        feature_hidden_size=0,
    )
    trainer = FGDRKHSTrainer(
        train_x=train_x,
        train_y=train_y,
        config=rkhs_config,
        device=device,
        feature_map=feature_map,
    )
    with torch.no_grad():
        predictions = model(trainer.train_x.to(torch.float32)).to(torch.float64)
        residual = predictions - trainer.train_y
        model_loss_before = float(residual.square().sum().item()) / (
            2.0 * residual.shape[0]
        )

    epoch_results: list[FGDRKHSEpochResult] = []
    for _ in range(max(1, config.fgd_rkhs.growth_epochs_per_cycle)):
        epoch_result = trainer.run_epoch()
        epoch_results.append(epoch_result)
        if epoch_result.converged:
            break
    final_loss = epoch_results[-1].train_functional_loss
    last_record = (
        epoch_results[-1].step_records[-1]
        if epoch_results[-1].step_records
        else None
    )
    tolerance = trainer.certificate_tolerance * (1.0 + abs(model_loss_before))
    accepted = final_loss < model_loss_before - tolerance
    descent_valid = all(
        record.descent_valid
        for result in epoch_results
        for record in result.step_records
    )
    return _RKHSPhaseResult(
        trainer=trainer,
        steps=trainer.total_steps,
        accepted=accepted,
        converged=trainer.converged,
        model_loss_before=model_loss_before,
        functional_loss_after=final_loss,
        last_record=last_record,
        descent_valid=descent_valid,
        global_bound=epoch_results[-1].global_bound,
        global_bound_valid=epoch_results[-1].global_bound_valid,
    )


def run_pipeline(
    config: PipelineConfig,
    progress: ProgressFn | None = print,
) -> PipelineResult:
    """Run the train-grow loop from the GroMo tutorial."""
    wandb_logger = build_wandb_logger(config.wandb)
    wandb_logger.start(
        run_name=config.run.name,
        config_payload=config_payload(config),
    )
    device = select_device(config.training.device)
    train_loader, validation_loader, test_loader = build_dataloaders(config, device)
    classification = is_classification_task(config)
    if config.training.method in ("fgd_rkhs", "fgd_rkhs_grow"):
        runner = (
            _run_fgd_rkhs_pipeline
            if config.training.method == "fgd_rkhs"
            else _run_fgd_rkhs_grow_pipeline
        )
        try:
            result = runner(
                config=config,
                device=device,
                train_loader=train_loader,
                validation_loader=validation_loader,
                test_loader=test_loader,
                classification=classification,
                wandb_logger=wandb_logger,
                progress=progress,
            )
        except Exception:
            wandb_logger.abort()
            raise
        wandb_logger.finish(history=result.history)
        return result
    model = build_model(config, device)
    loss_function = torch.nn.MSELoss()
    optimizer = build_optimizer(model, config.optimizer)
    lr_cycle_start_epoch = 0
    current_fgd_learning_rate = config.fgd_approx.theory_lr_initial
    initial_learning_rate = (
        current_fgd_learning_rate
        if (
            config.training.method == "fgd_approx"
            and config.fgd_approx.learning_rate_policy == "theory_interval"
        )
        else scheduled_learning_rate(
            config,
            epoch=0,
            cycle_start_epoch=lr_cycle_start_epoch,
        )
    )
    apply_learning_rate(optimizer, initial_learning_rate)

    history: list[HistoryEntry] = []
    growth_events: list[GrowthResult] = []

    if progress is not None:
        progress(f"Using device: {device}")
        progress(f"Training method: {config.training.method}")
        if wandb_logger.enabled:
            progress(
                f"W&B logging enabled: project={config.wandb.project}, "
                f"run={config.wandb.run_name or config.run.name}"
            )
        progress("Original model:")
        progress(str(model))

    try:
        train_metrics = evaluate_regression_metrics(
            model,
            train_loader,
            loss_function,
            device=device,
            accuracy_tolerance=config.training.accuracy_tolerance,
            classification=classification,
        )
        validation_metrics = evaluate_regression_metrics(
            model,
            validation_loader,
            loss_function,
            device=device,
            accuracy_tolerance=config.training.accuracy_tolerance,
            classification=classification,
        )
        test_metrics = evaluate_regression_metrics(
            model,
            test_loader,
            loss_function,
            device=device,
            accuracy_tolerance=config.training.accuracy_tolerance,
            classification=classification,
        )
        last_test_loss = test_metrics.loss
        initial_functional_loss = evaluate_functional_loss(
            model,
            validation_loader,
            device,
            config.fgd_approx.functional_loss,
        )
        previous_validation_functional_loss = initial_functional_loss
        theory_loss_star = config.fgd_approx.theory_loss_star
        initial_functional_gap = max(initial_functional_loss - theory_loss_star, 0.0)
        fgd_epoch_count = 0
        fgd_min_gradient_sq_norm: float | None = None
        fgd_min_positive_learning_rate: float | None = None
        fgd_min_descent_coefficient: float | None = None
        fgd_global_contraction_product = 1.0
        fgd_previous_train_loss: float | None = None
        fgd_stalled_epochs = 0
        # A rejected fallback family is skipped for a COOLDOWN of accepted
        # outer steps (family_rejection_cooldown), not forever: weight
        # updates change the tangent space, so the same architecture can
        # re-admit a family once the parameters have moved. Growth clears
        # all rejection state immediately.
        fgd_accepted_outer_steps = 0
        family_rejection_step: dict[str, int] = {}
        family_rejection_cooldown = config.fgd_approx.family_rejection_cooldown
        # Consecutive epochs with no committed step from any family; the
        # growth probe waits for fgd_approx.growth_patience of them.
        fgd_epochs_without_commit = 0
        # FIXED certification probes, materialized once for the whole run:
        # every certificate, family comparison and growth-layer trial solves
        # one joint shared-direction projection over the same sample.
        fgd_validation_probe: tuple[torch.Tensor, torch.Tensor] | None = None
        fgd_train_probe: tuple[torch.Tensor, torch.Tensor] | None = None
        if config.training.method == "fgd_approx":
            fgd_validation_probe = build_projection_probe(
                validation_loader,
                config.fgd_approx.probe_batches,
                device,
            )
            fgd_train_probe = build_projection_probe(
                train_loader,
                config.fgd_approx.probe_batches,
                device,
            )
        validation_certificate_for_next_epoch = None
        if (
            config.training.method == "fgd_approx"
            and config.fgd_approx.learning_rate_policy == "theory_interval"
            and config.fgd_approx.projection_solver != "gromo_layer"
        ):
            validation_certificate_for_next_epoch = (
                evaluate_fgd_validation_certificate(
                    model=model,
                    data_loader=validation_loader,
                    device=device,
                    config=config.fgd_approx,
                    learning_rate=current_fgd_learning_rate,
                    probe=fgd_validation_probe,
                )
            )

        def reset_fgd_certificate() -> None:
            """Re-anchor the per-mode FGD bounds at the current loss."""
            nonlocal initial_functional_gap, fgd_epoch_count
            nonlocal previous_validation_functional_loss
            nonlocal fgd_min_gradient_sq_norm
            nonlocal fgd_min_positive_learning_rate
            nonlocal fgd_min_descent_coefficient
            nonlocal fgd_global_contraction_product
            nonlocal fgd_previous_train_loss, fgd_stalled_epochs
            nonlocal validation_certificate_for_next_epoch
            previous_validation_functional_loss = evaluate_functional_loss(
                model,
                validation_loader,
                device,
                config.fgd_approx.functional_loss,
            )
            initial_functional_gap = max(
                previous_validation_functional_loss - theory_loss_star,
                0.0,
            )
            fgd_epoch_count = 0
            fgd_min_gradient_sq_norm = None
            fgd_min_positive_learning_rate = None
            fgd_min_descent_coefficient = None
            fgd_global_contraction_product = 1.0
            fgd_previous_train_loss = None
            fgd_stalled_epochs = 0
            validation_certificate_for_next_epoch = None
        init_entry = HistoryEntry(
            step=0,
            step_type="INIT",
            train_loss=train_metrics.loss,
            validation_loss=validation_metrics.loss,
            test_loss=test_metrics.loss,
            train_accuracy=train_metrics.accuracy,
            validation_accuracy=validation_metrics.accuracy,
            test_accuracy=test_metrics.accuracy,
            learning_rate=current_learning_rate(optimizer),
            num_params=count_parameters(model),
        )
        history.append(init_entry)
        wandb_logger.log_history_entry(init_entry)
        if progress is not None:
            progress(
                f"[INIT] Epoch 0, train_loss={train_metrics.loss:.4f}, "
                f"validation_loss={validation_metrics.loss:.4f}, "
                f"test_loss={test_metrics.loss:.4f}, "
                f"train_acc={train_metrics.accuracy:.3f}, "
                f"validation_acc={validation_metrics.accuracy:.3f}, "
                f"test_acc={test_metrics.accuracy:.3f}"
            )

        growth_count = 0
        last_growth_epoch: int | None = None
        for epoch in range(1, config.training.epochs + 1):
            use_fgd_theory_learning_rate = (
                config.training.method == "fgd_approx"
                and config.fgd_approx.learning_rate_policy == "theory_interval"
            )
            learning_rate_clipped_by_validation = False
            if use_fgd_theory_learning_rate:
                if validation_certificate_for_next_epoch is None:
                    validation_certificate_for_next_epoch = (
                        evaluate_fgd_validation_certificate(
                            model=model,
                            data_loader=validation_loader,
                            device=device,
                            config=config.fgd_approx,
                            learning_rate=current_fgd_learning_rate,
                            probe=fgd_validation_probe,
                        )
                    )
                lr_certificate = validation_certificate_for_next_epoch
                certified_learning_rate = certified_validation_learning_rate(
                    lr_certificate,
                    config.fgd_approx,
                )
                if certified_learning_rate is not None:
                    current_lr_in_interval = (
                        current_fgd_learning_rate
                        > config.fgd_approx.theory_lr_min
                        and current_fgd_learning_rate
                        <= certified_learning_rate + config.fgd_approx.eps
                    )
                    if (
                        config.fgd_approx.theory_lr_follow_bound
                        or not current_lr_in_interval
                    ):
                        learning_rate_clipped_by_validation = (
                            abs(
                                current_fgd_learning_rate
                                - certified_learning_rate
                            )
                            > config.fgd_approx.eps
                        )
                        current_fgd_learning_rate = certified_learning_rate
                    learning_rate = current_fgd_learning_rate
                else:
                    # No theoretically admissible step was certified. Keep the
                    # model fixed so validation can decide whether to grow.
                    learning_rate = 0.0
            else:
                learning_rate = scheduled_learning_rate(
                    config,
                    epoch=epoch,
                    cycle_start_epoch=lr_cycle_start_epoch,
                )
            apply_learning_rate(optimizer, learning_rate)

            rel_error: float | None = None
            selected_layer_index: int | None = None
            fgd_layer_rel_errors: list[FGDLayerRelError] = []
            fgd_output_rel_error: FGDOutputRelError | None = None
            fgd_learning_rate_upper_bound: float | None = None
            fgd_max_valid_learning_rate: float | None = None
            fgd_learning_rate_interval_valid: bool | None = None
            fgd_learning_rate_clipped_batches = int(
                learning_rate_clipped_by_validation
            )
            fgd_skipped_batches = 0
            fgd_relative_error_condition_valid: bool | None = None
            fgd_loss_descent_valid: bool | None = None
            fgd_loss_non_descent_batches = 0
            fgd_gradient_sq_norm: float | None = None
            fgd_min_gradient_sq_norm: float | None = None
            fgd_theory_descent_coefficient: float | None = None
            fgd_stationary_bound: float | None = None
            fgd_stationary_bound_valid: bool | None = None
            fgd_global_bound: float | None = None
            fgd_global_bound_valid: bool | None = None
            fgd_global_contraction: float | None = None
            fgd_theory_learning_rate_adjusted = False
            fgd_sensor_valid: bool | None = None
            fgd_sensor_invalid_batches = 0
            fgd_update_norm: float | None = None
            fgd_trial_sensor_failure = False
            diagnostic_trial: _FGDTrial | None = None
            fgd_growth_requested = False
            fgd_candidate_accepted: bool | None = None
            fgd_lr_search_trials = 0
            fgd_approximation_kind: str | None = (
                "tangent" if config.training.method == "fgd_approx" else None
            )
            fgd_rkhs_phase_attempted = False
            fgd_rkhs_phase_accepted: bool | None = None
            fgd_rkhs_phase_steps = 0
            fgd_growth_probe_improved: bool | None = None
            entry_learning_rate = current_learning_rate(optimizer)
            if config.training.method == "normal":
                epoch_result = train_one_epoch(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    optimizer=optimizer,
                    loss_function=loss_function,
                    device=device,
                    accuracy_tolerance=config.training.accuracy_tolerance,
                    gradient_clip_norm=config.training.gradient_clip_norm,
                    classification=classification,
                )
                step_type: StepType = "SGD"
            elif config.training.method == "fgd_approx":
                if use_fgd_theory_learning_rate:
                    max_outer_steps = max(
                        1,
                        config.fgd_approx.outer_steps_per_epoch,
                    )
                    accepted_steps_this_epoch = 0
                    # Multiple certified outer steps per epoch: each pass
                    # re-solves the shared direction at the CURRENT model and
                    # certifies it independently (k applications of the same
                    # per-step theorem). The epoch stops at the first
                    # rejected attempt; growth can only be requested when
                    # the FIRST attempt of the epoch fails.
                    for _outer_step_index in range(max_outer_steps):
                        theory_state = _FGDTheoryState(
                            epoch_count=fgd_epoch_count,
                            min_gradient_sq_norm=fgd_min_gradient_sq_norm,
                            min_positive_learning_rate=fgd_min_positive_learning_rate,
                            min_descent_coefficient=fgd_min_descent_coefficient,
                            global_contraction_product=fgd_global_contraction_product,
                            previous_validation_functional_loss=(
                                previous_validation_functional_loss
                            ),
                        )
                        _clear_inaccessible_tensor_caches(model)
                        frozen_train_batches = list(train_loader)

                        # One genuine outer step per epoch: solve the SHARED
                        # direction u* on the fixed train probe at the current
                        # model f_t, certify THAT direction on the fixed
                        # validation probe BEFORE any update, then search the
                        # step size eta for the single update theta - eta * u*.
                        tangent_direction: tuple[torch.Tensor, ...] | None = None
                        direction_stats: _FunctionalStepStats | None = None
                        maximum_learning_rate: float | None = None
                        direction_sensor_failure = False
                        direction_step = _compute_tangent_projection_step(
                            model=model,
                            x=fgd_train_probe[0],
                            y=fgd_train_probe[1],
                            config=config.fgd_approx,
                        )
                        if _projection_step_sensor_valid(
                            direction_step,
                            config.fgd_approx,
                        ):
                            tangent_direction = direction_step.parameter_updates
                            fgd_update_norm = math.sqrt(
                                sum(
                                    float(
                                        torch.sum(update.detach() ** 2).item()
                                    )
                                    for update in tangent_direction
                                )
                            )
                            direction_stats = measure_direction_projection(
                                model,
                                tangent_direction,
                                fgd_validation_probe[0],
                                fgd_validation_probe[1],
                                config.fgd_approx,
                            )
                            # The direction is only a projection on the TRAIN
                            # probe; on validation it is certified like a secant
                            # (finiteness sensor, Crel/interval as gates).
                            direction_certificate = (
                                certificate_from_projection_stats(
                                    stats=direction_stats,
                                    learning_rate=None,
                                    config=config.fgd_approx,
                                    projection_sensor=False,
                                )
                            )
                            if direction_certificate.sensor_valid:
                                maximum_learning_rate = (
                                    certified_validation_learning_rate(
                                        direction_certificate,
                                        config.fgd_approx,
                                    )
                                )
                            else:
                                direction_sensor_failure = True
                                direction_stats = None
                        else:
                            direction_sensor_failure = True

                        def evaluate_trial(candidate_learning_rate: float) -> _FGDTrial:
                            assert tangent_direction is not None
                            assert direction_stats is not None
                            return _evaluate_fgd_outer_trial(
                                base_model=model,
                                direction=tangent_direction,
                                direction_stats=direction_stats,
                                train_batches=frozen_train_batches,
                                validation_loader=validation_loader,
                                loss_function=loss_function,
                                device=device,
                                learning_rate=candidate_learning_rate,
                                accuracy_tolerance=config.training.accuracy_tolerance,
                                config=config,
                                classification=classification,
                                theory_state=theory_state,
                                initial_functional_gap=initial_functional_gap,
                                theory_loss_star=theory_loss_star,
                            )

                        if (
                            config.fgd_approx.tangent_measured_descent
                            and tangent_direction is not None
                            and direction_stats is not None
                        ):
                            # Paper-pure functional step: the tangent
                            # direction g = P_T r certified by MEASURED
                            # descent (Prop. 3.8), step size from a
                            # nonlinear line search instead of eta_max(eps).
                            search_result = _search_tangent_measured_descent(
                                base_model=model,
                                direction=tangent_direction,
                                direction_stats=direction_stats,
                                train_batches=frozen_train_batches,
                                validation_loader=validation_loader,
                                loss_function=loss_function,
                                device=device,
                                accuracy_tolerance=(
                                    config.training.accuracy_tolerance
                                ),
                                config=config,
                                theory_state=theory_state,
                                initial_functional_gap=initial_functional_gap,
                                theory_loss_star=theory_loss_star,
                            )
                        else:
                            search_result = (
                                _search_fgd_certified_trial(
                                    maximum_learning_rate=maximum_learning_rate,
                                    evaluate_trial=evaluate_trial,
                                    config=config.fgd_approx,
                                )
                                if maximum_learning_rate is not None
                                and direction_stats is not None
                                else _FGDSearchResult(
                                    None,
                                    None,
                                    0,
                                    direction_sensor_failure,
                                )
                            )
                        fgd_lr_search_trials += search_result.trial_count
                        fgd_trial_sensor_failure = (
                            fgd_trial_sensor_failure
                            or search_result.sensor_failure
                        )
                        accepted_trial = search_result.accepted
                        diagnostic_trial = search_result.last_trial

                        outer_step_trial = (
                            accepted_trial
                            if accepted_trial is not None
                            else diagnostic_trial
                        )
                        if progress is not None and outer_step_trial is not None:
                            step_error = (
                                outer_step_trial.certificate.output_relative_error
                            )
                            step_rel_error = (
                                outer_step_trial.certificate.relative_error
                            )
                            progress(
                                f"[FGD-STEP] Epoch {epoch}: "
                                "loss_before="
                                f"{theory_state.previous_validation_functional_loss:.6e}, "
                                "loss_after="
                                f"{outer_step_trial.validation_functional_loss:.6e}, "
                                "rel_err_before="
                                + (
                                    f"{step_rel_error:.4f}"
                                    if step_rel_error is not None
                                    else "n/a"
                                )
                                + ", |r|="
                                + (
                                    f"{step_error.target_norm:.4e}"
                                    if step_error is not None
                                    else "n/a"
                                )
                                + ", |g|="
                                + (
                                    f"{step_error.approximation_norm:.4e}"
                                    if step_error is not None
                                    else "n/a"
                                )
                                + ", |u|="
                                + (
                                    f"{fgd_update_norm:.4e}"
                                    if fgd_update_norm is not None
                                    else "n/a"
                                )
                                + ", eta="
                                f"{outer_step_trial.epoch_result.learning_rate:.4g}, "
                                f"accepted={accepted_trial is not None}"
                            )

                        if accepted_trial is None:
                            break
                        accepted_steps_this_epoch += 1
                        model = accepted_trial.model
                        fgd_epoch_result = accepted_trial.epoch_result
                        test_metrics = evaluate_regression_metrics(
                            model,
                            test_loader,
                            loss_function,
                            device=device,
                            accuracy_tolerance=config.training.accuracy_tolerance,
                            classification=classification,
                        )
                        fgd_epoch_result = replace(
                            fgd_epoch_result,
                            test_loss=test_metrics.loss,
                            test_accuracy=test_metrics.accuracy,
                        )
                        epoch_result = fgd_epoch_result
                        # The committed certificate describes the DIRECTION
                        # that moved f_t; the next epoch's state certificate
                        # must be measured at f_{t+1}, so it is recomputed
                        # lazily at the top of the next epoch.
                        validation_certificate = accepted_trial.certificate
                        validation_certificate_for_next_epoch = None
                        current_fgd_learning_rate = (
                            fgd_epoch_result.min_positive_learning_rate
                            or learning_rate
                        )
                        optimizer = build_optimizer(model, config.optimizer)
                        apply_learning_rate(optimizer, current_fgd_learning_rate)
                        entry_learning_rate = current_fgd_learning_rate
                        fgd_candidate_accepted = True
                        fgd_accepted_outer_steps += 1
                        fgd_theory_learning_rate_adjusted = (
                            abs(current_fgd_learning_rate - learning_rate)
                            > config.fgd_approx.eps
                        )
                        fgd_growth_requested = False

                        accepted_state = accepted_trial.theory_state
                        fgd_epoch_count = accepted_state.epoch_count
                        fgd_min_gradient_sq_norm = (
                            accepted_state.min_gradient_sq_norm
                        )
                        fgd_min_positive_learning_rate = (
                            accepted_state.min_positive_learning_rate
                        )
                        fgd_min_descent_coefficient = (
                            accepted_state.min_descent_coefficient
                        )
                        fgd_global_contraction_product = (
                            accepted_state.global_contraction_product
                        )
                        previous_validation_functional_loss = (
                            accepted_state.previous_validation_functional_loss
                        )
                        fgd_loss_descent_valid = (
                            accepted_trial.loss_descent_valid
                        )
                        fgd_stationary_bound = accepted_trial.stationary_bound
                        fgd_stationary_bound_valid = (
                            accepted_trial.stationary_bound_valid
                        )
                        fgd_global_bound = accepted_trial.global_bound
                        fgd_global_bound_valid = (
                            accepted_trial.global_bound_valid
                        )
                        fgd_global_contraction = (
                            accepted_trial.global_contraction
                        )
                    if accepted_steps_this_epoch == 0:
                        base_train_metrics = evaluate_regression_metrics(
                            model,
                            frozen_train_batches,
                            loss_function,
                            device=device,
                            accuracy_tolerance=config.training.accuracy_tolerance,
                            classification=classification,
                        )
                        base_test_metrics = evaluate_regression_metrics(
                            model,
                            test_loader,
                            loss_function,
                            device=device,
                            accuracy_tolerance=config.training.accuracy_tolerance,
                            classification=classification,
                        )
                        epoch_result = FGDApproxEpochResult(
                            train_loss=base_train_metrics.loss,
                            train_accuracy=base_train_metrics.accuracy,
                            test_loss=base_test_metrics.loss,
                            test_accuracy=base_test_metrics.accuracy,
                            learning_rate=0.0,
                            next_learning_rate=None,
                            learning_rate_upper_bound=None,
                            learning_rate_interval_valid=None,
                            learning_rate_clipped_batches=0,
                            skipped_batches=0,
                            relative_error_condition_valid=None,
                            loss_descent_valid=None,
                            loss_non_descent_batches=0,
                            gradient_sq_norm=None,
                            theory_descent_coefficient=None,
                            min_positive_learning_rate=None,
                            relative_error=None,
                            selected_layer_index=None,
                            layer_relative_errors=[],
                            output_relative_error=None,
                            sensor_valid=not search_result.sensor_failure,
                            sensor_invalid_batches=0,
                        )
                        validation_certificate = lr_certificate
                        validation_certificate_for_next_epoch = lr_certificate
                        entry_learning_rate = 0.0
                        fgd_candidate_accepted = False
                        fgd_growth_requested = (
                            config.growth_schedule.enabled
                            and lr_certificate.sensor_valid
                            and not search_result.sensor_failure
                        )
                        if (
                            fgd_growth_requested
                            and config.fgd_approx.growth_requires_admissibility_failure
                        ):
                            # Lemma 3.5 is the paper's structural criterion:
                            # capacity must increase only when the reachable
                            # set can no longer represent r_t, i.e. when
                            # eps >= rel_error_threshold. A failed transaction
                            # is NOT that signal on its own -- a step can fail
                            # for step-size or loss-plateau reasons while
                            # eps stays far below 1/2, and growing then throws
                            # parameters at a problem that is not capacity.
                            state_relative_error = lr_certificate.relative_error
                            fgd_growth_requested = (
                                state_relative_error is not None
                                and state_relative_error
                                >= config.fgd_approx.rel_error_threshold
                            )
                        if fgd_growth_requested:
                            selected_layer_index = select_tiny_growth_layer_index(
                                model=model,
                                train_loader=frozen_train_batches,
                                device=device,
                                config=config.fgd_approx,
                            )
                            epoch_result = replace(
                                epoch_result,
                                selected_layer_index=selected_layer_index,
                            )
                        if diagnostic_trial is not None:
                            fgd_loss_descent_valid = (
                                diagnostic_trial.loss_descent_valid
                            )
                            fgd_stationary_bound = (
                                diagnostic_trial.stationary_bound
                            )
                            fgd_stationary_bound_valid = (
                                diagnostic_trial.stationary_bound_valid
                            )
                            fgd_global_bound = diagnostic_trial.global_bound
                            fgd_global_bound_valid = (
                                diagnostic_trial.global_bound_valid
                            )
                            fgd_global_contraction = (
                                diagnostic_trial.global_contraction
                            )
                else:
                    fgd_epoch_result = train_one_epoch_fgd_approx(
                        model=model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        loss_function=loss_function,
                        device=device,
                        learning_rate=learning_rate,
                        accuracy_tolerance=config.training.accuracy_tolerance,
                        config=config.fgd_approx,
                        projection_group_size=max(
                            1,
                            config.fgd_approx.projection_group_size,
                        ),
                        classification=classification,
                    )
                    epoch_result = fgd_epoch_result
                    validation_certificate = evaluate_fgd_validation_certificate(
                        model=model,
                        data_loader=validation_loader,
                        device=device,
                        config=config.fgd_approx,
                        learning_rate=None,
                        probe=fgd_validation_probe,
                    )

                selected_layer_index = epoch_result.selected_layer_index
                fgd_layer_rel_errors = epoch_result.layer_relative_errors
                rel_error = validation_certificate.relative_error
                fgd_output_rel_error = validation_certificate.output_relative_error
                fgd_learning_rate_upper_bound = (
                    validation_certificate.learning_rate_upper_bound
                )
                fgd_max_valid_learning_rate = (
                    validation_certificate.max_valid_learning_rate
                )
                fgd_learning_rate_interval_valid = (
                    validation_certificate.learning_rate_interval_valid
                )
                fgd_skipped_batches = validation_certificate.skipped_batches
                fgd_relative_error_condition_valid = (
                    validation_certificate.relative_error_condition_valid
                )
                fgd_gradient_sq_norm = validation_certificate.gradient_sq_norm
                fgd_theory_descent_coefficient = (
                    validation_certificate.theory_descent_coefficient
                )
                fgd_sensor_valid = validation_certificate.sensor_valid
                fgd_sensor_invalid_batches = (
                    validation_certificate.sensor_invalid_batches
                )
                if fgd_trial_sensor_failure:
                    diagnostic_invalid_batches = (
                        diagnostic_trial.epoch_result.sensor_invalid_batches
                        + diagnostic_trial.certificate.sensor_invalid_batches
                        if diagnostic_trial is not None
                        else 0
                    )
                    fgd_sensor_valid = False
                    fgd_sensor_invalid_batches = max(
                        1,
                        fgd_sensor_invalid_batches,
                        diagnostic_invalid_batches,
                    )
                    rel_error = None
                    fgd_output_rel_error = None
                    fgd_relative_error_condition_valid = None
                fgd_loss_non_descent_batches = int(
                    fgd_loss_descent_valid is False
                )
                step_type = "FGD"
            else:
                raise ValueError(
                    f"Unsupported training method '{config.training.method}'. "
                    "Use one of: normal, fgd_approx, fgd_rkhs, fgd_rkhs_grow."
                )

            validation_metrics = evaluate_regression_metrics(
                model,
                validation_loader,
                loss_function,
                device=device,
                accuracy_tolerance=config.training.accuracy_tolerance,
                classification=classification,
            )
            epoch_entry = HistoryEntry(
                step=epoch,
                step_type=step_type,
                train_loss=epoch_result.train_loss,
                validation_loss=validation_metrics.loss,
                test_loss=epoch_result.test_loss,
                train_accuracy=epoch_result.train_accuracy,
                validation_accuracy=validation_metrics.accuracy,
                test_accuracy=epoch_result.test_accuracy,
                learning_rate=entry_learning_rate,
                num_params=count_parameters(model),
                rel_error=rel_error,
                selected_layer_index=selected_layer_index,
                fgd_layer_rel_errors=fgd_layer_rel_errors,
                fgd_output_rel_error=fgd_output_rel_error,
                fgd_learning_rate_upper_bound=fgd_learning_rate_upper_bound,
                fgd_max_valid_learning_rate=fgd_max_valid_learning_rate,
                fgd_learning_rate_interval_valid=fgd_learning_rate_interval_valid,
                fgd_learning_rate_clipped_batches=fgd_learning_rate_clipped_batches,
                fgd_skipped_batches=fgd_skipped_batches,
                fgd_relative_error_condition_valid=(
                    fgd_relative_error_condition_valid
                ),
                fgd_loss_descent_valid=fgd_loss_descent_valid,
                fgd_loss_non_descent_batches=fgd_loss_non_descent_batches,
                fgd_gradient_sq_norm=fgd_gradient_sq_norm,
                fgd_min_gradient_sq_norm=fgd_min_gradient_sq_norm,
                fgd_theory_descent_coefficient=fgd_theory_descent_coefficient,
                fgd_stationary_bound=fgd_stationary_bound,
                fgd_stationary_bound_valid=fgd_stationary_bound_valid,
                fgd_global_bound=fgd_global_bound,
                fgd_global_bound_valid=fgd_global_bound_valid,
                fgd_global_contraction=fgd_global_contraction,
                fgd_theory_learning_rate_adjusted=(
                    fgd_theory_learning_rate_adjusted
                ),
                fgd_sensor_valid=fgd_sensor_valid,
                fgd_sensor_invalid_batches=fgd_sensor_invalid_batches,
                fgd_update_norm=fgd_update_norm,
                fgd_candidate_accepted=fgd_candidate_accepted,
                fgd_lr_search_trials=fgd_lr_search_trials,
                fgd_approximation_kind=fgd_approximation_kind,
                fgd_rkhs_phase_attempted=fgd_rkhs_phase_attempted,
                fgd_rkhs_phase_accepted=fgd_rkhs_phase_accepted,
                fgd_rkhs_phase_steps=fgd_rkhs_phase_steps,
                fgd_growth_probe_improved=fgd_growth_probe_improved,
            )
            history.append(epoch_entry)
            wandb_logger.log_history_entry(epoch_entry)

            if progress is not None and should_log_epoch(epoch, config):
                delta = epoch_result.test_loss - last_test_loss
                rel_error_msg = (
                    f", rel_err={rel_error:.3f}" if rel_error is not None else ""
                )
                selected_layer_msg = (
                    f", selected_layer={selected_layer_index}"
                    if selected_layer_index is not None
                    else ""
                )
                progress(
                    f"[{step_type}] Epoch {epoch}, "
                    f"train_loss={epoch_result.train_loss:.4f}, "
                    f"validation_loss={validation_metrics.loss:.4f}, "
                    f"test_loss={epoch_result.test_loss:.4f} ({delta:+.4f}), "
                    f"train_acc={epoch_result.train_accuracy:.3f}, "
                    f"validation_acc={validation_metrics.accuracy:.3f}, "
                    f"test_acc={epoch_result.test_accuracy:.3f}, "
                    f"lr={entry_learning_rate:.4g}"
                    f"{rel_error_msg}"
                    f"{selected_layer_msg}"
                )
            if progress is not None and config.training.method == "fgd_approx":
                warnings = []
                if fgd_relative_error_condition_valid is False:
                    warnings.append("relative-error condition failed")
                if fgd_learning_rate_interval_valid is False:
                    warnings.append("learning-rate interval invalid")
                if fgd_learning_rate_clipped_batches > 0 and not (
                    config.fgd_approx.learning_rate_policy == "theory_interval"
                    and config.fgd_approx.theory_lr_follow_bound
                ):
                    warnings.append(
                        "learning-rate clipped by validation certificate"
                    )
                if fgd_skipped_batches > 0:
                    warnings.append(
                        "validation certificate rejected "
                        f"{fgd_skipped_batches} batch(es)"
                    )
                if fgd_loss_descent_valid is False:
                    warnings.append("validation functional loss increased")
                if fgd_sensor_valid is False:
                    warnings.append(
                        f"sensor invalid on "
                        f"{fgd_sensor_invalid_batches} validation batch(es)"
                    )
                diagnostic_bound_suffix = (
                    " (trajectory diagnostic, not an acceptance gate)"
                    if config.fgd_approx.local_acceptance_conditions
                    else ""
                )
                if fgd_stationary_bound_valid is False:
                    warnings.append(
                        "stationary-point bound failed"
                        + diagnostic_bound_suffix
                    )
                if fgd_global_bound_valid is False:
                    warnings.append(
                        "global-convergence bound failed"
                        + diagnostic_bound_suffix
                    )
                if fgd_theory_learning_rate_adjusted:
                    warnings.append(
                        "maximum validation-certified learning rate accepted "
                        "after transactional search"
                    )
                if fgd_candidate_accepted is False:
                    warnings.append(
                        "no learning rate satisfied all validation conditions; "
                        "model update rejected"
                    )
                if fgd_growth_requested:
                    warnings.append("FGD conditions request growth")
                if (
                    fgd_theory_descent_coefficient is not None
                    and fgd_theory_descent_coefficient <= 0.0
                ):
                    warnings.append("theory descent coefficient is non-positive")
                if warnings:
                    progress(f"[FGD-WARN] Epoch {epoch}: " + "; ".join(warnings))
            last_test_loss = epoch_result.test_loss

            if config.training.method == "normal":
                growth_triggered = should_grow(epoch, config.growth_schedule)
            else:
                growth_triggered = config.growth_schedule.enabled and (
                    fgd_growth_requested
                    or (
                        fgd_sensor_valid is True
                        and rel_error is not None
                        and should_trigger_fgd_growth(
                            relative_error=rel_error,
                            epoch=epoch,
                            last_growth_epoch=last_growth_epoch,
                            config=config.fgd_approx,
                        )
                    )
                )
                # Hard parameter budget: stop growing once the cap is
                # reached; the flow keeps training the fixed structure.
                max_parameters = config.fgd_approx.max_total_parameters
                if (
                    growth_triggered
                    and max_parameters is not None
                    and count_parameters(model) >= max_parameters
                ):
                    growth_triggered = False
                    if progress is not None:
                        progress(
                            f"[FGD] Epoch {epoch}: growth suppressed "
                            f"(parameter budget {max_parameters} reached: "
                            f"{count_parameters(model)} params)"
                        )

            if (
                config.training.method == "fgd_approx"
                and fgd_candidate_accepted is True
            ):
                # A committed tangent outer step: the structure is alive.
                fgd_epochs_without_commit = 0

            growth_probe: _GrowthProbe | None = None
            # The fallback families exist to act when the tangent outer step
            # cannot certify. Nesting them inside the GROWTH trigger means
            # that as soon as the structure becomes adequate (eps < 1/2, so
            # growth is correctly not requested) the ladder is skipped and
            # only the tangent family remains -- which is exactly when the
            # flow freezes. The two decisions are independent: families
            # handle "the tangent could not move", growth handles "the
            # reachable set cannot represent r".
            tangent_needs_fallback = (
                config.fgd_approx.families_available_without_growth
                and fgd_candidate_accepted is False
            )
            if (
                (growth_triggered or tangent_needs_fallback)
                and config.training.method == "fgd_approx"
                and config.fgd_approx.projection_solver != "gromo_layer"
                and config.fgd_approx.learning_rate_policy == "theory_interval"
            ):
                growth_train_batches = list(train_loader)

                def _attempt_rkhs_head_stage(in_ladder: bool) -> bool:
                    """Certified RKHS head phase; True iff a head was committed.

                    In the family ladder (in_ladder=True) the phase runs before any
                    growth probing and a rejection simply passes control to the next
                    family. In the legacy position (in_ladder=False) it runs only
                    after a failed growth probe, gated by secant_fgd.enabled,
                    exactly as before family_order existed.
                    """
                    nonlocal fgd_rkhs_phase_attempted, fgd_rkhs_phase_steps
                    nonlocal fgd_rkhs_phase_accepted, model, optimizer
                    nonlocal validation_certificate_for_next_epoch
                    nonlocal current_fgd_learning_rate
                    nonlocal previous_validation_functional_loss
                    nonlocal last_test_loss
                    nonlocal fgd_accepted_outer_steps
                    nonlocal fgd_epochs_without_commit
                    fgd_rkhs_phase_attempted = (
                        True if in_ladder else config.secant_fgd.enabled
                    )
                    phase: _RKHSPhaseResult | None = None
                    if fgd_rkhs_phase_attempted:
                        phase = _run_rkhs_head_phase(
                            model=model,
                            train_batches=growth_train_batches,
                            config=config,
                            device=device,
                        )
                        fgd_rkhs_phase_steps = phase.steps
                        fgd_rkhs_phase_accepted = phase.accepted

                    # In-ladder external gate: the phase's internal
                    # acceptance compares losses on its own (subsampled,
                    # reshuffled) train points, so an epoch-to-epoch
                    # subsample change can re-certify an epsilon
                    # "improvement" forever. The family only commits when
                    # the head genuinely improves the FULL validation
                    # functional by the configured relative margin —
                    # consistent with every other family gating on
                    # validation.
                    ladder_gate_declined = False
                    if in_ladder and phase is not None and phase.accepted:
                        gate_loss_before = evaluate_functional_loss(
                            model,
                            validation_loader,
                            device,
                            config.fgd_approx.functional_loss,
                        )
                        gate_candidate = copy.deepcopy(model)
                        _apply_certified_head(
                            gate_candidate,
                            phase.trainer.model,
                        )
                        gate_loss_after = evaluate_functional_loss(
                            gate_candidate,
                            validation_loader,
                            device,
                            config.fgd_approx.functional_loss,
                        )
                        required_improvement = (
                            config.fgd_approx
                            .rkhs_family_min_relative_improvement
                            * max(gate_loss_before, config.fgd_approx.eps)
                        )
                        gate_improvement = gate_loss_before - gate_loss_after
                        if not (
                            math.isfinite(gate_improvement)
                            and gate_improvement >= required_improvement
                        ):
                            ladder_gate_declined = True
                            fgd_rkhs_phase_accepted = False
                            if progress is not None:
                                progress(
                                    f"[RKHS] Epoch {epoch}: head phase "
                                    "validation improvement "
                                    f"{gate_improvement:.3e} is below the "
                                    "family margin "
                                    f"{required_improvement:.3e}; declining"
                                )

                    if (
                        phase is not None
                        and phase.accepted
                        and not ladder_gate_declined
                    ):
                        _apply_certified_head(model, phase.trainer.model)
                        optimizer = build_optimizer(model, config.optimizer)
                        validation_certificate_for_next_epoch = (
                            evaluate_fgd_validation_certificate(
                                model=model,
                                data_loader=validation_loader,
                                device=device,
                                config=config.fgd_approx,
                                learning_rate=None,
                                probe=fgd_validation_probe,
                            )
                        )
                        certified_learning_rate = (
                            certified_validation_learning_rate(
                                validation_certificate_for_next_epoch,
                                config.fgd_approx,
                            )
                        )
                        if certified_learning_rate is not None:
                            current_fgd_learning_rate = certified_learning_rate
                        apply_learning_rate(
                            optimizer,
                            current_fgd_learning_rate,
                        )
                        previous_validation_functional_loss = (
                            evaluate_functional_loss(
                                model,
                                validation_loader,
                                device,
                                config.fgd_approx.functional_loss,
                            )
                        )
                        phase_theory = phase.trainer.theory
                        phase_record = phase.last_record
                        phase_train_metrics = evaluate_regression_metrics(
                            model,
                            train_loader,
                            loss_function,
                            device=device,
                            accuracy_tolerance=(
                                config.training.accuracy_tolerance
                            ),
                            classification=classification,
                        )
                        phase_validation_metrics = evaluate_regression_metrics(
                            model,
                            validation_loader,
                            loss_function,
                            device=device,
                            accuracy_tolerance=(
                                config.training.accuracy_tolerance
                            ),
                            classification=classification,
                        )
                        phase_test_metrics = evaluate_regression_metrics(
                            model,
                            test_loader,
                            loss_function,
                            device=device,
                            accuracy_tolerance=(
                                config.training.accuracy_tolerance
                            ),
                            classification=classification,
                        )
                        rkhs_entry = HistoryEntry(
                            step=epoch,
                            step_type="RKHS",
                            train_loss=phase_train_metrics.loss,
                            validation_loss=phase_validation_metrics.loss,
                            test_loss=phase_test_metrics.loss,
                            train_accuracy=phase_train_metrics.accuracy,
                            validation_accuracy=(
                                phase_validation_metrics.accuracy
                            ),
                            test_accuracy=phase_test_metrics.accuracy,
                            learning_rate=current_fgd_learning_rate,
                            num_params=count_parameters(model),
                            rel_error=(
                                phase_record.relative_error
                                if phase_record is not None
                                else None
                            ),
                            fgd_learning_rate_upper_bound=(
                                phase_theory.learning_rate_upper_bound
                            ),
                            fgd_learning_rate_interval_valid=True,
                            fgd_relative_error_condition_valid=(
                                phase_record.relative_error_condition_valid
                                if phase_record is not None
                                else None
                            ),
                            fgd_loss_descent_valid=phase.descent_valid,
                            fgd_gradient_sq_norm=(
                                phase_record.gradient_sq_norm
                                if phase_record is not None
                                else None
                            ),
                            fgd_theory_descent_coefficient=(
                                phase_theory.descent_coefficient
                            ),
                            fgd_global_bound=phase.global_bound,
                            fgd_global_bound_valid=phase.global_bound_valid,
                            fgd_global_contraction=phase_theory.contraction,
                            fgd_candidate_accepted=True,
                            fgd_approximation_kind="rkhs_head",
                            fgd_rkhs_phase_attempted=True,
                            fgd_rkhs_phase_accepted=True,
                            fgd_rkhs_phase_steps=fgd_rkhs_phase_steps,
                            fgd_growth_probe_improved=False,
                            fgd_rkhs_dictionary_size=(
                                phase_record.dictionary_size
                                if phase_record is not None
                                else None
                            ),
                            fgd_rkhs_functional_loss=(
                                phase.functional_loss_after
                            ),
                            fgd_rkhs_loss_star=phase_theory.loss_star,
                        )
                        history.append(rkhs_entry)
                        wandb_logger.log_history_entry(rkhs_entry)
                        if progress is not None:
                            progress(
                                f"[RKHS] Epoch {epoch}: certified head phase "
                                "accepted (structure "
                                "not exhausted); functional loss "
                                f"{phase.model_loss_before:.4e} -> "
                                f"{phase.functional_loss_after:.4e} "
                                f"(ceiling L*={phase_theory.loss_star:.4e}, "
                                f"steps={phase.steps}, "
                                f"converged={phase.converged})"
                            )
                        last_test_loss = phase_test_metrics.loss
                        fgd_accepted_outer_steps += 1
                        fgd_epochs_without_commit = 0
                        family_rejection_step.pop("rkhs_head", None)
                        return True
                    else:
                        phase_theory = (
                            phase.trainer.theory if phase is not None else None
                        )
                        phase_record = (
                            phase.last_record if phase is not None else None
                        )
                        rejected_rkhs_entry = HistoryEntry(
                            step=epoch,
                            step_type="RKHS",
                            train_loss=epoch_result.train_loss,
                            validation_loss=validation_metrics.loss,
                            test_loss=epoch_result.test_loss,
                            train_accuracy=epoch_result.train_accuracy,
                            validation_accuracy=validation_metrics.accuracy,
                            test_accuracy=epoch_result.test_accuracy,
                            learning_rate=0.0,
                            num_params=count_parameters(model),
                            rel_error=(
                                phase_record.relative_error
                                if phase_record is not None
                                else None
                            ),
                            fgd_relative_error_condition_valid=(
                                phase_record.relative_error_condition_valid
                                if phase_record is not None
                                else None
                            ),
                            fgd_loss_descent_valid=(
                                phase.descent_valid
                                if phase is not None
                                else None
                            ),
                            fgd_global_bound=(
                                phase.global_bound
                                if phase is not None
                                else None
                            ),
                            fgd_global_bound_valid=(
                                phase.global_bound_valid
                                if phase is not None
                                else None
                            ),
                            fgd_candidate_accepted=False,
                            fgd_approximation_kind="rkhs_head",
                            fgd_rkhs_phase_attempted=fgd_rkhs_phase_attempted,
                            fgd_rkhs_phase_accepted=False,
                            fgd_rkhs_phase_steps=fgd_rkhs_phase_steps,
                            fgd_growth_probe_improved=False,
                            fgd_rkhs_functional_loss=(
                                phase.functional_loss_after
                                if phase is not None
                                else None
                            ),
                            fgd_rkhs_loss_star=(
                                phase_theory.loss_star
                                if phase_theory is not None
                                else None
                            ),
                        )
                        history.append(rejected_rkhs_entry)
                        wandb_logger.log_history_entry(rejected_rkhs_entry)
                        if progress is not None:
                            if in_ladder:
                                progress(
                                    f"[RKHS] Epoch {epoch}: certified head phase did "
                                    "not certify an improvement; trying the next "
                                    "family"
                                )
                            else:
                                progress(
                                    f"[RKHS-WARN] Epoch {epoch}: growth did not "
                                    "improve the FGD certificate and the output "
                                    "layer is already at the certified global "
                                    "optimum of the fixed structure "
                                    "(the architecture is exhausted at this "
                                    "point)"
                                )
                        return False

                def _attempt_parametric_stage(family_name: str) -> bool:
                    """Parametric secant families; True iff a step committed.

                    parametric_gd: screened by the output-projection cosine
                    and certified at the scale-optimal eta* through the full
                    relative-error certificate (Crel, interval, descent,
                    Cstat, Cglob). parametric_descent: same generation and
                    eta* calibration, but certified by the MEASURED descent
                    coefficient (Prop. 3.8 with the exact sum-MSE
                    function-space constants), with Cprog/Cstat/Cglob on the
                    same accumulators.
                    """
                    nonlocal model, optimizer
                    nonlocal validation_certificate_for_next_epoch
                    nonlocal current_fgd_learning_rate
                    nonlocal previous_validation_functional_loss
                    nonlocal fgd_epoch_count, fgd_min_gradient_sq_norm
                    nonlocal fgd_min_positive_learning_rate
                    nonlocal fgd_min_descent_coefficient
                    nonlocal fgd_global_contraction_product
                    nonlocal last_test_loss
                    nonlocal fgd_accepted_outer_steps
                    nonlocal fgd_epochs_without_commit
                    stage_theory_state = _FGDTheoryState(
                        epoch_count=fgd_epoch_count,
                        min_gradient_sq_norm=fgd_min_gradient_sq_norm,
                        min_positive_learning_rate=fgd_min_positive_learning_rate,
                        min_descent_coefficient=fgd_min_descent_coefficient,
                        global_contraction_product=fgd_global_contraction_product,
                        previous_validation_functional_loss=(
                            previous_validation_functional_loss
                        ),
                    )
                    if family_name == "parametric_gd":
                        stage_search = _search_parametric_gd_candidate(
                            base_model=model,
                            train_batches=growth_train_batches,
                            validation_loader=validation_loader,
                            loss_function=loss_function,
                            device=device,
                            accuracy_tolerance=(
                                config.training.accuracy_tolerance
                            ),
                            config=config,
                            probe=fgd_validation_probe,
                            classification=classification,
                            theory_state=stage_theory_state,
                            initial_functional_gap=initial_functional_gap,
                            theory_loss_star=theory_loss_star,
                        )
                    else:
                        stage_search = _search_parametric_descent_candidate(
                            base_model=model,
                            train_batches=growth_train_batches,
                            validation_loader=validation_loader,
                            loss_function=loss_function,
                            device=device,
                            accuracy_tolerance=(
                                config.training.accuracy_tolerance
                            ),
                            config=config,
                            classification=classification,
                            theory_state=stage_theory_state,
                            initial_functional_gap=initial_functional_gap,
                            theory_loss_star=theory_loss_star,
                        )
                    stage_label = (
                        "PGD" if family_name == "parametric_gd" else "PDESC"
                    )
                    stage_trial = stage_search.accepted
                    if stage_trial is None:
                        if progress is not None:
                            progress(
                                f"[{stage_label}] Epoch {epoch}: no "
                                f"{family_name} candidate passed its screen "
                                "and the full certificate "
                                f"({stage_search.trial_count} candidate(s) "
                                "evaluated); trying the next family"
                            )
                        return False
                    model = stage_trial.model
                    optimizer = build_optimizer(model, config.optimizer)
                    accepted_state = stage_trial.theory_state
                    fgd_epoch_count = accepted_state.epoch_count
                    fgd_min_gradient_sq_norm = accepted_state.min_gradient_sq_norm
                    fgd_min_positive_learning_rate = (
                        accepted_state.min_positive_learning_rate
                    )
                    fgd_min_descent_coefficient = (
                        accepted_state.min_descent_coefficient
                    )
                    fgd_global_contraction_product = (
                        accepted_state.global_contraction_product
                    )
                    previous_validation_functional_loss = (
                        accepted_state.previous_validation_functional_loss
                    )
                    validation_certificate_for_next_epoch = (
                        evaluate_fgd_validation_certificate(
                            model=model,
                            data_loader=validation_loader,
                            device=device,
                            config=config.fgd_approx,
                            learning_rate=None,
                            probe=fgd_validation_probe,
                        )
                    )
                    certified_learning_rate = certified_validation_learning_rate(
                        validation_certificate_for_next_epoch,
                        config.fgd_approx,
                    )
                    if certified_learning_rate is not None:
                        current_fgd_learning_rate = certified_learning_rate
                    apply_learning_rate(optimizer, current_fgd_learning_rate)
                    stage_validation_metrics = evaluate_regression_metrics(
                        model,
                        validation_loader,
                        loss_function,
                        device=device,
                        accuracy_tolerance=config.training.accuracy_tolerance,
                        classification=classification,
                    )
                    stage_test_metrics = evaluate_regression_metrics(
                        model,
                        test_loader,
                        loss_function,
                        device=device,
                        accuracy_tolerance=config.training.accuracy_tolerance,
                        classification=classification,
                    )
                    secant_entry = HistoryEntry(
                        step=epoch,
                        step_type="SEC",
                        train_loss=stage_trial.epoch_result.train_loss,
                        validation_loss=stage_validation_metrics.loss,
                        test_loss=stage_test_metrics.loss,
                        train_accuracy=stage_trial.epoch_result.train_accuracy,
                        validation_accuracy=stage_validation_metrics.accuracy,
                        test_accuracy=stage_test_metrics.accuracy,
                        learning_rate=stage_trial.epoch_result.learning_rate,
                        num_params=count_parameters(model),
                        rel_error=stage_trial.certificate.relative_error,
                        fgd_learning_rate_upper_bound=(
                            stage_trial.certificate.learning_rate_upper_bound
                        ),
                        fgd_max_valid_learning_rate=(
                            stage_trial.certificate.max_valid_learning_rate
                        ),
                        fgd_learning_rate_interval_valid=(
                            stage_trial.certificate.learning_rate_interval_valid
                        ),
                        fgd_relative_error_condition_valid=(
                            stage_trial.certificate.relative_error_condition_valid
                        ),
                        fgd_loss_descent_valid=stage_trial.loss_descent_valid,
                        fgd_gradient_sq_norm=stage_trial.certificate.gradient_sq_norm,
                        fgd_theory_descent_coefficient=(
                            stage_trial.certificate.theory_descent_coefficient
                        ),
                        fgd_stationary_bound=stage_trial.stationary_bound,
                        fgd_stationary_bound_valid=stage_trial.stationary_bound_valid,
                        fgd_global_bound=stage_trial.global_bound,
                        fgd_global_bound_valid=stage_trial.global_bound_valid,
                        fgd_global_contraction=stage_trial.global_contraction,
                        fgd_sensor_valid=True,
                        fgd_candidate_accepted=True,
                        fgd_approximation_kind=family_name,
                        fgd_growth_probe_improved=False,
                    )
                    history.append(secant_entry)
                    wandb_logger.log_history_entry(secant_entry)
                    if progress is not None:
                        stage_rel_error = (
                            stage_trial.certificate.relative_error
                        )
                        stage_cosine = math.sqrt(
                            max(0.0, 1.0 - stage_rel_error**2)
                        )
                        progress(
                            f"[{stage_label}] Epoch {epoch}: {family_name} "
                            "secant accepted "
                            f"(eta*={stage_trial.epoch_result.learning_rate:.4g}, "
                            f"cos={stage_cosine:.4f}, "
                            f"progress={_certified_trial_progress(stage_trial):.3e}, "
                            + (
                                "contraction="
                                f"{stage_trial.global_contraction:.6f}, "
                                if stage_trial.global_contraction is not None
                                # No PL constant for this functional, so no
                                # linear contraction is asserted.
                                else "contraction=n/a, "
                            )
                            + f"rel_err={stage_rel_error:.4f})"
                        )
                    last_test_loss = stage_test_metrics.loss
                    fgd_accepted_outer_steps += 1
                    fgd_epochs_without_commit = 0
                    family_rejection_step.pop(family_name, None)
                    return True

                # Fallback approximation families run in the configured order;
                # structural growth is probed only after every family fails.
                fallback_families = tuple(
                    name
                    for name in config.fgd_approx.family_order
                    if name != "tangent"
                )
                def _family_on_cooldown(name: str) -> bool:
                    return _family_rejection_active(
                        family_rejection_step.get(name),
                        fgd_accepted_outer_steps,
                        family_rejection_cooldown,
                    )

                skipped_families = [
                    name
                    for name in fallback_families
                    if _family_on_cooldown(name)
                ]
                if skipped_families and progress is not None:
                    progress(
                        f"[FGD] Epoch {epoch}: skipping "
                        + ", ".join(skipped_families)
                        + " (rejected recently; retried after "
                        f"{family_rejection_cooldown} accepted outer step(s) "
                        "or growth)"
                    )
                # A committed family step normally cancels growth: the
                # structure was not exhausted after all. That reasoning
                # breaks for a functional whose infimum is not attained
                # (cross-entropy: more confidence always lowers the loss),
                # because then SOME family step always certifies and growth
                # is postponed for ever, however inadequate the structure.
                # When Lemma 3.5 declares the reachable set unable to
                # represent r (eps >= rel_error_threshold), a family step
                # improves WITHIN an inadequate set and must not veto the
                # structural step.
                admissibility_failed = (
                    config.fgd_approx.admissibility_failure_forces_growth
                    and fgd_sensor_valid is True
                    and rel_error is not None
                    and rel_error >= config.fgd_approx.rel_error_threshold
                )
                # eps BEFORE the family step, for the stationarity test.
                epsilon_before_family = rel_error

                family_committed = False
                for family_name in fallback_families:
                    if family_committed:
                        break
                    if _family_on_cooldown(family_name):
                        continue
                    committed = False
                    if family_name == "rkhs_head":
                        committed = _attempt_rkhs_head_stage(in_ladder=True)
                    elif family_name in (
                        "parametric_gd",
                        "parametric_descent",
                    ):
                        committed = _attempt_parametric_stage(family_name)
                    if not committed:
                        # The family declined: remember it and let the next
                        # family in the ladder try.
                        family_rejection_step[family_name] = (
                            fgd_accepted_outer_steps
                        )
                        continue
                    # The family step IS kept either way -- it certified, so
                    # it commits. The only question is whether it also
                    # postpones the structural step.
                    family_committed = True
                    # R1 -- the structure's limit as stationarity of eps.
                    # A committed step that does not REDUCE the held-out
                    # relative error means training is no longer improving
                    # the reachable set's ability to express r: the descent
                    # is going into directions the structure cannot follow.
                    # That is the representation limit, so the step is kept
                    # but it must not postpone the structural step.
                    epsilon_stationary = False
                    if (
                        config.fgd_approx.growth_limit_criterion
                        == "epsilon_stationary"
                    ):
                        after = validation_certificate_for_next_epoch
                        epsilon_after_family = (
                            after.relative_error if after is not None else None
                        )
                        epsilon_stationary = (
                            epsilon_before_family is not None
                            and epsilon_after_family is not None
                            and epsilon_after_family
                            >= epsilon_before_family - config.fgd_approx.eps
                        )
                        if epsilon_stationary and progress is not None:
                            progress(
                                f"[FGD] Epoch {epoch}: {family_name} "
                                "committed but eps did not decrease "
                                f"({epsilon_before_family:.3f} -> "
                                f"{epsilon_after_family:.3f}): the structure "
                                "is at its representation limit, so the step "
                                "does not postpone growth"
                            )
                        # Generalised R1: eps is still (slowly) decreasing, so
                        # the stationarity test says "adequate" -- but on a
                        # rank-limited structure that verdict can be wrong.
                        # Only pay the look-ahead when about to stop, and only
                        # to CONFIRM the stop is real.
                        if (
                            not epsilon_stationary
                            and config.fgd_approx.growth_lookahead_adequacy
                            and epsilon_after_family is not None
                            and _growth_reduces_lookahead_epsilon(
                                model=model,
                                train_batches=growth_train_batches,
                                train_loader=train_loader,
                                validation_loader=validation_loader,
                                probe=fgd_validation_probe,
                                current_relative_error=epsilon_after_family,
                                device=device,
                                config=config,
                            )
                        ):
                            epsilon_stationary = True
                            if progress is not None:
                                progress(
                                    f"[FGD] Epoch {epoch}: eps still falls in "
                                    "place but growing the bottleneck reaches "
                                    "a strictly lower eps -- the structure is "
                                    "rank-limited, not adequate, so growth "
                                    "proceeds"
                                )
                    if not admissibility_failed and not epsilon_stationary:
                        growth_triggered = False
                        break
                    # Two different criteria reach this point; only the
                    # Lemma-3.5 one may claim eps >= threshold. R1 fires
                    # precisely when eps is BELOW the threshold but no longer
                    # decreasing, and has already logged its own reason.
                    if admissibility_failed and progress is not None:
                        progress(
                            f"[FGD] Epoch {epoch}: {family_name} committed, "
                            f"but eps={rel_error:.3f} >= "
                            f"{config.fgd_approx.rel_error_threshold} "
                            "(Lemma 3.5 fails), so it does not postpone "
                            "growth"
                        )
                    break

                if growth_triggered and not family_committed:
                    # Structure-burst patience: nothing committed this
                    # epoch; only probe structural growth after
                    # growth_patience consecutive exhausted epochs, so the
                    # stochastic families get real retries at the current
                    # structure first.
                    fgd_epochs_without_commit += 1
                    if (
                        fgd_epochs_without_commit
                        < config.fgd_approx.growth_patience
                    ):
                        growth_triggered = False
                        if progress is not None:
                            progress(
                                f"[FGD] Epoch {epoch}: growth deferred "
                                f"({fgd_epochs_without_commit}/"
                                f"{config.fgd_approx.growth_patience} "
                                "consecutive exhausted epochs at this "
                                "structure)"
                            )
                elif family_committed:
                    # A family committed within the ladder: the structure is
                    # not exhausted.
                    fgd_epochs_without_commit = 0

                if growth_triggered:
                    growth_probe = _probe_fgd_growth(
                        model=model,
                        train_batches=growth_train_batches,
                        validation_loader=validation_loader,
                        base_certificate=validation_certificate,
                        selected_layer_index=selected_layer_index,
                        growth_count=growth_count,
                        device=device,
                        config=config,
                        probe=fgd_validation_probe,
                    )
                    # The rel-error improvement gate is blind to delta
                    # growth (the GroMo optimal update reduces the loss but
                    # jumps the tangent linearization, so rel_err worsens and
                    # no candidate ever "improves"). When growth is selected
                    # by certified descent, a probe that realizes a genuine
                    # validation functional descent counts as an improvement
                    # — otherwise the flow cancels growth and stalls with the
                    # families already exhausted.
                    fgd_growth_probe_improved = bool(
                        growth_probe is not None
                        and (
                            growth_probe.improves_fgd
                            or (
                                config.fgd_approx.growth_select_by_descent
                                and growth_probe.functional_descent
                                > config.fgd_approx.eps
                            )
                        )
                    )
                    if not fgd_growth_probe_improved:
                        growth_triggered = False
                        if "rkhs_head" in fallback_families:
                            # The head phase already failed inside the ladder;
                            # re-running it here would duplicate the attempt.
                            if progress is not None:
                                progress(
                                    f"[FGD-STALL] Epoch {epoch}: every configured "
                                    "approximation family failed and the growth "
                                    "probe did not improve the certificate; "
                                    "model unchanged"
                                )
                        else:
                            _attempt_rkhs_head_stage(in_ladder=False)

            if growth_triggered:
                if config.training.method == "fgd_approx":
                    if (
                        config.fgd_approx.growth_selection
                        == "unified_expansion"
                    ):
                        # Width AND depth in one certified ranking.
                        #
                        # Both kinds are applied FUNCTION-PRESERVINGLY, so
                        # the structural step leaves f untouched: the loss
                        # cannot move, Prop. 3.8's descent certificate can
                        # never be violated by growing, and every change in
                        # eps is attributable to range(J) alone. Training
                        # descends; growth enlarges what can be descended
                        # along.
                        unified_kwargs = tiny_optimal_update_kwargs(
                            config.fgd_approx,
                            compute_delta=config.fgd_approx.growth_compute_delta,
                        )
                        growable = list(
                            range(len(getattr(model, "_growable_layers", [])))
                        )
                        neuron_costs = growable_neuron_costs(
                            model, config.data.in_features
                        )
                        base_parameters = count_parameters(model)
                        # rank J <= min_l w_l: while a location sits at the
                        # minimum, no purchase elsewhere can raise what the
                        # structure is able to express.
                        widths = [
                            int(layer.in_features)
                            for layer in model._growable_layers
                        ]
                        bottlenecks = set(rank_limiting_locations(widths))
                        ceiling_binds_precheck = (
                            validation_certificate.relative_error is not None
                            and validation_certificate.relative_error
                            >= config.fgd_approx.rel_error_threshold
                        )
                        # The rank cap mandates not only WHERE to buy but
                        # how far: while one location is the unique minimum
                        # it alone pins rank J, and the mandate ends exactly
                        # when the minimum becomes shared. Levelling there
                        # in one event is what the inequality already says;
                        # buying one neuron per event merely made each
                        # purchase wait for R1 again.
                        relief = bottleneck_relief_target(widths)
                        if (
                            relief is not None
                            and ceiling_binds_precheck
                        ):
                            relief_index, target_width = relief
                            added = 0
                            while (
                                int(
                                    model._growable_layers[
                                        relief_index
                                    ].in_features
                                )
                                < target_width
                            ):
                                try:
                                    grow_layer(
                                        model=model,
                                        train_loader=train_loader,
                                        layer_index=relief_index,
                                        device=device,
                                        line_search_config=(
                                            config.scaling_line_search
                                        ),
                                        optimal_update_kwargs=unified_kwargs,
                                        progress=None,
                                        function_preserving=True,
                                        preservation_tolerance=(
                                            config.fgd_approx
                                            .growth_preservation_tolerance
                                        ),
                                    )
                                except RuntimeError as error:
                                    if progress is not None:
                                        progress(
                                            f"[GRO-WARN] Epoch {epoch}: "
                                            f"bottleneck relief at "
                                            f"{relief_index} stopped: {error}"
                                        )
                                    break
                                added += 1
                            if added:
                                growth_result = GrowthResult(
                                    layer_index=relief_index,
                                    best_scaling_factor=1.0,
                                    best_train_loss=float("nan"),
                                    line_search=[],
                                )
                                layer_index = relief_index
                                selected_layer_index = relief_index
                                if progress is not None:
                                    progress(
                                        f"[GRO] Epoch {epoch}: rank cap "
                                        f"relieved at location "
                                        f"{relief_index}, widened to "
                                        f"{target_width} (+{added} neurons); "
                                        "the minimum is now shared, so the "
                                        "mandate ends"
                                    )
                                widths = [
                                    int(layer.in_features)
                                    for layer in model._growable_layers
                                ]
                                bottlenecks = set(
                                    rank_limiting_locations(widths)
                                )
                        ceiling_binds = (
                            validation_certificate.relative_error is not None
                            and validation_certificate.relative_error
                            >= config.fgd_approx.rel_error_threshold
                        )
                        candidates: list[Candidate] = []
                        trials: dict[tuple[str, int], GrowingMLP] = {}

                        def _certificate_for(trial: GrowingMLP) -> float | None:
                            measured = evaluate_fgd_validation_certificate(
                                model=trial,
                                data_loader=validation_loader,
                                device=device,
                                config=config.fgd_approx,
                                learning_rate=None,
                                probe=fgd_validation_probe,
                            )
                            return measured.relative_error

                        for candidate_layer in growable:
                            trial = copy.deepcopy(model)
                            try:
                                grow_layer(
                                    model=trial,
                                    train_loader=train_loader,
                                    layer_index=candidate_layer,
                                    device=device,
                                    line_search_config=config.scaling_line_search,
                                    optimal_update_kwargs=unified_kwargs,
                                    progress=None,
                                    function_preserving=True,
                                    preservation_tolerance=(
                                        config.fgd_approx
                                        .growth_preservation_tolerance
                                    ),
                                )
                            except RuntimeError as error:
                                # A skipped candidate is a candidate removed
                                # from the search; it must never be silent.
                                if progress is not None:
                                    progress(
                                        f"[GRO-WARN] Epoch {epoch}: width "
                                        f"candidate at {candidate_layer} "
                                        f"could not be built: {error}"
                                    )
                                continue
                            trials[("width", candidate_layer)] = trial
                            candidates.append(
                                Candidate(
                                    kind="width",
                                    index=candidate_layer,
                                    cost=max(
                                        count_parameters(trial)
                                        - base_parameters,
                                        1,
                                    ),
                                    relative_error_after=_certificate_for(trial),
                                    relieves_rank_ceiling=(
                                        candidate_layer in bottlenecks
                                    ),
                                )
                            )

                        for position in range(1, len(model.layers)):
                            trial = copy.deepcopy(model)
                            try:
                                insert_identity_layer(
                                    trial, position=position, device=device
                                )
                            except (ValueError, TypeError):
                                continue
                            trials[("depth", position)] = trial
                            candidates.append(
                                Candidate(
                                    kind="depth",
                                    index=position,
                                    cost=max(
                                        count_parameters(trial)
                                        - base_parameters,
                                        1,
                                    ),
                                    relative_error_after=_certificate_for(trial),
                                )
                            )

                        ranked = rank_candidates(
                            candidates,
                            relative_error_before=(
                                validation_certificate.relative_error
                            ),
                            gradient_sq_norm=(
                                validation_certificate.gradient_sq_norm
                            ),
                            statistical_threshold=(
                                config.fgd_approx.tiny_statistical_threshold
                            ),
                            rank_ceiling_binds=ceiling_binds,
                        )
                        if ranked:
                            # R3: buy the best proposal. Re-measuring after
                            # each purchase would be ideal but doubles the
                            # cost; one purchase per event keeps every step
                            # attributable to a single measured certificate.
                            chosen = ranked[0]
                            model = trials[(chosen.kind, chosen.index)]
                            growth_result = GrowthResult(
                                layer_index=chosen.index,
                                best_scaling_factor=1.0,
                                best_train_loss=float("nan"),
                                line_search=[],
                            )
                            layer_index = chosen.index
                            selected_layer_index = (
                                chosen.index if chosen.kind == "width" else None
                            )
                            if progress is not None:
                                progress(
                                    f"[GRO] Unified growth at epoch {epoch}: "
                                    f"{chosen.kind} at index {chosen.index} "
                                    f"(+{chosen.cost} params, eps "
                                    f"{validation_certificate.relative_error:.3f}"
                                    f" -> {chosen.relative_error_after:.3f}); "
                                    f"{len(candidates)} candidates considered"
                                )
                        else:
                            growth_triggered = False
                            if progress is not None:
                                progress(
                                    f"[GRO] Epoch {epoch}: no width or depth "
                                    "candidate enlarged the reachable set; "
                                    "structure left unchanged"
                                )
                    elif (
                        config.fgd_approx.growth_selection
                        == "expansion_per_parameter"
                    ):
                        # Every candidate NEURON from every location, pooled
                        # and ranked by certified first-order decrease per
                        # parameter it costs. The budget -- what uniform
                        # widening spends per event -- replaces a threshold,
                        # so no tuned constant decides what "worth it" means.
                        alloc_kwargs = tiny_optimal_update_kwargs(
                            config.fgd_approx,
                            compute_delta=config.fgd_approx.growth_compute_delta,
                        )
                        growable = list(
                            range(len(getattr(model, "_growable_layers", [])))
                        )
                        costs = growable_neuron_costs(
                            model, config.data.in_features
                        )
                        measured = [
                            expansion_spectrum(
                                model, train_loader, index, device, alloc_kwargs
                            )
                            for index in growable
                        ]
                        spectra = [item[0] for item in measured]
                        incumbents = [item[1] for item in measured]
                        # No budget: a neuron is admitted iff it buys at
                        # least as much certified first-order decrease per
                        # parameter as the layer's existing weights do by
                        # being re-optimised. Nothing has to be guessed
                        # about an unseen dataset, and the rule
                        # self-terminates as the structure becomes
                        # efficient.
                        allocation = allocate_by_expansion_per_parameter(
                            spectra,
                            costs,
                            incumbents,
                            config.fgd_approx.tiny_statistical_threshold,
                        )
                        growth_result = None
                        for index, neurons in enumerate(allocation):
                            if neurons <= 0:
                                continue
                            growth_result = grow_layer(
                                model=model,
                                train_loader=train_loader,
                                layer_index=growable[index],
                                device=device,
                                line_search_config=config.scaling_line_search,
                                optimal_update_kwargs={
                                    **alloc_kwargs,
                                    "maximum_added_neurons": neurons,
                                },
                                progress=None,
                                function_preserving=(
                                    config.fgd_approx.growth_function_preserving
                                ),
                                preservation_tolerance=(
                                    config.fgd_approx
                                    .growth_preservation_tolerance
                                ),
                                line_search_loader=(
                                    validation_loader
                                    if config.fgd_approx
                                    .growth_scaling_on_validation
                                    else None
                                ),
                            )
                        layer_index = (
                            growth_result.layer_index
                            if growth_result is not None
                            else 0
                        )
                        selected_layer_index = layer_index
                        if progress is not None:
                            progress(
                                f"[GRO] Expansion-per-parameter growth at "
                                f"epoch {epoch}: allocation {allocation} "
                                f"over neuron costs {costs}; growing beat "
                                f"tuning at "
                                f"{sum(1 for a in allocation if a)} of "
                                f"{len(allocation)} locations"
                            )
                    elif (
                        config.fgd_approx.growth_selection
                        == "natural_expansion"
                    ):
                        # SENN's where (arXiv:2307.04526). Its Ingredient 4
                        # adds at the best location and REPEATS -- picking a
                        # single layer per event is what starved the input
                        # layer under R2, so the loop is part of the method,
                        # not an embellishment.
                        #
                        # The addition budget is the number of growable
                        # layers: exactly what uniform growth spends per
                        # event, so the comparison isolates the ALLOCATION
                        # and not the amount. SENN's own stopping rule uses
                        # tuned tau/alpha thresholds, which this flow does
                        # not adopt; the threshold-free part -- stop when no
                        # location buys a first-order decrease -- is kept.
                        senn_kwargs = tiny_optimal_update_kwargs(
                            config.fgd_approx,
                            compute_delta=config.fgd_approx.growth_compute_delta,
                        )
                        growable = list(
                            range(len(getattr(model, "_growable_layers", [])))
                        )
                        growth_result = None
                        allocation: list[int] = []
                        for _ in growable:
                            scores = [
                                rank_layer_expansion_score(
                                    model,
                                    train_loader,
                                    candidate,
                                    device,
                                    senn_kwargs,
                                )
                                for candidate in growable
                            ]
                            best = max(range(len(growable)), key=scores.__getitem__)
                            if scores[best] <= config.fgd_approx.eps:
                                break
                            growth_result = grow_layer(
                                model=model,
                                train_loader=train_loader,
                                layer_index=growable[best],
                                device=device,
                                line_search_config=config.scaling_line_search,
                                optimal_update_kwargs=senn_kwargs,
                                progress=None,
                                function_preserving=(
                                    config.fgd_approx.growth_function_preserving
                                ),
                                preservation_tolerance=(
                                    config.fgd_approx
                                    .growth_preservation_tolerance
                                ),
                                line_search_loader=(
                                    validation_loader
                                    if config.fgd_approx
                                    .growth_scaling_on_validation
                                    else None
                                ),
                            )
                            allocation.append(growable[best])
                        layer_index = (
                            growth_result.layer_index
                            if growth_result is not None
                            else 0
                        )
                        selected_layer_index = layer_index
                        if progress is not None:
                            progress(
                                f"[GRO] SENN expansion-score growth at epoch "
                                f"{epoch}: added at layers {allocation} "
                                f"(scores recomputed after each addition)"
                            )
                    elif config.fgd_approx.growth_uniform:
                        # Uniform growth: widen EVERY hidden layer together,
                        # tracing the balanced dense nets (3xk) from the tiny
                        # start. Sidesteps the greedy input-layer credit
                        # problem. The delta of the last grown layer stands
                        # in as the reported growth_result.
                        growable = list(
                            range(len(getattr(model, "_growable_layers", [])))
                        )
                        uniform_kwargs = tiny_optimal_update_kwargs(
                            config.fgd_approx,
                            compute_delta=config.fgd_approx.growth_compute_delta,
                        )
                        growth_result = None
                        for uniform_layer in growable:
                            growth_result = grow_layer(
                                model=model,
                                train_loader=train_loader,
                                layer_index=uniform_layer,
                                device=device,
                                line_search_config=config.scaling_line_search,
                                optimal_update_kwargs=uniform_kwargs,
                                progress=None,
                                function_preserving=(
                                    config.fgd_approx.growth_function_preserving
                                ),
                                preservation_tolerance=(
                                    config.fgd_approx
                                    .growth_preservation_tolerance
                                ),
                                line_search_loader=(
                                    validation_loader
                                    if config.fgd_approx
                                    .growth_scaling_on_validation
                                    else None
                                ),
                            )
                        layer_index = (
                            growth_result.layer_index
                            if growth_result is not None
                            else 0
                        )
                        selected_layer_index = layer_index
                        if progress is not None:
                            progress(
                                f"[GRO] Uniform growth at epoch {epoch}: "
                                f"widened all {len(growable)} hidden layers"
                            )
                    elif growth_probe is not None:
                        model = growth_probe.model
                        growth_result = growth_probe.result
                        layer_index = growth_result.layer_index
                        selected_layer_index = layer_index
                        if progress is not None:
                            progress(
                                f"[GRO] Committing layer {layer_index} at epoch "
                                f"{epoch}; trial improved the FGD certificate"
                            )
                            for point in growth_result.line_search:
                                progress(
                                    f"  scaling={point.scaling_factor:.6g}, "
                                    f"train_loss={point.train_loss:.4f}"
                                )
                    else:
                        layer_index = (
                            selected_layer_index
                            if selected_layer_index is not None
                            else layer_index_for_growth(
                                growth_count=growth_count,
                                number_hidden_layers=(
                                    config.model.number_hidden_layers
                                ),
                                config=config.growth_schedule,
                            )
                        )
                        growth_result = grow_layer(
                            model=model,
                            train_loader=train_loader,
                            layer_index=layer_index,
                            device=device,
                            line_search_config=config.scaling_line_search,
                            optimal_update_kwargs=tiny_optimal_update_kwargs(
                                config.fgd_approx,
                                compute_delta=(
                                    config.fgd_approx.growth_compute_delta
                                ),
                            ),
                            progress=progress,
                            function_preserving=(
                                config.fgd_approx.growth_function_preserving
                            ),
                            preservation_tolerance=(
                                config.fgd_approx.growth_preservation_tolerance
                            ),
                            line_search_loader=(
                                validation_loader
                                if config.fgd_approx.growth_scaling_on_validation
                                else None
                            ),
                        )
                else:
                    layer_index = layer_index_for_growth(
                        growth_count=growth_count,
                        number_hidden_layers=config.model.number_hidden_layers,
                        config=config.growth_schedule,
                    )
                    if progress is not None:
                        progress(
                            f"[GRO] Growing layer {layer_index} at epoch {epoch}"
                        )
                    growth_result = grow_layer(
                        model=model,
                        train_loader=train_loader,
                        layer_index=layer_index,
                        device=device,
                        line_search_config=config.scaling_line_search,
                        optimal_update_kwargs=None,
                        progress=progress,
                    )
                growth_events.append(growth_result)
                growth_count += 1
                wandb_logger.log_growth_event(
                    event=growth_result,
                    epoch=epoch,
                    growth_count=growth_count,
                )
                last_growth_epoch = epoch
                lr_cycle_start_epoch = epoch
                if config.training.method == "fgd_approx":
                    # Growth is a mode switch: the accumulated stationary and
                    # global bounds certify a fixed architecture, so restart
                    # them from the post-growth loss. The new structure also
                    # re-offers approximation capacity, so all stale family
                    # rejections are cleared immediately.
                    family_rejection_step.clear()
                    fgd_epochs_without_commit = 0
                    reset_fgd_certificate()
                    if config.fgd_approx.learning_rate_policy == "theory_interval":
                        validation_certificate_for_next_epoch = (
                            evaluate_fgd_validation_certificate(
                                model=model,
                                data_loader=validation_loader,
                                device=device,
                                config=config.fgd_approx,
                                learning_rate=None,
                                probe=fgd_validation_probe,
                            )
                        )
                        post_growth_certified_learning_rate = (
                            certified_validation_learning_rate(
                                validation_certificate_for_next_epoch,
                                config.fgd_approx,
                            )
                        )
                        current_fgd_learning_rate = (
                            post_growth_certified_learning_rate
                            if post_growth_certified_learning_rate is not None
                            else 0.0
                        )
                        fgd_max_valid_learning_rate = (
                            post_growth_certified_learning_rate
                        )
                        rel_error = (
                            validation_certificate_for_next_epoch.relative_error
                        )
                        fgd_output_rel_error = (
                            validation_certificate_for_next_epoch.output_relative_error
                        )
                        fgd_learning_rate_upper_bound = (
                            validation_certificate_for_next_epoch.learning_rate_upper_bound
                        )
                        fgd_learning_rate_interval_valid = (
                            validation_certificate_for_next_epoch.learning_rate_interval_valid
                        )
                        fgd_relative_error_condition_valid = (
                            validation_certificate_for_next_epoch.relative_error_condition_valid
                        )
                        fgd_sensor_valid = (
                            validation_certificate_for_next_epoch.sensor_valid
                        )
                        fgd_sensor_invalid_batches = (
                            validation_certificate_for_next_epoch.sensor_invalid_batches
                        )
                optimizer = build_optimizer(model, config.optimizer)
                post_growth_learning_rate = (
                    current_fgd_learning_rate
                    if (
                        config.training.method == "fgd_approx"
                        and config.fgd_approx.learning_rate_policy
                        == "theory_interval"
                    )
                    else scheduled_learning_rate(
                        config,
                        epoch=epoch,
                        cycle_start_epoch=lr_cycle_start_epoch,
                    )
                )
                apply_learning_rate(optimizer, post_growth_learning_rate)

                train_metrics = evaluate_regression_metrics(
                    model,
                    train_loader,
                    loss_function,
                    device=device,
                    accuracy_tolerance=config.training.accuracy_tolerance,
                    classification=classification,
                )
                validation_metrics = evaluate_regression_metrics(
                    model,
                    validation_loader,
                    loss_function,
                    device=device,
                    accuracy_tolerance=config.training.accuracy_tolerance,
                    classification=classification,
                )
                test_metrics = evaluate_regression_metrics(
                    model,
                    test_loader,
                    loss_function,
                    device=device,
                    accuracy_tolerance=config.training.accuracy_tolerance,
                    classification=classification,
                )
                growth_entry = HistoryEntry(
                    step=epoch,
                    step_type="GRO",
                    train_loss=train_metrics.loss,
                    validation_loss=validation_metrics.loss,
                    test_loss=test_metrics.loss,
                    train_accuracy=train_metrics.accuracy,
                    validation_accuracy=validation_metrics.accuracy,
                    test_accuracy=test_metrics.accuracy,
                    learning_rate=current_learning_rate(optimizer),
                    num_params=count_parameters(model),
                    layer_index=layer_index,
                    scaling_factor=growth_result.best_scaling_factor,
                    rel_error=rel_error,
                    selected_layer_index=selected_layer_index,
                    fgd_layer_rel_errors=fgd_layer_rel_errors,
                    fgd_output_rel_error=fgd_output_rel_error,
                    fgd_learning_rate_upper_bound=fgd_learning_rate_upper_bound,
                    fgd_max_valid_learning_rate=fgd_max_valid_learning_rate,
                    fgd_learning_rate_interval_valid=fgd_learning_rate_interval_valid,
                    fgd_learning_rate_clipped_batches=(
                        fgd_learning_rate_clipped_batches
                    ),
                    fgd_skipped_batches=fgd_skipped_batches,
                    fgd_relative_error_condition_valid=(
                        fgd_relative_error_condition_valid
                    ),
                    fgd_loss_descent_valid=fgd_loss_descent_valid,
                    fgd_loss_non_descent_batches=fgd_loss_non_descent_batches,
                    fgd_gradient_sq_norm=fgd_gradient_sq_norm,
                    fgd_min_gradient_sq_norm=fgd_min_gradient_sq_norm,
                    fgd_theory_descent_coefficient=(
                        fgd_theory_descent_coefficient
                    ),
                    fgd_stationary_bound=fgd_stationary_bound,
                    fgd_stationary_bound_valid=fgd_stationary_bound_valid,
                    fgd_global_bound=fgd_global_bound,
                    fgd_global_bound_valid=fgd_global_bound_valid,
                    fgd_global_contraction=fgd_global_contraction,
                    fgd_theory_learning_rate_adjusted=(
                        fgd_theory_learning_rate_adjusted
                    ),
                    fgd_sensor_valid=fgd_sensor_valid,
                    fgd_sensor_invalid_batches=fgd_sensor_invalid_batches,
                    fgd_candidate_accepted=fgd_candidate_accepted,
                    fgd_lr_search_trials=fgd_lr_search_trials,
                    fgd_approximation_kind=fgd_approximation_kind,
                    fgd_rkhs_phase_attempted=fgd_rkhs_phase_attempted,
                    fgd_rkhs_phase_accepted=fgd_rkhs_phase_accepted,
                    fgd_rkhs_phase_steps=fgd_rkhs_phase_steps,
                    fgd_growth_probe_improved=fgd_growth_probe_improved,
                )
                history.append(growth_entry)
                wandb_logger.log_history_entry(growth_entry)

                if progress is not None:
                    delta = test_metrics.loss - last_test_loss
                    progress(
                        f"[GRO] Epoch {epoch}, train_loss={train_metrics.loss:.4f}, "
                        f"validation_loss={validation_metrics.loss:.4f}, "
                        f"test_loss={test_metrics.loss:.4f} ({delta:+.4f}), "
                        f"train_acc={train_metrics.accuracy:.3f}, "
                        f"validation_acc={validation_metrics.accuracy:.3f}, "
                        f"test_acc={test_metrics.accuracy:.3f}, "
                        f"scaling={growth_result.best_scaling_factor:.4g}"
                    )
                    progress("Model after growing:")
                    progress(str(model))
                last_test_loss = test_metrics.loss

        result = PipelineResult(
            config=config,
            history=history,
            growth_events=growth_events,
            model=model,
            device=str(device),
        )
        wandb_logger.finish(history=history)
        return result
    except Exception:
        wandb_logger.abort()
        raise


def result_payload(result: PipelineResult) -> dict[str, Any]:
    return {
        "config": config_payload(result.config),
        "device": result.device,
        "model": str(result.model),
        "history": [asdict(entry) for entry in result.history],
        "growth_events": [asdict(event) for event in result.growth_events],
    }


def save_result_json(result: PipelineResult, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_payload(result), indent=2),
        encoding="utf-8",
    )
    return output_path


def write_outputs(result: PipelineResult) -> dict[str, Path]:
    """Write JSON and optional plot outputs declared by the config."""
    run_config = result.config.run
    output_paths: dict[str, Path] = {}

    history_path = run_config.results_dir / f"{run_config.name}_history.json"
    output_paths["history"] = save_result_json(result, history_path)

    if run_config.save_plot:
        from stable_tiny.plotting import (
            plot_history,
            plot_parameters,
            plot_relative_error,
        )

        plot_path = run_config.results_dir / f"{run_config.name}_metrics.png"
        saved_plot = plot_history(
            result.history,
            output_path=plot_path,
            show=run_config.show_plot,
        )
        if saved_plot is not None:
            output_paths["metrics_plot"] = saved_plot

        parameters_path = run_config.results_dir / f"{run_config.name}_parameters.png"
        saved_parameters_plot = plot_parameters(
            result.history,
            output_path=parameters_path,
            show=run_config.show_plot,
        )
        if saved_parameters_plot is not None:
            output_paths["parameters_plot"] = saved_parameters_plot

        rel_error_path = run_config.results_dir / f"{run_config.name}_rel_error.png"
        saved_rel_error_plot = plot_relative_error(
            result.history,
            output_path=rel_error_path,
            show=run_config.show_plot,
            threshold=result.config.fgd_approx.rel_error_threshold,
        )
        if saved_rel_error_plot is not None:
            output_paths["rel_error_plot"] = saved_rel_error_plot

    return output_paths
