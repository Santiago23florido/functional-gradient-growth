"""Pipeline that joins config, data, train, grow, and outputs."""

from __future__ import annotations

import copy
import json
import math
import time
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
    _projection_step_sensor_reason,
    _trainable_named_parameters,
    batch_functional_loss,
    build_projection_probe,
    FUNCTIONAL_HAS_PL_CONSTANT,
    certificate_from_projection_stats,
    compute_expressivity_bottlenecks,
    evaluate_fgd_validation_certificate,
    evaluate_secant_validation_certificate,
    measure_direction_projection,
    functional_gradient,
    select_tiny_growth_layer_index,
    should_trigger_fgd_growth,
    theoretical_descent_coefficient,
    theoretical_learning_rate_upper_bound,
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
from fgdlib.profile import fallback, increment, timed
from fgdlib.search.certify import (
    exact_relative_error,
    grow_until_certified,
)
from fgdlib.search.families import (
    certify_parametric_step_swept,
)
from fgdlib.search.nonlinear import (
    NonlinearCandidate,
    NonlinearCertificateStats,
    build_nonlinear_probe,
    search_interpolated_step,
    stream_nonlinear_certificate,
    train_nonlinear_candidate,
)
from fgdlib.search.depth import insert_identity_layer
from fgdlib.search.unified import (
    Candidate,
    bottleneck_relief_target,
    rank_candidates,
    rank_candidates_by_certified_gain,
    rank_limiting_locations,
)
from fgdlib.search.damping import select_projection_damping
from fgdlib.search.realize import realization_damping, realize_functional_step
from fgdlib.search.linearization import certified_linear_learning_rate
from fgdlib.search.growth import (
    GrowthResult,
    ScalingLineSearchConfig,
    allocate_by_expansion_per_parameter,
    expansion_spectrum,
    grow_layer,
    growable_neuron_costs,
    rank_layer_expansion_score,
)
from fgdlib.search.schedule import (
    GrowthScheduleConfig,
    layer_index_for_growth,
    should_grow,
)
from fgdlib.training_utils.lr_scheduler import (
    LRSchedulerConfig,
    apply_learning_rate,
    learning_rate_for_epoch,
)
from fgdlib.training_utils.optim import OptimizerConfig, build_optimizer, current_learning_rate
from fgdlib.training_utils.loop import (
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
    # Dropout on the hidden layers' post-activation. 0.0 (default) builds the
    # plain MLP byte-identical, so MNIST is untouched. Certification-safe: the
    # certificate runs in eval where dropout is the identity; it only
    # regularizes the family training steps. See fgdlib/models/regularized_mlp.py.
    dropout_rate: float = 0.0
    # Per-feature batch normalization on the hidden layers. Default off, so
    # MNIST is byte-identical. It IS part of f (not eval-transparent like
    # dropout) but is function-preservingly growable because it is
    # per-feature; LayerNorm is intentionally unavailable (it couples features
    # and breaks preservation). See fgdlib/models/regularized_mlp.py.
    use_batchnorm: bool = False
    # Declarative architecture. When set, the model is built component by
    # component from this list instead of the uniform hidden_size /
    # number_hidden_layers + use_batchnorm / dropout_rate shorthand, so batch-
    # norm and dropout can be placed exactly where wanted. Each `mlp` is a
    # block (width, num_layers); widths may differ between blocks, and
    # `batchnorm` / `dropout` attach to the mlp above. Every mlp layer is
    # growable. See fgdlib/models/stack.py. Example:
    #   stack:
    #     - {mlp: [2, 1]}
    #     - batchnorm
    #     - {mlp: [2, 1]}
    #     - {dropout: 0.2}
    stack: tuple[Any, ...] | None = None


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
    nonlinear_functional_learning_rate: float | None = None
    nonlinear_inner_steps: int | None = None
    nonlinear_adamw_learning_rate: float | None = None
    nonlinear_weight_decay: float | None = None
    nonlinear_cosine: float | None = None
    nonlinear_relative_error: float | None = None
    nonlinear_certificate_valid: bool | None = None
    nonlinear_validation_descent_valid: bool | None = None
    nonlinear_committed_rate: float | None = None
    #: Interpolation actually committed. Distinct from the eta_f that generated
    #: the target and from the realized secant rate below.
    nonlinear_committed_alpha: float | None = None
    #: eta* = <Delta, r>/|r|^2 of the displacement that was APPLIED.
    nonlinear_effective_secant_rate: float | None = None
    #: Best cosine reached at this structure, over every candidate tried.
    nonlinear_best_cosine: float | None = None
    nonlinear_candidate_optimizer_steps: int | None = None
    nonlinear_candidate_epochs: float | None = None
    nonlinear_candidate_batches_seen: int | None = None
    nonlinear_candidate_examples_seen: int | None = None
    nonlinear_candidate_initial_objective: float | None = None
    nonlinear_candidate_final_objective: float | None = None
    nonlinear_candidate_objective_reduction: float | None = None
    nonlinear_candidate_parameter_displacement_norm: float | None = None
    nonlinear_growth_requested: bool = False
    nonlinear_candidate_training_seconds: float = 0.0
    nonlinear_certification_seconds: float = 0.0
    nonlinear_growth_statistics_seconds: float = 0.0
    nonlinear_growth_application_seconds: float = 0.0
    nonlinear_ladder_attempts: int = 0
    nonlinear_accepted_steps: int = 0
    nonlinear_failed_ladders: int = 0
    nonlinear_growth_events: int = 0
    nonlinear_full_jacobian_calls: int = 0
    nonlinear_tangent_system_calls: int = 0
    nonlinear_tangent_projection_solves: int = 0
    architecture_widths: tuple[int, ...] = ()


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
class _NonlinearDirectionalCertificate:
    """Lemma 3.5 conditions derived only from a streamed nonlinear secant."""

    learning_rate_upper_bound: float | None
    max_valid_learning_rate: float | None
    learning_rate_interval_valid: bool | None
    skipped_batches: int
    relative_error_condition_valid: bool | None
    gradient_sq_norm: float | None
    theory_descent_coefficient: float | None
    relative_error: float | None
    sensor_valid: bool
    sensor_invalid_batches: int
    non_finite_quantities: tuple[str, ...] = ()


@dataclass(frozen=True)
class _FGDTrial:
    model: GrowingMLP
    epoch_result: FGDApproxEpochResult
    certificate: FGDValidationCertificate | _NonlinearDirectionalCertificate
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
class _NonlinearPrimaryResult:
    """One nonlinear-only ladder outcome at the current structure."""

    accepted: _FGDTrial | None
    last_trial: _FGDTrial | None
    certificate: _NonlinearDirectionalCertificate
    stats: NonlinearCertificateStats | None
    candidate: NonlinearCandidate | None
    attempts: int
    candidate_training_seconds: float
    certification_seconds: float
    update_norm: float | None
    #: Interpolation actually committed, or ``None`` when nothing certified.
    committed_alpha: float | None = None
    #: Best cosine any candidate reached at this structure. Kept so a failed
    #: ladder reports how close it came instead of only its last attempt.
    best_cosine: float | None = None


@dataclass(frozen=True)
class _NonlinearGrowthOutcome:
    model: GrowingMLP | None
    result: GrowthResult | None
    layer_index: int | None
    statistics_seconds: float
    application_seconds: float
    preservation_valid: bool | None = None


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

    if section_type is FGDApproxConfig and "certify_family_functional_lrs" in values:
        values["certify_family_functional_lrs"] = tuple(
            float(value)
            for value in values["certify_family_functional_lrs"] or ()
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


def with_model_overrides(
    config: PipelineConfig,
    *,
    model_seed: int | None = None,
) -> PipelineConfig:
    """Return a config with command-line model overrides applied."""
    model_config = config.model
    if model_seed is not None:
        model_config = replace(model_config, model_seed=int(model_seed))
    return replace(config, model=model_config)


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


def _functional_tikhonov_probe(
    probe: tuple[torch.Tensor, torch.Tensor] | None,
    config: FGDApproxConfig,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Shrink a probe's targets to certify against L_data + gamma ||f||^2.

    For sum-MSE the regularised functional gradient r_gamma = 2((1+gamma)f - y)
    points exactly like the plain MSE gradient toward y/(1+gamma), and the
    certified parameter step is identical to plain MSE toward that shrunk
    target with L_s = 2. So functional Tikhonov is precisely this one
    operation on the direction probe; everything downstream -- growth,
    projection, realisation, GCV -- inherits it, while the reported metrics
    stay against the true targets in the loaders.

    Only for the sum-MSE functional: cross-entropy's ||f||^2 penalty does not
    reduce to a target shift, so gamma is ignored there rather than applied
    incorrectly.
    """
    if probe is None or config.functional_tikhonov_gamma <= 0.0:
        return probe
    if config.functional_loss != "mse":
        return probe
    x, y = probe
    return x, y / (1.0 + config.functional_tikhonov_gamma)


#: Cache of the certification rank per model, keyed by ``id(model)`` and its
#: current parameter count, so the rank is re-estimated only when the structure
#: has actually grown (once per growth, not once per outer step).
_CERTIFICATION_RANK_CACHE: dict[int, tuple[int, int]] = {}


def _estimate_certification_rank(
    config: PipelineConfig,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    sizing_batches: int,
) -> int | None:
    """Numerical rank of ``J`` on a small sizing probe -- the interpolation floor.

    eps collapses to a spurious 0 once the tangent space can represent ANY
    residual on the probe, i.e. once ``NK <= rank(J)`` (``J`` full row rank).
    The binding quantity is therefore ``rank(J)``, NOT the parameter count ``P``:
    on an input-heavy net (CIFAR: a 1024->2 first layer) ``rank(J) = 536 << P =
    2092``, so sizing by ``P`` over-samples ~4x and needlessly slows the where.
    This measures the numerical rank directly on a bounded probe.
    """
    from fgdlib.tangent import exact_tangent_system

    x, y = build_projection_probe(loader, sizing_batches, device)
    system = exact_tangent_system(model, x, y, config.fgd_approx)
    if system is None or system.jacobian.numel() == 0:
        return None
    singular_values = torch.linalg.svdvals(system.jacobian.to(torch.float32))
    if singular_values.numel() == 0:
        return None
    largest = float(singular_values.max())
    if not largest > 0.0:
        return None
    tolerance = largest * max(system.jacobian.shape) * 1e-6
    return int((singular_values > tolerance).sum())


def _bounded_probe_batches(
    config: PipelineConfig,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    current_batches: int | None = None,
) -> int:
    """Probe size for the certificate, bounded to the numerical rank of ``J``.

    Returns the configured ``probe_batches`` unchanged when
    ``certify_probe_kappa`` is 0 (every synthetic run). When positive the probe
    is sized so the number of Jacobian ROWS is ``NK = kappa * rank(J)``: enough
    to certify (MEASURED on the K=1 synthetic: ``NK ~ 8 rank`` matches the whole-
    dataset eps to +-0.05) and, crucially, kept ABOVE the interpolation floor
    ``NK > rank(J)`` where eps collapses to a spurious 0. ``rank(J)`` (not ``P``)
    is the right basis -- it is what actually bounds full-row-rank -- and it is
    typically ``<< P``, which is what keeps the where FAST at CIFAR scale.

    Capped at the dataset, so on a small dataset this is just the whole training
    set; it only shrinks the probe where the dataset is genuinely larger,
    decoupling the ``O(NK*P^2)`` where cost from the dataset size. ``rank(J)``
    grows during training, so it is re-estimated (cached per structure) and the
    probe re-materialised as it grows -- the floor holds for the CURRENT net.
    """
    fa = config.fgd_approx
    kappa = float(getattr(fa, "certify_probe_kappa", 0.0) or 0.0)
    if kappa <= 0.0:
        return fa.probe_batches
    batch_size = max(1, int(config.data.batch_size))
    out_features = max(1, int(getattr(config.data, "out_features", 1)))
    try:
        max_batches = len(loader)
    except TypeError:
        max_batches = fa.probe_batches
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Sizing probe: the current probe (grows reactively), starting from the
    # configured probe_batches. rank(J) <= its NK, so if rank saturates it the
    # sizing probe is grown to re-measure on the next step.
    base_batches = current_batches or fa.probe_batches or 4
    sizing_batches = max(1, min(int(base_batches), max_batches))

    cached = _CERTIFICATION_RANK_CACHE.get(id(model))
    if cached is not None and cached[0] == n_params:
        rank = cached[1]
    else:
        rank = _estimate_certification_rank(
            config, model, loader, device, sizing_batches
        )
        _CERTIFICATION_RANK_CACHE[id(model)] = (n_params, rank if rank else -1)
    if not rank or rank <= 0:
        # Could not measure: fall back to the conservative P bound rather than
        # risk an under-sized (interpolating) probe.
        rank = n_params

    sizing_rows = out_features * sizing_batches * batch_size
    if rank >= 0.6 * sizing_rows and sizing_batches < max_batches:
        # rank saturates the sizing probe -> it is under-measured; grow the probe
        # so the next estimate is unclamped.
        return min(max_batches, max(sizing_batches + 1, math.ceil(1.7 * sizing_batches)))

    target_rows = max(kappa, 2.0) * float(rank)   # NK > rank, no interpolation
    batches = math.ceil(target_rows / out_features / batch_size)
    return max(1, min(batches, max_batches))


def build_model(config: PipelineConfig, device: torch.device) -> GrowingMLP:
    data_config = config.data
    model_config = config.model
    torch.manual_seed(model_config.model_seed)
    import torch.nn as nn

    if model_config.stack is not None:
        # Declarative stack: place mlp / dropout / batchnorm exactly where the
        # config asks. Supersedes the uniform shorthand below.
        from fgdlib.models.stack import build_stack_model

        return build_stack_model(
            stack=list(model_config.stack),
            in_features=data_config.in_features,
            out_features=data_config.out_features,
            device=device,
        )

    kwargs: dict[str, Any] = {}
    if model_config.dropout_rate > 0.0 and not model_config.use_batchnorm:
        # Dropout-only: a single shared post-function suffices because dropout
        # is stateless. GroMo's GrowingMLP applies `activation` as each hidden
        # layer's post_layer_function (never on the output), so this
        # regularizes exactly the hidden representations. With rate 0 and no
        # batch-norm this branch is skipped and the model is byte-identical to
        # the plain MLP (the MNIST-preservation bar).
        from fgdlib.models.regularized_mlp import make_post_layer_function

        kwargs["activation"] = make_post_layer_function(
            nn.SELU(), model_config.dropout_rate
        )
    model = GrowingMLP(
        in_features=data_config.in_features,
        out_features=data_config.out_features,
        hidden_size=model_config.hidden_size,
        number_hidden_layers=model_config.number_hidden_layers,
        device=device,
        **kwargs,
    )
    if model_config.use_batchnorm:
        # Batch-norm needs a PER-LAYER instance (its own running statistics),
        # so replace each hidden layer's post-function individually. The
        # output layer (last) is never regularized. Growth keeps each norm in
        # sync via sync_normalization in grow_layer.
        from fgdlib.models.regularized_mlp import make_hidden_post_function

        for layer in list(model.layers)[:-1]:
            layer.post_layer_function = make_hidden_post_function(
                num_features=int(layer.out_features),
                activation=nn.SELU(),
                dropout_rate=model_config.dropout_rate,
                device=device,
            )
    return model


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
    certificate: FGDValidationCertificate | _NonlinearDirectionalCertificate,
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


def _certify_force_growth(
    config: FGDApproxConfig,
    *,
    previous_step_committed: bool,
    previous_failure_non_finite: bool,
) -> bool:
    """Decide grow_until_certified's `force`, discriminating WHY it failed.

    See FGDApproxConfig.certify_force_growth_on_finite_step_failure for
    the two measured incidents this balances: forcing unconditionally
    over-fires on a non-finite validation measurement (the model
    overflowing on unseen data, an overfitting symptom growth worsens);
    never forcing reintroduces the MNIST deadlock (eps certified, no rate
    produced held-out descent, nothing ever committed). Default False
    reproduces `force=False` unconditionally -- bit-identical to today.
    """
    if not config.certify_force_growth_on_finite_step_failure:
        return False
    if previous_step_committed:
        return False
    if previous_failure_non_finite:
        increment("certify_force_suppressed_nonfinite")
        return False
    increment("certify_forced_growths")
    return True


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


def lemma35_learning_rate(
    relative_error: float | None,
    config: FGDApproxConfig,
) -> float | None:
    """``theory_lr_safety * eta_bar(eps)`` -- 0.95 of the admissible interval.

    ``relative_error`` must be the eps that is actually CERTIFIED, i.e. the
    one measured on the probe the direction was solved on and driven below
    ``rel_error_threshold`` by the grow loop. That is the eps Lemma 3.5
    speaks about, and ``eta_bar(eps) = 2(1 - 2 eps) / (L_s (1 + 2 eps))`` is
    its interval.

    Returns ``None`` exactly when the certificate does not hold -- eps at or
    above the threshold, unmeasured, or an interval so degenerate that no
    rate sits strictly inside ``(theory_lr_min, eta_bar)``. That is the
    signal to grow, not to step.
    """
    if relative_error is None:
        return None
    if not relative_error < min(config.rel_error_threshold, 0.5):
        return None
    upper_bound = theoretical_learning_rate_upper_bound(relative_error, config)
    if upper_bound is None:
        return None
    # eta_bar is where Lemma 3.5's bracket VANISHES, so the guaranteed
    # decrease eta * bracket(eta) -- a parabola -- peaks at eta_bar/2. Taking
    # a fraction close to 1 maximises the step and minimises what it buys.
    fraction = 0.5 if config.certify_optimal_rate else config.theory_lr_safety
    learning_rate = fraction * upper_bound
    if learning_rate <= config.theory_lr_min + config.eps:
        return None
    return learning_rate


def _apply_lemma35_step(
    *,
    relative_error: float | None,
    evaluate_trial: Callable[[float], _FGDTrial],
    config: FGDApproxConfig,
    learning_rate: float | None = None,
    model: GrowingMLP | None = None,
    probe_inputs: torch.Tensor | None = None,
    updates: tuple[torch.Tensor, ...] | None = None,
    progress: Callable[[str], None] | None = None,
) -> _FGDSearchResult:
    """Take the certified step and commit it -- assume the lemma, don't check it.

    The cycle this serves: grow until eps < 1/2, then train with the tangent
    approximation at ``eta = 0.95 * eta_bar(eps)``, and keep training until
    eps stops satisfying the criterion -- at which point the caller grows
    again. Nothing else is verified. The additional condition is ASSUMED to
    hold, on the grounds that both of its premises do: the rate lies inside
    the admissible interval and the relative error satisfies the criterion.

    The ordinary path (:func:`_search_fgd_certified_trial`) instead sweeps
    rates downward from the bound and keeps the largest whose step is
    *observed* to descend on held-out data. That gate is what deadlocked
    this flow -- eps = 0.475 certified the structure so no growth fired,
    while no rate produced held-out descent so no step committed, and
    epochs 2, 3 and 4 came out bit-identical.

    Deriving the rate from the HELD-OUT certificate instead deadlocks it a
    second way, which is worth recording because it looks like a fix: that
    eps measures ~1.0 (on validation the direction is a secant, not a
    projection), so ``eta_bar`` is undefined there, no rate is produced and
    the flow sits still again. Only the certified eps has an interval at
    all.

    What still gates is the finiteness sensor. That is not a condition of
    the theory but of the arithmetic: a non-finite loss or a skipped batch
    means the measurement itself is meaningless.
    """
    if learning_rate is None:
        learning_rate = lemma35_learning_rate(relative_error, config)
    elif lemma35_learning_rate(relative_error, config) is None:
        # A rate supplied from outside never overrides the certificate: if
        # eps fails the criterion there is no admissible rate to supply.
        learning_rate = None
    else:
        # Already scored against the linearisation control by the selection
        # that produced it, so it is applied as given.
        return _finish_lemma35_step(learning_rate, evaluate_trial)
    if learning_rate is None:
        # The relative-error criterion is not satisfied, so this is where
        # the cycle turns: no step, grow instead. The caller's grow loop
        # runs before every outer step, so that happens on the next pass.
        return _FGDSearchResult(None, None, 0, False)

    if (
        config.certify_linearization_tolerance is not None
        and model is not None
        and probe_inputs is not None
        and updates is not None
    ):
        # Enforce the lemma's own hypothesis: that theta - eta u really is
        # the function-space step f - eta g the theorem is about. Narrows
        # eta INSIDE the certified interval; never enlarges it, never looks
        # at the loss.
        linearized = certified_linear_learning_rate(
            model, probe_inputs, updates, learning_rate, config
        )
        if progress is not None:
            progress(
                f"[LINEAR] eta {learning_rate:.4g} -> "
                + (
                    f"{linearized.learning_rate:.4g}"
                    if linearized.learning_rate is not None
                    else "none"
                )
                + f" (defect {linearized.defect:.3e}, "
                f"{linearized.backtracks} backtracks)"
            )
        if linearized.learning_rate is None:
            # No admissible rate puts this direction inside the regime the
            # lemma describes. The structure, not the step size, is what has
            # to change -- so take no step and let the grow loop act.
            return _FGDSearchResult(None, None, 0, False)
        learning_rate = linearized.learning_rate

    return _finish_lemma35_step(learning_rate, evaluate_trial)


def _finish_lemma35_step(
    learning_rate: float,
    evaluate_trial: Callable[[float], _FGDTrial],
) -> _FGDSearchResult:
    """Evaluate the certified rate once and commit it.

    Nothing here rejects, and that is deliberate. The sensors this used to
    consult are BOTH the held-out one: ``trial.certificate`` is built from
    ``direction_stats``, measured on the validation probe, and
    ``epoch_result.sensor_valid`` is copied from that same certificate --
    there is no train-side sensor in a trial at all. With the projector
    invariants off, as they are there, that sensor tests only finiteness, so
    failing it says the MODEL produced non-finite values on unseen data.

    That is a statement about the fit, not about admissibility. Admissibility
    is decided upstream, on the TRAIN probe, which is the sample Lemma 3.5
    speaks about -- and a non-finite measurement there already blocks the
    step, because ``lemma35_learning_rate`` withholds a rate whenever ``eps``
    fails ``eps < 1/2`` and NaN fails it.

    Letting the held-out sensor reject was the deadlock in its final form:
    MEASURED, ``eps = 0.4808`` certified so no growth fired, while this
    sensor rejected every step, and epochs 85-92 came out bit-identical at
    loss 0.1092 -- 1 committed step against 21 growths. The earlier ``force``
    workaround papered over it by buying capacity, which is the one response
    guaranteed to make an overflowing model overflow harder.
    """
    trial = evaluate_trial(learning_rate)
    return _FGDSearchResult(
        trial, trial, 1, not trial.certificate.sensor_valid
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
    device: torch.device,
    config: PipelineConfig,
) -> bool:
    """Generalised R1: is the structure inadequate despite eps < 1/2?

    Asks the only question that separates "this small net is enough" (MNIST)
    from "this net is rank-limited and grinding" (CIFAR): after the SAME
    amount of further training, does a structure that grew the bottleneck
    reach a strictly lower relative error than one that did not?

    The comparison must be apples-to-apples. More capacity almost always
    lowers eps a little, so comparing a grown-and-trained clone against the
    untrained current point trivially favours growth -- and does so even on
    MNIST, where the structure is already adequate. The fair test trains
    BOTH a grown clone and a stay clone for the same
    ``growth_lookahead_steps`` and compares them: growth is warranted only
    when growing reaches a lower eps than simply training longer would. That
    is exactly "the structure, not the training, is the binding constraint".

    The bottleneck is the width-minimum growable location -- ``rank J <=
    min_l w_l`` -- grown function-preservingly so the comparison isolates
    capacity, not a lucky re-initialisation. The margin reuses
    ``tiny_statistical_threshold``: no new constant, and numerical jitter
    cannot force endless growth.

    Returns True only when growth beats training. With the flag off (the
    default) this is never called and behaviour is unchanged.
    """
    widths = [int(layer.in_features) for layer in model._growable_layers]
    limiting = rank_limiting_locations(widths)
    if not limiting:
        return False
    bottleneck = limiting[0]
    steps = config.fgd_approx.growth_lookahead_steps

    def _epsilon_after_training(candidate: GrowingMLP) -> float | None:
        trained = _train_parametric_gd_candidate(
            base_model=candidate,
            train_batches=train_batches,
            device=device,
            functional_learning_rate=(
                config.parametric_descent.functional_learning_rates[0]
            ),
            steps=steps,
            config=config.parametric_descent,
            functional_loss=config.fgd_approx.functional_loss,
        )
        if trained is None:
            return None
        certificate = evaluate_fgd_validation_certificate(
            model=trained,
            data_loader=validation_loader,
            device=device,
            config=config.fgd_approx,
            learning_rate=None,
            probe=probe,
        )
        return certificate.relative_error

    # Stay: the current structure, trained the same number of steps.
    epsilon_stay = _epsilon_after_training(copy.deepcopy(model))
    if epsilon_stay is None:
        return False

    # Grow: the same, after relieving the bottleneck.
    grown = copy.deepcopy(model)
    optimal_update_kwargs = tiny_optimal_update_kwargs(
        config.fgd_approx, compute_delta=config.fgd_approx.growth_compute_delta
    )
    try:
        grow_layer(
            model=grown,
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
    epsilon_grow = _epsilon_after_training(grown)
    if epsilon_grow is None:
        return False

    margin = config.fgd_approx.tiny_statistical_threshold
    return epsilon_grow < epsilon_stay * (1.0 - margin)


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



def _wake_dormant_outgoing_weights(model: GrowingMLP, scale: float) -> None:
    """Break the w = 0 degeneracy on a DISPOSABLE scoring clone.

    A function-preserving neuron enters with its outgoing weight at zero, so
    df/d(its incoming weights) = 0 and those Jacobian columns are identically
    zero. Scoring there measures the neuron at its most useless moment, and
    unevenly: the closer the layer sits to a narrow output, the fewer live
    columns it has. Nudging the exactly-zero outgoing weights to `scale`
    activates the latent columns while leaving f, and therefore the residual,
    essentially unchanged.

    Only ever called on a clone that is thrown away after scoring.
    """
    with torch.no_grad():
        for layer in getattr(model, "layers", []):
            inner = getattr(layer, "layer", None)
            weight = getattr(inner, "weight", None)
            if weight is None or weight.ndim != 2:
                continue
            dormant = weight.abs().sum(dim=0) == 0
            if bool(dormant.any()):
                # Distinct directions, not one shared value. Assigning the
                # SAME scalar to every dormant column makes the new neurons'
                # outgoing weights identical, so their Jacobian columns are
                # collinear: that trades the "all zero" degeneracy for an
                # "all equal" one of rank 1. It is why a k-block lookahead
                # measured barely more than a single block -- the horizon it
                # probed spanned one direction, not k.
                generator = torch.Generator(device="cpu").manual_seed(
                    int(dormant.sum()) * 1000 + int(weight.shape[0])
                )
                noise = torch.randn(
                    (weight.shape[0], int(dormant.sum())),
                    generator=generator,
                ).to(weight.device, weight.dtype)
                noise = noise / noise.norm(dim=0, keepdim=True).clamp_min(1e-12)
                weight[:, dormant] = scale * noise


def _prune_negligible_units(
    model: GrowingMLP,
    probe: tuple[torch.Tensor, torch.Tensor],
    tolerance: float,
) -> int:
    """Drop hidden units whose removal moves f by less than `tolerance`.

    Growth is only allowed to change f by `growth_preservation_tolerance`;
    removing a unit that moves f by less than that same tolerance is
    function-preserving by exactly the same standard, so the certificate and
    the family ladder see the same function either way.

    This is the only mechanism here that lets capacity MOVE. Growth is
    otherwise greedy and monotone from 2-2-2: whatever the first events decide
    is sunk, and the reachable architectures at a given size are one path
    through the space rather than the space. Freeing a unit that stopped
    earning its place lets a later event buy it back somewhere it is worth
    more.

    MEASURED AND REFUTED as a source of improvement, but the negative is the
    informative part: across three N=1024 seeds this fired ZERO times, and the
    runs came out bit-identical to lookahead alone (0.921/0.899/0.903 at
    875/875/451). The helper is not broken -- given a unit with an
    exactly-zero outgoing column it removes it and leaves f untouched -- there
    simply is no dead capacity: after training, every unit earns more than the
    tolerance.

    That makes the tension explicit. Function preservation is WHY the search is
    monotone; moving capacity between layers requires changing f by more than
    the tolerance, which is precisely what the method forbids. So within the
    fixed constraints the search cannot be made non-monotone, and the only
    remaining lever is the quality of the measurement at the moment of an
    irreversible decision -- which is what lookahead improves.

    Returns the number of units removed.
    """
    layers = getattr(model, "layers", [])
    if len(layers) < 2:
        return 0
    x = probe[0]
    removed = 0
    with torch.no_grad():
        reference = model(x)
        for index in range(len(layers) - 1):
            inner = getattr(layers[index], "layer", None)
            nxt = getattr(layers[index + 1], "layer", None)
            if inner is None or nxt is None:
                continue
            width = inner.weight.shape[0]
            # Never prune a layer down to nothing: a width-zero layer is not a
            # narrower model, it is a disconnected one.
            keep = list(range(width))
            for unit in range(width):
                if len(keep) <= 1:
                    break
                saved = nxt.weight[:, unit].clone()
                nxt.weight[:, unit] = 0.0
                drift = float(torch.max(torch.abs(model(x) - reference)))
                if drift <= tolerance:
                    keep.remove(unit)
                else:
                    nxt.weight[:, unit] = saved
            if len(keep) == width:
                continue
            idx = torch.tensor(keep, device=inner.weight.device)
            new_in = torch.nn.Linear(
                inner.weight.shape[1], len(keep)
            ).to(inner.weight.device)
            new_in.weight.copy_(inner.weight[idx])
            if inner.bias is not None and new_in.bias is not None:
                new_in.bias.copy_(inner.bias[idx])
            new_out = torch.nn.Linear(
                len(keep), nxt.weight.shape[0]
            ).to(nxt.weight.device)
            new_out.weight.copy_(nxt.weight[:, idx])
            if nxt.bias is not None and new_out.bias is not None:
                new_out.bias.copy_(nxt.bias)
            layers[index].layer = new_in
            layers[index + 1].layer = new_out
            removed += width - len(keep)
    return removed


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


def _invalid_nonlinear_certificate() -> _NonlinearDirectionalCertificate:
    return _NonlinearDirectionalCertificate(
        learning_rate_upper_bound=None,
        max_valid_learning_rate=None,
        learning_rate_interval_valid=None,
        skipped_batches=0,
        relative_error_condition_valid=None,
        gradient_sq_norm=None,
        theory_descent_coefficient=None,
        relative_error=None,
        sensor_valid=False,
        sensor_invalid_batches=1,
        non_finite_quantities=("nonlinear_candidate",),
    )


def _nonlinear_directional_certificate(
    *,
    stats: NonlinearCertificateStats,
    learning_rate: float | None,
    config: FGDApproxConfig,
) -> _NonlinearDirectionalCertificate:
    """Build the nonlinear certificate directly from streaming scalars."""
    if not stats.sensor_valid or stats.relative_error is None or stats.cosine is None:
        return _invalid_nonlinear_certificate()

    upper_bound = theoretical_learning_rate_upper_bound(
        stats.relative_error,
        config,
    )
    safe_upper_bound = (
        config.theory_lr_safety * upper_bound if upper_bound is not None else None
    )
    interval_valid = bool(
        learning_rate is not None
        and safe_upper_bound is not None
        and safe_upper_bound > config.theory_lr_min + config.eps
        and learning_rate > config.theory_lr_min
        and learning_rate < safe_upper_bound + config.eps
    )
    descent_coefficient = (
        theoretical_descent_coefficient(
            stats.relative_error,
            learning_rate,
            config,
        )
        if interval_valid and learning_rate is not None
        else None
    )
    return _NonlinearDirectionalCertificate(
        learning_rate_upper_bound=upper_bound,
        max_valid_learning_rate=(safe_upper_bound if interval_valid else None),
        learning_rate_interval_valid=interval_valid,
        skipped_batches=int(not interval_valid),
        relative_error_condition_valid=stats.certified,
        gradient_sq_norm=stats.gradient_sq_norm,
        theory_descent_coefficient=descent_coefficient,
        relative_error=stats.relative_error,
        sensor_valid=True,
        sensor_invalid_batches=0,
    )


def _parameter_displacement_norm(
    base_model: torch.nn.Module,
    moved_model: torch.nn.Module,
) -> float:
    squared = 0.0
    with torch.no_grad():
        for base, moved in zip(base_model.parameters(), moved_model.parameters()):
            squared += float(torch.sum((moved - base).double().square()))
    return math.sqrt(squared)


def _nonlinear_certification_source(
    *,
    split: str,
    train_probe: tuple[torch.Tensor, torch.Tensor] | None,
    validation_loader,
):
    """Iterable of ``(x, y)`` the directional certificate is measured over.

    Under ``certificate_split="train"`` this is the SAME probe the candidate
    was trained on -- the ladder's semantics, where the certificate is a
    statement about the empirical objective the step is defined on. Under
    ``"validation"`` it is the validation loader, which is a generalization
    claim and a strictly different guarantee.
    """
    if split == "train":
        if train_probe is None:
            raise ValueError(
                "parametric_gd.certificate_split='train' requires a train probe."
            )
        return [(train_probe[0], train_probe[1])]
    return validation_loader


def _search_nonlinear_primary_candidate(
    *,
    base_model: GrowingMLP,
    train_loader,
    validation_loader,
    test_loader,
    loss_function: torch.nn.Module,
    device: torch.device,
    accuracy_tolerance: float,
    config: PipelineConfig,
    classification: bool,
    theory_state: _FGDTheoryState,
    initial_functional_gap: float,
    theory_loss_star: float,
    progress: ProgressFn | None,
    train_probe: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> _NonlinearPrimaryResult:
    """Try only the configured AdamW nonlinear ladder at ``theta_t``.

    Every functional rate ``eta_f`` is swept in the configured order and each
    is certified on its own -- an ``eta_f`` that fails says the clone could not
    realise THAT distance, not that the family is unavailable.

    The step that is COMMITTED is the step that was CERTIFIED. For each
    candidate the interpolations ``theta + alpha (theta' - theta)`` are
    measured again from scratch, and the accepted one carries its own cosine,
    its own relative error, and its own realized secant rate
    ``eta* = <Delta_alpha, r> / |r|^2``. Nothing derived from the full
    candidate is reused for a shorter step.
    """
    parametric = config.parametric_gd
    attempts = 0
    training_seconds = 0.0
    certification_seconds = 0.0
    last_stats: NonlinearCertificateStats | None = None
    last_candidate: NonlinearCandidate | None = None
    last_trial: _FGDTrial | None = None
    last_certificate = _invalid_nonlinear_certificate()
    last_update_norm: float | None = None
    best_cosine = -2.0

    certification_source = _nonlinear_certification_source(
        split=parametric.certificate_split,
        train_probe=train_probe,
        validation_loader=validation_loader,
    )
    # A train-probe certificate is one full-batch pass by construction; the
    # streaming cap only applies to a validation loader.
    certification_cap = (
        parametric.certification_batches
        if parametric.certificate_split == "validation"
        else None
    )

    with timed("nonlinear_total_seconds"):
        for functional_learning_rate in parametric.functional_learning_rates:
            for inner_steps in parametric.inner_steps:
                attempts += 1
                generated = train_nonlinear_candidate(
                    base_model=base_model,
                    train_loader=train_loader,
                    device=device,
                    functional_learning_rate=functional_learning_rate,
                    inner_steps=inner_steps,
                    config=parametric,
                    fgd_config=config.fgd_approx,
                    probe=train_probe,
                )
                last_candidate = generated
                training_seconds += generated.training_seconds
                if generated.model is None or not generated.sensor_valid:
                    last_certificate = _invalid_nonlinear_certificate()
                    if progress is not None:
                        progress(
                            f"[NONLINEAR] eta_f={functional_learning_rate:g}, "
                            f"inner_steps={inner_steps}: non-finite candidate; "
                            "rejected"
                        )
                    continue

                if progress is not None:
                    reduction = generated.objective_reduction
                    progress(
                        f"[NONLINEAR-CAND] eta_f={functional_learning_rate:g}, "
                        f"inner_steps={inner_steps} ({generated.step_unit}), "
                        f"optimizer_steps={generated.optimizer_steps}, "
                        f"epochs={generated.epochs:g}, "
                        f"batches={generated.batches_seen}, "
                        f"examples={generated.examples_seen}, "
                        f"objective {generated.initial_objective!r} -> "
                        f"{generated.final_objective!r} "
                        f"(reduction {reduction!r})"
                    )

                # The FULL candidate's certificate. Reported for diagnosis; it
                # never licenses a shorter step.
                candidate_stats = stream_nonlinear_certificate(
                    base_model=base_model,
                    candidate_model=generated.model,
                    certification_loader=certification_source,
                    device=device,
                    config=config.fgd_approx,
                    max_batches=certification_cap,
                )
                certification_seconds += candidate_stats.certification_seconds
                last_stats = candidate_stats
                if candidate_stats.cosine is not None:
                    best_cosine = max(best_cosine, candidate_stats.cosine)
                if progress is not None:
                    cosine = (
                        "n/a"
                        if candidate_stats.cosine is None
                        else f"{candidate_stats.cosine:.4f}"
                    )
                    epsilon = (
                        "n/a"
                        if candidate_stats.relative_error is None
                        else f"{candidate_stats.relative_error:.4f}"
                    )
                    secant = candidate_stats.effective_secant_rate
                    progress(
                        f"[NONLINEAR] eta_f={functional_learning_rate:g}, "
                        f"inner_steps={inner_steps}, cos={cosine}, "
                        f"eps={epsilon}, "
                        f"eta_star={'n/a' if secant is None else f'{secant:.4g}'}, "
                        f"param_disp="
                        f"{_parameter_displacement_norm(base_model, generated.model):.4g}, "
                        f"full_candidate_certified={candidate_stats.certified}"
                    )

                # Re-measure every interpolation. Each alpha is its own step
                # with its own certificate; the full candidate's is not reused.
                _, alpha_trials = search_interpolated_step(
                    base_model=base_model,
                    candidate_model=generated.model,
                    certification_loader=certification_source,
                    device=device,
                    config=config.fgd_approx,
                    alpha_grid=parametric.alpha_grid,
                    policy=parametric.alpha_policy,
                    max_batches=certification_cap,
                    progress=progress,
                )
                for trial_step in alpha_trials:
                    certification_seconds += trial_step.stats.certification_seconds

                admissible = [
                    step for step in alpha_trials if step.rejection_reason is None
                ]
                if parametric.alpha_policy == "max_descent":
                    admissible.sort(
                        key=lambda step: step.functional_descent,
                        reverse=True,
                    )
                else:
                    admissible.sort(key=lambda step: step.alpha, reverse=True)

                if not admissible:
                    last_certificate = _nonlinear_directional_certificate(
                        stats=candidate_stats,
                        learning_rate=candidate_stats.effective_secant_rate,
                        config=config.fgd_approx,
                    )
                    continue

                for step in admissible:
                    # The rate quoted with this certificate is the one this
                    # displacement REALIZED, so Lemma 3.5's interval is checked
                    # against the step actually being taken.
                    realized_rate = step.stats.effective_secant_rate
                    certificate = _nonlinear_directional_certificate(
                        stats=step.stats,
                        learning_rate=realized_rate,
                        config=config.fgd_approx,
                    )
                    last_certificate = certificate
                    if not certificate.learning_rate_interval_valid:
                        if progress is not None:
                            progress(
                                f"[NONLINEAR-ALPHA] alpha={step.alpha:g} rejected: "
                                f"realized eta*={realized_rate!r} outside the "
                                "Lemma 3.5 interval for its OWN eps "
                                f"{step.stats.relative_error!r}"
                            )
                        continue

                    last_update_norm = _parameter_displacement_norm(
                        base_model,
                        step.model,
                    )
                    train_metrics = evaluate_regression_metrics(
                        step.model,
                        train_loader,
                        loss_function,
                        device=device,
                        accuracy_tolerance=accuracy_tolerance,
                        classification=classification,
                    )
                    test_metrics = evaluate_regression_metrics(
                        step.model,
                        test_loader,
                        loss_function,
                        device=device,
                        accuracy_tolerance=accuracy_tolerance,
                        classification=classification,
                    )
                    epoch_result = FGDApproxEpochResult(
                        train_loss=train_metrics.loss,
                        train_accuracy=train_metrics.accuracy,
                        test_loss=test_metrics.loss,
                        test_accuracy=test_metrics.accuracy,
                        learning_rate=realized_rate,
                        next_learning_rate=realized_rate,
                        learning_rate_upper_bound=(
                            certificate.learning_rate_upper_bound
                        ),
                        learning_rate_interval_valid=(
                            certificate.learning_rate_interval_valid
                        ),
                        learning_rate_clipped_batches=0,
                        skipped_batches=certificate.skipped_batches,
                        relative_error_condition_valid=(
                            certificate.relative_error_condition_valid
                        ),
                        loss_descent_valid=None,
                        loss_non_descent_batches=0,
                        gradient_sq_norm=certificate.gradient_sq_norm,
                        theory_descent_coefficient=(
                            certificate.theory_descent_coefficient
                        ),
                        min_positive_learning_rate=realized_rate,
                        relative_error=step.stats.relative_error,
                        selected_layer_index=None,
                        layer_relative_errors=[],
                        output_relative_error=None,
                        sensor_valid=certificate.sensor_valid,
                        sensor_invalid_batches=certificate.sensor_invalid_batches,
                    )
                    trial = _certify_fgd_candidate(
                        candidate_model=step.model,
                        epoch_result=epoch_result,
                        certificate=certificate,
                        validation_loader=validation_loader,
                        device=device,
                        config=config,
                        theory_state=theory_state,
                        initial_functional_gap=initial_functional_gap,
                        theory_loss_star=theory_loss_star,
                    )
                    last_trial = trial
                    last_stats = step.stats
                    if not trial.all_conditions_valid:
                        if progress is not None:
                            progress(
                                f"[NONLINEAR-ALPHA] alpha={step.alpha:g} certified "
                                "but the transactional conditions rejected it"
                            )
                        continue

                    if progress is not None:
                        progress(
                            f"[NONLINEAR-ACCEPT] alpha={step.alpha:g}, "
                            f"eta_f={functional_learning_rate:g}, "
                            f"cos={step.stats.cosine:.4f}, "
                            f"eps={step.stats.relative_error:.4f}, "
                            f"eta_star={realized_rate:.4g}, "
                            f"descent={step.functional_descent:+.6e}, "
                            f"param_disp={last_update_norm:.4g}"
                        )
                    increment("nonlinear_accepted_steps")
                    return _NonlinearPrimaryResult(
                        accepted=trial,
                        last_trial=trial,
                        certificate=certificate,
                        stats=step.stats,
                        candidate=generated,
                        attempts=attempts,
                        candidate_training_seconds=training_seconds,
                        certification_seconds=certification_seconds,
                        update_norm=last_update_norm,
                        committed_alpha=step.alpha,
                        best_cosine=(best_cosine if best_cosine > -2.0 else None),
                    )

    if progress is not None:
        progress(
            "[NONLINEAR] no candidate certified the step it would apply "
            f"(best cos over {attempts} attempts: "
            f"{'n/a' if best_cosine <= -2.0 else f'{best_cosine:.4f}'})"
        )
    increment("nonlinear_failed_ladders")
    return _NonlinearPrimaryResult(
        accepted=None,
        last_trial=last_trial,
        certificate=last_certificate,
        stats=last_stats,
        candidate=last_candidate,
        attempts=attempts,
        candidate_training_seconds=training_seconds,
        certification_seconds=certification_seconds,
        update_norm=last_update_norm,
        committed_alpha=None,
        best_cosine=(best_cosine if best_cosine > -2.0 else None),
    )


def _architecture_widths(model: GrowingMLP) -> tuple[int, ...]:
    return tuple(
        int(layer.in_features) for layer in getattr(model, "_growable_layers", [])
    )


@torch.no_grad()
def _stream_max_function_drift(
    *,
    base_model: GrowingMLP,
    grown_model: GrowingMLP,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float | None:
    """Measure maximum output drift over a loader without retaining outputs."""
    base_was_training = base_model.training
    grown_was_training = grown_model.training
    base_model.eval()
    grown_model.eval()
    maximum = 0.0
    batches_seen = 0
    try:
        for x, _ in loader:
            batches_seen += 1
            x = x.to(device)
            before = base_model(x)
            after = grown_model(x)
            if not torch.isfinite(before).all() or not torch.isfinite(after).all():
                return None
            drift = float(torch.max(torch.abs(after - before)))
            if not math.isfinite(drift):
                return None
            maximum = max(maximum, drift)
    finally:
        base_model.train(base_was_training)
        grown_model.train(grown_was_training)
    return maximum if batches_seen else None


def _apply_nonlinear_primary_growth(
    *,
    model: GrowingMLP,
    train_loader,
    preservation_loader=None,
    device: torch.device,
    config: PipelineConfig,
    epoch: int,
    progress: ProgressFn | None,
) -> _NonlinearGrowthOutcome:
    """Select and transactionally apply one non-tangent structural growth."""
    if (
        config.fgd_approx.growth_where != "expressivity_bottleneck"
        or config.fgd_approx.growth_selection != "unified_expansion"
    ):
        raise ValueError(
            "Nonlinear primary growth requires growth_where="
            "'expressivity_bottleneck' and growth_selection='unified_expansion'."
        )
    if not getattr(model, "_growable_layers", None):
        return _NonlinearGrowthOutcome(None, None, None, 0.0, 0.0)

    started = time.perf_counter()
    with timed("nonlinear_growth_statistics_seconds"):
        bottlenecks = compute_expressivity_bottlenecks(
            model,
            train_loader,
            device,
            config.fgd_approx,
        )
        if progress is not None:
            progress(
                f"[NONLINEAR-BOTTLENECK] Epoch {epoch}, widths="
                f"{_architecture_widths(model)} "
                + " ".join(
                    f"L{index}={value:.4e}" for index, value in enumerate(bottlenecks)
                )
            )
        layer_index = (
            max(range(len(bottlenecks)), key=bottlenecks.__getitem__)
            if bottlenecks and max(bottlenecks) > 0.0
            else None
        )
    statistics_seconds = time.perf_counter() - started
    if layer_index is None:
        if progress is not None:
            progress(
                f"[NONLINEAR-GRO] Epoch {epoch}: no valid growth location; "
                "model unchanged"
            )
        return _NonlinearGrowthOutcome(
            None,
            None,
            None,
            statistics_seconds,
            0.0,
        )

    maximum_parameters = config.fgd_approx.max_total_parameters
    neuron_costs = growable_neuron_costs(model, config.data.in_features)
    projected_parameters = count_parameters(model) + neuron_costs[layer_index]
    if maximum_parameters is not None and projected_parameters > maximum_parameters:
        if progress is not None:
            progress(
                f"[NONLINEAR-GRO] Epoch {epoch}: layer {layer_index} growth "
                f"would exceed parameter budget ({projected_parameters} > "
                f"{maximum_parameters}); model unchanged"
            )
        return _NonlinearGrowthOutcome(
            None,
            None,
            layer_index,
            statistics_seconds,
            0.0,
        )

    before = _architecture_widths(model)
    grown_model = copy.deepcopy(model)
    started = time.perf_counter()
    try:
        with timed("nonlinear_growth_application_seconds"):
            result = grow_layer(
                model=grown_model,
                train_loader=train_loader,
                layer_index=layer_index,
                device=device,
                line_search_config=config.scaling_line_search,
                optimal_update_kwargs=tiny_optimal_update_kwargs(
                    config.fgd_approx,
                    compute_delta=False,
                ),
                progress=progress,
                function_preserving=True,
                preservation_tolerance=(
                    config.fgd_approx.growth_preservation_tolerance
                ),
            )
            maximum_drift = _stream_max_function_drift(
                base_model=model,
                grown_model=grown_model,
                loader=(
                    preservation_loader
                    if preservation_loader is not None
                    else train_loader
                ),
                device=device,
            )
            if (
                maximum_drift is None
                or maximum_drift > config.fgd_approx.growth_preservation_tolerance
            ):
                raise RuntimeError(
                    "Nonlinear growth failed its full-loader preservation "
                    f"check: drift={maximum_drift!r}, tolerance="
                    f"{config.fgd_approx.growth_preservation_tolerance:.3e}."
                )
    except (RuntimeError, ValueError) as error:
        application_seconds = time.perf_counter() - started
        if progress is not None:
            progress(
                f"[NONLINEAR-GRO] Epoch {epoch}: rejected transactional "
                f"growth at layer {layer_index}: {error}"
            )
        return _NonlinearGrowthOutcome(
            None,
            None,
            layer_index,
            statistics_seconds,
            application_seconds,
            False,
        )
    application_seconds = time.perf_counter() - started
    increment("nonlinear_growth_events")
    if progress is not None:
        progress(
            f"[NONLINEAR-GRO] Epoch {epoch}: layer={layer_index}, "
            f"widths {before} -> {_architecture_widths(grown_model)}, "
            f"max_drift={maximum_drift:.3e}"
        )
    return _NonlinearGrowthOutcome(
        grown_model,
        result,
        layer_index,
        statistics_seconds,
        application_seconds,
        True,
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


def _run_nonlinear_pipeline(
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
    """Run nonlinear training without entering the tangent outer-step loop."""
    model = build_model(config, device)
    loss_function = torch.nn.MSELoss()
    history: list[HistoryEntry] = []
    growth_events: list[GrowthResult] = []
    ladder_attempts = 0
    accepted_steps = 0
    failed_ladders = 0
    growth_count = 0

    def metrics(candidate: GrowingMLP, loader: torch.utils.data.DataLoader):
        return evaluate_regression_metrics(
            candidate,
            loader,
            loss_function,
            device=device,
            accuracy_tolerance=config.training.accuracy_tolerance,
            classification=classification,
        )

    if progress is not None:
        progress(f"Using device: {device}")
        progress(f"Training method: {config.training.method}")
        progress(
            "Primary approximation family: nonlinear "
            "(dedicated AdamW minibatch pipeline; tangent path unreachable)"
        )
        if wandb_logger.enabled:
            progress(
                f"W&B logging enabled: project={config.wandb.project}, "
                f"run={config.wandb.run_name or config.run.name}"
            )
        progress("Original model:")
        progress(str(model))

    train_metrics = metrics(model, train_loader)
    validation_metrics = metrics(model, validation_loader)
    test_metrics = metrics(model, test_loader)
    theory_loss_star = config.fgd_approx.theory_loss_star
    initial_functional_loss = evaluate_functional_loss(
        model,
        validation_loader,
        device,
        config.fgd_approx.functional_loss,
    )
    initial_functional_gap = max(initial_functional_loss - theory_loss_star, 0.0)
    theory_state = _FGDTheoryState(
        epoch_count=0,
        min_gradient_sq_norm=None,
        min_positive_learning_rate=None,
        min_descent_coefficient=None,
        global_contraction_product=1.0,
        previous_validation_functional_loss=initial_functional_loss,
    )

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
        num_params=count_parameters(model),
        fgd_approximation_kind="nonlinear",
        nonlinear_adamw_learning_rate=config.parametric_gd.inner_learning_rate,
        nonlinear_weight_decay=config.parametric_gd.weight_decay,
        architecture_widths=_architecture_widths(model),
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

    # The certification probe. build_projection_probe only CONCATENATES
    # minibatches -- no Jacobian, no tangent system, no projection solve -- so
    # the nonlinear-only guarantee is untouched. With the ladder's
    # probe_batches this is the entire training set, which is the empirical
    # objective the certified step is defined on.
    nonlinear_train_probe = build_nonlinear_probe(
        train_loader,
        config.fgd_approx.probe_batches,
        device,
    )
    if progress is not None:
        progress(
            f"[NONLINEAR] certificate split="
            f"{config.parametric_gd.certificate_split}, "
            f"train probe={nonlinear_train_probe[0].shape[0]} examples, "
            f"inner_step_unit={config.parametric_gd.inner_step_unit}, "
            f"eta_f sweep={list(config.parametric_gd.functional_learning_rates)}, "
            f"alpha grid={list(config.parametric_gd.alpha_grid) or [1.0]} "
            f"({config.parametric_gd.alpha_policy})"
        )

    last_test_loss = test_metrics.loss
    for epoch in range(1, config.training.epochs + 1):
        base_parameters = count_parameters(model)
        base_widths = _architecture_widths(model)
        nonlinear_result = _search_nonlinear_primary_candidate(
            base_model=model,
            train_loader=train_loader,
            train_probe=nonlinear_train_probe,
            validation_loader=validation_loader,
            test_loader=test_loader,
            loss_function=loss_function,
            device=device,
            accuracy_tolerance=config.training.accuracy_tolerance,
            config=config,
            classification=classification,
            theory_state=theory_state,
            initial_functional_gap=initial_functional_gap,
            theory_loss_star=theory_loss_star,
            progress=progress,
        )
        ladder_attempts += nonlinear_result.attempts
        accepted = nonlinear_result.accepted
        if accepted is not None:
            accepted_steps += 1
            model = accepted.model
            theory_state = accepted.theory_state
        else:
            failed_ladders += 1

        candidate = nonlinear_result.candidate
        stats = nonlinear_result.stats
        last_trial = nonlinear_result.last_trial
        committed_rate = (
            accepted.epoch_result.learning_rate if accepted is not None else None
        )
        growth_outcome: _NonlinearGrowthOutcome | None = None
        if accepted is None:
            if growth_count >= config.fgd_approx.certify_max_growths:
                if progress is not None:
                    progress(
                        f"[NONLINEAR-GRO] Epoch {epoch}: maximum growth events "
                        f"reached ({config.fgd_approx.certify_max_growths}); "
                        "model unchanged"
                    )
            else:
                growth_outcome = _apply_nonlinear_primary_growth(
                    model=model,
                    train_loader=train_loader,
                    preservation_loader=validation_loader,
                    device=device,
                    config=config,
                    epoch=epoch,
                    progress=progress,
                )

        step_metrics_model = accepted.model if accepted is not None else model
        train_metrics = metrics(step_metrics_model, train_loader)
        validation_metrics = metrics(step_metrics_model, validation_loader)
        test_metrics = metrics(step_metrics_model, test_loader)
        statistics_seconds = (
            growth_outcome.statistics_seconds if growth_outcome is not None else 0.0
        )
        application_seconds = (
            growth_outcome.application_seconds if growth_outcome is not None else 0.0
        )
        projected_growth_events = growth_count + int(
            growth_outcome is not None and growth_outcome.result is not None
        )
        epoch_entry = HistoryEntry(
            step=epoch,
            step_type="FGD",
            train_loss=train_metrics.loss,
            validation_loss=validation_metrics.loss,
            test_loss=test_metrics.loss,
            train_accuracy=train_metrics.accuracy,
            validation_accuracy=validation_metrics.accuracy,
            test_accuracy=test_metrics.accuracy,
            learning_rate=committed_rate or 0.0,
            num_params=(
                count_parameters(model) if accepted is not None else base_parameters
            ),
            fgd_learning_rate_upper_bound=(
                nonlinear_result.certificate.learning_rate_upper_bound
            ),
            fgd_learning_rate_interval_valid=(
                nonlinear_result.certificate.learning_rate_interval_valid
            ),
            fgd_relative_error_condition_valid=(
                nonlinear_result.certificate.relative_error_condition_valid
            ),
            fgd_loss_descent_valid=(
                last_trial.loss_descent_valid if last_trial is not None else None
            ),
            fgd_gradient_sq_norm=nonlinear_result.certificate.gradient_sq_norm,
            fgd_theory_descent_coefficient=(
                nonlinear_result.certificate.theory_descent_coefficient
            ),
            fgd_stationary_bound=(
                accepted.stationary_bound if accepted is not None else None
            ),
            fgd_stationary_bound_valid=(
                accepted.stationary_bound_valid if accepted is not None else None
            ),
            fgd_global_bound=(accepted.global_bound if accepted is not None else None),
            fgd_global_bound_valid=(
                accepted.global_bound_valid if accepted is not None else None
            ),
            fgd_global_contraction=(
                accepted.global_contraction if accepted is not None else None
            ),
            fgd_sensor_valid=nonlinear_result.certificate.sensor_valid,
            fgd_sensor_invalid_batches=(
                nonlinear_result.certificate.sensor_invalid_batches
            ),
            fgd_update_norm=nonlinear_result.update_norm,
            fgd_candidate_accepted=accepted is not None,
            fgd_lr_search_trials=nonlinear_result.attempts,
            fgd_approximation_kind="nonlinear",
            nonlinear_functional_learning_rate=(
                candidate.functional_learning_rate if candidate is not None else None
            ),
            nonlinear_inner_steps=(
                candidate.inner_steps if candidate is not None else None
            ),
            nonlinear_adamw_learning_rate=config.parametric_gd.inner_learning_rate,
            nonlinear_weight_decay=config.parametric_gd.weight_decay,
            nonlinear_cosine=(stats.cosine if stats is not None else None),
            nonlinear_relative_error=(
                stats.relative_error if stats is not None else None
            ),
            nonlinear_certificate_valid=(
                stats.certified if stats is not None else False
            ),
            nonlinear_validation_descent_valid=(
                last_trial.loss_descent_valid if last_trial is not None else None
            ),
            nonlinear_committed_rate=committed_rate,
            nonlinear_committed_alpha=nonlinear_result.committed_alpha,
            nonlinear_effective_secant_rate=(
                stats.effective_secant_rate if stats is not None else None
            ),
            nonlinear_best_cosine=nonlinear_result.best_cosine,
            nonlinear_candidate_optimizer_steps=(
                candidate.optimizer_steps if candidate is not None else None
            ),
            nonlinear_candidate_epochs=(
                candidate.epochs if candidate is not None else None
            ),
            nonlinear_candidate_batches_seen=(
                candidate.batches_seen if candidate is not None else None
            ),
            nonlinear_candidate_examples_seen=(
                candidate.examples_seen if candidate is not None else None
            ),
            nonlinear_candidate_initial_objective=(
                candidate.initial_objective if candidate is not None else None
            ),
            nonlinear_candidate_final_objective=(
                candidate.final_objective if candidate is not None else None
            ),
            nonlinear_candidate_objective_reduction=(
                candidate.objective_reduction if candidate is not None else None
            ),
            nonlinear_candidate_parameter_displacement_norm=(
                nonlinear_result.update_norm
            ),
            nonlinear_growth_requested=accepted is None,
            nonlinear_candidate_training_seconds=(
                nonlinear_result.candidate_training_seconds
            ),
            nonlinear_certification_seconds=nonlinear_result.certification_seconds,
            nonlinear_growth_statistics_seconds=statistics_seconds,
            nonlinear_growth_application_seconds=application_seconds,
            nonlinear_ladder_attempts=ladder_attempts,
            nonlinear_accepted_steps=accepted_steps,
            nonlinear_failed_ladders=failed_ladders,
            nonlinear_growth_events=projected_growth_events,
            nonlinear_full_jacobian_calls=0,
            nonlinear_tangent_system_calls=0,
            nonlinear_tangent_projection_solves=0,
            architecture_widths=base_widths,
        )
        history.append(epoch_entry)
        wandb_logger.log_history_entry(epoch_entry)

        if progress is not None and should_log_epoch(epoch, config):
            epsilon = (
                f"{stats.relative_error:.4f}"
                if stats is not None and stats.relative_error is not None
                else "n/a"
            )
            progress(
                f"[NONLINEAR] Epoch {epoch}, accepted={accepted is not None}, "
                f"eps={epsilon}, train_loss={train_metrics.loss:.4f}, "
                f"validation_loss={validation_metrics.loss:.4f}, "
                f"test_loss={test_metrics.loss:.4f} "
                f"({test_metrics.loss - last_test_loss:+.4f}), widths={base_widths}"
            )
        last_test_loss = test_metrics.loss

        if accepted is not None or growth_outcome is None:
            continue
        if growth_outcome.result is None or growth_outcome.model is None:
            continue

        model = growth_outcome.model
        growth_result = growth_outcome.result
        growth_events.append(growth_result)
        growth_count += 1
        # The clone passed the function-preservation check, so all metrics and
        # the functional loss are identical. Reusing them avoids four complete
        # loader passes after every structural event.
        post_growth_loss = theory_state.previous_validation_functional_loss
        initial_functional_gap = max(post_growth_loss - theory_loss_star, 0.0)
        theory_state = _FGDTheoryState(
            epoch_count=0,
            min_gradient_sq_norm=None,
            min_positive_learning_rate=None,
            min_descent_coefficient=None,
            global_contraction_product=1.0,
            previous_validation_functional_loss=post_growth_loss,
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
            learning_rate=0.0,
            num_params=count_parameters(model),
            layer_index=growth_outcome.layer_index,
            scaling_factor=growth_result.best_scaling_factor,
            selected_layer_index=growth_outcome.layer_index,
            fgd_candidate_accepted=False,
            fgd_approximation_kind="nonlinear",
            nonlinear_functional_learning_rate=(
                candidate.functional_learning_rate if candidate is not None else None
            ),
            nonlinear_inner_steps=(
                candidate.inner_steps if candidate is not None else None
            ),
            nonlinear_adamw_learning_rate=config.parametric_gd.inner_learning_rate,
            nonlinear_weight_decay=config.parametric_gd.weight_decay,
            nonlinear_cosine=(stats.cosine if stats is not None else None),
            nonlinear_relative_error=(
                stats.relative_error if stats is not None else None
            ),
            nonlinear_certificate_valid=(
                stats.certified if stats is not None else False
            ),
            nonlinear_validation_descent_valid=(
                last_trial.loss_descent_valid if last_trial is not None else None
            ),
            nonlinear_growth_requested=True,
            nonlinear_candidate_training_seconds=(
                nonlinear_result.candidate_training_seconds
            ),
            nonlinear_certification_seconds=nonlinear_result.certification_seconds,
            nonlinear_growth_statistics_seconds=growth_outcome.statistics_seconds,
            nonlinear_growth_application_seconds=growth_outcome.application_seconds,
            nonlinear_ladder_attempts=ladder_attempts,
            nonlinear_accepted_steps=accepted_steps,
            nonlinear_failed_ladders=failed_ladders,
            nonlinear_growth_events=growth_count,
            nonlinear_full_jacobian_calls=0,
            nonlinear_tangent_system_calls=0,
            nonlinear_tangent_projection_solves=0,
            architecture_widths=_architecture_widths(model),
        )
        history.append(growth_entry)
        wandb_logger.log_growth_event(
            event=growth_result,
            epoch=epoch,
            growth_count=growth_count,
            architecture_widths=_architecture_widths(model),
            statistics_seconds=growth_outcome.statistics_seconds,
            application_seconds=growth_outcome.application_seconds,
        )
        wandb_logger.log_history_entry(growth_entry)
        if progress is not None:
            progress(
                f"[GRO] Epoch {epoch}, layer={growth_outcome.layer_index}, "
                f"widths={_architecture_widths(model)}, "
                f"parameters={count_parameters(model)}"
            )
        last_test_loss = test_metrics.loss

    return PipelineResult(
        config=config,
        history=history,
        growth_events=growth_events,
        model=model,
        device=str(device),
    )


def run_pipeline(
    config: PipelineConfig,
    progress: ProgressFn | None = print,
) -> PipelineResult:
    """Run the train-grow loop from the GroMo tutorial."""
    nonlinear_selected = (
        config.training.method == "fgd_approx"
        and config.fgd_approx.family_order == ("nonlinear",)
    )
    if nonlinear_selected and config.parametric_gd.optimizer != "adamw":
        raise ValueError(
            "The nonlinear primary family requires parametric_gd.optimizer='adamw'."
        )
    if nonlinear_selected and (
        config.fgd_approx.growth_where != "expressivity_bottleneck"
        or config.fgd_approx.growth_selection != "unified_expansion"
    ):
        raise ValueError(
            "Nonlinear primary growth requires growth_where="
            "'expressivity_bottleneck' and growth_selection='unified_expansion'."
        )
    wandb_logger = build_wandb_logger(config.wandb)
    wandb_logger.start(
        run_name=config.run.name,
        config_payload=config_payload(config),
    )
    device = select_device(config.training.device)
    train_loader, validation_loader, test_loader = build_dataloaders(config, device)
    classification = is_classification_task(config)
    if nonlinear_selected:
        try:
            result = _run_nonlinear_pipeline(
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
    nonlinear_mode = (
        config.training.method == "fgd_approx"
        and config.fgd_approx.family_order == ("nonlinear",)
    )
    loss_function = torch.nn.MSELoss()
    optimizer = build_optimizer(model, config.optimizer)
    lr_cycle_start_epoch = 0
    current_fgd_learning_rate = config.fgd_approx.theory_lr_initial
    initial_learning_rate = (
        current_fgd_learning_rate
        if (
            config.training.method == "fgd_approx"
            and not nonlinear_mode
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
        if nonlinear_mode:
            progress(
                "Primary approximation family: nonlinear "
                "(AdamW minibatches; tangent operations disabled)"
            )
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
        # Grow-to-certify: remembers whether the previous epoch managed to
        # commit a step. A certified structure that still cannot step is the
        # deadlock this closes (see grow_until_certified's `force`).
        certify_previous_step_committed = True
        # Whether, when the previous epoch failed to commit, the failure's
        # own measurement was non-finite (see
        # certify_force_growth_on_finite_step_failure). Irrelevant while
        # certify_previous_step_committed is True, which is why the initial
        # value below never matters.
        certify_previous_failure_non_finite = False
        fgd_validation_probe: tuple[torch.Tensor, torch.Tensor] | None = None
        fgd_train_probe: tuple[torch.Tensor, torch.Tensor] | None = None
        if config.training.method == "fgd_approx" and not nonlinear_mode:
            # Bounded certification probe: sized to kappa*rank(J) when enabled,
            # else the fixed probe_batches. Re-evaluated each outer step (rank
            # grows). ``_probe_batches`` tracks the current train-probe size so
            # the rank estimate is measured on it and the probe grows reactively.
            _probe_batches = _bounded_probe_batches(
                config, model, train_loader, device
            )
            _validation_probe_batches = _bounded_probe_batches(
                config, model, validation_loader, device,
                current_batches=_probe_batches,
            )
            fgd_validation_probe = build_projection_probe(
                validation_loader,
                _validation_probe_batches,
                device,
            )
            fgd_train_probe = build_projection_probe(
                train_loader,
                _probe_batches,
                device,
            )
            # Functional Tikhonov: certify against L_data + gamma ||f||^2 by
            # shrinking the direction probe's targets. One point, and growth,
            # projection, realisation and GCV all inherit it.
            fgd_train_probe = _functional_tikhonov_probe(
                fgd_train_probe, config.fgd_approx
            )
            fgd_validation_probe = _functional_tikhonov_probe(
                fgd_validation_probe, config.fgd_approx
            )
        validation_certificate_for_next_epoch = None
        if (
            config.training.method == "fgd_approx"
            and not nonlinear_mode
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
                and not nonlinear_mode
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
            # Whether THIS epoch's (eventual) failure to commit is due to a
            # non-finite measurement rather than a finite one. Set inside
            # the outer-step loop below; defaults to False for training
            # methods that never reach it (normal SGD, fgd_approx without
            # theory_interval), where it stays unused.
            certify_step_failure_non_finite = False
            diagnostic_trial: _FGDTrial | None = None
            fgd_growth_requested = False
            fgd_candidate_accepted: bool | None = None
            fgd_lr_search_trials = 0
            fgd_approximation_kind: str | None = (
                (
                    "nonlinear"
                    if nonlinear_mode
                    else "tangent"
                )
                if config.training.method == "fgd_approx"
                else None
            )
            fgd_rkhs_phase_attempted = False
            fgd_rkhs_phase_accepted: bool | None = None
            fgd_rkhs_phase_steps = 0
            fgd_growth_probe_improved: bool | None = None
            nonlinear_functional_learning_rate: float | None = None
            nonlinear_inner_steps: int | None = None
            nonlinear_cosine: float | None = None
            nonlinear_certificate_valid: bool | None = None
            nonlinear_committed_rate: float | None = None
            nonlinear_growth_requested = False
            nonlinear_candidate_training_seconds = 0.0
            nonlinear_certification_seconds = 0.0
            nonlinear_growth_statistics_seconds = 0.0
            nonlinear_growth_application_seconds = 0.0
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
                if nonlinear_mode:
                    theory_state = _FGDTheoryState(
                        epoch_count=fgd_epoch_count,
                        min_gradient_sq_norm=fgd_min_gradient_sq_norm,
                        min_positive_learning_rate=(
                            fgd_min_positive_learning_rate
                        ),
                        min_descent_coefficient=fgd_min_descent_coefficient,
                        global_contraction_product=(
                            fgd_global_contraction_product
                        ),
                        previous_validation_functional_loss=(
                            previous_validation_functional_loss
                        ),
                    )
                    nonlinear_result = _search_nonlinear_primary_candidate(
                        base_model=model,
                        train_loader=train_loader,
                        validation_loader=validation_loader,
                        test_loader=test_loader,
                        loss_function=loss_function,
                        device=device,
                        accuracy_tolerance=(
                            config.training.accuracy_tolerance
                        ),
                        config=config,
                        classification=classification,
                        theory_state=theory_state,
                        initial_functional_gap=initial_functional_gap,
                        theory_loss_star=theory_loss_star,
                        progress=progress,
                    )
                    validation_certificate = nonlinear_result.certificate
                    validation_certificate_for_next_epoch = None
                    diagnostic_trial = nonlinear_result.last_trial
                    fgd_lr_search_trials = nonlinear_result.attempts
                    fgd_update_norm = nonlinear_result.update_norm
                    fgd_trial_sensor_failure = (
                        not nonlinear_result.certificate.sensor_valid
                    )
                    certify_step_failure_non_finite = fgd_trial_sensor_failure
                    nonlinear_candidate_training_seconds = (
                        nonlinear_result.candidate_training_seconds
                    )
                    nonlinear_certification_seconds = (
                        nonlinear_result.certification_seconds
                    )
                    if nonlinear_result.candidate is not None:
                        nonlinear_functional_learning_rate = (
                            nonlinear_result.candidate.functional_learning_rate
                        )
                        nonlinear_inner_steps = (
                            nonlinear_result.candidate.inner_steps
                        )
                    if nonlinear_result.stats is not None:
                        nonlinear_cosine = nonlinear_result.stats.cosine

                    accepted_trial = nonlinear_result.accepted
                    nonlinear_certificate_valid = accepted_trial is not None
                    if accepted_trial is not None:
                        model = accepted_trial.model
                        epoch_result = accepted_trial.epoch_result
                        optimizer = build_optimizer(model, config.optimizer)
                        committed_rate = epoch_result.learning_rate or 0.0
                        apply_learning_rate(optimizer, committed_rate)
                        current_fgd_learning_rate = committed_rate
                        entry_learning_rate = committed_rate
                        nonlinear_committed_rate = committed_rate
                        fgd_candidate_accepted = True
                        fgd_growth_requested = False
                        fgd_accepted_outer_steps += 1
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
                    else:
                        base_train_metrics = evaluate_regression_metrics(
                            model,
                            train_loader,
                            loss_function,
                            device=device,
                            accuracy_tolerance=(
                                config.training.accuracy_tolerance
                            ),
                            classification=classification,
                        )
                        base_test_metrics = evaluate_regression_metrics(
                            model,
                            test_loader,
                            loss_function,
                            device=device,
                            accuracy_tolerance=(
                                config.training.accuracy_tolerance
                            ),
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
                            relative_error=(
                                nonlinear_result.stats.relative_error
                                if nonlinear_result.stats is not None
                                else None
                            ),
                            selected_layer_index=None,
                            layer_relative_errors=[],
                            output_relative_error=(
                                validation_certificate.output_relative_error
                            ),
                            sensor_valid=validation_certificate.sensor_valid,
                            sensor_invalid_batches=(
                                validation_certificate.sensor_invalid_batches
                            ),
                        )
                        entry_learning_rate = 0.0
                        fgd_candidate_accepted = False
                        fgd_growth_requested = True
                        nonlinear_growth_requested = True
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
                        if progress is not None:
                            progress(
                                f"[NONLINEAR] Epoch {epoch}: all "
                                f"{nonlinear_result.attempts} nonlinear "
                                "candidates failed mandatory certificates; "
                                "requesting growth"
                            )
                elif use_fgd_theory_learning_rate:
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
                        if config.fgd_approx.probe_resample:
                            # A FIXED probe turns the method into Newton's
                            # method on one subsample: the certified step is
                            # now delivered in full, so the residual on those
                            # samples is driven to zero and the network
                            # interpolates them. MEASURED: train accuracy
                            # 0.320 against validation 0.150, with the logged
                            # tangent relative error exploding to 1077 --
                            # RelErr is normalised by ||g||, so it diverges
                            # precisely when there is no residual left on the
                            # probe to approximate.
                            #
                            # Drawing a fresh probe each outer step makes the
                            # functional gradient an unbiased estimate of the
                            # one over the dataset, which is the object the
                            # theory is about; the flow becomes stochastic FGD
                            # instead of exact descent on 256 fixed points.
                            _probe_batches = _bounded_probe_batches(
                                config, model, train_loader, device,
                                current_batches=_probe_batches,
                            )
                            fgd_train_probe = _functional_tikhonov_probe(
                                build_projection_probe(
                                    train_loader,
                                    _probe_batches,
                                    device,
                                ),
                                config.fgd_approx,
                            )
                            fgd_validation_probe = _functional_tikhonov_probe(
                                build_projection_probe(
                                    validation_loader,
                                    _bounded_probe_batches(
                                        config, model, validation_loader, device,
                                        current_batches=_probe_batches,
                                    ),
                                    device,
                                ),
                                config.fgd_approx,
                            )
                        elif config.fgd_approx.certify_probe_kappa > 0.0:
                            # Fixed bounded probe that GROWS with P: re-materialise
                            # only when kappa*P now needs more points than the probe
                            # holds, so the floor s > P holds for the current
                            # structure while candidate comparisons within a growth
                            # round still share one probe.
                            _needed = _bounded_probe_batches(
                                config, model, train_loader, device,
                                current_batches=_probe_batches,
                            )
                            if _needed > _probe_batches:
                                _probe_batches = _needed
                                fgd_train_probe = _functional_tikhonov_probe(
                                    build_projection_probe(
                                        train_loader, _probe_batches, device
                                    ),
                                    config.fgd_approx,
                                )
                                fgd_validation_probe = _functional_tikhonov_probe(
                                    build_projection_probe(
                                        validation_loader,
                                        _bounded_probe_batches(
                                            config, model, validation_loader, device,
                                            current_batches=_probe_batches,
                                        ),
                                        device,
                                    ),
                                    config.fgd_approx,
                                )
                        tangent_direction: tuple[torch.Tensor, ...] | None = None
                        direction_stats: _FunctionalStepStats | None = None
                        maximum_learning_rate: float | None = None
                        # The eps that is CERTIFIED: measured on the probe
                        # the direction is solved on, and the quantity the
                        # grow loop drives below the threshold. Lemma 3.5's
                        # interval is defined from this one.
                        certified_relative_error: float | None = None
                        # Set when the damping is chosen by measurement: the
                        # rate that selection already scored, so it is not
                        # recomputed from a damping that no longer applies.
                        selected_learning_rate: float | None = None
                        direction_sensor_failure = False
                        # Whether direction_sensor_failure (when set below)
                        # came from a non-finite measurement rather than a
                        # finite geometric/structural one -- the discriminator
                        # certify_force_growth_on_finite_step_failure needs.
                        direction_sensor_non_finite = False
                        tangent_system = None
                        if config.fgd_approx.grow_to_certify:
                            # GROW-TO-CERTIFY. Make the structure satisfy
                            # Lemma 3.5 BEFORE stepping, instead of stepping
                            # and growing when it fails. Every growth is
                            # function-preserving, so f does not move here --
                            # only the certified step below moves it -- and
                            # eps decreases monotonically, so this terminates.
                            # Certified family ladder: try a nonlinear
                            # within-MLP family before each growth, accepting
                            # its step only when its OWN relative error
                            # certifies. It certifies at a far smaller
                            # structure than the linear tangent, so it defers
                            # growth (MEASURED 57 -> 6 FP growths).
                            # "Grow now, or train now?" -- the organic turn
                            # question. Generalised R1 already implements it
                            # (train a grown clone and a stay clone the same
                            # number of steps, compare the eps they reach); it
                            # was only ever reachable from the fallback-family
                            # loop, which is dead under family_order:
                            # [tangent]. Bound here so the grow-to-certify
                            # loop can use it, which is where the front-loading
                            # actually happens.
                            def _growth_warranted_now(candidate_model):
                                return _growth_reduces_lookahead_epsilon(
                                    model=candidate_model,
                                    train_batches=frozen_train_batches,
                                    train_loader=train_loader,
                                    validation_loader=validation_loader,
                                    probe=fgd_validation_probe,
                                    device=device,
                                    config=config,
                                )

                            _family_step = None
                            if config.fgd_approx.certify_family_ladder:
                                _fp = fgd_train_probe

                                # One eta_f, or a swept list of them. The sweep
                                # certifies each candidate independently on
                                # this same probe against the same
                                # min(rel_error_threshold, 1/2), and stops at
                                # the first certificate -- more search at an
                                # unchanged bar, which is the shape of the
                                # measured gap: on N=1024 the four seeds
                                # certify the family 12/2/5/8 times and their
                                # capped accuracies rank 0.941/0.830/0.904/
                                # 0.925 in exactly that order. An eta_f that
                                # fails says the clone could not realise THAT
                                # distance, not that the family is unavailable.
                                def _family_step(candidate_model):
                                    return certify_parametric_step_swept(
                                        candidate_model,
                                        _fp[0],
                                        _fp[1],
                                        config.fgd_approx,
                                        config.fgd_approx.certify_family_functional_lrs,
                                        inner_steps=(
                                            config.fgd_approx.certify_family_inner_steps
                                        ),
                                        inner_learning_rate=(
                                            config.fgd_approx.certify_family_inner_learning_rate
                                        ),
                                        progress=progress,
                                    ).model
                            model, certify_result = grow_until_certified(
                                model=model,
                                x=fgd_train_probe[0],
                                y=fgd_train_probe[1],
                                train_loader=train_loader,
                                device=device,
                                config=config,
                                max_growths=(
                                    config.fgd_approx.certify_max_growths
                                ),
                                function_preserving=(
                                    config.fgd_approx.certify_function_preserving
                                ),
                                family_step=_family_step,
                                # "Is it growth's turn?", asked once per outer
                                # step. Passed as a callable for the same
                                # reason family_step is: this module imports
                                # certify, so the dependency cannot run back.
                                growth_warranted=(
                                    _growth_warranted_now
                                    if config.fgd_approx.certify_growth_lookahead
                                    else None
                                ),
                                # ONE criterion for the whole of growth. Passed
                                # as a callable for the same reason family_step
                                # is: this module imports certify, so the
                                # dependency cannot run back. Supplied only
                                # under expressivity_bottleneck -- under the
                                # other rules the adaptive count keeps its own
                                # gap test, so their results are untouched.
                                layer_bottlenecks=(
                                    (
                                        lambda m: compute_expressivity_bottlenecks(
                                            m, train_loader, device,
                                            config.fgd_approx,
                                        )
                                    )
                                    if (
                                        config.fgd_approx.growth_where
                                        == "expressivity_bottleneck"
                                        and config.fgd_approx
                                        .certify_adaptive_growth_by_bottleneck
                                    )
                                    else None
                                ),
                                # A step that did not commit while eps was
                                # already certified means the structure, not
                                # the step size, is what has to change --
                                # eps < 1/2 said the tangent space was
                                # adequate and the step still could not be
                                # taken, so grow anyway. MEASURED: without
                                # this the linearisation control froze the
                                # synthetic run, epochs 12-16 bit-identical
                                # at loss 0.1072 with eps = 0.388 (certified,
                                # so no growth) and no rate inside the
                                # regime the lemma describes (so no step).
                                # Growth is the right remedy here in a way it
                                # was not for the descent gate: the defect is
                                # measured on the SAME probe as eps, so
                                # failing it is a statement about this
                                # direction, and adding directions changes
                                # the tangent space and hence the direction.
                                # Growth fires on the CERTIFICATE and nothing
                                # else: eps >= 1/2 and only that. Forcing it
                                # whenever a step failed to commit was mine,
                                # added to break a deadlock whose cause -- the
                                # held-out descent gate -- has since been
                                # removed, so it now solves a problem that no
                                # longer exists while firing on problems it
                                # cannot fix. MEASURED on the small synthetic
                                # run: 262 growths against 7 committed steps,
                                # and 242 of those 262 grew with eps ALREADY
                                # certified, many at eps = 0.0000. The trigger
                                # was a non-finite validation measurement --
                                # the model overflowing on unseen data, which
                                # is a symptom of overfitting and which more
                                # capacity makes worse, not better.
                                #
                                # Both incidents are real, and what tells
                                # them apart is exactly that finiteness:
                                # sane numbers violating an invariant, or
                                # simply no admissible rate (MNIST), mean
                                # the STRUCTURE has to change; NaN/Inf (the
                                # over-firing run) means the MODEL failed,
                                # which growth cannot fix. See
                                # FGDApproxConfig.certify_force_growth_on_
                                # finite_step_failure for the full account;
                                # default False keeps force=False always,
                                # bit-identical to today.
                                force=_certify_force_growth(
                                    config.fgd_approx,
                                    previous_step_committed=(
                                        certify_previous_step_committed
                                    ),
                                    previous_failure_non_finite=(
                                        certify_previous_failure_non_finite
                                    ),
                                ),
                                progress=progress,
                            )
                            tangent_system = certify_result.tangent_system
                            if certify_result.growths or certify_result.family_steps:
                                growth_count += certify_result.growths
                                optimizer = build_optimizer(
                                    model, config.optimizer
                                )
                                _clear_inaccessible_tensor_caches(model)
                                if (
                                    certify_result.family_steps
                                    and progress is not None
                                ):
                                    progress(
                                        "[CERTIFY] family ladder took "
                                        f"{certify_result.family_steps} step(s) "
                                        f"in {certify_result.growths} growths"
                                    )
                            if progress is not None and not certify_result.certified:
                                # TWO lines are in play, and only one of them
                                # can kill the step: the growth TARGET the loop
                                # chases, and the step CERTIFICATE Lemma 3.5
                                # needs (eps < min(rel_error_threshold, 1/2),
                                # the same expression damping.py and
                                # lemma35_learning_rate use). Missing the
                                # target while clearing the certificate is a
                                # normal stop -- the step still commits, which
                                # is the whole point of separating them.
                                # Missing the certificate means no damping and
                                # no rate exist at all. The old single line read
                                # "could NOT reach eps < 0.3" for both, which is
                                # how the MNIST run with a dead tangent step
                                # looked exactly like a run doing fine.
                                step_certificate = min(
                                    config.fgd_approx.rel_error_threshold, 0.5
                                )
                                progress(
                                    f"[CERTIFY] Epoch {epoch}: growth target "
                                    f"eps < {certify_result.growth_target} NOT "
                                    f"reached (stopped at "
                                    f"{certify_result.relative_error:.4f} after "
                                    f"{certify_result.growths} growths, reason: "
                                    f"{certify_result.stop_reason}); step "
                                    f"certificate eps < {step_certificate} "
                                    + (
                                        "HOLDS, so a step can still commit"
                                        if certify_result.relative_error
                                        < step_certificate
                                        else "FAILS, so no rate is admissible"
                                    )
                                )
                        damping_choice = (
                            select_projection_damping(
                                model,
                                fgd_train_probe[0],
                                fgd_train_probe[1],
                                config.fgd_approx,
                                system=tangent_system,
                            )
                            if config.fgd_approx.projection_damping_auto
                            else None
                        )
                        if damping_choice is not None:
                            tangent_system = damping_choice.tangent_system
                            # The damping is the knob that arbitrates between
                            # the certificate (eps) and the realisability of
                            # the step (|u|); a fixed constant lands in the
                            # window satisfying both only by luck. Selection
                            # re-derives it from measurement, scored by the
                            # decrease Lemma 3.5 itself guarantees.
                            tangent_direction = damping_choice.parameter_updates
                            certified_relative_error = (
                                damping_choice.candidate.relative_error
                            )
                            selected_learning_rate = (
                                damping_choice.candidate.learning_rate
                            )
                            if (
                                config.fgd_approx.certify_realize_path
                                and damping_choice.candidate.certified_learning_rate
                            ):
                                # Realise the FULL certified functional step
                                # by integrating toward it, then express the
                                # path travelled as the equivalent single
                                # update so everything downstream -- the
                                # validation certificate, the trial, the
                                # accounting -- sees an ordinary outer step
                                # and reproduces this exact point.
                                nominal = (
                                    damping_choice.candidate
                                    .certified_learning_rate
                                )
                                base_model = copy.deepcopy(model)
                                walker = model
                                realization = realize_functional_step(
                                    walker,
                                    fgd_train_probe[0],
                                    fgd_train_probe[1],
                                    tangent_direction,
                                    nominal,
                                    config.fgd_approx,
                                    max_iterations=(
                                        config.fgd_approx
                                        .certify_realize_max_iterations
                                    ),
                                    tolerance=(
                                        config.fgd_approx
                                        .certify_realize_tolerance
                                    ),
                                    system=tangent_system,
                                    damping=realization_damping(
                                        config.fgd_approx,
                                        damping_choice.candidate.absolute_damping,
                                    ),
                                )
                                if realization.iterations > 0:
                                    with torch.no_grad():
                                        tangent_direction = tuple(
                                            (before.detach() - after.detach())
                                            / nominal
                                            for before, after in zip(
                                                base_model.parameters(),
                                                walker.parameters(),
                                            )
                                        )
                                    selected_learning_rate = nominal
                                    if progress is not None:
                                        progress(
                                            "[REALIZE] "
                                            f"eta={nominal:.4e} "
                                            "realised="
                                            f"{realization.realised_fraction:.1%} "
                                            "residual="
                                            f"{realization.residual_fraction:.1%} "
                                            f"iters={realization.iterations}"
                                        )
                                model = base_model
                                tangent_system = None
                                del walker
                            if progress is not None:
                                chosen = damping_choice.candidate
                                progress(
                                    f"[DAMPING] rho={chosen.relative_damping:.2e} "
                                    f"lambda={chosen.absolute_damping:.3e} "
                                    f"eps={chosen.relative_error:.4f} "
                                    f"|u|={chosen.update_norm:.3e} "
                                    f"eta={chosen.learning_rate:.4e} "
                                    f"decrease={chosen.guaranteed_decrease:.4e}"
                                )
                        elif config.fgd_approx.projection_damping_auto:
                            # No damping both certifies and realises a step:
                            # the structure has to change, so leave the
                            # direction unset and let growth act.
                            direction_sensor_failure = True
                        if (
                            tangent_direction is None
                            and not direction_sensor_failure
                        ):
                            direction_step = _compute_tangent_projection_step(
                                model=model,
                                x=fgd_train_probe[0],
                                y=fgd_train_probe[1],
                                config=config.fgd_approx,
                            )
                            direction_sensor_reason = _projection_step_sensor_reason(
                                direction_step,
                                config.fgd_approx,
                            )
                            if direction_sensor_reason is None:
                                tangent_direction = (
                                    direction_step.parameter_updates
                                )
                                certified_relative_error = (
                                    direction_step.output_error.relative_error
                                )
                            else:
                                direction_sensor_failure = True
                                # Only "non_finite" is the overfitting
                                # signal a forced growth must not act on;
                                # a geometric-invariant violation here is
                                # sane, finite numbers the structure failed
                                # to satisfy, which growth can still fix.
                                direction_sensor_non_finite = (
                                    direction_sensor_reason == "non_finite"
                                )
                        if tangent_direction is not None:
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
                            elif config.fgd_approx.certify_apply_in_interval:
                                # DIAGNOSTIC, not a gate. Admissibility is
                                # decided from eps on the TRAIN probe, the
                                # sample Lemma 3.5 speaks about, and the step
                                # is carried by selected_learning_rate under
                                # damping-auto. With the projector invariants
                                # off, this held-out sensor tests only
                                # finiteness, so failing it says the MODEL
                                # overflowed on unseen data -- a symptom of
                                # the fit, not a reason to withhold the step.
                                #
                                # So the flag is deliberately NOT set here.
                                # Setting it was the deadlock's final form:
                                # MEASURED, after two committed steps this
                                # sensor rejected every subsequent one and the
                                # run froze from epoch 2 to 400 (10 steps in
                                # all), while eps < 1/2 kept growth from
                                # firing. maximum_learning_rate simply stays
                                # None; the certified rate comes from the
                                # train-probe eps instead.
                                pass
                            else:
                                direction_sensor_failure = True
                                direction_stats = None
                                # This sensor runs with projection_sensor=
                                # False (only finiteness is checked, per
                                # the comment above), so reaching here
                                # always means a non-finite measurement.
                                direction_sensor_non_finite = True
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
                            config.fgd_approx.certify_apply_in_interval
                            and tangent_direction is not None
                            and direction_stats is not None
                        ):
                            # Lemma 3.5 as the paper applies it: take a rate
                            # inside the certified interval and step. Same
                            # rate the sweep would have started from -- only
                            # the verification is dropped, because descent is
                            # a CONCLUSION of the lemma, not a condition it
                            # asks to be checked, and demanding it on
                            # held-out data asks the lemma for a guarantee
                            # outside its domain.
                            search_result = _apply_lemma35_step(
                                relative_error=certified_relative_error,
                                learning_rate=selected_learning_rate,
                                evaluate_trial=evaluate_trial,
                                config=config.fgd_approx,
                                model=model,
                                probe_inputs=fgd_train_probe[0],
                                updates=tangent_direction,
                                progress=progress,
                            )
                        elif (
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
                        # certify_force_growth_on_finite_step_failure's
                        # discriminator. Every trial-level certificate this
                        # search touches runs with projection_sensor=False
                        # (finiteness only, see certificate_from_projection_
                        # stats above), so a sensor failure with a direction
                        # already in hand is unconditionally non-finite;
                        # with no direction at all (direction_stats is None)
                        # it carries direction_sensor_non_finite's own
                        # reason instead. Overwritten every attempt, so only
                        # the LAST (the one that may break the loop) is what
                        # reaches the epoch-level bookkeeping below.
                        certify_step_failure_non_finite = (
                            search_result.sensor_failure
                            and (
                                direction_stats is not None
                                or direction_sensor_non_finite
                            )
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
            # Drives the grow-to-certify `force`: a certified structure that
            # still could not commit a step is the deadlock to break.
            certify_previous_step_committed = bool(fgd_candidate_accepted)
            # Companion signal: whether THAT failure's own measurement was
            # non-finite. _certify_force_growth ignores this whenever
            # certify_previous_step_committed is True, so it is safe to
            # carry forward unconditionally here.
            certify_previous_failure_non_finite = certify_step_failure_non_finite
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
                nonlinear_functional_learning_rate=(
                    nonlinear_functional_learning_rate
                ),
                nonlinear_inner_steps=nonlinear_inner_steps,
                nonlinear_adamw_learning_rate=(
                    config.parametric_gd.inner_learning_rate
                    if nonlinear_mode
                    else None
                ),
                nonlinear_weight_decay=(
                    config.parametric_gd.weight_decay
                    if nonlinear_mode
                    else None
                ),
                nonlinear_cosine=nonlinear_cosine,
                nonlinear_certificate_valid=nonlinear_certificate_valid,
                nonlinear_committed_rate=nonlinear_committed_rate,
                nonlinear_growth_requested=nonlinear_growth_requested,
                nonlinear_candidate_training_seconds=(
                    nonlinear_candidate_training_seconds
                ),
                nonlinear_certification_seconds=(
                    nonlinear_certification_seconds
                ),
                nonlinear_growth_statistics_seconds=(
                    nonlinear_growth_statistics_seconds
                ),
                nonlinear_growth_application_seconds=(
                    nonlinear_growth_application_seconds
                ),
                architecture_widths=_architecture_widths(model),
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
                    reasons = getattr(
                        validation_certificate_for_next_epoch,
                        "non_finite_quantities",
                        (),
                    ) if validation_certificate_for_next_epoch else ()
                    warnings.append(
                        f"sensor invalid on "
                        f"{fgd_sensor_invalid_batches} validation batch(es)"
                        # Name what overflowed. Without it the message reads as
                        # a defect in the certificate machinery; with
                        # projection_sensor off the only test is finiteness, so
                        # what failed is the MODEL producing non-finite values
                        # on unseen data, not our arithmetic.
                        + (f" (non-finite: {', '.join(reasons)})" if reasons else "")
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
                    growth_triggered
                    and nonlinear_mode
                    and growth_count >= config.fgd_approx.certify_max_growths
                ):
                    growth_triggered = False
                    if progress is not None:
                        progress(
                            f"[NONLINEAR-GRO] Epoch {epoch}: growth suppressed "
                            "because the configured maximum of "
                            f"{config.fgd_approx.certify_max_growths} events "
                            "was reached"
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
                not nonlinear_mode
                and
                config.fgd_approx.families_available_without_growth
                and fgd_candidate_accepted is False
            )
            if (
                not nonlinear_mode
                and
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
                        # Only pay the look-ahead when it could actually change
                        # the decision: when eps is BELOW the threshold, so the
                        # normal criterion would not already trigger growth.
                        # Above it, growth is mandated anyway and the two
                        # trained clones would be pure waste -- this gate is
                        # what keeps the cost in the tail, not every epoch.
                        below_threshold = (
                            epsilon_before_family is not None
                            and epsilon_before_family
                            < config.fgd_approx.rel_error_threshold
                        )
                        if (
                            not epsilon_stationary
                            and below_threshold
                            and config.fgd_approx.growth_lookahead_adequacy
                            and epsilon_after_family is not None
                            and _growth_reduces_lookahead_epsilon(
                                model=model,
                                train_batches=growth_train_batches,
                                train_loader=train_loader,
                                validation_loader=validation_loader,
                                probe=fgd_validation_probe,
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
                if nonlinear_mode:
                    nonlinear_growth = _apply_nonlinear_primary_growth(
                        model=model,
                        train_loader=train_loader,
                        device=device,
                        config=config,
                        epoch=epoch,
                        progress=progress,
                    )
                    nonlinear_growth_statistics_seconds = (
                        nonlinear_growth.statistics_seconds
                    )
                    nonlinear_growth_application_seconds = (
                        nonlinear_growth.application_seconds
                    )
                    if nonlinear_growth.result is None:
                        # The failed nonlinear ladder requested growth, but a
                        # configured safety guard or the structural criterion
                        # declined it. The model remains transactional.
                        continue
                    if nonlinear_growth.model is None:
                        raise RuntimeError(
                            "Successful nonlinear growth did not return a model."
                        )
                    model = nonlinear_growth.model
                    growth_result = nonlinear_growth.result
                    layer_index = nonlinear_growth.layer_index
                    selected_layer_index = nonlinear_growth.layer_index
                elif config.training.method == "fgd_approx":
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
                        # "rank_ceiling" is the shipped path, byte-identical.
                        # "certified_gain" replaces the levelling, the
                        # bottleneck filter AND the pace in one go -- see the
                        # field's comment in tangent.py for why replacing only
                        # one of the three stalled growth at 53 parameters.
                        where_mode = getattr(
                            config.fgd_approx, "growth_where", "rank_ceiling"
                        )
                        # rank J <= min_l w_l: while a location sits at the
                        # minimum, no purchase elsewhere can raise what the
                        # structure is able to express.
                        widths = [
                            int(layer.in_features)
                            for layer in model._growable_layers
                        ]
                        # The rank cap both FILTERS candidates to the width
                        # minimum and MANDATES levelling it. Both rest on
                        # "rank J <= min_l w_l" (unified.py:53), which is
                        # FALSE as measured on the exact Jacobian: at NK=200,
                        # widths (20,3,20) reach rank 200, (30,4,30) reach
                        # 200, and even (2,2,2) reaches 25 = P, not 2. What
                        # bounds the rank is min(NK, P), not the narrowest
                        # layer. So the mandate levels the widths -- every
                        # seed lands on h,h,h+1 -- and bars the search from
                        # non-uniform shapes, for a reason that does not
                        # hold. Disabling it leaves the exact per-candidate
                        # eps ranking (unified.expansion_value) as the sole
                        # criterion, which measures what each location
                        # actually buys instead of assuming the minimum caps
                        # it. No certificate is touched.
                        free_shape = getattr(
                            config.fgd_approx, "growth_free_shape", False
                        )
                        # The filter and the levelling are separable, and
                        # they do different jobs. The filter protects the
                        # location a pure value ranking STARVES -- MEASURED,
                        # free shape drives the last hidden layer to width 2-4
                        # because a neuron there costs h2+2 and adds a single
                        # tangent direction, so it never wins on price even
                        # though the structure needs it. Its benefit is
                        # indirect: it raises the ceiling for later purchases
                        # elsewhere, which immediate eps cannot see. The
                        # levelling is what forces h,h,h+1. Keep the first,
                        # drop the second.
                        bottlenecks = (
                            set()
                            if free_shape
                            else set(rank_limiting_locations(widths))
                        )
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
                        # Also off under expressivity_bottleneck, and for the
                        # same reason it is off under certified_gain: this
                        # loop LEVELS min_l w_l on the "rank J <= min_l w_l"
                        # argument, so leaving it on hands the shape to a
                        # mandate that never consults the criterion. MEASURED
                        # with it left on by mistake: on the easy synthetic
                        # function the bottleneck had collapsed to 1e-10..1e-12
                        # by epoch 10 -- ten orders below its epoch-1 value of
                        # 4.9e-02, i.e. the criterion was saying "no width is
                        # missing" -- and the net still grew from 77 to 460
                        # parameters, every event a "rank cap relieved"
                        # levelling that widened whichever layer was narrowest.
                        relief = (
                            None
                            if free_shape
                            or where_mode in ("certified_gain", "expressivity_bottleneck")
                            else bottleneck_relief_target(widths)
                        )
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

                        # The TRAIN-probe eps, i.e. the same quantity the grow
                        # loop chases and Lemma 3.5's interval is built from.
                        # Scoring candidates on validation while the flow
                        # certifies on train is two certificates driving one
                        # decision; on the train probe function-preserving
                        # growth strictly enlarges range(J), so the gain is
                        # non-negative by certify.py's theorem, whereas on
                        # validation it is not -- which is exactly what
                        # expansion_value's max(..., 0) was clamping away.
                        # Kept as a single seam so the exact block-Schur fast
                        # path can be adopted here later behind its own flag.
                        def _train_epsilon(trial: GrowingMLP) -> float | None:
                            if fgd_train_probe is None:
                                return None
                            try:
                                return exact_relative_error(
                                    trial,
                                    fgd_train_probe[0],
                                    fgd_train_probe[1],
                                    config.fgd_approx,
                                )
                            except RuntimeError:
                                return None

                        def _ladder_residual(
                            trial: GrowingMLP,
                        ) -> float | None:
                            """How close the LADDER can bring this candidate
                            to the functional target it is already chasing.

                            Growth is function-preserving, so f -- and with it
                            the target f - eta * r -- is the same for the base
                            and for every candidate. All candidates therefore
                            aim at one fixed blank, which is what makes the
                            scores comparable without training a separate
                            control.
                            """
                            if fgd_train_probe is None:
                                return None
                            _pd = config.parametric_descent
                            try:
                                _rate = float(
                                    _pd.functional_learning_rates[0]
                                )
                            except (AttributeError, IndexError, TypeError):
                                return None
                            _batches = [
                                (
                                    fgd_train_probe[0].to(device),
                                    fgd_train_probe[1].to(device),
                                )
                            ]
                            _trained = _train_parametric_gd_candidate(
                                base_model=trial,
                                train_batches=_batches,
                                device=device,
                                functional_learning_rate=_rate,
                                steps=max(
                                    1,
                                    int(
                                        getattr(
                                            config.fgd_approx,
                                            "growth_where_ladder_steps",
                                            1,
                                        )
                                    ),
                                ),
                                config=_pd,
                                functional_loss=(
                                    config.fgd_approx.functional_loss
                                ),
                            )
                            if _trained is None:
                                return None
                            _x, _y = _batches[0]
                            with torch.no_grad():
                                _f = trial(_x)
                                _target = _f - _rate * functional_gradient(
                                    _f, _y, config.fgd_approx.functional_loss
                                )
                                # Report the FRACTION of the wanted functional
                                # move the ladder could not achieve. Since
                                # f - target = eta * g, this is
                                # ||trained - target|| / ||f - target||, which
                                # is eps's own meaning and eps's own scale --
                                # a raw MSE is not, and feeding one to a ranker
                                # and a threshold that both speak eps is what
                                # made this score produce 0-2 growth events.
                                _num = float(
                                    torch.linalg.vector_norm(
                                        _trained(_x) - _target
                                    )
                                )
                                _den = float(
                                    torch.linalg.vector_norm(_f - _target)
                                )
                            if _den <= 0.0:
                                return None
                            _res = _num / _den
                            return _res if math.isfinite(_res) else None

                        def _score_for(trial: GrowingMLP) -> float | None:
                            if where_mode != "certified_gain":
                                return _certificate_for(trial)
                            # Score the candidate AWAKE, not dormant, WITHOUT
                            # moving the target. Function-preserving growth
                            # enters the neuron with outgoing weight w = 0,
                            # which zeroes the derivative of ALL its incoming
                            # weights: the columns exist but are identically
                            # zero, so at insertion the neuron contributes one
                            # direction instead of fan_in + 1 + fan_out.
                            # MEASURED, rank(J) goes 211 -> 600 once w leaves
                            # zero. The bias is UNEVEN -- a neuron next to the
                            # 1-d output adds a single dormant direction and can
                            # never win on price, which is what drove the last
                            # hidden layer to width 2-4 -- so the rank cap was
                            # patching a measurement defect, not an
                            # architectural fact.
                            #
                            # Training the clone to wake it does NOT isolate
                            # this: it moves f, hence r, so eps is compared
                            # against a different target and every gain comes
                            # out negative (MEASURED: the ladder's own step
                            # raises eps 0.4497 -> 0.4806, and scoring that way
                            # produced zero growth events). Perturbing w
                            # instead breaks the degeneracy while leaving the
                            # residual essentially fixed, and the clone is
                            # discarded either way -- no certificate sees it.
                            scale = float(
                                getattr(
                                    config.fgd_approx,
                                    "growth_where_wake_scale",
                                    1e-3,
                                )
                            )
                            if scale > 0.0:
                                _wake_dormant_outgoing_weights(trial, scale)
                            _probe_mode = str(
                                getattr(
                                    config.fgd_approx,
                                    "growth_where_probe",
                                    "train",
                                )
                            )
                            if _probe_mode == "validation":
                                return _certificate_for(trial)
                            if _probe_mode == "both":
                                _tr = _train_epsilon(trial)
                                _va = _certificate_for(trial)
                                if _tr is None or _va is None:
                                    return _tr if _va is None else _va
                                return 0.5 * (_tr + _va)
                            return _train_epsilon(trial)

                        def _rank_score(trial: GrowingMLP) -> float | None:
                            """Score used to RANK where to grow.

                            Identical to _score_for unless the ladder score is
                            switched on, so the default path is unchanged.
                            """
                            if where_mode == "expressivity_bottleneck":
                                # Not consulted: that mode ranks by the
                                # per-layer expressivity bottleneck, so this
                                # would build the exact tangent system and its
                                # SVD once PER CANDIDATE for a number nothing
                                # reads. The candidate clones themselves are
                                # still built -- the winner is committed --
                                # but the exhaustive exact scoring is not.
                                return None
                            if where_mode == "certified_gain" and getattr(
                                config.fgd_approx,
                                "growth_where_ladder_score",
                                False,
                            ):
                                return _ladder_residual(trial)
                            return _score_for(trial)
                        if where_mode == "certified_gain" and getattr(
                            config.fgd_approx, "growth_where_prune", False
                        ) and fgd_train_probe is not None:
                            _freed = _prune_negligible_units(
                                model,
                                fgd_train_probe,
                                float(
                                    config.fgd_approx
                                    .growth_preservation_tolerance
                                ),
                            )
                            if _freed and progress is not None:
                                progress(
                                    f"[GRO] Epoch {epoch}: freed {_freed} "
                                    "unit(s) whose removal left f inside the "
                                    "growth tolerance"
                                )
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
                                    # The certified method's unified growth is
                                    # function-preserving BY DESIGN (f unchanged
                                    # so the certified steps do all the descent);
                                    # this is NOT governed by
                                    # growth_function_preserving, which only
                                    # affects the legacy paths. Validated on the
                                    # headline run: all 25 growth events left f
                                    # unchanged (delta 0), no non-FP mixing.
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
                            # Lookahead: measure this location over a
                            # HORIZON, not at a single block. The extra blocks
                            # go in the same way -- function-preserving, same
                            # tolerance -- so the trial stays a legal model,
                            # and it is this same trial that gets committed if
                            # the location wins, so the investment that was
                            # measured is the investment that is made.
                            _look = int(
                                getattr(
                                    config.fgd_approx,
                                    "growth_where_lookahead",
                                    1,
                                )
                            )
                            # SEE the horizon, BUY one block. Committing the
                            # whole horizon triples the spend for a worse mean
                            # (MEASURED: 0.925/0.908/0.874 at 1697/1348/1733,
                            # against 0.947/0.929/0.854 at 693/819/392), so
                            # the horizon informs WHERE while the pace stays
                            # exactly as it was. The scored model is a
                            # throwaway; `trial` -- one block -- is what gets
                            # committed if this location wins.
                            _probe = trial
                            if where_mode == "certified_gain" and _look > 1:
                                _probe = copy.deepcopy(trial)
                                for _ in range(_look - 1):
                                    try:
                                        grow_layer(
                                            model=_probe,
                                            train_loader=train_loader,
                                            layer_index=candidate_layer,
                                            device=device,
                                            line_search_config=(
                                                config.scaling_line_search
                                            ),
                                            optimal_update_kwargs=(
                                                unified_kwargs
                                            ),
                                            progress=None,
                                            function_preserving=True,
                                            preservation_tolerance=(
                                                config.fgd_approx
                                                .growth_preservation_tolerance
                                            ),
                                        )
                                    except RuntimeError:
                                        break
                            trials[("width", candidate_layer)] = trial
                            candidates.append(
                                Candidate(
                                    kind="width",
                                    index=candidate_layer,
                                    cost=max(
                                        count_parameters(_probe)
                                        - base_parameters,
                                        1,
                                    ),
                                    relative_error_after=_rank_score(_probe),
                                    relieves_rank_ceiling=(
                                        candidate_layer in bottlenecks
                                    ),
                                )
                            )

                        # JOINT moves: widen a layer AND the next one in the
                        # same event, priced as ONE purchase. A new feature
                        # upstream is worth only what the downstream can carry,
                        # so the value of widening layer l is conditional on
                        # widening l+1 -- a dependency no single-layer probe can
                        # express, and the reason a funnel is unreachable by
                        # one-layer-at-a-time growth (see Candidate.indices).
                        # The pair is carried in `indices`; `index` stays a real
                        # layer so every consumer that reads it keeps working.
                        if where_mode == "certified_gain" and getattr(
                            config.fgd_approx, "growth_where_joint", False
                        ):
                            for _first in growable:
                                _second = _first + 1
                                if _second not in growable:
                                    continue
                                _joint = copy.deepcopy(model)
                                _built = True
                                for _at in (_first, _second):
                                    try:
                                        grow_layer(
                                            model=_joint,
                                            train_loader=train_loader,
                                            layer_index=_at,
                                            device=device,
                                            line_search_config=(
                                                config.scaling_line_search
                                            ),
                                            optimal_update_kwargs=(
                                                unified_kwargs
                                            ),
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
                                                f"joint candidate "
                                                f"({_first},{_second}) could "
                                                f"not be built: {error}"
                                            )
                                        _built = False
                                        break
                                if not _built:
                                    continue
                                trials[("joint", _first)] = _joint
                                candidates.append(
                                    Candidate(
                                        kind="joint",
                                        index=_first,
                                        cost=max(
                                            count_parameters(_joint)
                                            - base_parameters,
                                            1,
                                        ),
                                        relative_error_after=(
                                            _rank_score(_joint)
                                        ),
                                        relieves_rank_ceiling=(
                                            _first in bottlenecks
                                            or _second in bottlenecks
                                        ),
                                        indices=(_first, _second),
                                    )
                                )

                        # Depth is excluded under certified_gain by default:
                        # the reference and the exhaustive 316-architecture
                        # search both live in the three-hidden-layer family, so
                        # letting the net deepen would compare against neither.
                        _allow_depth = where_mode != "certified_gain" or getattr(
                            config.fgd_approx, "growth_where_allow_depth", False
                        )
                        for position in (
                            range(1, len(model.layers)) if _allow_depth else ()
                        ):
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
                                    relative_error_after=_score_for(trial),
                                )
                            )

                        if where_mode == "certified_gain" and (
                            progress is not None
                        ):
                            # Per-candidate ledger. Without it there is no way
                            # to tell "the joint move does not win" from "the
                            # joint move was never evaluated" -- a distinction
                            # an earlier attempt got wrong.
                            _base_log = _rank_score(model)
                            if _base_log is not None:
                                _cols = []
                                for _cand in candidates:
                                    if _cand.relative_error_after is None:
                                        continue
                                    _gn = _base_log - _cand.relative_error_after
                                    _tag = (
                                        "+".join(str(i) for i in _cand.indices)
                                        if _cand.indices
                                        else str(_cand.index)
                                    )
                                    _cols.append(
                                        f"{_cand.kind[0].upper()}{_tag}"
                                        f":c={_cand.cost}"
                                        f",g={_gn:+.4f}"
                                        f",g/c={_gn / max(_cand.cost, 1):+.6f}"
                                    )
                                progress(
                                    f"[WHY] Epoch {epoch} widths={widths} "
                                    f"base={_base_log:.3f} " + " ".join(_cols)
                                )

                        if where_mode == "certified_gain":
                            # The base eps must come from the SAME probe the
                            # candidates were scored on, or the gain is a
                            # difference between two different certificates.
                            base_train_eps = _rank_score(model)
                            if base_train_eps is None or not math.isfinite(
                                base_train_eps
                            ):
                                # Degenerate base: fall through to the shipped
                                # rule for THIS event rather than rank on a
                                # quantity that does not exist.
                                fallback(
                                    "growth_where_base_unavailable",
                                    "train_epsilon_unavailable",
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
                            else:
                                _pool = candidates
                                if ceiling_binds:
                                    _binding = [
                                        c
                                        for c in candidates
                                        if c.relieves_rank_ceiling
                                    ]
                                    if _binding:
                                        _pool = _binding
                                ranked = rank_candidates_by_certified_gain(
                                    _pool,
                                    relative_error_before=base_train_eps,
                                    gradient_sq_norm=(
                                        validation_certificate.gradient_sq_norm
                                    ),
                                    cost_exponent=(
                                        config.fgd_approx.growth_where_cost_exponent
                                    ),
                                    min_gain_fraction=(
                                        config.fgd_approx.growth_where_min_gain_fraction
                                    ),
                                    # The clamped form, honoured at this new
                                    # site too. Numerically identical here
                                    # (threshold is 0.5) so it cannot perturb
                                    # the reference.
                                    certificate_binds=(
                                        base_train_eps
                                        >= min(
                                            config.fgd_approx.rel_error_threshold,
                                            0.5,
                                        )
                                    ),
                                )
                        else:
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
                        if where_mode == "expressivity_bottleneck":
                            # ONE neuron, into the layer that cannot express
                            # what is being asked of it. The ranking above
                            # scored candidates by what a step would gain;
                            # this replaces it with a measurement of the
                            # STRUCTURE -- TINY's extension term, without its
                            # parameter_update_decrease. See
                            # FGDApproxConfig.growth_where for why the other
                            # two rules run away on a solved task.
                            _bottlenecks = compute_expressivity_bottlenecks(
                                model, train_loader, device, config.fgd_approx
                            )
                            _width_only = [
                                c for c in candidates
                                if c.kind == "width"
                                and not c.indices
                                and 0 <= c.index < len(_bottlenecks)
                            ]
                            if progress is not None and _bottlenecks:
                                progress(
                                    f"[BOTTLENECK] Epoch {epoch} widths="
                                    f"{widths} "
                                    + " ".join(
                                        f"L{i}={v:.4e}"
                                        for i, v in enumerate(_bottlenecks)
                                    )
                                )
                            if _width_only and max(_bottlenecks) > 0.0:
                                ranked = sorted(
                                    _width_only,
                                    key=lambda c: _bottlenecks[c.index],
                                    reverse=True,
                                )
                            elif _width_only:
                                # Every layer expresses what is asked of it.
                                # That is the answer, not a missing one: no
                                # width is the bottleneck, so buy nothing.
                                fallback(
                                    "growth_where_no_bottleneck",
                                    "expressivity_bottleneck_all_zero",
                                )
                                if progress is not None:
                                    progress(
                                        f"[BOTTLENECK] Epoch {epoch}: no layer "
                                        "has an expressivity bottleneck; no "
                                        "growth"
                                    )
                                ranked = []

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
                                    f"[GRO] Unified growth at epoch {epoch} "
                                    f"[where={where_mode}]: "
                                    f"{chosen.kind} at index {chosen.index} "
                                    f"(+{chosen.cost} params, eps "
                                    f"{validation_certificate.relative_error:.3f}"
                                    + (
                                        " -> not scored"
                                        if chosen.relative_error_after is None
                                        else f" -> {chosen.relative_error_after:.3f}"
                                    )
                                    + "); "
                                    f"{len(candidates)} candidates considered"
                                    + (
                                        "; widths "
                                        + "-".join(
                                            str(int(layer.in_features))
                                            for layer in model._growable_layers
                                        )
                                        if where_mode == "certified_gain"
                                        else ""
                                    )
                                )
                            # ADAPTIVE COUNT: the "re-measuring after each
                            # purchase" ideal the note above defers. While the
                            # certificate still demands growth, keep buying at
                            # the chosen WIDTH location as long as each increment
                            # still pays -- its certificate improvement stays
                            # above min_gain of the remaining gap to threshold --
                            # stopping the instant it certifies or the returns
                            # diminish. Each purchase is re-measured by its OWN
                            # validation certificate, so attribution and every
                            # certificate condition are preserved; the COUNT is
                            # chosen by the criterion, not fixed. Off by default
                            # (one purchase per event, byte-identical).
                            # Under certified_gain this block is ALSO the
                            # pace replacement: the levelling loop it replaces
                            # bought several neurons per event, and without a
                            # burst the branch buys exactly one per epoch --
                            # arithmetically short of the 40 neurons the
                            # reference reaches in 25 epochs.
                            # ONE neuron per event under
                            # expressivity_bottleneck: the criterion names the
                            # layer that cannot express what is asked of it,
                            # and a single neuron changes that measurement, so
                            # buying more without re-measuring would spend on
                            # a bottleneck that may no longer be there. The
                            # burst's own stopping rule is the gain-per-
                            # parameter of the STEP, which is the quantity
                            # this mode exists to stop listening to.
                            _burst_on = where_mode != "expressivity_bottleneck" and (
                                getattr(
                                    config.fgd_approx,
                                    "certify_adaptive_growth",
                                    False,
                                )
                                or (
                                    where_mode == "certified_gain"
                                    and getattr(
                                        config.fgd_approx,
                                        "growth_where_burst",
                                        True,
                                    )
                                )
                            )
                            # ceiling_binds reads the VALIDATION certificate,
                            # which is the right signal for WHETHER to grow but
                            # not for how far: MEASURED, it left seed 2 at 1.0
                            # neurons per event and 241 parameters while seed 0
                            # got 1.9 and 611. Once the event has decided to
                            # buy, the burst's own marginal criterion is the
                            # brake. (Gating it on the TRAIN eps instead was
                            # tried and is worse: train certifies easily here,
                            # so growth never fired at all -- 0 events, 25
                            # parameters, accuracy 0.08-0.36.)
                            if (
                                _burst_on
                                and chosen.kind == "width"
                                and (
                                    ceiling_binds
                                    or where_mode == "certified_gain"
                                )
                            ):
                                # The burst buys while the target is unmet.
                                # Under certified_gain that target is
                                # certify_growth_target when set, so the
                                # aspiration -- not the certificate -- sets the
                                # pace. MEASURED without it, a seed whose first
                                # purchase already lands under 0.5 never bursts
                                # at all (seed 2: 0 bursts, 241 parameters,
                                # 0.611) while a seed landing above it bursts 6
                                # times (seed 1: 602 parameters, 0.900).
                                _thr = config.fgd_approx.rel_error_threshold
                                _min_gain = float(
                                    getattr(
                                        config.fgd_approx,
                                        "certify_adaptive_growth_min_gain",
                                        0.1,
                                    )
                                )
                                # HOW MUCH, without a target and without a
                                # constant. "Keep buying here while HERE is
                                # still the best place to spend": the runner-up
                                # of this event's own ranking is the reference,
                                # so the scale is the problem's own and
                                # recalibrates every event. A target instead
                                # (certify_growth_target driving the burst) was
                                # tried and rejected -- it works only if you
                                # already know where the problem ends, and it
                                # tuned this dataset rather than solving the
                                # rule. MEASURED at target 0.30: 0.893 mean at
                                # 761 parameters, i.e. the reference's budget
                                # for less accuracy.
                                # MEASURED AND REFUTED: buying at EVERY
                                # location whose gain clears the ranker's
                                # admission floor, to raise the pace. It does
                                # raise it -- seed 2 goes 392 -> 1143
                                # parameters -- but accuracy only moves 0.854
                                # -> 0.878 while the two good seeds DEGRADE
                                # (0.947 -> 0.925, 0.929 -> 0.895) and mean
                                # parameters reach 1190 against the
                                # reference's 774. It also refutes the
                                # "seed 2 is merely undersized" reading: at
                                # nearly double the reference's parameters it
                                # still trails the reference's 0.933 at 659,
                                # and 0.878 is about what an AdamW grid gives
                                # at that size, i.e. on that seed the
                                # certified step stops contributing the ~5
                                # points it contributes on the others. That is
                                # a separate pathology; more capacity does not
                                # buy it back.
                                _rival_rate = None
                                if where_mode == "certified_gain" and (
                                    len(ranked) > 1
                                ):
                                    _rival = ranked[1]
                                    _rival_gain = (
                                        base_train_eps - _rival.relative_error_after
                                    )
                                    _rival_rate = _rival_gain / max(
                                        float(_rival.cost), 1.0
                                    )
                                _eps_now = chosen.relative_error_after
                                _added = 0
                                # MEASURED AND REFUTED: letting the burst
                                # RELOCATE to whichever location currently
                                # rates best (instead of ending the event) and
                                # stopping on decay relative to the event's
                                # opening rate. A relocation commits nothing --
                                # it discards the trial and re-loops -- so
                                # events cost more and buy less: 0.915/0.392/
                                # 0.917 at 631/102/1220 parameters, against
                                # 0.947/0.929/0.854 at 693/819/392 for ending
                                # the event and re-ranking at the next one.
                                while _eps_now is not None and (
                                    _eps_now >= _thr or _rival_rate is not None
                                ):
                                    _trial = copy.deepcopy(model)
                                    try:
                                        grow_layer(
                                            model=_trial,
                                            train_loader=train_loader,
                                            layer_index=chosen.index,
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
                                    except RuntimeError:
                                        break
                                    _trial_eps = _score_for(_trial)
                                    if _trial_eps is None:
                                        break
                                    _gain = _eps_now - _trial_eps
                                    if _rival_rate is not None:
                                        # Keep buying here while HERE is still
                                        # the best place to spend, re-measuring
                                        # the alternatives after every
                                        # increment. Comparing against the
                                        # runner-up's rate from BEFORE the
                                        # purchase is stale: MEASURED, a seed
                                        # whose single purchases are very
                                        # effective (eps 1.072 -> 0.477) stops
                                        # after one and ends at 31 neurons
                                        # where the others reach 51.
                                        _rate = _gain / max(
                                            float(chosen.cost), 1.0
                                        )
                                        _still_best = True
                                        for _other in growable:
                                            if _other == chosen.index:
                                                continue
                                            _alt = copy.deepcopy(model)
                                            try:
                                                grow_layer(
                                                    model=_alt,
                                                    train_loader=train_loader,
                                                    layer_index=_other,
                                                    device=device,
                                                    line_search_config=(
                                                        config.scaling_line_search
                                                    ),
                                                    optimal_update_kwargs=(
                                                        unified_kwargs
                                                    ),
                                                    progress=None,
                                                    function_preserving=True,
                                                    preservation_tolerance=(
                                                        config.fgd_approx
                                                        .growth_preservation_tolerance
                                                    ),
                                                )
                                            except RuntimeError:
                                                continue
                                            _alt_eps = _score_for(_alt)
                                            if _alt_eps is None:
                                                continue
                                            _alt_cost = max(
                                                count_parameters(_alt)
                                                - count_parameters(model),
                                                1,
                                            )
                                            _alt_rate = (
                                                _eps_now - _alt_eps
                                            ) / float(_alt_cost)
                                            if _alt_rate >= _rate:
                                                _still_best = False
                                                break
                                        if not _still_best:
                                            break
                                    elif not (
                                        _gain
                                        > _min_gain * max(_eps_now - _thr, 1e-6)
                                    ):
                                        break
                                    model = _trial
                                    _eps_now = _trial_eps
                                    _added += 1
                                if _added and progress is not None:
                                    progress(
                                        f"[GRO] Epoch {epoch}: adaptive count "
                                        f"added {_added} more neuron-blocks at "
                                        f"index {chosen.index} (eps -> "
                                        f"{_eps_now:.3f}"
                                        + (
                                            "  certified"
                                            if _eps_now < _thr
                                            else ""
                                        )
                                        + ")"
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
                        config.fgd_approx.growth_selection == "grow_all_width"
                    ):
                        # Widen EVERY growable layer this event (each by GroMo's
                        # own min(fan-in,fan-out) count -- GroMo unchanged), so
                        # the widths bootstrap geometrically (2 -> 4 -> 8 -> ...)
                        # instead of spending the event on one layer. Every
                        # addition is function-preserving: f and the descent
                        # certificate are unchanged, only the reachable set grows
                        # faster. This is the fix for GroMo's min(fan-in,fan-out)
                        # cap on a net whose hidden layers all start at width 2.
                        all_kwargs = tiny_optimal_update_kwargs(
                            config.fgd_approx,
                            compute_delta=config.fgd_approx.growth_compute_delta,
                        )
                        widened = 0
                        for loc in list(
                            range(len(getattr(model, "_growable_layers", [])))
                        ):
                            before = count_parameters(model)
                            try:
                                grow_layer(
                                    model=model,
                                    train_loader=train_loader,
                                    layer_index=loc,
                                    device=device,
                                    line_search_config=config.scaling_line_search,
                                    optimal_update_kwargs=all_kwargs,
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
                                        f"[GRO-WARN] Epoch {epoch}: grow-all at "
                                        f"{loc} stopped: {error}"
                                    )
                                continue
                            if count_parameters(model) > before:
                                widened += 1
                        if widened:
                            growth_result = GrowthResult(
                                layer_index=0,
                                best_scaling_factor=1.0,
                                best_train_loss=float("nan"),
                                line_search=[],
                            )
                            layer_index = 0
                            selected_layer_index = 0
                            if progress is not None:
                                post = evaluate_fgd_validation_certificate(
                                    model=model,
                                    data_loader=validation_loader,
                                    device=device,
                                    config=config.fgd_approx,
                                    learning_rate=None,
                                    probe=fgd_validation_probe,
                                )
                                _post_eps = (
                                    post.relative_error
                                    if post.relative_error is not None
                                    else float("nan")
                                )
                                progress(
                                    f"[GRO] Epoch {epoch}: grew ALL {widened} "
                                    f"width locations (eps -> {_post_eps:.3f})"
                                )
                        else:
                            growth_triggered = False
                            if progress is not None:
                                progress(
                                    f"[GRO] Epoch {epoch}: grow-all added "
                                    "nothing; structure left unchanged"
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
                if nonlinear_mode and history:
                    history[-1] = replace(
                        history[-1],
                        num_params=count_parameters(model),
                        layer_index=selected_layer_index,
                        selected_layer_index=selected_layer_index,
                        nonlinear_growth_statistics_seconds=(
                            nonlinear_growth_statistics_seconds
                        ),
                        nonlinear_growth_application_seconds=(
                            nonlinear_growth_application_seconds
                        ),
                        architecture_widths=_architecture_widths(model),
                    )
                wandb_logger.log_growth_event(
                    event=growth_result,
                    epoch=epoch,
                    growth_count=growth_count,
                    architecture_widths=(
                        _architecture_widths(model) if nonlinear_mode else ()
                    ),
                    statistics_seconds=(
                        nonlinear_growth_statistics_seconds
                        if nonlinear_mode
                        else None
                    ),
                    application_seconds=(
                        nonlinear_growth_application_seconds
                        if nonlinear_mode
                        else None
                    ),
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
                    if (
                        not nonlinear_mode
                        and config.fgd_approx.learning_rate_policy
                        == "theory_interval"
                    ):
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
                    0.0
                    if nonlinear_mode
                    else current_fgd_learning_rate
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
                    nonlinear_functional_learning_rate=(
                        nonlinear_functional_learning_rate
                    ),
                    nonlinear_inner_steps=nonlinear_inner_steps,
                    nonlinear_adamw_learning_rate=(
                        config.parametric_gd.inner_learning_rate
                        if nonlinear_mode
                        else None
                    ),
                    nonlinear_weight_decay=(
                        config.parametric_gd.weight_decay
                        if nonlinear_mode
                        else None
                    ),
                    nonlinear_cosine=nonlinear_cosine,
                    nonlinear_certificate_valid=nonlinear_certificate_valid,
                    nonlinear_committed_rate=nonlinear_committed_rate,
                    nonlinear_growth_requested=nonlinear_growth_requested,
                    nonlinear_candidate_training_seconds=(
                        nonlinear_candidate_training_seconds
                    ),
                    nonlinear_certification_seconds=(
                        nonlinear_certification_seconds
                    ),
                    nonlinear_growth_statistics_seconds=(
                        nonlinear_growth_statistics_seconds
                    ),
                    nonlinear_growth_application_seconds=(
                        nonlinear_growth_application_seconds
                    ),
                    architecture_widths=_architecture_widths(model),
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
