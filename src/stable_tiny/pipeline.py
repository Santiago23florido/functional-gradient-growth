"""Pipeline that joins config, data, train, grow, and outputs."""

from __future__ import annotations

import json
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
from stable_tiny.fgd_approx import (
    FGDApproxConfig,
    FGDLayerRelError,
    FGDOutputRelError,
    batch_functional_mse_loss,
    evaluate_fgd_validation_certificate,
    select_certifying_growth_layer_index,
    should_trigger_fgd_growth,
    tiny_optimal_update_kwargs,
    train_one_epoch_fgd_approx,
)
from stable_tiny.gromo_setup import ensure_gromo_importable
from stable_tiny.grow import GrowthResult, ScalingLineSearchConfig, grow_layer
from stable_tiny.growth_schedule import (
    GrowthScheduleConfig,
    layer_index_for_growth,
    should_grow,
)
from stable_tiny.lr_scheduler import (
    LRSchedulerConfig,
    apply_learning_rate,
    learning_rate_for_epoch,
)
from stable_tiny.optim import OptimizerConfig, build_optimizer, current_learning_rate
from stable_tiny.train import (
    count_parameters,
    evaluate_regression_metrics,
    train_one_epoch,
)
from stable_tiny.wandb_logging import WandbConfig, build_wandb_logger


ensure_gromo_importable()

import torch

from gromo.containers.growing_mlp import GrowingMLP


ProgressFn = Callable[[str], None]
TrainingMethod = Literal["normal", "fgd_approx"]
StepType = Literal["INIT", "SGD", "FGD", "GRO"]
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


@dataclass
class PipelineResult:
    config: PipelineConfig
    history: list[HistoryEntry]
    growth_events: list[GrowthResult]
    model: GrowingMLP
    device: str


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
        "scaling_line_search",
        "growth_schedule",
        "wandb",
        "run",
    }
    unknown_sections = sorted(set(raw) - known_sections)
    if unknown_sections:
        joined = ", ".join(unknown_sections)
        raise ValueError(f"Unknown config sections in {config_path}: {joined}")

    return PipelineConfig(
        data=_section_dataclass("data", DataConfig, raw),
        model=_section_dataclass("model", ModelConfig, raw),
        training=_section_dataclass("training", TrainingConfig, raw),
        optimizer=_section_dataclass("optimizer", OptimizerConfig, raw),
        lr_scheduler=_section_dataclass("lr_scheduler", LRSchedulerConfig, raw),
        fgd_approx=_section_dataclass("fgd_approx", FGDApproxConfig, raw),
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
) -> float:
    model.eval()
    total_loss = 0.0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        total_loss += float(batch_functional_mse_loss(model(x), y).detach().item())
    return total_loss


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
        )
        theory_loss_star = config.fgd_approx.theory_loss_star
        initial_functional_gap = max(initial_functional_loss - theory_loss_star, 0.0)
        fgd_epoch_count = 0
        fgd_min_gradient_sq_norm: float | None = None
        fgd_min_positive_learning_rate: float | None = None
        fgd_min_descent_coefficient: float | None = None
        fgd_global_contraction_product = 1.0
        fgd_global_bound_only_failed_epochs = 0
        fgd_previous_train_loss: float | None = None
        fgd_stalled_epochs = 0
        projection_group_limit = max(
            1,
            config.fgd_approx.projection_group_max
            if config.fgd_approx.projection_group_max is not None
            else max(len(train_loader), len(validation_loader)),
        )
        current_projection_group_size = max(
            1,
            min(config.fgd_approx.projection_group_size, projection_group_limit),
        )

        def reset_fgd_certificate() -> None:
            """Re-anchor the per-mode FGD bounds at the current loss."""
            nonlocal initial_functional_gap, fgd_epoch_count
            nonlocal fgd_min_gradient_sq_norm
            nonlocal fgd_min_positive_learning_rate
            nonlocal fgd_min_descent_coefficient
            nonlocal fgd_global_contraction_product
            nonlocal fgd_global_bound_only_failed_epochs
            nonlocal fgd_previous_train_loss, fgd_stalled_epochs
            initial_functional_gap = max(
                evaluate_functional_loss(model, validation_loader, device)
                - theory_loss_star,
                0.0,
            )
            fgd_epoch_count = 0
            fgd_min_gradient_sq_norm = None
            fgd_min_positive_learning_rate = None
            fgd_min_descent_coefficient = None
            fgd_global_contraction_product = 1.0
            fgd_global_bound_only_failed_epochs = 0
            fgd_previous_train_loss = None
            fgd_stalled_epochs = 0
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
            learning_rate = (
                current_fgd_learning_rate
                if use_fgd_theory_learning_rate
                else scheduled_learning_rate(
                    config,
                    epoch=epoch,
                    cycle_start_epoch=lr_cycle_start_epoch,
                )
            )
            apply_learning_rate(optimizer, learning_rate)

            rel_error: float | None = None
            selected_layer_index: int | None = None
            fgd_layer_rel_errors: list[FGDLayerRelError] = []
            fgd_output_rel_error: FGDOutputRelError | None = None
            fgd_learning_rate_upper_bound: float | None = None
            fgd_max_valid_learning_rate: float | None = None
            fgd_learning_rate_interval_valid: bool | None = None
            fgd_learning_rate_clipped_batches = 0
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
            fgd_growth_requested = False
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
                fgd_epoch_result = train_one_epoch_fgd_approx(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    loss_function=loss_function,
                    device=device,
                    learning_rate=learning_rate,
                    accuracy_tolerance=config.training.accuracy_tolerance,
                    config=config.fgd_approx,
                    projection_group_size=current_projection_group_size,
                    classification=classification,
                )
                epoch_result = fgd_epoch_result
                selected_layer_index = fgd_epoch_result.selected_layer_index
                fgd_layer_rel_errors = fgd_epoch_result.layer_relative_errors
                fgd_learning_rate_clipped_batches = (
                    fgd_epoch_result.learning_rate_clipped_batches
                )
                fgd_loss_descent_valid = fgd_epoch_result.loss_descent_valid
                fgd_loss_non_descent_batches = (
                    fgd_epoch_result.loss_non_descent_batches
                )
                if fgd_epoch_result.next_learning_rate is not None:
                    current_fgd_learning_rate = fgd_epoch_result.next_learning_rate
                    apply_learning_rate(optimizer, current_fgd_learning_rate)
                if fgd_epoch_result.learning_rate is not None:
                    entry_learning_rate = fgd_epoch_result.learning_rate

                validation_certificate = evaluate_fgd_validation_certificate(
                    model=model,
                    data_loader=validation_loader,
                    device=device,
                    config=config.fgd_approx,
                    learning_rate=fgd_epoch_result.min_positive_learning_rate,
                    projection_group_size=current_projection_group_size,
                )
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

                if (
                    fgd_gradient_sq_norm is not None
                    and fgd_epoch_result.min_positive_learning_rate is not None
                ):
                    fgd_epoch_count += 1
                    fgd_min_gradient_sq_norm = (
                        fgd_gradient_sq_norm
                        if fgd_min_gradient_sq_norm is None
                        else min(fgd_min_gradient_sq_norm, fgd_gradient_sq_norm)
                    )

                if fgd_epoch_result.min_positive_learning_rate is not None:
                    fgd_min_positive_learning_rate = (
                        fgd_epoch_result.min_positive_learning_rate
                        if fgd_min_positive_learning_rate is None
                        else min(
                            fgd_min_positive_learning_rate,
                            fgd_epoch_result.min_positive_learning_rate,
                        )
                    )

                if (
                    fgd_theory_descent_coefficient is not None
                    and fgd_theory_descent_coefficient > 0.0
                ):
                    fgd_min_descent_coefficient = (
                        fgd_theory_descent_coefficient
                        if fgd_min_descent_coefficient is None
                        else min(
                            fgd_min_descent_coefficient,
                            fgd_theory_descent_coefficient,
                        )
                    )

                if (
                    fgd_epoch_count > 0
                    and fgd_min_positive_learning_rate is not None
                    and fgd_min_descent_coefficient is not None
                    and fgd_min_positive_learning_rate > 0.0
                    and fgd_min_descent_coefficient > 0.0
                ):
                    fgd_stationary_bound = initial_functional_gap / (
                        fgd_epoch_count
                        * fgd_min_positive_learning_rate
                        * fgd_min_descent_coefficient
                    )
                    fgd_stationary_bound_valid = (
                        fgd_min_gradient_sq_norm is not None
                        and fgd_min_gradient_sq_norm
                        <= fgd_stationary_bound + config.fgd_approx.eps
                    )

                    beta = config.fgd_approx.theory_beta
                    mu = config.fgd_approx.theory_mu
                    epoch_eta = fgd_epoch_result.min_positive_learning_rate
                    epoch_r = fgd_theory_descent_coefficient
                    if (
                        epoch_eta is not None
                        and epoch_r is not None
                        and beta > 0.0
                        and mu > 0.0
                    ):
                        contraction = 1.0 - (
                            2.0 * epoch_eta * mu * epoch_r / (beta**2)
                        )
                        fgd_global_contraction = contraction
                        fgd_global_contraction_product *= contraction
                        fgd_global_bound = (
                            fgd_global_contraction_product
                            * initial_functional_gap
                        )
                        current_functional_gap = max(
                            evaluate_functional_loss(
                                model,
                                validation_loader,
                                device,
                            )
                            - theory_loss_star,
                            0.0,
                        )
                        fgd_global_bound_valid = (
                            current_functional_gap
                            <= fgd_global_bound + config.fgd_approx.eps
                        )

                if (
                    fgd_gradient_sq_norm is not None
                    and fgd_min_positive_learning_rate is not None
                    and fgd_min_descent_coefficient is not None
                ):
                    if fgd_stationary_bound_valid is None:
                        fgd_stationary_bound_valid = False
                    if fgd_global_bound_valid is None:
                        fgd_global_bound_valid = False

                fgd_conditions_failed = (
                    fgd_relative_error_condition_valid is False
                    or fgd_stationary_bound_valid is False
                    or fgd_global_bound_valid is False
                )
                only_global_bound_failed = (
                    fgd_global_bound_valid is False
                    and fgd_relative_error_condition_valid is not False
                    and fgd_stationary_bound_valid is not False
                    and fgd_sensor_valid is not False
                )
                if only_global_bound_failed:
                    fgd_global_bound_only_failed_epochs += 1
                    if config.fgd_approx.global_bound_action == "ignore":
                        fgd_growth_requested = False
                    elif config.fgd_approx.global_bound_action == "grow":
                        fgd_growth_requested = (
                            config.growth_schedule.enabled
                            and fgd_conditions_failed
                            and should_trigger_fgd_growth(
                                relative_error=config.fgd_approx.rel_error_threshold,
                                epoch=epoch,
                                last_growth_epoch=last_growth_epoch,
                                config=config.fgd_approx,
                            )
                        )
                    else:
                        global_bound_patience_reached = (
                            fgd_global_bound_only_failed_epochs
                            >= max(1, config.fgd_approx.global_bound_lr_patience)
                        )
                        if (
                            use_fgd_theory_learning_rate
                            and global_bound_patience_reached
                            and fgd_max_valid_learning_rate is not None
                            and fgd_max_valid_learning_rate
                            > config.fgd_approx.theory_lr_min + config.fgd_approx.eps
                        ):
                            adjusted_learning_rate = min(
                                current_fgd_learning_rate
                                * config.fgd_approx.lr_backtrack,
                                fgd_max_valid_learning_rate,
                            )
                            if (
                                adjusted_learning_rate
                                > config.fgd_approx.theory_lr_min
                                + config.fgd_approx.eps
                                and adjusted_learning_rate
                                < current_fgd_learning_rate - config.fgd_approx.eps
                            ):
                                current_fgd_learning_rate = adjusted_learning_rate
                                apply_learning_rate(optimizer, current_fgd_learning_rate)
                                fgd_theory_learning_rate_adjusted = True
                                reset_fgd_certificate()
                            else:
                                fgd_growth_requested = (
                                    config.growth_schedule.enabled
                                    and fgd_conditions_failed
                                    and should_trigger_fgd_growth(
                                        relative_error=(
                                            config.fgd_approx.rel_error_threshold
                                        ),
                                        epoch=epoch,
                                        last_growth_epoch=last_growth_epoch,
                                        config=config.fgd_approx,
                                    )
                                )
                        elif not global_bound_patience_reached:
                            fgd_growth_requested = False
                        else:
                            fgd_growth_requested = (
                                config.growth_schedule.enabled
                                and fgd_conditions_failed
                                and should_trigger_fgd_growth(
                                    relative_error=config.fgd_approx.rel_error_threshold,
                                    epoch=epoch,
                                    last_growth_epoch=last_growth_epoch,
                                    config=config.fgd_approx,
                                )
                            )
                else:
                    fgd_global_bound_only_failed_epochs = 0
                    fgd_growth_requested = (
                        config.growth_schedule.enabled
                        and fgd_conditions_failed
                        and should_trigger_fgd_growth(
                            relative_error=config.fgd_approx.rel_error_threshold,
                            epoch=epoch,
                            last_growth_epoch=last_growth_epoch,
                            config=config.fgd_approx,
                        )
                    )
                if (
                    use_fgd_theory_learning_rate
                    and not fgd_growth_requested
                    and not only_global_bound_failed
                    and (
                        fgd_stationary_bound_valid is False
                        or fgd_global_bound_valid is False
                    )
                ):
                    adjusted_learning_rate = max(
                        config.fgd_approx.theory_lr_min,
                        current_fgd_learning_rate * config.fgd_approx.lr_backtrack,
                    )
                    if adjusted_learning_rate < current_fgd_learning_rate:
                        current_fgd_learning_rate = adjusted_learning_rate
                        apply_learning_rate(optimizer, current_fgd_learning_rate)
                        fgd_theory_learning_rate_adjusted = True
                step_type = "FGD"
            else:
                raise ValueError(
                    f"Unsupported training method '{config.training.method}'. "
                    "Use one of: normal, fgd_approx."
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
                        f"learning-rate clipped on "
                        f"{fgd_learning_rate_clipped_batches} batch(es)"
                    )
                if fgd_skipped_batches > 0:
                    warnings.append(f"skipped {fgd_skipped_batches} batch(es)")
                if fgd_loss_descent_valid is False:
                    warnings.append(
                        f"loss did not descend on "
                        f"{fgd_loss_non_descent_batches} batch(es)"
                    )
                if fgd_sensor_valid is False:
                    warnings.append(
                        f"sensor invalid on "
                        f"{fgd_sensor_invalid_batches} validation batch(es)"
                    )
                if fgd_stationary_bound_valid is False:
                    warnings.append("stationary-point bound failed")
                if fgd_global_bound_valid is False:
                    warnings.append("global-convergence bound failed")
                if fgd_theory_learning_rate_adjusted:
                    warnings.append(
                        "learning-rate adjusted for accumulated theory bounds"
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
                        rel_error is not None
                        and should_trigger_fgd_growth(
                            relative_error=rel_error,
                            epoch=epoch,
                            last_growth_epoch=last_growth_epoch,
                            config=config.fgd_approx,
                        )
                    )
                )

            if growth_triggered:
                if config.training.method == "fgd_approx":
                    if (
                        config.fgd_approx.layer_selection == "certifying"
                        and config.fgd_approx.projection_solver != "gromo_layer"
                    ):
                        certified_layer_index = select_certifying_growth_layer_index(
                            model=model,
                            train_loader=train_loader,
                            validation_loader=validation_loader,
                            device=device,
                            config=config.fgd_approx,
                            line_search_config=config.scaling_line_search,
                            projection_group_size=current_projection_group_size,
                        )
                    else:
                        certified_layer_index = None

                    layer_index = (
                        certified_layer_index
                        if certified_layer_index is not None
                        else selected_layer_index
                        if selected_layer_index is not None
                        else layer_index_for_growth(
                            growth_count=growth_count,
                            number_hidden_layers=config.model.number_hidden_layers,
                            config=config.growth_schedule,
                        )
                    )
                    selected_layer_index = layer_index
                    optimal_update_kwargs = tiny_optimal_update_kwargs(
                        config.fgd_approx,
                        compute_delta=config.fgd_approx.growth_compute_delta,
                    )
                else:
                    layer_index = layer_index_for_growth(
                        growth_count=growth_count,
                        number_hidden_layers=config.model.number_hidden_layers,
                        config=config.growth_schedule,
                    )
                    optimal_update_kwargs = None
                if progress is not None:
                    progress(f"[GRO] Growing layer {layer_index} at epoch {epoch}")

                growth_result = grow_layer(
                    model=model,
                    train_loader=train_loader,
                    layer_index=layer_index,
                    device=device,
                    line_search_config=config.scaling_line_search,
                    optimal_update_kwargs=optimal_update_kwargs,
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
                    # them from the post-growth loss.
                    initial_functional_gap = max(
                        evaluate_functional_loss(model, validation_loader, device)
                        - theory_loss_star,
                        0.0,
                    )
                    fgd_epoch_count = 0
                    fgd_min_gradient_sq_norm = None
                    fgd_min_positive_learning_rate = None
                    fgd_min_descent_coefficient = None
                    fgd_global_contraction_product = 1.0
                    fgd_global_bound_only_failed_epochs = 0
                    if (
                        config.fgd_approx.learning_rate_policy == "theory_interval"
                        and config.lr_scheduler.restart_on_growth
                    ):
                        current_fgd_learning_rate = max(
                            current_fgd_learning_rate,
                            config.fgd_approx.theory_lr_initial,
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
