"""Pipeline that joins config, data, train, grow, and outputs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from stable_tiny.data import MultiSinDataLoader, SmoothSinDataLoader
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


ensure_gromo_importable()

import torch

from gromo.containers.growing_mlp import GrowingMLP


ProgressFn = Callable[[str], None]
StepType = Literal["INIT", "SGD", "GRO"]
DataKind = Literal["multi_sin", "smooth_sin"]


@dataclass(frozen=True)
class DataConfig:
    kind: DataKind = "smooth_sin"
    in_features: int = 10
    out_features: int = 3
    train_batches: int = 10
    test_batches: int = 1
    batch_size: int = 1_000
    train_seed: int = 0
    test_seed: int = 1
    active_features: int = 2
    frequency: float = 1.0
    phase_shift: float = 0.5
    interaction_strength: float = 0.25
    linear_strength: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 2
    number_hidden_layers: int = 2
    model_seed: int = 0


@dataclass(frozen=True)
class TrainingConfig:
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
    scaling_line_search: ScalingLineSearchConfig = field(
        default_factory=ScalingLineSearchConfig
    )
    growth_schedule: GrowthScheduleConfig = field(default_factory=GrowthScheduleConfig)
    run: RunConfig = field(default_factory=RunConfig)


@dataclass(frozen=True)
class HistoryEntry:
    step: int
    step_type: StepType
    train_loss: float
    test_loss: float
    train_accuracy: float
    test_accuracy: float
    learning_rate: float
    num_params: int
    layer_index: int | None = None
    scaling_factor: float | None = None


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
        "scaling_line_search",
        "growth_schedule",
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


def select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_dataloaders(
    config: PipelineConfig,
    device: torch.device,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    data_config = config.data

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
            "Use one of: multi_sin, smooth_sin."
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
    test_loader = loader_class(
        nb_sample=data_config.test_batches,
        batch_size=data_config.batch_size,
        in_features=data_config.in_features,
        out_features=data_config.out_features,
        seed=data_config.test_seed,
        device=device,
        **extra_kwargs,
    )
    return train_loader, test_loader


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


def scheduled_learning_rate(config: PipelineConfig, epoch: int) -> float:
    return learning_rate_for_epoch(
        config.lr_scheduler,
        base_learning_rate=config.optimizer.learning_rate,
        epoch=epoch,
        total_epochs=config.training.epochs,
        growth_every=config.growth_schedule.every,
        first_growth_epoch=config.growth_schedule.first_epoch,
    )


def run_pipeline(
    config: PipelineConfig,
    progress: ProgressFn | None = print,
) -> PipelineResult:
    """Run the train-grow loop from the GroMo tutorial."""
    device = select_device(config.training.device)
    train_loader, test_loader = build_dataloaders(config, device)
    model = build_model(config, device)
    loss_function = torch.nn.MSELoss()
    optimizer = build_optimizer(model, config.optimizer)
    apply_learning_rate(optimizer, scheduled_learning_rate(config, epoch=0))

    history: list[HistoryEntry] = []
    growth_events: list[GrowthResult] = []

    if progress is not None:
        progress(f"Using device: {device}")
        progress("Original model:")
        progress(str(model))

    train_metrics = evaluate_regression_metrics(
        model,
        train_loader,
        loss_function,
        device=device,
        accuracy_tolerance=config.training.accuracy_tolerance,
    )
    test_metrics = evaluate_regression_metrics(
        model,
        test_loader,
        loss_function,
        device=device,
        accuracy_tolerance=config.training.accuracy_tolerance,
    )
    last_test_loss = test_metrics.loss
    history.append(
        HistoryEntry(
            step=0,
            step_type="INIT",
            train_loss=train_metrics.loss,
            test_loss=test_metrics.loss,
            train_accuracy=train_metrics.accuracy,
            test_accuracy=test_metrics.accuracy,
            learning_rate=current_learning_rate(optimizer),
            num_params=count_parameters(model),
        )
    )
    if progress is not None:
        progress(
            f"[INIT] Epoch 0, train_loss={train_metrics.loss:.4f}, "
            f"test_loss={test_metrics.loss:.4f}, "
            f"train_acc={train_metrics.accuracy:.3f}, "
            f"test_acc={test_metrics.accuracy:.3f}"
        )

    growth_count = 0
    for epoch in range(1, config.training.epochs + 1):
        apply_learning_rate(optimizer, scheduled_learning_rate(config, epoch=epoch))

        epoch_result = train_one_epoch(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            accuracy_tolerance=config.training.accuracy_tolerance,
            gradient_clip_norm=config.training.gradient_clip_norm,
        )
        history.append(
            HistoryEntry(
                step=epoch,
                step_type="SGD",
                train_loss=epoch_result.train_loss,
                test_loss=epoch_result.test_loss,
                train_accuracy=epoch_result.train_accuracy,
                test_accuracy=epoch_result.test_accuracy,
                learning_rate=current_learning_rate(optimizer),
                num_params=count_parameters(model),
            )
        )

        if progress is not None and should_log_epoch(epoch, config):
            delta = epoch_result.test_loss - last_test_loss
            progress(
                f"[SGD] Epoch {epoch}, train_loss={epoch_result.train_loss:.4f}, "
                f"test_loss={epoch_result.test_loss:.4f} ({delta:+.4f}), "
                f"train_acc={epoch_result.train_accuracy:.3f}, "
                f"test_acc={epoch_result.test_accuracy:.3f}, "
                f"lr={current_learning_rate(optimizer):.4g}"
            )
        last_test_loss = epoch_result.test_loss

        if should_grow(epoch, config.growth_schedule):
            layer_index = layer_index_for_growth(
                growth_count=growth_count,
                number_hidden_layers=config.model.number_hidden_layers,
                config=config.growth_schedule,
            )
            if progress is not None:
                progress(f"[GRO] Growing layer {layer_index} at epoch {epoch}")

            growth_result = grow_layer(
                model=model,
                train_loader=train_loader,
                layer_index=layer_index,
                device=device,
                line_search_config=config.scaling_line_search,
                progress=progress,
            )
            growth_events.append(growth_result)
            growth_count += 1
            optimizer = build_optimizer(model, config.optimizer)
            reset_epoch = 0 if config.lr_scheduler.restart_on_growth else epoch
            apply_learning_rate(
                optimizer,
                scheduled_learning_rate(config, epoch=reset_epoch),
            )

            train_metrics = evaluate_regression_metrics(
                model,
                train_loader,
                loss_function,
                device=device,
                accuracy_tolerance=config.training.accuracy_tolerance,
            )
            test_metrics = evaluate_regression_metrics(
                model,
                test_loader,
                loss_function,
                device=device,
                accuracy_tolerance=config.training.accuracy_tolerance,
            )
            history.append(
                HistoryEntry(
                    step=epoch,
                    step_type="GRO",
                    train_loss=train_metrics.loss,
                    test_loss=test_metrics.loss,
                    train_accuracy=train_metrics.accuracy,
                    test_accuracy=test_metrics.accuracy,
                    learning_rate=current_learning_rate(optimizer),
                    num_params=count_parameters(model),
                    layer_index=layer_index,
                    scaling_factor=growth_result.best_scaling_factor,
                )
            )

            if progress is not None:
                delta = test_metrics.loss - last_test_loss
                progress(
                    f"[GRO] Epoch {epoch}, train_loss={train_metrics.loss:.4f}, "
                    f"test_loss={test_metrics.loss:.4f} ({delta:+.4f}), "
                    f"train_acc={train_metrics.accuracy:.3f}, "
                    f"test_acc={test_metrics.accuracy:.3f}, "
                    f"scaling={growth_result.best_scaling_factor:.4g}"
                )
                progress("Model after growing:")
                progress(str(model))
            last_test_loss = test_metrics.loss

    return PipelineResult(
        config=config,
        history=history,
        growth_events=growth_events,
        model=model,
        device=str(device),
    )


def result_payload(result: PipelineResult) -> dict[str, Any]:
    config_payload = asdict(result.config)
    config_payload["run"]["results_dir"] = str(result.config.run.results_dir)
    return {
        "config": config_payload,
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
        from stable_tiny.plotting import plot_history, plot_parameters

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

    return output_paths
