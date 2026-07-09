"""Pipeline that joins config, data, train, grow, and outputs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from stable_tiny.data import MultiSinDataLoader
from stable_tiny.gromo_setup import ensure_gromo_importable
from stable_tiny.grow import GrowthResult, grow_layer
from stable_tiny.train import count_parameters, evaluate_loss, train_one_epoch


ensure_gromo_importable()

import torch

from gromo.containers.growing_mlp import GrowingMLP


ProgressFn = Callable[[str], None]
StepType = Literal["INIT", "SGD", "GRO"]


@dataclass(frozen=True)
class DataConfig:
    in_features: int = 10
    out_features: int = 3
    train_batches: int = 10
    test_batches: int = 1
    batch_size: int = 1_000
    train_seed: int = 0
    test_seed: int = 1


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 2
    number_hidden_layers: int = 2
    model_seed: int = 0


@dataclass(frozen=True)
class TrainingConfig:
    growth_steps: int = 4
    intermediate_epochs: int = 3
    learning_rate: float = 0.01
    scaling_factors: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0)
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
    run: RunConfig = field(default_factory=RunConfig)


@dataclass(frozen=True)
class HistoryEntry:
    step: int
    step_type: StepType
    test_loss: float
    num_params: int
    train_loss: float | None = None
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

    if section_type is TrainingConfig and "scaling_factors" in values:
        values["scaling_factors"] = tuple(float(value) for value in values["scaling_factors"])

    if section_type is RunConfig and "results_dir" in values:
        values["results_dir"] = Path(values["results_dir"])

    return section_type(**values)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load pipeline hyperparameters from YAML."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected a YAML mapping in {config_path}")

    known_sections = {"data", "model", "training", "run"}
    unknown_sections = sorted(set(raw) - known_sections)
    if unknown_sections:
        joined = ", ".join(unknown_sections)
        raise ValueError(f"Unknown config sections in {config_path}: {joined}")

    return PipelineConfig(
        data=_section_dataclass("data", DataConfig, raw),
        model=_section_dataclass("model", ModelConfig, raw),
        training=_section_dataclass("training", TrainingConfig, raw),
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
) -> tuple[MultiSinDataLoader, MultiSinDataLoader]:
    data_config = config.data
    train_loader = MultiSinDataLoader(
        nb_sample=data_config.train_batches,
        batch_size=data_config.batch_size,
        in_features=data_config.in_features,
        out_features=data_config.out_features,
        seed=data_config.train_seed,
        device=device,
    )
    test_loader = MultiSinDataLoader(
        nb_sample=data_config.test_batches,
        batch_size=data_config.batch_size,
        in_features=data_config.in_features,
        out_features=data_config.out_features,
        seed=data_config.test_seed,
        device=device,
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


def run_pipeline(
    config: PipelineConfig,
    progress: ProgressFn | None = print,
) -> PipelineResult:
    """Run the train-grow loop from the GroMo tutorial."""
    device = select_device(config.training.device)
    train_loader, test_loader = build_dataloaders(config, device)
    model = build_model(config, device)
    loss_function = torch.nn.MSELoss()

    history: list[HistoryEntry] = []
    growth_events: list[GrowthResult] = []

    if progress is not None:
        progress(f"Using device: {device}")
        progress("Original model:")
        progress(str(model))

    last_test_loss = evaluate_loss(model, test_loader, loss_function, device=device)
    history.append(
        HistoryEntry(
            step=0,
            step_type="INIT",
            test_loss=last_test_loss,
            num_params=count_parameters(model),
        )
    )
    if progress is not None:
        progress(f"[INIT] Step 0, test_loss={last_test_loss:.4f}")

    for growth_step in range(config.training.growth_steps):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
        )

        for epoch in range(1, config.training.intermediate_epochs + 1):
            epoch_result = train_one_epoch(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                loss_function=loss_function,
                device=device,
            )
            current_step = epoch + growth_step * (
                config.training.intermediate_epochs + 1
            )
            history.append(
                HistoryEntry(
                    step=current_step,
                    step_type="SGD",
                    train_loss=epoch_result.train_loss,
                    test_loss=epoch_result.test_loss,
                    num_params=count_parameters(model),
                )
            )

            if progress is not None:
                delta = epoch_result.test_loss - last_test_loss
                progress(
                    f"[SGD] Step {current_step}, "
                    f"test_loss={epoch_result.test_loss:.4f} ({delta:+.4f})"
                )
            last_test_loss = epoch_result.test_loss

        layer_index = growth_step % max(1, config.model.number_hidden_layers)
        if progress is not None:
            progress(f"[GRO] Growing layer {layer_index}")

        growth_result = grow_layer(
            model=model,
            train_loader=train_loader,
            layer_index=layer_index,
            device=device,
            scaling_factors=config.training.scaling_factors,
            progress=progress,
        )
        growth_events.append(growth_result)

        test_loss = evaluate_loss(model, test_loader, loss_function, device=device)
        current_step = (growth_step + 1) * (config.training.intermediate_epochs + 1)
        history.append(
            HistoryEntry(
                step=current_step,
                step_type="GRO",
                train_loss=growth_result.best_train_loss,
                test_loss=test_loss,
                num_params=count_parameters(model),
                layer_index=layer_index,
                scaling_factor=growth_result.best_scaling_factor,
            )
        )

        if progress is not None:
            delta = test_loss - last_test_loss
            progress(
                f"[GRO] Step {current_step}, test_loss={test_loss:.4f} "
                f"({delta:+.4f}), scaling={growth_result.best_scaling_factor:.4g}"
            )
            progress("Model after growing:")
            progress(str(model))
        last_test_loss = test_loss

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
        from stable_tiny.plotting import plot_history

        plot_path = run_config.results_dir / f"{run_config.name}_progress.png"
        saved_plot = plot_history(
            result.history,
            output_path=plot_path,
            show=run_config.show_plot,
        )
        if saved_plot is not None:
            output_paths["plot"] = saved_plot

    return output_paths
