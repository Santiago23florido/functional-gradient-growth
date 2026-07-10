"""Optional Weights & Biases logging for experiment runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal


WandbMode = Literal["online", "offline", "disabled"]


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "stable-tiny"
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    job_type: str = "train"
    tags: tuple[str, ...] = ()
    notes: str | None = None
    mode: WandbMode = "online"
    dir: str | None = "results"
    log_code: bool = False


class NullWandbLogger:
    """No-op logger used when W&B is disabled."""

    enabled = False

    def start(self, *, run_name: str, config_payload: dict[str, Any]) -> None:
        return None

    def log_history_entry(self, entry: Any) -> None:
        return None

    def log_growth_event(
        self,
        *,
        event: Any,
        epoch: int,
        growth_count: int,
    ) -> None:
        return None

    def finish(self, *, history: list[Any] | None = None) -> None:
        return None

    def abort(self) -> None:
        return None


class WandbRunLogger:
    """Thin wrapper around wandb so the rest of the pipeline stays optional."""

    enabled = True

    def __init__(self, config: WandbConfig) -> None:
        self.config = config
        self._wandb: Any | None = None
        self._run: Any | None = None
        self._growth_rows: list[list[Any]] = []
        self._line_search_rows: list[list[Any]] = []

    def start(self, *, run_name: str, config_payload: dict[str, Any]) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb.enabled is true, but the 'wandb' package is not installed. "
                "Install it inside the active environment with: pip install wandb"
            ) from exc

        self._wandb = wandb
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=False)

        self._run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=self.config.run_name or run_name,
            group=self.config.group,
            job_type=self.config.job_type,
            tags=list(self.config.tags),
            notes=self.config.notes,
            mode=self.config.mode,
            dir=self.config.dir,
            config=config_payload,
        )
        self._define_metrics()

        if self.config.log_code:
            self._run.log_code(".")

    def _define_metrics(self) -> None:
        if self._run is None:
            return

        self._run.define_metric("epoch")
        for pattern in (
            "train/*",
            "test/*",
            "optimizer/*",
            "model/*",
            "fgd/*",
            "growth/*",
        ):
            self._run.define_metric(pattern, step_metric="epoch")

    def log_history_entry(self, entry: Any) -> None:
        if self._run is None:
            return

        payload: dict[str, Any] = {
            "epoch": entry.step,
            "train/loss": entry.train_loss,
            "test/loss": entry.test_loss,
            "train/accuracy": entry.train_accuracy,
            "test/accuracy": entry.test_accuracy,
            "optimizer/learning_rate": entry.learning_rate,
            "model/num_params": entry.num_params,
            "event/step_type": entry.step_type,
        }

        if entry.rel_error is not None:
            payload["fgd/relative_error"] = entry.rel_error

        if getattr(entry, "selected_layer_index", None) is not None:
            payload["fgd/selected_layer_index"] = entry.selected_layer_index

        output_error = getattr(entry, "fgd_output_rel_error", None)
        if output_error is not None:
            payload["fgd/approximation_norm"] = output_error.approximation_norm
            payload["fgd/target_norm"] = output_error.target_norm
            payload["fgd/directional_cosine"] = output_error.directional_cosine
        else:
            selected_layer = None
            for layer_error in getattr(entry, "fgd_layer_rel_errors", []):
                if layer_error.layer_index == getattr(
                    entry, "selected_layer_index", None
                ):
                    selected_layer = layer_error
                    break

            if selected_layer is not None:
                payload["fgd/approximation_norm"] = selected_layer.approximation_norm
                payload["fgd/target_norm"] = selected_layer.target_norm
                payload["fgd/directional_cosine"] = selected_layer.directional_cosine

        if entry.layer_index is not None:
            payload["growth/layer_index"] = entry.layer_index

        if entry.scaling_factor is not None:
            payload["growth/scaling_factor"] = entry.scaling_factor

        self._run.log(payload)

    def log_growth_event(
        self,
        *,
        event: Any,
        epoch: int,
        growth_count: int,
    ) -> None:
        if self._run is None:
            return

        self._growth_rows.append(
            [
                growth_count,
                epoch,
                event.layer_index,
                event.best_scaling_factor,
                event.best_train_loss,
            ]
        )
        for point_index, point in enumerate(event.line_search):
            self._line_search_rows.append(
                [
                    growth_count,
                    epoch,
                    point_index,
                    event.layer_index,
                    point.scaling_factor,
                    point.train_loss,
                ]
            )

        self._run.log(
            {
                "epoch": epoch,
                "growth/count": growth_count,
                "growth/event": 1,
                "growth/best_scaling_factor": event.best_scaling_factor,
                "growth/best_train_loss": event.best_train_loss,
            }
        )

    def finish(self, *, history: list[Any] | None = None) -> None:
        if self._run is None or self._wandb is None:
            return

        if history:
            final_entry = history[-1]
            self._run.summary["final/train_loss"] = final_entry.train_loss
            self._run.summary["final/test_loss"] = final_entry.test_loss
            self._run.summary["final/train_accuracy"] = final_entry.train_accuracy
            self._run.summary["final/test_accuracy"] = final_entry.test_accuracy
            self._run.summary["final/num_params"] = final_entry.num_params

        if self._growth_rows:
            growth_table = self._wandb.Table(
                columns=[
                    "growth_count",
                    "epoch",
                    "layer_index",
                    "best_scaling_factor",
                    "best_train_loss",
                ],
                data=self._growth_rows,
            )
            self._run.log({"growth/events_table": growth_table})

        if self._line_search_rows:
            line_search_table = self._wandb.Table(
                columns=[
                    "growth_count",
                    "epoch",
                    "point_index",
                    "layer_index",
                    "scaling_factor",
                    "train_loss",
                ],
                data=self._line_search_rows,
            )
            self._run.log({"growth/line_search_table": line_search_table})

        self._run.finish()
        self._run = None
        self._wandb = None

    def abort(self) -> None:
        if self._run is None:
            return

        self._run.finish(exit_code=1)
        self._run = None
        self._wandb = None


def build_wandb_logger(config: WandbConfig) -> NullWandbLogger | WandbRunLogger:
    if not config.enabled:
        return NullWandbLogger()
    return WandbRunLogger(config)
