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
            "validation/*",
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
            "validation/loss": entry.validation_loss,
            "test/loss": entry.test_loss,
            "train/accuracy": entry.train_accuracy,
            "validation/accuracy": entry.validation_accuracy,
            "test/accuracy": entry.test_accuracy,
            "optimizer/learning_rate": entry.learning_rate,
            "model/num_params": entry.num_params,
            "event/step_type": entry.step_type,
        }

        if entry.rel_error is not None:
            payload["fgd/relative_error"] = entry.rel_error

        learning_rate_upper_bound = getattr(
            entry,
            "fgd_learning_rate_upper_bound",
            None,
        )
        if learning_rate_upper_bound is not None:
            payload["fgd/learning_rate_upper_bound"] = learning_rate_upper_bound

        max_valid_learning_rate = getattr(
            entry,
            "fgd_max_valid_learning_rate",
            None,
        )
        if max_valid_learning_rate is not None:
            payload["fgd/max_valid_learning_rate"] = max_valid_learning_rate

        learning_rate_interval_valid = getattr(
            entry,
            "fgd_learning_rate_interval_valid",
            None,
        )
        if learning_rate_interval_valid is not None:
            payload["fgd/learning_rate_interval_valid"] = learning_rate_interval_valid

        relative_error_condition_valid = getattr(
            entry,
            "fgd_relative_error_condition_valid",
            None,
        )
        if relative_error_condition_valid is not None:
            payload["fgd/relative_error_condition_valid"] = (
                relative_error_condition_valid
            )

        loss_descent_valid = getattr(entry, "fgd_loss_descent_valid", None)
        if loss_descent_valid is not None:
            payload["fgd/validation_loss_descent_valid"] = loss_descent_valid

        for attribute_name, metric_name in (
            ("fgd_gradient_sq_norm", "fgd/gradient_sq_norm"),
            ("fgd_min_gradient_sq_norm", "fgd/min_gradient_sq_norm"),
            (
                "fgd_theory_descent_coefficient",
                "fgd/theory_descent_coefficient",
            ),
            ("fgd_stationary_bound", "fgd/stationary_bound"),
            ("fgd_global_bound", "fgd/global_bound"),
            ("fgd_global_contraction", "fgd/global_contraction"),
        ):
            value = getattr(entry, attribute_name, None)
            if value is not None:
                payload[metric_name] = value

        stationary_bound_valid = getattr(
            entry,
            "fgd_stationary_bound_valid",
            None,
        )
        if stationary_bound_valid is not None:
            payload["fgd/stationary_bound_valid"] = stationary_bound_valid

        global_bound_valid = getattr(entry, "fgd_global_bound_valid", None)
        if global_bound_valid is not None:
            payload["fgd/global_bound_valid"] = global_bound_valid

        theory_learning_rate_adjusted = getattr(
            entry,
            "fgd_theory_learning_rate_adjusted",
            False,
        )
        if fgd_payload_active := (
            getattr(entry, "rel_error", None) is not None
            or learning_rate_interval_valid is not None
            or relative_error_condition_valid is not None
            or loss_descent_valid is not None
            or stationary_bound_valid is not None
            or global_bound_valid is not None
            or getattr(entry, "fgd_sensor_valid", None) is not None
        ):
            payload["fgd/theory_learning_rate_adjusted"] = (
                theory_learning_rate_adjusted
            )

        if fgd_payload_active:
            payload["fgd/learning_rate_clipped_by_validation"] = bool(
                getattr(entry, "fgd_learning_rate_clipped_batches", 0)
            )
            payload["fgd/validation_rejected_batches"] = getattr(
                entry,
                "fgd_skipped_batches",
                0,
            )
            payload["fgd/validation_loss_non_descent_epochs"] = getattr(
                entry,
                "fgd_loss_non_descent_batches",
                0,
            )
            sensor_valid = getattr(entry, "fgd_sensor_valid", None)
            if sensor_valid is not None:
                payload["fgd/sensor_valid"] = sensor_valid
            payload["fgd/sensor_invalid_batches"] = getattr(
                entry,
                "fgd_sensor_invalid_batches",
                0,
            )
            candidate_accepted = getattr(entry, "fgd_candidate_accepted", None)
            if candidate_accepted is not None:
                payload["fgd/candidate_accepted"] = candidate_accepted
            payload["fgd/lr_search_trials"] = getattr(
                entry,
                "fgd_lr_search_trials",
                0,
            )
            approximation_kind = getattr(
                entry,
                "fgd_approximation_kind",
                None,
            )
            if approximation_kind is not None:
                payload["fgd/approximation_kind"] = approximation_kind
            payload["fgd/secant_attempted"] = getattr(
                entry,
                "fgd_secant_attempted",
                False,
            )
            secant_accepted = getattr(entry, "fgd_secant_accepted", None)
            if secant_accepted is not None:
                payload["fgd/secant_accepted"] = secant_accepted
            payload["fgd/secant_trials"] = getattr(
                entry,
                "fgd_secant_trials",
                0,
            )
            growth_probe_improved = getattr(
                entry,
                "fgd_growth_probe_improved",
                None,
            )
            if growth_probe_improved is not None:
                payload["fgd/growth_probe_improved"] = growth_probe_improved

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
