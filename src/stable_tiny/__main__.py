"""Command-line entry point for the initial GroMo pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_tiny.pipeline import (
    load_pipeline_config,
    run_pipeline,
    with_run_overrides,
    with_wandb_overrides,
    write_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the GroMo GrowingMLP pipeline from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fgd/default.yaml"),
        help="YAML file with pipeline hyperparameters.",
    )
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot output.")
    parser.add_argument("--show-plot", action="store_true", help="Show plot window.")
    wandb_toggle = parser.add_mutually_exclusive_group()
    wandb_toggle.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging.",
    )
    wandb_toggle.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging.",
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tag", action="append", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        help="W&B mode override.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_pipeline_config(args.config)
    config = with_run_overrides(
        config,
        name=args.run_name,
        results_dir=args.results_dir,
        save_plot=False if args.no_plot else None,
        show_plot=True if args.show_plot else None,
    )
    config = with_wandb_overrides(
        config,
        enabled=True if args.wandb else False if args.no_wandb else None,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        mode=args.wandb_mode,
        tags=args.wandb_tag,
    )

    result = run_pipeline(config=config, progress=print)
    output_paths = write_outputs(result)
    for label, path in output_paths.items():
        print(f"Saved {label}: {path}")


if __name__ == "__main__":
    main()
