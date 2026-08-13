#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/dev/functional-gradient-growth}"
FGG_SCRATCH_ROOT="${FGG_SCRATCH_ROOT:-$HOME/datasets}"
GRID_CONFIG="${GRID_CONFIG:-grid_search/fixed_architectures_ladder_under600.yaml}"
RUN_NAME="${RUN_NAME:-fixed_grid_ladder_under600_v1}"
MAX_CONCURRENT="${MAX_CONCURRENT:-30}"
CONDA_ENV="${CONDA_ENV:-cct}"
SLURM_MEM="${SLURM_MEM:-2G}"
SLURM_TIME="${SLURM_TIME:-01:00:00}"

[[ -n "$RUN_NAME" ]] || { echo "RUN_NAME must not be empty" >&2; exit 2; }
[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_CONCURRENT must be a positive integer" >&2
  exit 2
}
[[ -e "$FGG_SCRATCH_ROOT" ]] || {
  echo "Scratch path does not exist: $FGG_SCRATCH_ROOT" >&2
  exit 2
}

REPO_ROOT="$(readlink -f "$REPO_ROOT")"
SCRATCH_ROOT="$(readlink -f "$FGG_SCRATCH_ROOT")"
[[ "$GRID_CONFIG" = /* ]] || GRID_CONFIG="$REPO_ROOT/$GRID_CONFIG"
GRID_CONFIG="$(readlink -f "$GRID_CONFIG")"
[[ -f "$GRID_CONFIG" ]] || { echo "Grid config not found: $GRID_CONFIG" >&2; exit 2; }

RUN_ROOT="$SCRATCH_ROOT/functional-gradient-growth/$RUN_NAME"
RUN_CONFIG="$RUN_ROOT/configs/$(basename "$GRID_CONFIG")"
RESULTS_DIR="$RUN_ROOT/results"
mkdir -p "$RUN_ROOT"/{configs,results,slurm_logs,metadata}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python - "$GRID_CONFIG" "$RUN_CONFIG" "$RESULTS_DIR" <<'PY'
from pathlib import Path
import sys

import yaml

source, destination, results_dir = map(Path, sys.argv[1:])
payload = yaml.safe_load(source.read_text(encoding="utf-8"))
payload["results_dir"] = str(results_dir)
rendered = yaml.safe_dump(payload, sort_keys=False)
if destination.exists() and destination.read_text(encoding="utf-8") != rendered:
    raise SystemExit(
        f"Refusing to change an existing run config: {destination}\n"
        "Choose a new RUN_NAME or restore the original source config."
    )
destination.write_text(rendered, encoding="utf-8")
print(f"Run config: {destination}")
PY

TRIAL_COUNT="$(python - "$RUN_CONFIG" <<'PY'
from pathlib import Path
import sys

from grid_search.run import enumerate_trials, load_grid

print(len(enumerate_trials(load_grid(Path(sys.argv[1])))))
PY
)"
[[ "$TRIAL_COUNT" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid trial count: $TRIAL_COUNT" >&2
  exit 2
}

export REPO_ROOT RUN_ROOT RUN_CONFIG CONDA_ENV
df -h "$SCRATCH_ROOT"
echo "repo_root=$REPO_ROOT"
echo "run_root=$RUN_ROOT"
echo "trials=$TRIAL_COUNT"
echo "array=0-$((TRIAL_COUNT - 1))%$MAX_CONCURRENT"

common=(--parsable --cpus-per-task=1 --mem="$SLURM_MEM" --export=ALL)
[[ -z "${SLURM_ACCOUNT:-}" ]] || common+=(--account="$SLURM_ACCOUNT")
[[ -z "${SLURM_PARTITION:-}" ]] || common+=(--partition="$SLURM_PARTITION")
[[ -z "${SLURM_QOS:-}" ]] || common+=(--qos="$SLURM_QOS")
extra=()
[[ -z "${SBATCH_EXTRA_ARGS:-}" ]] || read -r -a extra <<<"$SBATCH_EXTRA_ARGS"

array_spec="0-$((TRIAL_COUNT - 1))%$MAX_CONCURRENT"
array_script="$REPO_ROOT/cluster/slurm/fixed_architecture_grid_array.sbatch"
summary_script="$REPO_ROOT/cluster/slurm/fixed_architecture_grid_summarize.sbatch"

echo "Array: sbatch --array=$array_spec --time=$SLURM_TIME $array_script"
echo "Summary: submitted after successful completion of the array"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: config prepared; no jobs submitted."
  exit 0
fi
if [[ "${CONFIRM_SUBMIT:-0}" != "1" ]]; then
  read -r -p "Submit the fixed-architecture grid? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }
fi

array_job="$(sbatch "${common[@]}" "${extra[@]}" \
  --job-name=fixed-under600 \
  --array="$array_spec" --time="$SLURM_TIME" \
  --output="$RUN_ROOT/slurm_logs/grid_%A_%a.out" \
  --error="$RUN_ROOT/slurm_logs/grid_%A_%a.err" \
  "$array_script")"
summary_job="$(sbatch "${common[@]}" "${extra[@]}" \
  --job-name=fixed-summary \
  --dependency="afterok:$array_job" --time=00:10:00 \
  --output="$RUN_ROOT/slurm_logs/summary_%j.out" \
  --error="$RUN_ROOT/slurm_logs/summary_%j.err" \
  "$summary_script")"

printf 'ARRAY_JOB_ID=%q\nSUMMARY_JOB_ID=%q\n' "$array_job" "$summary_job" \
  >"$RUN_ROOT/metadata/slurm_job_ids.env"
echo "array_job=$array_job summary_job=$summary_job"
echo "Status: squeue -j $array_job,$summary_job"
echo "Results: $RESULTS_DIR"
