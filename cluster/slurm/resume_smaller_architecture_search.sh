#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/dev/functional-gradient-growth}"
FGG_SCRATCH_ROOT="${FGG_SCRATCH_ROOT:-$HOME/datasets}"
RUN_NAME="${RUN_NAME:-smaller_architectures_400_v1}"
RESUME_STAGE="${RESUME_STAGE:-stage1}"
MAX_CONCURRENT="${MAX_CONCURRENT:-30}"
STAGE1_NUM_SHARDS="${STAGE1_NUM_SHARDS:-1000}"
STAGE2_NUM_SHARDS="${STAGE2_NUM_SHARDS:-600}"
CONDA_ENV="${CONDA_ENV:-cct}"
SLURM_MEM="${SLURM_MEM:-2G}"
SCRATCH_ROOT="$(readlink -f "${FGG_SCRATCH_ROOT:-$HOME/datasets}")"
REPO_ROOT="$(readlink -f "$REPO_ROOT")"
RUN_ROOT="$SCRATCH_ROOT/functional-gradient-growth/$RUN_NAME"
export REPO_ROOT FGG_SCRATCH_ROOT SCRATCH_ROOT RUN_ROOT CONDA_ENV
export STAGE1_NUM_SHARDS STAGE2_NUM_SHARDS RETRY_FAILED=1
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [[ "$RESUME_STAGE" == "stage1" ]]; then
  num_shards="$STAGE1_NUM_SHARDS"; time_limit="${SLURM_TIME_STAGE1:-02:00:00}"
  script="$REPO_ROOT/cluster/slurm/budget_search_stage1.sbatch"
elif [[ "$RESUME_STAGE" == "stage2" ]]; then
  num_shards="$STAGE2_NUM_SHARDS"; time_limit="${SLURM_TIME_STAGE2:-01:00:00}"
  script="$REPO_ROOT/cluster/slurm/budget_search_stage2.sbatch"
else
  echo "RESUME_STAGE must be stage1 or stage2" >&2; exit 2
fi
indices="$(python -m grid_search.budget_search incomplete-shards \
  --config grid_search/smaller_architectures_400_stage1.yaml \
  --repo-root "$REPO_ROOT" --run-root "$RUN_ROOT" \
  --stage "$RESUME_STAGE" --num-shards "$num_shards")"
[[ -n "$indices" ]] || { echo "$RESUME_STAGE is already complete"; exit 0; }
echo "Resubmitting only $RESUME_STAGE shards: $indices"
common=(--cpus-per-task=1 --mem="$SLURM_MEM" --time="$time_limit")
[[ -z "${SLURM_ACCOUNT:-}" ]] || common+=(--account="$SLURM_ACCOUNT")
[[ -z "${SLURM_PARTITION:-}" ]] || common+=(--partition="$SLURM_PARTITION")
[[ -z "${SLURM_QOS:-}" ]] || common+=(--qos="$SLURM_QOS")
extra=()
[[ -z "${SBATCH_EXTRA_ARGS:-}" ]] || read -r -a extra <<<"$SBATCH_EXTRA_ARGS"
sbatch "${common[@]}" "${extra[@]}" \
  --array="${indices}%${MAX_CONCURRENT}" \
  --output="$RUN_ROOT/slurm_logs/${RESUME_STAGE}_resume_%A_%a.out" \
  --error="$RUN_ROOT/slurm_logs/${RESUME_STAGE}_resume_%A_%a.err" "$script"
