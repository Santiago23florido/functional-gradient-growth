#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/dev/functional-gradient-growth}"
FGG_SCRATCH_ROOT="${FGG_SCRATCH_ROOT:-$HOME/datasets}"
RUN_NAME="${RUN_NAME:-smaller_architectures_400_v1}"
MAX_CONCURRENT="${MAX_CONCURRENT:-30}"
STAGE1_NUM_SHARDS="${STAGE1_NUM_SHARDS:-1000}"
STAGE2_NUM_SHARDS="${STAGE2_NUM_SHARDS:-600}"
CONDA_ENV="${CONDA_ENV:-cct}"
SLURM_MEM="${SLURM_MEM:-2G}"
SLURM_TIME_STAGE1="${SLURM_TIME_STAGE1:-02:00:00}"
SLURM_TIME_STAGE2="${SLURM_TIME_STAGE2:-01:00:00}"
MIN_FREE_GB="${MIN_FREE_GB:-1}"

[[ -n "$RUN_NAME" ]] || { echo "RUN_NAME must not be empty" >&2; exit 2; }
[[ -e "$FGG_SCRATCH_ROOT" ]] || { echo "Scratch path does not exist: $FGG_SCRATCH_ROOT" >&2; exit 2; }
SCRATCH_ROOT="$(readlink -f "${FGG_SCRATCH_ROOT:-$HOME/datasets}")"
REPO_ROOT="$(readlink -f "$REPO_ROOT")"
RUN_ROOT="$SCRATCH_ROOT/functional-gradient-growth/$RUN_NAME"
export REPO_ROOT FGG_SCRATCH_ROOT SCRATCH_ROOT RUN_ROOT RUN_NAME MAX_CONCURRENT
export STAGE1_NUM_SHARDS STAGE2_NUM_SHARDS CONDA_ENV MIN_FREE_GB

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python -m grid_search.budget_search prepare-stage1 \
  --config grid_search/smaller_architectures_400_stage1.yaml \
  --repo-root "$REPO_ROOT" --run-root "$RUN_ROOT" \
  --min-free-gb "$MIN_FREE_GB" \
  --stage1-num-shards "$STAGE1_NUM_SHARDS" \
  --stage2-num-shards "$STAGE2_NUM_SHARDS" \
  --max-concurrent "$MAX_CONCURRENT"

if command -v scontrol >/dev/null 2>&1; then
  max_array_size="$(
    timeout "${SCONTROL_TIMEOUT_SECONDS:-10}" scontrol show config 2>/dev/null \
      | awk -F= '/MaxArraySize/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' \
      || true
  )"
  if [[ -n "$max_array_size" ]] && (( STAGE1_NUM_SHARDS > max_array_size || STAGE2_NUM_SHARDS > max_array_size )); then
    echo "Configured shards exceed Slurm MaxArraySize=$max_array_size." >&2
    echo "Lower STAGE1_NUM_SHARDS/STAGE2_NUM_SHARDS or ask the administrator." >&2
    exit 2
  elif [[ -z "$max_array_size" ]]; then
    echo "WARNING: could not read MaxArraySize within ${SCONTROL_TIMEOUT_SECONDS:-10}s; continuing." >&2
  fi
fi

common=(--parsable --cpus-per-task=1 --mem="$SLURM_MEM")
[[ -z "${SLURM_ACCOUNT:-}" ]] || common+=(--account="$SLURM_ACCOUNT")
[[ -z "${SLURM_PARTITION:-}" ]] || common+=(--partition="$SLURM_PARTITION")
[[ -z "${SLURM_QOS:-}" ]] || common+=(--qos="$SLURM_QOS")
extra=()
[[ -z "${SBATCH_EXTRA_ARGS:-}" ]] || read -r -a extra <<<"$SBATCH_EXTRA_ARGS"

stage1_array="0-$((STAGE1_NUM_SHARDS - 1))%$MAX_CONCURRENT"
stage2_array="0-$((STAGE2_NUM_SHARDS - 1))%$MAX_CONCURRENT"
stage1_script="$REPO_ROOT/cluster/slurm/budget_search_stage1.sbatch"
select_script="$REPO_ROOT/cluster/slurm/budget_search_select_stage1.sbatch"
stage2_script="$REPO_ROOT/cluster/slurm/budget_search_stage2.sbatch"
finalize_script="$REPO_ROOT/cluster/slurm/budget_search_finalize.sbatch"

echo "Stage1: sbatch --array=$stage1_array --time=$SLURM_TIME_STAGE1 $stage1_script"
echo "Select: sbatch --dependency=afterok:<stage1_job> $select_script"
echo "Stage2: sbatch --array=$stage2_array --time=$SLURM_TIME_STAGE2 --dependency=afterok:<select_job> $stage2_script"
echo "Finalize: sbatch --dependency=afterok:<stage2_job> $finalize_script"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: manifests verified; no jobs submitted."
  exit 0
fi
if [[ "${CONFIRM_SUBMIT:-0}" != "1" ]]; then
  read -r -p "Submit the full search chain? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }
fi

stage1_job="$(sbatch "${common[@]}" "${extra[@]}" --job-name=budget-s1 \
  --array="$stage1_array" --time="$SLURM_TIME_STAGE1" \
  --output="$RUN_ROOT/slurm_logs/stage1_%A_%a.out" \
  --error="$RUN_ROOT/slurm_logs/stage1_%A_%a.err" "$stage1_script")"
select_job="$(sbatch "${common[@]}" "${extra[@]}" --job-name=budget-select \
  --dependency="afterok:$stage1_job" --time=00:20:00 \
  --output="$RUN_ROOT/slurm_logs/select_%j.out" \
  --error="$RUN_ROOT/slurm_logs/select_%j.err" "$select_script")"
stage2_job="$(sbatch "${common[@]}" "${extra[@]}" --job-name=budget-s2 \
  --dependency="afterok:$select_job" --array="$stage2_array" \
  --time="$SLURM_TIME_STAGE2" \
  --output="$RUN_ROOT/slurm_logs/stage2_%A_%a.out" \
  --error="$RUN_ROOT/slurm_logs/stage2_%A_%a.err" "$stage2_script")"
finalize_job="$(sbatch "${common[@]}" "${extra[@]}" --job-name=budget-final \
  --dependency="afterok:$stage2_job" --time=00:20:00 \
  --output="$RUN_ROOT/slurm_logs/finalize_%j.out" \
  --error="$RUN_ROOT/slurm_logs/finalize_%j.err" "$finalize_script")"

printf 'STAGE1_JOB_ID=%q\nSELECT_JOB_ID=%q\nSTAGE2_JOB_ID=%q\nFINALIZE_JOB_ID=%q\n' \
  "$stage1_job" "$select_job" "$stage2_job" "$finalize_job" \
  >"$RUN_ROOT/metadata/slurm_job_ids.env"
echo "stage1=$stage1_job select=$select_job stage2=$stage2_job finalize=$finalize_job"
echo "Status: RUN_NAME=$RUN_NAME bash cluster/slurm/status_smaller_architecture_search.sh"
echo "Cancel: RUN_NAME=$RUN_NAME bash cluster/slurm/cancel_smaller_architecture_search.sh"
