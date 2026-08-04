#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${STAGE1_NUM_SHARDS:?STAGE1_NUM_SHARDS is required}"

# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-cct}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

retry_args=()
[[ "${RETRY_FAILED:-0}" != "1" ]] || retry_args+=(--retry-failed)
python -m grid_search.budget_search run-shard \
  --config grid_search/smaller_architectures_400_stage1.yaml \
  --repo-root "$REPO_ROOT" --run-root "$RUN_ROOT" \
  --stage stage1 --shard-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$STAGE1_NUM_SHARDS" "${retry_args[@]}"
