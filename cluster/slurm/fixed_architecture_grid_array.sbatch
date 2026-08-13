#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RUN_CONFIG:?RUN_CONFIG is required}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"

# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-cct}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m grid_search.run run \
  --config "$RUN_CONFIG" \
  --trial-index "$SLURM_ARRAY_TASK_ID"
