#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-cct}"
cd "${REPO_ROOT:?REPO_ROOT is required}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python -m grid_search.budget_search finalize \
  --config grid_search/smaller_architectures_400_stage1.yaml \
  --repo-root "$REPO_ROOT" --run-root "${RUN_ROOT:?RUN_ROOT is required}"
