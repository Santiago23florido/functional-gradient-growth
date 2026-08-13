#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT is required}"
: "${RUN_CONFIG:?RUN_CONFIG is required}"

# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-cct}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
python -m grid_search.run summarize --config "$RUN_CONFIG"
