#!/usr/bin/env bash
set -euo pipefail

FGG_SCRATCH_ROOT="${FGG_SCRATCH_ROOT:-$HOME/datasets}"
RUN_NAME="${RUN_NAME:-smaller_architectures_400_v1}"
SCRATCH_ROOT="$(readlink -f "${FGG_SCRATCH_ROOT:-$HOME/datasets}")"
RUN_ROOT="$SCRATCH_ROOT/functional-gradient-growth/$RUN_NAME"
ids="$RUN_ROOT/metadata/slurm_job_ids.env"
[[ -f "$ids" ]] || { echo "No job metadata at $ids" >&2; exit 2; }
# shellcheck disable=SC1090
source "$ids"
jobs=("${STAGE1_JOB_ID:-}" "${SELECT_JOB_ID:-}" "${STAGE2_JOB_ID:-}" "${FINALIZE_JOB_ID:-}")
for job in "${jobs[@]}"; do [[ -z "$job" ]] || scancel "$job"; done
echo "Cancellation requested. Results remain resumable at $RUN_ROOT"
