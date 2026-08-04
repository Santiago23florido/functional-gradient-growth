#!/usr/bin/env bash
set -euo pipefail

FGG_SCRATCH_ROOT="${FGG_SCRATCH_ROOT:-$HOME/datasets}"
RUN_NAME="${RUN_NAME:-smaller_architectures_400_v1}"
SCRATCH_ROOT="$(readlink -f "${FGG_SCRATCH_ROOT:-$HOME/datasets}")"
RUN_ROOT="$SCRATCH_ROOT/functional-gradient-growth/$RUN_NAME"
ids="$RUN_ROOT/metadata/slurm_job_ids.env"
# shellcheck disable=SC1090
[[ -f "$ids" ]] && source "$ids"
job_list="${STAGE1_JOB_ID:-},${SELECT_JOB_ID:-},${STAGE2_JOB_ID:-},${FINALIZE_JOB_ID:-}"
job_list="${job_list#,}"; job_list="${job_list%,}"; job_list="${job_list//,,/,}"
[[ -z "$job_list" ]] || squeue -j "$job_list" -o "%.18i %.24j %.10T %.10M %.30R"
for stage in stage1 stage2; do
  shards="$RUN_ROOT/shards/$stage"
  count=0
  [[ ! -d "$shards" ]] || count="$(find "$shards" -maxdepth 1 -name 'shard_*.jsonl' -type f | wc -l)"
  echo "$stage shard files: $count"
done
echo "RUN_ROOT=$RUN_ROOT"
