#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/fgd/nonlinear_family_ladder_N1024.yaml"
GROUP="nonlinear-n1024-2x2x2-3seeds"
OUTPUT_ROOT="${REPO_ROOT}/results/nonlinear_n1024_3seeds"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

for seed in 0 1 2; do
  run_name="nonlinear-n1024-2x2x2-model-seed-${seed}"
  output_dir="${OUTPUT_ROOT}/seed_${seed}"
  log_file="${output_dir}/run.log"
  mkdir -p "${output_dir}"

  echo "[RUN] model_seed=${seed} train_seed=0 architecture=2-2-2"
  echo "[RUN] output_dir=${output_dir}"

  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m stable_tiny \
      --config "${CONFIG}" \
      --model-seed "${seed}" \
      --run-name "${run_name}" \
      --results-dir "${output_dir}" \
      --wandb \
      --wandb-group "${GROUP}" \
      --no-plot \
      2>&1 | tee "${log_file}"
done
