#!/usr/bin/env bash
# MNIST depth/width sweep on Margaret: six runs as one SLURM array.
#
#   0 depth2_small   2 hidden layers, start 2-2
#   1 depth2_large   2 hidden layers, start 16-16
#   2 depth3_small   3 hidden layers, start 2-2-2
#   3 depth3_large   3 hidden layers, start 16-16-16
#   4 depth4_small   4 hidden layers, start 2-2-2-2
#   5 depth4_large   4 hidden layers, start 16-16-16-16
#
# NO WALLTIME. --time=0 asks for the partition maximum; growth is meant to end
# on the PARAMETER BUDGET, never on the clock. Confirm the partition allows it:
#   sinfo -p gpu -o "%P %l"
# If MaxTime is finite, --time=0 is refused and you must name that value.
#
#SBATCH --job-name=mnist-depth
#SBATCH --partition=gpu
#SBATCH --nodelist=margpu002
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=0
#SBATCH --array=0-5
#SBATCH --output=slurm_logs/mnist_depth_%A_%a.out
#SBATCH --error=slurm_logs/mnist_depth_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/dev/functional-gradient-growth}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets}"
CONDA_ENV="${CONDA_ENV:-base}"
WANDB_GROUP="${WANDB_GROUP:-mnist-depth-sweep}"
MODEL_SEED="${MODEL_SEED:-0}"
# 5 turns on the budget-free stopping criterion; 0 (default) leaves the
# parameter ceiling as the only thing that ends growth, which is what this
# sweep is asking about. See docs/budget_free_stopping.md.
CROSSFOLD="${CROSSFOLD:-0}"

VARIANTS=(depth2_small depth2_large depth3_small depth3_large depth4_small depth4_large)
VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"

# ---------------------------------------------------------------------------
# Node-local data. /local on margpu002 is 11T of RAID; $HOME is shared storage
# and reading 24,000 images off it every epoch is what makes these runs
# IO-bound. Copy once, read from the node.
# ---------------------------------------------------------------------------
LOCAL_ROOT="/local/${USER}/fgg"
mkdir -p "${LOCAL_ROOT}"
if [[ ! -d "${LOCAL_ROOT}/MNIST/raw" ]]; then
  echo "[DATA] staging MNIST to ${LOCAL_ROOT}"
  cp -r "${DATASET_ROOT}/MNIST" "${LOCAL_ROOT}/"
fi
ls "${LOCAL_ROOT}/MNIST/raw" >/dev/null

# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# W&B artefacts on the node, not on shared storage.
export WANDB_DIR="${LOCAL_ROOT}/wandb"
export WANDB_CACHE_DIR="${LOCAL_ROOT}/wandb-cache"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"
# If the compute nodes have no outbound network, set WANDB_MODE=offline before
# submitting and run `wandb sync` from the login node afterwards.
WANDB_RUN_MODE="${WANDB_MODE:-online}"

export FGD_PROFILE=1

CONFIG="configs/experiments/mnist_${VARIANT}.yaml"
RUN_NAME="mnist-${VARIANT//_/-}-seed-${MODEL_SEED}"

# There is no --data-dir on the CLI, so data_dir is patched into a node-local
# COPY of the config. The tracked file therefore stays exactly what the repo
# says it is, and the run still reads MNIST off /local.
RUN_CONFIG="${LOCAL_ROOT}/${VARIANT}.run.yaml"
sed "s#^  data_dir:.*#  data_dir: ${LOCAL_ROOT}#" "${CONFIG}" > "${RUN_CONFIG}"
grep -q "^  data_dir: ${LOCAL_ROOT}$" "${RUN_CONFIG}" \
  || { echo "[FATAL] data_dir not patched into ${RUN_CONFIG}"; exit 1; }

if [[ "${CROSSFOLD}" != "0" ]]; then
  sed -i "s/^  growth_where: expressivity_bottleneck$/  growth_where: expressivity_bottleneck\n  growth_bottleneck_crossfold_folds: ${CROSSFOLD}/" \
    "${RUN_CONFIG}"
  RUN_NAME="${RUN_NAME}-k${CROSSFOLD}"
  WANDB_GROUP="${WANDB_GROUP}-k${CROSSFOLD}"
fi

RESULTS_DIR="${LOCAL_ROOT}/results/${RUN_NAME}"
mkdir -p "${RESULTS_DIR}"

echo "[RUN] node=$(hostname) variant=${VARIANT} config=${RUN_CONFIG}"
echo "[RUN] crossfold=${CROSSFOLD} seed=${MODEL_SEED} wandb=${WANDB_RUN_MODE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun python -m stable_tiny \
  --config "${RUN_CONFIG}" \
  --model-seed "${MODEL_SEED}" \
  --run-name "${RUN_NAME}" \
  --results-dir "${RESULTS_DIR}" \
  --wandb --wandb-mode "${WANDB_RUN_MODE}" --wandb-group "${WANDB_GROUP}" \
  --no-plot

# Bring the history JSON back to shared storage; the node-local copy is not
# guaranteed to survive.
mkdir -p "${REPO_ROOT}/results/mnist_depth_sweep"
cp -r "${RESULTS_DIR}" "${REPO_ROOT}/results/mnist_depth_sweep/" || true
echo "[RUN] done: ${RUN_NAME}"
