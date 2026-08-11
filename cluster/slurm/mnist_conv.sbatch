#!/usr/bin/env bash
# The conv arm of the MNIST matrix-free comparison, on Margaret.
#
# configs/fgd/mnist_conv_matrix_free.yaml is configs/fgd/mnist_matrix_free.yaml
# with a convolutional model and nothing else changed that conv does not force,
# so a run of this against a run of that IS the comparison: same data, same
# probe, same lookahead, same budget, same certified core.
#
# Array is over MODEL SEEDS, because the linear ladder's seed spread at these
# sizes is documented at 11 points and a single seed decides nothing.
#
#   sbatch cluster/slurm/mnist_conv.sbatch                  # seeds 0-3, balance off
#   BALANCE=true sbatch cluster/slurm/mnist_conv.sbatch     # the width-vs-depth arm
#
# NO WALLTIME. --time=0 asks for the partition maximum; growth is meant to end
# on the parameter budget, never on the clock. Confirm the partition allows it:
#   sinfo -p gpu -o "%P %l"
#
# NODE AND PARTITION MUST MATCH or sbatch refuses before anything runs:
#   scontrol show node margpu010 | grep -i -E "NodeName|Partitions|Gres"
#   sbatch --partition=<P> --nodelist=<N> cluster/slurm/mnist_conv.sbatch
#
#SBATCH --job-name=mnist-conv
#SBATCH --partition=gpu
#SBATCH --nodelist=margpu010
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=0
#SBATCH --array=0-3
#SBATCH --output=slurm_logs/mnist_conv_%A_%a.out
#SBATCH --error=slurm_logs/mnist_conv_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/dev/functional-gradient-growth}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets}"
CONDA_ENV="${CONDA_ENV:-cct}"
WANDB_GROUP="${WANDB_GROUP:-mnist-conv}"
MODEL_SEED="${SLURM_ARRAY_TASK_ID:-0}"

# The width-vs-depth balance criterion. false is the honest incumbent (width
# only, exactly the branch the linear flow runs); true lets a NEW LAYER compete
# with the widenings in the same measured bottleneck per parameter. It is the
# only way the search can ever touch the widths GroMo's topology pins -- the
# last conv before each shape change -- so the two arms are not a tuning
# sweep, they are a question about whether that relief pays.
BALANCE="${BALANCE:-false}"

# THE TWO CHUNKS ARE A LAPTOP CONSTRAINT, NOT A METHOD CONSTRAINT, and this is
# the one place worth overriding on a real card.
#
# Both bound how many directions ride through one vmap, and vmap holds every
# direction's full forward activations at once. On an MLP an activation is
# (N, width); on a conv it is (N, C, H, W), larger by the spatial extent, so
# the values config 2 uses (jacobian_row_chunk 256, unchunked blocks) exhaust
# an 8 GB card outright. The tracked config carries 32/32, measured on that
# 8 GB card.
#
# margpu010 is 2x A100 40G. Five times the memory means five times the chunk,
# and chunking costs TIME, measured: the validation certificate is 4.5 s at
# row chunk 8 and 1.0 s at 32, same answer. So raise both here. Chunking
# changes the result by one ulp (reduction order), never more -- see
# tests/test_matrix_free_block_chunk.py -- so this is a speed knob, not a
# fidelity one.
ROW_CHUNK="${ROW_CHUNK:-256}"
BLOCK_CHUNK="${BLOCK_CHUNK:-256}"

# ---------------------------------------------------------------------------
# Node-local data. $HOME is shared storage; the mount point is not the same on
# every GPU node, so probe in preference order and say which one was taken.
# ---------------------------------------------------------------------------
LOCAL_ROOT=""
for candidate in /local/data /local2 /local /local/tmp "${TMPDIR:-/tmp}"; do
  [[ -d "${candidate}" ]] || continue
  if mkdir -p "${candidate}/${USER}/fgg" 2>/dev/null; then
    LOCAL_ROOT="${candidate}/${USER}/fgg"
    break
  fi
done
if [[ -z "${LOCAL_ROOT}" ]]; then
  echo "[FATAL] no writable node-local directory found on $(hostname)"; exit 1
fi
echo "[DATA] node-local root: ${LOCAL_ROOT}"

if [[ ! -d "${LOCAL_ROOT}/MNIST/raw" ]]; then
  echo "[DATA] staging MNIST from ${DATASET_ROOT}/MNIST"
  cp -r "${DATASET_ROOT}/MNIST" "${LOCAL_ROOT}/"
fi
ls "${LOCAL_ROOT}/MNIST/raw" >/dev/null

# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

export WANDB_DIR="${LOCAL_ROOT}/wandb"
export WANDB_CACHE_DIR="${LOCAL_ROOT}/wandb-cache"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"
WANDB_RUN_MODE="${WANDB_MODE:-online}"

export FGD_PROFILE=1

CONFIG="configs/fgd/mnist_conv_matrix_free.yaml"
RUN_NAME="mnist-conv-seed-${MODEL_SEED}"
[[ "${BALANCE}" == "true" ]] && RUN_NAME="${RUN_NAME}-balance"

# There is no --data-dir on the CLI, so data_dir and the two chunks are patched
# into a node-local COPY. The tracked file stays exactly what the repo says.
RUN_CONFIG="${LOCAL_ROOT}/conv_seed${MODEL_SEED}_${BALANCE}.run.yaml"
sed -e "s#^  data_dir:.*#  data_dir: ${LOCAL_ROOT}#" \
    -e "s#^  jacobian_row_chunk:.*#  jacobian_row_chunk: ${ROW_CHUNK}#" \
    -e "s#^  matrix_free_block_chunk:.*#  matrix_free_block_chunk: ${BLOCK_CHUNK}#" \
    -e "s#^  growth_where_balance:.*#  growth_where_balance: ${BALANCE}#" \
    "${CONFIG}" > "${RUN_CONFIG}"

for key in data_dir jacobian_row_chunk matrix_free_block_chunk growth_where_balance; do
  grep -q "^  ${key}: " "${RUN_CONFIG}" \
    || { echo "[FATAL] ${key} not patched into ${RUN_CONFIG}"; exit 1; }
done
grep -q "^  data_dir: ${LOCAL_ROOT}$" "${RUN_CONFIG}" \
  || { echo "[FATAL] data_dir not patched into ${RUN_CONFIG}"; exit 1; }

RESULTS_DIR="${LOCAL_ROOT}/results/${RUN_NAME}"
mkdir -p "${RESULTS_DIR}"

echo "[RUN] node=$(hostname) seed=${MODEL_SEED} balance=${BALANCE}"
echo "[RUN] row_chunk=${ROW_CHUNK} block_chunk=${BLOCK_CHUNK} wandb=${WANDB_RUN_MODE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun python -m stable_tiny \
  --config "${RUN_CONFIG}" \
  --model-seed "${MODEL_SEED}" \
  --run-name "${RUN_NAME}" \
  --results-dir "${RESULTS_DIR}" \
  --wandb --wandb-mode "${WANDB_RUN_MODE}" --wandb-group "${WANDB_GROUP}" \
  --no-plot

mkdir -p "${REPO_ROOT}/results/mnist_conv"
cp -r "${RESULTS_DIR}" "${REPO_ROOT}/results/mnist_conv/" || true
echo "[RUN] done: ${RUN_NAME}"
