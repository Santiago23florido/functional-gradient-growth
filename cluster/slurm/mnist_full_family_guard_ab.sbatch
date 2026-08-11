#!/usr/bin/env bash
# MNIST COMPLETO, A/B del guardian del paso de familia. Dos brazos:
#
#   0  guard=on   transactional_family_step: true   (el default)
#   1  guard=off  transactional_family_step: false  (el comportamiento medido
#                 como defectuoso: x3.97 / x6.21 / x15.44 del funcional real)
#
# Los dos brazos son IDENTICOS en todo lo demas -- misma semilla, mismos datos,
# mismo tope, mismo lookahead, mismo K-fold -- asi que cualquier diferencia es
# causal. El guardian del TANGENTE queda encendido en los dos: apagarlo para
# probar este compararia otra cosa.
#
# NO WALLTIME. --time=0 pide el maximo de la particion; si el MaxTime es finito
# sbatch lo rechaza y hay que nombrarlo (`sinfo -p gpu -o "%P %l"`).
#
# NODO Y PARTICION TIENEN QUE CASAR o sbatch responde "Requested nodes not in
# this partition" antes de ejecutar nada:
#   scontrol show node margpu010 | grep -i -E "NodeName|Partitions|Gres"
# Los dos se pisan desde la linea de comandos sin tocar este fichero.
#
#SBATCH --job-name=mnist-full-ab
#SBATCH --partition=gpu
#SBATCH --nodelist=margpu010
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=0
#SBATCH --array=0-1
#SBATCH --output=slurm_logs/mnist_full_ab_%A_%a.out
#SBATCH --error=slurm_logs/mnist_full_ab_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/dev/functional-gradient-growth}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets}"
CONDA_ENV="${CONDA_ENV:-cct}"
WANDB_GROUP="${WANDB_GROUP:-mnist-full-family-guard-ab}"
MODEL_SEED="${MODEL_SEED:-0}"
# Optional metadata-only suffix for reruns. Empty preserves every historical
# run name and results path; e.g. RUN_SUFFIX=probe-refine-r2 isolates a rerun.
RUN_SUFFIX="${RUN_SUFFIX:-}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) GUARD=true;  ARM=guard-on ;;
  1) GUARD=false; ARM=guard-off ;;
  *) echo "[FATAL] array index fuera de 0-1"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Datos en el disco del nodo. $HOME es almacenamiento compartido y leer 50000
# imagenes por epoch desde ahi es lo que vuelve la corrida IO-bound.
#
# El punto de montaje NO es el mismo en cada nodo GPU: margpu002/003 exponen
# /local, margpu008 y margpu011 exponen /local y /local2, y margpu009/010/012
# exponen SOLO /local/data y /local/tmp. Sondear en orden de preferencia.
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
  echo "[FATAL] ningun directorio local escribible en $(hostname)"; exit 1
fi
echo "[DATA] raiz local: ${LOCAL_ROOT}"

if [[ ! -d "${LOCAL_ROOT}/MNIST/raw" ]]; then
  echo "[DATA] copiando MNIST desde ${DATASET_ROOT}/MNIST"
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
# print() va a un fichero, no a un tty, asi que python lo bufferiza por bloques
# y el log parece congelado durante minutos. Sin esto no se puede vigilar nada.
export PYTHONUNBUFFERED=1

CONFIG="configs/experiments/mnist_full.yaml"
RUN_NAME="mnist-full-${ARM}-seed-${MODEL_SEED}"
[[ -n "${RUN_SUFFIX}" ]] && RUN_NAME="${RUN_NAME}-${RUN_SUFFIX}"

# data_dir y el brazo del A/B se parchean en una COPIA node-local, de modo que
# el fichero versionado sigue siendo exactamente lo que dice el repo.
RUN_CONFIG="${LOCAL_ROOT}/mnist_full_${ARM}.run.yaml"
sed -e "s#^  data_dir:.*#  data_dir: ${LOCAL_ROOT}#" \
    -e "s#^  transactional_realized_descent:.*#  transactional_realized_descent: true\n  transactional_family_step: ${GUARD}#" \
    "${CONFIG}" > "${RUN_CONFIG}"
grep -q "^  data_dir: ${LOCAL_ROOT}$" "${RUN_CONFIG}" \
  || { echo "[FATAL] data_dir no se parcheo"; exit 1; }
grep -q "^  transactional_family_step: ${GUARD}$" "${RUN_CONFIG}" \
  || { echo "[FATAL] transactional_family_step no se parcheo"; exit 1; }

RESULTS_DIR="${LOCAL_ROOT}/results/${RUN_NAME}"
mkdir -p "${RESULTS_DIR}"

echo "[RUN] node=$(hostname) arm=${ARM} family_guard=${GUARD}"
echo "[RUN] config=${RUN_CONFIG} seed=${MODEL_SEED} wandb=${WANDB_RUN_MODE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun python -m stable_tiny \
  --config "${RUN_CONFIG}" \
  --model-seed "${MODEL_SEED}" \
  --run-name "${RUN_NAME}" \
  --results-dir "${RESULTS_DIR}" \
  --wandb --wandb-mode "${WANDB_RUN_MODE}" --wandb-group "${WANDB_GROUP}" \
  --no-plot

mkdir -p "${REPO_ROOT}/results/mnist_full_ab"
cp -r "${RESULTS_DIR}" "${REPO_ROOT}/results/mnist_full_ab/" || true
echo "[RUN] done: ${RUN_NAME}"
