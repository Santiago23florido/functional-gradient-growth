#!/usr/bin/env bash
# MNIST COMPLETO con la sonda arreglada. UN SOLO BRAZO.
#
# El A/B de dos brazos que este fichero tenia ya se corrio y esta cerrado:
#
#   probe-fixed   (unguzxkq)  acc 0.7741 en 8 epochs, 1.58 h
#   probe-biased  (euepsdrk)  acc 0.4337 en 6 epochs, 2.31 h
#
# El brazo sesgado NO se relanza, por dos razones medidas. Su comportamiento
# esta fijado tres veces de forma identica -- yqfvy8l7, uesf9xgb y euepsdrk
# mueren en la MISMA epoch 6, mismo acc 0.4337, mismo P=9443, mismas 2.31 h --
# y muere de OOM de GPU: la telemetria de W&B lo lleva de 1.3 a 6.6 a 12.8 y a
# 40.1 GB, el 93.4% de una A100 40G. Gastar una A100 2.3 h en reproducir un
# OOM conocido no compra nada.
#
# Lo que este brazo mide ahora es lo siguiente que fallo, en el propio brazo
# arreglado: la sonda dejo de crecer cuando P crecio. MEDIDO en unguzxkq,
#
#   P=1612   NK=3840  NK/P=2.38  eps=0.441
#   P=9383   NK=5760  NK/P=0.61  eps=0.0131
#   P=12869  NK=5760  NK/P=0.45  eps=0.0410
#
# eps se desploma al cruzar NK/P el 1 mientras el rel_err de validacion sigue
# en 1.02. certify_probe_parameter_floor y matrix_free_block_chunk, ambos en la
# config, son la respuesta; este lanzador ya no parchea nada del A/B.
#
# NO WALLTIME. --time=0 pide el maximo de la particion; si el MaxTime es finito
# sbatch lo rechaza y hay que nombrarlo (`sinfo -p tau -o "%P %l"`).
#
# NODO Y PARTICION TIENEN QUE CASAR o sbatch responde "Requested nodes not in
# this partition" antes de ejecutar nada:
#   scontrol show node margpu008 | grep -i -E "NodeName|Partitions|Gres"
# Los dos se pisan desde la linea de comandos sin tocar este fichero.
#
#SBATCH --job-name=mnist-full-probe
#SBATCH --partition=tau
#SBATCH --nodelist=margpu008
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=0
#SBATCH --output=slurm_logs/mnist_full_probe_%j.out
#SBATCH --error=slurm_logs/mnist_full_probe_%j.err

set -euo pipefail

# La muerte anterior del brazo arreglado no se pudo atribuir: GPU al 24.7% de
# 40 GB y RSS del host en 2.4 GB de 480, ninguna de las dos agotada, y sin traza
# en el log. Sin el estado de salida de SLURM no hay forma de saberlo, asi que
# se registra pase lo que pase.
finish() {
  local status=$?
  echo "[EXIT] status=${status} job=${SLURM_JOB_ID:-none}"
  sacct -j "${SLURM_JOB_ID:-0}" \
    --format=JobID,JobName%22,State,ExitCode,DerivedExitCode,MaxRSS,MaxVMSize,Elapsed \
    2>/dev/null || echo "[EXIT] sacct no disponible"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
}
trap finish EXIT

REPO_ROOT="${REPO_ROOT:-$HOME/dev/functional-gradient-growth}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets}"
CONDA_ENV="${CONDA_ENV:-cct}"
WANDB_GROUP="${WANDB_GROUP:-mnist-full-probe-floor}"
MODEL_SEED="${MODEL_SEED:-0}"
# Optional metadata-only suffix for reruns. Empty preserves every historical
# run name and results path; e.g. RUN_SUFFIX=r2 isolates a rerun.
RUN_SUFFIX="${RUN_SUFFIX:-}"
ARM="probe-floor"

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

# Solo data_dir se parchea, en una COPIA node-local: ya no hay brazos que
# distinguir, asi que todo lo demas es literalmente lo que dice el repo.
RUN_CONFIG="${LOCAL_ROOT}/mnist_full_${ARM}.run.yaml"
sed -e "s#^  data_dir:.*#  data_dir: ${LOCAL_ROOT}#" "${CONFIG}" > "${RUN_CONFIG}"
grep -q "^  data_dir: ${LOCAL_ROOT}$" "${RUN_CONFIG}" \
  || { echo "[FATAL] data_dir no se parcheo"; exit 1; }

# Los tres ajustes que esta corrida existe para probar, verificados por VALOR y
# no solo por presencia: un lanzamiento que corra la config equivocada
# desperdicia dias de A100 y no se nota hasta leer los logs.
for pair in \
  "certify_probe_base_resample:true" \
  "certify_probe_parameter_floor:1.25" \
  "matrix_free_block_chunk:256"
do
  key="${pair%%:*}"; want="${pair##*:}"
  grep -q "^  ${key}: ${want}$" "${RUN_CONFIG}" \
    || { echo "[FATAL] ${key} no es ${want} en ${RUN_CONFIG}"; exit 1; }
  echo "[CONFIG] ${key} = ${want}"
done

RESULTS_DIR="${LOCAL_ROOT}/results/${RUN_NAME}"
mkdir -p "${RESULTS_DIR}"

echo "[RUN] node=$(hostname) arm=${ARM}"
echo "[RUN] config=${RUN_CONFIG} seed=${MODEL_SEED} wandb=${WANDB_RUN_MODE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun python -m stable_tiny \
  --config "${RUN_CONFIG}" \
  --model-seed "${MODEL_SEED}" \
  --run-name "${RUN_NAME}" \
  --results-dir "${RESULTS_DIR}" \
  --wandb --wandb-mode "${WANDB_RUN_MODE}" --wandb-group "${WANDB_GROUP}" \
  --no-plot

mkdir -p "${REPO_ROOT}/results/mnist_full_probe"
cp -r "${RESULTS_DIR}" "${REPO_ROOT}/results/mnist_full_probe/" || true
echo "[RUN] done: ${RUN_NAME}"
