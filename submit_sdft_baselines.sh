#!/bin/bash
# Submit FM-SDFT + ER-only training jobs for all 4 LIBERO suites.
#
# Usage:
#   bash submit_sdft_baselines.sh
#   BENCHMARKS="object spatial" bash submit_sdft_baselines.sh

set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${BASE}/logs"
mkdir -p "${LOG_DIR}"

BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"
PARTITION="${PARTITION:-gigabyte_a6000}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
SIF_IMAGE="${SIF_IMAGE:-/scratch2/yerincho04/sdft_fm.sif}"
DATA_DIR="${DATA_DIR:-/home/yerincho04/ing-chicken/cl_diffusion_ER/data}"
BENCHMARKS="${BENCHMARKS:-object spatial goal long}"

echo "Submitting FM-SDFT + ER-only baselines  [tag=${BATCH_TAG}]"
echo "  Partition : ${PARTITION}"
echo "  SIF       : ${SIF_IMAGE}"
echo "  Data      : ${DATA_DIR}"
echo "  Suites    : ${BENCHMARKS}"
echo "  wandb     : sdft-step"
echo ""

for BENCH in ${BENCHMARKS}; do
  CONFIG="${BASE}/configs/cl_${BENCH}_sdft.yaml"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found, skipping: ${CONFIG}" >&2
    continue
  fi

  LOG_PREFIX="${LOG_DIR}/sdft_${BENCH}_${BATCH_TAG}"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fm_sdft_${BENCH}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --output=${LOG_PREFIX}_%j.out
#SBATCH --error=${LOG_PREFIX}_%j.err

set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

exec singularity exec --nv --writable-tmpfs \\
  --bind "${BASE}:/workspace" \\
  --bind "${DATA_DIR}:${DATA_DIR}" \\
  "${SIF_IMAGE}" \\
  bash -lc '
    set -euo pipefail
    cd /workspace
    export DP_IMAGE_PYTHON_ROOT=/usr/local
    source /workspace/scripts/singularity/dp_image_env.sh
    export PYTHONUNBUFFERED=1
    export WANDB_API_KEY=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
    export HF_HUB_DISABLE_TELEMETRY=1
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    export MUJOCO_GL=osmesa
    export PYOPENGL_PLATFORM=osmesa
    python -m scripts.train_sequential_sdft \\
      --config /workspace/configs/cl_${BENCH}_sdft.yaml \\
      --skip-eval
  '
EOF

  echo "  Submitted: ${BENCH}  (logs: ${LOG_PREFIX}_<jobid>.out)"
done

echo ""
echo "Monitor with:  squeue -u ${USER}"
