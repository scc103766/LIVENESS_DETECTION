#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/supercloud/llm-code/scc/scc/Liveness_Detection}"
CONDA_ENV="${CONDA_ENV:-anti-spoofing_scc_175}"
PYTHON_BIN="${PYTHON_BIN:-/home/scc/anaconda3/envs/${CONDA_ENV}/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/dataset/flash_liveness_asset_archive_fixed_collect_protocol}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/flash_liveness_runs/flash_liveness_v3_fixed_protocol_gpu012}"

EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
GPU_FREE_MB="$(
  nvidia-smi --id="${GPU_IDS%%,*}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '
)"
GPU_FREE_MB="${GPU_FREE_MB:-0}"
if [[ -z "${BATCH_SIZE:-}" ]]; then
  if (( GPU_FREE_MB >= 32000 )); then
    BATCH_SIZE=4
  elif (( GPU_FREE_MB >= 24000 )); then
    BATCH_SIZE=3
  elif (( GPU_FREE_MB >= 16000 )); then
    BATCH_SIZE=2
  else
    BATCH_SIZE=1
  fi
fi
MAX_TRAIN_FRAMES="${MAX_TRAIN_FRAMES:-256}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-256}"
LOG_INTERVAL="${LOG_INTERVAL:-0}"
LOG_SECONDS="${LOG_SECONDS:-60}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-SILENT}"
OPENCV_FFMPEG_LOGLEVEL="${OPENCV_FFMPEG_LOGLEVEL:--8}"
# CUDA_VISIBLE_DEVICES remaps physical GPU_IDS to process-local cuda:0,cuda:1...
# This project should only use physical GPUs 0, 1, and 2. Do not use 3, 4, 5, or 6.
DEVICE="${DEVICE:-cuda:0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
AUTO_RESUME="${AUTO_RESUME:-true}"

mkdir -p "${OUTPUT_DIR}"

if [[ -z "${RESUME_CHECKPOINT}" && "${AUTO_RESUME}" == "true" && -f "${OUTPUT_DIR}/last_flash_liveness_model.pth" ]]; then
  RESUME_CHECKPOINT="${OUTPUT_DIR}/last_flash_liveness_model.pth"
fi

CMD=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/flash_liveness_project_v3.py"
  train
  --data-root "${DATA_ROOT}"
  --dataset-media videos
  --require-color-txt
  --output-dir "${OUTPUT_DIR}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --pin-memory
  --persistent-workers
  --amp
  --cudnn-benchmark
  --max-train-frames "${MAX_TRAIN_FRAMES}"
  --max-eval-frames "${MAX_EVAL_FRAMES}"
  --log-interval "${LOG_INTERVAL}"
  --log-seconds "${LOG_SECONDS}"
  --device "${DEVICE}"
  --multi-gpu
)

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  CMD+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi

printf 'Project root: %s\n' "${PROJECT_ROOT}"
printf 'Physical GPU ids: %s\n' "${GPU_IDS}"
printf 'Process device: %s\n' "${DEVICE}"
printf 'Detected free memory on first GPU: %s MiB\n' "${GPU_FREE_MB}"
printf 'Batch size: %s\n' "${BATCH_SIZE}"
printf 'Max frames train/eval: %s/%s\n' "${MAX_TRAIN_FRAMES}" "${MAX_EVAL_FRAMES}"
printf 'Batch log cadence: interval=%s, seconds=%s\n' "${LOG_INTERVAL}" "${LOG_SECONDS}"
printf 'Output dir: %s\n' "${OUTPUT_DIR}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  printf 'Resume checkpoint: %s\n' "${RESUME_CHECKPOINT}"
fi
printf 'Launching V3 training with command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL}" \
OPENCV_FFMPEG_LOGLEVEL="${OPENCV_FFMPEG_LOGLEVEL}" \
"${CMD[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
