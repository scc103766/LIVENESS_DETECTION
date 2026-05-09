#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/supercloud/llm-code/scc/scc/Liveness_Detection}"
CONDA_ENV="${CONDA_ENV:-anti-spoofing_scc_175}"
PYTHON_BIN="${PYTHON_BIN:-/home/scc/anaconda3/envs/${CONDA_ENV}/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/dataset/flash_liveness_asset_archive_fixed_collect_protocol_restore_original}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/flash_liveness_runs/flash_liveness_v3_1_fullframes_window_restore_original_gpu012}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
WINDOW_STRIDE="${WINDOW_STRIDE:-128}"
EVAL_WINDOW_SIZE="${EVAL_WINDOW_SIZE:-256}"
EVAL_WINDOW_STRIDE="${EVAL_WINDOW_STRIDE:-128}"
WINDOW_FUSION="${WINDOW_FUSION:-quality_lower_trimmed_mean}"
WINDOW_TRIM_RATIO="${WINDOW_TRIM_RATIO:-0.2}"
WINDOW_MIN_QUALITY="${WINDOW_MIN_QUALITY:-0.05}"
AUX_LOSS_MAX_RATIO="${AUX_LOSS_MAX_RATIO:-0.35}"
AUX_LOSS_TOTAL_MAX_RATIO="${AUX_LOSS_TOTAL_MAX_RATIO:-0.70}"
BALANCED_TRAIN_SAMPLER="${BALANCED_TRAIN_SAMPLER:-true}"
YAW_AUGMENT_PROB="${YAW_AUGMENT_PROB:-0.35}"
YAW_AUGMENT_MAX_RATIO="${YAW_AUGMENT_MAX_RATIO:-0.10}"
FLASH_WARMUP_SECONDS="${FLASH_WARMUP_SECONDS:-1.0}"
FLASH_HOLD_SECONDS="${FLASH_HOLD_SECONDS:-0.35}"
FLASH_RESTORE_SECONDS="${FLASH_RESTORE_SECONDS:-0.15}"
FLASH_TAIL_SECONDS="${FLASH_TAIL_SECONDS:-0.5}"
LOG_INTERVAL="${LOG_INTERVAL:-0}"
LOG_SECONDS="${LOG_SECONDS:-60}"
DEVICE="${DEVICE:-cuda:0}"
# Project GPU policy: only physical GPUs 0, 1, and 2 are allowed. Do not use 3, 4, 5, or 6.
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
AUTO_RESUME="${AUTO_RESUME:-true}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-SILENT}"
OPENCV_FFMPEG_LOGLEVEL="${OPENCV_FFMPEG_LOGLEVEL:--8}"

mkdir -p "${OUTPUT_DIR}"

if [[ -z "${RESUME_CHECKPOINT}" && "${AUTO_RESUME}" == "true" && -f "${OUTPUT_DIR}/last_flash_liveness_model.pth" ]]; then
  RESUME_CHECKPOINT="${OUTPUT_DIR}/last_flash_liveness_model.pth"
fi

CMD=(
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/flash_liveness_project_v3_1.py"
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
  --max-train-frames 0
  --max-eval-frames 0
  --window-size "${WINDOW_SIZE}"
  --window-stride "${WINDOW_STRIDE}"
  --eval-window-size "${EVAL_WINDOW_SIZE}"
  --eval-window-stride "${EVAL_WINDOW_STRIDE}"
  --window-fusion "${WINDOW_FUSION}"
  --window-trim-ratio "${WINDOW_TRIM_RATIO}"
  --window-min-quality "${WINDOW_MIN_QUALITY}"
  --aux-loss-max-ratio "${AUX_LOSS_MAX_RATIO}"
  --aux-loss-total-max-ratio "${AUX_LOSS_TOTAL_MAX_RATIO}"
  --yaw-augment-prob "${YAW_AUGMENT_PROB}"
  --yaw-augment-max-ratio "${YAW_AUGMENT_MAX_RATIO}"
  --flash-warmup-seconds "${FLASH_WARMUP_SECONDS}"
  --flash-hold-seconds "${FLASH_HOLD_SECONDS}"
  --flash-restore-seconds "${FLASH_RESTORE_SECONDS}"
  --flash-tail-seconds "${FLASH_TAIL_SECONDS}"
  --log-interval "${LOG_INTERVAL}"
  --log-seconds "${LOG_SECONDS}"
  --device "${DEVICE}"
  --no-multi-gpu
)

if [[ "${BALANCED_TRAIN_SAMPLER}" == "true" ]]; then
  CMD+=(--balanced-train-sampler)
else
  CMD+=(--no-balanced-train-sampler)
fi

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  CMD+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi

printf 'Project root: %s\n' "${PROJECT_ROOT}"
printf 'Physical GPU ids: %s\n' "${GPU_IDS}"
printf 'Process device: %s\n' "${DEVICE}"
printf 'Output dir: %s\n' "${OUTPUT_DIR}"
printf 'Full-frame CPU read: enabled\n'
printf 'Window train/eval: %s/%s, stride %s/%s\n' "${WINDOW_SIZE}" "${EVAL_WINDOW_SIZE}" "${WINDOW_STRIDE}" "${EVAL_WINDOW_STRIDE}"
printf 'Window fusion: %s, trim_ratio=%s, min_quality=%s\n' "${WINDOW_FUSION}" "${WINDOW_TRIM_RATIO}" "${WINDOW_MIN_QUALITY}"
printf 'Aux loss caps: per_term=%s, total=%s\n' "${AUX_LOSS_MAX_RATIO}" "${AUX_LOSS_TOTAL_MAX_RATIO}"
printf 'Balanced train sampler: %s\n' "${BALANCED_TRAIN_SAMPLER}"
printf 'Yaw augment: prob=%s, max_ratio=%s\n' "${YAW_AUGMENT_PROB}" "${YAW_AUGMENT_MAX_RATIO}"
printf 'Collect-flash timing: warmup=%s hold=%s restore=%s tail=%s\n' "${FLASH_WARMUP_SECONDS}" "${FLASH_HOLD_SECONDS}" "${FLASH_RESTORE_SECONDS}" "${FLASH_TAIL_SECONDS}"
printf 'Batch size: %s\n' "${BATCH_SIZE}"
printf 'Batch log cadence: interval=%s, seconds=%s\n' "${LOG_INTERVAL}" "${LOG_SECONDS}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  printf 'Resume checkpoint: %s\n' "${RESUME_CHECKPOINT}"
fi
printf 'Launching V3_1 window training with command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL}" \
OPENCV_FFMPEG_LOGLEVEL="${OPENCV_FFMPEG_LOGLEVEL}" \
"${CMD[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
