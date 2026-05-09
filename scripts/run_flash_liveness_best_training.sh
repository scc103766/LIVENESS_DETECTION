#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/supercloud/llm-code/scc/scc/Liveness_Detection"
CONDA_ENV="${CONDA_ENV:-anti-spoofing_scc_175}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/dataset/flash_liveness_best_protocol_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/flash_liveness_runs/flash_liveness_best_protocol_v1}"
GPU_IDS="${GPU_IDS:-0,1,2}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-30}"

cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$CONDA_ENV" python flash_liveness_project_v3_1.py train \
  --data-root "$DATA_ROOT" \
  --dataset-media videos \
  --require-color-txt \
  --missing-color-protocol neutral \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers 4 \
  --window-size 256 \
  --window-stride 128 \
  --eval-window-size 256 \
  --eval-window-stride 128 \
  --window-fusion quality_lower_trimmed_mean \
  --window-trim-ratio 0.2 \
  --use-physical-features \
  --lambda-depth 0.05 \
  --lambda-contrast 0.10 \
  --lambda-fft 0.05 \
  --aux-loss-max-ratio 0.35 \
  --aux-loss-total-max-ratio 0.70 \
  --balanced-train-sampler \
  --yaw-augment-prob 0.35 \
  --yaw-augment-max-ratio 0.10 \
  --device "$DEVICE" \
  --use-imagenet-pretrained \
  --imagenet-pretrained-path "$PROJECT_ROOT/resnet18-f37072fd.pth"
