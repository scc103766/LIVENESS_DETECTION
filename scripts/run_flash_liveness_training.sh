#!/usr/bin/env bash
set -euo pipefail

# 为 flash_liveness_project.py 提供一条可直接复用的训练命令。
# 默认使用我们刚整理好的视频数据集，并把控制台日志同步保存到 output_dir/train.log。

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_video_dataset_v1}"
RUN_NAME="${RUN_NAME:-flash_liveness_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/${RUN_NAME}}"

# 这套模型每个样本会读 16 帧 224x224 图像，显存压力不小。
# 推荐起步:
# - 24GB 显存: BATCH_SIZE=4
# - 12GB 显存: BATCH_SIZE=2
# - 8GB 显存或更小: BATCH_SIZE=1
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-20}"
NUM_FRAMES="${NUM_FRAMES:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
POS_WEIGHT="${POS_WEIGHT:-auto}"
DEVICE="${DEVICE:-cuda:0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
USE_IMAGENET_PRETRAINED="${USE_IMAGENET_PRETRAINED:-true}"
IMAGENET_PRETRAINED_PATH="${IMAGENET_PRETRAINED_PATH:-/supercloud/llm-code/scc/scc/Liveness_Detection/resnet18-f37072fd.pth}"

# 如需启用 YOLO 人脸检测，可设置 DETECTOR_MODEL 和 DETECTOR_DEVICE。
DETECTOR_MODEL="${DETECTOR_MODEL:-}"
DETECTOR_DEVICE="${DETECTOR_DEVICE:-}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}"
  "/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_project.py"
  train
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --num-frames "${NUM_FRAMES}"
  --image-size "${IMAGE_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --pos-weight "${POS_WEIGHT}"
  --device "${DEVICE}"
  --imagenet-pretrained-path "${IMAGENET_PRETRAINED_PATH}"
)

if [[ "${USE_IMAGENET_PRETRAINED}" == "true" ]]; then
  CMD+=(--use-imagenet-pretrained)
else
  CMD+=(--no-use-imagenet-pretrained)
fi

if [[ -n "${DETECTOR_MODEL}" ]]; then
  CMD+=(--detector-model "${DETECTOR_MODEL}")
fi

if [[ -n "${DETECTOR_DEVICE}" ]]; then
  CMD+=(--detector-device "${DETECTOR_DEVICE}")
fi

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  CMD+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi

printf 'Launching training with command:\n%s\n' "${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
