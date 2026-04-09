#!/usr/bin/env bash
set -euo pipefail

ROOT="/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档"
SAMPLE_DIR="$ROOT/ThunderGuard/data/sample"
EXPORT_DIR="$ROOT/dataset/tg_export"

mkdir -p "$SAMPLE_DIR"

for split in train test; do
  target="$SAMPLE_DIR/$split"
  source="$EXPORT_DIR/$split"

  if [ ! -e "$source" ]; then
    echo "missing source split: $source" >&2
    exit 1
  fi

  if [ -L "$target" ]; then
    rm -f "$target"
  elif [ -e "$target" ]; then
    echo "target already exists and is not a symlink: $target" >&2
    exit 1
  fi

  ln -s "$source" "$target"
  echo "linked $target -> $source"
done
