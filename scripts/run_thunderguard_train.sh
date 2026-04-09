#!/usr/bin/env bash
set -euo pipefail

ROOT="/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard"
ENV_ROOT="/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/environment/exp_yanyu_cv"
SCRIPT_ROOT="/supercloud/llm-code/scc/scc/Liveness_Detection/scripts"

SMOKE=0
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--smoke" ]]; then
    SMOKE=1
  else
    ARGS+=("$arg")
  fi
done

if [[ "$SMOKE" -eq 1 ]]; then
  python3 "$SCRIPT_ROOT/build_thunderguard_smoke_subset.py"
  ARGS+=(
    --data_path ../../dataset/tg_export_smoke
    --model_path ../../resources_smoke
    --epoch_num 1
    --batch_size 1
    --num_workers 0
  )
fi

cd "$ROOT/pytg"
export LD_LIBRARY_PATH="$ENV_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/thunderguard-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/thunderguard-cache}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME"

exec "$ENV_ROOT/bin/python" train.py "${ARGS[@]}"
