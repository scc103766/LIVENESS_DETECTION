#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/supercloud/llm-code/scc/scc/Liveness_Detection}"
RUN_TAG="${RUN_TAG:-v3_1_best_protocol_nan_guard_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/flash_liveness_runs/${RUN_TAG}}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/train.log}"
EXIT_CODE_PATH="${OUTPUT_DIR}/exit_code.txt"
ANALYSIS_PATH="${OUTPUT_DIR}/nan_failure_analysis.md"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

{
  printf 'Run tag: %s\n' "${RUN_TAG}"
  printf 'Output dir: %s\n' "${OUTPUT_DIR}"
  printf 'Log path: %s\n' "${LOG_PATH}"
  printf 'Started at: %s\n' "$(date -Is)"
  printf 'GPU_IDS: %s\n' "${GPU_IDS:-0,1,2}"
  printf '\n'
} | tee -a "${LOG_PATH}"

set +e
OUTPUT_DIR="${OUTPUT_DIR}" bash "${PROJECT_ROOT}/scripts/run_flash_liveness_best_training.sh" 2>&1 | tee -a "${LOG_PATH}"
train_rc=${PIPESTATUS[0]}
set -e

printf '%s\n' "${train_rc}" > "${EXIT_CODE_PATH}"

if grep -q "Non-finite V3_1 training loss detected" "${LOG_PATH}"; then
  {
    printf '# V3.1 NaN/Inf Loss Analysis\n\n'
    printf '- time: `%s`\n' "$(date -Is)"
    printf '- output_dir: `%s`\n' "${OUTPUT_DIR}"
    printf '- exit_code: `%s`\n\n' "${train_rc}"
    printf '## Detected Context\n\n'
    printf 'The training process stopped because `assert_finite_window_loss()` detected non-finite loss or tensor values.\n\n'
    printf '## Most Likely Causes\n\n'
    printf '- Input frames, color features, physical features, or FFT targets contain NaN/Inf.\n'
    printf '- AMP overflow happened before GradScaler could recover.\n'
    printf '- A particular video/window produced extreme values after face preprocessing or physical cue extraction.\n'
    printf '- Auxiliary losses became unstable despite caps; inspect weighted and raw auxiliary losses below.\n\n'
    printf '## Immediate Mitigations\n\n'
    printf '- Retry with `--no-amp` if AMP overflow is indicated.\n'
    printf '- Lower `--learning-rate`, `--aux-loss-max-ratio`, or `--aux-loss-total-max-ratio`.\n'
    printf '- Inspect the specific `sample_idx`, `window_start`, `window_end`, and tensor stats in the log excerpt.\n\n'
    printf '## Log Excerpt\n\n'
    printf '```text\n'
    grep -n -A160 -B30 "Non-finite V3_1 training loss detected" "${LOG_PATH}"
    printf '\n```\n'
  } > "${ANALYSIS_PATH}"
  printf 'NaN/Inf analysis written to: %s\n' "${ANALYSIS_PATH}" | tee -a "${LOG_PATH}"
fi

printf 'Finished at: %s\n' "$(date -Is)" | tee -a "${LOG_PATH}"
printf 'Exit code: %s\n' "${train_rc}" | tee -a "${LOG_PATH}"
exit "${train_rc}"
