#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_PATH="${1:-${SCRIPT_DIR}/best_flash_liveness_model.pth}"
PART_PREFIX="${SCRIPT_DIR}/best_flash_liveness_model.pth.part-"
EXPECTED_BYTES="214879434"
EXPECTED_SHA256="03145cccb43c958bb947a4f9c1d26a2e588fcc870c4974183614f590940e987b"

if ! compgen -G "${PART_PREFIX}*" > /dev/null; then
  echo "No checkpoint parts found: ${PART_PREFIX}*" >&2
  exit 1
fi

cat "${PART_PREFIX}"* > "${OUTPUT_PATH}"

actual_bytes="$(wc -c < "${OUTPUT_PATH}" | tr -d ' ')"
if [[ "${actual_bytes}" != "${EXPECTED_BYTES}" ]]; then
  echo "Size mismatch: expected ${EXPECTED_BYTES}, got ${actual_bytes}" >&2
  exit 1
fi

actual_sha256="$(sha256sum "${OUTPUT_PATH}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "SHA256 mismatch: expected ${EXPECTED_SHA256}, got ${actual_sha256}" >&2
  exit 1
fi

echo "Restored checkpoint: ${OUTPUT_PATH}"
echo "SHA256: ${actual_sha256}"
