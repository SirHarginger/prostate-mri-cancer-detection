#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/setup_nnunet_env.sh
source "${SCRIPT_DIR}/setup_nnunet_env.sh"

DATASET_DIR="${nnUNet_raw}/Dataset503_Prostate158_Lesion"

if [[ ! -f "${DATASET_DIR}/dataset.json" ]]; then
  echo "Missing Dataset503 dataset.json: ${DATASET_DIR}/dataset.json" >&2
  echo "Run scripts/create_nnunet_dataset503_prostate158_lesion.py first." >&2
  exit 1
fi

if ! command -v nnUNetv2_plan_and_preprocess >/dev/null 2>&1; then
  echo "nnUNetv2_plan_and_preprocess not found in PATH." >&2
  echo "Activate the Python environment where nnU-Net v2 is installed." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
nnUNetv2_plan_and_preprocess -d 503 --verify_dataset_integrity
