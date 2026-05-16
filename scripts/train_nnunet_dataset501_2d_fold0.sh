#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/setup_nnunet_env.sh
source "${SCRIPT_DIR}/setup_nnunet_env.sh"

DATASET_DIR="${nnUNet_raw}/Dataset501_ProstateROI_T2"

if [[ ! -f "${DATASET_DIR}/dataset.json" ]]; then
  echo "Missing Dataset501 dataset.json: ${DATASET_DIR}/dataset.json" >&2
  echo "Run scripts/create_nnunet_dataset501_prostate_roi_t2.py first." >&2
  exit 1
fi

if ! command -v nnUNetv2_train >/dev/null 2>&1; then
  echo "nnUNetv2_train not found in PATH." >&2
  echo "Activate the Python environment where nnU-Net v2 is installed." >&2
  exit 1
fi

export nnUNet_compile="${nnUNet_compile:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "nnUNet_compile=${nnUNet_compile}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

cd "${PROJECT_ROOT}"
python scripts/create_nnunet_dataset501_lowvram_plan.py
nnUNetv2_train 501 2d 0 -p nnUNetPlans_lowvram
