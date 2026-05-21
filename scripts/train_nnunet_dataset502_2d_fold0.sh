#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/setup_nnunet_env.sh
source "${SCRIPT_DIR}/setup_nnunet_env.sh"

DATASET_DIR="${nnUNet_raw}/Dataset502_Prostate158_Anatomy"
PREPROCESSED_DIR="${nnUNet_preprocessed}/Dataset502_Prostate158_Anatomy"

if [[ ! -f "${DATASET_DIR}/dataset.json" ]]; then
  echo "Missing Dataset502 dataset.json: ${DATASET_DIR}/dataset.json" >&2
  echo "Run scripts/create_nnunet_dataset502_prostate158_anatomy.py first." >&2
  exit 1
fi

if [[ ! -f "${PREPROCESSED_DIR}/nnUNetPlans.json" ]]; then
  echo "Missing Dataset502 nnUNetPlans.json: ${PREPROCESSED_DIR}/nnUNetPlans.json" >&2
  echo "Run scripts/run_nnunet_dataset502_preprocess.sh first." >&2
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
python scripts/create_nnunet_lowvram_plan.py \
  --preprocessed-dir data/nnunet/nnUNet_preprocessed/Dataset502_Prostate158_Anatomy \
  --two-d-batch-size 4 \
  --three-d-batch-size 1
nnUNetv2_train 502 2d 0 -tr nnUNetTrainer_100epochs -p nnUNetPlans_lowvram
