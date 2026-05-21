#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${PROJECT_ROOT}/scripts/setup_nnunet_env.sh"

INPUT_DIR="${PROJECT_ROOT}/data/interim/kaggle_prostate_mri_t2_nifti/imagesTs"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/predictions/kaggle_prostate_mri_anatomy_auto"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
LOG_PATH="${LOG_DIR}/kaggle_prostate_mri_anatomy_predict.log"

NNUNET_PREDICT="${PROJECT_ROOT}/.venv/bin/nnUNetv2_predict"
if [[ ! -x "${NNUNET_PREDICT}" ]]; then
  NNUNET_PREDICT="$(command -v nnUNetv2_predict || true)"
fi
if [[ -z "${NNUNET_PREDICT}" ]]; then
  echo "nnUNetv2_predict not found. Activate the project environment first." >&2
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Missing converted Kaggle imagesTs directory: ${INPUT_DIR}" >&2
  echo "Run scripts/prepare_kaggle_prostate_mri_t2.py first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

export nnUNet_compile=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${NNUNET_PREDICT}" \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_DIR}" \
  -d 502 \
  -c 2d \
  -f 0 \
  -tr nnUNetTrainer \
  -p nnUNetPlans_lowvram \
  -chk checkpoint_latest.pth \
  --continue_prediction \
  --disable_tta \
  --not_on_device \
  -npp 1 \
  -nps 1 \
  --disable_progress_bar \
  2>&1 | tee "${LOG_PATH}"
