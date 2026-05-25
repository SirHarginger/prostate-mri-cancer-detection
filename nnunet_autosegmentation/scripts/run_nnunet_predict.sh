#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-nnunet_autosegmentation/config/prostate_autoseg_config.json}"

if ! command -v nnUNetv2_predict >/dev/null 2>&1; then
  echo "ERROR: nnUNetv2_predict was not found in PATH."
  echo "Activate/install nnU-Net v2 before running this script."
  exit 2
fi

eval "$(
python - <<'PY' "${CONFIG_PATH}"
import json
import shlex
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text())
nnunet = config["nnunet"]

values = {
    "NNUNET_RAW": nnunet["raw"],
    "NNUNET_PREPROCESSED": nnunet["preprocessed"],
    "NNUNET_RESULTS": nnunet["results"],
    "INPUT_DIR": nnunet["input_dir"],
    "OUTPUT_DIR": nnunet["output_dir"],
    "DATASET_ID": str(nnunet["dataset_id"]),
    "CONFIGURATION": nnunet["configuration"],
    "TRAINER": nnunet["trainer"],
    "FOLDS": str(nnunet["folds"]),
}

for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

export nnUNet_raw="${NNUNET_RAW}"
export nnUNet_preprocessed="${NNUNET_PREPROCESSED}"
export nnUNet_results="${NNUNET_RESULTS}"

mkdir -p "${OUTPUT_DIR}"

MODEL_DIR="${NNUNET_RESULTS}/Dataset${DATASET_ID}_${DATASET_LABEL}/${TRAINER}__nnUNetPlans__${CONFIGURATION}"
CHECKPOINT_NAME="checkpoint_final.pth"

if [[ ! -f "${MODEL_DIR}/fold_${FOLDS}/${CHECKPOINT_NAME}" ]]; then
  if find "${MODEL_DIR}" -maxdepth 2 -name checkpoint_final.pth -type f | grep -q .; then
    CHECKPOINT_NAME="checkpoint_final.pth"
  elif find "${MODEL_DIR}" -maxdepth 2 -name checkpoint_best.pth -type f | grep -q .; then
    CHECKPOINT_NAME="checkpoint_best.pth"
  fi

  DETECTED_FOLDS="$(
    find "${MODEL_DIR}" -maxdepth 2 -path "*/fold_*/${CHECKPOINT_NAME}" -type f \
      | sed -E 's#.*/fold_([^/]+)/[^/]+#\1#' \
      | sort -V \
      | tr '\n' ' ' \
      | sed 's/[[:space:]]*$//'
  )"

  if [[ -n "${DETECTED_FOLDS}" ]]; then
    FOLDS="${DETECTED_FOLDS}"
  fi
fi

read -r -a FOLD_ARGS <<< "${FOLDS}"

echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "Model: ${MODEL_DIR}"
echo "Checkpoint: ${CHECKPOINT_NAME}"
echo "Folds: ${FOLDS}"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"

nnUNetv2_predict \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_DIR}" \
  -d "${DATASET_ID}" \
  -c "${CONFIGURATION}" \
  -tr "${TRAINER}" \
  -chk "${CHECKPOINT_NAME}" \
  -f "${FOLD_ARGS[@]}"
