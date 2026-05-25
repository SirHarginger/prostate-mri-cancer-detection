#!/usr/bin/env bash
set -euo pipefail

MODEL_URL="https://zenodo.org/records/15746134/files/Prostate_MRL_nnUNetTrainerNoMirroring_3d_fullres.zip?download=1"
MODEL_ZIP="nnunet_autosegmentation/data/nnunet_results/Prostate_MRL_nnUNetTrainerNoMirroring_3d_fullres.zip"
RESULTS_DIR="nnunet_autosegmentation/data/nnunet_results"
REPORT_PATH="nnunet_autosegmentation/outputs/reports/pretrained_nnunet_model_layout.txt"

mkdir -p "${RESULTS_DIR}"
mkdir -p "$(dirname "${REPORT_PATH}")"

if [[ ! -f "${MODEL_ZIP}" ]]; then
  echo "Downloading pretrained nnU-Net prostate/male pelvis model..."
  curl -L -o "${MODEL_ZIP}" "${MODEL_URL}"
else
  echo "Model zip already exists: ${MODEL_ZIP}"
fi

echo "Extracting pretrained model into ${RESULTS_DIR}..."
python - <<'PY' "${MODEL_ZIP}" "${RESULTS_DIR}"
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
results_dir = Path(sys.argv[2])

with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(results_dir)
PY

echo "Detected nnU-Net model layout:" | tee "${REPORT_PATH}"
find "${RESULTS_DIR}" -maxdepth 3 \( -name dataset.json -o -name plans.json -o -name "checkpoint*.pth" \) \
  | sort \
  | tee -a "${REPORT_PATH}"

echo
echo "If prediction still cannot find the model, use the Dataset*/trainer folder shown above to update:"
echo "  nnunet_autosegmentation/config/prostate_autoseg_config.json"
echo
echo "Expected trainer for this model:"
echo "  nnUNetTrainerNoMirroring"
