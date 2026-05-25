#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-nnunet_autosegmentation/config/prostate_autoseg_config.json}"

python - <<'PY' "${CONFIG_PATH}"
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
nnunet = config["nnunet"]

print("export nnUNet_raw=" + nnunet["raw"])
print("export nnUNet_preprocessed=" + nnunet["preprocessed"])
print("export nnUNet_results=" + nnunet["results"])
print("nnUNetv2_predict \\")
print(f"  -i {nnunet['input_dir']} \\")
print(f"  -o {nnunet['output_dir']} \\")
print(f"  -d {nnunet['dataset_id']} \\")
print(f"  -c {nnunet['configuration']} \\")
print(f"  -tr {nnunet['trainer']} \\")
print(f"  -f {nnunet['folds']}")
PY

echo
echo "Review the command above, source your nnU-Net environment, then run it on the cluster."
