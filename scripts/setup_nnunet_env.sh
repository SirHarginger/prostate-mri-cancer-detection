#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export nnUNet_raw="${PROJECT_ROOT}/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/outputs/nnunet_results"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "Directories are ready."
echo "Tip: source this script to keep variables in your current shell:"
echo "  source scripts/setup_nnunet_env.sh"

