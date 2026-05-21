"""Prepare Kaggle PROSTATE_MRI axial T2 series for nnU-Net inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prostate_detection.preprocessing.kaggle_prostate_mri import (
    KAGGLE_T2_AXIAL_SERIES_DESCRIPTION,
    build_kaggle_t2_inference_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Kaggle PROSTATE_MRI axial T2 DICOM series to nnU-Net imagesTs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/world-wide-covid-dataset/PROSTATE_MRI"),
        help="Extracted Kaggle PROSTATE_MRI directory containing metadata.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/kaggle_prostate_mri_t2_nifti"),
        help="Output directory for converted nnU-Net imagesTs.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/manifests/kaggle_prostate_mri_t2_manifest.csv"),
        help="Output manifest CSV for converted inference images.",
    )
    parser.add_argument(
        "--series-description",
        default=KAGGLE_T2_AXIAL_SERIES_DESCRIPTION,
        help="DICOM Series Description to select for inference.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing expected converted NIfTI outputs and manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selected series without writing converted images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    converted = build_kaggle_t2_inference_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest_output,
        series_description=args.series_description,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("Dry run completed.")
    else:
        print(f"Converted {len(converted)} Kaggle axial T2 series for nnU-Net inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
