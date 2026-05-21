"""Create QC overlays for Kaggle PROSTATE_MRI auto-segmentation masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prostate_detection.visualization.kaggle_auto_segmentation import (
    print_qc_summary,
    save_kaggle_auto_segmentation_qc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Kaggle PROSTATE_MRI T2 auto-segmentation outputs."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/kaggle_prostate_mri_t2_manifest.csv"),
        help="Kaggle T2 conversion manifest.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs/predictions/kaggle_prostate_mri_anatomy_auto"),
        help="Auto-segmented anatomy mask directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/qc/kaggle_prostate_mri_auto_segmentations"),
        help="Output directory for QC PNG overlays.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional maximum number of cases to visualize.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing QC figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = save_kaggle_auto_segmentation_qc(
        manifest_path=args.manifest,
        predictions_dir=args.predictions_dir,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        overwrite=args.overwrite,
    )
    print_qc_summary(paths, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
