"""Create nnU-Net Dataset501_ProstateROI_T2 from MSD Task05 Prostate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prostate_detection.preprocessing.nnunet_conversion import (
    create_dataset501_prostate_roi_t2,
    validate_dataset501,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MSD Task05 Prostate images and prepared binary ROI labels "
            "to nnU-Net Dataset501_ProstateROI_T2."
        )
    )
    parser.add_argument(
        "--msd-dir",
        type=Path,
        required=True,
        help="MSD Task05_Prostate directory containing dataset.json, imagesTr, and imagesTs.",
    )
    parser.add_argument(
        "--binary-roi-dir",
        type=Path,
        required=True,
        help="Directory from convert_msd_prostate_labels_to_binary_roi.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="nnU-Net raw dataset output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing expected nnU-Net output files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = create_dataset501_prostate_roi_t2(
        msd_dir=args.msd_dir,
        binary_roi_dir=args.binary_roi_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    validate_dataset501(args.output_dir)
    print(
        "Created Dataset501_ProstateROI_T2 with "
        f"{len(plan.training_cases)} training cases and {len(plan.test_cases)} test cases."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
