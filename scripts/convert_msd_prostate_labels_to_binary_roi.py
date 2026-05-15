"""Convert MSD Task05 prostate labels to binary whole-prostate ROI masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prostate_detection.preprocessing.msd_binary_roi import convert_msd_labels_to_binary_roi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MSD Task05 prostate PZ/TZ labels into binary ROI masks."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/public/Task05_Prostate"),
        help="MSD Task05_Prostate directory containing dataset.json, imagesTr, and labelsTr.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/public/msd_prostate_binary_roi"),
        help="Directory where binary labels and manifest.csv will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing binary labels and manifest.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report planned outputs without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = convert_msd_labels_to_binary_roi(
        args.input_dir,
        args.output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    action = "Validated" if args.dry_run else "Converted"
    print(f"{action} {len(records)} MSD prostate training labels.")
    print(f"Output directory: {args.output_dir}")
    print(f"Manifest: {args.output_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
