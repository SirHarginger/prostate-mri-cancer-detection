"""Build a validated manifest for the Prostate158 training dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prostate_detection.preprocessing.prostate158 import build_prostate158_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Prostate158 manifest and nnU-Net split file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Raw Prostate158 directory containing train.csv, valid.csv, and train/* cases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output manifest CSV path.",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=Path("data/manifests/splits/prostate158_nnunet_split.json"),
        help="Output nnU-Net-style train/validation split JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing manifest and split output files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = build_prostate158_manifest(
        input_dir=args.input_dir,
        manifest_path=args.output,
        split_output_path=args.split_output,
        overwrite=args.overwrite,
    )
    train_count = sum(case.split == "train" for case in cases)
    valid_count = sum(case.split == "valid" for case in cases)
    lesion_count = sum(case.lesion_present for case in cases)
    print(
        "Built Prostate158 manifest with "
        f"{len(cases)} cases ({train_count} train, {valid_count} valid, "
        f"{lesion_count} lesion-positive)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
