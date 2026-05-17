"""Create nnU-Net Dataset503_Prostate158_Lesion from a Prostate158 manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prostate_detection.preprocessing.prostate158 import (
    create_dataset503_prostate158_lesion,
    validate_dataset503,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Prostate158 T2/ADC/DWI and ADC lesion masks to nnU-Net Dataset503."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Prostate158 manifest CSV from build_prostate158_manifest.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="nnU-Net raw output directory for Dataset503_Prostate158_Lesion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing expected Dataset503 output files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = create_dataset503_prostate158_lesion(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    validate_dataset503(args.output_dir)
    print(f"Created Dataset503_Prostate158_Lesion with {len(records)} training cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
