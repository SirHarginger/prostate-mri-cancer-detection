"""Create nnU-Net Dataset502_Prostate158_Anatomy from a Prostate158 manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prostate_detection.preprocessing.prostate158 import (
    create_dataset502_prostate158_anatomy,
    validate_dataset502,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Prostate158 T2/anatomy labels to nnU-Net Dataset502."
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
        help="nnU-Net raw output directory for Dataset502_Prostate158_Anatomy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing expected Dataset502 output files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = create_dataset502_prostate158_anatomy(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    validate_dataset502(args.output_dir)
    print(f"Created Dataset502_Prostate158_Anatomy with {len(records)} training cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
