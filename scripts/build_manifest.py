"""Build a dataset manifest for a prostate MRI dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prostate_detection.utils.cli import not_implemented, summarize_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dataset manifest.")
    parser.add_argument("--dataset", required=True, help="Dataset identifier, e.g. prostate_mri.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Input dataset directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output manifest CSV/JSON path.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments without writing output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(f"Dry run: {summarize_args(args)}")
        return 0
    return not_implemented(Path(__file__).name, args)


if __name__ == "__main__":
    sys.exit(main())
