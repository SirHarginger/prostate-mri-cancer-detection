"""Evaluate segmentation predictions according to a config file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prostate_detection.utils.cli import not_implemented, summarize_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prostate MRI segmentation outputs.")
    parser.add_argument("--config", required=True, type=Path, help="Evaluation config path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(f"Dry run: {summarize_args(args)}")
        return 0
    return not_implemented(Path(__file__).name, args)


if __name__ == "__main__":
    sys.exit(main())
