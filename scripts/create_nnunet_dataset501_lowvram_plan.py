"""Create a low-VRAM nnU-Net plan for Dataset501 without touching raw data."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create nnUNetPlans_lowvram.json for Dataset501_ProstateROI_T2."
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("data/nnunet/nnUNet_preprocessed/Dataset501_ProstateROI_T2"),
        help="Dataset501 nnU-Net preprocessed directory.",
    )
    parser.add_argument(
        "--source-plans",
        default="nnUNetPlans.json",
        help="Source nnU-Net plans filename inside --preprocessed-dir.",
    )
    parser.add_argument(
        "--output-plans",
        default="nnUNetPlans_lowvram.json",
        help="Output low-VRAM plans filename inside --preprocessed-dir.",
    )
    parser.add_argument(
        "--three-d-batch-size",
        type=int,
        default=1,
        help="Batch size for 3d_fullres in the low-VRAM plan.",
    )
    parser.add_argument(
        "--two-d-batch-size",
        type=int,
        default=8,
        help="Batch size for 2d in the low-VRAM plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.preprocessed_dir / args.source_plans
    output_path = args.preprocessed_dir / args.output_plans

    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source plans file: {source_path}")
    if args.three_d_batch_size < 1:
        raise ValueError("--three-d-batch-size must be >= 1")
    if args.two_d_batch_size < 1:
        raise ValueError("--two-d-batch-size must be >= 1")

    with source_path.open("r", encoding="utf-8") as f:
        plans = json.load(f)

    lowvram = copy.deepcopy(plans)
    lowvram["plans_name"] = Path(args.output_plans).stem
    lowvram.setdefault("low_vram_notes", {})
    lowvram["low_vram_notes"] = {
        "source_plans": args.source_plans,
        "reason": "Reduce batch sizes for 4 GB GPUs such as Quadro P1000.",
        "raw_data_modified": False,
    }

    configurations = lowvram.get("configurations", {})
    if "3d_fullres" not in configurations:
        raise ValueError("Source plans do not contain a 3d_fullres configuration")
    configurations["3d_fullres"]["batch_size"] = args.three_d_batch_size
    if "2d" in configurations:
        configurations["2d"]["batch_size"] = args.two_d_batch_size

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(lowvram, f, indent=4)
        f.write("\n")

    print(f"Wrote low-VRAM plans: {output_path}")
    print(f"3d_fullres batch_size={args.three_d_batch_size}")
    if "2d" in configurations:
        print(f"2d batch_size={args.two_d_batch_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
