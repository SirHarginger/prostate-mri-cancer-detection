"""Prepare external prostate MRI cases for nnU-Net inference.

This script intentionally starts as a conservative manifest-based converter
stub. DICOM-to-NIfTI conversion differs by dataset export method, so the first
validated implementation should be based on the actual downloaded PROSTATEx
layout on the cluster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare external MRI data for nnU-Net inference.")
    parser.add_argument("--config", default="nnunet_autosegmentation/config/prostate_autoseg_config.json", type=Path)
    parser.add_argument("--report", default="nnunet_autosegmentation/outputs/reports/prepare_nnunet_dataset_report.json", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    report = {
        "stage": "prepare_nnunet_dataset",
        "status": "not_implemented_until_external_layout_is_available",
        "config": str(args.config),
        "expected_input_root": config["external_source"]["download_dir"],
        "expected_nnunet_input_dir": config["nnunet"]["input_dir"],
        "next_step": "Inspect downloaded PROSTATEx layout on the cluster, then implement conversion rules.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote report: {args.report}")
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
