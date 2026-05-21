"""Evaluate Prostate158 Dataset502 anatomy predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prostate_detection.evaluation.prostate158_predictions import (
    evaluate_dataset502_predictions,
    print_summary,
    save_dataset502_qc_figures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Prostate158 Dataset502 anatomy predictions."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/prostate158_manifest.csv"),
        help="Prostate158 manifest CSV.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/nnunet/nnUNet_raw/Dataset502_Prostate158_Anatomy/labelsTr"),
        help="Ground-truth Dataset502 labelsTr directory.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("outputs/predictions/dataset502_val_best"),
        help="Dataset502 prediction directory.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("outputs/metrics/prostate158_dataset502_anatomy_metrics.csv"),
        help="Output per-case metrics CSV.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/metrics/prostate158_dataset502_anatomy_summary.json"),
        help="Output summary metrics JSON.",
    )
    parser.add_argument(
        "--qc-output-dir",
        type=Path,
        default=Path("outputs/figures/qc/prostate158_predictions"),
        help="Optional output directory for QC figures.",
    )
    parser.add_argument(
        "--qc-count",
        type=int,
        default=0,
        help="Number of validation cases to render as QC figures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing metrics or QC figure outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = evaluate_dataset502_predictions(
        manifest_path=args.manifest,
        labels_dir=args.labels_dir,
        predictions_dir=args.predictions_dir,
        metrics_csv=args.metrics_csv,
        summary_json=args.summary_json,
        overwrite=args.overwrite,
    )
    print_summary(metrics, args.summary_json, args.metrics_csv)

    qc_paths = save_dataset502_qc_figures(
        manifest_path=args.manifest,
        labels_dir=args.labels_dir,
        predictions_dir=args.predictions_dir,
        output_dir=args.qc_output_dir,
        max_cases=args.qc_count,
        overwrite=args.overwrite,
    )
    if qc_paths:
        print(f"Wrote {len(qc_paths)} QC figures to {args.qc_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
