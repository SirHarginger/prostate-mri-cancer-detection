"""Command-line entrypoint for prostate MRI cancer detection workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from prostate_mri_cancer_detection.data import build_and_write_manifest
from prostate_mri_cancer_detection.preprocessing import validate_preprocessing_inputs


def build_parser() -> argparse.ArgumentParser:
    """Create the project CLI parser."""

    parser = argparse.ArgumentParser(
        prog="prostate-mri-cancer-detection",
        description="Utilities for the PI-CAI prostate MRI research workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Build the Stage 1 PI-CAI dataset inventory manifest.",
    )
    manifest_parser.add_argument(
        "--raw-root",
        default="data/raw/picai",
        type=Path,
        help="Path to the local PI-CAI raw root.",
    )
    manifest_parser.add_argument(
        "--output",
        default="data/interim/picai_manifest.csv",
        type=Path,
        help="CSV manifest output path.",
    )
    manifest_parser.add_argument(
        "--report",
        default="data/interim/picai_manifest_validation.json",
        type=Path,
        help="JSON validation report output path.",
    )
    manifest_parser.set_defaults(func=run_manifest)

    preprocessing_parser = subparsers.add_parser(
        "preprocessing-validate",
        help="Validate Stage 2 preprocessing inputs for a manifest sample.",
    )
    preprocessing_parser.add_argument(
        "--manifest",
        default="data/interim/picai_manifest.csv",
        type=Path,
        help="Stage 1 manifest CSV path.",
    )
    preprocessing_parser.add_argument(
        "--raw-root",
        default="data/raw/picai",
        type=Path,
        help="Path to the local PI-CAI raw root.",
    )
    preprocessing_parser.add_argument(
        "--report",
        default="outputs/reports/preprocessing_sample_validation.json",
        type=Path,
        help="JSON preprocessing validation report output path.",
    )
    preprocessing_parser.add_argument(
        "--sample-size",
        default=10,
        type=int,
        help="Number of sorted manifest cases to validate when --case-id is not used.",
    )
    preprocessing_parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Specific case ID to validate. Can be provided multiple times.",
    )
    preprocessing_parser.add_argument(
        "--fold",
        action="append",
        default=[],
        help="Fold to sample from. Can be provided multiple times.",
    )
    preprocessing_parser.set_defaults(func=run_preprocessing_validate)
    return parser


def run_manifest(args: argparse.Namespace) -> int:
    """Run the Stage 1 PI-CAI manifest command."""

    result = build_and_write_manifest(args.raw_root, args.output, args.report)
    report = result.report

    print(f"Wrote manifest: {args.output}")
    print(f"Wrote validation report: {args.report}")
    print(f"Total cases: {report['total_cases']}")
    print(f"Cases by fold: {report['cases_by_fold']}")
    print(f"Modality availability: {report['modality_available_counts']}")
    print(f"Clinical rows linked: {report['clinical_rows_linked']}")
    print(f"Gland mask cases linked: {report['gland_mask_cases_linked']}")
    print(f"Lesion mask cases linked: {report['lesion_mask_cases_linked']}")
    print(f"Non-manifest image files: {report['non_manifest_image_files_by_suffix']}")
    print(f"Skipped image files: {report['skipped_image_files_count']}")
    print(f"Missing data counts: {report['missing_data_counts']}")
    return 0


def run_preprocessing_validate(args: argparse.Namespace) -> int:
    """Run Stage 2 preprocessing input validation."""

    report = validate_preprocessing_inputs(
        manifest_path=args.manifest,
        raw_root=args.raw_root,
        report_path=args.report,
        sample_size=args.sample_size,
        case_ids=args.case_id,
        folds=args.fold,
    )
    summary = report["summary"]

    print(f"Wrote preprocessing validation report: {args.report}")
    print(f"Cases checked: {summary['cases_checked']}")
    print(f"Cases with blocking issues: {summary['cases_with_blocking_issues']}")
    print(f"Cases requiring resampling: {summary['cases_requiring_resampling']}")
    print(f"Readable modality headers: {summary['modality_headers_readable']}")
    print(f"Readable mask headers: {summary['mask_headers_readable']}")
    print(f"T2W-compatible mask cases: {summary['mask_t2w_compatible_cases']}")
    print(f"Blocking issue counts: {summary['issue_counts']}")
    print(f"Resampling required counts: {summary['resampling_required_counts']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch commands."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
