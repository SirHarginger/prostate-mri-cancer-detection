"""Command-line entrypoint for prostate MRI cancer detection workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from prostate_mri_cancer_detection.data import build_and_write_manifest


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
    print(f"Missing data counts: {report['missing_data_counts']}")
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
