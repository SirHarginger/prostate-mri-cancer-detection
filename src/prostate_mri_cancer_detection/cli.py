"""Command-line entrypoint for prostate MRI cancer detection workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from prostate_mri_cancer_detection.data import build_and_write_manifest
from prostate_mri_cancer_detection.evaluation import generate_evaluation_report, run_feature_baselines
from prostate_mri_cancer_detection.explainability import generate_explainability_report
from prostate_mri_cancer_detection.features import extract_radiomics_features
from prostate_mri_cancer_detection.modeling import extract_embedding_table
from prostate_mri_cancer_detection.preprocessing import validate_preprocessing_inputs, validate_resampling_plan


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

    resampling_parser = subparsers.add_parser(
        "resampling-validate",
        help="Validate SimpleITK ADC/HBV resampling to T2W grid without writing processed images.",
    )
    resampling_parser.add_argument(
        "--manifest",
        default="data/interim/picai_manifest.csv",
        type=Path,
        help="Stage 1 manifest CSV path.",
    )
    resampling_parser.add_argument(
        "--raw-root",
        default="data/raw/picai",
        type=Path,
        help="Path to the local PI-CAI raw root.",
    )
    resampling_parser.add_argument(
        "--sample-size",
        default=5,
        type=int,
        help="Number of sorted manifest cases to validate when --case-id is not used.",
    )
    resampling_parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Specific case ID to validate. Can be provided multiple times.",
    )
    resampling_parser.add_argument(
        "--report",
        default="outputs/reports/resampling_validation_sample.json",
        type=Path,
        help="JSON resampling validation report output path.",
    )
    resampling_parser.set_defaults(func=run_resampling_validate)

    radiomics_parser = subparsers.add_parser(
        "radiomics-extract",
        help="Extract Stage 3 first-order radiomics features for validated T2W-grid ROIs.",
    )
    radiomics_parser.add_argument(
        "--manifest",
        default="data/interim/picai_manifest.csv",
        type=Path,
        help="Stage 1 manifest CSV path.",
    )
    radiomics_parser.add_argument(
        "--preprocessing-report",
        default="outputs/reports/preprocessing_fold_sample_validation.json",
        type=Path,
        help="Stage 2 preprocessing validation report used for mask selection.",
    )
    radiomics_parser.add_argument(
        "--raw-root",
        default="data/raw/picai",
        type=Path,
        help="Path to the local PI-CAI raw root.",
    )
    radiomics_parser.add_argument(
        "--sequence",
        default="t2w",
        choices=["t2w"],
        help="MRI sequence for Stage 3 extraction. ADC/HBV wait for resampling.",
    )
    radiomics_parser.add_argument(
        "--roi",
        default="lesion",
        choices=["gland", "lesion"],
        help="ROI mask type to use.",
    )
    radiomics_parser.add_argument(
        "--sample-size",
        default=25,
        type=int,
        help="Number of manifest cases to extract if no preprocessing report/case IDs are provided.",
    )
    radiomics_parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Specific case ID to extract. Can be provided multiple times.",
    )
    radiomics_parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Extract all manifest cases. Use only after sample validation succeeds.",
    )
    radiomics_parser.add_argument(
        "--output",
        default="data/features/radiomics_t2w_lesion_sample.csv",
        type=Path,
        help="Feature table output path.",
    )
    radiomics_parser.add_argument(
        "--failure-log",
        default="outputs/reports/radiomics_t2w_lesion_failures.csv",
        type=Path,
        help="Per-case extraction failure log path.",
    )
    radiomics_parser.add_argument(
        "--settings",
        default="outputs/reports/radiomics_t2w_lesion_settings.json",
        type=Path,
        help="Reproducible extraction settings JSON path.",
    )
    radiomics_parser.set_defaults(func=run_radiomics_extract)

    embedding_parser = subparsers.add_parser(
        "embedding-extract",
        help="Extract Stage 4 split-safe prototype embeddings with provenance.",
    )
    embedding_parser.add_argument(
        "--manifest",
        default="data/interim/picai_manifest.csv",
        type=Path,
        help="Stage 1 manifest CSV path.",
    )
    embedding_parser.add_argument(
        "--preprocessing-report",
        default="outputs/reports/preprocessing_fold_sample_validation.json",
        type=Path,
        help="Stage 2 report used to choose the same validated sample when available.",
    )
    embedding_parser.add_argument(
        "--raw-root",
        default="data/raw/picai",
        type=Path,
        help="Path to the local PI-CAI raw root.",
    )
    embedding_parser.add_argument(
        "--sequence",
        default="t2w",
        choices=["t2w"],
        help="Sequence for prototype embeddings. ADC/HBV wait for resampling.",
    )
    embedding_parser.add_argument(
        "--embedding-dim",
        default=32,
        type=int,
        help="Number of embedding columns to write.",
    )
    embedding_parser.add_argument(
        "--sample-size-per-split",
        default=5,
        type=int,
        help="Cases per split when no preprocessing selected-case list is used.",
    )
    embedding_parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Specific case ID to extract. Can be provided multiple times.",
    )
    embedding_parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Extract all manifest cases. Use only after sample validation succeeds.",
    )
    embedding_parser.add_argument(
        "--augment-train",
        action="store_true",
        help="Apply deterministic train-only intensity perturbation for leakage checks.",
    )
    embedding_parser.add_argument(
        "--output",
        default="data/features/embeddings_t2w_prototype_sample.csv",
        type=Path,
        help="Embedding table output path.",
    )
    embedding_parser.add_argument(
        "--provenance",
        default="outputs/reports/embeddings_t2w_prototype_provenance.json",
        type=Path,
        help="Embedding provenance JSON path.",
    )
    embedding_parser.add_argument(
        "--report",
        default="outputs/reports/embeddings_t2w_prototype_report.json",
        type=Path,
        help="Embedding extraction validation report path.",
    )
    embedding_parser.set_defaults(func=run_embedding_extract)

    baseline_parser = subparsers.add_parser(
        "baseline-evaluate",
        help="Run Stage 5 aligned prototype radiomics/embedding/hybrid baselines.",
    )
    baseline_parser.add_argument(
        "--manifest",
        default="data/interim/picai_manifest.csv",
        type=Path,
        help="Stage 1 manifest CSV path.",
    )
    baseline_parser.add_argument(
        "--radiomics",
        default="data/features/radiomics_t2w_gland_sample.csv",
        type=Path,
        help="Radiomics feature table path.",
    )
    baseline_parser.add_argument(
        "--embeddings",
        default="data/features/embeddings_t2w_prototype_sample.csv",
        type=Path,
        help="Embedding table path.",
    )
    baseline_parser.add_argument(
        "--metrics",
        default="outputs/reports/prototype_baseline_metrics.json",
        type=Path,
        help="Baseline metrics JSON output path.",
    )
    baseline_parser.add_argument(
        "--predictions",
        default="outputs/reports/prototype_baseline_predictions.csv",
        type=Path,
        help="Baseline predictions CSV output path.",
    )
    baseline_parser.add_argument(
        "--report",
        default="outputs/reports/prototype_baseline_report.json",
        type=Path,
        help="Baseline validation report JSON output path.",
    )
    baseline_parser.set_defaults(func=run_baseline_evaluate)

    evaluation_parser = subparsers.add_parser(
        "evaluation-report",
        help="Generate Stage 6 metrics and error-analysis reports from predictions.",
    )
    evaluation_parser.add_argument(
        "--predictions",
        default="outputs/reports/prototype_baseline_predictions.csv",
        type=Path,
        help="Prediction CSV from Stage 5 baseline evaluation.",
    )
    evaluation_parser.add_argument(
        "--json-report",
        default="outputs/reports/prototype_evaluation_report.json",
        type=Path,
        help="Structured JSON evaluation report path.",
    )
    evaluation_parser.add_argument(
        "--markdown-report",
        default="outputs/reports/prototype_evaluation_report.md",
        type=Path,
        help="Markdown evaluation report path.",
    )
    evaluation_parser.add_argument(
        "--target-sensitivity",
        default=0.90,
        type=float,
        help="Target sensitivity for fixed-sensitivity analysis.",
    )
    evaluation_parser.set_defaults(func=run_evaluation_report)

    explainability_parser = subparsers.add_parser(
        "explainability-report",
        help="Generate Stage 7 prototype feature-importance reports.",
    )
    explainability_parser.add_argument(
        "--manifest",
        default="data/interim/picai_manifest.csv",
        type=Path,
        help="Stage 1 manifest CSV path.",
    )
    explainability_parser.add_argument(
        "--radiomics",
        default="data/features/radiomics_t2w_gland_sample.csv",
        type=Path,
        help="Radiomics feature table path.",
    )
    explainability_parser.add_argument(
        "--embeddings",
        default="data/features/embeddings_t2w_prototype_sample_all25.csv",
        type=Path,
        help="Embedding table path.",
    )
    explainability_parser.add_argument(
        "--json-report",
        default="outputs/reports/prototype_explainability_report.json",
        type=Path,
        help="Structured explainability JSON report path.",
    )
    explainability_parser.add_argument(
        "--csv-report",
        default="outputs/reports/prototype_feature_importance.csv",
        type=Path,
        help="Feature-importance CSV report path.",
    )
    explainability_parser.add_argument(
        "--top-n",
        default=20,
        type=int,
        help="Number of top features per baseline to report.",
    )
    explainability_parser.set_defaults(func=run_explainability_report)
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


def run_resampling_validate(args: argparse.Namespace) -> int:
    """Run SimpleITK resampling validation."""

    report = validate_resampling_plan(
        manifest_path=args.manifest,
        raw_root=args.raw_root,
        report_path=args.report,
        sample_size=args.sample_size,
        case_ids=args.case_id,
    )
    summary = report["summary"]

    print(f"Wrote resampling validation report: {args.report}")
    print(f"Cases checked: {summary['cases_checked']}")
    print(f"Cases with issues: {summary['cases_with_issues']}")
    print(f"ADC resampled matches reference: {summary['adc_resampled_matches_reference']}")
    print(f"HBV resampled matches reference: {summary['hbv_resampled_matches_reference']}")
    print(f"Gland reference-grid masks: {summary['gland_cases_with_reference_grid_mask']}")
    print(f"Lesion reference-grid masks: {summary['lesion_cases_with_reference_grid_mask']}")
    return 0


def run_radiomics_extract(args: argparse.Namespace) -> int:
    """Run Stage 3 first-order radiomics extraction."""

    summary = extract_radiomics_features(
        manifest_path=args.manifest,
        raw_root=args.raw_root,
        output_path=args.output,
        failure_log_path=args.failure_log,
        settings_path=args.settings,
        preprocessing_report_path=args.preprocessing_report,
        sequence=args.sequence,
        roi=args.roi,
        sample_size=args.sample_size,
        case_ids=args.case_id,
        all_cases=args.all_cases,
    )

    print(f"Wrote radiomics feature table: {args.output}")
    print(f"Wrote radiomics failure log: {args.failure_log}")
    print(f"Wrote radiomics settings: {args.settings}")
    print(f"Cases requested: {summary['cases_requested']}")
    print(f"Features written: {summary['features_written']}")
    print(f"Failures written: {summary['failures_written']}")
    return 0


def run_embedding_extract(args: argparse.Namespace) -> int:
    """Run Stage 4 prototype embedding extraction."""

    summary = extract_embedding_table(
        manifest_path=args.manifest,
        raw_root=args.raw_root,
        output_path=args.output,
        provenance_path=args.provenance,
        report_path=args.report,
        preprocessing_report_path=args.preprocessing_report,
        sequence=args.sequence,
        embedding_dim=args.embedding_dim,
        sample_size_per_split=args.sample_size_per_split,
        case_ids=args.case_id,
        all_cases=args.all_cases,
        augment_train=args.augment_train,
    )

    print(f"Wrote embedding table: {args.output}")
    print(f"Wrote embedding provenance: {args.provenance}")
    print(f"Wrote embedding report: {args.report}")
    print(f"Embeddings written: {summary['embeddings_written']}")
    print(f"Failures: {summary['failures']}")
    print(f"Embeddings by split: {summary['embeddings_by_split']}")
    print(f"Augmentation by split: {summary['augmentation_by_split']}")
    print(f"Validation/test augmented rows: {summary['validation_or_test_augmented_rows']}")
    return 0


def run_baseline_evaluate(args: argparse.Namespace) -> int:
    """Run Stage 5 prototype feature baselines."""

    report = run_feature_baselines(
        manifest_path=args.manifest,
        radiomics_path=args.radiomics,
        embeddings_path=args.embeddings,
        metrics_path=args.metrics,
        predictions_path=args.predictions,
        report_path=args.report,
    )

    print(f"Wrote baseline metrics: {args.metrics}")
    print(f"Wrote baseline predictions: {args.predictions}")
    print(f"Wrote baseline report: {args.report}")
    print(f"Aligned cases: {report['case_counts']['aligned']}")
    print(f"Split counts: {report['split_counts']}")
    print(f"Label counts: {report['label_counts']}")
    for name, metrics in report["metrics"].items():
        print(f"{name}: {metrics.get('status')}")
    return 0


def run_evaluation_report(args: argparse.Namespace) -> int:
    """Run Stage 6 evaluation report generation."""

    report = generate_evaluation_report(
        predictions_path=args.predictions,
        report_json_path=args.json_report,
        report_markdown_path=args.markdown_report,
        target_sensitivity=args.target_sensitivity,
    )

    print(f"Wrote evaluation JSON report: {args.json_report}")
    print(f"Wrote evaluation Markdown report: {args.markdown_report}")
    print(f"Prediction rows summarized: {report['total_prediction_rows']}")
    for baseline, splits in report["baselines"].items():
        print(f"{baseline}: {sorted(splits)}")
    return 0


def run_explainability_report(args: argparse.Namespace) -> int:
    """Run Stage 7 prototype explainability report generation."""

    report = generate_explainability_report(
        manifest_path=args.manifest,
        radiomics_path=args.radiomics,
        embeddings_path=args.embeddings,
        output_json_path=args.json_report,
        output_csv_path=args.csv_report,
        top_n=args.top_n,
    )

    print(f"Wrote explainability JSON report: {args.json_report}")
    print(f"Wrote feature-importance CSV: {args.csv_report}")
    print(f"Aligned cases: {report['aligned_cases']}")
    print(f"CNN visual explanation: {report['cnn_visual_explanation']['status']}")
    for baseline, payload in report["importances"].items():
        print(f"{baseline}: {payload['status']} top_features={len(payload['top_features'])}")
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
