"""Command-line entrypoint for prostate MRI cancer detection workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from prostate_mri_cancer_detection.cnn import run_cnn_smoke_training
from prostate_mri_cancer_detection.data import build_and_write_manifest
from prostate_mri_cancer_detection.evaluation import (
    generate_evaluation_report,
    run_feature_baselines,
    run_radiomics_cv_baseline,
    run_radiomics_ml_baseline,
)
from prostate_mri_cancer_detection.explainability import generate_explainability_report
from prostate_mri_cancer_detection.features import (
    extract_full_gland_multisequence_radiomics,
    extract_radiomics_features,
)
from prostate_mri_cancer_detection.modeling import extract_embedding_table
from prostate_mri_cancer_detection.preprocessing import (
    validate_preprocessing_inputs,
    validate_resampling_plan,
    write_preprocessed_sample,
)


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

    sample_parser = subparsers.add_parser(
        "preprocess-sample",
        help="Write a tiny SimpleITK ADC/HBV-to-T2W processed sample for inspection.",
    )
    sample_parser.add_argument(
        "--manifest",
        default="data/interim/picai_manifest.csv",
        type=Path,
        help="Stage 1 manifest CSV path.",
    )
    sample_parser.add_argument(
        "--raw-root",
        default="data/raw/picai",
        type=Path,
        help="Path to the local PI-CAI raw root.",
    )
    sample_parser.add_argument(
        "--output-root",
        default="data/processed/picai_sample",
        type=Path,
        help="Ignored output root for tiny processed sample.",
    )
    sample_parser.add_argument(
        "--report",
        default="outputs/reports/preprocessed_sample_report.json",
        type=Path,
        help="JSON report/provenance path.",
    )
    sample_parser.add_argument(
        "--sample-size",
        default=5,
        type=int,
        help="Number of sorted manifest cases to write when --case-id is not used.",
    )
    sample_parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Specific case ID to write. Can be provided multiple times.",
    )
    sample_parser.set_defaults(func=run_preprocess_sample)

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

    full_radiomics_parser = subparsers.add_parser(
        "radiomics-full-gland",
        help="Extract full whole-gland T2W + resampled ADC/HBV first-order radiomics.",
    )
    full_radiomics_parser.add_argument(
        "--manifest",
        default="data/interim/picai_manifest.csv",
        type=Path,
        help="Stage 1 manifest CSV path.",
    )
    full_radiomics_parser.add_argument(
        "--raw-root",
        default="data/raw/picai",
        type=Path,
        help="Path to the local PI-CAI raw root.",
    )
    full_radiomics_parser.add_argument(
        "--sample-size",
        default=25,
        type=int,
        help="Number of sorted manifest cases to extract when --all-cases is not used.",
    )
    full_radiomics_parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Specific case ID to extract. Can be provided multiple times.",
    )
    full_radiomics_parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Extract all manifest cases after sample validation succeeds.",
    )
    full_radiomics_parser.add_argument(
        "--output",
        default="data/features/radiomics_gland_multisequence_sample.csv",
        type=Path,
        help="Feature table output path.",
    )
    full_radiomics_parser.add_argument(
        "--failure-log",
        default="outputs/reports/radiomics_gland_multisequence_failures.csv",
        type=Path,
        help="Per-case extraction failure log path.",
    )
    full_radiomics_parser.add_argument(
        "--settings",
        default="outputs/reports/radiomics_gland_multisequence_settings.json",
        type=Path,
        help="Extraction settings JSON path.",
    )
    full_radiomics_parser.set_defaults(func=run_full_gland_radiomics)

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

    radiomics_ml_parser = subparsers.add_parser(
        "radiomics-ml-baseline",
        help="Run a full-table radiomics-only logistic-regression baseline.",
    )
    radiomics_ml_parser.add_argument(
        "--features",
        default="data/features/radiomics_gland_multisequence_full.csv",
        type=Path,
        help="Full multisequence whole-gland radiomics feature table.",
    )
    radiomics_ml_parser.add_argument(
        "--metrics",
        default="outputs/reports/radiomics_ml_metrics.json",
        type=Path,
        help="Metrics JSON output path.",
    )
    radiomics_ml_parser.add_argument(
        "--predictions",
        default="outputs/reports/radiomics_ml_predictions.csv",
        type=Path,
        help="Prediction CSV output path.",
    )
    radiomics_ml_parser.add_argument(
        "--report",
        default="outputs/reports/radiomics_ml_report.json",
        type=Path,
        help="Full baseline report JSON output path.",
    )
    radiomics_ml_parser.add_argument(
        "--target-sensitivity",
        default=0.90,
        type=float,
        help="Target sensitivity for fixed-sensitivity analysis.",
    )
    radiomics_ml_parser.set_defaults(func=run_radiomics_ml)

    radiomics_cv_parser = subparsers.add_parser(
        "radiomics-cv-baseline",
        help="Run a rotated-fold radiomics-only logistic-regression baseline.",
    )
    radiomics_cv_parser.add_argument(
        "--features",
        default="data/features/radiomics_gland_multisequence_full.csv",
        type=Path,
        help="Full multisequence whole-gland radiomics feature table.",
    )
    radiomics_cv_parser.add_argument(
        "--metrics",
        default="outputs/reports/radiomics_cv_metrics.json",
        type=Path,
        help="Rotated-fold metrics JSON output path.",
    )
    radiomics_cv_parser.add_argument(
        "--predictions",
        default="outputs/reports/radiomics_cv_predictions.csv",
        type=Path,
        help="Rotated-fold prediction CSV output path.",
    )
    radiomics_cv_parser.add_argument(
        "--report",
        default="outputs/reports/radiomics_cv_report.json",
        type=Path,
        help="Rotated-fold report JSON output path.",
    )
    radiomics_cv_parser.add_argument(
        "--target-sensitivity",
        default=0.90,
        type=float,
        help="Target sensitivity for validation-selected threshold analysis.",
    )
    radiomics_cv_parser.add_argument(
        "--c-value",
        action="append",
        default=[],
        type=float,
        help="Logistic-regression C value to evaluate. Can be provided multiple times.",
    )
    radiomics_cv_parser.set_defaults(func=run_radiomics_cv)

    cnn_parser = subparsers.add_parser(
        "cnn-smoke-train",
        help="Run a tiny split-safe multisequence CNN smoke training pass.",
    )
    cnn_parser.add_argument(
        "--manifest",
        default="data/interim/picai_manifest.csv",
        type=Path,
        help="Stage 1 manifest CSV path.",
    )
    cnn_parser.add_argument(
        "--raw-root",
        default="data/raw/picai",
        type=Path,
        help="Path to the local PI-CAI raw root.",
    )
    cnn_parser.add_argument(
        "--embeddings",
        default="data/features/cnn_smoke_embeddings.csv",
        type=Path,
        help="CNN smoke embedding table output path.",
    )
    cnn_parser.add_argument(
        "--predictions",
        default="outputs/reports/cnn_smoke_predictions.csv",
        type=Path,
        help="CNN smoke prediction CSV output path.",
    )
    cnn_parser.add_argument(
        "--report",
        default="outputs/reports/cnn_smoke_report.json",
        type=Path,
        help="CNN smoke report JSON output path.",
    )
    cnn_parser.add_argument(
        "--model",
        default="outputs/models/cnn_smoke_model.pt",
        type=Path,
        help="Ignored smoke-model checkpoint output path.",
    )
    cnn_parser.add_argument(
        "--sample-size-per-split",
        default=12,
        type=int,
        help="Balanced cases per split for smoke training when --all-cases is not used.",
    )
    cnn_parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Specific case ID to include. Can be provided multiple times.",
    )
    cnn_parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Use all manifest cases. Intended for later training, not the first smoke run.",
    )
    cnn_parser.add_argument(
        "--image-size",
        default=64,
        type=int,
        help="Square 2D crop size for CNN input.",
    )
    cnn_parser.add_argument(
        "--max-epochs",
        default=1,
        type=int,
        help="Number of smoke-training epochs.",
    )
    cnn_parser.add_argument(
        "--batch-size",
        default=4,
        type=int,
        help="Mini-batch size.",
    )
    cnn_parser.add_argument(
        "--learning-rate",
        default=1e-3,
        type=float,
        help="Adam learning rate.",
    )
    cnn_parser.add_argument(
        "--embedding-dim",
        default=32,
        type=int,
        help="CNN embedding dimension.",
    )
    cnn_parser.add_argument(
        "--augment-train",
        action="store_true",
        help="Apply deterministic augmentation to training rows only.",
    )
    cnn_parser.add_argument(
        "--target-sensitivity",
        default=0.90,
        type=float,
        help="Target sensitivity for report diagnostics.",
    )
    cnn_parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed.",
    )
    cnn_parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, for example cpu, cuda, or auto.",
    )
    cnn_parser.set_defaults(func=run_cnn_smoke)

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


def run_preprocess_sample(args: argparse.Namespace) -> int:
    """Write a tiny processed sample for inspection."""

    report = write_preprocessed_sample(
        manifest_path=args.manifest,
        raw_root=args.raw_root,
        output_root=args.output_root,
        report_path=args.report,
        sample_size=args.sample_size,
        case_ids=args.case_id,
    )
    summary = report["summary"]

    print(f"Wrote preprocessed sample report: {args.report}")
    print(f"Output root: {args.output_root}")
    print(f"Cases requested: {summary['cases_requested']}")
    print(f"Cases with issues: {summary['cases_with_issues']}")
    print(f"ADC written/matched: {summary['adc_written']}/{summary['adc_matches_reference']}")
    print(f"HBV written/matched: {summary['hbv_written']}/{summary['hbv_matches_reference']}")
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


def run_full_gland_radiomics(args: argparse.Namespace) -> int:
    """Run full whole-gland multisequence radiomics extraction."""

    summary = extract_full_gland_multisequence_radiomics(
        manifest_path=args.manifest,
        raw_root=args.raw_root,
        output_path=args.output,
        failure_log_path=args.failure_log,
        settings_path=args.settings,
        sample_size=args.sample_size,
        case_ids=args.case_id,
        all_cases=args.all_cases,
    )

    print(f"Wrote full gland radiomics table: {args.output}")
    print(f"Wrote full gland failure log: {args.failure_log}")
    print(f"Wrote full gland settings: {args.settings}")
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


def run_radiomics_ml(args: argparse.Namespace) -> int:
    """Run full-table radiomics-only ML baseline."""

    report = run_radiomics_ml_baseline(
        features_path=args.features,
        metrics_path=args.metrics,
        predictions_path=args.predictions,
        report_path=args.report,
        target_sensitivity=args.target_sensitivity,
    )

    print(f"Wrote radiomics ML metrics: {args.metrics}")
    print(f"Wrote radiomics ML predictions: {args.predictions}")
    print(f"Wrote radiomics ML report: {args.report}")
    print(f"Cases: {report['case_counts']}")
    print(f"Labels: {report['label_counts']}")
    for split, payload in report["metrics"].items():
        metrics = payload["metrics"]
        print(f"{split}: n={metrics['n']} auc={metrics['roc_auc']} sens={metrics['sensitivity']} spec={metrics['specificity']}")
    return 0


def run_radiomics_cv(args: argparse.Namespace) -> int:
    """Run rotated-fold radiomics-only ML baseline."""

    report = run_radiomics_cv_baseline(
        features_path=args.features,
        metrics_path=args.metrics,
        predictions_path=args.predictions,
        report_path=args.report,
        target_sensitivity=args.target_sensitivity,
        c_values=args.c_value or None,
    )

    print(f"Wrote radiomics CV metrics: {args.metrics}")
    print(f"Wrote radiomics CV predictions: {args.predictions}")
    print(f"Wrote radiomics CV report: {args.report}")
    print(f"Cases: {report['case_counts']}")
    print(f"Labels: {report['label_counts']}")
    print(f"Fold order: {report['fold_order']}")
    default_metrics = report["aggregate"]["pooled_test_default"]["metrics"]
    fixed = report["aggregate"]["validation_selected_fixed_sensitivity"]
    print(
        "Pooled held-out default: "
        f"n={default_metrics['n']} auc={default_metrics['roc_auc']} "
        f"sens={default_metrics['sensitivity']} spec={default_metrics['specificity']}"
    )
    print(
        "Validation-selected fixed sensitivity: "
        f"status={fixed['status']} metrics={fixed.get('metrics')}"
    )
    return 0


def run_cnn_smoke(args: argparse.Namespace) -> int:
    """Run CNN smoke training."""

    report = run_cnn_smoke_training(
        manifest_path=args.manifest,
        raw_root=args.raw_root,
        embeddings_path=args.embeddings,
        predictions_path=args.predictions,
        report_path=args.report,
        model_path=args.model,
        sample_size_per_split=args.sample_size_per_split,
        image_size=args.image_size,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        augment_train=args.augment_train,
        all_cases=args.all_cases,
        case_ids=args.case_id,
        target_sensitivity=args.target_sensitivity,
        seed=args.seed,
        device_name=args.device,
    )

    print(f"Wrote CNN smoke embeddings: {args.embeddings}")
    print(f"Wrote CNN smoke predictions: {args.predictions}")
    print(f"Wrote CNN smoke report: {args.report}")
    print(f"Wrote CNN smoke model: {args.model}")
    print(f"Summary: {report['summary']}")
    print(f"Case counts: {report['case_counts']}")
    print(f"Label counts: {report['label_counts']}")
    for split, payload in report["metrics"].items():
        metrics = payload["metrics"]
        print(f"{split}: n={metrics['n']} auc={metrics['roc_auc']} sens={metrics['sensitivity']} spec={metrics['specificity']}")
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
