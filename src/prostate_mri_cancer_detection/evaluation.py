"""Stage 5 prototype feature fusion and baseline evaluation."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

SPLIT_BY_FOLD = {
    "fold0": "train",
    "fold1": "train",
    "fold2": "train",
    "fold3": "validation",
    "fold4": "test",
}

NON_FEATURE_COLUMNS = {
    "case_id",
    "patient_id",
    "study_id",
    "fold",
    "split",
    "label",
    "label_cspca",
    "sequence",
    "roi",
    "image_path",
    "mask_path",
    "encoder_name",
    "encoder_type",
    "augmentation_applied",
}


def run_feature_baselines(
    manifest_path: str | Path,
    radiomics_path: str | Path,
    embeddings_path: str | Path,
    metrics_path: str | Path,
    predictions_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Run aligned radiomics, embedding, and hybrid prototype baselines."""

    manifest_rows = load_csv(manifest_path)
    radiomics_rows = load_csv(radiomics_path)
    embedding_rows = load_csv(embeddings_path)

    labels = {
        row["case_id"]: parse_label(row.get("label_cspca", ""))
        for row in manifest_rows
        if row.get("case_id")
    }
    folds = {
        row["case_id"]: row.get("fold", "")
        for row in manifest_rows
        if row.get("case_id")
    }
    radiomics_by_case = {row["case_id"]: row for row in radiomics_rows if row.get("case_id")}
    embeddings_by_case = {row["case_id"]: row for row in embedding_rows if row.get("case_id")}
    common_case_ids = sorted(set(labels) & set(radiomics_by_case) & set(embeddings_by_case))

    radiomics_feature_columns = numeric_feature_columns(radiomics_rows)
    embedding_feature_columns = numeric_feature_columns(embedding_rows)

    aligned_rows = []
    excluded_cases: list[dict[str, str]] = []
    for case_id in common_case_ids:
        label = labels.get(case_id)
        if label is None:
            excluded_cases.append({"case_id": case_id, "reason": "missing_or_unknown_label"})
            continue
        fold = folds.get(case_id, radiomics_by_case[case_id].get("fold", ""))
        split = embeddings_by_case[case_id].get("split") or SPLIT_BY_FOLD.get(fold, "unknown")
        aligned_rows.append(
            {
                "case_id": case_id,
                "fold": fold,
                "split": split,
                "label": label,
                "radiomics": feature_vector(radiomics_by_case[case_id], radiomics_feature_columns),
                "embedding": feature_vector(embeddings_by_case[case_id], embedding_feature_columns),
            }
        )

    baselines = {
        "radiomics_only": radiomics_feature_columns,
        "prototype_embedding_only": embedding_feature_columns,
        "hybrid_radiomics_embedding": radiomics_feature_columns + embedding_feature_columns,
    }

    all_predictions: list[dict[str, str]] = []
    metrics: dict[str, Any] = {}
    for baseline_name in baselines:
        result = evaluate_baseline(baseline_name, aligned_rows)
        metrics[baseline_name] = result["metrics"]
        all_predictions.extend(result["predictions"])

    report = {
        "schema_version": "1.0",
        "stage": "feature_fusion_and_prototype_baselines",
        "manifest_path": str(manifest_path),
        "radiomics_path": str(radiomics_path),
        "embeddings_path": str(embeddings_path),
        "case_counts": {
            "manifest": len(manifest_rows),
            "radiomics": len(radiomics_rows),
            "embeddings": len(embedding_rows),
            "aligned": len(aligned_rows),
            "excluded": len(excluded_cases),
        },
        "label_counts": dict(Counter(str(row["label"]) for row in aligned_rows)),
        "split_counts": dict(Counter(row["split"] for row in aligned_rows)),
        "feature_columns": {
            "radiomics_only": radiomics_feature_columns,
            "prototype_embedding_only": embedding_feature_columns,
            "hybrid_radiomics_embedding": radiomics_feature_columns + embedding_feature_columns,
        },
        "excluded_non_feature_columns": sorted(NON_FEATURE_COLUMNS),
        "excluded_cases": excluded_cases,
        "metrics": metrics,
        "claim_limits": [
            "Prototype embeddings are not trained CNN features.",
            "These baselines are implementation checks, not final model results.",
            "No clinical, deployment, localization, or biopsy-reduction claims are supported.",
        ],
    }

    write_json(metrics_path, metrics)
    write_predictions(predictions_path, all_predictions)
    write_json(report_path, report)
    return report


def generate_evaluation_report(
    predictions_path: str | Path,
    report_json_path: str | Path,
    report_markdown_path: str | Path,
    target_sensitivity: float = 0.90,
) -> dict[str, Any]:
    """Generate a Stage 6 evaluation report from prediction rows."""

    predictions = [
        row
        for row in load_csv(predictions_path)
        if row.get("status") == "ok"
    ]
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in predictions:
        grouped.setdefault(row["baseline"], {}).setdefault(row["split"], []).append(row)

    baselines: dict[str, Any] = {}
    for baseline, split_rows in sorted(grouped.items()):
        baselines[baseline] = {}
        for split, rows in sorted(split_rows.items()):
            baselines[baseline][split] = summarize_prediction_group(
                rows,
                target_sensitivity=target_sensitivity,
            )

    report = {
        "schema_version": "1.0",
        "stage": "evaluation_and_ablation_report",
        "predictions_path": str(predictions_path),
        "target_sensitivity": target_sensitivity,
        "total_prediction_rows": len(predictions),
        "baselines": baselines,
        "ablation_status": {
            "radiomics_only": "available if present in predictions",
            "prototype_embedding_only": "prototype only; not a trained CNN baseline",
            "hybrid_radiomics_embedding": "prototype only; not a final hybrid model",
            "sequence_contribution": "not implemented until ADC/HBV resampling and extraction are available",
            "augmentation_with_vs_without": "not implemented; Stage 4 only validates leakage guards",
            "pirads_comparison": "not implemented; requires verified PI-RADS linkage and threshold policy",
        },
        "claim_limits": [
            "This report summarizes prototype prediction outputs only.",
            "Do not describe these results as final radiomics, CNN, or hybrid performance.",
            "Do not claim external validation, lesion localization, clinical deployment readiness, or biopsy reduction.",
            "Fixed-sensitivity false-positive counts are exploratory and do not support biopsy-reduction claims by themselves.",
        ],
    }

    write_json(report_json_path, report)
    write_markdown_report(report_markdown_path, report)
    return report


def run_radiomics_ml_baseline(
    features_path: str | Path,
    metrics_path: str | Path,
    predictions_path: str | Path,
    report_path: str | Path,
    target_sensitivity: float = 0.90,
) -> dict[str, Any]:
    """Run a full-table radiomics-only logistic-regression baseline."""

    try:
        import numpy as np
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - depends on cluster env.
        raise RuntimeError("scikit-learn and numpy are required for the radiomics ML baseline") from error

    rows = load_csv(features_path)
    feature_columns = numeric_feature_columns(rows)
    model_rows = [
        {
            "case_id": row["case_id"],
            "fold": row["fold"],
            "split": SPLIT_BY_FOLD.get(row["fold"], "unknown"),
            "label": parse_label(row.get("label_cspca", "")),
            "features": [float(row[column]) for column in feature_columns],
        }
        for row in rows
        if parse_label(row.get("label_cspca", "")) is not None
    ]
    train_rows = [row for row in model_rows if row["split"] == "train"]
    if {row["label"] for row in train_rows} != {0, 1}:
        raise ValueError("training split must contain positive and negative cases")

    x_train = np.asarray([row["features"] for row in train_rows], dtype=float)
    y_train = np.asarray([row["label"] for row in train_rows], dtype=int)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    prediction_rows: list[dict[str, str]] = []
    for row in model_rows:
        x_row = np.asarray([row["features"]], dtype=float)
        probability = float(pipeline.predict_proba(x_row)[0, 1])
        prediction = 1 if probability >= 0.5 else 0
        prediction_rows.append(
            {
                "baseline": "radiomics_logistic_regression",
                "case_id": row["case_id"],
                "fold": row["fold"],
                "split": row["split"],
                "label": str(row["label"]),
                "score": format_float(probability),
                "probability": format_float(probability),
                "prediction": str(prediction),
                "status": "ok",
                "reason": "",
            }
        )

    split_metrics = {}
    for split in ("train", "validation", "test"):
        split_predictions = [row for row in prediction_rows if row["split"] == split]
        split_metrics[split] = summarize_prediction_group(
            split_predictions,
            target_sensitivity=target_sensitivity,
        )
    coefficients = pipeline.named_steps["model"].coef_[0]
    scaled_feature_importance = sorted(
        [
            {
                "feature": feature,
                "coefficient": float(coefficient),
                "abs_coefficient": abs(float(coefficient)),
            }
            for feature, coefficient in zip(feature_columns, coefficients)
        ],
        key=lambda item: item["abs_coefficient"],
        reverse=True,
    )

    report = {
        "schema_version": "1.0",
        "stage": "full_radiomics_only_ml_baseline",
        "features_path": str(features_path),
        "feature_count": len(feature_columns),
        "case_counts": {
            "total": len(model_rows),
            "train": sum(1 for row in model_rows if row["split"] == "train"),
            "validation": sum(1 for row in model_rows if row["split"] == "validation"),
            "test": sum(1 for row in model_rows if row["split"] == "test"),
        },
        "label_counts": dict(Counter(str(row["label"]) for row in model_rows)),
        "split_label_counts": {
            split: dict(Counter(str(row["label"]) for row in model_rows if row["split"] == split))
            for split in ("train", "validation", "test")
        },
        "model": {
            "name": "LogisticRegression",
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 5000,
            "random_state": 42,
            "preprocessing": ["median imputation", "standard scaling"],
        },
        "metrics": split_metrics,
        "top_coefficients": scaled_feature_importance[:25],
        "excluded_non_feature_columns": sorted(NON_FEATURE_COLUMNS),
        "claim_limits": [
            "This is an internal full-table radiomics-only baseline.",
            "No CNN or hybrid performance claim is supported by this run.",
            "No external validation, clinical deployment, lesion localization, or biopsy-reduction claim is supported.",
        ],
    }
    write_json(metrics_path, split_metrics)
    write_predictions(predictions_path, prediction_rows)
    write_json(report_path, report)

    try:
        report["sklearn_train_auc"] = float(
            roc_auc_score(y_train, pipeline.predict_proba(x_train)[:, 1])
        )
    except ValueError:
        report["sklearn_train_auc"] = None
    return report


def run_radiomics_cv_baseline(
    features_path: str | Path,
    metrics_path: str | Path,
    predictions_path: str | Path,
    report_path: str | Path,
    target_sensitivity: float = 0.90,
    c_values: list[float] | None = None,
) -> dict[str, Any]:
    """Run rotated-fold radiomics-only logistic-regression baselines."""

    try:
        import numpy as np
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - depends on cluster env.
        raise RuntimeError("scikit-learn and numpy are required for the radiomics CV baseline") from error

    c_grid = c_values or [0.01, 0.1, 1.0, 10.0]
    rows = load_csv(features_path)
    feature_columns = numeric_feature_columns(rows)
    model_rows = prepare_radiomics_model_rows(rows, feature_columns)
    fold_order = sorted({row["fold"] for row in model_rows})
    if len(fold_order) < 3:
        raise ValueError("rotated-fold evaluation requires at least three folds")

    prediction_rows: list[dict[str, str]] = []
    fold_reports: dict[str, Any] = {}
    coefficient_rows: list[dict[str, float]] = []
    fixed_test_predictions: list[dict[str, Any]] = []

    for index, test_fold in enumerate(fold_order):
        validation_fold = fold_order[(index + 1) % len(fold_order)]
        train_folds = [fold for fold in fold_order if fold not in {test_fold, validation_fold}]
        split_rows = {
            "train": [row for row in model_rows if row["fold"] in train_folds],
            "validation": [row for row in model_rows if row["fold"] == validation_fold],
            "test": [row for row in model_rows if row["fold"] == test_fold],
        }
        if {row["label"] for row in split_rows["train"]} != {0, 1}:
            raise ValueError(f"training folds for {test_fold} must contain positive and negative cases")

        selected = select_logistic_c(
            split_rows=split_rows,
            c_grid=c_grid,
            np_module=np,
            SimpleImputer=SimpleImputer,
            StandardScaler=StandardScaler,
            LogisticRegression=LogisticRegression,
            Pipeline=Pipeline,
        )
        pipeline = selected["pipeline"]
        fold_prediction_rows: list[dict[str, str]] = []
        for split, rows_for_split in split_rows.items():
            for row in rows_for_split:
                probability = radiomics_probability(np, pipeline, row)
                prediction = 1 if probability >= 0.5 else 0
                fold_prediction_rows.append(
                    {
                        "baseline": "radiomics_logistic_regression_cv",
                        "case_id": row["case_id"],
                        "fold": row["fold"],
                        "split": split,
                        "label": str(row["label"]),
                        "score": format_float(probability),
                        "probability": format_float(probability),
                        "prediction": str(prediction),
                        "status": "ok",
                        "reason": "",
                        "rotation_test_fold": test_fold,
                        "rotation_validation_fold": validation_fold,
                        "rotation_train_folds": ";".join(train_folds),
                        "selected_c": format_float(selected["c"]),
                    }
                )
        prediction_rows.extend(fold_prediction_rows)

        validation_rows = [row for row in fold_prediction_rows if row["split"] == "validation"]
        test_rows = [row for row in fold_prediction_rows if row["split"] == "test"]
        validation_threshold = fixed_sensitivity_analysis(
            rows=validation_rows,
            labels=[int(row["label"]) for row in validation_rows],
            probabilities=[float(row["probability"]) for row in validation_rows],
            target_sensitivity=target_sensitivity,
        )
        test_fixed = apply_threshold_from_validation(
            rows=test_rows,
            threshold_report=validation_threshold,
            target_sensitivity=target_sensitivity,
        )
        if test_fixed["status"] == "ok":
            fixed_test_predictions.extend(test_fixed["thresholded_rows"])

        coefficients = pipeline.named_steps["model"].coef_[0]
        for feature, coefficient in zip(feature_columns, coefficients):
            coefficient_rows.append(
                {
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "abs_coefficient": abs(float(coefficient)),
                }
            )

        fold_reports[test_fold] = {
            "test_fold": test_fold,
            "validation_fold": validation_fold,
            "train_folds": train_folds,
            "selected_c": selected["c"],
            "c_grid": c_grid,
            "validation_selection": selected["validation_scores"],
            "split_counts": {split: len(rows_for_split) for split, rows_for_split in split_rows.items()},
            "split_label_counts": {
                split: dict(Counter(str(row["label"]) for row in rows_for_split))
                for split, rows_for_split in split_rows.items()
            },
            "metrics": {
                split: summarize_prediction_group(
                    [row for row in fold_prediction_rows if row["split"] == split],
                    target_sensitivity=target_sensitivity,
                )
                for split in ("train", "validation", "test")
            },
            "validation_selected_fixed_sensitivity": {
                "validation": validation_threshold,
                "test": strip_thresholded_rows(test_fixed),
            },
        }

    aggregate = aggregate_cv_predictions(
        prediction_rows=prediction_rows,
        fixed_test_predictions=fixed_test_predictions,
        target_sensitivity=target_sensitivity,
    )
    report = {
        "schema_version": "1.0",
        "stage": "rotated_fold_radiomics_only_ml_baseline",
        "features_path": str(features_path),
        "feature_count": len(feature_columns),
        "fold_order": fold_order,
        "case_counts": {
            "total": len(model_rows),
            "folds": dict(Counter(row["fold"] for row in model_rows)),
        },
        "label_counts": dict(Counter(str(row["label"]) for row in model_rows)),
        "model": {
            "name": "LogisticRegression",
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 5000,
            "random_state": 42,
            "preprocessing": ["median imputation", "standard scaling"],
            "hyperparameter_selection": "C selected by validation ROC-AUC within each rotated fold",
            "c_grid": c_grid,
        },
        "rotation_policy": {
            "test_fold": "each PI-CAI fold is held out as test once",
            "validation_fold": "the next fold in sorted order is used for hyperparameter and threshold selection",
            "train_folds": "the remaining three folds",
        },
        "folds": fold_reports,
        "aggregate": aggregate,
        "top_coefficients": aggregate_coefficients(coefficient_rows)[:25],
        "excluded_non_feature_columns": sorted(NON_FEATURE_COLUMNS),
        "claim_limits": [
            "This is an internal rotated-fold radiomics-only baseline.",
            "Validation folds select model hyperparameters and fixed-sensitivity thresholds; test folds remain held out per rotation.",
            "No CNN or hybrid performance claim is supported by this run.",
            "No external validation, clinical deployment, lesion localization, or biopsy-reduction claim is supported.",
        ],
    }

    write_json(metrics_path, {"folds": fold_reports, "aggregate": aggregate})
    write_predictions(predictions_path, prediction_rows)
    write_json(report_path, report)
    return report


def run_hybrid_ml_baseline(
    radiomics_path: str | Path,
    embeddings_path: str | Path,
    metrics_path: str | Path,
    predictions_path: str | Path,
    report_path: str | Path,
    target_sensitivity: float = 0.90,
    c_values: list[float] | None = None,
) -> dict[str, Any]:
    """Run aligned radiomics-only, CNN-only, and hybrid ML baselines."""

    try:
        import numpy as np
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - depends on cluster env.
        raise RuntimeError("scikit-learn and numpy are required for the hybrid ML baseline") from error

    c_grid = c_values or [0.01, 0.1, 1.0, 10.0]
    radiomics_rows = load_csv(radiomics_path)
    embedding_rows = load_csv(embeddings_path)
    radiomics_feature_columns = numeric_feature_columns(radiomics_rows)
    embedding_feature_columns = numeric_feature_columns(embedding_rows)
    aligned_rows, excluded_cases = align_radiomics_and_embeddings(
        radiomics_rows=radiomics_rows,
        embedding_rows=embedding_rows,
        radiomics_feature_columns=radiomics_feature_columns,
        embedding_feature_columns=embedding_feature_columns,
    )
    baselines = {
        "radiomics_only": {
            "feature_names": [f"radiomics:{name}" for name in radiomics_feature_columns],
            "vector_key": "radiomics",
        },
        "cnn_embedding_only": {
            "feature_names": [f"cnn:{name}" for name in embedding_feature_columns],
            "vector_key": "cnn",
        },
        "hybrid_radiomics_cnn": {
            "feature_names": [f"radiomics:{name}" for name in radiomics_feature_columns]
            + [f"cnn:{name}" for name in embedding_feature_columns],
            "vector_key": "hybrid",
        },
    }

    all_predictions: list[dict[str, str]] = []
    baseline_reports: dict[str, Any] = {}
    top_coefficients: dict[str, list[dict[str, Any]]] = {}
    for baseline_name, config in baselines.items():
        result = train_aligned_logistic_baseline(
            baseline_name=baseline_name,
            aligned_rows=aligned_rows,
            vector_key=str(config["vector_key"]),
            feature_names=list(config["feature_names"]),
            target_sensitivity=target_sensitivity,
            c_grid=c_grid,
            np_module=np,
            SimpleImputer=SimpleImputer,
            StandardScaler=StandardScaler,
            LogisticRegression=LogisticRegression,
            Pipeline=Pipeline,
        )
        baseline_reports[baseline_name] = result["report"]
        top_coefficients[baseline_name] = result["top_coefficients"]
        all_predictions.extend(result["predictions"])
    prediction_groups = {
        baseline: [row for row in all_predictions if row["baseline"] == baseline and row["split"] == "test"]
        for baseline in baselines
    }

    report = {
        "schema_version": "1.0",
        "stage": "hybrid_radiomics_cnn_ml_baseline",
        "radiomics_path": str(radiomics_path),
        "embeddings_path": str(embeddings_path),
        "case_counts": {
            "radiomics": len(radiomics_rows),
            "embeddings": len(embedding_rows),
            "aligned": len(aligned_rows),
            "excluded": len(excluded_cases),
        },
        "split_counts": dict(Counter(row["split"] for row in aligned_rows)),
        "label_counts": dict(Counter(str(row["label"]) for row in aligned_rows)),
        "split_label_counts": {
            split: dict(Counter(str(row["label"]) for row in aligned_rows if row["split"] == split))
            for split in ("train", "validation", "test")
        },
        "feature_counts": {
            "radiomics_only": len(radiomics_feature_columns),
            "cnn_embedding_only": len(embedding_feature_columns),
            "hybrid_radiomics_cnn": len(radiomics_feature_columns) + len(embedding_feature_columns),
        },
        "model": {
            "name": "LogisticRegression",
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 5000,
            "random_state": 42,
            "preprocessing": ["median imputation", "standard scaling"],
            "hyperparameter_selection": "C selected by validation ROC-AUC for each representation",
            "c_grid": c_grid,
        },
        "baselines": baseline_reports,
        "paired_test_auc_deltas": {
            "hybrid_minus_radiomics": paired_auc_delta_ci(
                prediction_groups["hybrid_radiomics_cnn"],
                prediction_groups["radiomics_only"],
            ),
            "hybrid_minus_cnn": paired_auc_delta_ci(
                prediction_groups["hybrid_radiomics_cnn"],
                prediction_groups["cnn_embedding_only"],
            ),
            "cnn_minus_radiomics": paired_auc_delta_ci(
                prediction_groups["cnn_embedding_only"],
                prediction_groups["radiomics_only"],
            ),
        },
        "top_coefficients": top_coefficients,
        "excluded_cases": excluded_cases,
        "excluded_non_feature_columns": sorted(NON_FEATURE_COLUMNS),
        "claim_limits": [
            "This is an internal aligned-subset comparison using CNN embeddings from the current CNN baseline run.",
            "Radiomics-only, CNN-only, and hybrid rows use the same case IDs and split assignments.",
            "No external validation, clinical deployment, lesion localization, or biopsy-reduction claim is supported.",
        ],
    }

    write_json(metrics_path, baseline_reports)
    write_predictions(predictions_path, all_predictions)
    write_json(report_path, report)
    return report


def run_calibrated_fusion_baseline(
    radiomics_path: str | Path,
    cnn_predictions_path: str | Path,
    metrics_path: str | Path,
    predictions_path: str | Path,
    report_path: str | Path,
    target_sensitivity: float = 0.90,
    cnn_baseline_name: str = "cnn_smoke_multisequence",
    alpha_grid: list[float] | None = None,
    c_grid: list[float] | None = None,
) -> dict[str, Any]:
    """Run calibrated probability-level fusion using radiomics and CNN predictions."""

    try:
        import numpy as np
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - depends on cluster env.
        raise RuntimeError("NumPy and scikit-learn are required for calibrated fusion") from error

    alpha_grid = alpha_grid or [index / 20 for index in range(21)]
    c_grid = c_grid or [0.01, 0.1, 1.0, 10.0]

    radiomics_rows = load_csv(radiomics_path)
    cnn_rows = [
        row
        for row in load_csv(cnn_predictions_path)
        if row.get("status") == "ok" and row.get("baseline", cnn_baseline_name) == cnn_baseline_name
    ]
    radiomics_feature_columns = select_probability_fusion_feature_columns(radiomics_rows)
    aligned_rows, excluded_cases = prepare_probability_fusion_rows(
        radiomics_rows=radiomics_rows,
        cnn_rows=cnn_rows,
        radiomics_feature_columns=radiomics_feature_columns,
    )
    if not aligned_rows:
        raise ValueError("no aligned radiomics and CNN prediction rows were available")

    radiomics_result = train_aligned_logistic_baseline(
        baseline_name="radiomics_only",
        aligned_rows=aligned_rows,
        vector_key="radiomics_vector",
        feature_names=radiomics_feature_columns,
        target_sensitivity=target_sensitivity,
        c_grid=c_grid,
        np_module=np,
        SimpleImputer=SimpleImputer,
        StandardScaler=StandardScaler,
        LogisticRegression=LogisticRegression,
        Pipeline=Pipeline,
    )
    radiomics_predictions = radiomics_result["predictions"]
    radiomics_probability_by_case = {
        row["case_id"]: float(row["probability"])
        for row in radiomics_predictions
        if row.get("status") == "ok"
    }

    cnn_predictions = [
        probability_prediction_row(
            baseline_name="cnn_probability_only",
            row=row,
            probability=float(row["cnn_probability"]),
        )
        for row in aligned_rows
    ]
    probability_rows = [
        {
            **row,
            "radiomics_probability": radiomics_probability_by_case[row["case_id"]],
        }
        for row in aligned_rows
        if row["case_id"] in radiomics_probability_by_case
    ]

    weighted_result = run_weighted_probability_fusion(
        rows=probability_rows,
        alpha_grid=alpha_grid,
        target_sensitivity=target_sensitivity,
    )
    for row in probability_rows:
        row["stacking_vector"] = [row["radiomics_probability"], row["cnn_probability"]]
    stacked_result = train_aligned_logistic_baseline(
        baseline_name="stacked_probability_fusion",
        aligned_rows=probability_rows,
        vector_key="stacking_vector",
        feature_names=["radiomics_probability", "cnn_probability"],
        target_sensitivity=target_sensitivity,
        c_grid=c_grid,
        np_module=np,
        SimpleImputer=SimpleImputer,
        StandardScaler=StandardScaler,
        LogisticRegression=LogisticRegression,
        Pipeline=Pipeline,
    )

    baseline_reports = {
        "radiomics_only": radiomics_result["report"],
        "cnn_probability_only": summarize_calibrated_predictions(cnn_predictions, target_sensitivity),
        "weighted_probability_fusion": weighted_result["report"],
        "stacked_probability_fusion": stacked_result["report"],
    }
    all_predictions = (
        radiomics_predictions
        + cnn_predictions
        + weighted_result["predictions"]
        + stacked_result["predictions"]
    )
    prediction_groups = {
        baseline: [row for row in all_predictions if row["baseline"] == baseline and row["split"] == "test"]
        for baseline in baseline_reports
    }

    report = {
        "schema_version": "1.0",
        "stage": "calibrated_probability_fusion_baseline",
        "radiomics_path": str(radiomics_path),
        "cnn_predictions_path": str(cnn_predictions_path),
        "case_counts": {
            "radiomics": len(radiomics_rows),
            "cnn_predictions": len(cnn_rows),
            "aligned": len(aligned_rows),
            "excluded": len(excluded_cases),
        },
        "split_counts": dict(Counter(row["split"] for row in aligned_rows)),
        "label_counts": dict(Counter(str(row["label"]) for row in aligned_rows)),
        "feature_counts": {
            "radiomics_only": len(radiomics_feature_columns),
            "cnn_probability_only": 1,
            "weighted_probability_fusion": 2,
            "stacked_probability_fusion": 2,
        },
        "selection_policy": {
            "weighted_alpha": "selected on validation ROC-AUC only",
            "stacking_c": "selected on validation ROC-AUC only",
            "test_split": "used only for final held-out reporting",
            "alpha_grid": alpha_grid,
            "c_grid": c_grid,
        },
        "baselines": baseline_reports,
        "paired_test_auc_deltas": {
            "weighted_minus_cnn": paired_auc_delta_ci(
                prediction_groups["weighted_probability_fusion"],
                prediction_groups["cnn_probability_only"],
            ),
            "weighted_minus_radiomics": paired_auc_delta_ci(
                prediction_groups["weighted_probability_fusion"],
                prediction_groups["radiomics_only"],
            ),
            "stacked_minus_cnn": paired_auc_delta_ci(
                prediction_groups["stacked_probability_fusion"],
                prediction_groups["cnn_probability_only"],
            ),
            "stacked_minus_radiomics": paired_auc_delta_ci(
                prediction_groups["stacked_probability_fusion"],
                prediction_groups["radiomics_only"],
            ),
            "cnn_minus_radiomics": paired_auc_delta_ci(
                prediction_groups["cnn_probability_only"],
                prediction_groups["radiomics_only"],
            ),
        },
        "excluded_cases": excluded_cases,
        "claim_limits": [
            "This is an internal calibrated fusion ablation using existing radiomics and CNN predictions.",
            "Fusion weights and stacking hyperparameters are selected on validation only.",
            "No external validation, clinical deployment, lesion localization, or biopsy-reduction claim is supported.",
        ],
    }

    write_json(metrics_path, baseline_reports)
    write_predictions(predictions_path, all_predictions)
    write_json(report_path, report)
    return report


def select_probability_fusion_feature_columns(rows: list[dict[str, str]]) -> list[str]:
    """Select numeric radiomics feature columns for calibrated fusion."""

    if not rows:
        return []
    columns = []
    for column in rows[0]:
        if column in NON_FEATURE_COLUMNS or column.startswith("path_"):
            continue
        try:
            float(rows[0][column])
        except (TypeError, ValueError):
            continue
        columns.append(column)
    return columns


def prepare_probability_fusion_rows(
    radiomics_rows: list[dict[str, str]],
    cnn_rows: list[dict[str, str]],
    radiomics_feature_columns: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Align radiomics features and CNN probabilities by case_id."""

    radiomics_by_case = {
        row["case_id"]: row
        for row in radiomics_rows
        if row.get("case_id") and parse_label(row.get("label_cspca", "")) is not None
    }
    cnn_by_case = {
        row["case_id"]: row
        for row in cnn_rows
        if row.get("case_id") and row.get("probability") not in {"", None}
    }
    aligned_rows = []
    excluded_cases = []
    for case_id in sorted(set(radiomics_by_case) | set(cnn_by_case)):
        radiomics_row = radiomics_by_case.get(case_id)
        cnn_row = cnn_by_case.get(case_id)
        if radiomics_row is None or cnn_row is None:
            excluded_cases.append(
                {
                    "case_id": case_id,
                    "reason": "missing_radiomics" if radiomics_row is None else "missing_cnn_prediction",
                }
            )
            continue
        radiomics_label = parse_label(radiomics_row.get("label_cspca", ""))
        cnn_label = int(cnn_row.get("label", radiomics_label))
        if radiomics_label != cnn_label:
            excluded_cases.append({"case_id": case_id, "reason": "label_mismatch"})
            continue
        try:
            radiomics_vector = [float(radiomics_row[column]) for column in radiomics_feature_columns]
            cnn_probability = float(cnn_row["probability"])
        except (KeyError, TypeError, ValueError) as error:
            excluded_cases.append({"case_id": case_id, "reason": f"invalid_feature_or_probability:{error}"})
            continue
        aligned_rows.append(
            {
                "case_id": case_id,
                "fold": radiomics_row.get("fold", cnn_row.get("fold", "")),
                "split": cnn_row.get("split", split_for_fold(radiomics_row.get("fold", ""))),
                "label": int(radiomics_label),
                "radiomics_vector": radiomics_vector,
                "cnn_probability": cnn_probability,
            }
        )
    return aligned_rows, excluded_cases


def split_for_fold(fold: str) -> str:
    """Return the project split name for a PI-CAI fold."""

    if fold in {"fold0", "fold1", "fold2"}:
        return "train"
    if fold == "fold3":
        return "validation"
    if fold == "fold4":
        return "test"
    return "unknown"


def run_weighted_probability_fusion(
    rows: list[dict[str, Any]],
    alpha_grid: list[float],
    target_sensitivity: float,
) -> dict[str, Any]:
    """Select weighted probability fusion alpha on validation and report all splits."""

    validation_rows = [row for row in rows if row["split"] == "validation"]
    selection = []
    best_alpha = None
    best_key = (-1.0, float("-inf"))
    for alpha in alpha_grid:
        predictions = [
            probability_prediction_row(
                baseline_name="weighted_probability_fusion",
                row=row,
                probability=weighted_probability(row, alpha),
            )
            for row in validation_rows
        ]
        metrics = classification_metrics(predictions)
        auc = metrics.get("roc_auc")
        brier = calibration_diagnostics(predictions).get("brier_score")
        key = (
            auc if auc is not None else -1.0,
            -brier if brier is not None else float("-inf"),
        )
        selection.append({"alpha": alpha, "roc_auc": auc, "brier_score": brier})
        if key > best_key:
            best_key = key
            best_alpha = alpha
    if best_alpha is None:
        raise ValueError("weighted fusion alpha selection failed")

    predictions = [
        probability_prediction_row(
            baseline_name="weighted_probability_fusion",
            row=row,
            probability=weighted_probability(row, best_alpha),
        )
        for row in rows
    ]
    report = summarize_calibrated_predictions(predictions, target_sensitivity)
    report["selected_alpha"] = best_alpha
    report["validation_selection"] = selection
    return {"report": report, "predictions": predictions}


def weighted_probability(row: dict[str, Any], alpha: float) -> float:
    """Return alpha-weighted CNN/radiomics probability."""

    return alpha * float(row["cnn_probability"]) + (1 - alpha) * float(row["radiomics_probability"])


def probability_prediction_row(
    baseline_name: str,
    row: dict[str, Any],
    probability: float,
) -> dict[str, str]:
    """Create a prediction row from a calibrated probability."""

    prediction = 1 if probability >= 0.5 else 0
    return {
        "baseline": baseline_name,
        "case_id": row["case_id"],
        "fold": row.get("fold", ""),
        "split": row.get("split", ""),
        "label": str(row["label"]),
        "score": format_probability_value(probability),
        "probability": format_probability_value(probability),
        "prediction": str(prediction),
        "status": "ok",
        "reason": "",
    }


def format_probability_value(value: float) -> str:
    """Format a probability for stable CSV output."""

    return f"{float(value):.10g}"


def summarize_calibrated_predictions(
    predictions: list[dict[str, str]],
    target_sensitivity: float,
) -> dict[str, Any]:
    """Summarize calibrated prediction rows with validation-selected threshold."""

    validation_rows = [row for row in predictions if row["split"] == "validation"]
    test_rows = [row for row in predictions if row["split"] == "test"]
    validation_threshold = fixed_sensitivity_analysis(
        rows=validation_rows,
        labels=[int(row["label"]) for row in validation_rows],
        probabilities=[float(row["probability"]) for row in validation_rows],
        target_sensitivity=target_sensitivity,
    )
    test_fixed = apply_validation_threshold_to_prediction_rows(
        rows=test_rows,
        validation_threshold_report=validation_threshold,
        target_sensitivity=target_sensitivity,
    )
    return {
        "status": "ok",
        "split_counts": dict(Counter(row["split"] for row in predictions)),
        "metrics": {
            split: summarize_prediction_group(
                [row for row in predictions if row["split"] == split],
                target_sensitivity=target_sensitivity,
            )
            for split in ("train", "validation", "test")
        },
        "test_bootstrap_ci": bootstrap_metrics_ci(test_rows),
        "test_calibration": calibration_diagnostics(test_rows),
        "validation_selected_threshold": {
            "validation": validation_threshold,
            "test": strip_thresholded_rows(test_fixed),
        },
    }


def apply_validation_threshold_to_prediction_rows(
    rows: list[dict[str, str]],
    validation_threshold_report: dict[str, Any],
    target_sensitivity: float,
) -> dict[str, Any]:
    """Apply a validation-selected threshold to prediction rows."""

    if validation_threshold_report.get("status") != "ok":
        return {
            "status": "undefined",
            "target_sensitivity": target_sensitivity,
            "reason": "validation threshold was not available",
            "validation_threshold_status": validation_threshold_report.get("status"),
        }
    threshold = float(validation_threshold_report["threshold"])
    labels = [int(row["label"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    predicted = [1 if probability >= threshold else 0 for probability in probabilities]
    false_positives = [
        row["case_id"]
        for row, label, prediction in zip(rows, labels, predicted)
        if label == 0 and prediction == 1
    ]
    false_negatives = [
        row["case_id"]
        for row, label, prediction in zip(rows, labels, predicted)
        if label == 1 and prediction == 0
    ]
    return {
        "status": "ok",
        "target_sensitivity": target_sensitivity,
        "threshold_source": "validation",
        "threshold": threshold,
        "metrics": threshold_metrics(labels, predicted),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def generate_model_comparison_report(
    radiomics_cv_report_path: str | Path,
    cnn_report_path: str | Path,
    hybrid_report_path: str | Path,
    output_json_path: str | Path,
    output_markdown_path: str | Path,
) -> dict[str, Any]:
    """Generate a concise current-methodology comparison report."""

    radiomics_report = read_json(radiomics_cv_report_path)
    cnn_report = read_json(cnn_report_path)
    hybrid_report = read_json(hybrid_report_path)

    report = {
        "schema_version": "1.0",
        "stage": "current_model_comparison_report",
        "inputs": {
            "radiomics_cv_report": str(radiomics_cv_report_path),
            "cnn_report": str(cnn_report_path),
            "hybrid_report": str(hybrid_report_path),
        },
        "comparisons": {
            "full_radiomics_cv": summarize_full_radiomics_cv(radiomics_report),
            "cnn_aligned_subset": summarize_cnn_report(cnn_report),
            "hybrid_aligned_subset": summarize_hybrid_report(hybrid_report),
        },
        "interpretation": {
            "current_signal": [
                "Whole-gland multisequence radiomics remains the strongest full-cohort internal reference.",
                "The 2.5D CNN embeddings show ranking signal but are not yet a tuned final CNN baseline.",
                "Hybrid radiomics + CNN embeddings modestly improve aligned-subset AUC over radiomics-only.",
            ],
            "threshold_caution": [
                "Validation-selected fixed-sensitivity thresholds did not consistently achieve target sensitivity on held-out test data.",
                "Fixed-sensitivity behavior should be reported as exploratory threshold analysis, not biopsy-reduction evidence.",
            ],
            "next_decision": (
                "The current evidence supports continuing hybrid development while documenting that "
                "the observed AUC gain is modest and internal to PI-CAI folds."
            ),
        },
        "claim_limits": [
            "All results are internal PI-CAI fold or aligned-subset evaluations.",
            "The radiomics CV result uses all available radiomics rows, while CNN and hybrid results use the CNN-aligned subset.",
            "No external validation, clinical deployment, lesion localization, radiologist replacement, or biopsy-reduction claim is supported.",
            "The report is a methodology checkpoint and should be regenerated after any final naming or model-selection cleanup.",
        ],
    }
    write_json(output_json_path, report)
    write_text(output_markdown_path, markdown_for_model_comparison(report))
    return report


def evaluate_baseline(baseline_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Train/evaluate a dependency-free nearest-centroid baseline."""

    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    train_rows = split_rows["train"]
    train_labels = {row["label"] for row in train_rows}
    if train_labels != {0, 1}:
        reason = "train_split_must_contain_both_classes"
        predictions = [
            prediction_failure_row(baseline_name, row, reason)
            for row in rows
        ]
        return {
            "metrics": {
                "status": "failed",
                "reason": reason,
                "train_label_counts": dict(Counter(str(row["label"]) for row in train_rows)),
            },
            "predictions": predictions,
        }

    train_vectors = [baseline_vector(row, baseline_name) for row in train_rows]
    standardizer = fit_standardizer(train_vectors)
    standardized_train = [apply_standardizer(vector, standardizer) for vector in train_vectors]
    centroids = {
        label: mean_vector(
            vector for vector, row in zip(standardized_train, train_rows) if row["label"] == label
        )
        for label in (0, 1)
    }

    predictions: list[dict[str, str]] = []
    metrics_by_split: dict[str, Any] = {
        "status": "ok",
        "train_label_counts": dict(Counter(str(row["label"]) for row in train_rows)),
    }
    for split, split_data in split_rows.items():
        split_predictions = [
            predict_row(baseline_name, row, standardizer, centroids)
            for row in split_data
        ]
        predictions.extend(split_predictions)
        metrics_by_split[split] = classification_metrics(split_predictions)

    return {"metrics": metrics_by_split, "predictions": predictions}


def baseline_vector(row: dict[str, Any], baseline_name: str) -> list[float]:
    """Return the feature vector for a baseline name."""

    if baseline_name == "radiomics_only":
        return row["radiomics"]
    if baseline_name == "prototype_embedding_only":
        return row["embedding"]
    if baseline_name == "hybrid_radiomics_embedding":
        return row["radiomics"] + row["embedding"]
    raise ValueError(f"unknown baseline: {baseline_name}")


def predict_row(
    baseline_name: str,
    row: dict[str, Any],
    standardizer: tuple[list[float], list[float]],
    centroids: dict[int, list[float]],
) -> dict[str, str]:
    """Predict one row with a nearest-centroid model."""

    vector = apply_standardizer(baseline_vector(row, baseline_name), standardizer)
    distance_negative = euclidean_distance(vector, centroids[0])
    distance_positive = euclidean_distance(vector, centroids[1])
    score = distance_negative - distance_positive
    probability = sigmoid(score)
    prediction = 1 if probability >= 0.5 else 0
    return {
        "baseline": baseline_name,
        "case_id": row["case_id"],
        "fold": row["fold"],
        "split": row["split"],
        "label": str(row["label"]),
        "score": format_float(score),
        "probability": format_float(probability),
        "prediction": str(prediction),
        "status": "ok",
        "reason": "",
    }


def classification_metrics(predictions: list[dict[str, str]]) -> dict[str, Any]:
    """Compute basic classification metrics for prediction rows."""

    valid_rows = [row for row in predictions if row["status"] == "ok"]
    if not valid_rows:
        return {"n": 0, "status": "no_predictions"}

    labels = [int(row["label"]) for row in valid_rows]
    predicted = [int(row["prediction"]) for row in valid_rows]
    probabilities = [float(row["probability"]) for row in valid_rows]
    tp = sum(1 for label, pred in zip(labels, predicted) if label == 1 and pred == 1)
    tn = sum(1 for label, pred in zip(labels, predicted) if label == 0 and pred == 0)
    fp = sum(1 for label, pred in zip(labels, predicted) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, predicted) if label == 1 and pred == 0)
    precision = safe_divide(tp, tp + fp)
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = f1_score(precision, sensitivity)
    return {
        "n": len(valid_rows),
        "label_counts": dict(Counter(str(label) for label in labels)),
        "accuracy": safe_divide(tp + tn, len(valid_rows)),
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc(labels, probabilities),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def summarize_prediction_group(
    rows: list[dict[str, str]],
    target_sensitivity: float,
) -> dict[str, Any]:
    """Summarize metrics, errors, and fixed-sensitivity behavior for rows."""

    default_metrics = classification_metrics(rows)
    labels = [int(row["label"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    false_positives = [
        row["case_id"]
        for row, label, prediction in zip(rows, labels, predictions)
        if label == 0 and prediction == 1
    ]
    false_negatives = [
        row["case_id"]
        for row, label, prediction in zip(rows, labels, predictions)
        if label == 1 and prediction == 0
    ]

    return {
        "default_threshold": 0.5,
        "metrics": default_metrics,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "fixed_sensitivity": fixed_sensitivity_analysis(
            rows=rows,
            labels=labels,
            probabilities=probabilities,
            target_sensitivity=target_sensitivity,
        ),
    }


def fixed_sensitivity_analysis(
    rows: list[dict[str, str]],
    labels: list[int],
    probabilities: list[float],
    target_sensitivity: float,
) -> dict[str, Any]:
    """Find the highest-specificity threshold that reaches target sensitivity."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return {
            "status": "undefined",
            "reason": "requires at least one positive and one negative case",
        }

    thresholds = sorted(set(probabilities + [0.0, 1.0]), reverse=True)
    candidates = []
    for threshold in thresholds:
        predicted = [1 if probability >= threshold else 0 for probability in probabilities]
        metrics = threshold_metrics(labels, predicted)
        if metrics["sensitivity"] is not None and metrics["sensitivity"] >= target_sensitivity:
            candidates.append((threshold, metrics, predicted))

    if not candidates:
        return {
            "status": "not_reached",
            "target_sensitivity": target_sensitivity,
        }

    threshold, metrics, predicted = max(
        candidates,
        key=lambda item: (
            item[1]["specificity"] if item[1]["specificity"] is not None else -1,
            item[0],
        ),
    )
    false_positives = [
        row["case_id"]
        for row, label, prediction in zip(rows, labels, predicted)
        if label == 0 and prediction == 1
    ]
    false_negatives = [
        row["case_id"]
        for row, label, prediction in zip(rows, labels, predicted)
        if label == 1 and prediction == 0
    ]
    return {
        "status": "ok",
        "target_sensitivity": target_sensitivity,
        "threshold": threshold,
        "metrics": metrics,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def threshold_metrics(labels: list[int], predicted: list[int]) -> dict[str, Any]:
    """Compute metrics for thresholded predictions."""

    tp = sum(1 for label, pred in zip(labels, predicted) if label == 1 and pred == 1)
    tn = sum(1 for label, pred in zip(labels, predicted) if label == 0 and pred == 0)
    fp = sum(1 for label, pred in zip(labels, predicted) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, predicted) if label == 1 and pred == 0)
    precision = safe_divide(tp, tp + fp)
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    return {
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(precision, sensitivity),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def bootstrap_metrics_ci(
    rows: list[dict[str, str]],
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute bootstrap confidence intervals for prediction metrics."""

    valid_rows = [row for row in rows if row.get("status") == "ok"]
    if len(valid_rows) < 2:
        return {"status": "undefined", "reason": "requires at least two prediction rows"}

    random = __import__("random").Random(seed)
    metric_values: dict[str, list[float]] = {
        "roc_auc": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1": [],
    }
    for _ in range(n_bootstrap):
        sample = [valid_rows[random.randrange(len(valid_rows))] for _index in range(len(valid_rows))]
        metrics = classification_metrics(sample)
        for metric_name in metric_values:
            value = metrics.get(metric_name)
            if value is not None:
                metric_values[metric_name].append(float(value))

    return {
        "status": "ok",
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "metrics": {
            metric_name: percentile_interval(values, confidence)
            for metric_name, values in metric_values.items()
        },
    }


def paired_auc_delta_ci(
    left_rows: list[dict[str, str]],
    right_rows: list[dict[str, str]],
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute paired bootstrap CI for ROC-AUC difference left minus right."""

    left_by_case = {row["case_id"]: row for row in left_rows if row.get("status") == "ok"}
    right_by_case = {row["case_id"]: row for row in right_rows if row.get("status") == "ok"}
    case_ids = sorted(set(left_by_case) & set(right_by_case))
    if len(case_ids) < 2:
        return {"status": "undefined", "reason": "requires at least two aligned prediction rows"}

    labels = [int(left_by_case[case_id]["label"]) for case_id in case_ids]
    if sum(labels) == 0 or sum(labels) == len(labels):
        return {"status": "undefined", "reason": "requires positive and negative aligned rows"}

    left_scores = [float(left_by_case[case_id]["probability"]) for case_id in case_ids]
    right_scores = [float(right_by_case[case_id]["probability"]) for case_id in case_ids]
    observed = roc_auc(labels, left_scores) - roc_auc(labels, right_scores)
    random = __import__("random").Random(seed)
    deltas = []
    for _ in range(n_bootstrap):
        indices = [random.randrange(len(case_ids)) for _index in range(len(case_ids))]
        sample_labels = [labels[index] for index in indices]
        if sum(sample_labels) == 0 or sum(sample_labels) == len(sample_labels):
            continue
        sample_left = [left_scores[index] for index in indices]
        sample_right = [right_scores[index] for index in indices]
        deltas.append(roc_auc(sample_labels, sample_left) - roc_auc(sample_labels, sample_right))

    return {
        "status": "ok",
        "n": len(case_ids),
        "observed_delta": observed,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "ci": percentile_interval(deltas, confidence),
    }


def calibration_diagnostics(
    rows: list[dict[str, str]],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Brier score and simple calibration-bin diagnostics."""

    valid_rows = [row for row in rows if row.get("status") == "ok"]
    if not valid_rows:
        return {"status": "undefined", "reason": "no prediction rows"}
    labels = [int(row["label"]) for row in valid_rows]
    probabilities = [float(row["probability"]) for row in valid_rows]
    brier = sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(valid_rows)
    bins = []
    for bin_index in range(n_bins):
        low = bin_index / n_bins
        high = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            bin_rows = [
                (label, probability)
                for label, probability in zip(labels, probabilities)
                if low <= probability <= high
            ]
        else:
            bin_rows = [
                (label, probability)
                for label, probability in zip(labels, probabilities)
                if low <= probability < high
            ]
        if not bin_rows:
            bins.append({"bin": bin_index, "low": low, "high": high, "n": 0})
            continue
        bin_labels = [item[0] for item in bin_rows]
        bin_probabilities = [item[1] for item in bin_rows]
        bins.append(
            {
                "bin": bin_index,
                "low": low,
                "high": high,
                "n": len(bin_rows),
                "mean_probability": sum(bin_probabilities) / len(bin_probabilities),
                "observed_fraction": sum(bin_labels) / len(bin_labels),
            }
        )
    return {
        "status": "ok",
        "n": len(valid_rows),
        "brier_score": brier,
        "bins": bins,
    }


def percentile_interval(values: list[float], confidence: float) -> dict[str, Any]:
    """Return percentile confidence interval for values."""

    if not values:
        return {"mean": None, "lower": None, "upper": None, "n": 0}
    sorted_values = sorted(values)
    alpha = (1 - confidence) / 2
    lower_index = min(max(int(math.floor(alpha * (len(sorted_values) - 1))), 0), len(sorted_values) - 1)
    upper_index = min(max(int(math.ceil((1 - alpha) * (len(sorted_values) - 1))), 0), len(sorted_values) - 1)
    return {
        "mean": sum(sorted_values) / len(sorted_values),
        "lower": sorted_values[lower_index],
        "upper": sorted_values[upper_index],
        "n": len(sorted_values),
    }


def prepare_radiomics_model_rows(
    rows: list[dict[str, str]],
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    """Prepare labeled radiomics rows for scikit-learn baselines."""

    model_rows = []
    for row in rows:
        label = parse_label(row.get("label_cspca", ""))
        if label is None:
            continue
        model_rows.append(
            {
                "case_id": row["case_id"],
                "fold": row["fold"],
                "label": label,
                "features": [float(row[column]) for column in feature_columns],
            }
        )
    return model_rows


def select_logistic_c(
    split_rows: dict[str, list[dict[str, Any]]],
    c_grid: list[float],
    np_module: Any,
    SimpleImputer: Any,
    StandardScaler: Any,
    LogisticRegression: Any,
    Pipeline: Any,
) -> dict[str, Any]:
    """Select logistic-regression C by validation ROC-AUC."""

    train_rows = split_rows["train"]
    validation_rows = split_rows["validation"]
    x_train = np_module.asarray([row["features"] for row in train_rows], dtype=float)
    y_train = np_module.asarray([row["label"] for row in train_rows], dtype=int)
    validation_labels = [int(row["label"]) for row in validation_rows]

    best_pipeline = None
    best_c = None
    best_key = (-1.0, float("-inf"))
    validation_scores = []
    for c_value in c_grid:
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(c_value),
                        max_iter=5000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        )
        pipeline.fit(x_train, y_train)
        probabilities = [radiomics_probability(np_module, pipeline, row) for row in validation_rows]
        validation_auc = roc_auc(validation_labels, probabilities)
        validation_scores.append(
            {
                "c": float(c_value),
                "roc_auc": validation_auc,
            }
        )
        candidate_key = (validation_auc if validation_auc is not None else -1.0, -float(c_value))
        if candidate_key > best_key:
            best_key = candidate_key
            best_c = float(c_value)
            best_pipeline = pipeline

    if best_pipeline is None or best_c is None:
        raise ValueError("could not select a logistic-regression C value")
    return {
        "c": best_c,
        "pipeline": best_pipeline,
        "validation_scores": validation_scores,
    }


def radiomics_probability(np_module: Any, pipeline: Any, row: dict[str, Any]) -> float:
    """Predict positive-class probability for one radiomics row."""

    x_row = np_module.asarray([row["features"]], dtype=float)
    return float(pipeline.predict_proba(x_row)[0, 1])


def apply_threshold_from_validation(
    rows: list[dict[str, str]],
    threshold_report: dict[str, Any],
    target_sensitivity: float,
) -> dict[str, Any]:
    """Apply a validation-selected fixed-sensitivity threshold to test rows."""

    if threshold_report.get("status") != "ok":
        return {
            "status": "undefined",
            "target_sensitivity": target_sensitivity,
            "reason": "validation threshold was not available",
            "validation_threshold_status": threshold_report.get("status"),
            "thresholded_rows": [],
        }

    threshold = float(threshold_report["threshold"])
    labels = [int(row["label"]) for row in rows]
    probabilities = [float(row["probability"]) for row in rows]
    predicted = [1 if probability >= threshold else 0 for probability in probabilities]
    false_positives = [
        row["case_id"]
        for row, label, prediction in zip(rows, labels, predicted)
        if label == 0 and prediction == 1
    ]
    false_negatives = [
        row["case_id"]
        for row, label, prediction in zip(rows, labels, predicted)
        if label == 1 and prediction == 0
    ]
    thresholded_rows = [
        {
            "case_id": row["case_id"],
            "fold": row["fold"],
            "label": int(row["label"]),
            "probability": float(row["probability"]),
            "prediction": prediction,
            "rotation_test_fold": row.get("rotation_test_fold", ""),
            "threshold": threshold,
        }
        for row, prediction in zip(rows, predicted)
    ]
    return {
        "status": "ok",
        "target_sensitivity": target_sensitivity,
        "threshold_source": "validation",
        "threshold": threshold,
        "metrics": threshold_metrics(labels, predicted),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "thresholded_rows": thresholded_rows,
    }


def strip_thresholded_rows(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove verbose per-row threshold output from nested reports."""

    return {key: value for key, value in payload.items() if key != "thresholded_rows"}


def align_radiomics_and_embeddings(
    radiomics_rows: list[dict[str, str]],
    embedding_rows: list[dict[str, str]],
    radiomics_feature_columns: list[str],
    embedding_feature_columns: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Align full radiomics rows with CNN embedding rows by case ID."""

    radiomics_by_case = {row["case_id"]: row for row in radiomics_rows if row.get("case_id")}
    embeddings_by_case = {row["case_id"]: row for row in embedding_rows if row.get("case_id")}
    aligned = []
    excluded = []
    for case_id in sorted(set(radiomics_by_case) & set(embeddings_by_case)):
        radiomics_row = radiomics_by_case[case_id]
        embedding_row = embeddings_by_case[case_id]
        radiomics_label = parse_label(radiomics_row.get("label_cspca", ""))
        embedding_label = parse_label(embedding_row.get("label_cspca", ""))
        if radiomics_label is None:
            excluded.append({"case_id": case_id, "reason": "missing_radiomics_label"})
            continue
        if embedding_label is not None and embedding_label != radiomics_label:
            excluded.append({"case_id": case_id, "reason": "label_mismatch"})
            continue
        fold = radiomics_row.get("fold") or embedding_row.get("fold", "")
        split = embedding_row.get("split") or SPLIT_BY_FOLD.get(fold, "unknown")
        radiomics_vector = [float(radiomics_row[column]) for column in radiomics_feature_columns]
        cnn_vector = [float(embedding_row[column]) for column in embedding_feature_columns]
        aligned.append(
            {
                "case_id": case_id,
                "fold": fold,
                "split": split,
                "label": radiomics_label,
                "radiomics": radiomics_vector,
                "cnn": cnn_vector,
                "hybrid": radiomics_vector + cnn_vector,
            }
        )
    return aligned, excluded


def train_aligned_logistic_baseline(
    baseline_name: str,
    aligned_rows: list[dict[str, Any]],
    vector_key: str,
    feature_names: list[str],
    target_sensitivity: float,
    c_grid: list[float],
    np_module: Any,
    SimpleImputer: Any,
    StandardScaler: Any,
    LogisticRegression: Any,
    Pipeline: Any,
) -> dict[str, Any]:
    """Train one aligned logistic-regression representation baseline."""

    split_rows = {
        split: [
            {
                "case_id": row["case_id"],
                "fold": row["fold"],
                "split": row["split"],
                "label": row["label"],
                "features": row[vector_key],
            }
            for row in aligned_rows
            if row["split"] == split
        ]
        for split in ("train", "validation", "test")
    }
    if {row["label"] for row in split_rows["train"]} != {0, 1}:
        raise ValueError(f"{baseline_name} training split must contain positive and negative cases")

    selected = select_logistic_c(
        split_rows=split_rows,
        c_grid=c_grid,
        np_module=np_module,
        SimpleImputer=SimpleImputer,
        StandardScaler=StandardScaler,
        LogisticRegression=LogisticRegression,
        Pipeline=Pipeline,
    )
    pipeline = selected["pipeline"]
    predictions = []
    for split, rows_for_split in split_rows.items():
        for row in rows_for_split:
            probability = radiomics_probability(np_module, pipeline, row)
            prediction = 1 if probability >= 0.5 else 0
            predictions.append(
                {
                    "baseline": baseline_name,
                    "case_id": row["case_id"],
                    "fold": row["fold"],
                    "split": split,
                    "label": str(row["label"]),
                    "score": format_float(probability),
                    "probability": format_float(probability),
                    "prediction": str(prediction),
                    "status": "ok",
                    "reason": "",
                    "selected_c": format_float(selected["c"]),
                }
            )

    validation_rows = [row for row in predictions if row["split"] == "validation"]
    test_rows = [row for row in predictions if row["split"] == "test"]
    validation_threshold = fixed_sensitivity_analysis(
        rows=validation_rows,
        labels=[int(row["label"]) for row in validation_rows],
        probabilities=[float(row["probability"]) for row in validation_rows],
        target_sensitivity=target_sensitivity,
    )
    test_fixed = apply_threshold_from_validation(
        rows=test_rows,
        threshold_report=validation_threshold,
        target_sensitivity=target_sensitivity,
    )

    coefficients = pipeline.named_steps["model"].coef_[0]
    top_coefficients = sorted(
        [
            {
                "feature": feature,
                "coefficient": float(coefficient),
                "abs_coefficient": abs(float(coefficient)),
            }
            for feature, coefficient in zip(feature_names, coefficients)
        ],
        key=lambda item: item["abs_coefficient"],
        reverse=True,
    )[:25]
    report = {
        "status": "ok",
        "selected_c": selected["c"],
        "validation_selection": selected["validation_scores"],
        "feature_count": len(feature_names),
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_label_counts": {
            split: dict(Counter(str(row["label"]) for row in rows))
            for split, rows in split_rows.items()
        },
        "metrics": {
            split: summarize_prediction_group(
                [row for row in predictions if row["split"] == split],
                target_sensitivity=target_sensitivity,
            )
            for split in ("train", "validation", "test")
        },
        "test_bootstrap_ci": bootstrap_metrics_ci(
            [row for row in predictions if row["split"] == "test"]
        ),
        "test_calibration": calibration_diagnostics(
            [row for row in predictions if row["split"] == "test"]
        ),
        "validation_selected_threshold": {
            "validation": validation_threshold,
            "test": strip_thresholded_rows(test_fixed),
        },
    }
    return {
        "report": report,
        "predictions": predictions,
        "top_coefficients": top_coefficients,
    }


def aggregate_cv_predictions(
    prediction_rows: list[dict[str, str]],
    fixed_test_predictions: list[dict[str, Any]],
    target_sensitivity: float,
) -> dict[str, Any]:
    """Aggregate rotated-fold predictions across held-out test folds."""

    test_rows = [row for row in prediction_rows if row["split"] == "test"]
    test_rows_by_fold: dict[str, list[dict[str, str]]] = {}
    for row in test_rows:
        test_rows_by_fold.setdefault(row["rotation_test_fold"], []).append(row)
    fold_test_metrics = {
        fold: classification_metrics(rows)
        for fold, rows in sorted(test_rows_by_fold.items())
    }

    aggregate = {
        "pooled_test_default": summarize_prediction_group(
            test_rows,
            target_sensitivity=target_sensitivity,
        ),
        "fold_test_metrics": fold_test_metrics,
        "fold_test_metric_summary": summarize_metric_distribution(fold_test_metrics),
        "validation_selected_fixed_sensitivity": aggregate_fixed_threshold_predictions(
            fixed_test_predictions=fixed_test_predictions,
            target_sensitivity=target_sensitivity,
        ),
    }
    aggregate["pooled_test_default"]["fixed_sensitivity_note"] = (
        "This threshold is selected on pooled test scores and is included only "
        "for diagnostics. Use validation_selected_fixed_sensitivity for held-out "
        "fixed-sensitivity behavior."
    )
    return aggregate


def aggregate_fixed_threshold_predictions(
    fixed_test_predictions: list[dict[str, Any]],
    target_sensitivity: float,
) -> dict[str, Any]:
    """Aggregate test predictions thresholded by validation-selected cutoffs."""

    if not fixed_test_predictions:
        return {
            "status": "undefined",
            "target_sensitivity": target_sensitivity,
            "reason": "no validation-selected thresholds were available",
        }

    labels = [int(row["label"]) for row in fixed_test_predictions]
    predicted = [int(row["prediction"]) for row in fixed_test_predictions]
    false_positives = [
        row["case_id"]
        for row, label, prediction in zip(fixed_test_predictions, labels, predicted)
        if label == 0 and prediction == 1
    ]
    false_negatives = [
        row["case_id"]
        for row, label, prediction in zip(fixed_test_predictions, labels, predicted)
        if label == 1 and prediction == 0
    ]
    return {
        "status": "ok",
        "target_sensitivity": target_sensitivity,
        "threshold_source": "validation fold per rotation",
        "n": len(fixed_test_predictions),
        "metrics": threshold_metrics(labels, predicted),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def summarize_metric_distribution(fold_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize metric mean/std across rotated held-out folds."""

    summary = {}
    for metric_name in ("roc_auc", "sensitivity", "specificity", "precision", "f1", "accuracy"):
        values = [
            metrics.get(metric_name)
            for metrics in fold_metrics.values()
            if metrics.get(metric_name) is not None
        ]
        summary[metric_name] = mean_std(values)
    return summary


def aggregate_coefficients(rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    """Aggregate selected-model coefficients across rotations."""

    by_feature: dict[str, list[float]] = {}
    for row in rows:
        by_feature.setdefault(str(row["feature"]), []).append(float(row["coefficient"]))

    aggregated = []
    for feature, coefficients in by_feature.items():
        mean_coefficient = sum(coefficients) / len(coefficients)
        mean_abs_coefficient = sum(abs(value) for value in coefficients) / len(coefficients)
        aggregated.append(
            {
                "feature": feature,
                "mean_coefficient": mean_coefficient,
                "mean_abs_coefficient": mean_abs_coefficient,
                "rotations": len(coefficients),
            }
        )
    return sorted(
        aggregated,
        key=lambda item: item["mean_abs_coefficient"],
        reverse=True,
    )


def mean_std(values: list[float]) -> dict[str, Any]:
    """Return population mean/std for a metric list."""

    if not values:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance), "n": len(values)}


def summarize_full_radiomics_cv(report: dict[str, Any]) -> dict[str, Any]:
    """Extract compact fields from the rotated-fold radiomics report."""

    aggregate = report.get("aggregate", {})
    pooled = aggregate.get("pooled_test_default", {})
    fixed = aggregate.get("validation_selected_fixed_sensitivity", {})
    return {
        "scope": "full radiomics cohort",
        "case_counts": report.get("case_counts", {}),
        "label_counts": report.get("label_counts", {}),
        "feature_count": report.get("feature_count"),
        "default_test_metrics": pooled.get("metrics", {}),
        "validation_selected_fixed_sensitivity": fixed,
        "fold_metric_summary": aggregate.get("fold_test_metric_summary", {}),
        "limitations": [
            "Uses rotated internal PI-CAI folds, not external validation.",
            "Not directly case-count matched to CNN/hybrid aligned-subset results.",
        ],
    }


def summarize_cnn_report(report: dict[str, Any]) -> dict[str, Any]:
    """Extract compact fields from the CNN baseline report."""

    return {
        "scope": "CNN aligned subset",
        "case_counts": report.get("case_counts", {}),
        "label_counts": report.get("label_counts", {}),
        "model": {
            "name": report.get("model", {}).get("name"),
            "training_status": report.get("model", {}).get("training_status"),
            "input_channels": report.get("model", {}).get("input_channels"),
            "slice_window": report.get("model", {}).get("slice_window"),
            "best_epoch": report.get("model", {}).get("best_epoch"),
        },
        "default_test_metrics": report.get("metrics", {}).get("test", {}).get("metrics", {}),
        "validation_selected_fixed_sensitivity": report.get("validation_selected_threshold", {}).get("test", {}),
        "limitations": [
            "Uses a small 2.5D CNN architecture and limited training.",
            "Represents current CNN embeddings, not a final tuned CNN.",
        ],
    }


def summarize_hybrid_report(report: dict[str, Any]) -> dict[str, Any]:
    """Extract compact fields from the aligned hybrid report."""

    baselines = report.get("baselines", {})
    summarized_baselines = {}
    for name, payload in baselines.items():
        summarized_baselines[name] = {
            "selected_c": payload.get("selected_c"),
            "feature_count": payload.get("feature_count"),
            "default_test_metrics": payload.get("metrics", {}).get("test", {}).get("metrics", {}),
            "validation_selected_fixed_sensitivity": payload.get("validation_selected_threshold", {}).get("test", {}),
        }
    return {
        "scope": "radiomics and CNN aligned subset",
        "case_counts": report.get("case_counts", {}),
        "label_counts": report.get("label_counts", {}),
        "split_label_counts": report.get("split_label_counts", {}),
        "feature_counts": report.get("feature_counts", {}),
        "baselines": summarized_baselines,
        "top_hybrid_coefficients": report.get("top_coefficients", {}).get("hybrid_radiomics_cnn", [])[:15],
        "limitations": [
            "Limited to cases with CNN embeddings.",
            "Hybrid gain is internal and modest; further CNN tuning is needed before stronger claims.",
        ],
    }


def numeric_feature_columns(rows: list[dict[str, str]]) -> list[str]:
    """Return numeric columns that are safe to use as model features."""

    if not rows:
        return []
    columns = []
    for column in rows[0]:
        if column in NON_FEATURE_COLUMNS:
            continue
        values = [row.get(column, "") for row in rows]
        if values and all(is_float(value) for value in values if value != ""):
            columns.append(column)
    return columns


def feature_vector(row: dict[str, str], columns: list[str]) -> list[float]:
    """Convert selected columns to a numeric feature vector."""

    return [float(row[column]) for column in columns]


def fit_standardizer(vectors: list[list[float]]) -> tuple[list[float], list[float]]:
    """Fit column means and standard deviations."""

    if not vectors:
        return [], []
    width = len(vectors[0])
    means = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]
    stds = []
    for index, mean in enumerate(means):
        variance = sum((vector[index] - mean) ** 2 for vector in vectors) / len(vectors)
        stds.append(math.sqrt(variance) or 1.0)
    return means, stds


def apply_standardizer(vector: list[float], standardizer: tuple[list[float], list[float]]) -> list[float]:
    """Apply standardization."""

    means, stds = standardizer
    return [(value - mean) / std for value, mean, std in zip(vector, means, stds)]


def mean_vector(vectors: Any) -> list[float]:
    """Compute a centroid from an iterable of vectors."""

    vectors = list(vectors)
    if not vectors:
        return []
    width = len(vectors[0])
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def euclidean_distance(left: list[float], right: list[float]) -> float:
    """Compute Euclidean distance."""

    return math.sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)))


def roc_auc(labels: list[int], probabilities: list[float]) -> float | None:
    """Compute ROC-AUC with average ranks."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(enumerate(probabilities), key=lambda item: item[1])
    ranks = [0.0] * len(probabilities)
    index = 0
    while index < len(ranked):
        tie_end = index
        while tie_end + 1 < len(ranked) and ranked[tie_end + 1][1] == ranked[index][1]:
            tie_end += 1
        average_rank = (index + 1 + tie_end + 1) / 2.0
        for tied_index in range(index, tie_end + 1):
            ranks[ranked[tied_index][0]] = average_rank
        index = tie_end + 1
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def parse_label(value: str) -> int | None:
    """Parse PI-CAI csPCa labels."""

    normalized = value.strip().lower()
    if normalized in {"yes", "1", "true", "positive"}:
        return 1
    if normalized in {"no", "0", "false", "negative"}:
        return 0
    return None


def prediction_failure_row(baseline_name: str, row: dict[str, Any], reason: str) -> dict[str, str]:
    """Create a prediction failure row."""

    return {
        "baseline": baseline_name,
        "case_id": row["case_id"],
        "fold": row["fold"],
        "split": row["split"],
        "label": str(row["label"]),
        "score": "",
        "probability": "",
        "prediction": "",
        "status": "failed",
        "reason": reason,
    }


def load_csv(path: str | Path) -> list[dict[str, str]]:
    """Load CSV rows."""

    with Path(path).open("r", encoding="utf-8", newline="") as csv_file:
        return [
            {key: (value or "").strip() for key, value in row.items() if key}
            for row in csv.DictReader(csv_file)
        ]


def write_predictions(path: str | Path, rows: list[dict[str, str]]) -> None:
    """Write baseline prediction rows."""

    base_fieldnames = [
        "baseline",
        "case_id",
        "fold",
        "split",
        "label",
        "score",
        "probability",
        "prediction",
        "status",
        "reason",
    ]
    extra_fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in base_fieldnames
        }
    )
    fieldnames = base_fieldnames + extra_fieldnames
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: str | Path, report: dict[str, Any]) -> None:
    """Write a concise Markdown evaluation report."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_for_report(report), encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    """Write plain text content."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def markdown_for_report(report: dict[str, Any]) -> str:
    """Render a Stage 6 report as Markdown."""

    lines = [
        "# Prototype Evaluation Report",
        "",
        "This report summarizes prototype prediction outputs only. It is not a final scientific result.",
        "",
        f"Target sensitivity for fixed-sensitivity analysis: {report['target_sensitivity']}",
        "",
        "## Baselines",
        "",
    ]
    for baseline, splits in report["baselines"].items():
        lines.extend([f"### {baseline}", ""])
        for split, payload in splits.items():
            metrics = payload["metrics"]
            confusion = metrics.get("confusion_matrix", {})
            lines.extend(
                [
                    f"- Split: `{split}`",
                    f"- n: {metrics.get('n')}",
                    f"- ROC-AUC: {metrics.get('roc_auc')}",
                    f"- Sensitivity: {metrics.get('sensitivity')}",
                    f"- Specificity: {metrics.get('specificity')}",
                    f"- Precision: {metrics.get('precision')}",
                    f"- F1: {metrics.get('f1')}",
                    f"- Confusion matrix: {confusion}",
                    f"- False positives: {payload['false_positives']}",
                    f"- False negatives: {payload['false_negatives']}",
                    f"- Fixed-sensitivity analysis: {payload['fixed_sensitivity']}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Claim Limits",
            "",
            *[f"- {item}" for item in report["claim_limits"]],
            "",
        ]
    )
    return "\n".join(lines)


def markdown_for_model_comparison(report: dict[str, Any]) -> str:
    """Render the current model comparison as Markdown."""

    comparisons = report["comparisons"]
    radiomics = comparisons["full_radiomics_cv"]
    cnn = comparisons["cnn_aligned_subset"]
    hybrid = comparisons["hybrid_aligned_subset"]
    hybrid_baselines = hybrid["baselines"]

    lines = [
        "# Current Model Comparison",
        "",
        "This report is an internal methodology checkpoint. It is not external validation.",
        "",
        "## Summary",
        "",
        "| Representation | Scope | n | Test ROC-AUC | Sensitivity | Specificity | Fixed-sensitivity test |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        comparison_row(
            "Radiomics CV",
            radiomics["scope"],
            radiomics.get("case_counts", {}).get("total"),
            radiomics.get("default_test_metrics", {}),
            radiomics.get("validation_selected_fixed_sensitivity", {}).get("metrics", {}),
        ),
        comparison_row(
            "CNN 2.5D",
            cnn["scope"],
            cnn.get("case_counts", {}).get("loaded"),
            cnn.get("default_test_metrics", {}),
            cnn.get("validation_selected_fixed_sensitivity", {}).get("metrics", {}),
        ),
        comparison_row(
            "Aligned radiomics",
            hybrid["scope"],
            hybrid.get("case_counts", {}).get("aligned"),
            hybrid_baselines.get("radiomics_only", {}).get("default_test_metrics", {}),
            hybrid_baselines.get("radiomics_only", {}).get("validation_selected_fixed_sensitivity", {}).get("metrics", {}),
        ),
        comparison_row(
            "Aligned CNN embeddings",
            hybrid["scope"],
            hybrid.get("case_counts", {}).get("aligned"),
            hybrid_baselines.get("cnn_embedding_only", {}).get("default_test_metrics", {}),
            hybrid_baselines.get("cnn_embedding_only", {}).get("validation_selected_fixed_sensitivity", {}).get("metrics", {}),
        ),
        comparison_row(
            "Aligned hybrid",
            hybrid["scope"],
            hybrid.get("case_counts", {}).get("aligned"),
            hybrid_baselines.get("hybrid_radiomics_cnn", {}).get("default_test_metrics", {}),
            hybrid_baselines.get("hybrid_radiomics_cnn", {}).get("validation_selected_fixed_sensitivity", {}).get("metrics", {}),
        ),
        "",
        "## Interpretation",
        "",
    ]
    for item in report["interpretation"]["current_signal"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Threshold Caution", ""])
    for item in report["interpretation"]["threshold_caution"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Top Hybrid Coefficients", ""])
    for row in hybrid.get("top_hybrid_coefficients", [])[:10]:
        lines.append(
            f"- `{row.get('feature')}`: coefficient={row.get('coefficient')}, "
            f"abs={row.get('abs_coefficient')}"
        )
    lines.extend(["", "## Claim Limits", ""])
    for item in report["claim_limits"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def comparison_row(
    name: str,
    scope: str,
    n_value: Any,
    default_metrics: dict[str, Any],
    fixed_metrics: dict[str, Any],
) -> str:
    """Render one Markdown comparison row."""

    fixed_summary = (
        f"sens={metric_string(fixed_metrics.get('sensitivity'))}, "
        f"spec={metric_string(fixed_metrics.get('specificity'))}"
        if fixed_metrics
        else "unavailable"
    )
    return (
        f"| {name} | {scope} | {n_value} | "
        f"{metric_string(default_metrics.get('roc_auc'))} | "
        f"{metric_string(default_metrics.get('sensitivity'))} | "
        f"{metric_string(default_metrics.get('specificity'))} | "
        f"{fixed_summary} |"
    )


def metric_string(value: Any) -> str:
    """Format metrics for Markdown tables."""

    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON payload."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""

    with Path(path).open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def is_float(value: str) -> bool:
    """Return whether a string can be parsed as float."""

    try:
        float(value)
    except ValueError:
        return False
    return True


def safe_divide(numerator: float, denominator: float) -> float | None:
    """Divide while preserving undefined metrics as None."""

    if denominator == 0:
        return None
    return numerator / denominator


def f1_score(precision: float | None, sensitivity: float | None) -> float | None:
    """Compute F1 while preserving undefined precision/recall as None."""

    if precision is None or sensitivity is None:
        return None
    return safe_divide(2 * precision * sensitivity, precision + sensitivity)


def sigmoid(value: float) -> float:
    """Numerically stable sigmoid."""

    if value >= 0:
        exponent = math.exp(-value)
        return 1 / (1 + exponent)
    exponent = math.exp(value)
    return exponent / (1 + exponent)


def format_float(value: float) -> str:
    """Format floats consistently."""

    return f"{value:.10g}"
