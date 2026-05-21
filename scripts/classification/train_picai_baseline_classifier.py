#!/usr/bin/env python3
"""Train leakage-safe PI-CAI fold0 baseline csPCa classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMN = "case_cspca_binary"
CLINICAL_NUMERIC_COLUMNS = [
    "patient_age",
    "psa",
    "psad",
    "prostate_volume",
]
CATEGORICAL_COLUMNS = ["center"]
RADIOMICS_PREFIX = "t2w_wholegland_"
LEAKAGE_COLUMNS = {
    TARGET_COLUMN,
    "case_isup_int",
    "case_ISUP",
    "case_key",
    "patient_id",
    "study_id",
    "feature_error",
}
LEAKAGE_SUBSTRINGS = [
    "lesion",
    "gleason",
    "histopath",
    "diagnosis",
    "pathology",
    "pirads",
    "isup",
    "cspca",
]
ARTIFACT_FILENAMES = [
    "model.joblib",
    "preprocessing_pipeline.joblib",
    "feature_schema.json",
    "metrics.json",
    "training_summary.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train leakage-safe baseline classifiers from PI-CAI case-level "
            "clinical and T2W whole-gland radiomics features."
        )
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for baseline classifier training. "
            "Install scikit-learn in the active environment."
        ) from exc

    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError:  # pragma: no cover - depends on scikit-learn version.
        StratifiedGroupKFold = None

    return {
        "ColumnTransformer": ColumnTransformer,
        "RandomForestClassifier": RandomForestClassifier,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "average_precision_score": average_precision_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "roc_auc_score": roc_auc_score,
        "StratifiedGroupKFold": StratifiedGroupKFold,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    }


def require_joblib():
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required for saving classifier artifacts. Install joblib "
            "in the active environment."
        ) from exc
    return joblib


def parse_validation_size(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError("--validation-size must be between 0 and 1")
    return value


def load_feature_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Feature CSV does not exist: {path}")
    return pd.read_csv(path)


def assert_no_feature_errors(df: pd.DataFrame) -> None:
    if "feature_error" not in df.columns:
        return

    error_mask = df["feature_error"].fillna("").astype(str).str.strip().ne("")
    if error_mask.any():
        examples = df.loc[error_mask, "feature_error"].value_counts().head(5)
        raise ValueError(
            "Feature table contains non-empty feature_error values. "
            f"Top errors: {examples.to_dict()}"
        )


def validate_target(df: pd.DataFrame) -> pd.Series:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if target.isna().any():
        raise ValueError(f"Target column {TARGET_COLUMN} contains missing values")

    target = target.astype(int)
    unique_values = sorted(target.unique().tolist())
    if unique_values != [0, 1]:
        raise ValueError(
            f"Target column {TARGET_COLUMN} must contain exactly 0 and 1; "
            f"found {unique_values}"
        )

    class_counts = target.value_counts()
    if int(class_counts.min()) < 2:
        raise ValueError("Each target class needs at least two cases for validation")

    return target


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    missing_clinical = [
        col for col in [*CLINICAL_NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]
        if col not in df.columns
    ]
    if missing_clinical:
        raise ValueError(f"Missing required clinical feature columns: {missing_clinical}")

    radiomics_columns = [
        col for col in df.columns if col.startswith(RADIOMICS_PREFIX)
    ]
    if not radiomics_columns:
        raise ValueError(f"No {RADIOMICS_PREFIX} radiomics columns found")

    selected = [
        *CLINICAL_NUMERIC_COLUMNS,
        *CATEGORICAL_COLUMNS,
        *radiomics_columns,
    ]

    unsafe = [col for col in selected if is_leakage_column(col)]
    if unsafe:
        raise ValueError(f"Unsafe predictor columns selected: {unsafe}")

    return selected


def is_leakage_column(column: str) -> bool:
    normalized = column.lower()
    return column in LEAKAGE_COLUMNS or any(
        token in normalized for token in LEAKAGE_SUBSTRINGS
    )


def feature_schema(feature_columns: list[str]) -> dict[str, Any]:
    radiomics_columns = [
        col for col in feature_columns if col.startswith(RADIOMICS_PREFIX)
    ]
    return {
        "target_column": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "numeric_columns": [
            *CLINICAL_NUMERIC_COLUMNS,
            *radiomics_columns,
        ],
        "categorical_columns": CATEGORICAL_COLUMNS,
        "radiomics_prefix": RADIOMICS_PREFIX,
        "excluded_columns": sorted(LEAKAGE_COLUMNS),
        "excluded_substrings": LEAKAGE_SUBSTRINGS,
        "notes": [
            "Lesion radiomics are excluded from binary csPCa classification.",
            "case_isup_int and diagnosis-derived variables are not predictors.",
            "Research prototype only; not clinically validated.",
        ],
    }


def one_hot_encoder(sklearn: dict[str, Any]) -> Any:
    OneHotEncoder = sklearn["OneHotEncoder"]
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(
    feature_columns: list[str],
    sklearn: dict[str, Any],
    *,
    scale_numeric: bool,
) -> Any:
    ColumnTransformer = sklearn["ColumnTransformer"]
    Pipeline = sklearn["Pipeline"]
    SimpleImputer = sklearn["SimpleImputer"]
    StandardScaler = sklearn["StandardScaler"]

    numeric_columns = [
        col for col in feature_columns if col not in CATEGORICAL_COLUMNS
    ]

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot_encoder(sklearn)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def make_model_specs(
    feature_columns: list[str],
    sklearn: dict[str, Any],
    *,
    random_seed: int,
    n_estimators: int,
) -> dict[str, Any]:
    Pipeline = sklearn["Pipeline"]
    LogisticRegression = sklearn["LogisticRegression"]
    RandomForestClassifier = sklearn["RandomForestClassifier"]

    return {
        "logistic_regression": Pipeline(
            [
                (
                    "preprocess",
                    make_preprocessor(
                        feature_columns,
                        sklearn,
                        scale_numeric=True,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "preprocess",
                    make_preprocessor(
                        feature_columns,
                        sklearn,
                        scale_numeric=False,
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        class_weight="balanced",
                        random_state=random_seed,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
    }


def make_split(
    df: pd.DataFrame,
    y: pd.Series,
    sklearn: dict[str, Any],
    *,
    validation_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    train_test_split = sklearn["train_test_split"]

    indices = np.arange(len(df))
    patient_ids = (
        df["patient_id"].astype(str)
        if "patient_id" in df.columns
        else pd.Series([f"case_{idx}" for idx in indices])
    )

    if patient_ids.duplicated().any():
        StratifiedGroupKFold = sklearn["StratifiedGroupKFold"]
        if StratifiedGroupKFold is None:
            raise ImportError(
                "Duplicate patient IDs require StratifiedGroupKFold. Upgrade "
                "scikit-learn or provide one row per patient."
            )
        n_splits = max(2, int(round(1.0 / validation_size)))
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_seed,
        )
        train_idx, val_idx = next(splitter.split(indices, y, groups=patient_ids))
        split_method = f"StratifiedGroupKFold(n_splits={n_splits})"
    else:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_size,
            random_state=random_seed,
            stratify=y,
        )
        split_method = "stratified_case_split"

    train_patients = set(patient_ids.iloc[train_idx])
    val_patients = set(patient_ids.iloc[val_idx])
    overlap = train_patients & val_patients
    if overlap:
        raise ValueError(f"Patient leakage detected across splits: {sorted(overlap)[:5]}")

    return np.asarray(train_idx), np.asarray(val_idx), split_method


def metric_dict(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    sklearn: dict[str, Any],
) -> dict[str, Any]:
    roc_auc_score = sklearn["roc_auc_score"]
    average_precision_score = sklearn["average_precision_score"]
    balanced_accuracy_score = sklearn["balanced_accuracy_score"]
    confusion_matrix = sklearn["confusion_matrix"]
    f1_score = sklearn["f1_score"]

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": [[tn, fp], [fn, tp]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def train_and_evaluate_models(
    df: pd.DataFrame,
    feature_columns: list[str],
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    sklearn: dict[str, Any],
    *,
    random_seed: int,
    n_estimators: int,
    threshold: float,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    X = df[feature_columns].copy()
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    models = make_model_specs(
        feature_columns,
        sklearn,
        random_seed=random_seed,
        n_estimators=n_estimators,
    )
    metrics: dict[str, Any] = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        metrics[name] = metric_dict(y_val, y_prob, threshold, sklearn)

    selected_name = sorted(
        metrics,
        key=lambda name: (
            metrics[name]["roc_auc"],
            metrics[name]["pr_auc"],
        ),
        reverse=True,
    )[0]

    return models, selected_name, metrics


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        existing_artifacts = [
            output_dir / filename
            for filename in ARTIFACT_FILENAMES
            if (output_dir / filename).exists()
        ]
        if existing_artifacts and not overwrite:
            raise FileExistsError(
                "Output artifacts already exist. Use --overwrite to replace: "
                f"{[str(path) for path in existing_artifacts]}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def save_artifacts(
    output_dir: Path,
    selected_model: Any,
    schema: dict[str, Any],
    metrics_payload: dict[str, Any],
    summary: str,
    *,
    overwrite: bool,
) -> None:
    prepare_output_dir(output_dir, overwrite)
    joblib = require_joblib()

    joblib.dump(selected_model, output_dir / "model.joblib")
    joblib.dump(
        selected_model.named_steps["preprocess"],
        output_dir / "preprocessing_pipeline.joblib",
    )

    with (output_dir / "feature_schema.json").open("w") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics_payload, f, indent=2)
        f.write("\n")

    (output_dir / "training_summary.md").write_text(summary)


def make_training_summary(metrics_payload: dict[str, Any]) -> str:
    lines = [
        "# PI-CAI Fold0 Baseline Classifier Training Summary",
        "",
        "Research prototype only. Not clinically validated.",
        "",
        f"- Selected model: `{metrics_payload['selected_model']}`",
        f"- Target: `{metrics_payload['target_column']}`",
        f"- Split method: `{metrics_payload['split']['method']}`",
        f"- Train cases: {metrics_payload['split']['train_cases']}",
        f"- Validation cases: {metrics_payload['split']['validation_cases']}",
        f"- Feature count: {metrics_payload['feature_count']}",
        "",
        "## Validation Metrics",
        "",
        "| Model | ROC AUC | PR AUC | Sensitivity | Specificity | Balanced Accuracy | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name, metrics in metrics_payload["models"].items():
        lines.append(
            "| "
            f"{name} | "
            f"{metrics['roc_auc']:.4f} | "
            f"{metrics['pr_auc']:.4f} | "
            f"{metrics['sensitivity']:.4f} | "
            f"{metrics['specificity']:.4f} | "
            f"{metrics['balanced_accuracy']:.4f} | "
            f"{metrics['f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Leakage Controls",
            "",
            "- Lesion radiomics are excluded.",
            "- `case_isup_int` is excluded from predictors.",
            "- IDs, target columns, and feature errors are excluded from predictors.",
            "- Predictor columns are selected from an explicit allowlist.",
            "",
        ]
    )

    return "\n".join(lines)


def train_baseline_classifier(
    features_path: Path,
    output_dir: Path,
    *,
    validation_size: float = 0.2,
    random_seed: int = 42,
    n_estimators: int = 500,
    threshold: float = 0.5,
    overwrite: bool = False,
) -> dict[str, Any]:
    validation_size = parse_validation_size(validation_size)
    sklearn = require_sklearn()

    df = load_feature_table(features_path)
    assert_no_feature_errors(df)
    y = validate_target(df)
    feature_columns = select_feature_columns(df)
    prepare_output_dir(output_dir, overwrite)
    train_idx, val_idx, split_method = make_split(
        df,
        y,
        sklearn,
        validation_size=validation_size,
        random_seed=random_seed,
    )

    models, selected_model, model_metrics = train_and_evaluate_models(
        df,
        feature_columns,
        y,
        train_idx,
        val_idx,
        sklearn,
        random_seed=random_seed,
        n_estimators=n_estimators,
        threshold=threshold,
    )

    schema = feature_schema(feature_columns)
    schema.update(
        {
            "random_seed": random_seed,
            "validation_size": validation_size,
            "selected_model": selected_model,
        }
    )

    metrics_payload = {
        "target_column": TARGET_COLUMN,
        "selected_model": selected_model,
        "models": model_metrics,
        "feature_count": len(feature_columns),
        "input_shape": [int(df.shape[0]), int(df.shape[1])],
        "target_counts": {
            str(key): int(value)
            for key, value in y.value_counts().sort_index().items()
        },
        "split": {
            "method": split_method,
            "train_cases": int(len(train_idx)),
            "validation_cases": int(len(val_idx)),
            "train_target_counts": {
                str(key): int(value)
                for key, value in y.iloc[train_idx].value_counts().sort_index().items()
            },
            "validation_target_counts": {
                str(key): int(value)
                for key, value in y.iloc[val_idx].value_counts().sort_index().items()
            },
        },
        "random_seed": int(random_seed),
        "validation_size": float(validation_size),
        "notes": [
            "Experimental classifier; not clinically validated.",
            "Lesion radiomics and diagnosis-derived variables are excluded.",
        ],
    }

    summary = make_training_summary(metrics_payload)
    save_artifacts(
        output_dir,
        models[selected_model],
        schema,
        metrics_payload,
        summary,
        overwrite=overwrite,
    )

    return metrics_payload


def print_summary(metrics_payload: dict[str, Any], output_dir: Path) -> None:
    print(f"Saved classifier artifacts: {output_dir}")
    print(f"Input shape: {tuple(metrics_payload['input_shape'])}")
    print(f"Feature count: {metrics_payload['feature_count']}")
    print(f"Target counts: {metrics_payload['target_counts']}")
    print(
        "Split: "
        f"{metrics_payload['split']['method']} "
        f"({metrics_payload['split']['train_cases']} train / "
        f"{metrics_payload['split']['validation_cases']} validation)"
    )
    print(f"Selected model: {metrics_payload['selected_model']}")

    print("\nValidation metrics:")
    for name, metrics in metrics_payload["models"].items():
        print(
            f"{name}: "
            f"roc_auc={metrics['roc_auc']:.4f}, "
            f"pr_auc={metrics['pr_auc']:.4f}, "
            f"sensitivity={metrics['sensitivity']:.4f}, "
            f"specificity={metrics['specificity']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"f1={metrics['f1']:.4f}"
        )


def main() -> None:
    args = parse_args()
    metrics_payload = train_baseline_classifier(
        args.features,
        args.output_dir,
        validation_size=args.validation_size,
        random_seed=args.random_seed,
        n_estimators=args.n_estimators,
        threshold=args.threshold,
        overwrite=args.overwrite,
    )
    print_summary(metrics_payload, args.output_dir)


if __name__ == "__main__":
    main()
