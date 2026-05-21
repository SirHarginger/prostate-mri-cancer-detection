#!/usr/bin/env python3
"""Evaluate a saved PI-CAI baseline classifier and write reporting artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMN = "case_cspca_binary"
REQUIRED_MODEL_FILES = [
    "model.joblib",
    "feature_schema.json",
    "metrics.json",
]
OUTPUT_FILENAMES = [
    "evaluation_metrics.json",
    "roc_curve.png",
    "precision_recall_curve.png",
    "confusion_matrix.png",
    "model_card_draft.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved leakage-safe PI-CAI classifier and generate "
            "metrics, plots, and a model-card draft."
        )
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-size", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_joblib():
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required for loading classifier artifacts. Install joblib "
            "in the active environment."
        ) from exc
    return joblib


def require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_curve,
            roc_auc_score,
            roc_curve,
        )
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for classifier evaluation. Install "
            "scikit-learn in the active environment."
        ) from exc

    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError:  # pragma: no cover - depends on scikit-learn version.
        StratifiedGroupKFold = None

    return {
        "average_precision_score": average_precision_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_recall_curve": precision_recall_curve,
        "roc_auc_score": roc_auc_score,
        "roc_curve": roc_curve,
        "StratifiedGroupKFold": StratifiedGroupKFold,
        "train_test_split": train_test_split,
    }


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for evaluation plots. Install matplotlib "
            "in the active environment."
        ) from exc
    return plt


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open() as f:
        return json.load(f)


def load_feature_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Feature CSV does not exist: {path}")
    return pd.read_csv(path)


def load_model_artifacts(model_dir: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required model artifacts: {missing}")

    joblib = require_joblib()
    model = joblib.load(model_dir / "model.joblib")
    schema = load_json(model_dir / "feature_schema.json")
    training_metrics = load_json(model_dir / "metrics.json")
    return model, schema, training_metrics


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        existing = [
            output_dir / name
            for name in OUTPUT_FILENAMES
            if (output_dir / name).exists()
        ]
        if existing and not overwrite:
            raise FileExistsError(
                "Evaluation outputs already exist. Use --overwrite to replace: "
                f"{[str(path) for path in existing]}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)


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
    return target


def feature_columns_from_schema(schema: dict[str, Any], df: pd.DataFrame) -> list[str]:
    columns = schema.get("feature_columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("feature_schema.json must contain a non-empty feature_columns list")

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Feature table is missing schema columns: {missing}")

    unsafe = [col for col in columns if is_unsafe_feature_name(col)]
    if unsafe:
        raise ValueError(f"Unsafe predictor columns in feature schema: {unsafe}")

    return [str(col) for col in columns]


def is_unsafe_feature_name(column: str) -> bool:
    normalized = column.lower()
    unsafe_tokens = [
        "lesion",
        "gleason",
        "histopath",
        "diagnosis",
        "pathology",
        "pirads",
        "isup",
        "cspca",
    ]
    unsafe_exact = {
        TARGET_COLUMN,
        "case_isup_int",
        "case_ISUP",
        "case_key",
        "patient_id",
        "study_id",
        "feature_error",
    }
    return column in unsafe_exact or any(token in normalized for token in unsafe_tokens)


def resolve_validation_size(
    cli_value: float | None,
    schema: dict[str, Any],
    training_metrics: dict[str, Any],
) -> float:
    value = cli_value
    if value is None:
        value = schema.get("validation_size", training_metrics.get("validation_size", 0.2))
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("validation size must be between 0 and 1")
    return value


def resolve_random_seed(
    cli_value: int | None,
    schema: dict[str, Any],
    training_metrics: dict[str, Any],
) -> int:
    value = cli_value
    if value is None:
        value = schema.get("random_seed", training_metrics.get("random_seed", 42))
    return int(value)


def resolve_threshold(
    cli_value: float | None,
    training_metrics: dict[str, Any],
) -> float:
    if cli_value is not None:
        value = float(cli_value)
    else:
        selected = training_metrics.get("selected_model")
        model_metrics = training_metrics.get("models", {}).get(str(selected), {})
        value = float(model_metrics.get("threshold", 0.5))

    if not 0.0 <= value <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return value


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

    overlap = set(patient_ids.iloc[train_idx]) & set(patient_ids.iloc[val_idx])
    if overlap:
        raise ValueError(f"Patient leakage detected across splits: {sorted(overlap)[:5]}")

    return np.asarray(train_idx), np.asarray(val_idx), split_method


def metric_dict(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    sklearn: dict[str, Any],
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = sklearn["confusion_matrix"](y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "roc_auc": float(sklearn["roc_auc_score"](y_true, y_prob)),
        "pr_auc": float(sklearn["average_precision_score"](y_true, y_prob)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(sklearn["balanced_accuracy_score"](y_true, y_pred)),
        "f1": float(sklearn["f1_score"](y_true, y_pred, zero_division=0)),
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


def evaluate_model(
    features_path: Path,
    model_dir: Path,
    output_dir: Path,
    *,
    validation_size: float | None = None,
    random_seed: int | None = None,
    threshold: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    sklearn = require_sklearn()
    model, schema, training_metrics = load_model_artifacts(model_dir)
    df = load_feature_table(features_path)
    assert_no_feature_errors(df)
    y = validate_target(df)
    feature_columns = feature_columns_from_schema(schema, df)

    validation_size_value = resolve_validation_size(
        validation_size,
        schema,
        training_metrics,
    )
    random_seed_value = resolve_random_seed(random_seed, schema, training_metrics)
    threshold_value = resolve_threshold(threshold, training_metrics)

    train_idx, val_idx, split_method = make_split(
        df,
        y,
        sklearn,
        validation_size=validation_size_value,
        random_seed=random_seed_value,
    )

    X_val = df.iloc[val_idx][feature_columns].copy()
    y_val = y.iloc[val_idx]
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold_value).astype(int)

    metrics = metric_dict(y_val, y_prob, threshold_value, sklearn)
    payload = {
        "model_dir": str(model_dir),
        "features": str(features_path),
        "target_column": TARGET_COLUMN,
        "selected_model": training_metrics.get("selected_model", schema.get("selected_model")),
        "input_shape": [int(df.shape[0]), int(df.shape[1])],
        "feature_count": int(len(feature_columns)),
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
                for key, value in y_val.value_counts().sort_index().items()
            },
        },
        "metrics": metrics,
        "random_seed": int(random_seed_value),
        "validation_size": float(validation_size_value),
        "notes": [
            "Evaluation uses the saved leakage-safe feature schema.",
            "Research prototype only; not clinically validated.",
            "Decision-support only. Not a standalone diagnosis.",
        ],
    }

    prepare_output_dir(output_dir, overwrite)
    write_outputs(output_dir, payload, y_val, y_prob, y_pred, sklearn)
    return payload


def write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    y_true: pd.Series,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    sklearn: dict[str, Any],
) -> None:
    with (output_dir / "evaluation_metrics.json").open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    plot_roc_curve(output_dir / "roc_curve.png", y_true, y_prob, sklearn)
    plot_pr_curve(output_dir / "precision_recall_curve.png", y_true, y_prob, sklearn)
    plot_confusion_matrix(output_dir / "confusion_matrix.png", y_true, y_pred, sklearn)
    (output_dir / "model_card_draft.md").write_text(make_model_card(payload))


def plot_roc_curve(
    path: Path,
    y_true: pd.Series,
    y_prob: np.ndarray,
    sklearn: dict[str, Any],
) -> None:
    plt = require_matplotlib()
    fpr, tpr, _ = sklearn["roc_curve"](y_true, y_prob)
    auc = payload_float(sklearn["roc_auc_score"](y_true, y_prob))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("PI-CAI Fold0 ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pr_curve(
    path: Path,
    y_true: pd.Series,
    y_prob: np.ndarray,
    sklearn: dict[str, Any],
) -> None:
    plt = require_matplotlib()
    precision, recall, _ = sklearn["precision_recall_curve"](y_true, y_prob)
    auc = payload_float(sklearn["average_precision_score"](y_true, y_prob))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR AUC = {auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PI-CAI Fold0 Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(
    path: Path,
    y_true: pd.Series,
    y_pred: np.ndarray,
    sklearn: dict[str, Any],
) -> None:
    plt = require_matplotlib()
    cm = sklearn["confusion_matrix"](y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title("PI-CAI Fold0 Confusion Matrix")

    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(int(cm[row, col])), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def payload_float(value: Any) -> float:
    return float(value)


def make_model_card(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    cm = metrics["confusion_matrix"]
    return "\n".join(
        [
            "# PI-CAI Fold0 Classifier Model Card Draft",
            "",
            "Research prototype only. Not clinically validated.",
            "",
            "## Intended Use",
            "",
            "This model is an experimental case-level classifier for clinically "
            "significant prostate cancer research. It is decision-support only "
            "and not a standalone diagnosis.",
            "",
            "## Data and Features",
            "",
            f"- Input feature table: `{payload['features']}`",
            f"- Model directory: `{payload['model_dir']}`",
            f"- Input shape: {tuple(payload['input_shape'])}",
            f"- Feature count: {payload['feature_count']}",
            "- Features use safe clinical variables and T2W whole-gland radiomics.",
            "- Lesion radiomics and diagnosis-derived variables are excluded.",
            "",
            "## Validation Setup",
            "",
            f"- Split method: `{payload['split']['method']}`",
            f"- Train cases: {payload['split']['train_cases']}",
            f"- Validation cases: {payload['split']['validation_cases']}",
            f"- Target counts: {payload['target_counts']}",
            "",
            "## Validation Metrics",
            "",
            f"- ROC AUC: {metrics['roc_auc']:.4f}",
            f"- PR AUC: {metrics['pr_auc']:.4f}",
            f"- Sensitivity: {metrics['sensitivity']:.4f}",
            f"- Specificity: {metrics['specificity']:.4f}",
            f"- Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
            f"- F1: {metrics['f1']:.4f}",
            f"- Confusion matrix [[TN, FP], [FN, TP]]: {cm['matrix']}",
            "",
            "## Limitations",
            "",
            "- Fold0 validation result only.",
            "- Not externally validated.",
            "- Not clinically validated.",
            "- Performance is modest and needs stronger validation.",
            "",
        ]
    )


def print_summary(payload: dict[str, Any], output_dir: Path) -> None:
    metrics = payload["metrics"]
    print(f"Saved evaluation outputs: {output_dir}")
    print(f"Selected model: {payload['selected_model']}")
    print(f"Input shape: {tuple(payload['input_shape'])}")
    print(f"Feature count: {payload['feature_count']}")
    print(
        "Split: "
        f"{payload['split']['method']} "
        f"({payload['split']['train_cases']} train / "
        f"{payload['split']['validation_cases']} validation)"
    )
    print("\nValidation metrics:")
    print(f"roc_auc={metrics['roc_auc']:.4f}")
    print(f"pr_auc={metrics['pr_auc']:.4f}")
    print(f"sensitivity={metrics['sensitivity']:.4f}")
    print(f"specificity={metrics['specificity']:.4f}")
    print(f"balanced_accuracy={metrics['balanced_accuracy']:.4f}")
    print(f"f1={metrics['f1']:.4f}")


def main() -> None:
    args = parse_args()
    payload = evaluate_model(
        args.features,
        args.model_dir,
        args.output_dir,
        validation_size=args.validation_size,
        random_seed=args.random_seed,
        threshold=args.threshold,
        overwrite=args.overwrite,
    )
    print_summary(payload, args.output_dir)


if __name__ == "__main__":
    main()
