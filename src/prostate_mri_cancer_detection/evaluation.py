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

    fieldnames = [
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON payload."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


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
