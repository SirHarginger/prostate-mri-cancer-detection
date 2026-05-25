"""Stage 7 prototype explainability utilities.

The current project has dependency-light prototype baselines, not final trained
models. This module therefore reports centroid-based feature importance only and
explicitly marks CNN visual explanation as unavailable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from prostate_mri_cancer_detection.evaluation import (
    SPLIT_BY_FOLD,
    apply_standardizer,
    feature_vector,
    fit_standardizer,
    load_csv,
    mean_vector,
    numeric_feature_columns,
    parse_label,
)


BASELINES = {
    "radiomics_only": ("radiomics",),
    "prototype_embedding_only": ("embedding",),
    "hybrid_radiomics_embedding": ("radiomics", "embedding"),
}


def generate_explainability_report(
    manifest_path: str | Path,
    radiomics_path: str | Path,
    embeddings_path: str | Path,
    output_json_path: str | Path,
    output_csv_path: str | Path,
    top_n: int = 20,
) -> dict[str, Any]:
    """Generate prototype feature-importance reports for aligned baselines."""

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
    radiomics_columns = numeric_feature_columns(radiomics_rows)
    embedding_columns = numeric_feature_columns(embedding_rows)

    aligned_rows = []
    for case_id in sorted(set(labels) & set(radiomics_by_case) & set(embeddings_by_case)):
        label = labels.get(case_id)
        if label is None:
            continue
        fold = folds.get(case_id, radiomics_by_case[case_id].get("fold", ""))
        split = embeddings_by_case[case_id].get("split") or SPLIT_BY_FOLD.get(fold, "unknown")
        aligned_rows.append(
            {
                "case_id": case_id,
                "fold": fold,
                "split": split,
                "label": label,
                "radiomics": feature_vector(radiomics_by_case[case_id], radiomics_columns),
                "embedding": feature_vector(embeddings_by_case[case_id], embedding_columns),
            }
        )

    feature_sets = {
        "radiomics": radiomics_columns,
        "embedding": embedding_columns,
    }
    importances: dict[str, Any] = {}
    csv_rows: list[dict[str, str]] = []
    for baseline_name, feature_sources in BASELINES.items():
        baseline_feature_names = []
        for source in feature_sources:
            baseline_feature_names.extend(f"{source}:{name}" for name in feature_sets[source])
        rows = prototype_feature_importance(
            baseline_name=baseline_name,
            aligned_rows=aligned_rows,
            feature_names=baseline_feature_names,
        )
        importances[baseline_name] = {
            "status": "ok" if rows else "unavailable",
            "top_features": rows[:top_n],
        }
        csv_rows.extend(rows[:top_n])

    report = {
        "schema_version": "1.0",
        "stage": "prototype_explainability",
        "manifest_path": str(manifest_path),
        "radiomics_path": str(radiomics_path),
        "embeddings_path": str(embeddings_path),
        "aligned_cases": len(aligned_rows),
        "top_n": top_n,
        "method": "absolute standardized centroid difference between positive and negative training cases",
        "importances": importances,
        "cnn_visual_explanation": {
            "status": "not_available",
            "reason": "No trained CNN model or spatial activation maps exist in the current implementation.",
        },
        "claim_limits": [
            "Feature importance is a prototype model-inspection aid, not clinical proof.",
            "Prototype embeddings are not trained CNN features.",
            "No Grad-CAM or saliency map is valid until a trained CNN is implemented.",
            "Do not infer causality, lesion localization, or clinical readiness from this report.",
        ],
    }
    write_json(output_json_path, report)
    write_importance_csv(output_csv_path, csv_rows)
    return report


def prototype_feature_importance(
    baseline_name: str,
    aligned_rows: list[dict[str, Any]],
    feature_names: list[str],
) -> list[dict[str, str]]:
    """Compute centroid-difference feature importances for one baseline."""

    train_rows = [row for row in aligned_rows if row["split"] == "train"]
    if {row["label"] for row in train_rows} != {0, 1}:
        return []

    vectors = [baseline_vector(row, baseline_name) for row in train_rows]
    standardizer = fit_standardizer(vectors)
    standardized = [apply_standardizer(vector, standardizer) for vector in vectors]
    centroids = {
        label: mean_vector(
            vector for vector, row in zip(standardized, train_rows) if row["label"] == label
        )
        for label in (0, 1)
    }
    rows = []
    for index, feature_name in enumerate(feature_names):
        negative_value = centroids[0][index]
        positive_value = centroids[1][index]
        rows.append(
            {
                "baseline": baseline_name,
                "rank": "0",
                "feature": feature_name,
                "importance": format_float(abs(positive_value - negative_value)),
                "positive_centroid": format_float(positive_value),
                "negative_centroid": format_float(negative_value),
                "direction": "higher_in_positive" if positive_value > negative_value else "higher_in_negative",
            }
        )
    rows = sorted(rows, key=lambda row: float(row["importance"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = str(rank)
    return rows


def baseline_vector(row: dict[str, Any], baseline_name: str) -> list[float]:
    """Return a row vector for the requested baseline."""

    if baseline_name == "radiomics_only":
        return row["radiomics"]
    if baseline_name == "prototype_embedding_only":
        return row["embedding"]
    if baseline_name == "hybrid_radiomics_embedding":
        return row["radiomics"] + row["embedding"]
    raise ValueError(f"unknown baseline: {baseline_name}")


def write_importance_csv(path: str | Path, rows: list[dict[str, str]]) -> None:
    """Write feature importance rows."""

    fieldnames = [
        "baseline",
        "rank",
        "feature",
        "importance",
        "positive_centroid",
        "negative_centroid",
        "direction",
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


def format_float(value: float) -> str:
    """Format floats consistently."""

    return f"{value:.10g}"
