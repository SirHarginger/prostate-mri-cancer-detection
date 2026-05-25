"""Stage 4 split-safe embedding extraction utilities.

This module provides a minimal embedding pipeline that is safe to run before
full CNN training exists. The default encoder is a deterministic prototype over
original T2W voxel intensities; provenance labels it as such so it is not
mistaken for a trained clinical CNN.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from prostate_mri_cancer_detection.features import read_volume_data
from prostate_mri_cancer_detection.preprocessing import resolve_manifest_path

DEFAULT_SPLIT_BY_FOLD = {
    "fold0": "train",
    "fold1": "train",
    "fold2": "train",
    "fold3": "validation",
    "fold4": "test",
}
SUPPORTED_SEQUENCES = {"t2w": "path_t2w"}


def extract_embedding_table(
    manifest_path: str | Path,
    raw_root: str | Path,
    output_path: str | Path,
    provenance_path: str | Path,
    report_path: str | Path,
    preprocessing_report_path: str | Path | None = None,
    sequence: str = "t2w",
    embedding_dim: int = 32,
    sample_size_per_split: int = 5,
    case_ids: Iterable[str] | None = None,
    all_cases: bool = False,
    augment_train: bool = False,
) -> dict[str, Any]:
    """Extract deterministic prototype embeddings with split provenance."""

    if sequence not in SUPPORTED_SEQUENCES:
        raise ValueError("Stage 4 supports T2W embeddings only until ADC/HBV resampling is implemented")

    manifest_path = Path(manifest_path)
    raw_root = Path(raw_root)
    rows = load_manifest_rows(manifest_path)
    preprocessing_report = load_optional_json(preprocessing_report_path)
    selected_rows = select_embedding_rows(rows, preprocessing_report, sample_size_per_split, case_ids, all_cases)

    embedding_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for row in selected_rows:
        split = DEFAULT_SPLIT_BY_FOLD.get(row.get("fold", ""), "unknown")
        augmentation_applied = bool(augment_train and split == "train")
        image_path = resolve_manifest_path(row.get(SUPPORTED_SEQUENCES[sequence], ""), raw_root)
        try:
            embedding = extract_deterministic_embedding(
                image_path=image_path,
                embedding_dim=embedding_dim,
                augmentation_applied=augmentation_applied,
            )
            embedding_rows.append(
                build_embedding_row(
                    row=row,
                    split=split,
                    sequence=sequence,
                    image_path=image_path,
                    embedding=embedding,
                    augmentation_applied=augmentation_applied,
                )
            )
        except Exception as error:  # noqa: BLE001 - per-case failures belong in report.
            failures.append(
                {
                    "case_id": row.get("case_id", ""),
                    "fold": row.get("fold", ""),
                    "split": split,
                    "sequence": sequence,
                    "image_path": str(image_path),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )

    provenance = build_embedding_provenance(
        manifest_path=manifest_path,
        raw_root=raw_root,
        preprocessing_report_path=preprocessing_report_path,
        sequence=sequence,
        embedding_dim=embedding_dim,
        sample_size_per_split=sample_size_per_split,
        all_cases=all_cases,
        augment_train=augment_train,
        selected_case_ids=[row.get("case_id", "") for row in selected_rows],
    )
    report = build_embedding_report(
        embedding_rows=embedding_rows,
        failures=failures,
        provenance=provenance,
    )

    write_embedding_csv(output_path, embedding_rows, embedding_dim)
    write_json(provenance_path, provenance)
    write_json(report_path, report)
    return report["summary"]


def extract_deterministic_embedding(
    image_path: Path,
    embedding_dim: int,
    augmentation_applied: bool = False,
    max_sampled_voxels: int = 200_000,
) -> list[float]:
    """Create a fixed-length prototype embedding from original voxel values."""

    volume = read_volume_data(image_path)
    values = volume.values
    if not values:
        raise ValueError("empty image volume")

    stride = max(1, len(values) // max_sampled_voxels)
    sampled = [float(value) for value in values[::stride]][:max_sampled_voxels]
    if augmentation_applied:
        sampled = [value * 1.01 for value in sampled]

    mean = sum(sampled) / len(sampled)
    variance = sum((value - mean) ** 2 for value in sampled) / len(sampled)
    std = math.sqrt(variance) or 1.0
    normalized = [(value - mean) / std for value in sampled]

    return segment_means(normalized, embedding_dim)


def segment_means(values: list[float], embedding_dim: int) -> list[float]:
    """Convert a sequence of values into fixed-length segment means."""

    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if not values:
        return [0.0] * embedding_dim

    embedding = []
    for index in range(embedding_dim):
        start = math.floor(index * len(values) / embedding_dim)
        end = math.floor((index + 1) * len(values) / embedding_dim)
        segment = values[start:end] or [values[min(start, len(values) - 1)]]
        embedding.append(sum(segment) / len(segment))
    return embedding


def build_embedding_row(
    row: dict[str, str],
    split: str,
    sequence: str,
    image_path: Path,
    embedding: list[float],
    augmentation_applied: bool,
) -> dict[str, str]:
    """Build one embedding table row."""

    output = {
        "case_id": row.get("case_id", ""),
        "fold": row.get("fold", ""),
        "split": split,
        "label_cspca": row.get("label_cspca", ""),
        "sequence": sequence,
        "encoder_name": "deterministic_t2w_intensity_embedding_v1",
        "encoder_type": "prototype_not_trained_cnn",
        "augmentation_applied": str(augmentation_applied),
        "image_path": str(image_path),
    }
    for index, value in enumerate(embedding):
        output[f"embedding_{index:03d}"] = format_float(value)
    return output


def select_embedding_rows(
    rows: list[dict[str, str]],
    preprocessing_report: dict[str, Any] | None,
    sample_size_per_split: int,
    case_ids: Iterable[str] | None,
    all_cases: bool,
) -> list[dict[str, str]]:
    """Select rows without leaking augmented copies across folds."""

    rows_by_case = {row.get("case_id", ""): row for row in rows}
    requested_case_ids = sorted(set(case_id for case_id in case_ids or [] if case_id))
    if requested_case_ids:
        return [rows_by_case[case_id] for case_id in requested_case_ids if case_id in rows_by_case]

    allowed_case_ids: set[str] | None = None
    if preprocessing_report is not None and preprocessing_report.get("selected_case_ids"):
        allowed_case_ids = set(preprocessing_report["selected_case_ids"])

    candidates = [
        row
        for row in rows
        if row.get("case_id", "") and (allowed_case_ids is None or row.get("case_id", "") in allowed_case_ids)
    ]
    candidates = sorted(candidates, key=lambda item: item.get("case_id", ""))
    if all_cases:
        return candidates

    selected: list[dict[str, str]] = []
    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_split[DEFAULT_SPLIT_BY_FOLD.get(row.get("fold", ""), "unknown")].append(row)
    for split in ("train", "validation", "test", "unknown"):
        selected.extend(by_split[split][: max(sample_size_per_split, 0)])
    return selected


def build_embedding_provenance(
    manifest_path: Path,
    raw_root: Path,
    preprocessing_report_path: str | Path | None,
    sequence: str,
    embedding_dim: int,
    sample_size_per_split: int,
    all_cases: bool,
    augment_train: bool,
    selected_case_ids: list[str],
) -> dict[str, Any]:
    """Record model/config provenance for Stage 4 embeddings."""

    return {
        "schema_version": "1.0",
        "stage": "cnn_embedding_pipeline_prototype",
        "manifest_path": str(manifest_path),
        "raw_root": str(raw_root),
        "preprocessing_report_path": str(preprocessing_report_path or ""),
        "sequence": sequence,
        "embedding_dim": embedding_dim,
        "sample_size_per_split": sample_size_per_split,
        "all_cases": all_cases,
        "selected_case_ids": selected_case_ids,
        "split_by_fold": DEFAULT_SPLIT_BY_FOLD,
        "encoder_name": "deterministic_t2w_intensity_embedding_v1",
        "encoder_type": "prototype_not_trained_cnn",
        "training_status": "not_trained",
        "augmentation_policy": {
            "train_augmentation_enabled": augment_train,
            "validation_augmentation_enabled": False,
            "test_augmentation_enabled": False,
            "saved_augmented_copies": False,
        },
        "paired_transform_policy": {
            "t2w": "native-grid deterministic resize-to-vector prototype",
            "adc": "deferred until resampling/alignment to T2W grid is implemented",
            "hbv": "deferred until resampling/alignment to T2W grid is implemented",
            "masks": "not used by this embedding prototype; ROI-aware CNN embeddings are deferred",
        },
        "claim_limits": [
            "This is an embedding pipeline scaffold, not a trained CNN model.",
            "Do not compare as CNN-only baseline until a real encoder/training policy is implemented.",
            "No clinical, localization, or deployment claims are supported by this output.",
        ],
    }


def build_embedding_report(
    embedding_rows: list[dict[str, str]],
    failures: list[dict[str, str]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact Stage 4 validation report."""

    split_counter = Counter(row["split"] for row in embedding_rows)
    augmentation_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in embedding_rows:
        augmentation_by_split[row["split"]][row["augmentation_applied"]] += 1

    return {
        "schema_version": "1.0",
        "stage": "cnn_embedding_pipeline_prototype",
        "summary": {
            "embeddings_written": len(embedding_rows),
            "failures": len(failures),
            "embeddings_by_split": dict(sorted(split_counter.items())),
            "augmentation_by_split": {
                split: dict(counter)
                for split, counter in sorted(augmentation_by_split.items())
            },
            "validation_or_test_augmented_rows": sum(
                1
                for row in embedding_rows
                if row["split"] in {"validation", "test"} and row["augmentation_applied"] == "True"
            ),
        },
        "failures": failures,
        "provenance": provenance,
    }


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Load Stage 1 manifest rows."""

    with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
        return [
            {key: (value or "").strip() for key, value in row.items() if key}
            for row in csv.DictReader(csv_file)
        ]


def load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    """Load optional JSON if it exists."""

    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_embedding_csv(path: str | Path, rows: list[dict[str, str]], embedding_dim: int) -> None:
    """Write an embedding table."""

    columns = [
        "case_id",
        "fold",
        "split",
        "label_cspca",
        "sequence",
        "encoder_name",
        "encoder_type",
        "augmentation_applied",
        "image_path",
    ] + [f"embedding_{index:03d}" for index in range(embedding_dim)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
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
    """Format floats consistently for CSV output."""

    return f"{value:.10g}"
