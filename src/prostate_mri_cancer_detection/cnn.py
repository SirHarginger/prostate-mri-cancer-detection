"""Minimal split-safe CNN smoke training for PI-CAI bpMRI.

The functions in this module are intentionally small. They validate the real
CNN data path before expensive training: T2W is the reference grid, ADC/HBV are
resampled in memory, whole-gland masks choose a prostate-centered slice, and
augmentation is restricted to the training split.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from prostate_mri_cancer_detection.evaluation import (
    classification_metrics,
    parse_label,
    summarize_prediction_group,
)
from prostate_mri_cancer_detection.preprocessing import (
    image_has_positive_voxels,
    load_manifest_rows,
    resample_to_reference,
    resolve_manifest_path,
    signatures_match,
    simpleitk_signature,
    split_pipe_value,
)


SPLIT_BY_FOLD = {
    "fold0": "train",
    "fold1": "train",
    "fold2": "train",
    "fold3": "validation",
    "fold4": "test",
}

CNN_FEATURE_COLUMNS = [f"cnn_embedding_{index:03d}" for index in range(32)]


def run_cnn_smoke_training(
    manifest_path: str | Path,
    raw_root: str | Path,
    embeddings_path: str | Path,
    predictions_path: str | Path,
    report_path: str | Path,
    model_path: str | Path,
    sample_size_per_split: int = 12,
    image_size: int = 64,
    max_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    embedding_dim: int = 32,
    augment_train: bool = False,
    all_cases: bool = False,
    case_ids: Iterable[str] | None = None,
    target_sensitivity: float = 0.90,
    seed: int = 42,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Run a tiny multisequence CNN smoke training pass."""

    try:
        import numpy as np
        import SimpleITK as sitk
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:  # pragma: no cover - depends on cluster env.
        raise RuntimeError("NumPy, SimpleITK, and PyTorch are required for CNN smoke training") from error

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    manifest_path = Path(manifest_path)
    raw_root = Path(raw_root)
    manifest_rows = load_manifest_rows(manifest_path)
    selected_rows = select_cnn_rows(
        rows=manifest_rows,
        sample_size_per_split=sample_size_per_split,
        all_cases=all_cases,
        case_ids=case_ids,
    )
    examples, failures = prepare_cnn_examples(
        rows=selected_rows,
        raw_root=raw_root,
        image_size=image_size,
        sitk=sitk,
        np_module=np,
    )
    split_examples = {
        split: [example for example in examples if example["split"] == split]
        for split in ("train", "validation", "test")
    }
    train_labels = {example["label"] for example in split_examples["train"]}
    if train_labels != {0, 1}:
        raise ValueError("CNN smoke training requires positive and negative training examples")

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_class = build_tensor_dataset_class(Dataset, torch)
    train_dataset = dataset_class(
        split_examples["train"],
        augment=augment_train,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, batch_size),
        shuffle=True,
        num_workers=0,
    )

    model = TinyMultisequenceCNN(embedding_dim=embedding_dim, nn=nn).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pos_weight = positive_class_weight(split_examples["train"], torch, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    epoch_losses = []
    for _epoch in range(max(0, max_epochs)):
        model.train()
        losses = []
        for images, labels, _case_ids in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _embeddings = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        epoch_losses.append(sum(losses) / len(losses) if losses else None)

    prediction_rows, embedding_rows = evaluate_cnn_examples(
        examples=examples,
        model=model,
        dataset_class=dataset_class,
        DataLoader=DataLoader,
        torch=torch,
        device=device,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
    )
    metrics_by_split = {
        split: summarize_prediction_group(
            [row for row in prediction_rows if row["split"] == split],
            target_sensitivity=target_sensitivity,
        )
        for split in ("train", "validation", "test")
    }
    summary = {
        "selected_cases": len(selected_rows),
        "examples_loaded": len(examples),
        "failures": len(failures),
        "embeddings_written": len(embedding_rows),
        "predictions_written": len(prediction_rows),
        "validation_or_test_augmented_rows": 0,
    }
    report = {
        "schema_version": "1.0",
        "stage": "cnn_smoke_training",
        "manifest_path": str(manifest_path),
        "raw_root": str(raw_root),
        "summary": summary,
        "case_counts": {
            "manifest": len(manifest_rows),
            "selected": len(selected_rows),
            "loaded": len(examples),
            "by_split": dict(Counter(example["split"] for example in examples)),
        },
        "label_counts": dict(Counter(str(example["label"]) for example in examples)),
        "split_label_counts": {
            split: dict(Counter(str(example["label"]) for example in split_examples[split]))
            for split in ("train", "validation", "test")
        },
        "model": {
            "name": "TinyMultisequenceCNN",
            "training_status": "smoke_trained",
            "input_channels": 3,
            "input_sequences": ["t2w", "adc_resampled_to_t2w", "hbv_resampled_to_t2w"],
            "image_size": image_size,
            "embedding_dim": embedding_dim,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "device": str(device),
            "epoch_losses": epoch_losses,
        },
        "split_policy": SPLIT_BY_FOLD,
        "augmentation_policy": {
            "train_augmentation_enabled": augment_train,
            "validation_augmentation_enabled": False,
            "test_augmentation_enabled": False,
            "saved_augmented_copies": False,
            "transform": "deterministic horizontal flip and small intensity scale for selected training cases",
        },
        "preprocessing_policy": {
            "reference_grid": "T2W",
            "adc": "resampled in memory to T2W grid",
            "hbv": "resampled in memory to T2W grid",
            "roi": "non-empty T2W-grid whole-gland mask selects the axial slice and crop when available",
            "normalization": "per-case per-sequence percentile clipping followed by z-score normalization",
            "writes_processed_images": False,
        },
        "metrics": metrics_by_split,
        "overall_loaded_metrics": classification_metrics(prediction_rows),
        "failures": failures,
        "output_paths": {
            "embeddings": str(embeddings_path),
            "predictions": str(predictions_path),
            "report": str(report_path),
            "model": str(model_path),
        },
        "claim_limits": [
            "This is a CNN smoke test for data loading, split safety, and training mechanics.",
            "It is not a final CNN baseline and should not be compared against full radiomics results.",
            "No clinical, localization, deployment, or biopsy-reduction claim is supported by this run.",
        ],
    }

    write_embedding_csv(embeddings_path, embedding_rows, embedding_dim)
    write_prediction_csv(predictions_path, prediction_rows)
    write_model_checkpoint(
        path=model_path,
        model=model,
        torch=torch,
        report=report,
    )
    write_json(report_path, report)
    return report


def select_cnn_rows(
    rows: list[dict[str, str]],
    sample_size_per_split: int,
    all_cases: bool,
    case_ids: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    """Select labeled rows deterministically while preserving splits."""

    requested_case_ids = sorted(set(case_id for case_id in case_ids or [] if case_id))
    labeled_rows = [
        row for row in rows
        if row.get("case_id") and parse_label(row.get("label_cspca", "")) is not None
    ]
    rows_by_case = {row["case_id"]: row for row in labeled_rows}
    if requested_case_ids:
        return [rows_by_case[case_id] for case_id in requested_case_ids if case_id in rows_by_case]

    labeled_rows = sorted(labeled_rows, key=lambda row: row["case_id"])
    if all_cases:
        return labeled_rows

    selected = []
    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in labeled_rows:
        rows_by_split[SPLIT_BY_FOLD.get(row.get("fold", ""), "unknown")].append(row)
    for split in ("train", "validation", "test"):
        selected.extend(select_balanced_rows(rows_by_split[split], sample_size_per_split))
    return selected


def select_balanced_rows(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Select up to limit rows with both labels represented when possible."""

    if limit <= 0 or len(rows) <= limit:
        return rows
    positives = [row for row in rows if parse_label(row.get("label_cspca", "")) == 1]
    negatives = [row for row in rows if parse_label(row.get("label_cspca", "")) == 0]
    first_quota = limit // 2
    selected = positives[:first_quota] + negatives[: limit - first_quota]
    if len(selected) < limit:
        selected_ids = {row["case_id"] for row in selected}
        selected.extend(row for row in rows if row["case_id"] not in selected_ids)
    return sorted(selected[:limit], key=lambda row: row["case_id"])


def prepare_cnn_examples(
    rows: list[dict[str, str]],
    raw_root: Path,
    image_size: int,
    sitk: Any,
    np_module: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Load selected rows into fixed-size multisequence tensors."""

    examples = []
    failures = []
    for row in rows:
        split = SPLIT_BY_FOLD.get(row.get("fold", ""), "unknown")
        try:
            tensor, metadata = load_case_tensor(
                row=row,
                raw_root=raw_root,
                image_size=image_size,
                sitk=sitk,
                np_module=np_module,
            )
            examples.append(
                {
                    "case_id": row["case_id"],
                    "fold": row.get("fold", ""),
                    "split": split,
                    "label": int(parse_label(row.get("label_cspca", ""))),
                    "label_cspca": row.get("label_cspca", ""),
                    "tensor": tensor,
                    "metadata": metadata,
                }
            )
        except Exception as error:  # noqa: BLE001 - per-case failures belong in report.
            failures.append(
                {
                    "case_id": row.get("case_id", ""),
                    "fold": row.get("fold", ""),
                    "split": split,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    return examples, failures


def load_case_tensor(
    row: dict[str, str],
    raw_root: Path,
    image_size: int,
    sitk: Any,
    np_module: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load one case as a three-channel T2W-grid 2D tensor."""

    t2w_path = resolve_manifest_path(row.get("path_t2w", ""), raw_root)
    adc_path = resolve_manifest_path(row.get("path_adc", ""), raw_root)
    hbv_path = resolve_manifest_path(row.get("path_hbv", ""), raw_root)
    reference = sitk.ReadImage(str(t2w_path))
    adc = resample_to_reference(sitk.ReadImage(str(adc_path)), reference, sitk.sitkLinear, sitk)
    hbv = resample_to_reference(sitk.ReadImage(str(hbv_path)), reference, sitk.sitkLinear, sitk)
    mask, mask_path = find_reference_gland_mask(row, raw_root, reference, sitk)

    arrays = {
        "t2w": np_module.asarray(sitk.GetArrayFromImage(reference), dtype=np_module.float32),
        "adc": np_module.asarray(sitk.GetArrayFromImage(adc), dtype=np_module.float32),
        "hbv": np_module.asarray(sitk.GetArrayFromImage(hbv), dtype=np_module.float32),
    }
    mask_array = None
    if mask is not None:
        mask_array = np_module.asarray(sitk.GetArrayFromImage(mask) > 0)

    slice_index = choose_slice_index(arrays["t2w"], mask_array, np_module)
    mask_slice = mask_array[slice_index] if mask_array is not None else None
    channels = []
    for sequence in ("t2w", "adc", "hbv"):
        normalized = normalize_slice(arrays[sequence][slice_index], mask_slice, np_module)
        channels.append(crop_resize_2d(normalized, mask_slice, image_size, np_module))

    tensor = np_module.stack(channels, axis=0).astype(np_module.float32)
    metadata = {
        "slice_index": int(slice_index),
        "used_gland_mask": mask_path is not None,
        "gland_mask_path": str(mask_path or ""),
        "source_paths": {
            "t2w": str(t2w_path),
            "adc": str(adc_path),
            "hbv": str(hbv_path),
        },
    }
    return tensor, metadata


def find_reference_gland_mask(row: dict[str, str], raw_root: Path, reference: Any, sitk: Any) -> tuple[Any | None, Path | None]:
    """Return the first non-empty whole-gland mask already on the T2W grid."""

    reference_signature = simpleitk_signature(reference)
    for value in split_pipe_value(row.get("path_gland_mask", "")):
        mask_path = resolve_manifest_path(value, raw_root)
        mask = sitk.ReadImage(str(mask_path))
        if signatures_match(simpleitk_signature(mask), reference_signature) and image_has_positive_voxels(mask, sitk):
            return mask, mask_path
    return None, None


def choose_slice_index(volume: Any, mask: Any | None, np_module: Any) -> int:
    """Choose the axial slice for a case."""

    if mask is not None and bool(mask.any()):
        per_slice = mask.reshape(mask.shape[0], -1).sum(axis=1)
        return int(np_module.argmax(per_slice))
    return int(volume.shape[0] // 2)


def normalize_slice(slice_2d: Any, mask_2d: Any | None, np_module: Any) -> Any:
    """Normalize a slice using per-case foreground statistics."""

    values = slice_2d[mask_2d] if mask_2d is not None and bool(mask_2d.any()) else slice_2d.reshape(-1)
    low, high = np_module.percentile(values, [1, 99])
    if math.isclose(float(low), float(high)):
        low = float(values.min())
        high = float(values.max())
    clipped = np_module.clip(slice_2d, low, high)
    clipped_values = clipped[mask_2d] if mask_2d is not None and bool(mask_2d.any()) else clipped.reshape(-1)
    mean = float(clipped_values.mean())
    std = float(clipped_values.std()) or 1.0
    return (clipped - mean) / std


def crop_resize_2d(slice_2d: Any, mask_2d: Any | None, image_size: int, np_module: Any) -> Any:
    """Crop around the gland mask when available and resize with nearest sampling."""

    if mask_2d is not None and bool(mask_2d.any()):
        y_indices, x_indices = np_module.where(mask_2d)
        y_min, y_max = int(y_indices.min()), int(y_indices.max()) + 1
        x_min, x_max = int(x_indices.min()), int(x_indices.max()) + 1
        y_pad = max(1, int((y_max - y_min) * 0.25))
        x_pad = max(1, int((x_max - x_min) * 0.25))
        y_min = max(0, y_min - y_pad)
        y_max = min(slice_2d.shape[0], y_max + y_pad)
        x_min = max(0, x_min - x_pad)
        x_max = min(slice_2d.shape[1], x_max + x_pad)
        cropped = slice_2d[y_min:y_max, x_min:x_max]
    else:
        cropped = center_crop(slice_2d)
    return resize_nearest(cropped, image_size, np_module)


def center_crop(slice_2d: Any) -> Any:
    """Return a square center crop."""

    height, width = slice_2d.shape
    size = min(height, width)
    y_start = (height - size) // 2
    x_start = (width - size) // 2
    return slice_2d[y_start : y_start + size, x_start : x_start + size]


def resize_nearest(slice_2d: Any, image_size: int, np_module: Any) -> Any:
    """Resize a 2D array with deterministic nearest-neighbor indexing."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if slice_2d.size == 0:
        raise ValueError("cannot resize an empty crop")
    y_index = np_module.linspace(0, slice_2d.shape[0] - 1, image_size).round().astype(int)
    x_index = np_module.linspace(0, slice_2d.shape[1] - 1, image_size).round().astype(int)
    return slice_2d[y_index[:, None], x_index[None, :]]


def build_tensor_dataset_class(Dataset: Any, torch: Any) -> Any:
    """Build a tiny torch Dataset class without importing torch at module load."""

    class TensorDataset(Dataset):
        def __init__(self, examples: list[dict[str, Any]], augment: bool, seed: int) -> None:
            self.examples = examples
            self.augment = augment
            self.seed = seed

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> tuple[Any, Any, str]:
            example = self.examples[index]
            tensor = torch.as_tensor(example["tensor"], dtype=torch.float32)
            if self.augment and example["split"] == "train":
                tensor = augment_tensor(tensor, example["case_id"], self.seed, torch)
            label = torch.tensor(float(example["label"]), dtype=torch.float32)
            return tensor, label, example["case_id"]

    return TensorDataset


def augment_tensor(tensor: Any, case_id: str, seed: int, torch: Any) -> Any:
    """Apply deterministic train-only augmentation."""

    stable_value = sum(ord(char) for char in f"{case_id}:{seed}")
    if stable_value % 2 == 0:
        tensor = torch.flip(tensor, dims=[2])
    scale = 1.0 + ((stable_value % 5) - 2) * 0.01
    return tensor * scale


class TinyMultisequenceCNN:
    """Small torch module wrapper created with injected nn module."""

    def __new__(cls, embedding_dim: int, nn: Any) -> Any:
        class _TinyMultisequenceCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 8, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(8, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.embedding = nn.Linear(16, embedding_dim)
                self.classifier = nn.Linear(embedding_dim, 1)

            def forward(self, images: Any) -> tuple[Any, Any]:
                pooled = self.features(images).flatten(1)
                embeddings = self.embedding(pooled)
                logits = self.classifier(nn.functional.relu(embeddings)).squeeze(1)
                return logits, embeddings

        return _TinyMultisequenceCNN()


def positive_class_weight(examples: list[dict[str, Any]], torch: Any, device: Any) -> Any:
    """Compute BCE positive-class weight from training examples."""

    positives = sum(1 for example in examples if example["label"] == 1)
    negatives = sum(1 for example in examples if example["label"] == 0)
    weight = negatives / positives if positives else 1.0
    return torch.tensor([weight], dtype=torch.float32, device=device)


def evaluate_cnn_examples(
    examples: list[dict[str, Any]],
    model: Any,
    dataset_class: Any,
    DataLoader: Any,
    torch: Any,
    device: Any,
    batch_size: int,
    embedding_dim: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Run unaugmented inference and return prediction and embedding rows."""

    dataset = dataset_class(examples, augment=False, seed=0)
    loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=False, num_workers=0)
    examples_by_case = {example["case_id"]: example for example in examples}
    prediction_rows = []
    embedding_rows = []
    model.eval()
    with torch.no_grad():
        for images, _labels, case_ids in loader:
            images = images.to(device)
            logits, embeddings = model(images)
            probabilities = torch.sigmoid(logits).detach().cpu().tolist()
            embedding_values = embeddings.detach().cpu().tolist()
            for case_id, probability, embedding in zip(case_ids, probabilities, embedding_values):
                example = examples_by_case[str(case_id)]
                prediction = 1 if float(probability) >= 0.5 else 0
                prediction_rows.append(
                    {
                        "baseline": "cnn_smoke_multisequence",
                        "case_id": example["case_id"],
                        "fold": example["fold"],
                        "split": example["split"],
                        "label": str(example["label"]),
                        "score": format_float(float(probability)),
                        "probability": format_float(float(probability)),
                        "prediction": str(prediction),
                        "status": "ok",
                        "reason": "",
                    }
                )
                embedding_row = {
                    "case_id": example["case_id"],
                    "fold": example["fold"],
                    "split": example["split"],
                    "label_cspca": example["label_cspca"],
                    "encoder_name": "tiny_multisequence_cnn_smoke_v1",
                    "encoder_type": "smoke_trained_cnn",
                    "augmentation_applied": "False",
                }
                for index in range(embedding_dim):
                    embedding_row[f"cnn_embedding_{index:03d}"] = format_float(float(embedding[index]))
                embedding_rows.append(embedding_row)
    return prediction_rows, embedding_rows


def write_embedding_csv(path: str | Path, rows: list[dict[str, str]], embedding_dim: int) -> None:
    """Write CNN embedding rows."""

    fieldnames = [
        "case_id",
        "fold",
        "split",
        "label_cspca",
        "encoder_name",
        "encoder_type",
        "augmentation_applied",
    ] + [f"cnn_embedding_{index:03d}" for index in range(embedding_dim)]
    write_csv(path, rows, fieldnames)


def write_prediction_csv(path: str | Path, rows: list[dict[str, str]]) -> None:
    """Write CNN prediction rows."""

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
    write_csv(path, rows, fieldnames)


def write_csv(path: str | Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write CSV rows."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_model_checkpoint(path: str | Path, model: Any, torch: Any, report: dict[str, Any]) -> None:
    """Write a small smoke-test model checkpoint under ignored outputs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stage": report["stage"],
            "model": report["model"],
            "split_policy": report["split_policy"],
            "augmentation_policy": report["augmentation_policy"],
            "claim_limits": report["claim_limits"],
        },
        str(path),
    )


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
