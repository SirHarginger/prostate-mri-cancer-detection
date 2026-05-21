"""Evaluate Prostate158 nnU-Net prediction outputs."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import numpy as np

from prostate_detection.preprocessing.msd_binary_roi import display_path, require_nibabel
from prostate_detection.preprocessing.prostate158 import (
    Prostate158ManifestRecord,
    read_prostate158_manifest,
)


ANATOMY_LABELS = {
    1: "anatomy_label_1",
    2: "anatomy_label_2",
}

METRIC_FIELDS = [
    "case_id",
    "split",
    "label_value",
    "label_name",
    "gt_voxels",
    "pred_voxels",
    "tp",
    "fp",
    "fn",
    "dice",
    "precision",
    "recall",
]


@dataclass(frozen=True)
class LabelMetrics:
    """Per-case binary segmentation metrics for one label."""

    case_id: str
    split: str
    label_value: int
    label_name: str
    gt_voxels: int
    pred_voxels: int
    tp: int
    fp: int
    fn: int
    dice: float
    precision: float
    recall: float

    def to_row(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "split": self.split,
            "label_value": str(self.label_value),
            "label_name": self.label_name,
            "gt_voxels": str(self.gt_voxels),
            "pred_voxels": str(self.pred_voxels),
            "tp": str(self.tp),
            "fp": str(self.fp),
            "fn": str(self.fn),
            "dice": f"{self.dice:.6f}",
            "precision": f"{self.precision:.6f}",
            "recall": f"{self.recall:.6f}",
        }


def binary_dice(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute Dice/F1 for two binary masks."""
    gt = np.asarray(ground_truth, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    gt_count = int(gt.sum())
    pred_count = int(pred.sum())
    denominator = gt_count + pred_count
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(gt, pred).sum())
    return float((2.0 * intersection) / denominator)


def binary_precision_recall(ground_truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    """Compute precision and recall for two binary masks."""
    gt = np.asarray(ground_truth, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())

    precision = 1.0 if tp + fp == 0 else float(tp / (tp + fp))
    recall = 1.0 if tp + fn == 0 else float(tp / (tp + fn))
    return precision, recall


def compute_label_metrics(
    *,
    case_id: str,
    split: str,
    label_value: int,
    label_name: str,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> LabelMetrics:
    """Compute per-label metrics for one case."""
    gt = np.asarray(ground_truth == label_value)
    pred = np.asarray(prediction == label_value)
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    precision, recall = binary_precision_recall(gt, pred)
    return LabelMetrics(
        case_id=case_id,
        split=split,
        label_value=label_value,
        label_name=label_name,
        gt_voxels=int(gt.sum()),
        pred_voxels=int(pred.sum()),
        tp=tp,
        fp=fp,
        fn=fn,
        dice=binary_dice(gt, pred),
        precision=precision,
        recall=recall,
    )


def evaluate_dataset502_predictions(
    *,
    manifest_path: Path,
    labels_dir: Path,
    predictions_dir: Path,
    metrics_csv: Path,
    summary_json: Path,
    overwrite: bool = False,
) -> list[LabelMetrics]:
    """Evaluate Dataset502 anatomy predictions against nnU-Net labels."""
    records = read_prostate158_manifest(manifest_path)
    _ensure_outputs_can_be_written([metrics_csv, summary_json], overwrite=overwrite)

    all_metrics: list[LabelMetrics] = []
    nib = require_nibabel()
    for record in records:
        label_path = labels_dir / f"{record.nnunet_case_id}.nii.gz"
        prediction_path = predictions_dir / f"{record.nnunet_case_id}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing Dataset502 label: {label_path}")
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing Dataset502 prediction: {prediction_path}")

        label_img = nib.load(str(label_path))
        prediction_img = nib.load(str(prediction_path))
        if label_img.shape != prediction_img.shape:
            raise ValueError(
                "Shape mismatch for "
                f"{record.nnunet_case_id}: label {label_img.shape}, "
                f"prediction {prediction_img.shape}"
            )

        label_data = np.asanyarray(label_img.dataobj).astype(np.int16)
        prediction_data = np.asanyarray(prediction_img.dataobj).astype(np.int16)
        label_values = set(np.unique(label_data).astype(int).tolist())
        prediction_values = set(np.unique(prediction_data).astype(int).tolist())
        if not label_values.issubset({0, 1, 2}):
            raise ValueError(f"Unexpected label values in {label_path}: {sorted(label_values)}")
        if not prediction_values.issubset({0, 1, 2}):
            raise ValueError(
                f"Unexpected prediction values in {prediction_path}: {sorted(prediction_values)}"
            )

        for label_value, label_name in ANATOMY_LABELS.items():
            all_metrics.append(
                compute_label_metrics(
                    case_id=record.case_id,
                    split=record.split,
                    label_value=label_value,
                    label_name=label_name,
                    ground_truth=label_data,
                    prediction=prediction_data,
                )
            )

    _write_metrics_csv(all_metrics, metrics_csv)
    _write_summary_json(all_metrics, summary_json)
    return all_metrics


def save_dataset502_qc_figures(
    *,
    manifest_path: Path,
    labels_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
    split: str = "valid",
    max_cases: int = 6,
    overwrite: bool = False,
) -> list[Path]:
    """Save lightweight anatomy overlay figures for visual QC."""
    if max_cases <= 0:
        return []

    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "prostate_mri_matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("matplotlib is required for QC figures") from exc

    records = [record for record in read_prostate158_manifest(manifest_path) if record.split == split]
    if not records:
        raise ValueError(f"No Prostate158 records found for split: {split}")

    output_dir.mkdir(parents=True, exist_ok=True)
    nib = require_nibabel()
    written: list[Path] = []
    for record in records[:max_cases]:
        output_path = output_dir / f"dataset502_{record.nnunet_case_id}_qc.png"
        _ensure_outputs_can_be_written([output_path], overwrite=overwrite)

        image = np.asanyarray(nib.load(str(record.t2w_path)).dataobj).astype(np.float32)
        label = np.asanyarray(
            nib.load(str(labels_dir / f"{record.nnunet_case_id}.nii.gz")).dataobj
        ).astype(np.int16)
        prediction = np.asanyarray(
            nib.load(str(predictions_dir / f"{record.nnunet_case_id}.nii.gz")).dataobj
        ).astype(np.int16)
        if image.shape != label.shape or image.shape != prediction.shape:
            raise ValueError(f"QC shape mismatch for {record.nnunet_case_id}")

        slice_index = _choose_mask_slice((label > 0) | (prediction > 0))
        image_slice = _window_image(image[:, :, slice_index])
        label_slice = label[:, :, slice_index]
        prediction_slice = prediction[:, :, slice_index]

        fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
        panels = [
            ("T2", None),
            ("Ground truth anatomy", label_slice),
            ("Prediction anatomy", prediction_slice),
        ]
        for axis, (title, mask) in zip(axes, panels):
            axis.imshow(np.rot90(image_slice), cmap="gray")
            if mask is not None:
                _overlay_label(axis, mask)
            axis.set_title(f"{record.nnunet_case_id} {title}")
            axis.axis("off")
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        written.append(output_path)
    return written


def summarize_metrics(metrics: list[LabelMetrics]) -> dict[str, Any]:
    """Summarize metrics by split and label."""
    summary: dict[str, Any] = {"labels": ANATOMY_LABELS, "splits": {}}
    splits = sorted({metric.split for metric in metrics})
    for split in [*splits, "all"]:
        split_metrics = metrics if split == "all" else [metric for metric in metrics if metric.split == split]
        split_summary: dict[str, Any] = {"n_cases": len({metric.case_id for metric in split_metrics})}
        label_means: list[float] = []
        for label_value, label_name in ANATOMY_LABELS.items():
            label_metrics = [
                metric for metric in split_metrics if metric.label_value == label_value
            ]
            dice_values = [metric.dice for metric in label_metrics]
            precision_values = [metric.precision for metric in label_metrics]
            recall_values = [metric.recall for metric in label_metrics]
            if not dice_values:
                continue
            label_means.append(mean(dice_values))
            split_summary[label_name] = {
                "label_value": label_value,
                "n": len(dice_values),
                "mean_dice": mean(dice_values),
                "median_dice": median(dice_values),
                "std_dice": pstdev(dice_values) if len(dice_values) > 1 else 0.0,
                "mean_precision": mean(precision_values),
                "mean_recall": mean(recall_values),
            }
        split_summary["macro_mean_dice"] = mean(label_means) if label_means else None
        summary["splits"][split] = split_summary
    return summary


def _ensure_outputs_can_be_written(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output exists and --overwrite was not set: {preview}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _write_metrics_csv(metrics: list[LabelMetrics], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric.to_row())


def _write_summary_json(metrics: list[LabelMetrics], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_metrics(metrics)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def _choose_mask_slice(mask: np.ndarray) -> int:
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask.shape}")
    areas = mask.reshape((-1, mask.shape[-1])).sum(axis=0)
    if int(areas.max()) == 0:
        return mask.shape[-1] // 2
    return int(np.argmax(areas))


def _window_image(image_slice: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image_slice, [1, 99])
    if high <= low:
        return image_slice
    return np.clip((image_slice - low) / (high - low), 0, 1)


def _overlay_label(axis: Any, mask_slice: np.ndarray) -> None:
    rotated = np.rot90(mask_slice)
    label_1 = np.ma.masked_where(rotated != 1, rotated)
    label_2 = np.ma.masked_where(rotated != 2, rotated)
    axis.imshow(label_1, cmap="spring", alpha=0.35, vmin=0, vmax=2)
    axis.imshow(label_2, cmap="winter", alpha=0.35, vmin=0, vmax=2)


def print_summary(metrics: list[LabelMetrics], summary_json: Path, metrics_csv: Path) -> None:
    """Print concise evaluation summary for CLI users."""
    summary = summarize_metrics(metrics)
    valid = summary["splits"].get("valid", {})
    print(f"Wrote metrics CSV: {display_path(metrics_csv)}")
    print(f"Wrote summary JSON: {display_path(summary_json)}")
    if valid:
        print("Validation split summary:")
        print(f"  cases: {valid.get('n_cases')}")
        print(f"  macro_mean_dice: {valid.get('macro_mean_dice'):.4f}")
        for label_name in ANATOMY_LABELS.values():
            label_summary = valid.get(label_name)
            if label_summary:
                print(f"  {label_name} mean_dice: {label_summary['mean_dice']:.4f}")
