"""Visual QC for Kaggle PROSTATE_MRI auto-segmentation outputs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from prostate_detection.preprocessing.msd_binary_roi import display_path, require_nibabel


def save_kaggle_auto_segmentation_qc(
    *,
    manifest_path: Path,
    predictions_dir: Path,
    output_dir: Path,
    max_cases: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Save T2 plus auto-segmentation overlays for Kaggle PROSTATE_MRI cases."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "prostate_mri_matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("matplotlib is required for QC figures") from exc

    rows = _read_manifest(manifest_path)
    if max_cases is not None:
        rows = rows[:max_cases]

    nib = require_nibabel()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for row in rows:
        case_id = row["case_id"]
        image_path = Path(row["image_path"])
        prediction_path = predictions_dir / f"{case_id}.nii.gz"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing converted Kaggle image: {image_path}")
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing Kaggle auto-segmentation: {prediction_path}")

        output_path = output_dir / f"{case_id}_auto_segmentation_qc.png"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"QC figure exists and --overwrite was not set: {output_path}")

        image = np.asanyarray(nib.load(str(image_path)).dataobj).astype(np.float32)
        prediction = np.asanyarray(nib.load(str(prediction_path)).dataobj).astype(np.int16)
        if image.shape != prediction.shape:
            raise ValueError(f"Shape mismatch for {case_id}: {image.shape} vs {prediction.shape}")

        slice_index = choose_prediction_slice(prediction)
        image_slice = _window_image(image[:, :, slice_index])
        prediction_slice = prediction[:, :, slice_index]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        axes[0].imshow(np.rot90(image_slice), cmap="gray")
        axes[0].set_title(f"{case_id} T2")
        axes[0].axis("off")
        axes[1].imshow(np.rot90(image_slice), cmap="gray")
        _overlay_prediction(axes[1], prediction_slice)
        axes[1].set_title("Auto anatomy mask")
        axes[1].axis("off")
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        written.append(output_path)
    return written


def choose_prediction_slice(prediction: np.ndarray) -> int:
    """Choose the axial slice with the largest predicted anatomy area."""
    if prediction.ndim != 3:
        raise ValueError(f"Expected 3D prediction, got shape {prediction.shape}")
    mask = prediction > 0
    areas = mask.reshape((-1, mask.shape[-1])).sum(axis=0)
    if int(areas.max()) == 0:
        return prediction.shape[-1] // 2
    return int(np.argmax(areas))


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Kaggle T2 manifest: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _window_image(image_slice: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image_slice, [1, 99])
    if high <= low:
        return image_slice
    return np.clip((image_slice - low) / (high - low), 0, 1)


def _overlay_prediction(axis: Any, prediction_slice: np.ndarray) -> None:
    rotated = np.rot90(prediction_slice)
    label_1 = np.ma.masked_where(rotated != 1, rotated)
    label_2 = np.ma.masked_where(rotated != 2, rotated)
    axis.imshow(label_1, cmap="spring", alpha=0.35, vmin=0, vmax=2)
    axis.imshow(label_2, cmap="winter", alpha=0.35, vmin=0, vmax=2)


def print_qc_summary(paths: list[Path], output_dir: Path) -> None:
    print(f"Wrote {len(paths)} Kaggle auto-segmentation QC figures to {display_path(output_dir)}")
