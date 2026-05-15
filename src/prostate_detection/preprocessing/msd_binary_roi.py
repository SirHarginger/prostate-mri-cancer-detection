"""Convert MSD prostate zone labels into binary whole-prostate ROI masks."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MsdTrainingCase:
    """Resolved MSD training image and label paths."""

    case_id: str
    image_path: Path
    original_label_path: Path


@dataclass(frozen=True)
class BinaryRoiRecord:
    """Manifest row for one converted binary ROI mask."""

    case_id: str
    image_path: Path
    original_label_path: Path
    binary_label_path: Path


def strip_nii_suffix(path: Path) -> str:
    """Return a case identifier from a NIfTI path."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def is_nifti_file(path: Path) -> bool:
    """Return true for NIfTI files, ignoring macOS AppleDouble sidecar files."""
    return not path.name.startswith("._") and (
        path.name.endswith(".nii") or path.name.endswith(".nii.gz")
    )


def load_dataset_json(input_dir: Path) -> dict[str, Any]:
    """Load MSD dataset metadata."""
    dataset_json_path = input_dir / "dataset.json"
    if not dataset_json_path.is_file():
        raise FileNotFoundError(f"Missing dataset.json: {dataset_json_path}")
    with dataset_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_msd_path(input_dir: Path, value: str) -> Path:
    """Resolve an MSD dataset.json relative path under the dataset directory."""
    return input_dir / value.removeprefix("./")


def discover_training_cases(input_dir: Path) -> list[MsdTrainingCase]:
    """Read and validate MSD training image/label pairs."""
    metadata = load_dataset_json(input_dir)
    training = metadata.get("training")
    if not isinstance(training, list):
        raise ValueError("dataset.json must contain a list field named 'training'")

    num_training = metadata.get("numTraining")
    if num_training is not None and int(num_training) != len(training):
        raise ValueError(
            f"dataset.json numTraining={num_training} but training has {len(training)} entries"
        )

    images_tr = input_dir / "imagesTr"
    labels_tr = input_dir / "labelsTr"
    if not images_tr.is_dir():
        raise FileNotFoundError(f"Missing imagesTr directory: {images_tr}")
    if not labels_tr.is_dir():
        raise FileNotFoundError(f"Missing labelsTr directory: {labels_tr}")

    image_files = sorted(path for path in images_tr.iterdir() if is_nifti_file(path))
    label_files = sorted(path for path in labels_tr.iterdir() if is_nifti_file(path))
    if len(image_files) != len(training):
        raise ValueError(
            f"imagesTr contains {len(image_files)} NIfTI files but dataset.json lists "
            f"{len(training)} training entries"
        )
    if len(label_files) != len(training):
        raise ValueError(
            f"labelsTr contains {len(label_files)} NIfTI files but dataset.json lists "
            f"{len(training)} training entries"
        )

    cases: list[MsdTrainingCase] = []
    seen_case_ids: set[str] = set()
    for item in training:
        if not isinstance(item, dict) or "image" not in item or "label" not in item:
            raise ValueError("Each training entry must contain 'image' and 'label' paths")
        image_path = resolve_msd_path(input_dir, str(item["image"]))
        label_path = resolve_msd_path(input_dir, str(item["label"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing training image: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing training label: {label_path}")

        image_case_id = strip_nii_suffix(image_path)
        label_case_id = strip_nii_suffix(label_path)
        if image_case_id != label_case_id:
            raise ValueError(
                f"Image/label case mismatch: {image_path.name} vs {label_path.name}"
            )
        if image_case_id in seen_case_ids:
            raise ValueError(f"Duplicate training case_id: {image_case_id}")
        seen_case_ids.add(image_case_id)
        cases.append(
            MsdTrainingCase(
                case_id=image_case_id,
                image_path=image_path,
                original_label_path=label_path,
            )
        )
    return cases


def require_nibabel():
    """Import nibabel with a clear medical-imaging dependency error."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "nibabel is required to read and write NIfTI files. Install project "
            "requirements before running this converter."
        ) from exc
    return nib


def _import_nibabel():
    return require_nibabel()


def convert_label_to_binary_roi(original_label_path: Path, binary_label_path: Path) -> None:
    """Convert one label volume to a uint8 whole-prostate ROI mask."""
    nib = _import_nibabel()
    label_img = nib.load(str(original_label_path))
    label_data = np.asanyarray(label_img.dataobj)
    binary_data = (label_data != 0).astype(np.uint8)

    header = label_img.header.copy()
    header.set_data_dtype(np.uint8)
    binary_img = nib.Nifti1Image(binary_data, label_img.affine, header=header)
    binary_img.set_qform(label_img.get_qform(), code=int(label_img.header["qform_code"]))
    binary_img.set_sform(label_img.get_sform(), code=int(label_img.header["sform_code"]))

    binary_label_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(binary_img, str(binary_label_path))


def display_path(path: Path, base_dir: Path | None = None) -> str:
    """Return a portable path string, relative to base_dir when possible."""
    if base_dir is None:
        base_dir = Path.cwd()
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_manifest(records: list[BinaryRoiRecord], manifest_path: Path) -> None:
    """Write the binary ROI conversion manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "original_label_path",
                "binary_label_path",
                "case_id",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": display_path(record.image_path),
                    "original_label_path": display_path(record.original_label_path),
                    "binary_label_path": display_path(record.binary_label_path),
                    "case_id": record.case_id,
                }
            )


def convert_msd_labels_to_binary_roi(
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[BinaryRoiRecord]:
    """Convert all MSD training labels into binary whole-prostate ROI masks."""
    cases = discover_training_cases(input_dir)
    binary_label_dir = output_dir / "labelsTr"
    manifest_path = output_dir / "manifest.csv"

    records = [
        BinaryRoiRecord(
            case_id=case.case_id,
            image_path=case.image_path,
            original_label_path=case.original_label_path,
            binary_label_path=binary_label_dir / case.original_label_path.name,
        )
        for case in cases
    ]

    existing_outputs = [
        record.binary_label_path
        for record in records
        if record.binary_label_path.exists() and not overwrite
    ]
    if manifest_path.exists() and not overwrite:
        existing_outputs.append(manifest_path)
    if existing_outputs:
        preview = ", ".join(str(path) for path in existing_outputs[:5])
        raise FileExistsError(
            f"Output exists and --overwrite was not set. Existing paths: {preview}"
        )

    if dry_run:
        return records

    for record in records:
        convert_label_to_binary_roi(record.original_label_path, record.binary_label_path)
    write_manifest(records, manifest_path)
    return records
