"""Prepare Kaggle PROSTATE_MRI DICOM series for nnU-Net inference."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prostate_detection.preprocessing.msd_binary_roi import display_path, require_nibabel


KAGGLE_EXPECTED_SUBJECTS = 26
KAGGLE_T2_AXIAL_SERIES_DESCRIPTION = "T2 TSE ax hi"

KAGGLE_MANIFEST_FIELDS = [
    "case_id",
    "subject_id",
    "series_uid",
    "series_description",
    "source_series_dir",
    "image_path",
    "number_of_dicoms",
    "shape",
    "spacing",
    "source",
    "notes",
]


@dataclass(frozen=True)
class KaggleProstateSeries:
    """Selected Kaggle prostate MRI DICOM series for one subject."""

    case_id: str
    subject_id: str
    series_uid: str
    series_description: str
    series_dir: Path
    number_of_dicoms: int

    @property
    def nnunet_case_id(self) -> str:
        return self.case_id


@dataclass(frozen=True)
class ConvertedKaggleSeries:
    """Converted NIfTI output metadata for one Kaggle series."""

    series: KaggleProstateSeries
    image_path: Path
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    notes: str


def build_kaggle_t2_inference_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    series_description: str = KAGGLE_T2_AXIAL_SERIES_DESCRIPTION,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[ConvertedKaggleSeries]:
    """Select axial T2 DICOM series and convert them to nnU-Net imagesTs."""
    series = select_kaggle_t2_series(input_dir, series_description=series_description)
    expected_paths = [
        output_dir / "imagesTs" / f"{item.nnunet_case_id}_0000.nii.gz" for item in series
    ]
    expected_paths.append(manifest_path)
    _ensure_outputs_can_be_written(expected_paths, overwrite=overwrite)

    if dry_run:
        for item in series:
            print(
                f"Dry run: {item.subject_id} {item.series_description} "
                f"with {item.number_of_dicoms} DICOM files"
            )
        return []

    converted: list[ConvertedKaggleSeries] = []
    for item in series:
        output_path = output_dir / "imagesTs" / f"{item.nnunet_case_id}_0000.nii.gz"
        print(f"Converting {item.subject_id} -> {display_path(output_path)}")
        shape, spacing = convert_dicom_series_to_nifti(item.series_dir, output_path)
        converted.append(
            ConvertedKaggleSeries(
                series=item,
                image_path=output_path,
                shape=shape,
                spacing=spacing,
                notes="converted_from_t2_tse_ax_hi_dicom",
            )
        )

    write_kaggle_t2_manifest(converted, manifest_path)
    print(f"Wrote Kaggle T2 manifest: {display_path(manifest_path)}")
    return converted


def select_kaggle_t2_series(
    input_dir: Path,
    *,
    series_description: str = KAGGLE_T2_AXIAL_SERIES_DESCRIPTION,
) -> list[KaggleProstateSeries]:
    """Read metadata.csv and select one axial T2 series per subject."""
    metadata_path = _require_file(input_dir / "metadata.csv", "Kaggle PROSTATE_MRI metadata.csv")
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if row["Series Description"].strip() == series_description
        ]

    if len(rows) != KAGGLE_EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {KAGGLE_EXPECTED_SUBJECTS} {series_description!r} series, "
            f"found {len(rows)}"
        )

    selected: list[KaggleProstateSeries] = []
    seen_subjects: set[str] = set()
    for row in sorted(rows, key=lambda item: item["Subject ID"]):
        subject_id = row["Subject ID"].strip()
        if subject_id in seen_subjects:
            raise ValueError(f"Duplicate selected series for subject {subject_id}")
        seen_subjects.add(subject_id)

        series_dir = _resolve_series_dir(input_dir, row["File Location"])
        dicom_count = len(list(series_dir.glob("*.dcm")))
        expected_count = int(row["Number of Images"])
        if dicom_count != expected_count:
            raise ValueError(
                f"DICOM count mismatch for {subject_id}: metadata {expected_count}, "
                f"files {dicom_count}"
            )

        selected.append(
            KaggleProstateSeries(
                case_id=_case_id_from_subject(subject_id),
                subject_id=subject_id,
                series_uid=row["Series UID"],
                series_description=row["Series Description"],
                series_dir=series_dir,
                number_of_dicoms=dicom_count,
            )
        )
    return selected


def convert_dicom_series_to_nifti(series_dir: Path, output_path: Path) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """Convert one single-frame DICOM series into a 3D NIfTI image."""
    pydicom = _require_pydicom()
    dicom_paths = sorted(series_dir.glob("*.dcm"))
    if not dicom_paths:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")

    datasets = [pydicom.dcmread(str(path), stop_before_pixels=False) for path in dicom_paths]
    datasets = sorted(datasets, key=_slice_sort_key)

    arrays = [_scaled_pixel_array(ds) for ds in datasets]
    first_shape = arrays[0].shape
    if any(array.shape != first_shape for array in arrays):
        raise ValueError(f"Inconsistent slice shapes in {series_dir}")

    volume = np.stack(arrays, axis=-1).astype(np.float32)
    spacing = _spacing_from_datasets(datasets)
    affine = _affine_from_datasets(datasets, spacing)

    nib = require_nibabel()
    image = nib.Nifti1Image(volume, affine)
    image.header.set_data_dtype(np.float32)
    image.header.set_zooms(spacing)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))
    return tuple(int(value) for value in volume.shape), spacing


def write_kaggle_t2_manifest(
    converted: list[ConvertedKaggleSeries],
    manifest_path: Path,
) -> None:
    """Write converted Kaggle T2 inference manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KAGGLE_MANIFEST_FIELDS)
        writer.writeheader()
        for item in converted:
            writer.writerow(
                {
                    "case_id": item.series.case_id,
                    "subject_id": item.series.subject_id,
                    "series_uid": item.series.series_uid,
                    "series_description": item.series.series_description,
                    "source_series_dir": display_path(item.series.series_dir),
                    "image_path": display_path(item.image_path),
                    "number_of_dicoms": str(item.series.number_of_dicoms),
                    "shape": "x".join(str(value) for value in item.shape),
                    "spacing": "x".join(f"{value:.6g}" for value in item.spacing),
                    "source": "Kaggle PROSTATE_MRI",
                    "notes": item.notes,
                }
            )


def _case_id_from_subject(subject_id: str) -> str:
    return subject_id.lower().replace("-", "_")


def _resolve_series_dir(input_dir: Path, file_location: str) -> Path:
    relative = file_location.strip()
    if relative.startswith("./"):
        relative = relative[2:]
    series_dir = input_dir / relative
    if not series_dir.is_dir():
        raise FileNotFoundError(f"Missing DICOM series directory: {series_dir}")
    return series_dir


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _require_pydicom() -> Any:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError("pydicom is required to convert Kaggle PROSTATE_MRI DICOM files") from exc
    return pydicom


def _slice_sort_key(dataset: Any) -> tuple[float, int]:
    position = getattr(dataset, "ImagePositionPatient", None)
    orientation = getattr(dataset, "ImageOrientationPatient", None)
    if position is not None and orientation is not None:
        row = np.asarray([float(value) for value in orientation[:3]], dtype=float)
        col = np.asarray([float(value) for value in orientation[3:]], dtype=float)
        normal = np.cross(row, col)
        distance = float(np.dot(np.asarray([float(value) for value in position]), normal))
    else:
        distance = float(getattr(dataset, "SliceLocation", 0.0))
    instance = int(getattr(dataset, "InstanceNumber", 0))
    return distance, instance


def _scaled_pixel_array(dataset: Any) -> np.ndarray:
    array = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    return array * slope + intercept


def _spacing_from_datasets(datasets: list[Any]) -> tuple[float, float, float]:
    first = datasets[0]
    pixel_spacing = [float(value) for value in first.PixelSpacing]
    row_spacing = pixel_spacing[0]
    col_spacing = pixel_spacing[1]

    if len(datasets) > 1:
        positions = [
            np.asarray([float(value) for value in ds.ImagePositionPatient], dtype=float)
            for ds in datasets
            if getattr(ds, "ImagePositionPatient", None) is not None
        ]
        if len(positions) == len(datasets):
            distances = [
                float(np.linalg.norm(positions[index + 1] - positions[index]))
                for index in range(len(positions) - 1)
            ]
            slice_spacing = float(np.median(distances))
        else:
            slice_spacing = float(
                getattr(first, "SpacingBetweenSlices", getattr(first, "SliceThickness", 1.0))
            )
    else:
        slice_spacing = float(
            getattr(first, "SpacingBetweenSlices", getattr(first, "SliceThickness", 1.0))
        )
    return row_spacing, col_spacing, slice_spacing


def _affine_from_datasets(datasets: list[Any], spacing: tuple[float, float, float]) -> np.ndarray:
    first = datasets[0]
    orientation = getattr(first, "ImageOrientationPatient", None)
    position = getattr(first, "ImagePositionPatient", None)
    affine = np.eye(4, dtype=float)
    if orientation is None or position is None:
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]
        return affine

    row_lps = np.asarray([float(value) for value in orientation[:3]], dtype=float)
    col_lps = np.asarray([float(value) for value in orientation[3:]], dtype=float)
    slice_lps = np.cross(row_lps, col_lps)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    origin_ras = lps_to_ras @ np.asarray([float(value) for value in position], dtype=float)

    # The saved array is [row, column, slice]. These columns map voxel axes to RAS.
    affine[:3, 0] = lps_to_ras @ (col_lps * spacing[0])
    affine[:3, 1] = lps_to_ras @ (row_lps * spacing[1])
    affine[:3, 2] = lps_to_ras @ (slice_lps * spacing[2])
    affine[:3, 3] = origin_ras
    return affine


def _ensure_outputs_can_be_written(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing[:5])
        raise FileExistsError(f"Output exists and --overwrite was not set: {preview}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
