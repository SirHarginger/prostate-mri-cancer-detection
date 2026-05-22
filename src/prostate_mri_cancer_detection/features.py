"""Stage 3 radiomics feature extraction utilities.

This module implements a small, dependency-light first-order radiomics
extractor for validated T2W-grid ROIs. It does not augment images, train
models, or write features outside ignored project output directories.
"""

from __future__ import annotations

import array
import csv
import gzip
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from prostate_mri_cancer_detection.preprocessing import (
    floats_close,
    parse_metaimage_header,
    read_image_metadata,
    read_nifti_header_bytes,
    resolve_manifest_path,
    split_pipe_value,
)

SEQUENCE_PATH_COLUMNS = {
    "t2w": "path_t2w",
}
ROI_PATH_COLUMNS = {
    "gland": "path_gland_mask",
    "lesion": "path_lesion_mask",
}

FEATURE_COLUMNS = [
    "case_id",
    "fold",
    "sequence",
    "roi",
    "image_path",
    "mask_path",
    "voxel_count",
    "mask_volume_mm3",
    "mask_fraction",
    "intensity_min",
    "intensity_max",
    "intensity_mean",
    "intensity_std",
    "intensity_median",
    "intensity_p10",
    "intensity_p90",
    "intensity_iqr",
    "intensity_energy",
    "intensity_entropy_32bin",
]
FAILURE_COLUMNS = ["case_id", "fold", "sequence", "roi", "reason", "image_path", "mask_path"]

ARRAY_TYPES = {
    "MET_UCHAR": ("B", 1),
    "MET_CHAR": ("b", 1),
    "MET_USHORT": ("H", 2),
    "MET_SHORT": ("h", 2),
    "MET_UINT": ("I", 4),
    "MET_INT": ("i", 4),
    "MET_FLOAT": ("f", 4),
    "MET_DOUBLE": ("d", 8),
}
NIFTI_ARRAY_TYPES = {
    2: ("B", 1),
    4: ("h", 2),
    8: ("i", 4),
    16: ("f", 4),
    64: ("d", 8),
    256: ("b", 1),
    512: ("H", 2),
    768: ("I", 4),
}


@dataclass(frozen=True)
class VolumeData:
    """Voxel data plus spatial metadata."""

    path: str
    values: array.array
    shape: list[int]
    spacing: list[float]
    element_type: str


def extract_radiomics_features(
    manifest_path: str | Path,
    raw_root: str | Path,
    output_path: str | Path,
    failure_log_path: str | Path,
    settings_path: str | Path,
    preprocessing_report_path: str | Path | None = None,
    sequence: str = "t2w",
    roi: str = "lesion",
    sample_size: int = 25,
    case_ids: Iterable[str] | None = None,
    all_cases: bool = False,
) -> dict[str, Any]:
    """Extract first-order radiomics features for validated ROI masks."""

    if sequence not in SEQUENCE_PATH_COLUMNS:
        raise ValueError(f"Unsupported sequence for Stage 3 radiomics: {sequence}")
    if roi not in ROI_PATH_COLUMNS:
        raise ValueError(f"Unsupported ROI for Stage 3 radiomics: {roi}")

    manifest_path = Path(manifest_path)
    raw_root = Path(raw_root)
    rows = load_manifest_rows(manifest_path)
    preprocessing_report = load_optional_json(preprocessing_report_path)
    selected_case_ids = select_case_ids(rows, preprocessing_report, sample_size, case_ids, all_cases)
    rows_by_case = {row.get("case_id", ""): row for row in rows}

    feature_rows: list[dict[str, str]] = []
    failure_rows: list[dict[str, str]] = []

    for case_id in selected_case_ids:
        row = rows_by_case.get(case_id)
        if row is None:
            failure_rows.append(failure_row(case_id, "", sequence, roi, "case_id_not_in_manifest", "", ""))
            continue

        image_path = resolve_manifest_path(row.get(SEQUENCE_PATH_COLUMNS[sequence], ""), raw_root)
        mask_path = choose_mask_path(row, raw_root, preprocessing_report, roi)
        fold = row.get("fold", "")

        try:
            feature_rows.append(
                extract_case_first_order_features(
                    case_id=case_id,
                    fold=fold,
                    sequence=sequence,
                    roi=roi,
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )
        except Exception as error:  # noqa: BLE001 - per-case extraction failures go to the failure log.
            failure_rows.append(
                failure_row(
                    case_id,
                    fold,
                    sequence,
                    roi,
                    f"{type(error).__name__}: {error}",
                    str(image_path),
                    str(mask_path) if mask_path is not None else "",
                )
            )

    settings = build_radiomics_settings(
        manifest_path=manifest_path,
        raw_root=raw_root,
        preprocessing_report_path=preprocessing_report_path,
        sequence=sequence,
        roi=roi,
        sample_size=sample_size,
        all_cases=all_cases,
        selected_case_ids=selected_case_ids,
    )
    write_csv(output_path, FEATURE_COLUMNS, feature_rows)
    write_csv(failure_log_path, FAILURE_COLUMNS, failure_rows)
    write_json(settings_path, settings)

    return {
        "schema_version": "1.0",
        "stage": "radiomics_feature_extraction",
        "sequence": sequence,
        "roi": roi,
        "cases_requested": len(selected_case_ids),
        "features_written": len(feature_rows),
        "failures_written": len(failure_rows),
        "output_path": str(output_path),
        "failure_log_path": str(failure_log_path),
        "settings_path": str(settings_path),
    }


def extract_case_first_order_features(
    case_id: str,
    fold: str,
    sequence: str,
    roi: str,
    image_path: Path,
    mask_path: Path | None,
) -> dict[str, str]:
    """Extract first-order features for one image and one ROI mask."""

    if mask_path is None:
        raise ValueError("no compatible mask path found")
    validate_image_mask_metadata(image_path, mask_path)
    image_volume = read_volume_data(image_path)
    mask_volume = read_volume_data(mask_path)
    if image_volume.shape != mask_volume.shape:
        raise ValueError(f"image/mask shape mismatch: {image_volume.shape} vs {mask_volume.shape}")

    values = [
        float(image_value)
        for image_value, mask_value in zip(image_volume.values, mask_volume.values)
        if mask_value > 0
    ]
    if not values:
        raise ValueError("empty mask")

    stats = first_order_statistics(values)
    voxel_volume = math.prod(image_volume.spacing) if image_volume.spacing else 0.0
    voxel_count = len(values)
    total_voxels = len(image_volume.values)

    return {
        "case_id": case_id,
        "fold": fold,
        "sequence": sequence,
        "roi": roi,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "voxel_count": str(voxel_count),
        "mask_volume_mm3": format_float(voxel_count * voxel_volume),
        "mask_fraction": format_float(voxel_count / total_voxels if total_voxels else 0.0),
        **{key: format_float(value) for key, value in stats.items()},
    }


def validate_image_mask_metadata(image_path: Path, mask_path: Path) -> None:
    """Validate image and mask headers before reading voxel arrays."""

    image_metadata = read_image_metadata(image_path)
    mask_metadata = read_image_metadata(mask_path)
    if not image_metadata.readable:
        raise ValueError(f"unreadable image header: {image_metadata.error}")
    if not mask_metadata.readable:
        raise ValueError(f"unreadable mask header: {mask_metadata.error}")
    if image_metadata.shape != mask_metadata.shape:
        raise ValueError(f"image/mask shape mismatch: {image_metadata.shape} vs {mask_metadata.shape}")
    if image_metadata.spacing and mask_metadata.spacing and not floats_close(
        image_metadata.spacing,
        mask_metadata.spacing,
    ):
        raise ValueError(
            f"image/mask spacing mismatch: {image_metadata.spacing} vs {mask_metadata.spacing}"
        )


def choose_mask_path(
    row: dict[str, str],
    raw_root: Path,
    preprocessing_report: dict[str, Any] | None,
    roi: str,
) -> Path | None:
    """Choose the first T2W-compatible mask from the preprocessing report."""

    case_id = row.get("case_id", "")
    if preprocessing_report is not None:
        for case_report in preprocessing_report.get("cases", []):
            if case_report.get("case_id") != case_id:
                continue
            for mask_report in case_report.get("masks", {}).get(roi, []):
                if mask_report.get("alignment_to_t2w") == "t2w_compatible":
                    return Path(mask_report["path"])

    for value in split_pipe_value(row.get(ROI_PATH_COLUMNS[roi], "")):
        return resolve_manifest_path(value, raw_root)
    return None


def select_case_ids(
    rows: list[dict[str, str]],
    preprocessing_report: dict[str, Any] | None,
    sample_size: int,
    case_ids: Iterable[str] | None,
    all_cases: bool,
) -> list[str]:
    """Select cases safely for extraction."""

    requested = [case_id for case_id in case_ids or [] if case_id]
    if requested:
        return sorted(set(requested))
    if preprocessing_report is not None and preprocessing_report.get("selected_case_ids"):
        return list(preprocessing_report["selected_case_ids"])
    manifest_case_ids = sorted(row.get("case_id", "") for row in rows if row.get("case_id", ""))
    if all_cases:
        return manifest_case_ids
    return manifest_case_ids[: max(sample_size, 0)]


def read_volume_data(path: str | Path) -> VolumeData:
    """Read a supported image volume into a one-dimensional array."""

    path = Path(path)
    lower_name = path.name.lower()
    if lower_name.endswith((".mha", ".mhd")):
        return read_metaimage_volume(path)
    if lower_name.endswith((".nii", ".nii.gz")):
        return read_nifti_volume(path)
    raise ValueError(f"unsupported image format: {path}")


def read_metaimage_volume(path: Path) -> VolumeData:
    """Read voxel data from a MetaImage file."""

    header = parse_metaimage_header(path)
    element_data_file = header.get("ElementDataFile", "")

    shape = [int(value) for value in header["DimSize"].split()]
    spacing = [float(value) for value in header.get("ElementSpacing", "").split()]
    element_type = header["ElementType"]
    if element_type not in ARRAY_TYPES:
        raise ValueError(f"unsupported MetaImage element type: {element_type}")
    type_code, item_size = ARRAY_TYPES[element_type]
    count = math.prod(shape)
    expected_bytes = count * item_size
    data = read_metaimage_payload(path, header, expected_bytes)
    if len(data) < expected_bytes:
        raise ValueError("MetaImage voxel data are shorter than expected")
    values = bytes_to_array(data[:expected_bytes], type_code, element_byte_order(header), count)
    return VolumeData(str(path), values, shape, spacing, element_type)


def read_metaimage_payload(path: Path, header: dict[str, str], expected_bytes: int) -> bytes:
    """Read and optionally decompress MetaImage voxel payload bytes."""

    element_data_file = header.get("ElementDataFile", "")
    if element_data_file.upper() == "LOCAL":
        payload = path.read_bytes()
        data_offset = find_metaimage_data_offset(payload)
        data = payload[data_offset:]
    else:
        data_path = path.parent / element_data_file
        if not data_path.exists():
            raise ValueError(f"MetaImage external data file does not exist: {data_path}")
        data = data_path.read_bytes()

    if is_metaimage_compressed(header):
        try:
            data = zlib.decompress(data)
        except zlib.error as error:
            raise ValueError(f"could not decompress MetaImage payload: {error}") from error

    if len(data) < expected_bytes:
        raise ValueError("MetaImage voxel data are shorter than expected")
    return data


def read_nifti_volume(path: Path) -> VolumeData:
    """Read voxel data from a NIfTI-1 file."""

    payload = gzip.open(path, "rb").read() if path.name.lower().endswith(".gz") else path.read_bytes()
    header = read_nifti_header_bytes(path)
    endian = infer_nifti_endian_from_bytes(header)
    dim = struct.unpack(endian + "8h", header[40:56])
    pixdim = struct.unpack(endian + "8f", header[76:108])
    datatype = struct.unpack(endian + "h", header[70:72])[0]
    vox_offset = struct.unpack(endian + "f", header[108:112])[0]
    if datatype not in NIFTI_ARRAY_TYPES:
        raise ValueError(f"unsupported NIfTI datatype: {datatype}")

    ndim = int(dim[0])
    shape = [int(value) for value in dim[1 : ndim + 1] if value > 0]
    spacing = [float(value) for value in pixdim[1 : ndim + 1] if value > 0]
    type_code, item_size = NIFTI_ARRAY_TYPES[datatype]
    count = math.prod(shape)
    data_offset = int(vox_offset) if vox_offset >= 348 else 348
    data = payload[data_offset : data_offset + count * item_size]
    if len(data) < count * item_size:
        raise ValueError("NIfTI voxel data are shorter than expected")
    values = bytes_to_array(data, type_code, endian, count)
    return VolumeData(str(path), values, shape, spacing, f"nifti_datatype_{datatype}")


def first_order_statistics(values: list[float]) -> dict[str, float]:
    """Compute reproducible first-order ROI intensity features."""

    sorted_values = sorted(values)
    count = len(sorted_values)
    mean = sum(sorted_values) / count
    variance = sum((value - mean) ** 2 for value in sorted_values) / count
    p10 = percentile(sorted_values, 0.10)
    median = percentile(sorted_values, 0.50)
    p90 = percentile(sorted_values, 0.90)
    return {
        "intensity_min": sorted_values[0],
        "intensity_max": sorted_values[-1],
        "intensity_mean": mean,
        "intensity_std": math.sqrt(variance),
        "intensity_median": median,
        "intensity_p10": p10,
        "intensity_p90": p90,
        "intensity_iqr": percentile(sorted_values, 0.75) - percentile(sorted_values, 0.25),
        "intensity_energy": sum(value * value for value in sorted_values),
        "intensity_entropy_32bin": entropy(sorted_values, bins=32),
    }


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Compute a linear-interpolated percentile from sorted values."""

    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def entropy(sorted_values: list[float], bins: int) -> float:
    """Compute simple fixed-bin entropy for ROI intensities."""

    minimum = sorted_values[0]
    maximum = sorted_values[-1]
    if minimum == maximum:
        return 0.0
    width = (maximum - minimum) / bins
    counts = [0] * bins
    for value in sorted_values:
        index = min(int((value - minimum) / width), bins - 1)
        counts[index] += 1
    total = len(sorted_values)
    return -sum((count / total) * math.log2(count / total) for count in counts if count)


def find_metaimage_data_offset(payload: bytes) -> int:
    """Find the start of local binary payload in an `.mha` file."""

    marker = b"ElementDataFile"
    marker_index = payload.find(marker)
    if marker_index == -1:
        raise ValueError("ElementDataFile marker not found")
    newline_index = payload.find(b"\n", marker_index)
    if newline_index == -1:
        raise ValueError("ElementDataFile line is not terminated")
    return newline_index + 1


def element_byte_order(header: dict[str, str]) -> str:
    """Return struct byte order for MetaImage payload."""

    value = header.get("ElementByteOrderMSB", header.get("BinaryDataByteOrderMSB", "False"))
    return ">" if value.lower() in {"true", "1"} else "<"


def is_metaimage_compressed(header: dict[str, str]) -> bool:
    """Return whether MetaImage payload bytes are zlib compressed."""

    return header.get("CompressedData", "False").lower() in {"true", "1"}


def bytes_to_array(data: bytes, type_code: str, endian: str, count: int) -> array.array:
    """Convert binary bytes to an array with byte-order handling."""

    values = array.array(type_code)
    values.frombytes(data)
    del values[count:]
    native_little = struct.pack("=H", 1) == struct.pack("<H", 1)
    data_little = endian == "<"
    if type_code not in {"b", "B"} and native_little != data_little:
        values.byteswap()
    return values


def infer_nifti_endian_from_bytes(header: bytes) -> str:
    """Infer NIfTI header endianness from raw header bytes."""

    little = struct.unpack("<i", header[:4])[0]
    if little == 348:
        return "<"
    big = struct.unpack(">i", header[:4])[0]
    if big == 348:
        return ">"
    raise ValueError("NIfTI sizeof_hdr is not 348")


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Load Stage 1 manifest rows."""

    with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
        return [
            {key: (value or "").strip() for key, value in row.items() if key}
            for row in csv.DictReader(csv_file)
        ]


def load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    """Load an optional JSON document."""

    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_radiomics_settings(
    manifest_path: Path,
    raw_root: Path,
    preprocessing_report_path: str | Path | None,
    sequence: str,
    roi: str,
    sample_size: int,
    all_cases: bool,
    selected_case_ids: list[str],
) -> dict[str, Any]:
    """Record reproducible extraction settings."""

    return {
        "schema_version": "1.0",
        "stage": "radiomics_first_order_extraction",
        "manifest_path": str(manifest_path),
        "raw_root": str(raw_root),
        "preprocessing_report_path": str(preprocessing_report_path or ""),
        "sequence": sequence,
        "roi": roi,
        "sample_size": sample_size,
        "all_cases": all_cases,
        "selected_case_ids": selected_case_ids,
        "image_source": "original manifest image; no augmentation",
        "alignment_policy": "exact image/mask shape and spacing match required",
        "mask_policy": "prefer first T2W-compatible mask from Stage 2 preprocessing report",
        "feature_family": "first_order_intensity_and_roi_size",
        "feature_columns": FEATURE_COLUMNS,
        "limitations": [
            "T2W-only until ADC/HBV resampling is implemented",
            "dependency-light first-order features; not full PyRadiomics texture extraction",
            "not publication-grade final radiomics until image/mask geometry policy is finalized",
        ],
    }


def write_csv(path: str | Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write rows to CSV with a stable schema."""

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


def failure_row(
    case_id: str,
    fold: str,
    sequence: str,
    roi: str,
    reason: str,
    image_path: str,
    mask_path: str,
) -> dict[str, str]:
    """Create a failure-log row."""

    return {
        "case_id": case_id,
        "fold": fold,
        "sequence": sequence,
        "roi": roi,
        "reason": reason,
        "image_path": image_path,
        "mask_path": mask_path,
    }


def format_float(value: float) -> str:
    """Format floating values consistently for CSV output."""

    return f"{value:.10g}"
