"""Stage 2 preprocessing validation utilities.

The functions here inspect image headers and manifest consistency only. They do
not resample, normalize, crop, or write processed medical images.
"""

from __future__ import annotations

import csv
import gzip
import json
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REQUIRED_MODALITIES = ("t2w", "adc", "hbv")
MODALITY_PATH_COLUMNS = {
    "t2w": "path_t2w",
    "adc": "path_adc",
    "hbv": "path_hbv",
}
MASK_PATH_COLUMNS = {
    "gland": "path_gland_mask",
    "lesion": "path_lesion_mask",
}

DEFAULT_NORMALIZATION_PLAN = {
    "stage2_status": "planned_not_applied",
    "scope": "per-case and per-modality after train/validation split rules are fixed",
    "default_method": "percentile clipping followed by z-score normalization",
    "roi_preference": "use prostate gland mask when available; otherwise flag fallback explicitly",
    "leakage_guard": "do not compute dataset-level statistics across validation or test folds",
}

DEFAULT_ROI_PLAN = {
    "stage2_status": "validate_paths_and_header_alignment_only",
    "gland_mask_use": "candidate case-level prostate ROI for later cropping or foreground statistics",
    "lesion_mask_use": "candidate lesion ROI for later radiomics or lesion-aware experiments only",
    "missing_mask_policy": "flag missing masks; do not synthesize or overwrite masks",
}


@dataclass(frozen=True)
class ImageMetadata:
    """Header metadata needed for lightweight preprocessing checks."""

    path: str
    format: str
    readable: bool
    ndim: int | None = None
    shape: list[int] | None = None
    spacing: list[float] | None = None
    direction: list[float] | None = None
    origin: list[float] | None = None
    element_type: str = ""
    error: str = ""


def validate_preprocessing_inputs(
    manifest_path: str | Path,
    raw_root: str | Path,
    report_path: str | Path | None = None,
    sample_size: int = 10,
    case_ids: Iterable[str] | None = None,
    folds: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate a deterministic sample of manifest rows for Stage 2 readiness."""

    manifest_path = Path(manifest_path)
    raw_root = Path(raw_root)
    rows = load_manifest_rows(manifest_path)
    selected_rows = select_manifest_rows(rows, sample_size, case_ids, folds)

    case_reports = [validate_case_row(row, raw_root) for row in selected_rows]
    report = build_preprocessing_report(
        manifest_path=manifest_path,
        raw_root=raw_root,
        rows=rows,
        selected_rows=selected_rows,
        case_reports=case_reports,
        sample_size=sample_size,
        case_ids=case_ids,
        folds=folds,
    )

    if report_path is not None:
        write_preprocessing_report(report, report_path)
    return report


def validate_resampling_plan(
    manifest_path: str | Path,
    raw_root: str | Path,
    report_path: str | Path | None = None,
    sample_size: int = 5,
    case_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate SimpleITK resampling of ADC/HBV to each case's T2W grid."""

    try:
        import SimpleITK as sitk
    except ImportError as error:  # pragma: no cover - exercised on systems without SimpleITK.
        raise RuntimeError("SimpleITK is required for resampling validation") from error

    manifest_path = Path(manifest_path)
    raw_root = Path(raw_root)
    rows = select_manifest_rows(
        load_manifest_rows(manifest_path),
        sample_size=sample_size,
        case_ids=case_ids,
    )
    case_reports = [
        validate_case_resampling_plan(row=row, raw_root=raw_root, sitk=sitk)
        for row in rows
    ]
    report = {
        "schema_version": "1.0",
        "stage": "simpleitk_resampling_validation",
        "manifest_path": str(manifest_path),
        "raw_root": str(raw_root),
        "sample_size_requested": sample_size,
        "selected_case_ids": [row.get("case_id", "") for row in rows],
        "resampling_policy": {
            "reference": "T2W image grid",
            "moving_images": ["ADC", "HBV"],
            "image_interpolator": "linear",
            "mask_interpolator": "nearest_neighbor",
            "output_pixel_type": "preserve input image pixel type",
            "write_processed_images": False,
        },
        "summary": summarize_resampling_cases(case_reports),
        "cases": case_reports,
    }
    if report_path is not None:
        write_preprocessing_report(report, report_path)
    return report


def write_preprocessed_sample(
    manifest_path: str | Path,
    raw_root: str | Path,
    output_root: str | Path,
    report_path: str | Path,
    sample_size: int = 5,
    case_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Write a tiny inspected sample of ADC/HBV resampled to T2W grid."""

    try:
        import SimpleITK as sitk
    except ImportError as error:  # pragma: no cover - exercised on systems without SimpleITK.
        raise RuntimeError("SimpleITK is required for preprocessing sample writing") from error

    manifest_path = Path(manifest_path)
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    ensure_output_not_inside_raw(output_root, raw_root)

    rows = select_manifest_rows(
        load_manifest_rows(manifest_path),
        sample_size=sample_size,
        case_ids=case_ids,
    )
    case_reports = [
        write_case_preprocessed_sample(row=row, raw_root=raw_root, output_root=output_root, sitk=sitk)
        for row in rows
    ]
    report = {
        "schema_version": "1.0",
        "stage": "preprocessed_sample_writer",
        "manifest_path": str(manifest_path),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "sample_size_requested": sample_size,
        "selected_case_ids": [row.get("case_id", "") for row in rows],
        "write_policy": {
            "processed_outputs": ["adc_resampled_to_t2w", "hbv_resampled_to_t2w"],
            "reference_t2w": "not copied; raw path recorded",
            "masks": "not copied; reference-grid candidates recorded",
            "image_interpolator": "linear",
            "mask_interpolator": "not applied in this stage",
        },
        "summary": summarize_preprocessed_sample_cases(case_reports),
        "cases": case_reports,
    }
    write_preprocessing_report(report, report_path)
    return report


def write_case_preprocessed_sample(row: dict[str, str], raw_root: Path, output_root: Path, sitk: Any) -> dict[str, Any]:
    """Write resampled ADC/HBV outputs for one case."""

    case_id = row.get("case_id", "")
    case_output_root = output_root / case_id
    t2w_path = resolve_manifest_path(row.get("path_t2w", ""), raw_root)
    case_report: dict[str, Any] = {
        "case_id": case_id,
        "fold": row.get("fold", ""),
        "output_dir": str(case_output_root),
        "reference_t2w_path": str(t2w_path),
        "outputs": {},
        "masks": {},
        "issues": [],
    }

    try:
        reference = sitk.ReadImage(str(t2w_path))
    except Exception as error:  # noqa: BLE001 - per-case errors belong in report.
        case_report["issues"].append(f"unreadable_t2w:{type(error).__name__}: {error}")
        return case_report

    reference_signature = simpleitk_signature(reference)
    case_report["reference_signature"] = reference_signature
    case_output_root.mkdir(parents=True, exist_ok=True)

    for modality, column in (("adc", "path_adc"), ("hbv", "path_hbv")):
        moving_path = resolve_manifest_path(row.get(column, ""), raw_root)
        output_path = case_output_root / f"{case_id}_{modality}_to_t2w.mha"
        output_report: dict[str, Any] = {
            "source_path": str(moving_path),
            "output_path": str(output_path),
            "written": False,
            "native_signature": {},
            "output_signature": {},
            "matches_reference": False,
            "error": "",
        }
        try:
            moving = sitk.ReadImage(str(moving_path))
            output_report["native_signature"] = simpleitk_signature(moving)
            resampled = resample_to_reference(moving, reference, sitk.sitkLinear, sitk)
            sitk.WriteImage(resampled, str(output_path))
            written = sitk.ReadImage(str(output_path))
            output_report["written"] = output_path.exists()
            output_report["output_signature"] = simpleitk_signature(written)
            output_report["matches_reference"] = signatures_match(
                output_report["output_signature"],
                reference_signature,
            )
            if not output_report["matches_reference"]:
                case_report["issues"].append(f"{modality}_written_grid_mismatch")
        except Exception as error:  # noqa: BLE001 - per-output failure belongs in report.
            output_report["error"] = f"{type(error).__name__}: {error}"
            case_report["issues"].append(f"{modality}_write_failed")
        case_report["outputs"][modality] = output_report

    for mask_name, column in MASK_PATH_COLUMNS.items():
        case_report["masks"][mask_name] = validate_mask_candidates(
            mask_paths=split_pipe_value(row.get(column, "")),
            raw_root=raw_root,
            reference=reference,
            reference_signature=reference_signature,
            sitk=sitk,
        )
    return case_report


def ensure_output_not_inside_raw(output_root: Path, raw_root: Path) -> None:
    """Prevent accidental writes inside raw medical imaging data."""

    try:
        output_root.resolve().relative_to(raw_root.resolve())
    except ValueError:
        return
    raise ValueError(f"Refusing to write processed outputs inside raw_root: {output_root}")


def summarize_preprocessed_sample_cases(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize written preprocessed sample outputs."""

    return {
        "cases_requested": len(case_reports),
        "cases_with_issues": sum(1 for report in case_reports if report["issues"]),
        "adc_written": sum(1 for report in case_reports if report.get("outputs", {}).get("adc", {}).get("written")),
        "hbv_written": sum(1 for report in case_reports if report.get("outputs", {}).get("hbv", {}).get("written")),
        "adc_matches_reference": sum(
            1 for report in case_reports if report.get("outputs", {}).get("adc", {}).get("matches_reference")
        ),
        "hbv_matches_reference": sum(
            1 for report in case_reports if report.get("outputs", {}).get("hbv", {}).get("matches_reference")
        ),
        "gland_cases_with_reference_grid_mask": sum(
            1 for report in case_reports if report.get("masks", {}).get("gland", {}).get("has_reference_grid_candidate")
        ),
        "lesion_cases_with_reference_grid_mask": sum(
            1 for report in case_reports if report.get("masks", {}).get("lesion", {}).get("has_reference_grid_candidate")
        ),
    }


def validate_case_resampling_plan(row: dict[str, str], raw_root: Path, sitk: Any) -> dict[str, Any]:
    """Validate one case's ADC/HBV resampling against its T2W reference."""

    case_id = row.get("case_id", "")
    t2w_path = resolve_manifest_path(row.get("path_t2w", ""), raw_root)
    case_report: dict[str, Any] = {
        "case_id": case_id,
        "fold": row.get("fold", ""),
        "reference_path": str(t2w_path),
        "modalities": {},
        "masks": {},
        "issues": [],
    }

    try:
        reference = sitk.ReadImage(str(t2w_path))
    except Exception as error:  # noqa: BLE001 - per-case errors belong in report.
        case_report["issues"].append(f"unreadable_t2w:{type(error).__name__}: {error}")
        return case_report

    reference_signature = simpleitk_signature(reference)
    case_report["reference_signature"] = reference_signature
    for modality, column in (("adc", "path_adc"), ("hbv", "path_hbv")):
        moving_path = resolve_manifest_path(row.get(column, ""), raw_root)
        case_report["modalities"][modality] = validate_resampled_image(
            moving_path=moving_path,
            reference=reference,
            reference_signature=reference_signature,
            sitk=sitk,
        )
        if not case_report["modalities"][modality]["resampled_matches_reference"]:
            case_report["issues"].append(f"{modality}_resampling_validation_failed")

    for mask_name, column in MASK_PATH_COLUMNS.items():
        case_report["masks"][mask_name] = validate_mask_candidates(
            mask_paths=split_pipe_value(row.get(column, "")),
            raw_root=raw_root,
            reference=reference,
            reference_signature=reference_signature,
            sitk=sitk,
        )
        if not case_report["masks"][mask_name]["has_reference_grid_candidate"]:
            case_report["issues"].append(f"missing_reference_grid_{mask_name}_mask")
    return case_report


def validate_resampled_image(
    moving_path: Path,
    reference: Any,
    reference_signature: dict[str, Any],
    sitk: Any,
) -> dict[str, Any]:
    """Read and resample a moving image in memory, then compare with reference."""

    report: dict[str, Any] = {
        "path": str(moving_path),
        "readable": False,
        "native_signature": {},
        "resampled_signature": {},
        "resampled_matches_reference": False,
        "error": "",
    }
    try:
        moving = sitk.ReadImage(str(moving_path))
        report["readable"] = True
        report["native_signature"] = simpleitk_signature(moving)
        resampled = resample_to_reference(moving, reference, sitk.sitkLinear, sitk)
        report["resampled_signature"] = simpleitk_signature(resampled)
        report["resampled_matches_reference"] = signatures_match(
            report["resampled_signature"],
            reference_signature,
        )
    except Exception as error:  # noqa: BLE001 - record per-image failure.
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def validate_mask_candidates(
    mask_paths: list[str],
    raw_root: Path,
    reference: Any,
    reference_signature: dict[str, Any],
    sitk: Any,
) -> dict[str, Any]:
    """Validate mask candidates and reference-grid availability."""

    candidates = []
    for mask_value in mask_paths:
        mask_path = resolve_manifest_path(mask_value, raw_root)
        candidate: dict[str, Any] = {
            "path": str(mask_path),
            "readable": False,
            "native_signature": {},
            "matches_reference_grid": False,
            "non_empty": None,
            "error": "",
        }
        try:
            mask = sitk.ReadImage(str(mask_path))
            candidate["readable"] = True
            candidate["native_signature"] = simpleitk_signature(mask)
            candidate["matches_reference_grid"] = signatures_match(
                candidate["native_signature"],
                reference_signature,
            )
            candidate["non_empty"] = image_has_positive_voxels(mask, sitk)
        except Exception as error:  # noqa: BLE001 - record per-mask failure.
            candidate["error"] = f"{type(error).__name__}: {error}"
        candidates.append(candidate)

    return {
        "candidate_count": len(candidates),
        "readable_count": sum(1 for candidate in candidates if candidate["readable"]),
        "reference_grid_candidate_count": sum(
            1 for candidate in candidates if candidate["matches_reference_grid"]
        ),
        "non_empty_reference_grid_candidate_count": sum(
            1
            for candidate in candidates
            if candidate["matches_reference_grid"] and candidate["non_empty"]
        ),
        "has_reference_grid_candidate": any(
            candidate["matches_reference_grid"] for candidate in candidates
        ),
        "candidates": candidates,
    }


def resample_to_reference(moving: Any, reference: Any, interpolator: int, sitk: Any) -> Any:
    """Resample a SimpleITK image to a reference image grid in memory."""

    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        interpolator,
        0,
        moving.GetPixelID(),
    )


def simpleitk_signature(image: Any) -> dict[str, Any]:
    """Return grid-defining SimpleITK metadata."""

    return {
        "size": list(image.GetSize()),
        "spacing": [float(value) for value in image.GetSpacing()],
        "origin": [float(value) for value in image.GetOrigin()],
        "direction": [float(value) for value in image.GetDirection()],
        "pixel_id": image.GetPixelIDTypeAsString(),
    }


def signatures_match(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1e-4) -> bool:
    """Compare image grid signatures while ignoring pixel type."""

    return (
        left.get("size") == right.get("size")
        and floats_close(left.get("spacing", []), right.get("spacing", []), tolerance)
        and floats_close(left.get("origin", []), right.get("origin", []), tolerance)
        and floats_close(left.get("direction", []), right.get("direction", []), tolerance)
    )


def image_has_positive_voxels(image: Any, sitk: Any) -> bool:
    """Return whether a mask has positive voxels using SimpleITK statistics."""

    statistics = sitk.StatisticsImageFilter()
    statistics.Execute(image)
    return statistics.GetMaximum() > 0


def summarize_resampling_cases(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize resampling validation reports."""

    return {
        "cases_checked": len(case_reports),
        "cases_with_issues": sum(1 for report in case_reports if report["issues"]),
        "adc_resampled_matches_reference": sum(
            1 for report in case_reports if report.get("modalities", {}).get("adc", {}).get("resampled_matches_reference")
        ),
        "hbv_resampled_matches_reference": sum(
            1 for report in case_reports if report.get("modalities", {}).get("hbv", {}).get("resampled_matches_reference")
        ),
        "gland_cases_with_reference_grid_mask": sum(
            1 for report in case_reports if report.get("masks", {}).get("gland", {}).get("has_reference_grid_candidate")
        ),
        "lesion_cases_with_reference_grid_mask": sum(
            1 for report in case_reports if report.get("masks", {}).get("lesion", {}).get("has_reference_grid_candidate")
        ),
    }


def load_manifest_rows(manifest_path: str | Path) -> list[dict[str, str]]:
    """Load a Stage 1 CSV manifest."""

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
        return [
            {key: (value or "").strip() for key, value in row.items() if key}
            for row in csv.DictReader(csv_file)
        ]


def select_manifest_rows(
    rows: list[dict[str, str]],
    sample_size: int,
    case_ids: Iterable[str] | None = None,
    folds: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    """Select rows deterministically by case ID after optional filtering."""

    requested_cases = set(case_ids or [])
    requested_folds = set(folds or [])
    selected = rows
    if requested_cases:
        selected = [row for row in selected if row.get("case_id") in requested_cases]
    if requested_folds:
        selected = [row for row in selected if row.get("fold") in requested_folds]

    selected = sorted(selected, key=lambda row: row.get("case_id", ""))
    if requested_cases:
        return selected
    return selected[: max(sample_size, 0)]


def validate_case_row(row: dict[str, str], raw_root: Path) -> dict[str, Any]:
    """Validate image and mask paths plus header compatibility for one case."""

    case_report: dict[str, Any] = {
        "case_id": row.get("case_id", ""),
        "fold": row.get("fold", ""),
        "modalities": {},
        "masks": {},
        "checks": {},
        "resampling_required": [],
        "blocking_issues": [],
        "issues": [],
    }

    modality_metadata: dict[str, ImageMetadata] = {}
    for modality, column in MODALITY_PATH_COLUMNS.items():
        metadata = metadata_from_manifest_value(row.get(column, ""), raw_root)
        modality_metadata[modality] = metadata
        case_report["modalities"][modality] = asdict(metadata)
        if not row.get(column, ""):
            case_report["blocking_issues"].append(f"missing_manifest_path_{modality}")
        elif not Path(metadata.path).exists():
            case_report["blocking_issues"].append(f"missing_file_{modality}")
        elif not metadata.readable:
            case_report["blocking_issues"].append(f"unreadable_header_{modality}")

    reference = modality_metadata["t2w"]
    case_report["modalities"]["t2w"]["alignment_to_t2w"] = "reference"
    case_report["modalities"]["t2w"]["mismatched_fields"] = []
    for modality in ("adc", "hbv"):
        issue_fields = compare_metadata(reference, modality_metadata[modality])
        case_report["modalities"][modality]["mismatched_fields"] = issue_fields
        if issue_fields:
            case_report["modalities"][modality]["alignment_to_t2w"] = "requires_resampling"
            case_report["resampling_required"].append(
                f"{modality}_to_t2w_grid:{','.join(issue_fields)}"
            )
        else:
            case_report["modalities"][modality]["alignment_to_t2w"] = "already_aligned"

    for mask_name, column in MASK_PATH_COLUMNS.items():
        mask_reports = []
        mask_paths = split_pipe_value(row.get(column, ""))
        if not mask_paths:
            case_report["blocking_issues"].append(f"missing_manifest_path_{mask_name}_mask")
        readable_mask_count = 0
        t2w_compatible_mask_count = 0
        for index, mask_path in enumerate(mask_paths):
            metadata = metadata_from_manifest_value(mask_path, raw_root)
            mask_report = asdict(metadata)
            mask_report["index"] = index
            mask_report["mismatched_fields"] = []
            mask_report["alignment_to_t2w"] = "unreadable"
            mask_reports.append(mask_report)
            if not Path(metadata.path).exists():
                case_report["blocking_issues"].append(f"missing_file_{mask_name}_mask")
            elif not metadata.readable:
                case_report["blocking_issues"].append(f"unreadable_header_{mask_name}_mask")
            else:
                readable_mask_count += 1
                issue_fields = compare_metadata(reference, metadata, compare_direction=False)
                mask_report["mismatched_fields"] = issue_fields
                if issue_fields:
                    mask_report["alignment_to_t2w"] = "different_grid"
                else:
                    mask_report["alignment_to_t2w"] = "t2w_compatible"
                    t2w_compatible_mask_count += 1
        case_report["masks"][mask_name] = mask_reports
        if readable_mask_count and not t2w_compatible_mask_count:
            case_report["blocking_issues"].append(f"no_t2w_compatible_{mask_name}_mask")

    case_report["issues"] = list(case_report["blocking_issues"])
    case_report["checks"] = {
        "all_target_modality_paths_resolve": all(
            Path(metadata.path).exists() for metadata in modality_metadata.values()
        ),
        "all_target_modality_headers_readable": all(
            metadata.readable for metadata in modality_metadata.values()
        ),
        "target_modalities_align_with_t2w": not case_report["resampling_required"],
        "target_modalities_need_resampling_to_t2w": bool(case_report["resampling_required"]),
        "masks_have_t2w_compatible_candidate": not any(
            issue.startswith("no_t2w_compatible_") for issue in case_report["blocking_issues"]
        ),
        "has_blocking_issues": bool(case_report["blocking_issues"]),
    }
    return case_report


def metadata_from_manifest_value(value: str, raw_root: Path) -> ImageMetadata:
    """Resolve a manifest path value and read lightweight image metadata."""

    path = resolve_manifest_path(value, raw_root) if value else raw_root / "__missing__"
    if not value:
        return ImageMetadata(path=str(path), format="", readable=False, error="missing manifest path")
    if not path.exists():
        return ImageMetadata(path=str(path), format="", readable=False, error="file does not exist")
    return read_image_metadata(path)


def resolve_manifest_path(value: str, raw_root: Path) -> Path:
    """Resolve absolute, repo-relative, or raw-root-relative manifest paths."""

    path = Path(value)
    if path.is_absolute():
        return path

    raw_relative = raw_root / path
    if raw_relative.exists():
        return raw_relative
    if path.exists():
        return path
    return raw_relative


def read_image_metadata(path: str | Path) -> ImageMetadata:
    """Read header metadata from MetaImage or NIfTI files without voxel loading."""

    path = Path(path)
    lower_name = path.name.lower()
    try:
        if lower_name.endswith((".mha", ".mhd")):
            return read_metaimage_metadata(path)
        if lower_name.endswith((".nii", ".nii.gz")):
            return read_nifti_metadata(path)
    except Exception as error:  # noqa: BLE001 - report the header-read failure.
        return ImageMetadata(
            path=str(path),
            format=path.suffix.lower().lstrip("."),
            readable=False,
            error=f"{type(error).__name__}: {error}",
        )
    return ImageMetadata(
        path=str(path),
        format=path.suffix.lower().lstrip("."),
        readable=False,
        error="unsupported image format for header-only validation",
    )


def read_metaimage_metadata(path: Path) -> ImageMetadata:
    """Read `.mha` or `.mhd` header metadata."""

    header = parse_metaimage_header(path)
    shape = parse_int_list(header.get("DimSize", ""))
    spacing = parse_float_list(header.get("ElementSpacing", ""))
    direction = parse_float_list(header.get("TransformMatrix", ""))
    origin = parse_float_list(header.get("Offset", header.get("Position", "")))
    ndim = parse_optional_int(header.get("NDims", ""))

    return ImageMetadata(
        path=str(path),
        format="metaimage",
        readable=True,
        ndim=ndim,
        shape=shape,
        spacing=spacing,
        direction=direction,
        origin=origin,
        element_type=header.get("ElementType", ""),
    )


def parse_metaimage_header(path: Path, max_bytes: int = 131072) -> dict[str, str]:
    """Parse the ASCII header of a MetaImage file."""

    raw = path.read_bytes()[:max_bytes]
    text = raw.decode("latin-1", errors="ignore")
    header: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        header[key] = value
        if key == "ElementDataFile":
            break
    return header


def read_nifti_metadata(path: Path) -> ImageMetadata:
    """Read NIfTI-1 header metadata from `.nii` or `.nii.gz` files."""

    header = read_nifti_header_bytes(path)
    endian = infer_nifti_endian(header)
    dim = struct.unpack(endian + "8h", header[40:56])
    pixdim = struct.unpack(endian + "8f", header[76:108])
    datatype = struct.unpack(endian + "h", header[70:72])[0]
    ndim = int(dim[0])
    shape = [int(value) for value in dim[1 : ndim + 1] if value > 0]
    spacing = [float(value) for value in pixdim[1 : ndim + 1] if value > 0]
    sform_code = struct.unpack(endian + "h", header[254:256])[0]
    direction: list[float] | None = None
    origin: list[float] | None = None
    if sform_code > 0:
        srow_x = struct.unpack(endian + "4f", header[280:296])
        srow_y = struct.unpack(endian + "4f", header[296:312])
        srow_z = struct.unpack(endian + "4f", header[312:328])
        direction = [
            float(value)
            for row in (srow_x[:3], srow_y[:3], srow_z[:3])
            for value in row
        ]
        origin = [float(srow_x[3]), float(srow_y[3]), float(srow_z[3])]

    return ImageMetadata(
        path=str(path),
        format="nifti1",
        readable=True,
        ndim=ndim,
        shape=shape,
        spacing=spacing,
        direction=direction,
        origin=origin,
        element_type=nifti_datatype_name(datatype),
    )


def read_nifti_header_bytes(path: Path) -> bytes:
    """Return the first 348 bytes from a NIfTI file."""

    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "rb") as file_obj:
            header = file_obj.read(348)
    else:
        with path.open("rb") as file_obj:
            header = file_obj.read(348)
    if len(header) < 348:
        raise ValueError("NIfTI header is shorter than 348 bytes")
    return header


def infer_nifti_endian(header: bytes) -> str:
    """Infer NIfTI header endianness."""

    little = struct.unpack("<i", header[:4])[0]
    if little == 348:
        return "<"
    big = struct.unpack(">i", header[:4])[0]
    if big == 348:
        return ">"
    raise ValueError("NIfTI sizeof_hdr is not 348")


def compare_metadata(
    reference: ImageMetadata,
    candidate: ImageMetadata,
    compare_direction: bool = True,
) -> list[str]:
    """Return metadata fields that mismatch where both sides are readable."""

    if not reference.readable or not candidate.readable:
        return []

    mismatches: list[str] = []
    if reference.ndim is not None and candidate.ndim is not None and reference.ndim != candidate.ndim:
        mismatches.append("ndim")
    if reference.shape and candidate.shape and reference.shape != candidate.shape:
        mismatches.append("shape")
    if reference.spacing and candidate.spacing and not floats_close(reference.spacing, candidate.spacing):
        mismatches.append("spacing")
    if (
        compare_direction
        and reference.direction
        and candidate.direction
        and not floats_close(reference.direction, candidate.direction)
    ):
        mismatches.append("direction")
    return mismatches


def build_preprocessing_report(
    manifest_path: Path,
    raw_root: Path,
    rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
    case_reports: list[dict[str, Any]],
    sample_size: int,
    case_ids: Iterable[str] | None,
    folds: Iterable[str] | None,
) -> dict[str, Any]:
    """Build the Stage 2 preprocessing validation report."""

    blocking_counter: Counter[str] = Counter(
        issue
        for case_report in case_reports
        for issue in case_report["blocking_issues"]
    )
    resampling_counter: Counter[str] = Counter(
        item
        for case_report in case_reports
        for item in case_report["resampling_required"]
    )
    modality_readable_counts = {
        modality: sum(
            1
            for case_report in case_reports
            if case_report["modalities"][modality]["readable"]
        )
        for modality in REQUIRED_MODALITIES
    }
    mask_readable_counts = {
        mask_name: sum(
            1
            for case_report in case_reports
            if any(mask_report["readable"] for mask_report in case_report["masks"][mask_name])
        )
        for mask_name in MASK_PATH_COLUMNS
    }
    mask_t2w_compatible_counts = {
        mask_name: sum(
            1
            for case_report in case_reports
            if any(
                mask_report["alignment_to_t2w"] == "t2w_compatible"
                for mask_report in case_report["masks"][mask_name]
            )
        )
        for mask_name in MASK_PATH_COLUMNS
    }

    return {
        "schema_version": "1.0",
        "stage": "preprocessing_input_validation",
        "manifest_path": str(manifest_path),
        "raw_root": str(raw_root),
        "total_manifest_rows": len(rows),
        "sample_size_requested": sample_size,
        "case_ids_requested": sorted(set(case_ids or [])),
        "folds_requested": sorted(set(folds or [])),
        "selected_case_ids": [row.get("case_id", "") for row in selected_rows],
        "normalization_plan": DEFAULT_NORMALIZATION_PLAN,
        "roi_plan": DEFAULT_ROI_PLAN,
        "summary": {
            "cases_checked": len(case_reports),
            "cases_with_issues": sum(
                1 for case_report in case_reports if case_report["blocking_issues"]
            ),
            "cases_with_blocking_issues": sum(
                1 for case_report in case_reports if case_report["blocking_issues"]
            ),
            "cases_requiring_resampling": sum(
                1 for case_report in case_reports if case_report["resampling_required"]
            ),
            "issue_counts": dict(sorted(blocking_counter.items())),
            "resampling_required_counts": dict(sorted(resampling_counter.items())),
            "modality_headers_readable": modality_readable_counts,
            "mask_headers_readable": mask_readable_counts,
            "mask_t2w_compatible_cases": mask_t2w_compatible_counts,
        },
        "cases": case_reports,
    }


def write_preprocessing_report(report: dict[str, Any], report_path: str | Path) -> None:
    """Write a Stage 2 validation report as JSON."""

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, sort_keys=True)
        report_file.write("\n")


def parse_int_list(value: str) -> list[int] | None:
    """Parse a whitespace-separated integer list."""

    if not value:
        return None
    return [int(item) for item in value.split()]


def parse_float_list(value: str) -> list[float] | None:
    """Parse a whitespace-separated float list."""

    if not value:
        return None
    return [float(item) for item in value.split()]


def parse_optional_int(value: str) -> int | None:
    """Parse an integer if available."""

    if not value:
        return None
    return int(value)


def floats_close(left: list[float], right: list[float], tolerance: float = 1e-4) -> bool:
    """Return whether two float vectors are equal within tolerance."""

    if len(left) != len(right):
        return False
    return all(abs(left_value - right_value) <= tolerance for left_value, right_value in zip(left, right))


def split_pipe_value(value: str) -> list[str]:
    """Split a pipe-delimited manifest field."""

    if not value:
        return []
    return [item for item in value.split("|") if item]


def nifti_datatype_name(datatype: int) -> str:
    """Return a compact NIfTI datatype name."""

    names = {
        2: "uint8",
        4: "int16",
        8: "int32",
        16: "float32",
        64: "float64",
        256: "int8",
        512: "uint16",
        768: "uint32",
        1024: "int64",
        1280: "uint64",
    }
    return names.get(datatype, f"datatype_{datatype}")
