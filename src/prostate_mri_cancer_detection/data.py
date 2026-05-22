"""PI-CAI dataset inventory and manifest utilities.

This module intentionally stays at the file-inventory level. It does not load,
preprocess, resample, or otherwise modify medical images.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MODALITY_SUFFIXES = {
    "t2w": "_t2w",
    "adc": "_adc",
    "hbv": "_hbv",
}

NON_MANIFEST_IMAGE_SUFFIXES = {
    "cor": "_cor",
    "sag": "_sag",
}

MEDICAL_IMAGE_EXTENSIONS = (
    ".nii.gz",
    ".nii",
    ".mha",
    ".mhd",
    ".nrrd",
    ".nhdr",
)

CASE_ID_RE = re.compile(r"(?P<patient_id>\d+)_(?P<study_id>\d+)")
FOLD_RE = re.compile(r"^fold\d+$")

MANIFEST_COLUMNS = [
    "case_id",
    "patient_id",
    "study_id",
    "fold",
    "path_t2w",
    "path_adc",
    "path_hbv",
    "available_sequences",
    "clinical_row_found",
    "label_cspca",
    "pirads_score",
    "path_gland_mask",
    "path_lesion_mask",
    "has_gland_mask",
    "has_lesion_mask",
    "missing_data_flags",
]


@dataclass(frozen=True)
class ParsedImageName:
    """Case and modality parsed from a PI-CAI image filename."""

    case_id: str
    patient_id: str
    study_id: str
    modality: str


@dataclass(frozen=True)
class ManifestBuildResult:
    """Rows plus validation summary produced by a manifest build."""

    rows: list[dict[str, str]]
    report: dict[str, Any]


def build_and_write_manifest(
    raw_root: str | Path,
    output_path: str | Path,
    report_path: str | Path,
) -> ManifestBuildResult:
    """Build the PI-CAI manifest and write CSV plus JSON validation report."""

    result = build_picai_manifest(raw_root)
    write_manifest_csv(result.rows, output_path)
    write_manifest_report(result.report, report_path)
    return result


def build_picai_manifest(raw_root: str | Path) -> ManifestBuildResult:
    """Create a case-level PI-CAI manifest from a local raw data directory."""

    raw_root = Path(raw_root)
    image_root = raw_root / "images"
    label_root = raw_root / "picai_labels"

    if not raw_root.exists():
        raise FileNotFoundError(f"PI-CAI raw root does not exist: {raw_root}")
    if not image_root.exists():
        raise FileNotFoundError(f"PI-CAI images directory does not exist: {image_root}")

    image_index, image_diagnostics = index_picai_images(image_root)
    clinical_rows, clinical_info = load_clinical_rows(label_root)
    gland_masks = index_label_masks(label_root / "anatomical_delineations")
    lesion_masks = index_label_masks(label_root / "csPCa_lesion_delineations")

    rows: list[dict[str, str]] = []
    fold_mismatch_cases: list[str] = []

    for case_id in sorted(image_index):
        case_modalities = image_index[case_id]
        patient_id, study_id = split_case_id(case_id)
        folds = sorted(
            {
                item["fold"]
                for paths in case_modalities.values()
                for item in paths
                if item["fold"]
            }
        )
        if len(folds) > 1:
            fold_mismatch_cases.append(case_id)

        clinical_row = clinical_rows.get(case_id)
        clinical_row_found = clinical_row is not None
        label_cspca = ""
        pirads_score = ""
        if clinical_row is not None:
            label_cspca = clinical_row.get("case_csPCa", "")
            pirads_score = format_pirads_value(clinical_row, clinical_info["pirads_columns"])

        missing_flags: list[str] = []
        for modality in MODALITY_SUFFIXES:
            if modality not in case_modalities:
                missing_flags.append(f"missing_{modality}")
            elif len(case_modalities[modality]) > 1:
                missing_flags.append(f"duplicate_{modality}")
        if not clinical_row_found:
            missing_flags.append("missing_clinical_row")
        elif not label_cspca:
            missing_flags.append("missing_case_cspca")
        if not gland_masks.get(case_id):
            missing_flags.append("missing_gland_mask")
        if not lesion_masks.get(case_id):
            missing_flags.append("missing_lesion_mask")
        if len(folds) > 1:
            missing_flags.append("fold_mismatch")

        rows.append(
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "study_id": study_id,
                "fold": folds[0] if folds else "",
                "path_t2w": first_relative_path(case_modalities.get("t2w", []), raw_root),
                "path_adc": first_relative_path(case_modalities.get("adc", []), raw_root),
                "path_hbv": first_relative_path(case_modalities.get("hbv", []), raw_root),
                "available_sequences": pipe_join(
                    modality for modality in MODALITY_SUFFIXES if modality in case_modalities
                ),
                "clinical_row_found": str(clinical_row_found),
                "label_cspca": label_cspca,
                "pirads_score": pirads_score,
                "path_gland_mask": paths_to_string(gland_masks.get(case_id, []), raw_root),
                "path_lesion_mask": paths_to_string(lesion_masks.get(case_id, []), raw_root),
                "has_gland_mask": str(bool(gland_masks.get(case_id))),
                "has_lesion_mask": str(bool(lesion_masks.get(case_id))),
                "missing_data_flags": pipe_join(missing_flags),
            }
        )

    report = build_validation_report(
        rows=rows,
        raw_root=raw_root,
        image_root=image_root,
        label_root=label_root,
        clinical_rows=clinical_rows,
        clinical_info=clinical_info,
        gland_masks=gland_masks,
        lesion_masks=lesion_masks,
        image_diagnostics=image_diagnostics,
        fold_mismatch_cases=fold_mismatch_cases,
    )
    return ManifestBuildResult(rows=rows, report=report)


def index_picai_images(image_root: str | Path) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    """Index PI-CAI images by case ID and bpMRI modality."""

    image_root = Path(image_root)
    image_index: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    non_manifest_files: dict[str, list[str]] = defaultdict(list)
    skipped_files: list[str] = []

    for image_path in sorted(iter_medical_image_files(image_root)):
        parsed = parse_image_filename(image_path)
        if parsed is None:
            non_manifest = parse_non_manifest_image_filename(image_path)
            if non_manifest is None:
                skipped_files.append(str(image_path))
            else:
                non_manifest_files[non_manifest.modality].append(str(image_path))
            continue
        image_index[parsed.case_id][parsed.modality].append(
            {
                "path": image_path,
                "fold": find_fold_name(image_path.relative_to(image_root)),
            }
        )

    duplicates = []
    for case_id, modalities in image_index.items():
        for modality, paths in modalities.items():
            if len(paths) > 1:
                duplicates.append(
                    {
                        "case_id": case_id,
                        "modality": modality,
                        "paths": [str(item["path"]) for item in paths],
                    }
                )

    return dict(image_index), {
        "non_manifest_image_files": dict(non_manifest_files),
        "skipped_image_files": skipped_files,
        "duplicate_image_modalities": duplicates,
    }


def parse_image_filename(path: str | Path) -> ParsedImageName | None:
    """Parse PI-CAI image filenames like ``10000_1000000_t2w.mha``."""

    return parse_image_filename_with_suffixes(path, MODALITY_SUFFIXES)


def parse_non_manifest_image_filename(path: str | Path) -> ParsedImageName | None:
    """Parse recognized PI-CAI image planes excluded from the Stage 1 manifest."""

    return parse_image_filename_with_suffixes(path, NON_MANIFEST_IMAGE_SUFFIXES)


def parse_image_filename_with_suffixes(
    path: str | Path,
    suffixes: dict[str, str],
) -> ParsedImageName | None:
    """Parse a PI-CAI image filename with explicit modality suffixes."""

    stem = strip_medical_extension(Path(path).name)
    lower_stem = stem.lower()
    for modality, suffix in suffixes.items():
        if lower_stem.endswith(suffix):
            case_id = stem[: -len(suffix)]
            patient_id, study_id = split_case_id(case_id)
            if not patient_id or not study_id:
                return None
            return ParsedImageName(
                case_id=case_id,
                patient_id=patient_id,
                study_id=study_id,
                modality=modality,
            )
    return None


def split_case_id(case_id: str) -> tuple[str, str]:
    """Return patient and study IDs from a PI-CAI case ID."""

    match = CASE_ID_RE.search(case_id)
    if match:
        return match.group("patient_id"), match.group("study_id")

    parts = case_id.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", ""


def load_clinical_rows(label_root: str | Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Load PI-CAI clinical marksheet rows keyed by ``patient_id_study_id``."""

    marksheet_path = Path(label_root) / "clinical_information" / "marksheet.csv"
    info: dict[str, Any] = {
        "marksheet_path": str(marksheet_path),
        "marksheet_found": marksheet_path.exists(),
        "fieldnames": [],
        "pirads_columns": [],
        "missing_required_columns": [],
        "row_count": 0,
    }
    if not marksheet_path.exists():
        return {}, info

    rows: dict[str, dict[str, str]] = {}
    with marksheet_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        info["fieldnames"] = fieldnames
        info["pirads_columns"] = find_pirads_columns(fieldnames)

        required_columns = {"patient_id", "study_id", "case_csPCa"}
        info["missing_required_columns"] = sorted(required_columns - set(fieldnames))
        if "patient_id" not in fieldnames or "study_id" not in fieldnames:
            return {}, info

        for row in reader:
            normalized_row = {key: (value or "").strip() for key, value in row.items() if key}
            patient_id = normalized_row.get("patient_id", "")
            study_id = normalized_row.get("study_id", "")
            if not patient_id or not study_id:
                continue
            rows[f"{patient_id}_{study_id}"] = normalized_row
        info["row_count"] = len(rows)
    return rows, info


def index_label_masks(mask_root: str | Path) -> dict[str, list[Path]]:
    """Index anatomical or lesion label masks by case ID."""

    mask_root = Path(mask_root)
    masks: dict[str, list[Path]] = defaultdict(list)
    if not mask_root.exists():
        return {}

    for mask_path in sorted(iter_medical_image_files(mask_root)):
        case_id = parse_case_id_from_label(mask_path)
        if case_id:
            masks[case_id].append(mask_path)
    return dict(masks)


def parse_case_id_from_label(path: str | Path) -> str | None:
    """Parse a PI-CAI case ID from a label filename or parent path."""

    text = str(path)
    match = CASE_ID_RE.search(text)
    if match:
        return f"{match.group('patient_id')}_{match.group('study_id')}"
    return None


def build_validation_report(
    rows: list[dict[str, str]],
    raw_root: Path,
    image_root: Path,
    label_root: Path,
    clinical_rows: dict[str, dict[str, str]],
    clinical_info: dict[str, Any],
    gland_masks: dict[str, list[Path]],
    lesion_masks: dict[str, list[Path]],
    image_diagnostics: dict[str, Any],
    fold_mismatch_cases: list[str],
) -> dict[str, Any]:
    """Build a compact validation report for manifest quality control."""

    case_ids = {row["case_id"] for row in rows}
    missing_counter: Counter[str] = Counter()
    for row in rows:
        for flag in split_pipe_value(row["missing_data_flags"]):
            missing_counter[flag] += 1

    modality_counts = {
        "t2w": sum(1 for row in rows if row["path_t2w"]),
        "adc": sum(1 for row in rows if row["path_adc"]),
        "hbv": sum(1 for row in rows if row["path_hbv"]),
    }
    fold_counts = Counter(row["fold"] or "unknown" for row in rows)
    orphan_clinical = sorted(set(clinical_rows) - case_ids)
    orphan_gland = sorted(set(gland_masks) - case_ids)
    orphan_lesion = sorted(set(lesion_masks) - case_ids)
    non_manifest_files = image_diagnostics["non_manifest_image_files"]
    non_manifest_counts = {
        suffix: len(paths)
        for suffix, paths in sorted(non_manifest_files.items())
    }
    non_manifest_sample = [
        path
        for suffix in sorted(non_manifest_files)
        for path in non_manifest_files[suffix][:10]
    ][:20]

    warnings = []
    if not label_root.exists():
        warnings.append("picai_labels directory was not found")
    if clinical_info["missing_required_columns"]:
        warnings.append(
            "clinical marksheet is missing required columns: "
            + ", ".join(clinical_info["missing_required_columns"])
        )

    return {
        "schema_version": "1.0",
        "raw_root": str(raw_root),
        "image_root": str(image_root),
        "label_root": str(label_root),
        "manifest_columns": MANIFEST_COLUMNS,
        "total_cases": len(rows),
        "cases_by_fold": dict(sorted(fold_counts.items())),
        "modality_available_counts": modality_counts,
        "clinical_marksheet": clinical_info,
        "clinical_rows_linked": sum(1 for row in rows if row["clinical_row_found"] == "True"),
        "clinical_rows_orphaned_count": len(orphan_clinical),
        "clinical_rows_orphaned_sample": orphan_clinical[:20],
        "gland_mask_cases_linked": sum(1 for row in rows if row["has_gland_mask"] == "True"),
        "lesion_mask_cases_linked": sum(1 for row in rows if row["has_lesion_mask"] == "True"),
        "orphan_gland_mask_cases_count": len(orphan_gland),
        "orphan_gland_mask_cases_sample": orphan_gland[:20],
        "orphan_lesion_mask_cases_count": len(orphan_lesion),
        "orphan_lesion_mask_cases_sample": orphan_lesion[:20],
        "missing_data_counts": dict(sorted(missing_counter.items())),
        "duplicate_image_modalities_count": len(image_diagnostics["duplicate_image_modalities"]),
        "duplicate_image_modalities_sample": image_diagnostics["duplicate_image_modalities"][:20],
        "non_manifest_image_files_count": sum(non_manifest_counts.values()),
        "non_manifest_image_files_by_suffix": non_manifest_counts,
        "non_manifest_image_files_sample": non_manifest_sample,
        "skipped_image_files_count": len(image_diagnostics["skipped_image_files"]),
        "skipped_image_files_sample": image_diagnostics["skipped_image_files"][:20],
        "fold_mismatch_cases_count": len(fold_mismatch_cases),
        "fold_mismatch_cases_sample": fold_mismatch_cases[:20],
        "warnings": warnings,
    }


def write_manifest_csv(rows: list[dict[str, str]], output_path: str | Path) -> None:
    """Write manifest rows as CSV using the fixed Stage 1 schema."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest_report(report: dict[str, Any], report_path: str | Path) -> None:
    """Write the validation report as formatted JSON."""

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, sort_keys=True)
        report_file.write("\n")


def iter_medical_image_files(root: str | Path) -> Iterable[Path]:
    """Yield files with common medical image extensions under ``root``."""

    root = Path(root)
    if not root.exists():
        return []
    return (
        path
        for path in root.rglob("*")
        if path.is_file() and has_medical_image_extension(path)
    )


def has_medical_image_extension(path: str | Path) -> bool:
    """Return whether ``path`` has a supported medical image file extension."""

    lower_name = Path(path).name.lower()
    return any(lower_name.endswith(extension) for extension in MEDICAL_IMAGE_EXTENSIONS)


def strip_medical_extension(filename: str) -> str:
    """Remove a supported medical image extension from a filename."""

    lower_name = filename.lower()
    for extension in MEDICAL_IMAGE_EXTENSIONS:
        if lower_name.endswith(extension):
            return filename[: -len(extension)]
    return Path(filename).stem


def find_fold_name(relative_path: Path) -> str:
    """Find the PI-CAI fold directory name from an image path."""

    for part in relative_path.parts:
        if FOLD_RE.match(part):
            return part
    return ""


def find_pirads_columns(fieldnames: list[str]) -> list[str]:
    """Return PI-RADS columns by name without inventing scores."""

    return [
        fieldname
        for fieldname in fieldnames
        if "pirads" in re.sub(r"[^a-z0-9]", "", fieldname.lower())
    ]


def format_pirads_value(row: dict[str, str], pirads_columns: list[str]) -> str:
    """Format available PI-RADS value(s) from explicit PI-RADS columns."""

    values = [(column, row.get(column, "")) for column in pirads_columns]
    values = [(column, value) for column, value in values if value]
    if not values:
        return ""
    if len(values) == 1:
        return values[0][1]
    return pipe_join(f"{column}={value}" for column, value in values)


def first_relative_path(items: list[dict[str, Any]], root: Path) -> str:
    """Return the first sorted relative path from indexed image items."""

    if not items:
        return ""
    return relative_to_root(sorted(item["path"] for item in items)[0], root)


def paths_to_string(paths: list[Path], root: Path) -> str:
    """Return pipe-separated relative paths for mask fields."""

    return pipe_join(relative_to_root(path, root) for path in sorted(paths))


def relative_to_root(path: Path, root: Path) -> str:
    """Return a POSIX-style path relative to ``root`` when possible."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def pipe_join(values: Iterable[str]) -> str:
    """Join non-empty values with a stable pipe delimiter."""

    return "|".join(str(value) for value in values if str(value))


def split_pipe_value(value: str) -> list[str]:
    """Split a pipe-delimited manifest field into non-empty values."""

    if not value:
        return []
    return [item for item in value.split("|") if item]
