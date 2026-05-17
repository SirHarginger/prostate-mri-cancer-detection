"""Prepare Prostate158 manifests and nnU-Net raw datasets."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prostate_detection.preprocessing.msd_binary_roi import display_path, require_nibabel


PROSTATE158_EXPECTED_CASES = 139
PROSTATE158_EXPECTED_TRAIN = 119
PROSTATE158_EXPECTED_VALID = 20
DATASET502_ID = "Dataset502_Prostate158_Anatomy"
DATASET503_ID = "Dataset503_Prostate158_Lesion"

MANIFEST_FIELDS = [
    "case_id",
    "patient_id",
    "split",
    "dataset_name",
    "t2w_path",
    "adc_path",
    "dwi_path",
    "anatomy_mask_path",
    "adc_lesion_mask_path",
    "t2_lesion_mask_path",
    "adc_lesion_reader2_path",
    "lesion_present",
    "shape",
    "spacing",
    "source",
    "notes",
]


@dataclass(frozen=True)
class Prostate158Case:
    """Resolved Prostate158 paths and metadata for one case."""

    case_id: str
    patient_id: str
    split: str
    t2w_path: Path
    adc_path: Path
    dwi_path: Path
    anatomy_mask_path: Path
    adc_lesion_mask_path: Path
    t2_lesion_mask_path: Path | None
    adc_lesion_reader2_path: Path | None
    lesion_present: bool
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    notes: str

    @property
    def nnunet_case_id(self) -> str:
        return f"prostate158_{self.case_id}"


@dataclass(frozen=True)
class Prostate158ManifestRecord:
    """Manifest record resolved back to filesystem paths."""

    case_id: str
    patient_id: str
    split: str
    t2w_path: Path
    adc_path: Path
    dwi_path: Path
    anatomy_mask_path: Path
    adc_lesion_mask_path: Path
    t2_lesion_mask_path: Path | None
    adc_lesion_reader2_path: Path | None
    lesion_present: bool
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    notes: str

    @property
    def nnunet_case_id(self) -> str:
        return f"prostate158_{self.case_id}"


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _resolve_csv_path(input_dir: Path, value: str, *, required: bool = True) -> Path | None:
    if not value:
        return None
    path = input_dir / value
    if required:
        _require_file(path, value)
    return path


def _load_image(path: Path) -> Any:
    nib = require_nibabel()
    return nib.load(str(path))


def _same_affine(left: Any, right: Any) -> bool:
    return bool(np.allclose(left.affine, right.affine, atol=1e-4))


def _unique_int_values(path: Path) -> set[int]:
    image = _load_image(path)
    return set(np.unique(np.asanyarray(image.dataobj)).astype(int).tolist())


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split("x")]
    if len(parts) != 3:
        raise ValueError(f"Expected shape as XxYxZ, got: {value}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def _parse_spacing(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split("x")]
    if len(parts) != 3:
        raise ValueError(f"Expected spacing as XxYxZ, got: {value}")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def _resolve_manifest_path(value: str, manifest_path: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _iter_csv_rows(input_dir: Path, split: str) -> list[dict[str, str]]:
    csv_path = _require_file(input_dir / f"{split}.csv", f"Prostate158 {split}.csv")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        row["_split"] = split
    return rows


def discover_prostate158_cases(input_dir: Path) -> list[Prostate158Case]:
    """Read Prostate158 train/valid CSV files and validate image/mask geometry."""
    rows = _iter_csv_rows(input_dir, "train") + _iter_csv_rows(input_dir, "valid")
    if len(rows) != PROSTATE158_EXPECTED_CASES:
        raise ValueError(f"Expected {PROSTATE158_EXPECTED_CASES} cases, found {len(rows)}")
    if sum(row["_split"] == "train" for row in rows) != PROSTATE158_EXPECTED_TRAIN:
        raise ValueError("Unexpected Prostate158 train case count")
    if sum(row["_split"] == "valid" for row in rows) != PROSTATE158_EXPECTED_VALID:
        raise ValueError("Unexpected Prostate158 valid case count")

    cases: list[Prostate158Case] = []
    seen: set[str] = set()
    for row in rows:
        case_id = str(row["ID"]).zfill(3)
        if case_id in seen:
            raise ValueError(f"Duplicate Prostate158 case_id: {case_id}")
        seen.add(case_id)

        t2w_path = _resolve_csv_path(input_dir, row["t2"])
        adc_path = _resolve_csv_path(input_dir, row["adc"])
        dwi_path = _resolve_csv_path(input_dir, row["dwi"])
        anatomy_path = _resolve_csv_path(input_dir, row["t2_anatomy_reader1"])
        adc_lesion_path = _resolve_csv_path(input_dir, row["adc_tumor_reader1"])
        t2_lesion_path = _resolve_csv_path(
            input_dir, row.get("t2_tumor_reader1", ""), required=False
        )
        adc_reader2_path = _resolve_csv_path(
            input_dir, row.get("adc_tumor_reader2", ""), required=False
        )

        assert t2w_path is not None
        assert adc_path is not None
        assert dwi_path is not None
        assert anatomy_path is not None
        assert adc_lesion_path is not None

        path_case_id = t2w_path.parent.name
        if path_case_id != case_id:
            raise ValueError(f"CSV ID/path mismatch for case {case_id}: {t2w_path}")

        t2_img = _load_image(t2w_path)
        adc_img = _load_image(adc_path)
        dwi_img = _load_image(dwi_path)
        anatomy_img = _load_image(anatomy_path)
        adc_lesion_img = _load_image(adc_lesion_path)

        for name, image in [
            ("adc", adc_img),
            ("dwi", dwi_img),
            ("t2_anatomy_reader1", anatomy_img),
            ("adc_tumor_reader1", adc_lesion_img),
        ]:
            if image.shape != t2_img.shape:
                raise ValueError(
                    f"Shape mismatch for case {case_id}: {name} {image.shape} vs T2 {t2_img.shape}"
                )

        if not _same_affine(adc_img, t2_img):
            raise ValueError(f"ADC affine does not match T2 for case {case_id}")
        if not _same_affine(dwi_img, t2_img):
            raise ValueError(f"DWI affine does not match T2 for case {case_id}")
        if not _same_affine(anatomy_img, t2_img):
            raise ValueError(f"Anatomy affine does not match T2 for case {case_id}")

        anatomy_values = _unique_int_values(anatomy_path)
        if not anatomy_values.issubset({0, 1, 2}):
            raise ValueError(
                f"Unexpected anatomy labels for case {case_id}: {sorted(anatomy_values)}"
            )

        adc_lesion_values = _unique_int_values(adc_lesion_path)
        if not adc_lesion_values.issubset({0, 1}):
            raise ValueError(
                f"Unexpected ADC lesion labels for case {case_id}: {sorted(adc_lesion_values)}"
            )
        lesion_present = any(value != 0 for value in adc_lesion_values)
        if lesion_present and not _same_affine(adc_lesion_img, adc_img):
            raise ValueError(f"Positive ADC lesion mask affine mismatch for case {case_id}")

        notes: list[str] = []
        if not lesion_present:
            notes.append("negative_case_zero_mask_generated_for_nnunet")
            if not _same_affine(adc_lesion_img, adc_img):
                notes.append("raw_empty_mask_affine_mismatch")

        cases.append(
            Prostate158Case(
                case_id=case_id,
                patient_id=case_id,
                split=row["_split"],
                t2w_path=t2w_path,
                adc_path=adc_path,
                dwi_path=dwi_path,
                anatomy_mask_path=anatomy_path,
                adc_lesion_mask_path=adc_lesion_path,
                t2_lesion_mask_path=t2_lesion_path,
                adc_lesion_reader2_path=adc_reader2_path,
                lesion_present=lesion_present,
                shape=tuple(int(dim) for dim in t2_img.shape),
                spacing=tuple(float(value) for value in t2_img.header.get_zooms()[:3]),
                notes=";".join(notes),
            )
        )
    return cases


def write_prostate158_manifest(
    cases: list[Prostate158Case],
    manifest_path: Path,
    split_output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write Prostate158 manifest CSV and nnU-Net split JSON."""
    existing = [path for path in [manifest_path, split_output_path] if path.exists()]
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output exists and --overwrite was not set: {preview}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "patient_id": case.patient_id,
                    "split": case.split,
                    "dataset_name": "Prostate158",
                    "t2w_path": display_path(case.t2w_path),
                    "adc_path": display_path(case.adc_path),
                    "dwi_path": display_path(case.dwi_path),
                    "anatomy_mask_path": display_path(case.anatomy_mask_path),
                    "adc_lesion_mask_path": display_path(case.adc_lesion_mask_path),
                    "t2_lesion_mask_path": display_path(case.t2_lesion_mask_path)
                    if case.t2_lesion_mask_path
                    else "",
                    "adc_lesion_reader2_path": display_path(case.adc_lesion_reader2_path)
                    if case.adc_lesion_reader2_path
                    else "",
                    "lesion_present": str(case.lesion_present).lower(),
                    "shape": "x".join(str(value) for value in case.shape),
                    "spacing": "x".join(f"{value:.6g}" for value in case.spacing),
                    "source": "Prostate158",
                    "notes": case.notes,
                }
            )

    split_output_path.parent.mkdir(parents=True, exist_ok=True)
    split = {
        "train": [case.nnunet_case_id for case in cases if case.split == "train"],
        "val": [case.nnunet_case_id for case in cases if case.split == "valid"],
    }
    with split_output_path.open("w", encoding="utf-8") as f:
        json.dump([split], f, indent=2)
        f.write("\n")


def build_prostate158_manifest(
    input_dir: Path,
    manifest_path: Path,
    split_output_path: Path,
    *,
    overwrite: bool = False,
) -> list[Prostate158Case]:
    """Build and write a validated Prostate158 manifest."""
    cases = discover_prostate158_cases(input_dir)
    write_prostate158_manifest(
        cases, manifest_path, split_output_path, overwrite=overwrite
    )
    return cases


def read_prostate158_manifest(manifest_path: Path) -> list[Prostate158ManifestRecord]:
    """Read the validated Prostate158 manifest."""
    _require_file(manifest_path, "Prostate158 manifest")
    records: list[Prostate158ManifestRecord] = []
    seen: set[str] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = set(MANIFEST_FIELDS) - fieldnames
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        for row in reader:
            case_id = row["case_id"]
            if case_id in seen:
                raise ValueError(f"Duplicate case_id in manifest: {case_id}")
            seen.add(case_id)
            records.append(
                Prostate158ManifestRecord(
                    case_id=case_id,
                    patient_id=row["patient_id"],
                    split=row["split"],
                    t2w_path=_require_file(
                        _resolve_manifest_path(row["t2w_path"], manifest_path),
                        f"T2 image for case {case_id}",
                    ),
                    adc_path=_require_file(
                        _resolve_manifest_path(row["adc_path"], manifest_path),
                        f"ADC image for case {case_id}",
                    ),
                    dwi_path=_require_file(
                        _resolve_manifest_path(row["dwi_path"], manifest_path),
                        f"DWI image for case {case_id}",
                    ),
                    anatomy_mask_path=_require_file(
                        _resolve_manifest_path(row["anatomy_mask_path"], manifest_path),
                        f"anatomy mask for case {case_id}",
                    ),
                    adc_lesion_mask_path=_require_file(
                        _resolve_manifest_path(row["adc_lesion_mask_path"], manifest_path),
                        f"ADC lesion mask for case {case_id}",
                    ),
                    t2_lesion_mask_path=_resolve_manifest_path(
                        row["t2_lesion_mask_path"], manifest_path
                    ),
                    adc_lesion_reader2_path=_resolve_manifest_path(
                        row["adc_lesion_reader2_path"], manifest_path
                    ),
                    lesion_present=row["lesion_present"].strip().lower() == "true",
                    shape=_parse_shape(row["shape"]),
                    spacing=_parse_spacing(row["spacing"]),
                    notes=row["notes"],
                )
            )
    if len(records) != PROSTATE158_EXPECTED_CASES:
        raise ValueError(f"Expected {PROSTATE158_EXPECTED_CASES} records, found {len(records)}")
    return records


def _copy_nifti(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)


def _copy_qform_sform(source_img: Any, target_img: Any) -> None:
    target_img.set_qform(source_img.get_qform(), code=int(source_img.header["qform_code"]))
    target_img.set_sform(source_img.get_sform(), code=int(source_img.header["sform_code"]))


def _save_uint8_label(source_path: Path, output_path: Path, allowed_values: set[int]) -> None:
    nib = require_nibabel()
    source = nib.load(str(source_path))
    data = np.asanyarray(source.dataobj)
    values = set(np.unique(data).astype(int).tolist())
    if not values.issubset(allowed_values):
        raise ValueError(f"Unexpected label values {sorted(values)} in {source_path}")
    label_data = np.asarray(data, dtype=np.uint8)
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    label = nib.Nifti1Image(label_data, source.affine, header=header)
    _copy_qform_sform(source, label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(label, str(output_path))


def _save_zero_mask_like(reference_path: Path, output_path: Path) -> None:
    nib = require_nibabel()
    reference = nib.load(str(reference_path))
    data = np.zeros(reference.shape, dtype=np.uint8)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    label = nib.Nifti1Image(data, reference.affine, header=header)
    _copy_qform_sform(reference, label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(label, str(output_path))


def _expected_dataset502_paths(
    records: list[Prostate158ManifestRecord], output_dir: Path
) -> list[Path]:
    paths = [output_dir / "dataset.json"]
    for record in records:
        paths.append(output_dir / "imagesTr" / f"{record.nnunet_case_id}_0000.nii.gz")
        paths.append(output_dir / "labelsTr" / f"{record.nnunet_case_id}.nii.gz")
    return paths


def _expected_dataset503_paths(
    records: list[Prostate158ManifestRecord], output_dir: Path
) -> list[Path]:
    paths = [output_dir / "dataset.json"]
    for record in records:
        paths.extend(
            [
                output_dir / "imagesTr" / f"{record.nnunet_case_id}_0000.nii.gz",
                output_dir / "imagesTr" / f"{record.nnunet_case_id}_0001.nii.gz",
                output_dir / "imagesTr" / f"{record.nnunet_case_id}_0002.nii.gz",
                output_dir / "labelsTr" / f"{record.nnunet_case_id}.nii.gz",
            ]
        )
    return paths


def _ensure_output_ready(paths: list[Path], output_dir: Path, *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing[:5])
        raise FileExistsError(f"Output exists and --overwrite was not set: {preview}")
    for subdir in ["imagesTr", "labelsTr", "imagesTs"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def _write_dataset_json(output_dir: Path, metadata: dict[str, Any]) -> None:
    with (output_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def _dataset502_json(num_training: int) -> dict[str, Any]:
    return {
        "name": DATASET502_ID,
        "description": (
            "T2-only Prostate158 anatomy segmentation dataset. Label names are "
            "kept conservative until the official numeric mapping is cited."
        ),
        "reference": "Prostate158 public dataset",
        "release": "see Prostate158 source",
        "license": "see Prostate158 license/source",
        "tensorImageSize": "3D",
        "channel_names": {"0": "T2"},
        "labels": {"background": 0, "anatomy_label_1": 1, "anatomy_label_2": 2},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }


def _dataset503_json(num_training: int) -> dict[str, Any]:
    return {
        "name": DATASET503_ID,
        "description": (
            "Multimodal Prostate158 suspicious lesion segmentation dataset using "
            "T2, ADC, and DWI images with binary ADC reader1 lesion labels."
        ),
        "reference": "Prostate158 public dataset",
        "release": "see Prostate158 source",
        "license": "see Prostate158 license/source",
        "tensorImageSize": "3D",
        "channel_names": {"0": "T2", "1": "ADC", "2": "DWI"},
        "labels": {"background": 0, "suspicious_lesion": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }


def create_dataset502_prostate158_anatomy(
    manifest_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> list[Prostate158ManifestRecord]:
    """Create nnU-Net Dataset502_Prostate158_Anatomy."""
    records = read_prostate158_manifest(manifest_path)
    _ensure_output_ready(_expected_dataset502_paths(records, output_dir), output_dir, overwrite=overwrite)

    for record in records:
        image_path = output_dir / "imagesTr" / f"{record.nnunet_case_id}_0000.nii.gz"
        label_path = output_dir / "labelsTr" / f"{record.nnunet_case_id}.nii.gz"
        print(f"Converting anatomy {record.case_id} -> {display_path(image_path)}")
        _copy_nifti(record.t2w_path, image_path)
        _save_uint8_label(record.anatomy_mask_path, label_path, {0, 1, 2})

    _write_dataset_json(output_dir, _dataset502_json(len(records)))
    print(f"Wrote nnU-Net dataset metadata: {display_path(output_dir / 'dataset.json')}")
    return records


def create_dataset503_prostate158_lesion(
    manifest_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> list[Prostate158ManifestRecord]:
    """Create nnU-Net Dataset503_Prostate158_Lesion."""
    records = read_prostate158_manifest(manifest_path)
    _ensure_output_ready(_expected_dataset503_paths(records, output_dir), output_dir, overwrite=overwrite)

    for record in records:
        image_t2 = output_dir / "imagesTr" / f"{record.nnunet_case_id}_0000.nii.gz"
        image_adc = output_dir / "imagesTr" / f"{record.nnunet_case_id}_0001.nii.gz"
        image_dwi = output_dir / "imagesTr" / f"{record.nnunet_case_id}_0002.nii.gz"
        label_path = output_dir / "labelsTr" / f"{record.nnunet_case_id}.nii.gz"
        print(f"Converting lesion {record.case_id} -> {display_path(label_path)}")
        _copy_nifti(record.t2w_path, image_t2)
        _copy_nifti(record.adc_path, image_adc)
        _copy_nifti(record.dwi_path, image_dwi)
        if record.lesion_present:
            _save_uint8_label(record.adc_lesion_mask_path, label_path, {0, 1})
        else:
            _save_zero_mask_like(record.adc_path, label_path)

    _write_dataset_json(output_dir, _dataset503_json(len(records)))
    print(f"Wrote nnU-Net dataset metadata: {display_path(output_dir / 'dataset.json')}")
    return records


def validate_dataset502(output_dir: Path) -> None:
    """Validate Dataset502 structure and label values."""
    _validate_dataset_common(
        output_dir=output_dir,
        expected_channels=1,
        expected_labels={"background": 0, "anatomy_label_1": 1, "anatomy_label_2": 2},
        allowed_label_values={0, 1, 2},
        dataset_name=DATASET502_ID,
    )


def validate_dataset503(output_dir: Path) -> None:
    """Validate Dataset503 structure and label values."""
    _validate_dataset_common(
        output_dir=output_dir,
        expected_channels=3,
        expected_labels={"background": 0, "suspicious_lesion": 1},
        allowed_label_values={0, 1},
        dataset_name=DATASET503_ID,
    )


def _validate_dataset_common(
    *,
    output_dir: Path,
    expected_channels: int,
    expected_labels: dict[str, int],
    allowed_label_values: set[int],
    dataset_name: str,
) -> None:
    nib = require_nibabel()
    dataset_json_path = _require_file(output_dir / "dataset.json", f"{dataset_name} dataset.json")
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing imagesTr directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labelsTr directory: {labels_dir}")

    labels = sorted(labels_dir.glob("*.nii.gz"))
    images = sorted(images_dir.glob("*.nii.gz"))
    if len(labels) != PROSTATE158_EXPECTED_CASES:
        raise ValueError(f"Expected {PROSTATE158_EXPECTED_CASES} labels, found {len(labels)}")
    expected_images = PROSTATE158_EXPECTED_CASES * expected_channels
    if len(images) != expected_images:
        raise ValueError(f"Expected {expected_images} images, found {len(images)}")

    with dataset_json_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("name") != dataset_name:
        raise ValueError(f"Unexpected dataset name: {metadata.get('name')}")
    if metadata.get("labels") != expected_labels:
        raise ValueError(f"Unexpected labels: {metadata.get('labels')}")
    if metadata.get("numTraining") != PROSTATE158_EXPECTED_CASES:
        raise ValueError(f"Unexpected numTraining: {metadata.get('numTraining')}")
    if metadata.get("file_ending") != ".nii.gz":
        raise ValueError(f"Unexpected file_ending: {metadata.get('file_ending')}")

    for label_path in labels:
        label = nib.load(str(label_path))
        values = set(np.unique(np.asanyarray(label.dataobj)).astype(int).tolist())
        if not values.issubset(allowed_label_values):
            raise ValueError(f"Unexpected values {sorted(values)} in {label_path}")
