"""Create nnU-Net raw datasets from prepared prostate MRI preprocessing outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prostate_detection.preprocessing.msd_binary_roi import (
    display_path,
    is_nifti_file,
    load_dataset_json,
    require_nibabel,
    resolve_msd_path,
    strip_nii_suffix,
)


EXPECTED_DATASET_ID = "Dataset501_ProstateROI_T2"
EXPECTED_NUM_TRAINING = 32
EXPECTED_NUM_TEST = 16


@dataclass(frozen=True)
class NnUNetTrainingCase:
    """A resolved MSD training case and its nnU-Net output names."""

    source_case_id: str
    nnunet_case_id: str
    image_path: Path
    binary_label_path: Path
    output_image_path: Path
    output_label_path: Path


@dataclass(frozen=True)
class NnUNetTestCase:
    """A resolved MSD test case and its nnU-Net output name."""

    source_case_id: str
    nnunet_case_id: str
    image_path: Path
    output_image_path: Path


@dataclass(frozen=True)
class NnUNetDatasetPlan:
    """All cases and metadata required to create an nnU-Net raw dataset."""

    metadata: dict[str, Any]
    training_cases: list[NnUNetTrainingCase]
    test_cases: list[NnUNetTestCase]
    output_dir: Path


def resolve_manifest_path(path_value: str, manifest_path: Path) -> Path:
    """Resolve a manifest path written as absolute or repo-relative text."""
    raw_path = Path(path_value)
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        Path.cwd() / raw_path,
        manifest_path.parent / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_binary_roi_manifest(binary_roi_dir: Path) -> dict[str, Path]:
    """Read and validate the manifest from the binary ROI conversion step."""
    manifest_path = binary_roi_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing binary ROI manifest: {manifest_path}")

    required_fields = {"image_path", "original_label_path", "binary_label_path", "case_id"}
    mapping: dict[str, Path] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = required_fields - fieldnames
        if missing:
            raise ValueError(
                f"Binary ROI manifest is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            case_id = str(row["case_id"])
            if case_id in mapping:
                raise ValueError(f"Duplicate case_id in binary ROI manifest: {case_id}")
            binary_label_path = resolve_manifest_path(row["binary_label_path"], manifest_path)
            if not binary_label_path.is_file():
                raise FileNotFoundError(
                    f"Binary ROI label listed in manifest does not exist: {binary_label_path}"
                )
            mapping[case_id] = binary_label_path

    if len(mapping) != EXPECTED_NUM_TRAINING:
        raise ValueError(
            f"Expected {EXPECTED_NUM_TRAINING} binary ROI manifest rows, found {len(mapping)}"
        )
    return mapping


def verify_binary_label_count(binary_roi_dir: Path) -> None:
    """Verify the expected number of binary label NIfTI files exists."""
    labels_dir = binary_roi_dir / "labelsTr"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing binary ROI labelsTr directory: {labels_dir}")
    label_files = sorted(path for path in labels_dir.iterdir() if is_nifti_file(path))
    if len(label_files) != EXPECTED_NUM_TRAINING:
        raise ValueError(
            f"Expected {EXPECTED_NUM_TRAINING} binary ROI labels, found {len(label_files)} "
            f"in {labels_dir}"
        )


def discover_msd_test_images(msd_dir: Path, metadata: dict[str, Any]) -> list[Path]:
    """Resolve and validate MSD test image paths."""
    test_entries = metadata.get("test")
    if not isinstance(test_entries, list):
        raise ValueError("MSD dataset.json must contain a list field named 'test'")

    num_test = metadata.get("numTest")
    if num_test is not None and int(num_test) != len(test_entries):
        raise ValueError(
            f"dataset.json numTest={num_test} but test has {len(test_entries)} entries"
        )
    if len(test_entries) != EXPECTED_NUM_TEST:
        raise ValueError(f"Expected {EXPECTED_NUM_TEST} test images, found {len(test_entries)}")

    images_ts_dir = msd_dir / "imagesTs"
    if not images_ts_dir.is_dir():
        raise FileNotFoundError(f"Missing imagesTs directory: {images_ts_dir}")
    image_files = sorted(path for path in images_ts_dir.iterdir() if is_nifti_file(path))
    if len(image_files) != EXPECTED_NUM_TEST:
        raise ValueError(
            f"Expected {EXPECTED_NUM_TEST} imagesTs NIfTI files, found {len(image_files)}"
        )

    resolved: list[Path] = []
    seen: set[str] = set()
    for entry in test_entries:
        image_path = resolve_msd_path(msd_dir, str(entry))
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing MSD test image: {image_path}")
        case_id = strip_nii_suffix(image_path)
        if case_id in seen:
            raise ValueError(f"Duplicate MSD test case_id: {case_id}")
        seen.add(case_id)
        resolved.append(image_path)
    return resolved


def build_dataset501_plan(
    msd_dir: Path,
    binary_roi_dir: Path,
    output_dir: Path,
) -> NnUNetDatasetPlan:
    """Create a validated conversion plan for Dataset501_ProstateROI_T2."""
    metadata = load_dataset_json(msd_dir)
    training = metadata.get("training")
    if not isinstance(training, list):
        raise ValueError("MSD dataset.json must contain a list field named 'training'")
    num_training = metadata.get("numTraining")
    if num_training is not None and int(num_training) != len(training):
        raise ValueError(
            f"dataset.json numTraining={num_training} but training has {len(training)} entries"
        )
    if len(training) != EXPECTED_NUM_TRAINING:
        raise ValueError(
            f"Expected {EXPECTED_NUM_TRAINING} training images, found {len(training)}"
        )

    binary_label_by_case = read_binary_roi_manifest(binary_roi_dir)
    verify_binary_label_count(binary_roi_dir)

    images_tr_dir = msd_dir / "imagesTr"
    if not images_tr_dir.is_dir():
        raise FileNotFoundError(f"Missing imagesTr directory: {images_tr_dir}")
    image_files = sorted(path for path in images_tr_dir.iterdir() if is_nifti_file(path))
    if len(image_files) != EXPECTED_NUM_TRAINING:
        raise ValueError(
            f"Expected {EXPECTED_NUM_TRAINING} imagesTr NIfTI files, found {len(image_files)}"
        )

    training_cases: list[NnUNetTrainingCase] = []
    seen_case_ids: set[str] = set()
    for index, item in enumerate(training):
        if not isinstance(item, dict) or "image" not in item:
            raise ValueError("Each MSD training entry must contain an 'image' path")
        image_path = resolve_msd_path(msd_dir, str(item["image"]))
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing MSD training image: {image_path}")
        source_case_id = strip_nii_suffix(image_path)
        if source_case_id in seen_case_ids:
            raise ValueError(f"Duplicate MSD training case_id: {source_case_id}")
        seen_case_ids.add(source_case_id)
        if source_case_id not in binary_label_by_case:
            raise FileNotFoundError(
                f"No binary ROI label found for MSD training case: {source_case_id}"
            )

        nnunet_case_id = f"prostate_{index:03d}"
        training_cases.append(
            NnUNetTrainingCase(
                source_case_id=source_case_id,
                nnunet_case_id=nnunet_case_id,
                image_path=image_path,
                binary_label_path=binary_label_by_case[source_case_id],
                output_image_path=output_dir / "imagesTr" / f"{nnunet_case_id}_0000.nii.gz",
                output_label_path=output_dir / "labelsTr" / f"{nnunet_case_id}.nii.gz",
            )
        )

    test_cases = [
        NnUNetTestCase(
            source_case_id=strip_nii_suffix(image_path),
            nnunet_case_id=f"prostate_{index:03d}",
            image_path=image_path,
            output_image_path=output_dir / "imagesTs" / f"prostate_{index:03d}_0000.nii.gz",
        )
        for index, image_path in enumerate(discover_msd_test_images(msd_dir, metadata))
    ]

    return NnUNetDatasetPlan(
        metadata=metadata,
        training_cases=training_cases,
        test_cases=test_cases,
        output_dir=output_dir,
    )


def ensure_output_ready(plan: NnUNetDatasetPlan, overwrite: bool) -> None:
    """Create output folders and fail clearly on existing outputs unless overwritten."""
    output_paths = [plan.output_dir / "dataset.json"]
    output_paths.extend(case.output_image_path for case in plan.training_cases)
    output_paths.extend(case.output_label_path for case in plan.training_cases)
    output_paths.extend(case.output_image_path for case in plan.test_cases)

    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing[:5])
        raise FileExistsError(f"Output exists and --overwrite was not set: {preview}")

    for subdir in ["imagesTr", "labelsTr", "imagesTs"]:
        (plan.output_dir / subdir).mkdir(parents=True, exist_ok=True)


def _copy_qform_sform(source_img: Any, target_img: Any) -> None:
    target_img.set_qform(source_img.get_qform(), code=int(source_img.header["qform_code"]))
    target_img.set_sform(source_img.get_sform(), code=int(source_img.header["sform_code"]))


def save_t2_channel_as_3d(image_path: Path, output_image_path: Path) -> None:
    """Extract channel 0 from an MSD 4D image and save it as nnU-Net 3D T2."""
    nib = require_nibabel()
    image = nib.load(str(image_path))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 4:
        raise ValueError(f"Expected a 4D MSD image, got shape {data.shape}: {image_path}")
    if data.shape[-1] < 1:
        raise ValueError(f"MSD image has no channel 0: {image_path}")

    t2_data = np.asarray(data[..., 0], dtype=data.dtype)
    header = image.header.copy()
    header.set_data_dtype(t2_data.dtype)
    t2_image = nib.Nifti1Image(t2_data, image.affine, header=header)
    _copy_qform_sform(image, t2_image)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(t2_image, str(output_image_path))


def save_binary_label_for_nnunet(binary_label_path: Path, output_label_path: Path) -> None:
    """Copy a prepared binary ROI label into nnU-Net labelsTr naming."""
    nib = require_nibabel()
    label = nib.load(str(binary_label_path))
    data = np.asanyarray(label.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D binary label, got shape {data.shape}: {binary_label_path}")

    values = set(np.unique(data).astype(int).tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"Label contains non-binary values {sorted(values)}: {binary_label_path}")

    label_data = np.asarray(data, dtype=np.uint8)
    header = label.header.copy()
    header.set_data_dtype(np.uint8)
    nnunet_label = nib.Nifti1Image(label_data, label.affine, header=header)
    _copy_qform_sform(label, nnunet_label)
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nnunet_label, str(output_label_path))


def build_nnunet_dataset_json(metadata: dict[str, Any]) -> dict[str, Any]:
    """Create nnU-Net v2 dataset.json metadata for Dataset501."""
    return {
        "name": EXPECTED_DATASET_ID,
        "description": (
            "T2-only whole-prostate ROI baseline derived from MSD Task05 Prostate. "
            "Original PZ and TZ labels were merged into one prostate_roi label."
        ),
        "reference": metadata.get("reference", "Radboud University, Nijmegen Medical Centre"),
        "release": metadata.get("release", metadata.get("relase", "unknown")),
        "license": metadata.get("licence", "see MSD license/source"),
        "tensorImageSize": "3D",
        "channel_names": {"0": "T2"},
        "labels": {"background": 0, "prostate_roi": 1},
        "numTraining": EXPECTED_NUM_TRAINING,
        "file_ending": ".nii.gz",
    }


def write_nnunet_dataset_json(plan: NnUNetDatasetPlan) -> None:
    dataset_json_path = plan.output_dir / "dataset.json"
    with dataset_json_path.open("w", encoding="utf-8") as f:
        json.dump(build_nnunet_dataset_json(plan.metadata), f, indent=2)
        f.write("\n")


def create_dataset501_prostate_roi_t2(
    msd_dir: Path,
    binary_roi_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> NnUNetDatasetPlan:
    """Convert prepared MSD binary ROI data to nnU-Net raw Dataset501."""
    plan = build_dataset501_plan(msd_dir, binary_roi_dir, output_dir)
    ensure_output_ready(plan, overwrite=overwrite)

    for case in plan.training_cases:
        print(
            "Converting train "
            f"{case.source_case_id} -> {display_path(case.output_image_path)}; "
            f"label -> {display_path(case.output_label_path)}"
        )
        save_t2_channel_as_3d(case.image_path, case.output_image_path)
        save_binary_label_for_nnunet(case.binary_label_path, case.output_label_path)

    for case in plan.test_cases:
        print(f"Converting test {case.source_case_id} -> {display_path(case.output_image_path)}")
        save_t2_channel_as_3d(case.image_path, case.output_image_path)

    write_nnunet_dataset_json(plan)
    print(f"Wrote nnU-Net dataset metadata: {display_path(plan.output_dir / 'dataset.json')}")
    return plan


def validate_dataset501(output_dir: Path) -> None:
    """Validate the expected Dataset501 folder structure and contents."""
    nib = require_nibabel()
    dataset_json_path = output_dir / "dataset.json"
    images_tr = sorted((output_dir / "imagesTr").glob("*.nii.gz"))
    labels_tr = sorted((output_dir / "labelsTr").glob("*.nii.gz"))
    images_ts = sorted((output_dir / "imagesTs").glob("*.nii.gz"))

    for required_dir in [output_dir / "imagesTr", output_dir / "labelsTr", output_dir / "imagesTs"]:
        if not required_dir.is_dir():
            raise FileNotFoundError(f"Missing nnU-Net directory: {required_dir}")
    if not dataset_json_path.is_file():
        raise FileNotFoundError(f"Missing nnU-Net dataset.json: {dataset_json_path}")
    if len(images_tr) != EXPECTED_NUM_TRAINING:
        raise ValueError(f"Expected {EXPECTED_NUM_TRAINING} imagesTr files, found {len(images_tr)}")
    if len(labels_tr) != EXPECTED_NUM_TRAINING:
        raise ValueError(f"Expected {EXPECTED_NUM_TRAINING} labelsTr files, found {len(labels_tr)}")
    if len(images_ts) != EXPECTED_NUM_TEST:
        raise ValueError(f"Expected {EXPECTED_NUM_TEST} imagesTs files, found {len(images_ts)}")

    with dataset_json_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("channel_names") != {"0": "T2"}:
        raise ValueError(f"Unexpected channel_names: {metadata.get('channel_names')}")
    if metadata.get("labels") != {"background": 0, "prostate_roi": 1}:
        raise ValueError(f"Unexpected labels: {metadata.get('labels')}")
    if metadata.get("numTraining") != EXPECTED_NUM_TRAINING:
        raise ValueError(f"Unexpected numTraining: {metadata.get('numTraining')}")
    if metadata.get("file_ending") != ".nii.gz":
        raise ValueError(f"Unexpected file_ending: {metadata.get('file_ending')}")

    for image_path in images_tr:
        image = nib.load(str(image_path))
        if len(image.shape) != 3:
            raise ValueError(f"Training image is not 3D: {image_path} shape={image.shape}")
    for label_path in labels_tr:
        label = nib.load(str(label_path))
        values = set(np.unique(np.asanyarray(label.dataobj)).astype(int).tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"Label contains non-binary values {sorted(values)}: {label_path}")
