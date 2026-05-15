import json
from pathlib import Path

import numpy as np

from prostate_detection.preprocessing.nnunet_conversion import (
    EXPECTED_NUM_TEST,
    EXPECTED_NUM_TRAINING,
)
from prostate_detection.preprocessing.msd_binary_roi import require_nibabel


DATASET_DIR = Path("data/nnunet/nnUNet_raw/Dataset501_ProstateROI_T2")


def test_nnunet_dataset501_folder_structure_exists() -> None:
    assert (DATASET_DIR / "dataset.json").is_file()
    assert (DATASET_DIR / "imagesTr").is_dir()
    assert (DATASET_DIR / "labelsTr").is_dir()
    assert (DATASET_DIR / "imagesTs").is_dir()


def test_nnunet_dataset501_expected_file_counts() -> None:
    assert len(list((DATASET_DIR / "imagesTr").glob("*.nii.gz"))) == EXPECTED_NUM_TRAINING
    assert len(list((DATASET_DIR / "labelsTr").glob("*.nii.gz"))) == EXPECTED_NUM_TRAINING
    assert len(list((DATASET_DIR / "imagesTs").glob("*.nii.gz"))) == EXPECTED_NUM_TEST


def test_nnunet_dataset501_training_images_are_3d() -> None:
    nib = require_nibabel()
    for image_path in sorted((DATASET_DIR / "imagesTr").glob("*.nii.gz")):
        image = nib.load(str(image_path))
        assert len(image.shape) == 3, f"{image_path} has shape {image.shape}"


def test_nnunet_dataset501_labels_are_binary() -> None:
    nib = require_nibabel()
    for label_path in sorted((DATASET_DIR / "labelsTr").glob("*.nii.gz")):
        label = nib.load(str(label_path))
        values = set(np.unique(np.asanyarray(label.dataobj)).astype(int).tolist())
        assert values.issubset({0, 1}), f"{label_path} has values {sorted(values)}"


def test_nnunet_dataset501_dataset_json_definitions() -> None:
    with (DATASET_DIR / "dataset.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["channel_names"] == {"0": "T2"}
    assert metadata["labels"] == {"background": 0, "prostate_roi": 1}
    assert metadata["numTraining"] == EXPECTED_NUM_TRAINING
    assert metadata["file_ending"] == ".nii.gz"
