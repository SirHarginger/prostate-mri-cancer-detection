import json
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from prostate_detection.preprocessing.prostate158 import PROSTATE158_EXPECTED_CASES


DATASET502_DIR = Path("data/nnunet/nnUNet_raw/Dataset502_Prostate158_Anatomy")
DATASET503_DIR = Path("data/nnunet/nnUNet_raw/Dataset503_Prostate158_Lesion")
MANIFEST_PATH = Path("data/manifests/prostate158_manifest.csv")


def _manifest_rows() -> list[dict[str, str]]:
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_dataset502_structure_and_counts() -> None:
    assert (DATASET502_DIR / "dataset.json").is_file()
    assert (DATASET502_DIR / "imagesTr").is_dir()
    assert (DATASET502_DIR / "labelsTr").is_dir()
    assert len(list((DATASET502_DIR / "imagesTr").glob("*.nii.gz"))) == PROSTATE158_EXPECTED_CASES
    assert len(list((DATASET502_DIR / "labelsTr").glob("*.nii.gz"))) == PROSTATE158_EXPECTED_CASES


def test_dataset503_structure_and_counts() -> None:
    assert (DATASET503_DIR / "dataset.json").is_file()
    assert (DATASET503_DIR / "imagesTr").is_dir()
    assert (DATASET503_DIR / "labelsTr").is_dir()
    assert len(list((DATASET503_DIR / "imagesTr").glob("*.nii.gz"))) == (
        PROSTATE158_EXPECTED_CASES * 3
    )
    assert len(list((DATASET503_DIR / "labelsTr").glob("*.nii.gz"))) == PROSTATE158_EXPECTED_CASES


def test_dataset502_json_definitions() -> None:
    with (DATASET502_DIR / "dataset.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["channel_names"] == {"0": "T2"}
    assert metadata["labels"] == {
        "background": 0,
        "anatomy_label_1": 1,
        "anatomy_label_2": 2,
    }
    assert metadata["numTraining"] == PROSTATE158_EXPECTED_CASES
    assert metadata["file_ending"] == ".nii.gz"


def test_dataset503_json_definitions() -> None:
    with (DATASET503_DIR / "dataset.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["channel_names"] == {"0": "T2", "1": "ADC", "2": "DWI"}
    assert metadata["labels"] == {"background": 0, "suspicious_lesion": 1}
    assert metadata["numTraining"] == PROSTATE158_EXPECTED_CASES
    assert metadata["file_ending"] == ".nii.gz"


def test_dataset502_labels_are_anatomy_values() -> None:
    for label_path in sorted((DATASET502_DIR / "labelsTr").glob("*.nii.gz")):
        values = set(np.unique(np.asanyarray(nib.load(str(label_path)).dataobj)).astype(int).tolist())
        assert values.issubset({0, 1, 2}), f"{label_path} has values {sorted(values)}"


def test_dataset503_labels_are_binary() -> None:
    for label_path in sorted((DATASET503_DIR / "labelsTr").glob("*.nii.gz")):
        values = set(np.unique(np.asanyarray(nib.load(str(label_path)).dataobj)).astype(int).tolist())
        assert values.issubset({0, 1}), f"{label_path} has values {sorted(values)}"


def test_dataset503_generated_negative_masks_match_adc_geometry() -> None:
    negative_rows = [row for row in _manifest_rows() if row["lesion_present"] == "false"]
    assert negative_rows
    for row in negative_rows:
        case_id = f"prostate158_{row['case_id']}"
        adc = nib.load(row["adc_path"])
        label = nib.load(str(DATASET503_DIR / "labelsTr" / f"{case_id}.nii.gz"))
        assert label.shape == adc.shape
        assert np.allclose(label.affine, adc.affine)
        values = set(np.unique(np.asanyarray(label.dataobj)).astype(int).tolist())
        assert values == {0}
