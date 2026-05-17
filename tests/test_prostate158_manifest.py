import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from prostate_detection.preprocessing.prostate158 import (
    PROSTATE158_EXPECTED_CASES,
    PROSTATE158_EXPECTED_TRAIN,
    PROSTATE158_EXPECTED_VALID,
)


MANIFEST_PATH = Path("data/manifests/prostate158_manifest.csv")


def _rows() -> list[dict[str, str]]:
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_prostate158_manifest_counts_and_splits() -> None:
    rows = _rows()
    assert len(rows) == PROSTATE158_EXPECTED_CASES
    assert sum(row["split"] == "train" for row in rows) == PROSTATE158_EXPECTED_TRAIN
    assert sum(row["split"] == "valid" for row in rows) == PROSTATE158_EXPECTED_VALID


def test_prostate158_manifest_paths_exist_and_shapes_match() -> None:
    for row in _rows():
        paths = [
            Path(row["t2w_path"]),
            Path(row["adc_path"]),
            Path(row["dwi_path"]),
            Path(row["anatomy_mask_path"]),
            Path(row["adc_lesion_mask_path"]),
        ]
        for path in paths:
            assert path.is_file(), f"Missing manifest path: {path}"

        images = [nib.load(str(path)) for path in paths]
        shapes = {image.shape for image in images}
        assert len(shapes) == 1, f"Shape mismatch for case {row['case_id']}: {shapes}"


def test_prostate158_manifest_label_values_are_expected() -> None:
    for row in _rows():
        anatomy = np.asanyarray(nib.load(row["anatomy_mask_path"]).dataobj)
        anatomy_values = set(np.unique(anatomy).astype(int).tolist())
        assert anatomy_values.issubset({0, 1, 2})

        lesion = np.asanyarray(nib.load(row["adc_lesion_mask_path"]).dataobj)
        lesion_values = set(np.unique(lesion).astype(int).tolist())
        assert lesion_values.issubset({0, 1})


def test_prostate158_split_file_matches_manifest() -> None:
    split_path = Path("data/manifests/splits/prostate158_nnunet_split.json")
    assert split_path.is_file()
