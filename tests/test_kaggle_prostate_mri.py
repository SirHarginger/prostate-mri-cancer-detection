from pathlib import Path

import pytest

from prostate_detection.preprocessing.kaggle_prostate_mri import (
    KAGGLE_EXPECTED_SUBJECTS,
    _case_id_from_subject,
    select_kaggle_t2_series,
)


KAGGLE_INPUT_DIR = Path("data/raw/world-wide-covid-dataset/PROSTATE_MRI")


def test_kaggle_case_id_from_subject_is_nnunet_safe() -> None:
    assert _case_id_from_subject("MIP-PROSTATE-01-0001") == "mip_prostate_01_0001"


@pytest.mark.skipif(
    not (KAGGLE_INPUT_DIR / "metadata.csv").is_file(),
    reason="Kaggle PROSTATE_MRI raw data is not present.",
)
def test_select_kaggle_t2_axial_series_count_and_paths() -> None:
    selected = select_kaggle_t2_series(KAGGLE_INPUT_DIR)

    assert len(selected) == KAGGLE_EXPECTED_SUBJECTS
    assert len({series.subject_id for series in selected}) == KAGGLE_EXPECTED_SUBJECTS
    assert all(series.series_description == "T2 TSE ax hi" for series in selected)
    assert all(series.series_dir.is_dir() for series in selected)
    assert all(series.number_of_dicoms > 0 for series in selected)
