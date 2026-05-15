from pathlib import Path

from prostate_detection.preprocessing.msd_binary_roi import (
    is_nifti_file,
    strip_nii_suffix,
)


def test_strip_nii_suffix() -> None:
    assert strip_nii_suffix(Path("prostate_00.nii.gz")) == "prostate_00"
    assert strip_nii_suffix(Path("prostate_00.nii")) == "prostate_00"


def test_is_nifti_file_ignores_appledouble_sidecars() -> None:
    assert is_nifti_file(Path("prostate_00.nii.gz"))
    assert is_nifti_file(Path("prostate_00.nii"))
    assert not is_nifti_file(Path("._prostate_00.nii.gz"))
    assert not is_nifti_file(Path("prostate_00.json"))
