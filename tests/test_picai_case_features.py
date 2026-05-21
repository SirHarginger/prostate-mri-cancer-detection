import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path("scripts/classification/extract_picai_case_features.py")


def load_module():
    spec = importlib.util.spec_from_file_location(
        "extract_picai_case_features", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_picai_case_features_imports() -> None:
    module = load_module()

    assert module.RADIOMICS_PREFIX == "t2w_wholegland_"


def test_read_core_manifest_filters_to_core_bpmri(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    module = load_module()

    manifest = tmp_path / "manifest.csv"
    df = pd.DataFrame(
        [
            {
                "case_key": "case_1",
                "patient_id": "p1",
                "study_id": "s1",
                "has_core_bpMRI": True,
                "t2w_image_path": "/tmp/t2w_1.mha",
                "whole_gland_mask_path": "/tmp/mask_1.nii.gz",
                "patient_age": 65,
                "psa": 7.2,
                "psad": 0.14,
                "prostate_volume": 52,
                "center": "A",
                "case_cspca_binary": 1,
                "case_isup_int": 2,
            },
            {
                "case_key": "case_2",
                "patient_id": "p2",
                "study_id": "s2",
                "has_core_bpMRI": False,
                "t2w_image_path": "",
                "whole_gland_mask_path": "/tmp/mask_2.nii.gz",
                "patient_age": 61,
                "psa": 4.1,
                "psad": 0.09,
                "prostate_volume": 45,
                "center": "B",
                "case_cspca_binary": 0,
                "case_isup_int": 0,
            },
        ]
    )
    df.to_csv(manifest, index=False)

    out = module.read_core_manifest(manifest)

    assert out["case_key"].tolist() == ["case_1"]


def test_format_radiomics_result_prefixes_and_drops_diagnostics() -> None:
    module = load_module()

    out = module.format_radiomics_result(
        {
            "diagnostics_Versions_PyRadiomics": "ignored",
            "original_firstorder_Mean": 42.0,
        }
    )

    assert out == {"t2w_wholegland_original_firstorder_Mean": 42.0}


def test_binarize_mask_uses_positive_voxels_only() -> None:
    np = pytest.importorskip("numpy")
    sitk = pytest.importorskip("SimpleITK")
    module = load_module()

    arr = np.zeros((2, 4, 4), dtype=np.int16)
    arr[0, 1, 1] = 2
    arr[1, 2, 3] = 5
    mask = sitk.GetImageFromArray(arr)

    binary = module.binarize_mask(mask)
    binary_arr = sitk.GetArrayFromImage(binary)

    assert binary.GetPixelID() == sitk.sitkUInt8
    assert set(binary_arr.flatten().tolist()) == {0, 1}
    assert int(binary_arr.sum()) == 2


def test_resample_mask_to_image_matches_reference_geometry() -> None:
    np = pytest.importorskip("numpy")
    sitk = pytest.importorskip("SimpleITK")
    module = load_module()

    image = sitk.Image([6, 6, 2], sitk.sitkFloat32)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))

    arr = np.zeros((2, 3, 3), dtype=np.uint8)
    arr[:, 1, 1] = 1
    mask = sitk.GetImageFromArray(arr)
    mask.SetSpacing((2.0, 2.0, 1.0))
    mask.SetOrigin((0.0, 0.0, 0.0))

    resampled = module.resample_mask_to_image(mask, image)
    resampled_arr = sitk.GetArrayFromImage(resampled)

    assert module.images_have_same_geometry(image, resampled)
    assert resampled.GetPixelID() == sitk.sitkUInt8
    assert set(resampled_arr.flatten().tolist()).issubset({0, 1})
    assert module.nonzero_voxel_count(resampled) > 0


def test_resample_mask_to_image_normalizes_tiny_metadata_offsets() -> None:
    sitk = pytest.importorskip("SimpleITK")
    module = load_module()

    image = sitk.Image([4, 4, 2], sitk.sitkFloat32)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))

    mask = sitk.Image([4, 4, 2], sitk.sitkUInt8)
    mask.SetSpacing((1.0, 1.0, 1.0))
    mask.SetOrigin((0.000001, 0.0, 0.0))
    mask.SetPixel(1, 1, 0, 1)

    assert module.images_have_same_geometry(image, mask)

    resampled = module.resample_mask_to_image(mask, image)

    assert resampled.GetOrigin() == image.GetOrigin()
    assert resampled.GetSpacing() == image.GetSpacing()
    assert resampled.GetDirection() == image.GetDirection()
    assert module.images_have_same_geometry(image, resampled)
