from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - exercised when optional deps are absent.
    np = None
    sitk = None


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "nnunet_autosegmentation" / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(sitk is None or np is None, "SimpleITK and NumPy are required")
class NnUNetAutosegmentationTests(unittest.TestCase):
    def test_combined_label_image_lets_lesion_override_gland(self) -> None:
        prepare = load_script("prepare_picai_nnunet_training.py")
        reference = image_from_array(np.zeros((2, 4, 4), dtype=np.float32))
        gland = image_from_array(mask_with_voxels((2, 4, 4), [(0, 1, 1), (0, 1, 2), (0, 2, 2)]))
        lesion = image_from_array(mask_with_voxels((2, 4, 4), [(0, 1, 2)]))

        label, counts = prepare.combined_label_image(reference, gland, lesion, sitk, np)
        label_array = sitk.GetArrayFromImage(label)

        self.assertEqual(int(label_array[0, 1, 1]), 1)
        self.assertEqual(int(label_array[0, 1, 2]), 2)
        self.assertEqual(counts["gland_voxels"], 3)
        self.assertEqual(counts["label_1_voxels"], 2)
        self.assertEqual(counts["label_2_voxels"], 1)

    def test_prepare_case_writes_three_channels_and_label(self) -> None:
        prepare = load_script("prepare_picai_nnunet_training.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = root / "raw"
            dataset_dir = root / "nnunet" / "Dataset910_PI_CAIGlandLesion"
            write_image(raw_root / "images/fold0/10000_1000000_t2w.mha", np.ones((2, 4, 4), dtype=np.float32))
            write_image(raw_root / "images/fold0/10000_1000000_adc.mha", np.full((2, 4, 4), 2, dtype=np.float32))
            write_image(raw_root / "images/fold0/10000_1000000_hbv.mha", np.full((2, 4, 4), 3, dtype=np.float32))
            write_image(
                raw_root / "picai_labels/anatomical_delineations/10000_1000000.nii.gz",
                mask_with_voxels((2, 4, 4), [(0, 1, 1), (0, 1, 2)]),
            )
            write_image(
                raw_root / "picai_labels/csPCa_lesion_delineations/10000_1000000.nii.gz",
                mask_with_voxels((2, 4, 4), [(0, 1, 2)]),
            )
            row = {
                "case_id": "10000_1000000",
                "fold": "fold0",
                "path_t2w": "images/fold0/10000_1000000_t2w.mha",
                "path_adc": "images/fold0/10000_1000000_adc.mha",
                "path_hbv": "images/fold0/10000_1000000_hbv.mha",
                "path_gland_mask": "picai_labels/anatomical_delineations/10000_1000000.nii.gz",
                "path_lesion_mask": "picai_labels/csPCa_lesion_delineations/10000_1000000.nii.gz",
            }

            result = prepare.prepare_case(row, raw_root, dataset_dir, sitk, np)

            self.assertTrue(result["written"])
            self.assertTrue((dataset_dir / "imagesTr/10000_1000000_0000.nii.gz").exists())
            self.assertTrue((dataset_dir / "imagesTr/10000_1000000_0001.nii.gz").exists())
            self.assertTrue((dataset_dir / "imagesTr/10000_1000000_0002.nii.gz").exists())
            self.assertTrue((dataset_dir / "labelsTr/10000_1000000.nii.gz").exists())

    def test_zero_like_reference_preserves_shape_for_missing_external_adc(self) -> None:
        prepare_external = load_script("prepare_nnunet_dataset.py")
        reference = image_from_array(np.ones((3, 5, 7), dtype=np.float32))

        zero = prepare_external.zero_like_reference(sitk, reference)
        arr = sitk.GetArrayFromImage(zero)

        self.assertEqual(arr.shape, (3, 5, 7))
        self.assertEqual(float(arr.max()), 0.0)
        self.assertEqual(zero.GetSize(), reference.GetSize())

    def test_label_specific_feature_rows_handle_empty_lesion(self) -> None:
        extract = load_script("extract_features_from_masks.py")
        image = image_from_array(np.arange(32, dtype=np.float32).reshape(2, 4, 4))
        image_array = sitk.GetArrayFromImage(image)
        prostate_mask = np.zeros((2, 4, 4), dtype=bool)
        prostate_mask[0, 1:3, 1:3] = True
        lesion_mask = np.zeros((2, 4, 4), dtype=bool)

        prostate = extract.feature_row(
            case_id="case",
            image_path=Path("case_0000.nii.gz"),
            mask_path=Path("case.nii.gz"),
            label_name="prostate_gland",
            label_value=1,
            image=image,
            image_array=image_array,
            label_mask=prostate_mask,
            np=np,
        )
        lesion = extract.feature_row(
            case_id="case",
            image_path=Path("case_0000.nii.gz"),
            mask_path=Path("case.nii.gz"),
            label_name="cspca_lesion",
            label_value=2,
            image=image,
            image_array=image_array,
            label_mask=lesion_mask,
            np=np,
        )

        self.assertEqual(prostate["empty_mask"], "False")
        self.assertEqual(prostate["voxel_count"], 4)
        self.assertEqual(lesion["empty_mask"], "True")
        self.assertEqual(lesion["voxel_count"], 0)


def image_from_array(array):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 3.0))
    return image


def mask_with_voxels(shape, coords):
    array = np.zeros(shape, dtype=np.uint8)
    for coord in coords:
        array[coord] = 1
    return array


def write_image(path: Path, array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image_from_array(array), str(path))


if __name__ == "__main__":
    unittest.main()
