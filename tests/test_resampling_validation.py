from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from prostate_mri_cancer_detection.preprocessing import validate_resampling_plan, write_preprocessed_sample


try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - depends on local environment.
    sitk = None


@unittest.skipIf(sitk is None, "SimpleITK is not installed")
class ResamplingValidationTests(unittest.TestCase):
    def test_validates_adc_hbv_resampling_to_t2w_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_resampling_fixture(root)
            manifest = write_manifest(root)
            report_path = root / "outputs" / "reports" / "resampling.json"

            report = validate_resampling_plan(
                manifest_path=manifest,
                raw_root=raw_root,
                report_path=report_path,
                sample_size=1,
            )

            self.assertTrue(report_path.exists())
            self.assertEqual(report["summary"]["cases_checked"], 1)
            self.assertEqual(report["summary"]["cases_with_issues"], 0)
            self.assertEqual(report["summary"]["adc_resampled_matches_reference"], 1)
            self.assertEqual(report["summary"]["hbv_resampled_matches_reference"], 1)
            self.assertEqual(report["summary"]["gland_cases_with_reference_grid_mask"], 1)
            self.assertEqual(report["summary"]["lesion_cases_with_reference_grid_mask"], 1)
            self.assertFalse((root / "data" / "processed").exists())

    def test_writes_tiny_preprocessed_sample_outside_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_resampling_fixture(root)
            manifest = write_manifest(root)
            output_root = root / "data" / "processed" / "picai_sample"
            report_path = root / "outputs" / "reports" / "preprocessed_sample.json"

            report = write_preprocessed_sample(
                manifest_path=manifest,
                raw_root=raw_root,
                output_root=output_root,
                report_path=report_path,
                sample_size=1,
            )

            self.assertTrue(report_path.exists())
            self.assertEqual(report["summary"]["cases_with_issues"], 0)
            self.assertEqual(report["summary"]["adc_written"], 1)
            self.assertEqual(report["summary"]["hbv_written"], 1)
            self.assertEqual(report["summary"]["adc_matches_reference"], 1)
            self.assertEqual(report["summary"]["hbv_matches_reference"], 1)
            case_dir = output_root / "10000_1000000"
            self.assertTrue((case_dir / "10000_1000000_adc_to_t2w.mha").exists())
            self.assertTrue((case_dir / "10000_1000000_hbv_to_t2w.mha").exists())

    def test_refuses_to_write_preprocessed_sample_inside_raw_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_resampling_fixture(root)
            manifest = write_manifest(root)

            with self.assertRaises(ValueError):
                write_preprocessed_sample(
                    manifest_path=manifest,
                    raw_root=raw_root,
                    output_root=raw_root / "processed",
                    report_path=root / "outputs" / "reports" / "preprocessed_sample.json",
                    sample_size=1,
                )


def create_resampling_fixture(root: Path) -> Path:
    raw_root = root / "data" / "raw" / "picai"
    case_id = "10000_1000000"
    write_image(raw_root / "images" / "fold0" / case_id / f"{case_id}_t2w.mha", size=(8, 8, 4), spacing=(0.5, 0.5, 2.0))
    write_image(raw_root / "images" / "fold0" / case_id / f"{case_id}_adc.mha", size=(4, 4, 4), spacing=(1.0, 1.0, 2.0))
    write_image(raw_root / "images" / "fold0" / case_id / f"{case_id}_hbv.mha", size=(4, 4, 4), spacing=(1.0, 1.0, 2.0))
    write_mask(raw_root / "picai_labels" / "anatomical_delineations" / f"{case_id}.nii.gz", size=(8, 8, 4), spacing=(0.5, 0.5, 2.0))
    write_mask(raw_root / "picai_labels" / "csPCa_lesion_delineations" / f"{case_id}.nii.gz", size=(8, 8, 4), spacing=(0.5, 0.5, 2.0))
    return raw_root


def write_manifest(root: Path) -> Path:
    manifest = root / "data" / "interim" / "picai_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "case_id",
                "fold",
                "path_t2w",
                "path_adc",
                "path_hbv",
                "path_gland_mask",
                "path_lesion_mask",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "10000_1000000",
                "fold": "fold0",
                "path_t2w": "images/fold0/10000_1000000/10000_1000000_t2w.mha",
                "path_adc": "images/fold0/10000_1000000/10000_1000000_adc.mha",
                "path_hbv": "images/fold0/10000_1000000/10000_1000000_hbv.mha",
                "path_gland_mask": "picai_labels/anatomical_delineations/10000_1000000.nii.gz",
                "path_lesion_mask": "picai_labels/csPCa_lesion_delineations/10000_1000000.nii.gz",
            }
        )
    return manifest


def write_image(path: Path, size: tuple[int, int, int], spacing: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.Image(size, sitk.sitkUInt16)
    image.SetSpacing(spacing)
    image += 10
    sitk.WriteImage(image, str(path))


def write_mask(path: Path, size: tuple[int, int, int], spacing: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.Image(size, sitk.sitkUInt8)
    image.SetSpacing(spacing)
    image[1, 1, 1] = 1
    sitk.WriteImage(image, str(path))


if __name__ == "__main__":
    unittest.main()
