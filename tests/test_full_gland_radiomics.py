from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from prostate_mri_cancer_detection.features import extract_full_gland_multisequence_radiomics


try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - depends on local environment.
    sitk = None


@unittest.skipIf(sitk is None, "SimpleITK is not installed")
class FullGlandRadiomicsTests(unittest.TestCase):
    def test_extracts_t2w_adc_hbv_whole_gland_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_fixture(root)
            manifest = write_manifest(root)
            output = root / "data" / "features" / "radiomics_full.csv"
            failures = root / "outputs" / "reports" / "failures.csv"
            settings = root / "outputs" / "reports" / "settings.json"

            summary = extract_full_gland_multisequence_radiomics(
                manifest_path=manifest,
                raw_root=raw_root,
                output_path=output,
                failure_log_path=failures,
                settings_path=settings,
                all_cases=True,
            )

            self.assertEqual(summary["features_written"], 1)
            self.assertEqual(summary["failures_written"], 0)
            row = read_csv_rows(output)[0]
            self.assertEqual(row["case_id"], "10000_1000000")
            self.assertEqual(row["label_cspca"], "YES")
            self.assertIn("t2w_intensity_mean", row)
            self.assertIn("adc_intensity_mean", row)
            self.assertIn("hbv_intensity_mean", row)
            self.assertEqual(read_csv_rows(failures), [])
            self.assertTrue(settings.exists())

    def test_logs_missing_non_empty_gland_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_fixture(root, mask_value=0)
            manifest = write_manifest(root)
            output = root / "data" / "features" / "radiomics_full.csv"
            failures = root / "outputs" / "reports" / "failures.csv"
            settings = root / "outputs" / "reports" / "settings.json"

            summary = extract_full_gland_multisequence_radiomics(
                manifest_path=manifest,
                raw_root=raw_root,
                output_path=output,
                failure_log_path=failures,
                settings_path=settings,
                all_cases=True,
            )

            self.assertEqual(summary["features_written"], 0)
            self.assertEqual(summary["failures_written"], 1)
            self.assertIn("no non-empty T2W-grid gland mask", read_csv_rows(failures)[0]["reason"])


def create_fixture(root: Path, mask_value: int = 1) -> Path:
    raw_root = root / "data" / "raw" / "picai"
    case_id = "10000_1000000"
    case_dir = raw_root / "images" / "fold0" / case_id
    write_image(case_dir / f"{case_id}_t2w.mha", size=(8, 8, 4), spacing=(0.5, 0.5, 2.0), value=10)
    write_image(case_dir / f"{case_id}_adc.mha", size=(4, 4, 4), spacing=(1.0, 1.0, 2.0), value=20)
    write_image(case_dir / f"{case_id}_hbv.mha", size=(4, 4, 4), spacing=(1.0, 1.0, 2.0), value=30)
    write_mask(
        raw_root / "picai_labels" / "anatomical_delineations" / f"{case_id}.nii.gz",
        size=(8, 8, 4),
        spacing=(0.5, 0.5, 2.0),
        value=mask_value,
    )
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
                "label_cspca",
                "path_t2w",
                "path_adc",
                "path_hbv",
                "path_gland_mask",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "10000_1000000",
                "fold": "fold0",
                "label_cspca": "YES",
                "path_t2w": "images/fold0/10000_1000000/10000_1000000_t2w.mha",
                "path_adc": "images/fold0/10000_1000000/10000_1000000_adc.mha",
                "path_hbv": "images/fold0/10000_1000000/10000_1000000_hbv.mha",
                "path_gland_mask": "picai_labels/anatomical_delineations/10000_1000000.nii.gz",
            }
        )
    return manifest


def write_image(path: Path, size: tuple[int, int, int], spacing: tuple[float, float, float], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.Image(size, sitk.sitkUInt16)
    image.SetSpacing(spacing)
    image += value
    sitk.WriteImage(image, str(path))


def write_mask(path: Path, size: tuple[int, int, int], spacing: tuple[float, float, float], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.Image(size, sitk.sitkUInt8)
    image.SetSpacing(spacing)
    if value:
        for z in range(1, 3):
            for y in range(1, 7):
                for x in range(1, 7):
                    image[x, y, z] = value
    sitk.WriteImage(image, str(path))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


if __name__ == "__main__":
    unittest.main()
