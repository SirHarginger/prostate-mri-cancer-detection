from __future__ import annotations

import csv
import gzip
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from prostate_mri_cancer_detection.features import extract_radiomics_features
from prostate_mri_cancer_detection.features import read_volume_data


class RadiomicsFeatureTests(unittest.TestCase):
    def test_extracts_first_order_features_for_valid_t2w_roi(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_radiomics_fixture(root)
            manifest = write_manifest(root)
            preprocessing_report = write_preprocessing_report(root, raw_root)
            output = root / "data" / "features" / "radiomics.csv"
            failures = root / "outputs" / "reports" / "failures.csv"
            settings = root / "outputs" / "reports" / "settings.json"

            summary = extract_radiomics_features(
                manifest_path=manifest,
                raw_root=raw_root,
                preprocessing_report_path=preprocessing_report,
                output_path=output,
                failure_log_path=failures,
                settings_path=settings,
                roi="lesion",
            )

            self.assertEqual(summary["features_written"], 1)
            self.assertEqual(summary["failures_written"], 0)
            row = read_csv_rows(output)[0]
            self.assertEqual(row["case_id"], "10000_1000000")
            self.assertEqual(row["sequence"], "t2w")
            self.assertEqual(row["roi"], "lesion")
            self.assertEqual(row["voxel_count"], "3")
            self.assertEqual(row["intensity_min"], "2")
            self.assertEqual(row["intensity_max"], "8")
            self.assertEqual(row["intensity_mean"], "5")
            self.assertEqual(read_csv_rows(failures), [])

            settings_payload = json.loads(settings.read_text(encoding="utf-8"))
            self.assertEqual(settings_payload["image_source"], "original manifest image; no augmentation")
            self.assertEqual(settings_payload["sequence"], "t2w")
            self.assertIn("T2W-only until ADC/HBV resampling is implemented", settings_payload["limitations"])

    def test_empty_mask_is_logged_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_radiomics_fixture(root, mask_values=[0] * 8)
            manifest = write_manifest(root)
            preprocessing_report = write_preprocessing_report(root, raw_root)
            output = root / "data" / "features" / "radiomics.csv"
            failures = root / "outputs" / "reports" / "failures.csv"
            settings = root / "outputs" / "reports" / "settings.json"

            summary = extract_radiomics_features(
                manifest_path=manifest,
                raw_root=raw_root,
                preprocessing_report_path=preprocessing_report,
                output_path=output,
                failure_log_path=failures,
                settings_path=settings,
                roi="lesion",
            )

            self.assertEqual(summary["features_written"], 0)
            self.assertEqual(summary["failures_written"], 1)
            failure = read_csv_rows(failures)[0]
            self.assertIn("empty mask", failure["reason"])

    def test_shape_mismatch_is_logged_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_radiomics_fixture(root, mask_shape=(2, 2, 1), mask_values=[1, 0, 0, 0])
            manifest = write_manifest(root)
            preprocessing_report = write_preprocessing_report(root, raw_root)
            output = root / "data" / "features" / "radiomics.csv"
            failures = root / "outputs" / "reports" / "failures.csv"
            settings = root / "outputs" / "reports" / "settings.json"

            summary = extract_radiomics_features(
                manifest_path=manifest,
                raw_root=raw_root,
                preprocessing_report_path=preprocessing_report,
                output_path=output,
                failure_log_path=failures,
                settings_path=settings,
                roi="lesion",
            )

            self.assertEqual(summary["features_written"], 0)
            self.assertEqual(summary["failures_written"], 1)
            self.assertIn("shape mismatch", read_csv_rows(failures)[0]["reason"])

    def test_reads_compressed_metaimage_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "compressed.mha"
            write_mha_volume(
                image_path,
                shape=(2, 2, 1),
                spacing=(1.0, 1.0, 1.0),
                values=[1, 2, 3, 4],
                compressed=True,
            )

            volume = read_volume_data(image_path)

            self.assertEqual(volume.shape, [2, 2, 1])
            self.assertEqual(list(volume.values), [1, 2, 3, 4])


def create_radiomics_fixture(
    root: Path,
    mask_shape: tuple[int, int, int] = (2, 2, 2),
    mask_values: list[int] | None = None,
) -> Path:
    raw_root = root / "data" / "raw" / "picai"
    image_path = raw_root / "images" / "fold0" / "10000_1000000_t2w.mha"
    mask_path = raw_root / "picai_labels" / "csPCa_lesion_delineations" / "10000_1000000.nii.gz"
    write_mha_volume(
        image_path,
        shape=(2, 2, 2),
        spacing=(1.0, 1.0, 2.0),
        values=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    write_nifti_volume(
        mask_path,
        shape=mask_shape,
        spacing=(1.0, 1.0, 2.0),
        values=mask_values if mask_values is not None else [0, 1, 0, 0, 1, 0, 0, 1],
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
                "path_t2w",
                "path_gland_mask",
                "path_lesion_mask",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "10000_1000000",
                "fold": "fold0",
                "path_t2w": "images/fold0/10000_1000000_t2w.mha",
                "path_gland_mask": "",
                "path_lesion_mask": "picai_labels/csPCa_lesion_delineations/10000_1000000.nii.gz",
            }
        )
    return manifest


def write_preprocessing_report(root: Path, raw_root: Path) -> Path:
    report = root / "outputs" / "reports" / "preprocessing.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "selected_case_ids": ["10000_1000000"],
                "cases": [
                    {
                        "case_id": "10000_1000000",
                        "masks": {
                            "lesion": [
                                {
                                    "path": str(
                                        raw_root
                                        / "picai_labels"
                                        / "csPCa_lesion_delineations"
                                        / "10000_1000000.nii.gz"
                                    ),
                                    "alignment_to_t2w": "t2w_compatible",
                                }
                            ],
                            "gland": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return report


def write_mha_volume(
    path: Path,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    values: list[int],
    compressed: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            f"DimSize = {shape[0]} {shape[1]} {shape[2]}",
            f"ElementSpacing = {spacing[0]} {spacing[1]} {spacing[2]}",
            "TransformMatrix = 1 0 0 0 1 0 0 0 1",
            "Offset = 0 0 0",
            "ElementType = MET_USHORT",
            f"CompressedData = {str(compressed)}",
            "ElementDataFile = LOCAL",
        ]
    )
    payload = struct.pack("<" + "H" * len(values), *values)
    if compressed:
        payload = zlib.compress(payload)
    path.write_bytes(header.encode("latin-1") + b"\n" + payload)


def write_nifti_volume(
    path: Path,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    values: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, shape[0], shape[1], shape[2], 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 2)
    struct.pack_into("<h", header, 72, 8)
    struct.pack_into("<8f", header, 76, 0.0, spacing[0], spacing[1], spacing[2], 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<h", header, 254, 1)
    struct.pack_into("<4f", header, 280, spacing[0], 0.0, 0.0, 0.0)
    struct.pack_into("<4f", header, 296, 0.0, spacing[1], 0.0, 0.0)
    struct.pack_into("<4f", header, 312, 0.0, 0.0, spacing[2], 0.0)
    header[344:348] = b"n+1\0"
    payload = bytes(values)
    with gzip.open(path, "wb") as file_obj:
        file_obj.write(header + payload)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


if __name__ == "__main__":
    unittest.main()
