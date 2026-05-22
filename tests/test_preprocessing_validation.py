from __future__ import annotations

import csv
import gzip
import struct
import tempfile
import unittest
from pathlib import Path

from prostate_mri_cancer_detection.preprocessing import (
    read_image_metadata,
    validate_preprocessing_inputs,
)


class PreprocessingValidationTests(unittest.TestCase):
    def test_reads_metaimage_header_without_voxel_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "10000_1000000_t2w.mha"
            write_mha_header(image_path, shape=(4, 5, 6), spacing=(0.5, 0.5, 3.0))

            metadata = read_image_metadata(image_path)

            self.assertTrue(metadata.readable)
            self.assertEqual(metadata.format, "metaimage")
            self.assertEqual(metadata.ndim, 3)
            self.assertEqual(metadata.shape, [4, 5, 6])
            self.assertEqual(metadata.spacing, [0.5, 0.5, 3.0])

    def test_reads_nifti_header_without_external_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = Path(tmpdir) / "10000_1000000.nii.gz"
            write_nifti_header(mask_path, shape=(4, 5, 6), spacing=(0.5, 0.5, 3.0))

            metadata = read_image_metadata(mask_path)

            self.assertTrue(metadata.readable)
            self.assertEqual(metadata.format, "nifti1")
            self.assertEqual(metadata.shape, [4, 5, 6])
            self.assertEqual(metadata.spacing, [0.5, 0.5, 3.0])

    def test_validates_manifest_sample_and_reports_plans(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_preprocessing_fixture(root)
            manifest_path = root / "data" / "interim" / "picai_manifest.csv"
            report_path = root / "outputs" / "reports" / "preprocessing.json"
            write_manifest(manifest_path)

            report = validate_preprocessing_inputs(
                manifest_path=manifest_path,
                raw_root=raw_root,
                report_path=report_path,
                sample_size=1,
            )

            self.assertTrue(report_path.exists())
            self.assertEqual(report["summary"]["cases_checked"], 1)
            self.assertEqual(report["summary"]["cases_with_issues"], 0)
            self.assertEqual(report["summary"]["cases_with_blocking_issues"], 0)
            self.assertEqual(report["summary"]["cases_requiring_resampling"], 0)
            self.assertEqual(
                report["summary"]["modality_headers_readable"],
                {"t2w": 1, "adc": 1, "hbv": 1},
            )
            self.assertEqual(report["summary"]["mask_headers_readable"], {"gland": 1, "lesion": 1})
            self.assertEqual(report["summary"]["mask_t2w_compatible_cases"], {"gland": 1, "lesion": 1})
            self.assertEqual(report["normalization_plan"]["stage2_status"], "planned_not_applied")
            self.assertEqual(
                report["roi_plan"]["stage2_status"],
                "validate_paths_and_header_alignment_only",
            )

    def test_reports_shape_mismatch_without_writing_processed_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_preprocessing_fixture(root, adc_shape=(8, 5, 6))
            manifest_path = root / "data" / "interim" / "picai_manifest.csv"
            write_manifest(manifest_path)

            report = validate_preprocessing_inputs(
                manifest_path=manifest_path,
                raw_root=raw_root,
                sample_size=1,
            )

            self.assertEqual(report["summary"]["cases_with_blocking_issues"], 0)
            self.assertEqual(report["summary"]["cases_requiring_resampling"], 1)
            self.assertEqual(
                report["summary"]["resampling_required_counts"]["adc_to_t2w_grid:shape"],
                1,
            )
            self.assertFalse((root / "data" / "processed").exists())

    def test_accepts_one_t2w_compatible_mask_among_alternate_grids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_preprocessing_fixture(root)
            write_nifti_header(
                raw_root / "picai_labels" / "anatomical_delineations" / "10000_1000000_adc_grid.nii.gz",
                shape=(2, 3, 6),
                spacing=(2.0, 2.0, 3.0),
            )
            manifest_path = root / "data" / "interim" / "picai_manifest.csv"
            write_manifest(
                manifest_path,
                gland_mask=(
                    "picai_labels/anatomical_delineations/10000_1000000.nii.gz|"
                    "picai_labels/anatomical_delineations/10000_1000000_adc_grid.nii.gz"
                ),
            )

            report = validate_preprocessing_inputs(
                manifest_path=manifest_path,
                raw_root=raw_root,
                sample_size=1,
            )

            self.assertEqual(report["summary"]["cases_with_blocking_issues"], 0)
            gland_masks = report["cases"][0]["masks"]["gland"]
            self.assertEqual(
                [mask["alignment_to_t2w"] for mask in gland_masks],
                ["t2w_compatible", "different_grid"],
            )


def create_preprocessing_fixture(
    root: Path,
    adc_shape: tuple[int, int, int] = (4, 5, 6),
) -> Path:
    raw_root = root / "data" / "raw" / "picai"
    image_dir = raw_root / "images" / "fold0"
    mask_dir = raw_root / "picai_labels"

    write_mha_header(image_dir / "10000_1000000_t2w.mha", shape=(4, 5, 6))
    write_mha_header(image_dir / "10000_1000000_adc.mha", shape=adc_shape)
    write_mha_header(image_dir / "10000_1000000_hbv.mha", shape=(4, 5, 6))
    write_nifti_header(
        mask_dir / "anatomical_delineations" / "10000_1000000.nii.gz",
        shape=(4, 5, 6),
    )
    write_nifti_header(
        mask_dir / "csPCa_lesion_delineations" / "10000_1000000.nii.gz",
        shape=(4, 5, 6),
    )
    return raw_root


def write_manifest(
    path: Path,
    gland_mask: str = "picai_labels/anatomical_delineations/10000_1000000.nii.gz",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
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
                "path_t2w": "images/fold0/10000_1000000_t2w.mha",
                "path_adc": "images/fold0/10000_1000000_adc.mha",
                "path_hbv": "images/fold0/10000_1000000_hbv.mha",
                "path_gland_mask": gland_mask,
                "path_lesion_mask": "picai_labels/csPCa_lesion_delineations/10000_1000000.nii.gz",
            }
        )


def write_mha_header(
    path: Path,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float] = (0.5, 0.5, 3.0),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ObjectType = Image",
                "NDims = 3",
                f"DimSize = {shape[0]} {shape[1]} {shape[2]}",
                f"ElementSpacing = {spacing[0]} {spacing[1]} {spacing[2]}",
                "TransformMatrix = 1 0 0 0 1 0 0 0 1",
                "Offset = 0 0 0",
                "ElementType = MET_SHORT",
                "ElementDataFile = LOCAL",
            ]
        )
        + "\n",
        encoding="latin-1",
    )


def write_nifti_header(
    path: Path,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float] = (0.5, 0.5, 3.0),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, shape[0], shape[1], shape[2], 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 4)
    struct.pack_into("<h", header, 72, 16)
    struct.pack_into("<8f", header, 76, 0.0, spacing[0], spacing[1], spacing[2], 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<h", header, 254, 1)
    struct.pack_into("<4f", header, 280, spacing[0], 0.0, 0.0, 0.0)
    struct.pack_into("<4f", header, 296, 0.0, spacing[1], 0.0, 0.0)
    struct.pack_into("<4f", header, 312, 0.0, 0.0, spacing[2], 0.0)
    header[344:348] = b"n+1\0"

    with gzip.open(path, "wb") as file_obj:
        file_obj.write(header)


if __name__ == "__main__":
    unittest.main()
