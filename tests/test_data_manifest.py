from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prostate_mri_cancer_detection.data import (
    build_and_write_manifest,
    build_picai_manifest,
    parse_case_id_from_label,
    parse_image_filename,
)


class PicaiManifestTests(unittest.TestCase):
    def test_parse_image_filename_uses_exact_modality_suffix(self) -> None:
        parsed = parse_image_filename("10000_1000000_t2w.mha")

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.case_id, "10000_1000000")
        self.assertEqual(parsed.patient_id, "10000")
        self.assertEqual(parsed.study_id, "1000000")
        self.assertEqual(parsed.modality, "t2w")
        self.assertIsNone(parse_image_filename("10000_1000000_t2w_extra.mha"))

    def test_parse_label_case_id_from_nested_path(self) -> None:
        case_id = parse_case_id_from_label(
            "picai_labels/csPCa_lesion_delineations/human/10000_1000000.nii.gz"
        )

        self.assertEqual(case_id, "10000_1000000")

    def test_build_manifest_links_images_clinical_rows_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = create_picai_fixture(Path(tmpdir))

            result = build_picai_manifest(raw_root)
            rows_by_case = {row["case_id"]: row for row in result.rows}

            self.assertEqual(set(rows_by_case), {"10000_1000000", "10001_1000001"})
            complete_case = rows_by_case["10000_1000000"]
            self.assertEqual(complete_case["fold"], "fold0")
            self.assertEqual(complete_case["path_t2w"], "images/fold0/10000_1000000_t2w.mha")
            self.assertEqual(complete_case["path_adc"], "images/fold0/10000_1000000_adc.mha")
            self.assertEqual(complete_case["path_hbv"], "images/fold0/10000_1000000_hbv.mha")
            self.assertEqual(complete_case["available_sequences"], "t2w|adc|hbv")
            self.assertEqual(complete_case["clinical_row_found"], "True")
            self.assertEqual(complete_case["label_cspca"], "1")
            self.assertEqual(complete_case["pirads_score"], "4")
            self.assertEqual(complete_case["has_gland_mask"], "True")
            self.assertEqual(complete_case["has_lesion_mask"], "True")
            self.assertEqual(complete_case["missing_data_flags"], "")

            partial_case = rows_by_case["10001_1000001"]
            self.assertEqual(partial_case["fold"], "fold1")
            self.assertIn("missing_hbv", partial_case["missing_data_flags"])
            self.assertIn("missing_gland_mask", partial_case["missing_data_flags"])
            self.assertIn("missing_lesion_mask", partial_case["missing_data_flags"])

            report = result.report
            self.assertEqual(report["total_cases"], 2)
            self.assertEqual(report["cases_by_fold"], {"fold0": 1, "fold1": 1})
            self.assertEqual(report["modality_available_counts"], {"t2w": 2, "adc": 2, "hbv": 1})
            self.assertEqual(report["clinical_rows_linked"], 2)
            self.assertEqual(report["gland_mask_cases_linked"], 1)
            self.assertEqual(report["lesion_mask_cases_linked"], 1)
            self.assertEqual(report["missing_data_counts"]["missing_hbv"], 1)

    def test_duplicate_modality_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = create_picai_fixture(Path(tmpdir))
            touch(raw_root / "images" / "fold0" / "duplicate" / "10000_1000000_t2w.mha")

            result = build_picai_manifest(raw_root)
            row = next(row for row in result.rows if row["case_id"] == "10000_1000000")

            self.assertIn("duplicate_t2w", row["missing_data_flags"])
            self.assertEqual(result.report["duplicate_image_modalities_count"], 1)

    def test_build_and_write_manifest_creates_interim_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_picai_fixture(root)
            output = root / "data" / "interim" / "picai_manifest.csv"
            report = root / "data" / "interim" / "picai_manifest_validation.json"

            build_and_write_manifest(raw_root, output, report)

            self.assertTrue(output.exists())
            self.assertTrue(report.exists())
            self.assertIn("case_id", output.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn('"total_cases": 2', report.read_text(encoding="utf-8"))


def create_picai_fixture(root: Path) -> Path:
    raw_root = root / "data" / "raw" / "picai"

    for modality in ("t2w", "adc", "hbv"):
        touch(raw_root / "images" / "fold0" / f"10000_1000000_{modality}.mha")
    for modality in ("t2w", "adc"):
        touch(raw_root / "images" / "fold1" / f"10001_1000001_{modality}.mha")

    clinical_dir = raw_root / "picai_labels" / "clinical_information"
    clinical_dir.mkdir(parents=True, exist_ok=True)
    (clinical_dir / "marksheet.csv").write_text(
        "\n".join(
            [
                "patient_id,study_id,case_csPCa,PI-RADS",
                "10000,1000000,1,4",
                "10001,1000001,0,2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    touch(
        raw_root
        / "picai_labels"
        / "anatomical_delineations"
        / "human_expert"
        / "10000_1000000.nii.gz"
    )
    touch(
        raw_root
        / "picai_labels"
        / "csPCa_lesion_delineations"
        / "human_expert"
        / "10000_1000000.nii.gz"
    )
    return raw_root


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


if __name__ == "__main__":
    unittest.main()
