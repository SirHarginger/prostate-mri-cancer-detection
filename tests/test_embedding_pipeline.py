from __future__ import annotations

import csv
import json
import struct
import tempfile
import unittest
from pathlib import Path

from prostate_mri_cancer_detection.modeling import extract_embedding_table


class EmbeddingPipelineTests(unittest.TestCase):
    def test_extracts_split_safe_embeddings_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_embedding_fixture(root)
            manifest = write_manifest(root)
            preprocessing_report = write_preprocessing_report(root)
            output = root / "data" / "features" / "embeddings.csv"
            provenance = root / "outputs" / "reports" / "provenance.json"
            report = root / "outputs" / "reports" / "report.json"

            summary = extract_embedding_table(
                manifest_path=manifest,
                raw_root=raw_root,
                output_path=output,
                provenance_path=provenance,
                report_path=report,
                preprocessing_report_path=preprocessing_report,
                embedding_dim=4,
                sample_size_per_split=1,
            )

            rows = read_csv_rows(output)
            self.assertEqual(summary["embeddings_written"], 3)
            self.assertEqual(summary["failures"], 0)
            self.assertEqual(summary["embeddings_by_split"], {"test": 1, "train": 1, "validation": 1})
            self.assertEqual(summary["validation_or_test_augmented_rows"], 0)
            self.assertEqual({row["augmentation_applied"] for row in rows}, {"False"})
            self.assertEqual({row["encoder_type"] for row in rows}, {"prototype_not_trained_cnn"})
            self.assertIn("embedding_003", rows[0])

            provenance_payload = json.loads(provenance.read_text(encoding="utf-8"))
            self.assertEqual(provenance_payload["training_status"], "not_trained")
            self.assertEqual(
                provenance_payload["paired_transform_policy"]["adc"],
                "deferred until resampling/alignment to T2W grid is implemented",
            )

    def test_train_augmentation_never_marks_validation_or_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_embedding_fixture(root)
            manifest = write_manifest(root)
            output = root / "data" / "features" / "embeddings.csv"
            provenance = root / "outputs" / "reports" / "provenance.json"
            report = root / "outputs" / "reports" / "report.json"

            summary = extract_embedding_table(
                manifest_path=manifest,
                raw_root=raw_root,
                output_path=output,
                provenance_path=provenance,
                report_path=report,
                embedding_dim=4,
                sample_size_per_split=1,
                augment_train=True,
            )

            rows_by_split = {row["split"]: row for row in read_csv_rows(output)}
            self.assertEqual(rows_by_split["train"]["augmentation_applied"], "True")
            self.assertEqual(rows_by_split["validation"]["augmentation_applied"], "False")
            self.assertEqual(rows_by_split["test"]["augmentation_applied"], "False")
            self.assertEqual(summary["validation_or_test_augmented_rows"], 0)

    def test_requested_case_ids_preserve_manifest_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_embedding_fixture(root)
            manifest = write_manifest(root)
            output = root / "data" / "features" / "embeddings.csv"
            provenance = root / "outputs" / "reports" / "provenance.json"
            report = root / "outputs" / "reports" / "report.json"

            summary = extract_embedding_table(
                manifest_path=manifest,
                raw_root=raw_root,
                output_path=output,
                provenance_path=provenance,
                report_path=report,
                embedding_dim=4,
                case_ids=["10002_1000002"],
            )

            rows = read_csv_rows(output)
            self.assertEqual(summary["embeddings_written"], 1)
            self.assertEqual(rows[0]["case_id"], "10002_1000002")
            self.assertEqual(rows[0]["split"], "test")


def create_embedding_fixture(root: Path) -> Path:
    raw_root = root / "data" / "raw" / "picai"
    for case_id, fold, offset in [
        ("10000_1000000", "fold0", 0),
        ("10001_1000001", "fold3", 10),
        ("10002_1000002", "fold4", 20),
    ]:
        write_mha_volume(
            raw_root / "images" / fold / case_id / f"{case_id}_t2w.mha",
            values=[offset + value for value in range(1, 9)],
        )
    return raw_root


def write_manifest(root: Path) -> Path:
    manifest = root / "data" / "interim" / "picai_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["case_id", "fold", "label_cspca", "path_t2w"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "10000_1000000",
                "fold": "fold0",
                "label_cspca": "NO",
                "path_t2w": "images/fold0/10000_1000000/10000_1000000_t2w.mha",
            }
        )
        writer.writerow(
            {
                "case_id": "10001_1000001",
                "fold": "fold3",
                "label_cspca": "YES",
                "path_t2w": "images/fold3/10001_1000001/10001_1000001_t2w.mha",
            }
        )
        writer.writerow(
            {
                "case_id": "10002_1000002",
                "fold": "fold4",
                "label_cspca": "NO",
                "path_t2w": "images/fold4/10002_1000002/10002_1000002_t2w.mha",
            }
        )
    return manifest


def write_preprocessing_report(root: Path) -> Path:
    report = root / "outputs" / "reports" / "preprocessing.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"selected_case_ids": ["10000_1000000", "10001_1000001", "10002_1000002"]}),
        encoding="utf-8",
    )
    return report


def write_mha_volume(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            "DimSize = 2 2 2",
            "ElementSpacing = 1 1 1",
            "TransformMatrix = 1 0 0 0 1 0 0 0 1",
            "Offset = 0 0 0",
            "ElementType = MET_USHORT",
            "ElementDataFile = LOCAL",
        ]
    )
    payload = struct.pack("<" + "H" * len(values), *values)
    path.write_bytes(header.encode("latin-1") + b"\n" + payload)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


if __name__ == "__main__":
    unittest.main()
