from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from prostate_mri_cancer_detection.explainability import generate_explainability_report


class ExplainabilityReportTests(unittest.TestCase):
    def test_generates_centroid_feature_importance_and_guardrails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = write_manifest(root)
            radiomics = write_radiomics(root)
            embeddings = write_embeddings(root)
            json_report = root / "outputs" / "reports" / "explainability.json"
            csv_report = root / "outputs" / "reports" / "importance.csv"

            report = generate_explainability_report(
                manifest_path=manifest,
                radiomics_path=radiomics,
                embeddings_path=embeddings,
                output_json_path=json_report,
                output_csv_path=csv_report,
                top_n=3,
            )

            self.assertEqual(report["aligned_cases"], 6)
            self.assertEqual(report["cnn_visual_explanation"]["status"], "not_available")
            self.assertEqual(report["importances"]["radiomics_only"]["status"], "ok")
            self.assertEqual(len(report["importances"]["radiomics_only"]["top_features"]), 2)
            self.assertTrue(json_report.exists())
            csv_rows = read_csv_rows(csv_report)
            self.assertEqual({row["baseline"] for row in csv_rows}, set(report["importances"]))
            self.assertTrue(all(int(row["rank"]) <= 3 for row in csv_rows))
            payload = json.loads(json_report.read_text(encoding="utf-8"))
            self.assertIn("Feature importance is a prototype", payload["claim_limits"][0])

    def test_importance_unavailable_when_train_split_lacks_both_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = write_manifest(root, train_labels=("NO", "NO", "NO", "NO"))
            radiomics = write_radiomics(root)
            embeddings = write_embeddings(root)
            json_report = root / "outputs" / "reports" / "explainability.json"
            csv_report = root / "outputs" / "reports" / "importance.csv"

            report = generate_explainability_report(
                manifest_path=manifest,
                radiomics_path=radiomics,
                embeddings_path=embeddings,
                output_json_path=json_report,
                output_csv_path=csv_report,
            )

            self.assertEqual(report["importances"]["radiomics_only"]["status"], "unavailable")
            self.assertEqual(read_csv_rows(csv_report), [])


def write_manifest(root: Path, train_labels: tuple[str, str, str, str] = ("NO", "YES", "NO", "YES")) -> Path:
    path = root / "data" / "interim" / "picai_manifest.csv"
    rows = [
        ("10000_1000000", "fold0", train_labels[0]),
        ("10001_1000001", "fold0", train_labels[1]),
        ("10002_1000002", "fold1", train_labels[2]),
        ("10003_1000003", "fold2", train_labels[3]),
        ("10004_1000004", "fold3", "NO"),
        ("10005_1000005", "fold4", "YES"),
    ]
    write_csv(path, ["case_id", "fold", "label_cspca"], rows)
    return path


def write_radiomics(root: Path) -> Path:
    path = root / "data" / "features" / "radiomics.csv"
    rows = [
        ("10000_1000000", "fold0", "gland", "10", "1.0"),
        ("10001_1000001", "fold0", "gland", "20", "2.0"),
        ("10002_1000002", "fold1", "gland", "11", "1.1"),
        ("10003_1000003", "fold2", "gland", "21", "2.1"),
        ("10004_1000004", "fold3", "gland", "12", "1.2"),
        ("10005_1000005", "fold4", "gland", "22", "2.2"),
    ]
    write_csv(path, ["case_id", "fold", "roi", "voxel_count", "intensity_mean"], rows)
    return path


def write_embeddings(root: Path) -> Path:
    path = root / "data" / "features" / "embeddings.csv"
    rows = [
        ("10000_1000000", "fold0", "train", "NO", "0.1", "0.2"),
        ("10001_1000001", "fold0", "train", "YES", "0.8", "0.7"),
        ("10002_1000002", "fold1", "train", "NO", "0.2", "0.1"),
        ("10003_1000003", "fold2", "train", "YES", "0.9", "0.8"),
        ("10004_1000004", "fold3", "validation", "NO", "0.3", "0.2"),
        ("10005_1000005", "fold4", "test", "YES", "1.0", "0.9"),
    ]
    write_csv(
        path,
        ["case_id", "fold", "split", "label_cspca", "embedding_000", "embedding_001"],
        rows,
    )
    return path


def write_csv(path: Path, fieldnames: list[str], rows: list[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


if __name__ == "__main__":
    unittest.main()
