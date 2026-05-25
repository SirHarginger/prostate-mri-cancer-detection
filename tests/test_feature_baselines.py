from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from prostate_mri_cancer_detection.evaluation import run_feature_baselines


class FeatureBaselineTests(unittest.TestCase):
    def test_runs_aligned_prototype_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = write_manifest(root)
            radiomics = write_radiomics(root)
            embeddings = write_embeddings(root)
            metrics = root / "outputs" / "reports" / "metrics.json"
            predictions = root / "outputs" / "reports" / "predictions.csv"
            report_path = root / "outputs" / "reports" / "report.json"

            report = run_feature_baselines(
                manifest_path=manifest,
                radiomics_path=radiomics,
                embeddings_path=embeddings,
                metrics_path=metrics,
                predictions_path=predictions,
                report_path=report_path,
            )

            self.assertEqual(report["case_counts"]["aligned"], 6)
            self.assertEqual(report["split_counts"], {"train": 4, "validation": 1, "test": 1})
            self.assertEqual(report["metrics"]["radiomics_only"]["status"], "ok")
            self.assertEqual(report["metrics"]["prototype_embedding_only"]["status"], "ok")
            self.assertEqual(report["metrics"]["hybrid_radiomics_embedding"]["status"], "ok")
            self.assertNotIn("label_cspca", report["feature_columns"]["radiomics_only"])
            self.assertNotIn("fold", report["feature_columns"]["prototype_embedding_only"])
            prediction_rows = read_csv_rows(predictions)
            self.assertEqual(len(prediction_rows), 18)
            self.assertTrue(metrics.exists())
            self.assertTrue(report_path.exists())

    def test_fails_cleanly_when_train_split_has_one_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = write_manifest(root, train_labels=("NO", "NO", "NO", "NO"))
            radiomics = write_radiomics(root)
            embeddings = write_embeddings(root)
            metrics = root / "outputs" / "reports" / "metrics.json"
            predictions = root / "outputs" / "reports" / "predictions.csv"
            report_path = root / "outputs" / "reports" / "report.json"

            report = run_feature_baselines(
                manifest_path=manifest,
                radiomics_path=radiomics,
                embeddings_path=embeddings,
                metrics_path=metrics,
                predictions_path=predictions,
                report_path=report_path,
            )

            self.assertEqual(
                report["metrics"]["radiomics_only"]["reason"],
                "train_split_must_contain_both_classes",
            )
            rows = read_csv_rows(predictions)
            self.assertTrue(all(row["status"] == "failed" for row in rows))


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
