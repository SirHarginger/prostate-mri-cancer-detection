from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from prostate_mri_cancer_detection.evaluation import (
    run_radiomics_cv_baseline,
    run_radiomics_ml_baseline,
)


try:
    import sklearn  # noqa: F401
except ImportError:  # pragma: no cover - depends on local environment.
    sklearn = None


@unittest.skipIf(sklearn is None, "scikit-learn is not installed")
class RadiomicsMLBaselineTests(unittest.TestCase):
    def test_runs_full_table_radiomics_logistic_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            features = write_features(root)
            metrics = root / "outputs" / "reports" / "metrics.json"
            predictions = root / "outputs" / "reports" / "predictions.csv"
            report_path = root / "outputs" / "reports" / "report.json"

            report = run_radiomics_ml_baseline(
                features_path=features,
                metrics_path=metrics,
                predictions_path=predictions,
                report_path=report_path,
            )

            self.assertEqual(report["case_counts"], {"total": 8, "train": 4, "validation": 2, "test": 2})
            self.assertEqual(report["feature_count"], 3)
            self.assertIn("test", report["metrics"])
            self.assertTrue(metrics.exists())
            self.assertEqual(len(read_csv_rows(predictions)), 8)
            self.assertTrue(report_path.exists())
            self.assertNotIn("label_cspca", report["top_coefficients"][0]["feature"])

    def test_runs_rotated_fold_radiomics_logistic_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            features = write_cv_features(root)
            metrics = root / "outputs" / "reports" / "cv_metrics.json"
            predictions = root / "outputs" / "reports" / "cv_predictions.csv"
            report_path = root / "outputs" / "reports" / "cv_report.json"

            report = run_radiomics_cv_baseline(
                features_path=features,
                metrics_path=metrics,
                predictions_path=predictions,
                report_path=report_path,
                c_values=[0.1, 1.0],
            )

            prediction_rows = read_csv_rows(predictions)
            test_rows = [row for row in prediction_rows if row["split"] == "test"]

            self.assertEqual(report["stage"], "rotated_fold_radiomics_only_ml_baseline")
            self.assertEqual(report["feature_count"], 3)
            self.assertEqual(len(report["folds"]), 5)
            self.assertEqual(report["aggregate"]["pooled_test_default"]["metrics"]["n"], 10)
            self.assertEqual(len(prediction_rows), 50)
            self.assertEqual(len(test_rows), 10)
            self.assertEqual(
                {row["case_id"]: sum(1 for test_row in test_rows if test_row["case_id"] == row["case_id"]) for row in test_rows},
                {row["case_id"]: 1 for row in test_rows},
            )
            self.assertEqual(
                report["aggregate"]["validation_selected_fixed_sensitivity"]["status"],
                "ok",
            )
            self.assertTrue(metrics.exists())
            self.assertTrue(report_path.exists())
            self.assertNotIn("label_cspca", report["top_coefficients"][0]["feature"])


def write_features(root: Path) -> Path:
    path = root / "data" / "features" / "radiomics_full.csv"
    rows = [
        ("10000_1000000", "fold0", "NO", "1.0", "0.1", "10"),
        ("10001_1000001", "fold0", "YES", "3.0", "1.1", "20"),
        ("10002_1000002", "fold1", "NO", "1.2", "0.2", "11"),
        ("10003_1000003", "fold2", "YES", "3.2", "1.2", "21"),
        ("10004_1000004", "fold3", "NO", "1.1", "0.3", "12"),
        ("10005_1000005", "fold3", "YES", "3.1", "1.3", "22"),
        ("10006_1000006", "fold4", "NO", "1.4", "0.4", "13"),
        ("10007_1000007", "fold4", "YES", "3.4", "1.4", "23"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "case_id",
                "fold",
                "label_cspca",
                "t2w_intensity_mean",
                "adc_intensity_mean",
                "hbv_voxel_count",
            ]
        )
        writer.writerows(rows)
    return path


def write_cv_features(root: Path) -> Path:
    path = root / "data" / "features" / "radiomics_full_cv.csv"
    rows = []
    for fold_index in range(5):
        fold = f"fold{fold_index}"
        rows.append(
            (
                f"100{fold_index}0_10000{fold_index}0",
                fold,
                "NO",
                str(1.0 + fold_index / 10),
                str(0.2 + fold_index / 100),
                str(10 + fold_index),
            )
        )
        rows.append(
            (
                f"100{fold_index}1_10000{fold_index}1",
                fold,
                "YES",
                str(3.0 + fold_index / 10),
                str(1.2 + fold_index / 100),
                str(20 + fold_index),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "case_id",
                "fold",
                "label_cspca",
                "t2w_intensity_mean",
                "adc_intensity_mean",
                "hbv_voxel_count",
            ]
        )
        writer.writerows(rows)
    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


if __name__ == "__main__":
    unittest.main()
