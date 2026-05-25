from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from prostate_mri_cancer_detection.evaluation import run_hybrid_ml_baseline


try:
    import sklearn  # noqa: F401
except ImportError:  # pragma: no cover - depends on local environment.
    sklearn = None


@unittest.skipIf(sklearn is None, "scikit-learn is not installed")
class HybridMLBaselineTests(unittest.TestCase):
    def test_runs_aligned_radiomics_cnn_and_hybrid_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            radiomics = write_radiomics(root)
            embeddings = write_embeddings(root)
            metrics = root / "outputs" / "reports" / "hybrid_metrics.json"
            predictions = root / "outputs" / "reports" / "hybrid_predictions.csv"
            report_path = root / "outputs" / "reports" / "hybrid_report.json"

            report = run_hybrid_ml_baseline(
                radiomics_path=radiomics,
                embeddings_path=embeddings,
                metrics_path=metrics,
                predictions_path=predictions,
                report_path=report_path,
                c_values=[0.1, 1.0],
            )

            prediction_rows = read_csv_rows(predictions)

            self.assertEqual(report["stage"], "hybrid_radiomics_cnn_ml_baseline")
            self.assertEqual(report["case_counts"]["aligned"], 8)
            self.assertEqual(report["case_counts"]["excluded"], 0)
            self.assertEqual(report["feature_counts"]["radiomics_only"], 3)
            self.assertEqual(report["feature_counts"]["cnn_embedding_only"], 2)
            self.assertEqual(report["feature_counts"]["hybrid_radiomics_cnn"], 5)
            self.assertEqual(set(report["baselines"]), {"radiomics_only", "cnn_embedding_only", "hybrid_radiomics_cnn"})
            self.assertEqual(len(prediction_rows), 24)
            self.assertTrue(metrics.exists())
            self.assertTrue(report_path.exists())
            self.assertNotIn("label_cspca", report["top_coefficients"]["hybrid_radiomics_cnn"][0]["feature"])


def write_radiomics(root: Path) -> Path:
    path = root / "data" / "features" / "radiomics.csv"
    rows = feature_rows()
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
                "hbv_intensity_std",
            ]
        )
        for case_id, fold, label, offset in rows:
            writer.writerow(
                [
                    case_id,
                    fold,
                    label,
                    str(1.0 + offset),
                    str(0.2 + offset),
                    str(2.0 + offset),
                ]
            )
    return path


def write_embeddings(root: Path) -> Path:
    path = root / "data" / "features" / "cnn_embeddings.csv"
    rows = feature_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "case_id",
                "fold",
                "split",
                "label_cspca",
                "encoder_name",
                "encoder_type",
                "augmentation_applied",
                "cnn_embedding_000",
                "cnn_embedding_001",
            ]
        )
        for case_id, fold, label, offset in rows:
            writer.writerow(
                [
                    case_id,
                    fold,
                    split_for_fold(fold),
                    label,
                    "tiny_multisequence_cnn_baseline_v1",
                    "baseline_trained_cnn",
                    "False",
                    str(0.5 + offset),
                    str(1.5 + offset),
                ]
            )
    return path


def feature_rows() -> list[tuple[str, str, str, float]]:
    return [
        ("10000_1000000", "fold0", "NO", 0.0),
        ("10001_1000001", "fold0", "YES", 2.0),
        ("10002_1000002", "fold1", "NO", 0.1),
        ("10003_1000003", "fold2", "YES", 2.1),
        ("10004_1000004", "fold3", "NO", 0.2),
        ("10005_1000005", "fold3", "YES", 2.2),
        ("10006_1000006", "fold4", "NO", 0.3),
        ("10007_1000007", "fold4", "YES", 2.3),
    ]


def split_for_fold(fold: str) -> str:
    if fold in {"fold0", "fold1", "fold2"}:
        return "train"
    if fold == "fold3":
        return "validation"
    return "test"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


if __name__ == "__main__":
    unittest.main()
