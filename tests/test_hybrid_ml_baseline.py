from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from prostate_mri_cancer_detection.evaluation import (
    bootstrap_metrics_ci,
    calibration_diagnostics,
    paired_auc_delta_ci,
    run_calibrated_fusion_baseline,
    run_hybrid_ml_baseline,
)


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
            self.assertEqual(report["paired_test_auc_deltas"]["hybrid_minus_radiomics"]["status"], "ok")
            self.assertEqual(report["baselines"]["hybrid_radiomics_cnn"]["test_calibration"]["status"], "ok")
            self.assertEqual(report["baselines"]["hybrid_radiomics_cnn"]["test_bootstrap_ci"]["status"], "ok")
            self.assertTrue(metrics.exists())
            self.assertTrue(report_path.exists())
            self.assertNotIn("label_cspca", report["top_coefficients"]["hybrid_radiomics_cnn"][0]["feature"])

    def test_runs_calibrated_probability_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            radiomics = write_radiomics(root)
            cnn_predictions = write_cnn_predictions(root)
            metrics = root / "outputs" / "reports" / "fusion_metrics.json"
            predictions = root / "outputs" / "reports" / "fusion_predictions.csv"
            report_path = root / "outputs" / "reports" / "fusion_report.json"

            report = run_calibrated_fusion_baseline(
                radiomics_path=radiomics,
                cnn_predictions_path=cnn_predictions,
                metrics_path=metrics,
                predictions_path=predictions,
                report_path=report_path,
                alpha_grid=[0.0, 0.5, 1.0],
                c_grid=[0.1, 1.0],
            )

            prediction_rows = read_csv_rows(predictions)

            self.assertEqual(report["stage"], "calibrated_probability_fusion_baseline")
            self.assertEqual(report["case_counts"]["aligned"], 8)
            self.assertEqual(report["case_counts"]["excluded"], 0)
            self.assertEqual(
                set(report["baselines"]),
                {
                    "radiomics_only",
                    "cnn_probability_only",
                    "weighted_probability_fusion",
                    "stacked_probability_fusion",
                },
            )
            self.assertEqual(len(prediction_rows), 32)
            self.assertEqual(report["baselines"]["weighted_probability_fusion"]["test_calibration"]["status"], "ok")
            self.assertIn("weighted_minus_cnn", report["paired_test_auc_deltas"])
            self.assertTrue(metrics.exists())
            self.assertTrue(report_path.exists())

    def test_metric_rigor_helpers(self) -> None:
        left = prediction_rows("left", [0.1, 0.2, 0.8, 0.9])
        right = prediction_rows("right", [0.2, 0.3, 0.7, 0.8])

        ci = bootstrap_metrics_ci(left, n_bootstrap=20, seed=1)
        delta = paired_auc_delta_ci(left, right, n_bootstrap=20, seed=1)
        calibration = calibration_diagnostics(left, n_bins=2)

        self.assertEqual(ci["status"], "ok")
        self.assertEqual(delta["status"], "ok")
        self.assertEqual(calibration["status"], "ok")
        self.assertIn("brier_score", calibration)


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


def write_cnn_predictions(root: Path) -> Path:
    path = root / "outputs" / "reports" / "cnn_predictions.csv"
    rows = feature_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "baseline",
                "case_id",
                "fold",
                "split",
                "label",
                "score",
                "probability",
                "prediction",
                "status",
                "reason",
            ]
        )
        for case_id, fold, label, offset in rows:
            probability = 0.2 + offset / 3.0
            writer.writerow(
                [
                    "cnn_smoke_multisequence",
                    case_id,
                    fold,
                    split_for_fold(fold),
                    str(1 if label == "YES" else 0),
                    str(probability),
                    str(probability),
                    str(1 if probability >= 0.5 else 0),
                    "ok",
                    "",
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


def prediction_rows(baseline_name: str, probabilities: list[float]) -> list[dict[str, str]]:
    labels = [0, 0, 1, 1]
    return [
        {
            "baseline": baseline_name,
            "case_id": f"case_{index}",
            "fold": "fold4",
            "split": "test",
            "label": str(label),
            "score": str(probability),
            "probability": str(probability),
            "prediction": str(int(probability >= 0.5)),
            "status": "ok",
            "reason": "",
        }
        for index, (label, probability) in enumerate(zip(labels, probabilities))
    ]


if __name__ == "__main__":
    unittest.main()
