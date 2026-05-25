from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from prostate_mri_cancer_detection.evaluation import generate_evaluation_report


class EvaluationReportTests(unittest.TestCase):
    def test_generates_metrics_error_lists_and_fixed_sensitivity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            predictions = write_predictions(root)
            json_report = root / "outputs" / "reports" / "evaluation.json"
            markdown_report = root / "outputs" / "reports" / "evaluation.md"

            report = generate_evaluation_report(
                predictions_path=predictions,
                report_json_path=json_report,
                report_markdown_path=markdown_report,
                target_sensitivity=0.90,
            )

            payload = report["baselines"]["radiomics_only"]["test"]
            self.assertEqual(payload["metrics"]["confusion_matrix"], {"tp": 1, "tn": 1, "fp": 1, "fn": 1})
            self.assertEqual(payload["false_positives"], ["10002_1000002"])
            self.assertEqual(payload["false_negatives"], ["10003_1000003"])
            self.assertEqual(payload["fixed_sensitivity"]["status"], "ok")
            self.assertEqual(
                payload["fixed_sensitivity"]["metrics"]["confusion_matrix"],
                {"tp": 2, "tn": 1, "fp": 1, "fn": 0},
            )
            self.assertTrue(json_report.exists())
            self.assertIn("Prototype Evaluation Report", markdown_report.read_text(encoding="utf-8"))
            self.assertIn("prototype only", report["ablation_status"]["prototype_embedding_only"])

    def test_fixed_sensitivity_is_undefined_for_single_class_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            predictions = write_predictions(root, single_class_validation=True)
            json_report = root / "outputs" / "reports" / "evaluation.json"
            markdown_report = root / "outputs" / "reports" / "evaluation.md"

            report = generate_evaluation_report(
                predictions_path=predictions,
                report_json_path=json_report,
                report_markdown_path=markdown_report,
            )

            payload = report["baselines"]["radiomics_only"]["validation"]
            self.assertEqual(payload["fixed_sensitivity"]["status"], "undefined")
            self.assertIsNone(payload["metrics"]["roc_auc"])


def write_predictions(root: Path, single_class_validation: bool = False) -> Path:
    path = root / "outputs" / "reports" / "predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("radiomics_only", "10000_1000000", "fold4", "test", "0", "0.10", "0"),
        ("radiomics_only", "10001_1000001", "fold4", "test", "1", "0.80", "1"),
        ("radiomics_only", "10002_1000002", "fold4", "test", "0", "0.70", "1"),
        ("radiomics_only", "10003_1000003", "fold4", "test", "1", "0.40", "0"),
        ("prototype_embedding_only", "10000_1000000", "fold4", "test", "0", "0.20", "0"),
        ("prototype_embedding_only", "10001_1000001", "fold4", "test", "1", "0.90", "1"),
    ]
    if single_class_validation:
        rows.append(("radiomics_only", "10004_1000004", "fold3", "validation", "0", "0.20", "0"))
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
        for baseline, case_id, fold, split, label, probability, prediction in rows:
            writer.writerow(
                [
                    baseline,
                    case_id,
                    fold,
                    split,
                    label,
                    "",
                    probability,
                    prediction,
                    "ok",
                    "",
                ]
            )
    return path


if __name__ == "__main__":
    unittest.main()
