from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prostate_mri_cancer_detection.evaluation import generate_model_comparison_report


class ModelComparisonReportTests(unittest.TestCase):
    def test_generates_current_model_comparison_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            radiomics = write_json(root / "radiomics.json", radiomics_report())
            cnn = write_json(root / "cnn.json", cnn_report())
            hybrid = write_json(root / "hybrid.json", hybrid_report())
            output_json = root / "outputs" / "reports" / "comparison.json"
            output_md = root / "outputs" / "reports" / "comparison.md"

            report = generate_model_comparison_report(
                radiomics_cv_report_path=radiomics,
                cnn_report_path=cnn,
                hybrid_report_path=hybrid,
                output_json_path=output_json,
                output_markdown_path=output_md,
            )

            self.assertEqual(report["stage"], "current_model_comparison_report")
            self.assertEqual(report["comparisons"]["full_radiomics_cv"]["feature_count"], 39)
            self.assertEqual(
                report["comparisons"]["hybrid_aligned_subset"]["baselines"]["hybrid_radiomics_cnn"]["selected_c"],
                0.1,
            )
            self.assertTrue(output_json.exists())
            self.assertIn("Current Model Comparison", output_md.read_text(encoding="utf-8"))
            self.assertIn("Aligned hybrid", output_md.read_text(encoding="utf-8"))


def radiomics_report() -> dict:
    return {
        "case_counts": {"total": 1500},
        "label_counts": {"0": 1075, "1": 425},
        "feature_count": 39,
        "aggregate": {
            "pooled_test_default": {
                "metrics": {
                    "n": 1500,
                    "roc_auc": 0.7348,
                    "sensitivity": 0.6565,
                    "specificity": 0.6856,
                }
            },
            "validation_selected_fixed_sensitivity": {
                "status": "ok",
                "metrics": {"sensitivity": 0.8965, "specificity": 0.3572},
            },
            "fold_test_metric_summary": {},
        },
    }


def cnn_report() -> dict:
    return {
        "case_counts": {"loaded": 576},
        "label_counts": {"0": 307, "1": 269},
        "model": {
            "name": "TinyMultisequenceCNN",
            "training_status": "baseline_trained",
            "input_channels": 15,
            "slice_window": 5,
            "best_epoch": 4,
        },
        "metrics": {
            "test": {
                "metrics": {
                    "n": 192,
                    "roc_auc": 0.6846,
                    "sensitivity": 0.0,
                    "specificity": 1.0,
                }
            }
        },
        "validation_selected_threshold": {
            "test": {
                "status": "ok",
                "metrics": {"sensitivity": 0.8778, "specificity": 0.3333},
            }
        },
    }


def hybrid_report() -> dict:
    return {
        "case_counts": {"aligned": 576},
        "label_counts": {"0": 307, "1": 269},
        "split_label_counts": {"test": {"0": 102, "1": 90}},
        "feature_counts": {
            "radiomics_only": 39,
            "cnn_embedding_only": 32,
            "hybrid_radiomics_cnn": 71,
        },
        "baselines": {
            "radiomics_only": baseline(0.7169, 0.6444, 0.6078, 0.1),
            "cnn_embedding_only": baseline(0.6939, 0.6, 0.6569, 0.01),
            "hybrid_radiomics_cnn": baseline(0.7304, 0.6333, 0.6667, 0.1),
        },
        "top_coefficients": {
            "hybrid_radiomics_cnn": [
                {"feature": "radiomics:hbv_intensity_energy", "coefficient": 0.3, "abs_coefficient": 0.3}
            ]
        },
    }


def baseline(auc: float, sensitivity: float, specificity: float, selected_c: float) -> dict:
    return {
        "selected_c": selected_c,
        "feature_count": 1,
        "metrics": {
            "test": {
                "metrics": {
                    "n": 192,
                    "roc_auc": auc,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                }
            }
        },
        "validation_selected_threshold": {
            "test": {
                "status": "ok",
                "metrics": {"sensitivity": 0.8, "specificity": 0.5},
            }
        },
    }


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
