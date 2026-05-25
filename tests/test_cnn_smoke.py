from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from prostate_mri_cancer_detection.cnn import (
    run_cnn_smoke_training,
    select_cnn_rows,
    windowed_slice_indices,
)


try:
    import numpy as np  # noqa: F401
    import SimpleITK as sitk
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - depends on local environment.
    np = None
    sitk = None
    torch = None


class CNNSmokeSelectionTests(unittest.TestCase):
    def test_selects_balanced_rows_per_split(self) -> None:
        rows = []
        for fold in range(5):
            rows.append(row(case_id=f"100{fold}0_10000{fold}0", fold=f"fold{fold}", label="NO"))
            rows.append(row(case_id=f"100{fold}1_10000{fold}1", fold=f"fold{fold}", label="YES"))
        selected = select_cnn_rows(rows, sample_size_per_split=2, all_cases=False)

        self.assertEqual(len(selected), 6)
        self.assertEqual(
            [item["case_id"] for item in selected],
            [
                "10000_1000000",
                "10001_1000001",
                "10030_1000030",
                "10031_1000031",
                "10040_1000040",
                "10041_1000041",
            ],
        )

    def test_windowed_slice_indices_are_centered_and_clamped(self) -> None:
        self.assertEqual(windowed_slice_indices(center_index=2, depth=5, slice_window=3), [1, 2, 3])
        self.assertEqual(windowed_slice_indices(center_index=0, depth=5, slice_window=5), [0, 0, 0, 1, 2])
        self.assertEqual(windowed_slice_indices(center_index=4, depth=5, slice_window=5), [2, 3, 4, 4, 4])


@unittest.skipIf(torch is None or sitk is None or np is None, "PyTorch, NumPy, and SimpleITK are required")
class CNNSmokeTrainingTests(unittest.TestCase):
    def test_runs_tiny_split_safe_cnn_smoke_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = create_cnn_fixture(root)
            manifest = write_manifest(root)
            embeddings = root / "data" / "features" / "cnn_embeddings.csv"
            predictions = root / "outputs" / "reports" / "cnn_predictions.csv"
            report_path = root / "outputs" / "reports" / "cnn_report.json"
            model_path = root / "outputs" / "models" / "cnn_model.pt"

            report = run_cnn_smoke_training(
                manifest_path=manifest,
                raw_root=raw_root,
                embeddings_path=embeddings,
                predictions_path=predictions,
                report_path=report_path,
                model_path=model_path,
                sample_size_per_split=4,
                image_size=8,
                slice_window=3,
                max_epochs=1,
                batch_size=2,
                embedding_dim=4,
                augment_train=True,
            )

            embedding_rows = read_csv_rows(embeddings)
            prediction_rows = read_csv_rows(predictions)

            self.assertEqual(report["summary"]["examples_loaded"], 8)
            self.assertEqual(report["summary"]["failures"], 0)
            self.assertEqual(report["summary"]["validation_or_test_augmented_rows"], 0)
            self.assertEqual(report["case_counts"]["by_split"], {"test": 2, "train": 4, "validation": 2})
            self.assertEqual(len(report["model"]["epoch_history"]), 1)
            self.assertEqual(report["model"]["best_epoch"], 1)
            self.assertEqual(report["model"]["input_channels"], 9)
            self.assertEqual(report["model"]["slice_window"], 3)
            self.assertEqual(report["validation_selected_threshold"]["test"]["status"], "ok")
            self.assertEqual(len(embedding_rows), 8)
            self.assertEqual(len(prediction_rows), 8)
            self.assertEqual({row["encoder_type"] for row in embedding_rows}, {"smoke_trained_cnn"})
            self.assertEqual({row["augmentation_applied"] for row in embedding_rows}, {"False"})
            self.assertTrue(model_path.exists())
            self.assertTrue(report_path.exists())


def row(case_id: str, fold: str, label: str) -> dict[str, str]:
    return {
        "case_id": case_id,
        "fold": fold,
        "label_cspca": label,
        "path_t2w": "",
        "path_adc": "",
        "path_hbv": "",
        "path_gland_mask": "",
    }


def create_cnn_fixture(root: Path) -> Path:
    raw_root = root / "data" / "raw" / "picai"
    for fold in range(5):
        for label_index, label in enumerate(("NO", "YES")):
            case_id = f"100{fold}{label_index}_10000{fold}{label_index}"
            case_dir = raw_root / "images" / f"fold{fold}" / case_id
            base_value = 10 + fold * 5 + label_index * 20
            write_image(case_dir / f"{case_id}_t2w.mha", size=(8, 8, 4), spacing=(0.5, 0.5, 2.0), value=base_value)
            write_image(case_dir / f"{case_id}_adc.mha", size=(4, 4, 4), spacing=(1.0, 1.0, 2.0), value=base_value + 5)
            write_image(case_dir / f"{case_id}_hbv.mha", size=(4, 4, 4), spacing=(1.0, 1.0, 2.0), value=base_value + 10)
            write_mask(
                raw_root / "picai_labels" / "anatomical_delineations" / f"{case_id}.nii.gz",
                size=(8, 8, 4),
                spacing=(0.5, 0.5, 2.0),
            )
    return raw_root


def write_manifest(root: Path) -> Path:
    manifest = root / "data" / "interim" / "picai_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "case_id",
                "fold",
                "label_cspca",
                "path_t2w",
                "path_adc",
                "path_hbv",
                "path_gland_mask",
            ],
        )
        writer.writeheader()
        for fold in range(5):
            for label_index, label in enumerate(("NO", "YES")):
                case_id = f"100{fold}{label_index}_10000{fold}{label_index}"
                writer.writerow(
                    {
                        "case_id": case_id,
                        "fold": f"fold{fold}",
                        "label_cspca": label,
                        "path_t2w": f"images/fold{fold}/{case_id}/{case_id}_t2w.mha",
                        "path_adc": f"images/fold{fold}/{case_id}/{case_id}_adc.mha",
                        "path_hbv": f"images/fold{fold}/{case_id}/{case_id}_hbv.mha",
                        "path_gland_mask": f"picai_labels/anatomical_delineations/{case_id}.nii.gz",
                    }
                )
    return manifest


def write_image(path: Path, size: tuple[int, int, int], spacing: tuple[float, float, float], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.Image(size, sitk.sitkUInt16)
    image.SetSpacing(spacing)
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                image[x, y, z] = value + x + y + z
    sitk.WriteImage(image, str(path))


def write_mask(path: Path, size: tuple[int, int, int], spacing: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.Image(size, sitk.sitkUInt8)
    image.SetSpacing(spacing)
    for z in range(1, 3):
        for y in range(2, 6):
            for x in range(2, 6):
                image[x, y, z] = 1
    sitk.WriteImage(image, str(path))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


if __name__ == "__main__":
    unittest.main()
