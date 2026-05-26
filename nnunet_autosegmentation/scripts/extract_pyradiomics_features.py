#!/usr/bin/env python
"""Extract PyRadiomics features from nnU-Net prostate and lesion labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

LABELS = {
    "prostate_gland": 1,
    "cspca_lesion_candidate": 2,
}

MODALITIES = {
    "t2w": "0000",
    "adc": "0001",
    "hbv": "0002",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PyRadiomics features from nnU-Net masks.")
    parser.add_argument(
        "--config",
        default="nnunet_autosegmentation/config/picai_gland_lesion_nnunet_config.json",
        type=Path,
    )
    parser.add_argument(
        "--output",
        default="nnunet_autosegmentation/outputs/features/prostate_mri_pyradiomics_features.csv",
        type=Path,
    )
    parser.add_argument(
        "--report",
        default="nnunet_autosegmentation/outputs/reports/prostate_mri_pyradiomics_feature_report.json",
        type=Path,
    )
    parser.add_argument("--bin-width", type=float, default=25.0)
    return parser.parse_args()


def load_dependencies():
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor
    except ImportError as error:
        raise RuntimeError(
            "PyRadiomics and SimpleITK are required. Install with: python -m pip install pyradiomics"
        ) from error
    return sitk, featureextractor


def build_extractor(featureextractor: Any, bin_width: float):
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=bin_width,
        force2D=False,
        normalize=False,
    )
    extractor.disableAllFeatures()
    for feature_class in ("firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        extractor.enableFeatureClassByName(feature_class)
    return extractor


def clean_feature_name(name: str) -> str:
    return name.replace("original_", "").replace("-", "_")


def scalar_value(value: Any) -> str | float | int:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


def mask_has_label(sitk: Any, mask_path: Path, label: int) -> bool:
    mask = sitk.ReadImage(str(mask_path))
    binary = sitk.BinaryThreshold(mask, lowerThreshold=label, upperThreshold=label, insideValue=1, outsideValue=0)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(binary)
    return stats.GetSum() > 0


def extract_case_label_modality(
    extractor: Any,
    sitk: Any,
    *,
    case_id: str,
    image_path: Path,
    mask_path: Path,
    label_name: str,
    label_value: int,
    modality: str,
) -> tuple[dict[str, Any], str]:
    base = {
        "case_id": case_id,
        "label_name": label_name,
        "label_value": label_value,
        "modality": modality,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "status": "ok",
    }
    if not image_path.exists():
        base["status"] = "missing_image"
        return base, "missing_image"
    if not mask_has_label(sitk, mask_path, label_value):
        base["status"] = "empty_mask"
        return base, "empty_mask"

    try:
        features = extractor.execute(str(image_path), str(mask_path), label=label_value)
    except Exception as error:  # noqa: BLE001
        base["status"] = f"{type(error).__name__}: {error}"
        return base, base["status"]

    for key, value in features.items():
        if key.startswith("diagnostics_"):
            continue
        base[clean_feature_name(key)] = scalar_value(value)
    return base, "ok"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["case_id", "label_name", "label_value", "modality", "status", "image_path", "mask_path"]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    sitk, featureextractor = load_dependencies()
    extractor = build_extractor(featureextractor, args.bin_width)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    image_dir = Path(config["feature_extraction"]["image_dir"])
    mask_dir = Path(config["feature_extraction"]["mask_dir"])

    rows = []
    status_counts: dict[str, int] = {}
    for mask_path in sorted(mask_dir.glob("*.nii.gz")):
        case_id = mask_path.name.replace(".nii.gz", "")
        for label_name, label_value in LABELS.items():
            for modality, suffix in MODALITIES.items():
                image_path = image_dir / f"{case_id}_{suffix}.nii.gz"
                row, status = extract_case_label_modality(
                    extractor,
                    sitk,
                    case_id=case_id,
                    image_path=image_path,
                    mask_path=mask_path,
                    label_name=label_name,
                    label_value=label_value,
                    modality=modality,
                )
                rows.append(row)
                status_counts[status] = status_counts.get(status, 0) + 1

    write_csv(args.output, rows)
    report = {
        "stage": "extract_pyradiomics_features_from_nnunet_masks",
        "config": str(args.config),
        "output_csv": str(args.output),
        "mask_dir": str(mask_dir),
        "image_dir": str(image_dir),
        "labels": LABELS,
        "modalities": MODALITIES,
        "bin_width": args.bin_width,
        "summary": {
            "masks_found": len(list(mask_dir.glob("*.nii.gz"))),
            "rows_written": len(rows),
            "status_counts": status_counts,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote PyRadiomics features: {args.output}")
    print(f"Wrote report: {args.report}")
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
