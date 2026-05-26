#!/usr/bin/env python
"""Extract label-specific features from nnU-Net predicted prostate masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

LABEL_OUTPUTS = {
    "prostate_gland": 1,
    "cspca_lesion": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract label-specific features from nnU-Net predictions.")
    parser.add_argument(
        "--config",
        default="nnunet_autosegmentation/config/picai_gland_lesion_nnunet_config.json",
        type=Path,
    )
    return parser.parse_args()


def load_dependencies():
    try:
        import numpy as np
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError("NumPy and SimpleITK are required for feature extraction") from error
    return np, sitk


def empty_feature_row(case_id: str, image_path: Path, mask_path: Path, label_name: str, label_value: int) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "label_name": label_name,
        "label_value": label_value,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "empty_mask": "True",
        "voxel_count": 0,
        "volume_mm3": 0.0,
        "intensity_mean": "",
        "intensity_std": "",
        "intensity_min": "",
        "intensity_max": "",
        "bbox_z_min": "",
        "bbox_z_max": "",
        "bbox_y_min": "",
        "bbox_y_max": "",
        "bbox_x_min": "",
        "bbox_x_max": "",
        "centroid_x_mm": "",
        "centroid_y_mm": "",
        "centroid_z_mm": "",
    }


def feature_row(
    *,
    case_id: str,
    image_path: Path,
    mask_path: Path,
    label_name: str,
    label_value: int,
    image: Any,
    image_array: Any,
    label_mask: Any,
    np: Any,
) -> dict[str, Any]:
    if not bool(label_mask.any()):
        return empty_feature_row(case_id, image_path, mask_path, label_name, label_value)

    values = image_array[label_mask]
    coords_zyx = np.argwhere(label_mask)
    z_min, y_min, x_min = coords_zyx.min(axis=0).tolist()
    z_max, y_max, x_max = coords_zyx.max(axis=0).tolist()
    centroid_zyx = coords_zyx.mean(axis=0)
    centroid_xyz_index = (
        float(centroid_zyx[2]),
        float(centroid_zyx[1]),
        float(centroid_zyx[0]),
    )
    centroid_xyz_mm = image.TransformContinuousIndexToPhysicalPoint(centroid_xyz_index)
    spacing = image.GetSpacing()
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])

    return {
        "case_id": case_id,
        "label_name": label_name,
        "label_value": label_value,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "empty_mask": "False",
        "voxel_count": int(label_mask.sum()),
        "volume_mm3": float(label_mask.sum() * voxel_volume),
        "intensity_mean": float(values.mean()),
        "intensity_std": float(values.std()),
        "intensity_min": float(values.min()),
        "intensity_max": float(values.max()),
        "bbox_z_min": int(z_min),
        "bbox_z_max": int(z_max),
        "bbox_y_min": int(y_min),
        "bbox_y_max": int(y_max),
        "bbox_x_min": int(x_min),
        "bbox_x_max": int(x_max),
        "centroid_x_mm": float(centroid_xyz_mm[0]),
        "centroid_y_mm": float(centroid_xyz_mm[1]),
        "centroid_z_mm": float(centroid_xyz_mm[2]),
    }


def find_image_for_case(image_dir: Path, case_id: str) -> Path | None:
    candidates = sorted(image_dir.glob(f"{case_id}_0000.nii.gz"))
    if candidates:
        return candidates[0]
    candidates = sorted(image_dir.glob(f"{case_id}*.nii.gz"))
    return candidates[0] if candidates else None


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "label_name",
        "label_value",
        "image_path",
        "mask_path",
        "empty_mask",
        "voxel_count",
        "volume_mm3",
        "intensity_mean",
        "intensity_std",
        "intensity_min",
        "intensity_max",
        "bbox_z_min",
        "bbox_z_max",
        "bbox_y_min",
        "bbox_y_max",
        "bbox_x_min",
        "bbox_x_max",
        "centroid_x_mm",
        "centroid_y_mm",
        "centroid_z_mm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    feature_config = config["feature_extraction"]
    if "prostate_output_csv" in feature_config and "lesion_output_csv" in feature_config:
        return {
            "prostate_gland": Path(feature_config["prostate_output_csv"]),
            "cspca_lesion": Path(feature_config["lesion_output_csv"]),
        }
    legacy_output = Path(feature_config["output_csv"])
    return {
        "prostate_gland": legacy_output.with_name("prostate_gland_features.csv"),
        "cspca_lesion": legacy_output.with_name("cspca_lesion_candidate_features.csv"),
    }


def main() -> int:
    args = parse_args()
    np, sitk = load_dependencies()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    image_dir = Path(config["feature_extraction"]["image_dir"])
    mask_dir = Path(config["feature_extraction"]["mask_dir"])
    report_json = Path(config["feature_extraction"]["report_json"])
    paths = output_paths(config)

    rows_by_label = {label_name: [] for label_name in LABEL_OUTPUTS}
    failures = []

    for mask_path in sorted(mask_dir.glob("*.nii.gz")):
        case_id = mask_path.name.replace(".nii.gz", "")
        image_path = find_image_for_case(image_dir, case_id)
        if image_path is None:
            failures.append({"case_id": case_id, "reason": "missing_t2w_channel"})
            continue

        try:
            image = sitk.ReadImage(str(image_path))
            mask = sitk.ReadImage(str(mask_path))
            image_array = np.asarray(sitk.GetArrayFromImage(image), dtype=np.float32)
            mask_array = np.asarray(sitk.GetArrayFromImage(mask))
            if image_array.shape != mask_array.shape:
                failures.append(
                    {
                        "case_id": case_id,
                        "reason": "shape_mismatch",
                        "image_shape": list(image_array.shape),
                        "mask_shape": list(mask_array.shape),
                    }
                )
                continue

            for label_name, label_value in LABEL_OUTPUTS.items():
                rows_by_label[label_name].append(
                    feature_row(
                        case_id=case_id,
                        image_path=image_path,
                        mask_path=mask_path,
                        label_name=label_name,
                        label_value=label_value,
                        image=image,
                        image_array=image_array,
                        label_mask=mask_array == label_value,
                        np=np,
                    )
                )
        except Exception as error:  # noqa: BLE001 - per-case extraction failure
            failures.append({"case_id": case_id, "reason": f"{type(error).__name__}: {error}"})

    for label_name, rows in rows_by_label.items():
        write_rows(paths[label_name], rows)

    report = {
        "stage": "extract_label_specific_nnunet_features",
        "config": str(args.config),
        "mask_dir": str(mask_dir),
        "image_dir": str(image_dir),
        "outputs": {name: str(path) for name, path in paths.items()},
        "summary": {
            "prediction_masks_found": len(list(mask_dir.glob("*.nii.gz"))),
            "prostate_rows": len(rows_by_label["prostate_gland"]),
            "lesion_rows": len(rows_by_label["cspca_lesion"]),
            "empty_prostate_rows": sum(row["empty_mask"] == "True" for row in rows_by_label["prostate_gland"]),
            "empty_lesion_rows": sum(row["empty_mask"] == "True" for row in rows_by_label["cspca_lesion"]),
            "failures": len(failures),
        },
        "failures_sample": failures[:20],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote prostate features: {paths['prostate_gland']}")
    print(f"Wrote lesion candidate features: {paths['cspca_lesion']}")
    print(f"Wrote report: {report_json}")
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
