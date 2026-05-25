"""Extract simple image statistics from nnU-Net predicted masks.

This is a lightweight starter extractor for autosegmentation outputs. It avoids
committing generated features and is intentionally separate from the main
PI-CAI classification pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract features from nnU-Net predicted masks.")
    parser.add_argument("--config", default="nnunet_autosegmentation/config/prostate_autoseg_config.json", type=Path)
    args = parser.parse_args()

    try:
        import numpy as np
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError("NumPy and SimpleITK are required for feature extraction") from error

    config = json.loads(args.config.read_text())
    image_dir = Path(config["feature_extraction"]["image_dir"])
    mask_dir = Path(config["feature_extraction"]["mask_dir"])
    output_csv = Path(config["feature_extraction"]["output_csv"])
    report_json = Path(config["feature_extraction"]["report_json"])

    rows = []
    failures = []
    for mask_path in sorted(mask_dir.glob("*.nii.gz")):
        case_id = mask_path.name.replace(".nii.gz", "")
        image_candidates = sorted(image_dir.glob(f"{case_id}*.nii.gz"))
        if not image_candidates:
            failures.append({"case_id": case_id, "reason": "missing_image"})
            continue
        try:
            image = sitk.ReadImage(str(image_candidates[0]))
            mask = sitk.ReadImage(str(mask_path))
            image_array = np.asarray(sitk.GetArrayFromImage(image), dtype=np.float32)
            mask_array = np.asarray(sitk.GetArrayFromImage(mask)) > 0
            if image_array.shape != mask_array.shape:
                failures.append({"case_id": case_id, "reason": "shape_mismatch"})
                continue
            if not bool(mask_array.any()):
                failures.append({"case_id": case_id, "reason": "empty_mask"})
                continue
            values = image_array[mask_array]
            rows.append(
                {
                    "case_id": case_id,
                    "image_path": str(image_candidates[0]),
                    "mask_path": str(mask_path),
                    "voxel_count": int(mask_array.sum()),
                    "intensity_mean": float(values.mean()),
                    "intensity_std": float(values.std()),
                    "intensity_min": float(values.min()),
                    "intensity_max": float(values.max()),
                }
            )
        except Exception as error:  # noqa: BLE001 - per-case extraction failure
            failures.append({"case_id": case_id, "reason": f"{type(error).__name__}: {error}"})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = [
            "case_id",
            "image_path",
            "mask_path",
            "voxel_count",
            "intensity_mean",
            "intensity_std",
            "intensity_min",
            "intensity_max",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "stage": "extract_features_from_nnunet_masks",
        "features_written": len(rows),
        "failures": len(failures),
        "output_csv": str(output_csv),
        "failures_sample": failures[:20],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote features: {output_csv}")
    print(f"Wrote report: {report_json}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
