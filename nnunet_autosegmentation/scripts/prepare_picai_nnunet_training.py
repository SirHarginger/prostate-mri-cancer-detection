#!/usr/bin/env python
"""Prepare PI-CAI gland and csPCa lesion labels for nnU-Net training."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PI-CAI images and labels into a supervised nnU-Net dataset."
    )
    parser.add_argument(
        "--config",
        default="nnunet_autosegmentation/config/picai_gland_lesion_nnunet_config.json",
        type=Path,
    )
    parser.add_argument(
        "--report",
        default="nnunet_autosegmentation/outputs/reports/prepare_picai_gland_lesion_nnunet_report.json",
        type=Path,
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional case limit. Use 0 for all usable cases.")
    parser.add_argument(
        "--require-lesion-positive",
        action="store_true",
        help="For small smoke datasets, keep scanning until the requested limit of non-empty lesion cases is written.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing Dataset910 output before writing. Use this for repeated smoke runs.",
    )
    return parser.parse_args()


def load_simpleitk():
    try:
        import numpy as np
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit("NumPy and SimpleITK are required for PI-CAI nnU-Net preparation.") from exc
    return np, sitk


def split_pipe_value(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split("|") if part.strip()]


def resolve_path(value: str, raw_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return raw_root / path


def first_existing_path(value: str, raw_root: Path) -> Path | None:
    for item in split_pipe_value(value):
        path = resolve_path(item, raw_root)
        if path.exists():
            return path
    return None


def resample_to_reference(moving: Any, reference: Any, interpolator: int, sitk: Any) -> Any:
    return sitk.Resample(moving, reference, sitk.Transform(), interpolator, 0, moving.GetPixelID())


def binary_mask_array(mask: Any, reference: Any, sitk: Any, np: Any) -> Any:
    resampled = resample_to_reference(mask, reference, sitk.sitkNearestNeighbor, sitk)
    return np.asarray(sitk.GetArrayFromImage(resampled)) > 0


def combined_label_image(reference: Any, gland: Any, lesion: Any | None, sitk: Any, np: Any) -> tuple[Any, dict[str, int]]:
    gland_array = binary_mask_array(gland, reference, sitk, np)
    label_array = np.zeros(gland_array.shape, dtype=np.uint8)
    label_array[gland_array] = 1

    lesion_voxels = 0
    if lesion is not None:
        lesion_array = binary_mask_array(lesion, reference, sitk, np)
        lesion_voxels = int(lesion_array.sum())
        label_array[lesion_array] = 2

    label = sitk.GetImageFromArray(label_array)
    label.CopyInformation(reference)
    return label, {
        "gland_voxels": int(gland_array.sum()),
        "lesion_voxels": lesion_voxels,
        "label_1_voxels": int((label_array == 1).sum()),
        "label_2_voxels": int((label_array == 2).sum()),
    }


def write_dataset_json(dataset_dir: Path) -> None:
    payload = {
        "channel_names": {
            "0": "T2W",
            "1": "ADC",
            "2": "HBV",
        },
        "labels": {
            "background": 0,
            "prostate_gland": 1,
            "cspca_lesion": 2,
        },
        "numTraining": len(list((dataset_dir / "labelsTr").glob("*.nii.gz"))),
        "file_ending": ".nii.gz",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def cleanup_case_outputs(dataset_dir: Path, case_id: str) -> None:
    for suffix in ("0000", "0001", "0002"):
        path = dataset_dir / "imagesTr" / f"{case_id}_{suffix}.nii.gz"
        if path.exists():
            path.unlink()
    label_path = dataset_dir / "labelsTr" / f"{case_id}.nii.gz"
    if label_path.exists():
        label_path.unlink()


def prepare_case(row: dict[str, str], raw_root: Path, dataset_dir: Path, sitk: Any, np: Any) -> dict[str, Any]:
    case_id = row.get("case_id", "")
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "case_id": case_id,
        "fold": row.get("fold", ""),
        "written": False,
        "empty_gland": False,
        "empty_lesion": False,
        "paths": {},
        "label_counts": {},
        "issues": [],
    }

    required_paths = {
        "t2w": first_existing_path(row.get("path_t2w", ""), raw_root),
        "adc": first_existing_path(row.get("path_adc", ""), raw_root),
        "hbv": first_existing_path(row.get("path_hbv", ""), raw_root),
        "gland": first_existing_path(row.get("path_gland_mask", ""), raw_root),
    }
    lesion_path = first_existing_path(row.get("path_lesion_mask", ""), raw_root)

    for name, path in required_paths.items():
        if path is None:
            result["issues"].append(f"missing_{name}")
        else:
            result["paths"][name] = str(path)
    if result["issues"]:
        return result

    try:
        reference = sitk.ReadImage(str(required_paths["t2w"]))
        adc = sitk.ReadImage(str(required_paths["adc"]))
        hbv = sitk.ReadImage(str(required_paths["hbv"]))
        gland = sitk.ReadImage(str(required_paths["gland"]))
        lesion = sitk.ReadImage(str(lesion_path)) if lesion_path is not None else None

        sitk.WriteImage(reference, str(images_tr / f"{case_id}_0000.nii.gz"))
        sitk.WriteImage(resample_to_reference(adc, reference, sitk.sitkLinear, sitk), str(images_tr / f"{case_id}_0001.nii.gz"))
        sitk.WriteImage(resample_to_reference(hbv, reference, sitk.sitkLinear, sitk), str(images_tr / f"{case_id}_0002.nii.gz"))

        label, counts = combined_label_image(reference, gland, lesion, sitk, np)
        result["label_counts"] = counts
        result["empty_gland"] = counts["gland_voxels"] == 0
        result["empty_lesion"] = counts["lesion_voxels"] == 0
        if result["empty_gland"]:
            result["issues"].append("empty_gland")
            return result

        sitk.WriteImage(label, str(labels_tr / f"{case_id}.nii.gz"))
        result["written"] = True
    except Exception as exc:  # noqa: BLE001
        result["issues"].append(f"{type(exc).__name__}: {exc}")

    return result


def main() -> int:
    args = parse_args()
    np, sitk = load_simpleitk()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = Path(config["picai_source"]["manifest"])
    raw_root = Path(config["picai_source"]["raw_root"])
    dataset_dir = Path(config["nnunet"]["dataset_dir"])
    report_path = args.report

    if args.clean_output and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    selected = sorted(rows, key=lambda row: row.get("case_id", ""))

    case_reports = []
    attempted_rows = []
    if args.require_lesion_positive and args.limit:
        for row in selected:
            attempted_rows.append(row)
            case_report = prepare_case(row, raw_root, dataset_dir, sitk, np)
            if case_report["written"] and not case_report["empty_lesion"]:
                case_reports.append(case_report)
            else:
                cleanup_case_outputs(dataset_dir, row.get("case_id", ""))
            if len(case_reports) >= args.limit:
                break
    else:
        if args.limit:
            selected = selected[: args.limit]
        attempted_rows = selected
        case_reports = [prepare_case(row, raw_root, dataset_dir, sitk, np) for row in selected]

    write_dataset_json(dataset_dir)

    issue_counts = Counter(issue for report in case_reports for issue in report["issues"])
    summary = {
        "manifest_rows": len(rows),
        "attempted_cases": len(attempted_rows),
        "selected_cases": len(case_reports),
        "written_cases": sum(1 for report in case_reports if report["written"]),
        "failed_or_skipped_cases": sum(1 for report in case_reports if not report["written"]),
        "empty_gland_cases": sum(1 for report in case_reports if report["empty_gland"]),
        "empty_lesion_cases": sum(1 for report in case_reports if report["empty_lesion"]),
        "issue_counts": dict(issue_counts),
        "dataset_dir": str(dataset_dir),
    }
    report = {
        "stage": "prepare_picai_gland_lesion_nnunet_training",
        "config": str(args.config),
        "manifest": str(manifest_path),
        "raw_root": str(raw_root),
        "summary": summary,
        "cases": case_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote PI-CAI nnU-Net dataset: {dataset_dir}")
    print(f"Wrote preparation report: {report_path}")
    print(f"Summary: {summary}")
    return 0 if summary["written_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
