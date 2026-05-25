#!/usr/bin/env python
"""Prepare external prostate MRI DICOM series for nnU-Net inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert selected PROSTATE-MRI DICOM series to nnU-Net imagesTs NIfTI files."
    )
    parser.add_argument(
        "--config",
        default="nnunet_autosegmentation/config/prostate_autoseg_config.json",
        help="Autosegmentation config JSON.",
    )
    parser.add_argument(
        "--report",
        default="nnunet_autosegmentation/outputs/reports/prepare_nnunet_dataset_report.json",
        help="Output preparation report JSON.",
    )
    parser.add_argument(
        "--series-description",
        default="T2 TSE ax hi",
        help="DICOM Series Description to convert.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of cases to convert. Use 0 for all matching cases.",
    )
    return parser.parse_args()


def load_simpleitk():
    try:
        import SimpleITK as sitk  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "SimpleITK is required for DICOM to NIfTI conversion. "
            "Install it in the active environment, for example: pip install SimpleITK"
        ) from exc
    return sitk


def find_metadata(download_dir: Path) -> tuple[Path, Path]:
    candidates = [
        download_dir / "PROSTATE_MRI" / "metadata.csv",
        download_dir / "metadata.csv",
    ]
    for metadata_path in candidates:
        if metadata_path.exists():
            return metadata_path, metadata_path.parent
    raise FileNotFoundError(
        "Could not find metadata.csv under "
        f"{download_dir / 'PROSTATE_MRI'} or {download_dir}"
    )


def dicom_series_files(sitk, dicom_dir: Path) -> list[str]:
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        return sorted(str(path) for path in dicom_dir.glob("*.dcm"))
    if len(series_ids) == 1:
        return list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0]))

    best_files: list[str] = []
    for series_id in series_ids:
        files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_id))
        if len(files) > len(best_files):
            best_files = files
    return best_files


def write_dataset_json(dataset_root: Path) -> None:
    payload = {
        "channel_names": {"0": "T2"},
        "labels": {"background": 0, "prostate": 1},
        "numTraining": 0,
        "file_ending": ".nii.gz",
    }
    (dataset_root / "dataset.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    sitk = load_simpleitk()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())

    download_dir = Path(config["external_source"]["download_dir"])
    input_dir = Path(config["nnunet"]["input_dir"])
    dataset_root = input_dir.parent
    report_path = Path(args.report)

    input_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_dataset_json(dataset_root)

    metadata_path, metadata_root = find_metadata(download_dir)
    rows = list(csv.DictReader(metadata_path.open(newline="", encoding="utf-8", errors="replace")))

    converted = []
    failures = []
    seen_subjects: set[str] = set()

    for row in rows:
        if row.get("Series Description") != args.series_description:
            continue

        subject_id = row["Subject ID"]
        if subject_id in seen_subjects:
            continue
        if args.limit and len(converted) >= args.limit:
            break

        seen_subjects.add(subject_id)
        dicom_dir = metadata_root / row["File Location"].lstrip("./")
        output_path = input_dir / f"{subject_id}_0000.nii.gz"

        try:
            files = dicom_series_files(sitk, dicom_dir)
            if not files:
                raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(files)
            image = reader.Execute()
            sitk.WriteImage(image, str(output_path))

            converted.append(
                {
                    "case_id": subject_id,
                    "series_description": row["Series Description"],
                    "number_of_images_metadata": row.get("Number of Images"),
                    "dicom_files_read": len(files),
                    "dicom_dir": str(dicom_dir),
                    "output_path": str(output_path),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "case_id": subject_id,
                    "series_description": row.get("Series Description"),
                    "dicom_dir": str(dicom_dir),
                    "error": str(exc),
                }
            )

    report = {
        "config": str(config_path),
        "metadata": str(metadata_path),
        "series_description": args.series_description,
        "summary": {
            "metadata_rows": len(rows),
            "matching_subjects": len(seen_subjects),
            "converted": len(converted),
            "failures": len(failures),
            "input_dir": str(input_dir),
        },
        "converted": converted,
        "failures": failures,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote nnU-Net input images: {input_dir}")
    print(f"Wrote preparation report: {report_path}")
    print(f"Summary: {report['summary']}")
    return 0 if converted else 1


if __name__ == "__main__":
    raise SystemExit(main())
