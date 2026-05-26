#!/usr/bin/env python
"""Prepare external PROSTATE-MRI DICOM series for nnU-Net inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PROSTATE-MRI DICOM series to nnU-Net imagesTs.")
    parser.add_argument(
        "--config",
        default="nnunet_autosegmentation/config/picai_gland_lesion_nnunet_config.json",
        help="Autosegmentation config JSON.",
    )
    parser.add_argument(
        "--report",
        default="nnunet_autosegmentation/outputs/reports/prepare_nnunet_dataset_report.json",
        help="Output preparation report JSON.",
    )
    parser.add_argument(
        "--series-description",
        default="",
        help="Override T2W DICOM Series Description. Defaults to config external_source.series.t2w.",
    )
    parser.add_argument("--adc-series-description", default="", help="Optional ADC Series Description.")
    parser.add_argument(
        "--hbv-series-description",
        default="",
        help="Override HBV/DWI Series Description. Defaults to config external_source.series.hbv.",
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
        "channel_names": {"0": "T2W", "1": "ADC", "2": "HBV"},
        "labels": {"background": 0, "prostate_gland": 1, "cspca_lesion": 2},
        "numTraining": 0,
        "file_ending": ".nii.gz",
    }
    (dataset_root / "dataset.json").write_text(json.dumps(payload, indent=2) + "\n")


def index_rows_by_subject_and_series(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    index: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        subject = row.get("Subject ID", "")
        series = row.get("Series Description", "")
        if not subject or not series:
            continue
        index.setdefault(subject, {})
        index[subject].setdefault(series, row)
    return index


def read_series_image(sitk, metadata_root: Path, row: dict[str, str]):
    dicom_dir = metadata_root / row["File Location"].lstrip("./")
    files = dicom_series_files(sitk, dicom_dir)
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    return reader.Execute(), str(dicom_dir), len(files)


def resample_to_reference(sitk, moving, reference, interpolator):
    return sitk.Resample(moving, reference, sitk.Transform(), interpolator, 0, moving.GetPixelID())


def zero_like_reference(sitk, reference):
    image = sitk.Image(reference.GetSize(), reference.GetPixelID())
    image.CopyInformation(reference)
    return image


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
    series_config = config.get("external_source", {}).get("series", {})
    t2w_series = args.series_description or series_config.get("t2w") or "T2 TSE ax hi"
    adc_series = args.adc_series_description or series_config.get("adc", "")
    hbv_series = args.hbv_series_description or series_config.get("hbv") or "SShDWI FAST"

    row_index = index_rows_by_subject_and_series(rows)
    converted = []
    failures = []
    subject_ids = sorted(subject for subject, series_rows in row_index.items() if t2w_series in series_rows)
    if args.limit:
        subject_ids = subject_ids[: args.limit]

    for subject_id in subject_ids:
        if args.limit and len(converted) >= args.limit:
            break

        try:
            subject_rows = row_index[subject_id]
            t2w, t2w_dir, t2w_file_count = read_series_image(sitk, metadata_root, subject_rows[t2w_series])
            sitk.WriteImage(t2w, str(input_dir / f"{subject_id}_0000.nii.gz"))

            adc_source = "zero_filled_missing_adc"
            adc_dir = ""
            adc_file_count = 0
            if adc_series and adc_series in subject_rows:
                adc, adc_dir, adc_file_count = read_series_image(sitk, metadata_root, subject_rows[adc_series])
                adc = resample_to_reference(sitk, adc, t2w, sitk.sitkLinear)
                adc_source = adc_series
            else:
                adc = zero_like_reference(sitk, t2w)
            sitk.WriteImage(adc, str(input_dir / f"{subject_id}_0001.nii.gz"))

            if hbv_series not in subject_rows:
                raise FileNotFoundError(f"Missing HBV/DWI series '{hbv_series}' for {subject_id}")
            hbv, hbv_dir, hbv_file_count = read_series_image(sitk, metadata_root, subject_rows[hbv_series])
            hbv = resample_to_reference(sitk, hbv, t2w, sitk.sitkLinear)
            sitk.WriteImage(hbv, str(input_dir / f"{subject_id}_0002.nii.gz"))

            converted.append(
                {
                    "case_id": subject_id,
                    "channels": {
                        "0000": {
                            "modality": "t2w",
                            "series_description": t2w_series,
                            "dicom_dir": t2w_dir,
                            "dicom_files_read": t2w_file_count,
                        },
                        "0001": {
                            "modality": "adc",
                            "series_description": adc_source,
                            "dicom_dir": adc_dir,
                            "dicom_files_read": adc_file_count,
                        },
                        "0002": {
                            "modality": "hbv",
                            "series_description": hbv_series,
                            "dicom_dir": hbv_dir,
                            "dicom_files_read": hbv_file_count,
                        },
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "case_id": subject_id,
                    "error": str(exc),
                }
            )

    report = {
        "config": str(config_path),
        "metadata": str(metadata_path),
        "series_descriptions": {
            "t2w": t2w_series,
            "adc": adc_series or "zero_filled_missing_adc",
            "hbv": hbv_series,
        },
        "summary": {
            "metadata_rows": len(rows),
            "matching_subjects": len(subject_ids),
            "converted": len(converted),
            "failures": len(failures),
            "adc_zero_filled_cases": sum(
                1
                for row in converted
                if row["channels"]["0001"]["series_description"] == "zero_filled_missing_adc"
            ),
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
