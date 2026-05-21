#!/usr/bin/env python3
"""Audit PI-CAI fold0 readiness for safe deep-learning classifier prototyping."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


CORE_MODALITIES = ["t2w", "adc", "hbv"]
REQUIRED_COLUMNS = [
    "has_core_bpMRI",
    "case_cspca_binary",
    "t2w_image_path",
    "adc_image_path",
    "hbv_image_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit PI-CAI fold0 image manifest readiness for a leakage-safe "
            "deep-learning classifier prototype. This does not train a model."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def optional_module_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return {
            "installed": False,
            "version": None,
            "error": str(exc),
        }

    return {
        "installed": True,
        "version": str(getattr(module, "__version__", "unknown")),
        "error": "",
    }


def torch_status() -> dict[str, Any]:
    status = optional_module_status("torch")
    if not status["installed"]:
        status.update(
            {
                "cuda_available": False,
                "cuda_device_count": 0,
                "cuda_devices": [],
            }
        )
        return status

    torch = importlib.import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices = [
        str(torch.cuda.get_device_name(idx))
        for idx in range(device_count)
    ]
    status.update(
        {
            "cuda_available": cuda_available,
            "cuda_device_count": device_count,
            "cuda_devices": devices,
        }
    )
    return status


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required manifest columns: {missing}")


def read_core_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    df = pd.read_csv(manifest_path)
    validate_columns(df)
    return df[bool_series(df["has_core_bpMRI"])].copy().reset_index(drop=True)


def path_present(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def modality_availability(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    availability: dict[str, dict[str, int]] = {}

    for modality in CORE_MODALITIES:
        column = f"{modality}_image_path"
        present = path_present(df[column])
        exists = df.loc[present, column].map(lambda value: Path(str(value)).is_file())
        availability[modality] = {
            "path_present": int(present.sum()),
            "path_exists": int(exists.sum()),
        }

    return availability


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    target = pd.to_numeric(df["case_cspca_binary"], errors="coerce")
    return {
        str(key): int(value)
        for key, value in target.value_counts(dropna=False).sort_index().items()
    }


def image_info(path: str, sitk_module: Any) -> dict[str, Any]:
    image = sitk_module.ReadImage(str(path))
    return {
        "size": [int(value) for value in image.GetSize()],
        "spacing": [round(float(value), 6) for value in image.GetSpacing()],
        "origin": [round(float(value), 6) for value in image.GetOrigin()],
        "direction": [round(float(value), 6) for value in image.GetDirection()],
    }


def inspect_image_geometry(df: pd.DataFrame, limit: int | None) -> dict[str, Any]:
    sitk_status = optional_module_status("SimpleITK")
    if not sitk_status["installed"]:
        return {
            "enabled": False,
            "reason": "SimpleITK is not installed.",
            "cases_requested": int(limit or 0),
            "cases": [],
        }

    sitk = importlib.import_module("SimpleITK")
    sample_count = 5 if limit is None else max(0, int(limit))
    sample = df.head(sample_count)
    cases: list[dict[str, Any]] = []

    for _, row in sample.iterrows():
        case_key = str(row.get("case_key", "unknown"))
        case: dict[str, Any] = {
            "case_key": case_key,
            "case_cspca_binary": to_json_value(row["case_cspca_binary"]),
            "modalities": {},
            "error": "",
        }

        for modality in CORE_MODALITIES:
            column = f"{modality}_image_path"
            path = str(row[column])
            try:
                case["modalities"][modality] = image_info(path, sitk)
            except Exception as exc:
                case["modalities"][modality] = {
                    "path": path,
                    "error": str(exc),
                }

        cases.append(case)

    return {
        "enabled": True,
        "cases_requested": int(sample_count),
        "cases": cases,
    }


def to_json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def build_readiness_summary(manifest_path: Path, limit: int | None = None) -> dict[str, Any]:
    core = read_core_manifest(manifest_path)
    dependencies = {
        "SimpleITK": optional_module_status("SimpleITK"),
        "torch": torch_status(),
        "monai": optional_module_status("monai"),
    }
    all_core_paths_present = all(
        availability["path_present"] == len(core)
        for availability in modality_availability(core).values()
    )

    return {
        "manifest": str(manifest_path),
        "total_core_bpmri_cases": int(len(core)),
        "label_counts": label_counts(core),
        "modality_availability": modality_availability(core),
        "dependencies": dependencies,
        "image_geometry_sample": inspect_image_geometry(core, limit),
        "readiness_verdict": {
            "ready_for_small_fold0_prototype": bool(
                len(core) > 0 and all_core_paths_present
            ),
            "ready_for_serious_training": False,
            "reason": (
                "Fold0 is sufficient for a prototype audit, but serious "
                "deep-learning training should wait for folds 1-4, validated "
                "preprocessing, and confirmed GPU/PyTorch/MONAI readiness."
            ),
        },
        "safety_notes": [
            "Do not use lesion masks, lesion crops, or lesion-derived features "
            "for binary csPCa detection.",
            "Use whole-gland masks only for anatomical cropping/localization.",
            "Do not save generated image arrays inside Git.",
            "This audit does not train a model.",
        ],
    }


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def print_summary(summary: dict[str, Any]) -> None:
    verdict = summary["readiness_verdict"]

    print(f"Manifest: {summary['manifest']}")
    print(f"Core bpMRI cases: {summary['total_core_bpmri_cases']}")
    print(f"Label counts: {summary['label_counts']}")
    print(f"Modality availability: {summary['modality_availability']}")
    print("\nDependency status:")
    for name, status in summary["dependencies"].items():
        print(f"{name}: installed={status['installed']}, version={status['version']}")
        if name == "torch":
            print(
                "torch CUDA: "
                f"available={status['cuda_available']}, "
                f"device_count={status['cuda_device_count']}, "
                f"devices={status['cuda_devices']}"
            )

    print("\nReadiness verdict:")
    print(f"ready_for_small_fold0_prototype={verdict['ready_for_small_fold0_prototype']}")
    print(f"ready_for_serious_training={verdict['ready_for_serious_training']}")
    print(verdict["reason"])


def main() -> None:
    args = parse_args()
    summary = build_readiness_summary(args.manifest, args.limit)

    if args.output is not None:
        write_summary(summary, args.output)
        print(f"Saved audit summary: {args.output}")

    print_summary(summary)


if __name__ == "__main__":
    main()
