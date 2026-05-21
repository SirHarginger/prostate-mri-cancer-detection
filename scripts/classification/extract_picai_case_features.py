#!/usr/bin/env python3
"""Extract leakage-safe PI-CAI case-level classifier features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    sitk = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional progress sugar.

    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


CLINICAL_FEATURE_COLUMNS = [
    "patient_age",
    "psa",
    "psad",
    "prostate_volume",
    "center",
]

# These columns are kept for supervised training targets and case metadata.
# Do not use them as predictor inputs for binary csPCa classification.
LABEL_COLUMNS = [
    "case_cspca_binary",
    "case_isup_int",
]

OUTPUT_CASE_COLUMNS = [
    *CLINICAL_FEATURE_COLUMNS,
    *LABEL_COLUMNS,
]

IDENTIFIER_COLUMNS = [
    "case_key",
    "patient_id",
    "study_id",
]

REQUIRED_COLUMNS = [
    "has_core_bpMRI",
    "t2w_image_path",
    "whole_gland_mask_path",
    *OUTPUT_CASE_COLUMNS,
]

RADIOMICS_PREFIX = "t2w_wholegland_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract leakage-safe PI-CAI case-level clinical and T2W "
            "whole-gland radiomics features."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--radiomics-config", type=Path, default=None)
    return parser.parse_args()


def require_sitk():
    if sitk is None:
        raise ImportError(
            "SimpleITK is required for feature extraction. Install SimpleITK "
            "in the active environment."
        )
    return sitk


def create_radiomics_extractor(config_path: Path | None = None) -> Any:
    try:
        from radiomics import featureextractor
    except ImportError as exc:
        raise ImportError(
            "PyRadiomics is required for feature extraction. Install pyradiomics "
            "in the active environment."
        ) from exc

    if config_path is not None:
        if not config_path.is_file():
            raise FileNotFoundError(f"Radiomics config does not exist: {config_path}")
        return featureextractor.RadiomicsFeatureExtractor(str(config_path))
    return featureextractor.RadiomicsFeatureExtractor()


def validate_manifest_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required manifest columns: {missing}")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def read_core_manifest(manifest_path: Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    validate_manifest_columns(df)

    df = df[bool_series(df["has_core_bpMRI"])].copy()

    if limit is not None:
        df = df.head(limit)

    return df.reset_index(drop=True)


def images_have_same_geometry(image: Any, mask: Any, tolerance: float = 1e-5) -> bool:
    return (
        image.GetSize() == mask.GetSize()
        and np.allclose(image.GetSpacing(), mask.GetSpacing(), atol=tolerance, rtol=0)
        and np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=tolerance, rtol=0)
        and np.allclose(image.GetDirection(), mask.GetDirection(), atol=tolerance, rtol=0)
    )


def binarize_mask(mask: Any) -> Any:
    sitk_module = require_sitk()
    return sitk_module.Cast(mask > 0, sitk_module.sitkUInt8)


def resample_mask_to_image(mask: Any, reference_image: Any) -> Any:
    sitk_module = require_sitk()
    mask = binarize_mask(mask)

    resampled = sitk_module.Resample(
        mask,
        reference_image,
        sitk_module.Transform(),
        sitk_module.sitkNearestNeighbor,
        0,
        sitk_module.sitkUInt8,
    )
    resampled = binarize_mask(resampled)
    resampled.CopyInformation(reference_image)
    return resampled


def nonzero_voxel_count(mask: Any) -> int:
    sitk_module = require_sitk()
    return int(np.count_nonzero(sitk_module.GetArrayViewFromImage(mask)))


def require_existing_path(value: Any, column: str) -> Path:
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError(f"Missing path in column {column}")

    path = Path(str(value))
    if not path.is_file():
        raise FileNotFoundError(f"{column} does not exist: {path}")

    return path


def to_python_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()

    return value


def extract_t2w_wholegland_radiomics(row: pd.Series, extractor: Any) -> dict[str, Any]:
    sitk_module = require_sitk()

    image_path = require_existing_path(row["t2w_image_path"], "t2w_image_path")
    mask_path = require_existing_path(
        row["whole_gland_mask_path"], "whole_gland_mask_path"
    )

    image = sitk_module.ReadImage(str(image_path))
    mask = sitk_module.ReadImage(str(mask_path))
    mask = resample_mask_to_image(mask, image)

    if nonzero_voxel_count(mask) == 0:
        raise ValueError("Whole-gland mask is empty after resampling to T2W geometry")

    return format_radiomics_result(extractor.execute(image, mask, label=1))


def format_radiomics_result(result: dict[Any, Any]) -> dict[str, Any]:
    features: dict[str, Any] = {}

    for key, value in result.items():
        key = str(key)
        if key.startswith("diagnostics_"):
            continue
        features[f"{RADIOMICS_PREFIX}{key}"] = to_python_value(value)

    return features


def base_output_row(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}

    for col in IDENTIFIER_COLUMNS:
        if col in row.index:
            out[col] = row[col]

    for col in OUTPUT_CASE_COLUMNS:
        out[col] = row[col]

    out["feature_error"] = ""
    return out


def extract_features(df: pd.DataFrame, extractor: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # Lesion radiomics are intentionally excluded for binary csPCa classification:
    # in PI-CAI fold0, empty/non-empty lesion masks perfectly correlate with the
    # case label, so lesion-mask features would leak the target.
    for _, row in tqdm(df.iterrows(), total=len(df)):
        out = base_output_row(row)

        try:
            out.update(extract_t2w_wholegland_radiomics(row, extractor))
        except Exception as exc:
            out["feature_error"] = str(exc)

        rows.append(out)

    return pd.DataFrame(rows)


def print_summary(features: pd.DataFrame, output_path: Path) -> None:
    print(f"Saved case features: {output_path}")
    print(f"Shape: {features.shape}")

    print("\ncase_cspca_binary counts:")
    print(features["case_cspca_binary"].value_counts(dropna=False))

    error_mask = features["feature_error"].astype(str).str.len() > 0
    print("\nFeature errors:")
    print(int(error_mask.sum()))

    if error_mask.any():
        print("\nTop feature errors:")
        print(features.loc[error_mask, "feature_error"].value_counts().head(10))


def main() -> None:
    args = parse_args()

    df = read_core_manifest(args.manifest, args.limit)
    extractor = create_radiomics_extractor(args.radiomics_config)
    features = extract_features(df, extractor)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)

    print_summary(features, args.output)


if __name__ == "__main__":
    main()
