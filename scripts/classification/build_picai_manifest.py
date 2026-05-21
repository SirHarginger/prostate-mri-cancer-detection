#!/usr/bin/env python3
"""Build PI-CAI clinical manifest for classifier training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean PI-CAI clinical manifest from marksheet.csv"
    )
    parser.add_argument(
        "--marksheet",
        type=Path,
        required=True,
        help="Path to PI-CAI clinical_information/marksheet.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV manifest path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.marksheet)

    required = [
        "patient_id",
        "study_id",
        "patient_age",
        "psa",
        "psad",
        "prostate_volume",
        "histopath_type",
        "lesion_GS",
        "lesion_ISUP",
        "case_ISUP",
        "case_csPCa",
        "center",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    manifest = df[required].copy()

    manifest["patient_id"] = manifest["patient_id"].astype(str)
    manifest["study_id"] = manifest["study_id"].astype(str)

    manifest["case_cspca_binary"] = (
        manifest["case_csPCa"].astype(str).str.upper().map({"YES": 1, "NO": 0})
    )

    manifest["case_isup_int"] = pd.to_numeric(
        manifest["case_ISUP"], errors="coerce"
    ).astype("Int64")

    manifest["patient_age"] = pd.to_numeric(manifest["patient_age"], errors="coerce")
    manifest["psa"] = pd.to_numeric(manifest["psa"], errors="coerce")
    manifest["psad"] = pd.to_numeric(manifest["psad"], errors="coerce")
    manifest["prostate_volume"] = pd.to_numeric(
        manifest["prostate_volume"], errors="coerce"
    )

    manifest["case_key"] = (
        manifest["patient_id"].astype(str) + "_" + manifest["study_id"].astype(str)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    print(f"Saved manifest: {args.output}")
    print(f"Shape: {manifest.shape}")
    print("\ncase_cspca_binary counts:")
    print(manifest["case_cspca_binary"].value_counts(dropna=False))
    print("\ncase_isup_int counts:")
    print(manifest["case_isup_int"].value_counts(dropna=False).sort_index())
    print("\nMissing values:")
    print(manifest.isna().sum())


if __name__ == "__main__":
    main()
