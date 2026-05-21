#!/usr/bin/env python3
"""Build PI-CAI image manifest for downloaded image folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODALITIES = ["t2w", "adc", "hbv", "cor", "sag"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PI-CAI image manifest.")
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Root containing downloaded PI-CAI folds, e.g. .../images",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def case_key_from_image(path: Path, modality: str) -> str:
    suffix = f"_{modality}.mha"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected image filename for {modality}: {path.name}")
    return path.name.replace(suffix, "")


def index_modality(images_root: Path, modality: str) -> dict[str, Path]:
    paths = sorted(images_root.rglob(f"*_{modality}.mha"))
    indexed: dict[str, Path] = {}

    for path in paths:
        key = case_key_from_image(path, modality)
        indexed.setdefault(key, path)

    return indexed


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.mask_manifest)
    df["patient_id"] = df["patient_id"].astype(str)
    df["study_id"] = df["study_id"].astype(str)
    df["case_key"] = df["patient_id"] + "_" + df["study_id"]

    modality_indexes = {
        modality: index_modality(args.images_root, modality)
        for modality in MODALITIES
    }

    for modality in MODALITIES:
        col = f"{modality}_image_path"
        has_col = f"has_{modality}_image"

        df[col] = df["case_key"].map(
            lambda key: str(modality_indexes[modality].get(key, ""))
        )
        df[has_col] = df[col].ne("")

    core_modalities = ["t2w", "adc", "hbv"]
    df["has_core_bpMRI"] = df[[f"has_{m}_image" for m in core_modalities]].all(axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Saved image manifest: {args.output}")
    print(f"Shape: {df.shape}")

    print("\nDownloaded image counts by modality:")
    for modality in MODALITIES:
        print(f"{modality}: {len(modality_indexes[modality])}")

    print("\nAvailability in manifest:")
    for modality in MODALITIES:
        print(f"{modality}: {int(df[f'has_{modality}_image'].sum())}")

    print("\nCore bpMRI cases:")
    print(int(df["has_core_bpMRI"].sum()))

    print("\nLabel counts for downloaded core bpMRI cases:")
    print(df[df["has_core_bpMRI"]]["case_cspca_binary"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
