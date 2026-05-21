#!/usr/bin/env python3
"""Build PI-CAI classifier mask manifest.

Combines the clinical manifest with available lesion, whole-gland, and zonal mask paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PI-CAI mask manifest.")
    parser.add_argument("--clinical-manifest", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def case_key_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name.replace(".nii.gz", "")
    if name.endswith(".nii"):
        return name.replace(".nii", "")
    return path.stem


def index_masks(root: Path) -> dict[str, Path]:
    paths = sorted(root.rglob("*.nii.gz"))
    indexed: dict[str, Path] = {}

    for path in paths:
        key = case_key_from_path(path)

        # If duplicates exist, keep the first sorted path for deterministic behavior.
        # Later we can make this source-specific.
        indexed.setdefault(key, path)

    return indexed


def main() -> None:
    args = parse_args()

    clinical = pd.read_csv(args.clinical_manifest)
    clinical["patient_id"] = clinical["patient_id"].astype(str)
    clinical["study_id"] = clinical["study_id"].astype(str)
    clinical["case_key"] = clinical["patient_id"] + "_" + clinical["study_id"]

    label_root = args.label_root

    lesion_root = label_root / "csPCa_lesion_delineations" / "human_expert"
    whole_gland_root = label_root / "anatomical_delineations" / "whole_gland" / "AI"
    zonal_root = label_root / "anatomical_delineations" / "zonal_pz_tz" / "AI"

    lesion_masks = index_masks(lesion_root)
    whole_gland_masks = index_masks(whole_gland_root)
    zonal_masks = index_masks(zonal_root)

    clinical["lesion_mask_path"] = clinical["case_key"].map(
        lambda key: str(lesion_masks.get(key, ""))
    )
    clinical["whole_gland_mask_path"] = clinical["case_key"].map(
        lambda key: str(whole_gland_masks.get(key, ""))
    )
    clinical["zonal_mask_path"] = clinical["case_key"].map(
        lambda key: str(zonal_masks.get(key, ""))
    )

    clinical["has_lesion_mask"] = clinical["lesion_mask_path"].ne("")
    clinical["has_whole_gland_mask"] = clinical["whole_gland_mask_path"].ne("")
    clinical["has_zonal_mask"] = clinical["zonal_mask_path"].ne("")

    clinical["lesion_mask_source"] = "human_expert"
    clinical["whole_gland_mask_source"] = "AI"
    clinical["zonal_mask_source"] = "AI"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clinical.to_csv(args.output, index=False)

    print(f"Saved mask manifest: {args.output}")
    print(f"Shape: {clinical.shape}")
    print("\nMask availability:")
    print("lesion:", clinical["has_lesion_mask"].sum())
    print("whole_gland:", clinical["has_whole_gland_mask"].sum())
    print("zonal:", clinical["has_zonal_mask"].sum())

    print("\nLabel counts:")
    print(clinical["case_cspca_binary"].value_counts(dropna=False))

    missing = clinical[
        ~clinical["has_lesion_mask"]
        | ~clinical["has_whole_gland_mask"]
        | ~clinical["has_zonal_mask"]
    ]

    if not missing.empty:
        print("\nMissing mask cases:")
        print(missing[["case_key", "has_lesion_mask", "has_whole_gland_mask", "has_zonal_mask"]].head(20))


if __name__ == "__main__":
    main()
