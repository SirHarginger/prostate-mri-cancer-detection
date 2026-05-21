#!/usr/bin/env python3
"""Check PI-CAI mask voxel contents before radiomics extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


MASK_COLUMNS = [
    "lesion_mask_path",
    "whole_gland_mask_path",
    "zonal_mask_path",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mask_stats(path: str) -> dict:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)

    return {
        "voxel_count_nonzero": int((arr > 0).sum()),
        "unique_values": ",".join(map(str, sorted(set(arr.flatten().tolist()))[:20])),
        "size": str(img.GetSize()),
        "spacing": str(tuple(round(x, 6) for x in img.GetSpacing())),
    }


def main():
    args = parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["has_core_bpMRI"]].copy()

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        out = {
            "case_key": row["case_key"],
            "case_cspca_binary": row["case_cspca_binary"],
        }

        for col in MASK_COLUMNS:
            try:
                stats = mask_stats(row[col])
                prefix = col.replace("_path", "")
                for k, v in stats.items():
                    out[f"{prefix}_{k}"] = v
            except Exception as exc:
                out[f"{col}_error"] = str(exc)

        rows.append(out)

    qc = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print("\nLesion nonzero summary:")
    print(qc["lesion_mask_voxel_count_nonzero"].describe())

    print("\nEmpty lesion masks by label:")
    qc["lesion_is_empty"] = qc["lesion_mask_voxel_count_nonzero"] == 0
    print(pd.crosstab(qc["case_cspca_binary"], qc["lesion_is_empty"]))


if __name__ == "__main__":
    main()
