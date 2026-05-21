#!/usr/bin/env python3
"""Check geometry compatibility for PI-CAI images and masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


CHECK_COLUMNS = [
    "t2w_image_path",
    "adc_image_path",
    "hbv_image_path",
    "lesion_mask_path",
    "whole_gland_mask_path",
    "zonal_mask_path",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def image_info(path: str) -> dict:
    img = sitk.ReadImage(path)
    return {
        "size": img.GetSize(),
        "spacing": tuple(round(x, 5) for x in img.GetSpacing()),
        "origin": tuple(round(x, 5) for x in img.GetOrigin()),
        "direction": tuple(round(x, 5) for x in img.GetDirection()),
    }


def same_geometry(a: dict, b: dict) -> bool:
    return (
        a["size"] == b["size"]
        and a["spacing"] == b["spacing"]
        and a["origin"] == b["origin"]
        and a["direction"] == b["direction"]
    )


def main():
    args = parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["has_core_bpMRI"]].copy()

    if args.limit:
        df = df.head(args.limit)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        case_key = row["case_key"]

        result = {
            "case_key": case_key,
            "geometry_ok": True,
            "error": "",
        }

        try:
            infos = {col: image_info(row[col]) for col in CHECK_COLUMNS}

            ref = infos["t2w_image_path"]

            for col in CHECK_COLUMNS:
                result[f"{col}_size"] = str(infos[col]["size"])
                result[f"{col}_spacing"] = str(infos[col]["spacing"])
                result[f"{col}_matches_t2w"] = same_geometry(ref, infos[col])

            result["geometry_ok"] = all(
                result[f"{col}_matches_t2w"] for col in CHECK_COLUMNS
            )

        except Exception as exc:
            result["geometry_ok"] = False
            result["error"] = str(exc)

        rows.append(result)

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(out["geometry_ok"].value_counts(dropna=False))

    if not out["geometry_ok"].all():
        print("\nFailed examples:")
        print(out[~out["geometry_ok"]][["case_key", "error"]].head(20))


if __name__ == "__main__":
    main()
