#!/usr/bin/env python
"""Collect nnU-Net training logs, metrics, and figures into ignored outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

EPOCH_RE = re.compile(r"Epoch (?P<epoch>\d+)$")
LR_RE = re.compile(r"Current learning rate: (?P<value>[-+0-9.eE]+)")
TRAIN_LOSS_RE = re.compile(r"train_loss (?P<value>[-+0-9.eE]+)")
VAL_LOSS_RE = re.compile(r"val_loss (?P<value>[-+0-9.eE]+)")
PSEUDO_DICE_RE = re.compile(r"Pseudo dice (?P<value>\[.*\])")
EPOCH_TIME_RE = re.compile(r"Epoch time: (?P<value>[-+0-9.eE]+) s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect nnU-Net training artifacts for reporting.")
    parser.add_argument(
        "--config",
        default="nnunet_autosegmentation/config/picai_gland_lesion_nnunet_config.json",
        type=Path,
    )
    parser.add_argument("--fold", default="0")
    parser.add_argument("--configuration", default="", help="Defaults to config nnunet.configuration.")
    parser.add_argument("--trainer", default="", help="Defaults to config nnunet.trainer.")
    parser.add_argument("--output-root", default="nnunet_autosegmentation/outputs", type=Path)
    return parser.parse_args()


def model_dir(config: dict[str, Any], configuration: str, trainer: str) -> Path:
    nnunet = config["nnunet"]
    return (
        Path(nnunet["results"])
        / f"Dataset{nnunet['dataset_id']}_{nnunet['dataset_label']}"
        / f"{trainer}__nnUNetPlans__{configuration}"
    )


def parse_float(value: str) -> float:
    return float(value.strip())


def parse_pseudo_dice(value: str) -> list[float]:
    cleaned = value.replace("np.float32(", "").replace(")", "")
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    return [float(item) for item in parsed]


def parse_training_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split(": ", 1)[-1].strip()
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            if current is not None:
                rows.append(current)
            current = {"epoch": int(epoch_match.group("epoch"))}
            continue
        if current is None:
            continue

        for key, pattern in (
            ("learning_rate", LR_RE),
            ("train_loss", TRAIN_LOSS_RE),
            ("val_loss", VAL_LOSS_RE),
            ("epoch_time_seconds", EPOCH_TIME_RE),
        ):
            match = pattern.search(line)
            if match:
                current[key] = parse_float(match.group("value"))

        dice_match = PSEUDO_DICE_RE.search(line)
        if dice_match:
            dice_values = parse_pseudo_dice(dice_match.group("value"))
            current["pseudo_dice"] = dice_values
            if dice_values:
                current["pseudo_dice_mean"] = sum(dice_values) / len(dice_values)
                if len(dice_values) > 0:
                    current["pseudo_dice_prostate_gland"] = dice_values[0]
                if len(dice_values) > 1:
                    current["pseudo_dice_cspca_lesion"] = dice_values[1]

    if current is not None:
        rows.append(current)
    return rows


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "val_loss",
        "pseudo_dice_mean",
        "pseudo_dice_prostate_gland",
        "pseudo_dice_cspca_lesion",
        "epoch_time_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def maybe_write_figures(figure_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    if not rows:
        return []

    figure_paths: list[str] = []
    epochs = [row["epoch"] for row in rows]

    loss_path = figure_dir / "nnunet_training_loss.png"
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row.get("train_loss") for row in rows], marker="o", label="train_loss")
    plt.plot(epochs, [row.get("val_loss") for row in rows], marker="o", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("nnU-Net Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=160)
    plt.close()
    figure_paths.append(str(loss_path))

    dice_path = figure_dir / "nnunet_pseudo_dice.png"
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row.get("pseudo_dice_prostate_gland") for row in rows], marker="o", label="prostate_gland")
    plt.plot(epochs, [row.get("pseudo_dice_cspca_lesion") for row in rows], marker="o", label="cspca_lesion")
    plt.plot(epochs, [row.get("pseudo_dice_mean") for row in rows], marker="o", label="mean")
    plt.xlabel("Epoch")
    plt.ylabel("Pseudo Dice")
    plt.title("nnU-Net Pseudo Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dice_path, dpi=160)
    plt.close()
    figure_paths.append(str(dice_path))

    return figure_paths


def copy_existing_figures(source_dir: Path, figure_dir: Path) -> list[str]:
    copied = []
    figure_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.glob("*.png")):
        target = figure_dir / path.name
        shutil.copy2(path, target)
        copied.append(str(target))
    return copied


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# nnU-Net Training Smoke Report",
        "",
        "This report summarizes a CPU smoke training run. It is not a final segmentation model.",
        "",
        "## Summary",
        "",
        f"- Model directory: `{report['model_dir']}`",
        f"- Fold: `{report['fold']}`",
        f"- Epochs parsed: {summary['epochs_parsed']}",
        f"- Last epoch: {summary.get('last_epoch', '')}",
        f"- Mean epoch time seconds: {summary.get('mean_epoch_time_seconds', '')}",
        f"- Estimated 1000-epoch CPU time hours: {summary.get('estimated_1000_epoch_hours', '')}",
        f"- Best mean pseudo Dice: {summary.get('best_pseudo_dice_mean', '')}",
        f"- Best lesion pseudo Dice: {summary.get('best_pseudo_dice_cspca_lesion', '')}",
        "",
        "## Compute Note",
        "",
        "Default nnU-Net training on CPU is expected to be slow. Use this run as a pipeline smoke test only; publication-grade training should use GPU or an explicitly documented reduced-epoch trainer.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"epochs_parsed": 0}
    epoch_times = [float(row["epoch_time_seconds"]) for row in rows if "epoch_time_seconds" in row]
    mean_time = sum(epoch_times) / len(epoch_times) if epoch_times else None
    mean_dice_rows = [row for row in rows if "pseudo_dice_mean" in row]
    lesion_dice_rows = [row for row in rows if "pseudo_dice_cspca_lesion" in row]
    return {
        "epochs_parsed": len(rows),
        "last_epoch": rows[-1].get("epoch"),
        "mean_epoch_time_seconds": mean_time,
        "estimated_1000_epoch_hours": (mean_time * 1000 / 3600) if mean_time else None,
        "best_pseudo_dice_mean": max((row["pseudo_dice_mean"] for row in mean_dice_rows), default=None),
        "best_pseudo_dice_cspca_lesion": max((row["pseudo_dice_cspca_lesion"] for row in lesion_dice_rows), default=None),
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    configuration = args.configuration or config["nnunet"]["configuration"]
    trainer = args.trainer or config["nnunet"]["trainer"]
    source_dir = model_dir(config, configuration, trainer)
    fold_dir = source_dir / f"fold_{args.fold}"

    log_candidates = sorted(fold_dir.glob("training_log*.txt")) + sorted(source_dir.glob("training_log*.txt"))
    rows: list[dict[str, Any]] = []
    log_path = None
    if log_candidates:
        log_path = log_candidates[-1]
        rows = parse_training_log(log_path)

    report_dir = args.output_root / "reports"
    figure_dir = args.output_root / "figures" / f"Dataset{config['nnunet']['dataset_id']}_{configuration}_fold{args.fold}"
    metrics_csv = report_dir / f"nnunet_training_metrics_Dataset{config['nnunet']['dataset_id']}_{configuration}_fold{args.fold}.csv"
    report_json = report_dir / f"nnunet_training_report_Dataset{config['nnunet']['dataset_id']}_{configuration}_fold{args.fold}.json"
    report_md = report_dir / f"nnunet_training_report_Dataset{config['nnunet']['dataset_id']}_{configuration}_fold{args.fold}.md"

    write_metrics_csv(metrics_csv, rows)
    figures = maybe_write_figures(figure_dir, rows)
    copied_figures = copy_existing_figures(fold_dir if fold_dir.exists() else source_dir, figure_dir)

    checkpoint_paths = [
        str(path)
        for path in sorted(fold_dir.glob("checkpoint*.pth"))
    ]
    report = {
        "stage": "collect_nnunet_training_artifacts",
        "config": str(args.config),
        "model_dir": str(source_dir),
        "fold_dir": str(fold_dir),
        "fold": args.fold,
        "configuration": configuration,
        "trainer": trainer,
        "training_log": str(log_path) if log_path else "",
        "metrics_csv": str(metrics_csv),
        "figures": figures,
        "copied_figures": copied_figures,
        "checkpoints_found": checkpoint_paths,
        "summary": summarize(rows),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report_md, report)

    print(f"Wrote metrics CSV: {metrics_csv}")
    print(f"Wrote report JSON: {report_json}")
    print(f"Wrote report Markdown: {report_md}")
    print(f"Figures: {figures + copied_figures}")
    print(f"Summary: {report['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
