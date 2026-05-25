"""CNN command handlers.

This module keeps the current model-training command path small and
config-driven. Older experimental CLI handlers remain available in cli.py until
they are deliberately retired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prostate_mri_cancer_detection.cnn import run_cnn_candidate_training


def run_cnn_candidate_command(args: argparse.Namespace) -> int:
    """Run publication-candidate CNN training from CLI arguments or config."""

    args = load_cnn_candidate_config(args)
    report = run_cnn_candidate_training(
        manifest_path=args.manifest,
        raw_root=args.raw_root,
        embeddings_path=args.embeddings,
        predictions_path=args.predictions,
        report_path=args.report,
        model_path=args.model,
        architecture=args.architecture,
        tensor_mode=args.tensor_mode,
        sample_size_per_split=args.sample_size_per_split,
        image_size=args.image_size,
        slice_window=args.slice_window,
        volume_depth=args.volume_depth,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        early_stopping_patience=args.early_stopping_patience,
        embedding_dim=args.embedding_dim,
        augment_train=args.augment_train,
        all_cases=args.all_cases,
        case_ids=args.case_id,
        target_sensitivity=args.target_sensitivity,
        seed=args.seed,
        device_name=args.device,
    )

    print(f"Wrote CNN candidate embeddings: {args.embeddings}")
    print(f"Wrote CNN candidate predictions: {args.predictions}")
    print(f"Wrote CNN candidate report: {args.report}")
    print(f"Wrote CNN candidate model: {args.model}")
    print(f"Summary: {report['summary']}")
    print(f"Case counts: {report['case_counts']}")
    print(f"Label counts: {report['label_counts']}")
    print(f"Best epoch: {report['model']['best_epoch']}")
    for split, payload in report["metrics"].items():
        metrics = payload["metrics"]
        print(f"{split}: n={metrics['n']} auc={metrics['roc_auc']} sens={metrics['sensitivity']} spec={metrics['specificity']}")
    fixed = report["validation_selected_threshold"]["test"]
    print(f"Validation-selected test threshold: status={fixed['status']} metrics={fixed.get('metrics')}")
    return 0


def load_cnn_candidate_config(args: argparse.Namespace) -> argparse.Namespace:
    """Apply an optional JSON config to CNN candidate args."""

    config_path = getattr(args, "config", None)
    if config_path is None:
        return args

    config = read_json_config(config_path)
    path_keys = {"manifest", "raw_root", "embeddings", "predictions", "report", "model"}
    for key, value in config.items():
        if key == "outputs":
            for output_key, output_value in value.items():
                setattr(args, output_key, Path(output_value))
            continue
        if key in path_keys:
            setattr(args, key, Path(value))
        else:
            setattr(args, key, value)
    return args


def read_json_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON config file."""

    with Path(path).open("r", encoding="utf-8") as config_file:
        return json.load(config_file)
