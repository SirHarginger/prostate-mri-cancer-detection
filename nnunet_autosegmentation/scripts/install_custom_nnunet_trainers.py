#!/usr/bin/env python
"""Install small custom nnU-Net trainer variants into the active environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install a custom fixed-epoch nnU-Net trainer.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--class-name", default="", help="Defaults to nnUNetTrainer_<epochs>epochs.")
    parser.add_argument(
        "--report",
        default="nnunet_autosegmentation/outputs/reports/custom_nnunet_trainer_install_report.json",
        type=Path,
    )
    return parser.parse_args()


def trainer_source(class_name: str, epochs: int) -> str:
    return f'''"""Custom fixed-length trainer installed by the prostate MRI project."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class {class_name}(nnUNetTrainer):
    """nnU-Net trainer capped at {epochs} epochs for CPU smoke training."""

    def initialize(self):
        self.num_epochs = {epochs}
        return super().initialize()
'''


def install_trainer(class_name: str, epochs: int) -> Path:
    try:
        import nnunetv2
    except ImportError as exc:
        raise SystemExit("nnunetv2 is not installed in the active Python environment.") from exc

    package_root = Path(nnunetv2.__file__).resolve().parent
    trainer_dir = package_root / "training" / "nnUNetTrainer" / "variants" / "training_length"
    trainer_dir.mkdir(parents=True, exist_ok=True)
    init_path = trainer_dir / "__init__.py"
    init_path.touch(exist_ok=True)
    trainer_path = trainer_dir / f"{class_name}.py"
    trainer_path.write_text(trainer_source(class_name, epochs), encoding="utf-8")
    return trainer_path


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    class_name = args.class_name or f"nnUNetTrainer_{args.epochs}epochs"
    trainer_path = install_trainer(class_name, args.epochs)
    report = {
        "stage": "install_custom_nnunet_trainer",
        "class_name": class_name,
        "epochs": args.epochs,
        "trainer_path": str(trainer_path),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Installed {class_name}: {trainer_path}")
    print(f"Wrote report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
