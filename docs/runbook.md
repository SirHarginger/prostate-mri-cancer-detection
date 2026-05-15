# Runbook

Operational notes for running project workflows.

## Scaffold Commands

The current scripts are CLI stubs. They define the planned interface and fail
clearly until implemented.

Examples:

```bash
python scripts/build_manifest.py --dataset prostate_mri --input-dir data/raw --output data/manifests/prostate_mri_manifest.csv --dry-run
python scripts/verify_dataset.py --manifest data/manifests/prostate_mri_manifest.csv --dry-run
python scripts/preprocess_dataset.py --config configs/preprocess/example.yaml --dry-run
python scripts/train_segmentation.py --config configs/train/example.yaml --dry-run
python scripts/evaluate_segmentation.py --config configs/eval/example.yaml --dry-run
python scripts/visualize_qc.py --manifest data/manifests/prostate_mri_manifest.csv --output-dir outputs/figures/qc --dry-run
```

