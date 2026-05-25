# Production Stage F CNN Smoke Training

Production Stage F validates the real CNN data path before any expensive model
training. It trains a tiny multisequence CNN on a small balanced sample and
writes ignored predictions, embeddings, a checkpoint, and a provenance report.

This is a smoke test, not a final CNN baseline.

## Environment

The cluster environment must include NumPy, SimpleITK, and PyTorch.

```bash
python -m pip install numpy SimpleITK torch
```

## Command

Run from the cluster repository root:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli cnn-smoke-train \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --sample-size-per-split 12 \
  --image-size 64 \
  --max-epochs 1 \
  --batch-size 4 \
  --embedding-dim 32 \
  --augment-train \
  --embeddings data/features/cnn_smoke_embeddings.csv \
  --predictions outputs/reports/cnn_smoke_predictions.csv \
  --report outputs/reports/cnn_smoke_report.json \
  --model outputs/models/cnn_smoke_model.pt
```

Outputs are written under ignored project paths.

## Split Policy

- Training: `fold0`, `fold1`, `fold2`
- Validation: `fold3`
- Test: `fold4`

Training augmentation is optional and, when enabled, is applied only to training
rows. Validation and test rows are never augmented.

## Input Construction

For each selected case:

- T2W is used as the reference grid.
- ADC and high b-value DWI are resampled to the T2W grid in memory.
- A non-empty whole-gland mask on the T2W grid selects the axial slice and crop
  when available.
- Each sequence is normalized per case and per sequence.
- No processed image copies are written.

## Reported Outputs

- split and label counts
- train/validation/test metrics
- augmentation leakage check
- per-case loading failures
- CNN embeddings
- prediction rows
- a small model checkpoint

## Claim Limits

- This stage validates data loading, split safety, and training mechanics.
- It is not a final CNN-only baseline.
- It does not support hybrid-model, lesion-localization, clinical deployment, or
  biopsy-reduction claims.
