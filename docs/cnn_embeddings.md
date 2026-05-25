# Stage 4 CNN Embedding Pipeline

Stage 4 adds a split-safe embedding pipeline scaffold. The current
implementation writes deterministic T2W prototype embeddings with model/config
provenance so later CNN-only and hybrid baselines can use the same table shape
and split metadata.

This is not a trained CNN model. It must not be reported as a final CNN-only
baseline or clinical model. It is a minimal working pipeline for data loading,
split tracking, augmentation guards, embedding table schema, and provenance.

## Command

Run from the cluster repository root after Stage 2 validation has passed:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli embedding-extract \
  --manifest data/interim/picai_manifest.csv \
  --preprocessing-report outputs/reports/preprocessing_fold_sample_validation.json \
  --raw-root data/raw/picai \
  --sequence t2w \
  --embedding-dim 32 \
  --output data/features/embeddings_t2w_prototype_sample.csv \
  --provenance outputs/reports/embeddings_t2w_prototype_provenance.json \
  --report outputs/reports/embeddings_t2w_prototype_report.json
```

Outputs are written under `data/features` and `outputs/reports`, both ignored by
Git.

## Split Policy

The initial split mapping is fold-based:

- `fold0`, `fold1`, `fold2`: training
- `fold3`: validation
- `fold4`: test

The embedding report records rows by split and verifies that validation/test
rows are never marked as augmented.

## Augmentation Policy

No augmentation is applied by default. The optional `--augment-train` flag
applies only a deterministic train-split intensity perturbation for leakage
testing. It does not save augmented image copies, and validation/test rows must
remain unaugmented.

## Paired Modality Policy

Stage 2 showed ADC and high b-value DWI are on different native grids from T2W.
Therefore Stage 4 currently extracts T2W-only prototype embeddings. ADC/HBV and
ROI-aware paired embeddings are deferred until a resampling/alignment
implementation exists.

## Claim Limits

- These embeddings are a pipeline prototype, not trained CNN features.
- Do not use them as the final CNN-only baseline.
- Do not make lesion localization, diagnosis, deployment, or biopsy-reduction
  claims from this output.
