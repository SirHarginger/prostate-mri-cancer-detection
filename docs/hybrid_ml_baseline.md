# Hybrid Radiomics + CNN Baseline

This stage compares three aligned case-level representations on the same cases:

- whole-gland multisequence radiomics
- CNN embeddings from the controlled multisequence CNN run
- concatenated radiomics + CNN embeddings

It is an internal PI-CAI comparison. It is not external validation.

## Inputs

The command expects:

- `data/features/radiomics_gland_multisequence_full.csv`
- a CNN embedding table from `cnn-train-baseline`

The two tables are aligned by `case_id`. Cases with missing labels or label
mismatches are excluded and reported.

## Command

Run from the cluster repository root:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli hybrid-ml-baseline \
  --radiomics data/features/radiomics_gland_multisequence_full.csv \
  --embeddings data/features/cnn_baseline_25d_embeddings.csv \
  --metrics outputs/reports/hybrid_ml_metrics.json \
  --predictions outputs/reports/hybrid_ml_predictions.csv \
  --report outputs/reports/hybrid_ml_report.json \
  --target-sensitivity 0.90
```

Outputs are written under ignored project paths.

## Model

Each representation uses the same scikit-learn logistic-regression workflow:

- median imputation
- standard scaling
- balanced class weights
- `liblinear` solver
- `C` selected by validation ROC-AUC

The split assignment comes from the CNN embedding table when present, with the
standard fold mapping as fallback.

## Threshold Policy

For each representation, the validation split selects a fixed-sensitivity
threshold. That threshold is then applied to the held-out test split.

## Claim Limits

- The comparison is restricted to cases that have both radiomics and CNN
  embeddings.
- CNN embeddings reflect the current controlled CNN baseline, not a final tuned
  CNN model.
- No clinical deployment, lesion localization, external validation, or
  biopsy-reduction claims are supported by this stage.
