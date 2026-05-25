# Production Stage E Rotated-Fold Radiomics Baseline

Production Stage E strengthens the radiomics-only baseline by rotating the
PI-CAI folds. Each fold is held out as test once, the next fold is used for
validation, and the remaining three folds are used for training.

This stage still uses only whole-gland multisequence radiomics. It does not use
CNN embeddings, lesion ROI features, PI-RADS, or hybrid fusion.

## Command

Run from the cluster repository root after the full radiomics table exists:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli radiomics-cv-baseline \
  --features data/features/radiomics_gland_multisequence_full.csv \
  --metrics outputs/reports/radiomics_cv_metrics.json \
  --predictions outputs/reports/radiomics_cv_predictions.csv \
  --report outputs/reports/radiomics_cv_report.json \
  --target-sensitivity 0.90
```

Outputs are ignored by Git.

## Rotation Policy

- Test: each PI-CAI fold is held out once.
- Validation: the next fold in sorted order.
- Training: the remaining three folds.

For example, when `fold0` is test, `fold1` is validation and `fold2`, `fold3`,
and `fold4` are training folds.

## Model Selection

The model is scikit-learn logistic regression with median imputation, standard
scaling, balanced class weights, and the `liblinear` solver. The regularization
parameter `C` is selected within each rotation by validation ROC-AUC.

The default grid is:

```text
0.01, 0.1, 1.0, 10.0
```

Additional runs can provide repeated `--c-value` flags to override the default.

## Fixed-Sensitivity Analysis

The report includes two threshold summaries:

- pooled held-out default metrics at threshold `0.5`
- validation-selected fixed-sensitivity thresholds applied to the held-out test
  fold for each rotation

The validation-selected fixed-sensitivity summary is the appropriate diagnostic
for threshold behavior because the threshold is not selected on the test fold.

## Claim Limits

- This is internal rotated-fold evaluation on PI-CAI public data.
- It is not external validation.
- It does not support CNN, hybrid, lesion localization, clinical deployment, or
  biopsy-reduction claims.
