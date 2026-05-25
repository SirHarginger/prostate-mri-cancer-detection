# Production Stage D Radiomics-Only ML Baseline

Production Stage D trains a full-table radiomics-only baseline on
`radiomics_gland_multisequence_full.csv`. This is the first serious
case-level modeling stage.

It does not use CNN embeddings, lesion ROI features, PI-RADS, or hybrid fusion.

## Environment

The cluster environment should include:

```bash
python -m pip install numpy SimpleITK scikit-learn
```

## Command

Run from the cluster repository root:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli radiomics-ml-baseline \
  --features data/features/radiomics_gland_multisequence_full.csv \
  --metrics outputs/reports/radiomics_ml_metrics.json \
  --predictions outputs/reports/radiomics_ml_predictions.csv \
  --report outputs/reports/radiomics_ml_report.json \
  --target-sensitivity 0.90
```

Outputs are ignored by Git.

## Split Policy

- Training: `fold0`, `fold1`, `fold2`
- Validation: `fold3`
- Test: `fold4`

Training-split medians and scaling statistics are learned only from training
rows.

## Model

The initial model is scikit-learn logistic regression:

- median imputation
- standard scaling
- balanced class weights
- `liblinear` solver
- fixed random seed

## Reported Outputs

- train/validation/test metrics
- ROC-AUC
- sensitivity, specificity, precision, F1
- confusion matrices
- false positives and false negatives
- fixed-sensitivity analysis
- top absolute logistic-regression coefficients

## Claim Limits

- This is an internal full-table radiomics-only baseline.
- It is not external validation.
- It does not support CNN, hybrid, lesion localization, clinical deployment, or
  biopsy-reduction claims.
