# Stage 6 Evaluation Report

Stage 6 generates structured metrics and error-analysis reports from Stage 5
prediction rows. It does not train new models, tune thresholds on hidden data,
or add new claims. For now it summarizes prototype baselines only.

## Command

Run from the cluster repository root after Stage 5 baseline predictions exist:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli evaluation-report \
  --predictions outputs/reports/prototype_baseline_predictions.csv \
  --json-report outputs/reports/prototype_evaluation_report.json \
  --markdown-report outputs/reports/prototype_evaluation_report.md \
  --target-sensitivity 0.90
```

Reports are written under `outputs/reports`, which is ignored by Git.

## Contents

For each baseline and split, the report includes:

- ROC-AUC when both classes are present
- sensitivity
- specificity
- precision
- F1-score
- confusion matrix
- false-positive case IDs
- false-negative case IDs
- fixed-sensitivity threshold analysis

Undefined metrics remain `null` in JSON when a split lacks positives,
negatives, or predicted positives.

## Fixed-Sensitivity Analysis

The fixed-sensitivity section searches prediction thresholds and reports the
highest-specificity threshold that reaches the requested sensitivity. This is an
exploratory analysis of model outputs. It does not support biopsy-reduction
claims unless later validated with a final model, clinically meaningful split,
and documented decision policy.

## Ablation Status

Current Stage 6 output can summarize:

- radiomics-only prototype predictions
- prototype embedding-only predictions
- prototype hybrid predictions

It does not yet complete sequence contribution, augmentation versus no
augmentation, PI-RADS comparison, or final CNN-only ablations.

## Claim Limits

- These are prototype pipeline checks, not final scientific results.
- The embedding baseline is not a trained CNN baseline.
- Results are internal split/sample outputs, not external validation.
- Do not claim lesion localization, clinical deployment readiness,
  radiologist replacement, or biopsy reduction from this stage.
