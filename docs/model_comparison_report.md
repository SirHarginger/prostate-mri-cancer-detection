# Current Model Comparison Report

This report is a concise checkpoint across the current internal experiments:

- full-cohort rotated-fold radiomics-only baseline
- controlled 2.5D CNN baseline
- aligned radiomics-only, CNN-only, and hybrid baselines

It is designed for methodology tracking. It is not a final publication table.

## Command

Run from the cluster repository root after the input reports exist:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli model-comparison-report \
  --radiomics-cv-report outputs/reports/radiomics_cv_report.json \
  --cnn-report outputs/reports/cnn_baseline_25d_report.json \
  --hybrid-report outputs/reports/hybrid_ml_report.json \
  --json-report outputs/reports/current_model_comparison.json \
  --markdown-report outputs/reports/current_model_comparison.md
```

Outputs are written under `outputs/reports`, which is ignored by Git.

## Contents

The report summarizes:

- case counts and feature counts
- default test ROC-AUC, sensitivity, and specificity
- validation-selected fixed-sensitivity test behavior
- top hybrid coefficients
- limitations and claim boundaries

## Interpretation Boundaries

The full radiomics CV result uses the full radiomics table. The CNN and hybrid
results use only cases with CNN embeddings. These scopes are intentionally shown
separately and should not be described as one unified external-validation
comparison.

The report must not be used to claim clinical deployment readiness, lesion
localization, radiologist replacement, external validation, or biopsy reduction.
