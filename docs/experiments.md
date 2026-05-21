# Experiments

This document will track experiment configurations, runs, metrics, outputs, and
known limitations.

## Experiment Records

Each meaningful run should record:

- Dataset and manifest.
- Split strategy.
- Model or method.
- Config path.
- Output directory.
- Metrics.
- Known failures or warnings.

## PI-CAI Fold0 Baseline Classifier V1

Date recorded: after first cluster run of
`scripts/classification/train_picai_baseline_classifier.py`.

Dataset and manifest:

- PI-CAI fold0 case feature table.
- Feature CSV:
  `/home/degboh/prostate_mri_cancer_detection/data/features/picai_fold0_case_features.csv`
- Input shape: 300 rows x 118 columns.
- Target: `case_cspca_binary`.
- Target counts: 213 non-csPCa and 87 csPCa.

Feature policy:

- Used 112 predictor columns.
- Included safe clinical variables, encoded `center`, and
  `t2w_wholegland_*` radiomics.
- Excluded lesion features, IDs, `case_isup_int`, `feature_error`, and
  target-derived fields.

Split strategy:

- `StratifiedGroupKFold(n_splits=5)`.
- 240 training cases and 60 validation cases.

Artifacts:

```text
/home/degboh/prostate_mri_cancer_detection/artifacts/classifier_v1_fold0/
```

Validation metrics:

| Model | ROC AUC | PR AUC | Sensitivity | Specificity | Balanced Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6908 | 0.4732 | 0.5882 | 0.7209 | 0.6546 | 0.5128 |
| Random Forest | 0.5978 | 0.3549 | 0.2353 | 0.8837 | 0.5595 | 0.3077 |

Selected model:

- Logistic Regression.

Known limitations:

- Initial fold0 validation result only.
- Not clinically validated.
- Requires evaluation plots, model-card draft, and stronger validation before
  any backend-facing inference package.

Next reporting command:

```bash
python scripts/classification/evaluate_picai_classifier.py \
  --features /home/degboh/prostate_mri_cancer_detection/data/features/picai_fold0_case_features.csv \
  --model-dir /home/degboh/prostate_mri_cancer_detection/artifacts/classifier_v1_fold0 \
  --output-dir /home/degboh/prostate_mri_cancer_detection/reports/classifier_v1_fold0 \
  --overwrite
```
