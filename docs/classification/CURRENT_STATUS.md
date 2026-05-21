# Current PI-CAI Classification Status

This document records the current classifier-first state of the project.

## Current Direction

The project is currently focused on a leakage-safe, case-level binary classifier
for clinically significant prostate cancer using PI-CAI data. Auto-segmentation
is paused as the primary path.

This is research software only and is not clinically validated.

## Existing Classification Scripts

Current scripts:

- `scripts/classification/build_picai_manifest.py`
- `scripts/classification/build_picai_mask_manifest.py`
- `scripts/classification/build_picai_image_manifest.py`
- `scripts/classification/check_picai_geometry.py`
- `scripts/classification/check_picai_mask_content.py`
- `scripts/classification/extract_picai_case_features.py`
- `scripts/classification/train_picai_baseline_classifier.py`
- `scripts/classification/evaluate_picai_classifier.py`
- `scripts/classification/audit_deep_learning_readiness.py`

The baseline classifier training script exists and has completed the first
fold0 cluster run.

## Data Status

Full PI-CAI clinical manifest:

- 1500 cases.
- 1075 non-csPCa cases.
- 425 csPCa cases.
- Binary target: `case_cspca_binary`.

PI-CAI fold0 image manifest:

- 300 cases with core bpMRI.
- Core bpMRI means T2W, ADC, and HBV are present.
- Fold0 labels: 213 non-csPCa and 87 csPCa.

PI-CAI all-fold image manifest:

```text
/home/degboh/prostate_mri_cancer_detection/data/features/picai_all_folds_image_manifest.csv
```

- 1500 cases.
- 1500 cases with core bpMRI.
- T2W: 1500 available.
- ADC: 1500 available.
- HBV: 1500 available.
- Coronal T2W: 1497 available.
- Sagittal T2W: 1498 available.
- All-fold labels: 1075 non-csPCa and 425 csPCa.

## Feature Extraction Status

Current feature table:

```text
/home/degboh/prostate_mri_cancer_detection/data/features/picai_fold0_case_features.csv
```

Current result:

- Shape: 300 rows x 118 columns.
- All 300 fold0 core bpMRI cases represented.
- Label balance preserved: 213 non-csPCa and 87 csPCa.
- Feature errors: 0.
- Radiomics extracted from T2W whole-gland region.
- Lesion radiomics excluded.
- Zonal features excluded.

The PyRadiomics message below is informational and not a failure:

```text
GLCM is symmetrical, therefore Sum Average = 2 * Joint Average, only 1 needs to be calculated
```

## Important Findings

Leakage finding:

- Lesion masks are empty for all non-csPCa fold0 cases.
- Lesion masks are non-empty for all csPCa fold0 cases.
- Lesion-mask radiomics must not be used for binary csPCa classification.

Geometry finding from example case `10000_1000000`:

- T2W image: 640 x 640 x 31.
- ADC image: 116 x 114 x 31.
- HBV image: 116 x 114 x 31.
- Lesion mask: 116 x 114 x 31.
- Whole-gland mask: 640 x 640 x 31.
- Zonal mask: 640 x 640 x 25.

Interpretation:

- Lesion mask aligns with ADC/HBV space.
- Whole-gland mask aligns with T2W space.
- Zonal mask is not cleanly aligned for v1.

## Predictor Policy for Next Task

Target:

- `case_cspca_binary`

Allowed predictors:

- `patient_age`
- `psa`
- `psad`
- `prostate_volume`
- Encoded `center`
- `t2w_wholegland_*` radiomics columns

Exclude from predictors:

- `case_cspca_binary`
- `case_isup_int`
- `case_key`
- `patient_id`
- `study_id`
- `feature_error`
- Any lesion-related column.
- Any target-derived or diagnosis-derived variable.

## Baseline Classifier Status

Cluster command used:

```bash
python scripts/classification/train_picai_baseline_classifier.py \
  --features /home/degboh/prostate_mri_cancer_detection/data/features/picai_fold0_case_features.csv \
  --output-dir /home/degboh/prostate_mri_cancer_detection/artifacts/classifier_v1_fold0 \
  --overwrite
```

The script trains baseline Logistic Regression and Random Forest models from the
fold0 feature CSV, saves artifacts outside Git, and reports ROC AUC, PR AUC,
sensitivity, specificity, balanced accuracy, F1, and confusion matrix.

Artifact directory:

```text
/home/degboh/prostate_mri_cancer_detection/artifacts/classifier_v1_fold0/
```

Run result:

- Input shape: 300 rows x 118 columns.
- Feature count used by classifier: 112.
- Target counts: 213 non-csPCa and 87 csPCa.
- Split: `StratifiedGroupKFold(n_splits=5)`.
- Train/validation cases: 240 train and 60 validation.
- Selected model: Logistic Regression.

Validation metrics:

| Model | ROC AUC | PR AUC | Sensitivity | Specificity | Balanced Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6908 | 0.4732 | 0.5882 | 0.7209 | 0.6546 | 0.5128 |
| Random Forest | 0.5978 | 0.3549 | 0.2353 | 0.8837 | 0.5595 | 0.3077 |

## Evaluation and Reporting Status

Cluster command used:

```bash
python scripts/classification/evaluate_picai_classifier.py \
  --features /home/degboh/prostate_mri_cancer_detection/data/features/picai_fold0_case_features.csv \
  --model-dir /home/degboh/prostate_mri_cancer_detection/artifacts/classifier_v1_fold0 \
  --output-dir /home/degboh/prostate_mri_cancer_detection/reports/classifier_v1_fold0 \
  --overwrite
```

Evaluation outputs were saved under:

```text
/home/degboh/prostate_mri_cancer_detection/reports/classifier_v1_fold0/
```

Expected report artifacts:

- `evaluation_metrics.json`
- `roc_curve.png`
- `precision_recall_curve.png`
- `confusion_matrix.png`
- `model_card_draft.md`

The evaluator reproduced the saved fold0 validation metrics:

- ROC AUC: 0.6908.
- PR AUC: 0.4732.
- Sensitivity: 0.5882.
- Specificity: 0.7209.
- Balanced accuracy: 0.6546.
- F1: 0.5128.

## Research Framework

The controlling design document for the next phase is:

```text
docs/classification/RESEARCH_FRAMEWORK.md
```

It defines the experiment ladder, leakage rules, preprocessing architecture,
split policy, metrics, and immediate all-fold implementation tasks.

## Next Task

Run an all-fold readiness/data audit on the cluster using the all-fold image
manifest:

```bash
python scripts/classification/audit_deep_learning_readiness.py \
  --manifest /home/degboh/prostate_mri_cancer_detection/data/features/picai_all_folds_image_manifest.csv \
  --output /home/degboh/prostate_mri_cancer_detection/outputs/deep_learning_readiness_all_folds.json \
  --limit 50
```

The audit should confirm all-fold image availability, sample geometry,
whole-gland crop feasibility, and torch/MONAI/GPU readiness before any
deep-learning prototype is implemented.

## Known Limitations

- Full PI-CAI core bpMRI image availability is now confirmed in the all-fold
  image manifest, but all-fold feature extraction and all-fold modeling have not
  yet been run.
- Current radiomics are T2W whole-gland only.
- ADC/HBV whole-gland features are deferred.
- Zonal features are deferred due to geometry mismatch.
- Lesion features are forbidden for binary csPCa v1 due to leakage.
- Baseline metrics are from one fold0 validation split only.
- Evaluation/reporting artifacts exist for fold0 only.
- Logistic Regression performance is modest and needs stronger validation.
- Deep learning is not ready for serious training until all-fold preprocessing,
  crop QC, split policy, and GPU/PyTorch/MONAI readiness are verified.
- No model is clinically validated.
