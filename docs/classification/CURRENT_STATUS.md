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

No baseline classifier training script has been added yet.

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

## Next Task

Implement:

```text
scripts/classification/train_picai_baseline_classifier.py
```

The script should train baseline Logistic Regression and Random Forest models
from the fold0 feature CSV, save artifacts outside Git, and report ROC AUC, PR
AUC, sensitivity, specificity, balanced accuracy, F1, and confusion matrix.

Planned artifact directory:

```text
/home/degboh/prostate_mri_cancer_detection/artifacts/classifier_v1_fold0/
```

## Known Limitations

- Current work is fold0 only.
- Current radiomics are T2W whole-gland only.
- ADC/HBV whole-gland features are deferred.
- Zonal features are deferred due to geometry mismatch.
- Lesion features are forbidden for binary csPCa v1 due to leakage.
- No baseline classifier has been trained yet.
- No model is clinically validated.
