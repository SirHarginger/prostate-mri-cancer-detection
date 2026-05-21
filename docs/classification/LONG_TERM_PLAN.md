# Classifier-First Long-Term Plan

This document tracks the classifier-first PI-CAI workflow. The project is a
research prototype and is not clinically validated.

## Phase 0 - Documentation and Agent Guidance

Goal: make the repository self-explanatory for Codex sessions, humans, and
future collaborators.

Status: current documentation task.

Tasks:

- Update `AGENTS.md` with the classifier-first direction.
- Add the long-term plan, scientific guardrails, cluster workflow, and current
  status documents under `docs/classification/`.
- Keep generated data, outputs, reports, and model artifacts outside Git.

## Phase 1 - Leakage-Safe Fold0 Feature Extraction

Goal: produce one feature row per PI-CAI fold0 case without using target-leaking
features.

Status: mostly complete.

Inputs:

- PI-CAI fold0 image manifest.
- T2W images.
- Whole-gland masks.
- Safe clinical variables.

Output outside Git:

```text
/home/degboh/prostate_mri_cancer_detection/data/features/picai_fold0_case_features.csv
```

Current result:

- 300 rows x 118 columns.
- 213 non-csPCa and 87 csPCa cases.
- Feature errors: 0.
- T2W whole-gland radiomics only.

## Phase 2 - Baseline Fold0 Classifier

Goal: train defensible baseline binary csPCa classifiers from the fold0 feature
table.

Status: initial fold0 cluster run complete.

Created:

```text
scripts/classification/train_picai_baseline_classifier.py
```

Input:

```text
/home/degboh/prostate_mri_cancer_detection/data/features/picai_fold0_case_features.csv
```

Requirements:

- Target: `case_cspca_binary`.
- Exclude leakage columns and diagnosis-derived variables.
- Use a stratified train/validation split.
- Handle missing values.
- Standardize numeric features where needed.
- One-hot encode `center`.
- Train at least Logistic Regression and Random Forest.

Artifacts outside Git:

```text
/home/degboh/prostate_mri_cancer_detection/artifacts/classifier_v1_fold0/
```

Expected artifacts:

- `model.joblib`
- `preprocessing_pipeline.joblib`
- `feature_schema.json`
- `metrics.json`
- `training_summary.md`

Metrics:

- ROC AUC.
- PR AUC.
- Sensitivity.
- Specificity.
- Balanced accuracy.
- F1.
- Confusion matrix.

Initial fold0 validation result:

- Selected model: Logistic Regression.
- Logistic Regression ROC AUC: 0.6908.
- Logistic Regression PR AUC: 0.4732.
- Logistic Regression balanced accuracy: 0.6546.
- Random Forest underperformed this first baseline split.

## Phase 3 - Evaluation and Reporting

Goal: evaluate the selected fold0 baseline and produce reproducible reports.

Status: initial fold0 report generation complete.

Created:

```text
scripts/classification/evaluate_picai_classifier.py
```

Outputs outside Git:

- ROC curve.
- PR curve.
- Confusion matrix.
- Metrics JSON.
- Model card draft.

Current report directory:

```text
/home/degboh/prostate_mri_cancer_detection/reports/classifier_v1_fold0/
```

## Phase 4 - Backend-Ready Inference Package

Goal: move stable inference logic into reusable package code.

Create:

```text
src/prostate_detection/inference/schema.py
src/prostate_detection/inference/predictor.py
```

Expose:

```python
predict_case(
    t2w_image_path: str,
    whole_gland_mask_path: str,
    clinical_metadata: dict,
    model_dir: str,
) -> dict
```

Expected output shape:

```json
{
  "case_id": "...",
  "prediction_target": "case_csPCa",
  "probability_cspca": 0.0,
  "risk_category": "low/intermediate/high",
  "model_version": "...",
  "features_used": [],
  "warnings": [],
  "notes": ["Decision-support only. Not a standalone diagnosis."]
}
```

## Phase 5 - Scale to All PI-CAI Folds

Goal: expand from fold0 to the full PI-CAI dataset after the fold0 workflow is
stable.

Tasks:

- Download folds 1-4 on the cluster.
- Rebuild the image manifest.
- Confirm all 1500 cases have T2W, ADC, and HBV where expected.
- Run QC checks.
- Extract all-case leakage-safe features.
- Train the full PI-CAI classifier.
- Use stronger validation, including possible center-aware validation.

## Phase 6 - Add ADC/HBV Whole-Gland Features

Goal: test whether additional bpMRI modalities improve performance without
introducing lesion-mask leakage.

Tasks:

- Resample whole-gland masks from T2W space to ADC/HBV space.
- Extract ADC whole-gland radiomics.
- Extract HBV whole-gland radiomics.
- Compare clinical only, T2W only, clinical plus T2W, and clinical plus
  T2W/ADC/HBV feature sets.

## Phase 7 - Lesion Characterization Only

Goal: use lesion masks only in a separate positive-case or lesion-level task.

Possible targets:

- ISUP grade among positive cases.
- Lesion burden.
- Lesion texture analysis.

Do not mix lesion-mask features into binary csPCa detection.

## Phase 8 - Backend-Ready ML Service

Goal: expose stable inference through a service only after model artifacts and
package inference code are reliable.

Tasks:

- Add a FastAPI backend.
- Load versioned model artifacts from the configured model directory.
- Return decision-support outputs and warnings.
- Keep frontend work out of scope until the backend is stable.

## Phase 9 - Operator UI

Goal: build a later operator-facing interface once the ML service is stable.

Possible UI capabilities:

- 2D MRI viewer.
- T2W/ADC/HBV switching.
- Mask overlay.
- Opacity control.
- Classifier output.
- Warning panel.
- Report export.
