# PI-CAI Classifier-First Research Framework

This document defines the research design for the classifier-first prostate MRI
cancer detection work. It is the controlling plan for the next implementation
phase.

The goal is a leakage-safe, case-level classifier for clinically significant
prostate cancer using PI-CAI bpMRI. This is research software only. It is not a
clinical diagnostic device and is not clinically validated.

## 1. Research Objective

Primary task:

- Predict case-level clinically significant prostate cancer.

Primary target:

- `case_cspca_binary`

Primary dataset:

- PI-CAI public prostate MRI dataset.

Primary imaging inputs:

- T2W.
- ADC.
- HBV.

Primary non-image inputs:

- Safe clinical variables only.

The first serious classifier should answer a narrow question:

> Given leakage-safe PI-CAI case metadata and bpMRI images, can a model estimate
> case-level csPCa risk better than a simple clinical/radiomics baseline?

The model output must be framed as experimental decision-support research, not
as a standalone diagnosis.

## 2. Current Data Foundation

All five PI-CAI image folds have been downloaded on the cluster.

Cluster image root:

```text
/home/degboh/prostate_mri_cancer_detection/data/raw/picai/images
```

All-fold image manifest:

```text
/home/degboh/prostate_mri_cancer_detection/data/features/picai_all_folds_image_manifest.csv
```

Current all-fold manifest status:

- Shape: 1500 rows x 35 columns.
- T2W available: 1500 cases.
- ADC available: 1500 cases.
- HBV available: 1500 cases.
- Coronal T2W available: 1497 cases.
- Sagittal T2W available: 1498 cases.
- Core bpMRI cases: 1500.
- Non-csPCa cases: 1075.
- csPCa cases: 425.

This means the project is no longer limited to a fold0 prototype. The next
stage can prepare a serious all-fold feature and image-classification workflow,
provided leakage and preprocessing checks remain strict.

## 3. Scientific Guardrails

These rules are hard constraints.

### Lesion Masks Are Forbidden for Binary Detection

In fold0, lesion masks are empty for all non-csPCa cases and non-empty for all
csPCa cases. That makes lesion-mask existence a direct proxy for the target.

Do not use any of the following for binary csPCa classification:

- Lesion mask as model input.
- Lesion crop as model input.
- Lesion-mask existence.
- Lesion-mask volume.
- Lesion-mask radiomics.
- Lesion bounding boxes derived from lesion masks.
- Any feature whose availability depends on lesion annotation.

Lesion masks may be used later only for a separate lesion-characterization task
among positive or suspicious cases.

### Diagnosis-Derived Variables Are Not Predictors

Do not use these as predictors:

- `case_cspca_binary`
- `case_isup_int`
- `case_ISUP`
- Gleason or pathology-derived fields.
- Any diagnosis-derived variable.
- Any variable created after knowing the target.

`case_isup_int` is allowed only as metadata for reporting or future separate
tasks, not as an input to the binary csPCa classifier.

### No Blind Column Use

Never train from all columns by default. Every predictor must be explicitly
allowed by policy.

Allowed initial tabular predictors:

- `patient_age`
- `psa`
- `psad`
- `prostate_volume`
- Encoded `center`
- Leakage-safe whole-gland radiomics columns.

Forbidden initial predictors:

- Labels.
- IDs.
- File paths.
- Feature-error columns.
- Lesion columns.
- Target-derived columns.
- Diagnosis-derived columns.

### Patient/Case-Level Splits Only

Do not split slices or crops independently across train and validation.

The same patient/case must not appear in both train and validation. If multiple
studies for a patient exist, the split must prevent patient-level leakage.

### Decision-Support Wording Only

Use language such as:

- Research prototype.
- Experimental model.
- Case-level csPCa risk estimate.
- Decision-support research.
- Not clinically validated.

Do not claim diagnosis, clinical deployment readiness, or patient management
utility.

## 4. Geometry and Preprocessing Facts

Known example case `10000_1000000`:

- T2W image: 640 x 640 x 31.
- ADC image: 116 x 114 x 31.
- HBV image: 116 x 114 x 31.
- Lesion mask: 116 x 114 x 31.
- Whole-gland mask: 640 x 640 x 31.
- Zonal mask: 640 x 640 x 25.

Interpretation:

- Whole-gland mask aligns with T2W space.
- ADC and HBV are lower-resolution images.
- Lesion masks align with ADC/HBV space but are forbidden for binary detection.
- Zonal masks are not cleanly aligned for v1 and are deferred.

Initial safe crop policy:

- Use the whole-gland mask only to localize/crop the prostate.
- Do not feed the mask itself as a diagnostic input channel in the first model.
- Use mask-derived crop coordinates consistently for positive and negative
  cases.
- Save crop metadata and preprocessing parameters for reproducibility.

## 5. Experiment Ladder

The project should move in controlled layers. Each layer should be kept
reproducible and compared to the previous one.

### Experiment 0: Data and Split Audit

Purpose:

- Confirm all 1500 core bpMRI cases are readable.
- Confirm labels and centers.
- Confirm patient/case uniqueness.
- Confirm image shape, spacing, and modality availability.
- Confirm whole-gland mask availability and usability.

Expected output:

- JSON audit summary outside Git.
- Split proposal outside Git.
- Updated status documentation in Git.

No model training is required in this step.

### Experiment 1: All-Fold Classical Baseline

Purpose:

- Scale the current leakage-safe fold0 classical baseline to all 1500 cases.

Inputs:

- All-fold image manifest.
- T2W images.
- Whole-gland masks.
- Safe clinical variables.

Features:

- Clinical variables.
- T2W whole-gland radiomics.

Models:

- Logistic Regression.
- Random Forest.
- Optionally Gradient Boosting after the simple baselines.

This becomes the reference baseline that deep learning must beat.

### Experiment 2: Image Preprocessing Prototype

Purpose:

- Build a reproducible prostate-centered crop pipeline before deep training.

Inputs:

- T2W, ADC, HBV.
- Whole-gland mask for crop localization only.

Required checks:

- Crop bounding boxes.
- Crop sizes.
- Spacing summaries.
- Intensity distributions.
- Missing or empty whole-gland masks.
- Modality alignment strategy.

No deep model should be trained until this step produces trustworthy QC.

### Experiment 3: Simple Deep Image Prototype

Purpose:

- Test whether a simple image model can learn anything useful without leakage.

Preferred first model:

- 2.5D axial CNN on prostate-centered crops.

Reason:

- It is simpler and less memory-heavy than a 3D CNN.
- It exposes preprocessing mistakes quickly.
- It can use neighboring slices without requiring a full-volume network.

Inputs:

- T2W crop first.
- Then T2W + ADC + HBV once modality handling is validated.

Target:

- `case_cspca_binary`

Validation:

- Patient/case-level split only.
- Stratified by label.
- Record center distribution.

### Experiment 4: Multimodal Deep Model

Purpose:

- Learn from T2W, ADC, and HBV together.

Candidate designs:

- Three-channel resampled crop.
- Modality-specific branches merged before classification.
- Late-fusion image model with clinical metadata.

This should come only after the single-modality prototype and crop QC are
stable.

### Experiment 5: Hybrid Model

Purpose:

- Combine deep image features with safe clinical variables and radiomics.

Inputs:

- Deep image embedding.
- Safe clinical variables.
- Leakage-safe radiomics.

Rules:

- The hybrid model must use the same split as the image model.
- Radiomics and clinical preprocessing must be fitted on training data only.
- It must be compared against the all-fold classical baseline.

### Experiment 6: Center-Aware and External Validation

Purpose:

- Estimate whether performance generalizes beyond random or fold-based splits.

Validation extensions:

- Center-aware split.
- Leave-one-center-out analysis if feasible.
- External validation later if a suitable dataset is available.

This step is required before any serious claim about generalization.

## 6. Safe Deep-Learning Formulations

### Preferred: Whole-Gland Crop Classifier

Input:

- Prostate-centered crop from T2W, ADC, and HBV.

Target:

- `case_cspca_binary`

Mask use:

- Whole-gland mask only for cropping/localization.

Why this is preferred:

- Uses anatomical context available for all cases.
- Avoids lesion-mask leakage.
- Reduces background/scanner shortcut risk compared with full-image models.

### Acceptable but Riskier: Full-Image Classifier

Input:

- Full T2W, ADC, and HBV images.

Target:

- `case_cspca_binary`

Main risks:

- Scanner/site shortcuts.
- Background anatomy shortcuts.
- Larger memory requirements.
- Harder QC.

This can be an exploratory baseline but should not be the main path.

### Invalid for Binary Detection: Lesion-Crop Classifier

Input:

- Lesion crop, lesion mask, or lesion-localized patch.

Target:

- `case_cspca_binary`

Status:

- Not valid for binary detection because lesion-mask existence leaks the label.

### Later: Hybrid Image + Clinical + Radiomics

Input:

- Deep image features.
- Safe clinical variables.
- Leakage-safe whole-gland radiomics.

Status:

- Valid later, after the image pipeline is stable and the classical baseline is
  established.

## 7. Preprocessing Architecture

The preprocessing code should be explicit, auditable, and separate from model
training.

Recommended stages:

1. Read all-fold manifest.
2. Filter to `has_core_bpMRI == True`.
3. Validate file existence for T2W, ADC, HBV, and whole-gland mask.
4. Load images with SimpleITK or MONAI.
5. Normalize whole-gland mask to binary.
6. Resample mask if needed for crop localization.
7. Build a prostate-centered crop around the whole-gland bounding box.
8. Decide modality handling:
   - Resample ADC/HBV to the T2W crop space, or
   - keep modality-specific crops and use modality-specific branches.
9. Normalize intensities per image/crop.
10. Save only lightweight manifests/QC summaries unless explicitly generating
    processed arrays outside Git.

Generated arrays, caches, model files, and reports must stay under the cluster
storage root, not inside Git.

Preferred generated locations:

```text
/home/degboh/prostate_mri_cancer_detection/data/features
/home/degboh/prostate_mri_cancer_detection/data/processed
/home/degboh/prostate_mri_cancer_detection/outputs
/home/degboh/prostate_mri_cancer_detection/artifacts
/home/degboh/prostate_mri_cancer_detection/logs
/home/degboh/prostate_mri_cancer_detection/reports
```

## 8. Split Strategy

Initial all-fold split policy:

- Use deterministic patient/case-level splits.
- Stratify by `case_cspca_binary`.
- Preserve case-level grouping.
- Record center distribution in each split.
- Save split definitions outside Git unless they are lightweight and safe to
  version.

Recommended validation ladder:

1. Stratified patient/case-level train/validation split for fast iteration.
2. Stratified group cross-validation for classical baselines.
3. Center-aware validation once the pipeline is stable.
4. External validation later.

Do not regenerate splits silently. Every training run must record the split
file or split seed.

## 9. Metrics

Primary metrics:

- ROC AUC.
- PR AUC.
- Sensitivity.
- Specificity.
- Balanced accuracy.
- F1.
- Confusion matrix.

Secondary metrics:

- Calibration curve.
- Brier score.
- Threshold-specific sensitivity/specificity.
- Center-stratified performance.

Because the dataset is imbalanced, PR AUC and sensitivity/specificity tradeoffs
must be reported alongside ROC AUC.

## 10. Engineering Architecture

Keep scripts thin and reusable logic under `src/` once implementation begins.

Recommended future structure:

```text
scripts/classification/
|-- audit_picai_all_folds.py
|-- build_picai_splits.py
|-- extract_picai_case_features.py
|-- train_picai_baseline_classifier.py
|-- evaluate_picai_classifier.py
|-- build_picai_prostate_crops.py
`-- train_picai_image_classifier.py

src/prostate_detection/
|-- classification/
|   |-- splits.py
|   |-- tabular.py
|   `-- metrics.py
|-- imaging/
|   |-- io.py
|   |-- geometry.py
|   |-- crops.py
|   `-- normalization.py
|-- datasets/
|   `-- picai.py
|-- models/
|   `-- image_classifier.py
`-- inference/
    |-- schema.py
    `-- predictor.py
```

Do not write one large training script that hides preprocessing, splits, model
definition, metrics, and artifact writing in one place.

## 11. Immediate Next Implementation Tasks

The next coding tasks should happen in this order.

### Task 1: All-Fold Manifest Audit

Create or extend an audit script that reads:

```text
/home/degboh/prostate_mri_cancer_detection/data/features/picai_all_folds_image_manifest.csv
```

It should report:

- Number of core bpMRI cases.
- Label counts.
- Center counts.
- Patient/case uniqueness.
- T2W, ADC, HBV shape and spacing summaries.
- Whole-gland mask availability and empty-mask count.
- Any path or geometry failures.

No model training.

### Task 2: All-Fold T2W Whole-Gland Radiomics

Scale the existing leakage-safe feature extractor to all 1500 cases.

Output outside Git:

```text
/home/degboh/prostate_mri_cancer_detection/data/features/picai_all_folds_case_features.csv
```

### Task 3: All-Fold Classical Baseline

Train and evaluate classical models on the all-fold feature table.

This establishes the serious baseline before deep learning.

### Task 4: Prostate Crop QC Pipeline

Build crop metadata and QC before training any CNN.

Outputs should include:

- Crop size summary.
- Example overlay figures.
- Failure report.
- Preprocessing config.

### Task 5: Small Image Prototype

Only after crop QC passes, implement a small 2D or 2.5D image classifier.

## 12. Definition of Ready for Deep Learning

The project is ready for serious deep-learning training only when all items are
true:

- All 1500 T2W, ADC, and HBV files are readable.
- Whole-gland masks are available and non-empty for the intended cases.
- Crop sizes and spacings are summarized.
- Modality geometry handling is decided and tested.
- PyTorch is installed on the cluster.
- MONAI is installed or a deliberate no-MONAI path is chosen.
- GPU availability is confirmed.
- Patient/case-level split is fixed.
- Leakage predictor policy is encoded in code.
- Classical all-fold baseline exists.
- Generated arrays and artifacts write outside Git.

Current state:

- Data availability for core bpMRI is ready.
- Serious deep-learning preprocessing is not ready until all-fold crop QC and
  GPU/PyTorch/MONAI readiness are confirmed.

## 13. Tomorrow's Coding Rule

Tomorrow's implementation should start from this classifier-first framework and
the `scripts/classification/` path only.

Old segmentation-first, nnU-Net, Kaggle, Prostate158, and auto-segmentation
scripts are historical context. They should not guide the PI-CAI classifier
architecture unless a specific reusable utility is deliberately reintroduced.
