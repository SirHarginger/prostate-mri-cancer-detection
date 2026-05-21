# Scientific Guardrails for PI-CAI Classification

This project is research software. It is not clinically validated and must not
be used as a standalone diagnostic system.

## Leakage Rules

Do not use lesion-mask radiomics for the v1 binary csPCa classifier.

Observed PI-CAI fold0 finding:

```text
Lesion masks are empty for all non-csPCa fold0 cases and non-empty for all
csPCa fold0 cases.
```

That means lesion-mask emptiness perfectly correlates with the binary target in
fold0. Any lesion-mask feature would leak the label into the model.

Lesion features may be considered later only for a separate
lesion-characterization task, such as positive-case grading or lesion texture
analysis. They must not be mixed into binary csPCa detection.

## Diagnosis-Derived Variables

Do not use these as binary csPCa predictors:

- `case_cspca_binary`
- `case_ISUP`
- `case_isup_int`
- Gleason-derived fields.
- Histopathology-derived fields.
- Any target-derived or diagnosis-derived variable.

`case_cspca_binary` is the target. `case_ISUP` and `case_isup_int` are closely
related pathology outputs and must be treated as labels or metadata, not model
inputs.

## Allowed V1 Predictors

For the fold0 binary classifier, use only predictors that are available for all
cases and do not encode the target:

- `patient_age`
- `psa`
- `psad`
- `prostate_volume`
- Encoded `center`
- `t2w_wholegland_*` radiomics columns

Do not blindly train on all columns from the feature CSV.

Exclude from predictors:

- `case_cspca_binary`
- `case_isup_int`
- `case_key`
- `patient_id`
- `study_id`
- `feature_error`
- Any lesion-related column.
- Any diagnosis-derived or target-derived variable.

## Geometry Rules

Example geometry finding for case `10000_1000000`:

```text
T2W image:         640 x 640 x 31
ADC image:         116 x 114 x 31
HBV image:         116 x 114 x 31
Lesion mask:       116 x 114 x 31
Whole-gland mask:  640 x 640 x 31
Zonal mask:        640 x 640 x 25
```

Interpretation:

- Lesion masks align with ADC/HBV space.
- Whole-gland masks align with T2W space.
- Zonal masks are not cleanly aligned for v1.

For v1, use T2W whole-gland radiomics and safe clinical variables. Defer zonal
features. If ADC/HBV whole-gland features are added later, resample the
whole-gland mask into ADC/HBV space with explicit QC.

## Split and Evaluation Rules

- Use case-level or patient-level splits only.
- Never split 2D slices across train and validation/test.
- Use stratification for fold0 baseline experiments where practical.
- Prefer official PI-CAI folds when scaling beyond fold0.
- Report patient-level or case-level metrics for classification.
- Include ROC AUC, PR AUC, sensitivity, specificity, balanced accuracy, F1, and
  confusion matrix for baseline classifiers.

## Claim and Reporting Rules

Use careful language:

- Research prototype.
- Experimental classifier.
- Decision-support only.
- Not clinically validated.
- Not a standalone diagnosis.

Do not claim:

- The model diagnoses patients.
- The model is clinically validated.
- The system should be used for real clinical decisions.
- Segmentation or radiomics performance proves cancer detection ability.

Generated outputs, feature CSVs, model artifacts, logs, and reports must stay
outside Git.
