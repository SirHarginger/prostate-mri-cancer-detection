# Production Stage C Full Whole-Gland Radiomics

Production Stage C extracts case-level whole-gland radiomics for T2W, ADC, and
high b-value DWI. T2W is used on its native grid. ADC and high b-value DWI are
resampled in memory to each case's T2W grid using SimpleITK linear
interpolation. The selected gland mask must already be a non-empty T2W-grid
candidate.

This stage writes feature tables only. It does not train models, augment
images, write full processed image datasets, or use lesion ROI features for
case-level classification.

## Sample Command

Run a small sample first:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli radiomics-full-gland \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --sample-size 25 \
  --output data/features/radiomics_gland_multisequence_sample.csv \
  --failure-log outputs/reports/radiomics_gland_multisequence_sample_failures.csv \
  --settings outputs/reports/radiomics_gland_multisequence_sample_settings.json
```

## Full Command

After the sample succeeds, run the full 1500-case extraction:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli radiomics-full-gland \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --all-cases \
  --output data/features/radiomics_gland_multisequence_full.csv \
  --failure-log outputs/reports/radiomics_gland_multisequence_full_failures.csv \
  --settings outputs/reports/radiomics_gland_multisequence_full_settings.json
```

Generated feature tables and reports are ignored by Git.

## Feature Scope

Each case row includes sequence-prefixed first-order features:

- voxel count
- mask volume
- mask fraction
- intensity min, max, mean, standard deviation, median, p10, p90, IQR
- intensity energy
- 32-bin intensity entropy

## Safety And Validity Rules

- Raw files are never modified.
- ADC/HBV are resampled in memory only.
- Gland masks must be non-empty and T2W-grid compatible.
- Failures are logged per case.
- Labels, folds, paths, and sequence names are metadata, not predictive
  features.

## Current Limitations

- Features are dependency-light first-order radiomics, not full PyRadiomics
  texture features.
- Lesion ROI radiomics remain separate because empty lesion masks are expected
  for negative cases.
- This feature table alone does not support model-performance or clinical
  claims.
