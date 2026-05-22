# Stage 3 Radiomics Features

Stage 3 adds a minimal radiomics extraction pipeline for validated T2W-grid
ROIs. It is intentionally conservative: it extracts first-order ROI intensity
and size features from original T2W images only. ADC and high b-value DWI
radiomics are deferred until an explicit resampling implementation aligns those
modalities to the T2W reference grid.

This stage does not train models, augment images, generate CNN embeddings, or
claim lesion localization performance.

## Command

Run a sample extraction from the cluster repository root after Stage 2
preprocessing validation has passed:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli radiomics-extract \
  --manifest data/interim/picai_manifest.csv \
  --preprocessing-report outputs/reports/preprocessing_fold_sample_validation.json \
  --raw-root data/raw/picai \
  --sequence t2w \
  --roi lesion \
  --output data/features/radiomics_t2w_lesion_sample.csv \
  --failure-log outputs/reports/radiomics_t2w_lesion_failures.csv \
  --settings outputs/reports/radiomics_t2w_lesion_settings.json
```

The feature table is written under `data/features`, and logs/settings are
written under `outputs/reports`; both locations are ignored by Git.

## Feature Schema

The CSV contains one row per successfully extracted case/ROI:

- identifiers: `case_id`, `fold`, `sequence`, `roi`
- provenance: `image_path`, `mask_path`
- ROI size: `voxel_count`, `mask_volume_mm3`, `mask_fraction`
- first-order intensity features: minimum, maximum, mean, standard deviation,
  median, p10, p90, IQR, energy, and 32-bin entropy

The failure log records `case_id`, `fold`, `sequence`, `roi`, `reason`,
`image_path`, and `mask_path` for every failed extraction.

The extractor supports uncompressed or zlib-compressed MetaImage payloads and
NIfTI-1 masks using standard library readers. It still validates image/mask
shape and spacing before computing features.

## ROI And Alignment Policy

- The extractor prefers the first `t2w_compatible` mask candidate from the
  Stage 2 preprocessing validation report.
- Image and mask shape must match exactly.
- Image and mask spacing must match within tolerance.
- Empty masks are skipped and logged as failures.
- Alternate-grid masks are not used when a T2W-compatible candidate is
  available.

## Current Limitations

- This is a dependency-light first-order extractor, not full PyRadiomics texture
  extraction.
- Only original T2W images are used at this stage.
- ADC/HBV radiomics require a later resampling implementation.
- Features are not publication-grade final radiomics until the image/mask
  geometry policy and preprocessing choices are fully locked.
