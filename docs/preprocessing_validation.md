# Stage 2 Preprocessing Validation

Stage 2 validates that the Stage 1 manifest can support reproducible
preprocessing. It performs lightweight header checks only. It does not resample,
normalize, crop, augment, extract radiomics, train CNNs, or write processed
image datasets.

The current cluster environment does not require SimpleITK for this validation.
The validator reads `.mha`, `.mhd`, `.nii`, and `.nii.gz` headers with standard
library code and records shape, spacing, direction where available, origin, and
element type.

## Command

Run from the cluster repository root after the Stage 1 manifest exists:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli preprocessing-validate \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --sample-size 10 \
  --report outputs/reports/preprocessing_sample_validation.json
```

To validate specific cases:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli preprocessing-validate \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --case-id 10000_1000000 \
  --case-id 10001_1000001 \
  --report outputs/reports/preprocessing_selected_cases.json
```

Reports are written under `outputs/reports`, which is ignored by Git.

## Checks

- T2W, ADC, and high b-value DWI paths resolve from the manifest.
- Target modality headers are readable without voxel loading.
- ADC and high b-value DWI shape, spacing, and direction are compared with T2W
  where metadata are available.
- Gland and lesion mask paths resolve where present.
- Mask shape and spacing are compared with T2W where mask headers are readable.
- Missing paths, unreadable headers, and mismatches are reported per case.

## Normalization Plan

No normalization is applied in Stage 2. The report records the planned default:
per-case and per-modality percentile clipping followed by z-score
normalization, preferably inside the prostate gland ROI when available.
Dataset-level statistics must not be computed across validation or test folds.

## ROI Plan

Stage 2 only validates ROI availability and header compatibility. Gland masks
are candidates for later case-level prostate ROI cropping or foreground
statistics. Lesion masks are candidates for later lesion-aware experiments and
radiomics only after alignment and mask validity are confirmed.

## Claim Limits

This stage does not prove preprocessing quality, model performance, clinical
utility, lesion localization, or biopsy-reduction potential. It only checks
whether the manifest-linked files are structurally ready for a minimal
preprocessing implementation.
