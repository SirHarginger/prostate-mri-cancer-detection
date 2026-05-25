# Production Stage B Preprocessed Sample

This stage writes a tiny processed sample for manual inspection after
SimpleITK resampling validation has passed. It is intentionally small and does
not process the full PI-CAI dataset.

## What Is Written

For each selected case:

- ADC resampled to the T2W grid
- high b-value DWI resampled to the T2W grid
- JSON provenance/report

T2W images are not copied. Masks are not copied or resampled in this stage;
reference-grid mask candidates are recorded in the report.

## Command

Run from the cluster repository root:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli preprocess-sample \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --sample-size 5 \
  --output-root data/processed/picai_sample \
  --report outputs/reports/preprocessed_sample_report.json
```

To write specific cases:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli preprocess-sample \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --case-id 10000_1000000 \
  --case-id 10005_1000005 \
  --output-root data/processed/picai_sample \
  --report outputs/reports/preprocessed_sample_selected_report.json
```

Outputs are ignored by Git:

- `data/processed/picai_sample/`
- `outputs/reports/preprocessed_sample_report.json`

## Safety Rules

- The command refuses to write inside `data/raw`.
- Raw images and labels are never modified.
- The command overwrites only generated files under the selected processed
  output root.
- Do not scale to all 1500 cases until this sample has been inspected.

## Success Criteria

- Cases with issues: `0`
- ADC written count equals selected case count
- HBV written count equals selected case count
- ADC/HBV output signatures match each case's T2W reference grid
- Reference-grid gland and lesion mask candidates are recorded
- `git status --short` remains clean
