# Production Stage A Resampling Validation

This stage bridges the prototype workflow to a serious end-to-end pipeline. It
validates, in memory, that ADC and high b-value DWI can be resampled to each
case's T2W grid using SimpleITK. It does not write processed images or create a
full processed dataset.

## Requirement

SimpleITK must be installed in the cluster environment:

```bash
python -m pip install SimpleITK
```

## Command

Run from the cluster repository root:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli resampling-validate \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --sample-size 10 \
  --report outputs/reports/resampling_validation_sample.json
```

To validate specific cases:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli resampling-validate \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --case-id 10000_1000000 \
  --case-id 10005_1000005 \
  --report outputs/reports/resampling_validation_selected.json
```

Reports are written under `outputs/reports`, which is ignored by Git.

## Policy Validated

- T2W is the reference grid.
- ADC and high b-value DWI are resampled in memory to the T2W grid.
- Linear interpolation is used for images.
- Nearest-neighbor interpolation is reserved for masks.
- Processed images are not written in this stage.
- Mask candidates are checked for existing T2W-grid compatibility.

## Success Criteria

- T2W, ADC, and high b-value DWI are readable.
- In-memory resampled ADC/HBV signatures match the T2W reference grid.
- At least one gland mask candidate matches the T2W grid.
- At least one lesion mask candidate matches the T2W grid where expected.
- No files are written under `data/processed`.

## Next Step After Validation

If this report passes on a representative sample, the next serious
implementation is a controlled preprocessing writer that stores a small,
documented sample of resampled ADC/HBV outputs under ignored `data/processed`
for inspection before any full-dataset processing.
