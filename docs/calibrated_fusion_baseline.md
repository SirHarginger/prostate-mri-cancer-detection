# Calibrated Probability Fusion Baseline

This stage tests whether probability-level fusion improves over the current
regularized 2.5D CNN candidate. It is an internal PI-CAI case-level csPCa
classification experiment, not external validation.

## Command

Run from the cluster repository root after radiomics features and CNN candidate
predictions exist:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli fusion-calibrated-baseline \
  --radiomics data/features/radiomics_gland_multisequence_full.csv \
  --cnn-predictions outputs/reports/cnn_candidate_25d_resnet_regularized_seed42_predictions.csv \
  --metrics outputs/reports/fusion_calibrated_seed42_metrics.json \
  --predictions outputs/reports/fusion_calibrated_seed42_predictions.csv \
  --report outputs/reports/fusion_calibrated_seed42_report.json \
  --target-sensitivity 0.90
```

Repeat the command for seeds `123` and `2026` by changing the CNN prediction
input and output filenames.

## Baselines

The report includes:

- `radiomics_only`: logistic model trained from radiomics features.
- `cnn_probability_only`: CNN candidate probability rows.
- `weighted_probability_fusion`: validation-selected alpha ensemble of CNN and
  radiomics probabilities.
- `stacked_probability_fusion`: logistic stacker trained on radiomics
  probability and CNN probability only.

Raw feature concatenation remains a separate ablation through
`hybrid-ml-baseline`.

## Validation Policy

- Case alignment is by `case_id`.
- Labels must match between radiomics rows and CNN prediction rows.
- Alpha and stacking hyperparameters are selected on validation only.
- Test split is used only for held-out reporting.
- Generated outputs remain under ignored report paths.

## Claim Limits

Calibrated fusion can support an internal model-selection conclusion only. It
does not support external validation, clinical deployment, lesion localization,
radiologist replacement, or biopsy-reduction claims.
