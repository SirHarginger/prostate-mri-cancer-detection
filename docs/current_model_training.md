# Current Model Training

This is the clean command path for the current leading model family:
regularized 2.5D CNN case-level csPCa classification.

The model config is:

```text
config/cnn_candidate_25d_regularized.json
```

It records the model architecture, preprocessing size, training hyperparameters,
seed, and output paths. Edit that config to run a controlled experiment.

## Train

Run from the cluster repository root:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli cnn-train-candidate \
  --config config/cnn_candidate_25d_regularized.json
```

The command writes ignored outputs:

```text
data/features/cnn_candidate_25d_resnet_regularized_seed42_embeddings.csv
outputs/reports/cnn_candidate_25d_resnet_regularized_seed42_predictions.csv
outputs/reports/cnn_candidate_25d_resnet_regularized_seed42_report.json
outputs/models/cnn_candidate_25d_resnet_regularized_seed42_model.pt
```

## Current Role

This model is the current leading CNN candidate. Radiomics-only, raw hybrid, and
calibrated fusion remain ablations until final model selection is locked.

## Claim Limits

This is an internal PI-CAI research model. It is not externally validated and
does not support clinical deployment, lesion localization, radiologist
replacement, or biopsy-reduction claims.
