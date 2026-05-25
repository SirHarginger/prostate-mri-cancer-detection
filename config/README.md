# Experiment Configs

This directory stores reproducible experiment configs.

The current primary model config is:

```text
config/cnn_candidate_25d_regularized.json
```

Train it from the repository root:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli cnn-train-candidate \
  --config config/cnn_candidate_25d_regularized.json
```

## Editing Rules

- Change hyperparameters in the config file, not by editing long shell commands.
- Keep raw data paths relative to the repository root.
- Keep generated outputs under ignored locations:
  - `data/features/`
  - `outputs/reports/`
  - `outputs/models/`
- Create a new config file for a meaningfully different experiment.
- Do not rename a candidate config to a final model config until model selection
  is complete.

## Current Model Role

`cnn_candidate_25d_regularized.json` is the leading CNN candidate configuration.
It is still an internal PI-CAI research model, not a clinically deployed model.
