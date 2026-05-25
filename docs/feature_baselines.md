# Stage 5 Feature Fusion And Prototype Baselines

Stage 5 aligns Stage 3 radiomics features and Stage 4 prototype embeddings by
`case_id`, then runs dependency-free prototype classifiers on the same cases and
splits.

These are implementation-check baselines. The embedding table is still marked
`prototype_not_trained_cnn`, so the embedding-only and hybrid results must not be
reported as final CNN-only or final hybrid model performance.

## Required Inputs

- `data/interim/picai_manifest.csv`
- a whole-gland radiomics table, for example
  `data/features/radiomics_t2w_gland_sample.csv`
- a prototype embedding table, preferably extracted for the same sample cases,
  for example `data/features/embeddings_t2w_prototype_sample_all25.csv`

Generated features and reports remain ignored by Git.

## Recommended Sample Commands

First regenerate Stage 4 embeddings for all cases from the Stage 2 sample report
so the baseline has the same 25 cases as the gland radiomics sample:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli embedding-extract \
  --manifest data/interim/picai_manifest.csv \
  --preprocessing-report outputs/reports/preprocessing_fold_sample_validation.json \
  --raw-root data/raw/picai \
  --sequence t2w \
  --embedding-dim 32 \
  --all-cases \
  --output data/features/embeddings_t2w_prototype_sample_all25.csv \
  --provenance outputs/reports/embeddings_t2w_prototype_sample_all25_provenance.json \
  --report outputs/reports/embeddings_t2w_prototype_sample_all25_report.json
```

Then run aligned baselines:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli baseline-evaluate \
  --manifest data/interim/picai_manifest.csv \
  --radiomics data/features/radiomics_t2w_gland_sample.csv \
  --embeddings data/features/embeddings_t2w_prototype_sample_all25.csv \
  --metrics outputs/reports/prototype_baseline_metrics.json \
  --predictions outputs/reports/prototype_baseline_predictions.csv \
  --report outputs/reports/prototype_baseline_report.json
```

## What Is Checked

- Rows are aligned by `case_id`.
- The same aligned cases and split labels are used for radiomics-only,
  prototype-embedding-only, and hybrid baselines.
- Labels, fold IDs, split IDs, paths, ROI names, sequence names, encoder names,
  and augmentation flags are excluded from predictive features.
- The training split must contain both positive and negative cases.
- Missing or unknown labels are excluded and reported.

## Current Model

The classifier is a small nearest-centroid prototype:

- standardize features using training-split statistics only
- compute positive and negative train centroids
- score validation/test rows by relative distance to class centroids

This is suitable for validating table alignment, split safety, and metric
plumbing. It is not a final model-selection method.

## Claim Limits

- Do not report these as final radiomics/CNN/hybrid results.
- Do not use the prototype embedding baseline as a trained CNN-only baseline.
- Do not make clinical, localization, deployment, or biopsy-reduction claims
  from this stage.
