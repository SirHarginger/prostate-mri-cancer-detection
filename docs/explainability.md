# Stage 7 Explainability

Stage 7 provides prototype feature-importance reporting for the current
radiomics, prototype embedding, and hybrid baselines. It does not provide CNN
visual explanations because the repository does not yet contain a trained CNN
encoder or spatial activation maps.

## Command

Run from the cluster repository root after Stage 5 features and embeddings
exist:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli explainability-report \
  --manifest data/interim/picai_manifest.csv \
  --radiomics data/features/radiomics_t2w_gland_sample.csv \
  --embeddings data/features/embeddings_t2w_prototype_sample_all25.csv \
  --json-report outputs/reports/prototype_explainability_report.json \
  --csv-report outputs/reports/prototype_feature_importance.csv \
  --top-n 20
```

Outputs are written under `outputs/reports`, which is ignored by Git.

## Method

The current report uses the absolute standardized centroid difference between
positive and negative training cases. Larger values indicate features that
separate the prototype class centroids more strongly in the training split.

This is a model-inspection aid for the nearest-centroid prototype baseline. It
is not SHAP, causal evidence, or clinical proof.

## CNN Visual Explanation Status

Grad-CAM, saliency maps, and attention visualizations are not implemented yet.
They are not technically valid until the project has a trained CNN model with
spatial feature maps and a documented image preprocessing policy.

## Claim Limits

- Feature importance is not clinical proof.
- Prototype embeddings are not trained CNN features.
- Do not infer lesion localization from centroid-based feature importance.
- Do not claim radiologist replacement, clinical deployment readiness, external
  validation, or biopsy reduction from this stage.
