# Current Experiment Runbook

This runbook records the current internal PI-CAI experiment chain. It is a
methodology checkpoint, not a final manuscript result.

The project direction remains a clinically interpretable hybrid radiomics + CNN
feature framework for case-level clinically significant prostate cancer
classification. Current results support continued hybrid development, but they
do not support clinical deployment, lesion localization, radiologist
replacement, or biopsy-reduction claims.

## Execution Environment

Use the cluster repository for all data-dependent commands:

```bash
conda activate prostate-mri
cd /home/degboh/ben/prostate-mri-cancer-detection
```

Current required packages:

```text
numpy
SimpleITK
scikit-learn
torch
```

Raw PI-CAI data must remain under `data/raw/picai`. Generated feature tables,
reports, models, and processed samples are ignored by Git.

## Naming Policy

Development names such as `smoke`, `baseline`, and `25d` are allowed while a
method is still provisional. They prevent accidental overclaiming.

Final publication-facing outputs should use method-descriptive names instead of
development-stage names. Example final-name direction:

```text
data/features/radiomics_multisequence_gland_features.csv
data/features/cnn_multisequence_embeddings.csv
outputs/reports/radiomics_multisequence_cv_report.json
outputs/reports/cnn_multisequence_report.json
outputs/reports/hybrid_radiomics_cnn_report.json
outputs/models/cnn_multisequence_model.pt
```

Before final naming, confirm that the workflow is the selected final method.
Do not rename provisional outputs into final names just because a command ran
successfully.

## Architecture Policy

The current `TinyMultisequenceCNN` and `cnn-train-baseline` path are pipeline
validators. They prove that split-safe loading, paired T2W/ADC/HBV transforms,
train-only augmentation, checkpointing, embeddings, predictions, and reports
work end to end.

They are not publication-grade architecture claims.

Current publication-candidate model names are:

```text
cnn_candidate_25d_resnet
cnn_candidate_3d_densenet
hybrid_radiomics_cnn_candidate
```

CNN performance is a hypothesis. The project should only claim CNN or hybrid
benefit when candidate CNN embeddings improve over radiomics on the same
case IDs, split policy, threshold policy, and evaluation metrics.

## Regeneration Commands

### Dataset Manifest

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli manifest \
  --raw-root data/raw/picai \
  --output data/interim/picai_manifest.csv \
  --report data/interim/picai_manifest_validation.json
```

Expected current summary:

```text
total cases: 1500
T2W/ADC/HBV linked: 1500 each
clinical rows linked: 1500
gland mask cases linked: 1500
lesion mask cases linked: 1500
```

### Resampling Validation

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli resampling-validate \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --sample-size 10 \
  --report outputs/reports/resampling_validation_sample.json
```

Current validated behavior:

```text
ADC and HBV resample to T2W grid in memory.
Reference-grid gland and lesion mask candidates are available in sampled cases.
No raw data are modified.
```

### Full Whole-Gland Radiomics

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli radiomics-full-gland \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --all-cases \
  --output data/features/radiomics_gland_multisequence_full.csv \
  --failure-log outputs/reports/radiomics_gland_multisequence_full_failures.csv \
  --settings outputs/reports/radiomics_gland_multisequence_full_settings.json
```

Current result:

```text
features written: 1500
failures: 0
label counts: 1075 NO, 425 YES
```

### Radiomics-Only Cross-Validated Baseline

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli radiomics-cv-baseline \
  --features data/features/radiomics_gland_multisequence_full.csv \
  --metrics outputs/reports/radiomics_cv_metrics.json \
  --predictions outputs/reports/radiomics_cv_predictions.csv \
  --report outputs/reports/radiomics_cv_report.json \
  --target-sensitivity 0.90
```

Current full-cohort internal result:

```text
pooled held-out ROC-AUC: 0.7348
sensitivity at default threshold: 0.6565
specificity at default threshold: 0.6856
validation-selected fixed-sensitivity test behavior: sensitivity 0.8965, specificity 0.3572
```

### Controlled CNN Baseline

Current provisional command:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli cnn-train-baseline \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --sample-size-per-split 192 \
  --image-size 96 \
  --slice-window 5 \
  --max-epochs 10 \
  --batch-size 8 \
  --embedding-dim 32 \
  --augment-train \
  --device cpu \
  --embeddings data/features/cnn_baseline_25d_embeddings.csv \
  --predictions outputs/reports/cnn_baseline_25d_predictions.csv \
  --report outputs/reports/cnn_baseline_25d_report.json \
  --model outputs/models/cnn_baseline_25d_model.pt
```

Current aligned-subset result:

```text
loaded cases: 576
failures: 0
best epoch: 4
test ROC-AUC: 0.6846
default threshold sensitivity/specificity: 0.0000 / 1.0000
validation-selected threshold test sensitivity/specificity: 0.8778 / 0.3333
```

The default threshold is poorly calibrated. Use validation-selected threshold
results only as exploratory threshold behavior.

### Candidate CNN Tensor Cache

Use this to prepare ignored gland-centered tensor caches for candidate CNN
experiments. Caching is optional for small runs, but useful for repeatable
cluster experiments.

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli cnn-prepare-tensors \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --tensor-mode 25d \
  --sample-size-per-split 4 \
  --image-size 64 \
  --slice-window 5 \
  --output-root data/processed/cnn_candidate_25d_tensor_sample \
  --report outputs/reports/cnn_candidate_25d_tensor_sample_report.json
```

For a small 3D cache check:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli cnn-prepare-tensors \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --tensor-mode 3d \
  --sample-size-per-split 2 \
  --image-size 64 \
  --volume-depth 12 \
  --output-root data/processed/cnn_candidate_3d_tensor_sample \
  --report outputs/reports/cnn_candidate_3d_tensor_sample_report.json
```

### Candidate CNN Training

Run a tiny candidate first to validate the command path:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli cnn-train-candidate \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --architecture cnn_candidate_25d_resnet \
  --tensor-mode 25d \
  --sample-size-per-split 12 \
  --image-size 64 \
  --slice-window 5 \
  --max-epochs 1 \
  --batch-size 4 \
  --embedding-dim 64 \
  --augment-train \
  --device cpu \
  --embeddings data/features/cnn_candidate_25d_resnet_smoke_embeddings.csv \
  --predictions outputs/reports/cnn_candidate_25d_resnet_smoke_predictions.csv \
  --report outputs/reports/cnn_candidate_25d_resnet_smoke_report.json \
  --model outputs/models/cnn_candidate_25d_resnet_smoke_model.pt
```

Then run a controlled 2.5D candidate. `--sample-size-per-split 0` means use
all labeled cases available in each split, rather than a balanced subset:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli cnn-train-candidate \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --architecture cnn_candidate_25d_resnet \
  --tensor-mode 25d \
  --sample-size-per-split 0 \
  --image-size 96 \
  --slice-window 5 \
  --max-epochs 10 \
  --batch-size 8 \
  --embedding-dim 64 \
  --augment-train \
  --device cpu \
  --embeddings data/features/cnn_candidate_25d_resnet_embeddings.csv \
  --predictions outputs/reports/cnn_candidate_25d_resnet_predictions.csv \
  --report outputs/reports/cnn_candidate_25d_resnet_report.json \
  --model outputs/models/cnn_candidate_25d_resnet_model.pt
```

Run the 3D candidate only after the small 3D cache check succeeds and runtime is
acceptable:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli cnn-train-candidate \
  --manifest data/interim/picai_manifest.csv \
  --raw-root data/raw/picai \
  --architecture cnn_candidate_3d_densenet \
  --tensor-mode 3d \
  --sample-size-per-split 4 \
  --image-size 64 \
  --volume-depth 12 \
  --max-epochs 1 \
  --batch-size 2 \
  --embedding-dim 64 \
  --augment-train \
  --device cpu \
  --embeddings data/features/cnn_candidate_3d_densenet_smoke_embeddings.csv \
  --predictions outputs/reports/cnn_candidate_3d_densenet_smoke_predictions.csv \
  --report outputs/reports/cnn_candidate_3d_densenet_smoke_report.json \
  --model outputs/models/cnn_candidate_3d_densenet_smoke_model.pt
```

### Hybrid Radiomics + CNN Baseline

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli hybrid-ml-baseline \
  --radiomics data/features/radiomics_gland_multisequence_full.csv \
  --embeddings data/features/cnn_baseline_25d_embeddings.csv \
  --metrics outputs/reports/hybrid_ml_metrics.json \
  --predictions outputs/reports/hybrid_ml_predictions.csv \
  --report outputs/reports/hybrid_ml_report.json \
  --target-sensitivity 0.90
```

Current aligned-subset result:

```text
aligned cases: 576
radiomics-only test ROC-AUC: 0.7169
CNN embedding-only test ROC-AUC: 0.6939
hybrid test ROC-AUC: 0.7304
```

The hybrid representation modestly improves aligned-subset AUC over radiomics
alone. This supports continued hybrid development but not a strong final claim.

### Current Comparison Report

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli model-comparison-report \
  --radiomics-cv-report outputs/reports/radiomics_cv_report.json \
  --cnn-report outputs/reports/cnn_baseline_25d_report.json \
  --hybrid-report outputs/reports/hybrid_ml_report.json \
  --json-report outputs/reports/current_model_comparison.json \
  --markdown-report outputs/reports/current_model_comparison.md
```

The Markdown output is the current concise checkpoint across methods.

## Current Evidence Summary

| Representation | Scope | n | Test ROC-AUC | Sensitivity | Specificity |
| --- | --- | ---: | ---: | ---: | ---: |
| Radiomics CV | full radiomics cohort | 1500 | 0.7348 | 0.6565 | 0.6856 |
| CNN 2.5D | CNN aligned subset | 576 | 0.6846 | 0.0000 | 1.0000 |
| Aligned radiomics | radiomics + CNN subset | 576 | 0.7169 | 0.6444 | 0.6078 |
| Aligned CNN embeddings | radiomics + CNN subset | 576 | 0.6939 | 0.6000 | 0.6569 |
| Aligned hybrid | radiomics + CNN subset | 576 | 0.7304 | 0.6333 | 0.6667 |

## Interpretation

- Whole-gland multisequence radiomics is the strongest current full-cohort
  internal reference.
- The 2.5D CNN captures ranking signal, but it is not yet a tuned final CNN.
- CNN embeddings add complementary information to radiomics on the aligned
  subset, producing a modest AUC gain.
- Fixed-sensitivity threshold behavior is unstable across methods and must be
  described as exploratory.

## Allowed Claims

- A reproducible PI-CAI manifest and validation workflow was implemented.
- ADC and high b-value DWI are resampled to the T2W grid in memory for current
  radiomics/CNN workflows.
- Whole-gland multisequence radiomics provides a full-cohort internal reference.
- CNN embeddings from the current 2.5D model show some internal ranking signal.
- Hybrid radiomics + CNN embeddings modestly improve aligned-subset internal
  ROC-AUC over radiomics alone.

## Disallowed Claims

- Do not claim external validation.
- Do not claim lesion detection or tumor localization from these case-level
  workflows.
- Do not claim clinical deployment readiness.
- Do not claim radiologist replacement.
- Do not claim biopsy reduction. Fixed-sensitivity false-positive analysis is
  exploratory only.
- Do not present the current CNN as a final tuned CNN architecture.

## Cleanup Before Final Outputs

Before final names are assigned:

1. Decide whether the current CNN is final enough or whether it needs further
   tuning.
2. Decide whether final reporting uses the full radiomics cohort, the aligned
   subset, or both with separate tables.
3. Regenerate selected outputs using canonical final filenames.
4. Keep generated final outputs ignored unless they are small documentation
   summaries.
5. Update methodology docs to remove superseded prototype wording.
6. Keep the raw data, generated feature files, model checkpoints, and large
   reports out of Git.

## Safe Next Work

The safest next engineering step is candidate CNN model selection:

1. Validate tensor preparation for 2.5D and 3D modes.
2. Run a tiny 2.5D ResNet-style candidate.
3. Run a controlled 2.5D candidate on all available split cases.
4. Run a tiny 3D Dense-style candidate if CPU runtime is acceptable.
5. Re-run aligned hybrid evaluation using the strongest candidate embeddings.
6. Regenerate the comparison report with bootstrap, paired AUC delta, and
   calibration diagnostics.

Only after this should any CNN or hybrid filename be promoted from provisional
to final.
