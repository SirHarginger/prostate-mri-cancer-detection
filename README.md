# Prostate MRI Cancer Detection Research

Research repository for classifier-first prostate MRI cancer detection using
PI-CAI bpMRI data.

This project is a research prototype. It is not clinically validated and must
not be used for real clinical decisions.

## Current Focus

The project has pivoted to a leakage-safe, case-level classifier for clinically
significant prostate cancer.

Primary dataset:

```text
PI-CAI public prostate MRI dataset
```

Cluster data root:

```text
/home/degboh/prostate_mri_cancer_detection
```

Current all-fold image manifest:

```text
/home/degboh/prostate_mri_cancer_detection/data/features/picai_all_folds_image_manifest.csv
```

Current all-fold status:

- 1500 PI-CAI cases.
- 1500 cases with core bpMRI: T2W, ADC, and HBV.
- 1075 non-csPCa cases.
- 425 csPCa cases.

The current research design is documented in:

```text
docs/classification/RESEARCH_FRAMEWORK.md
```

## Scientific Guardrails

- Do not use lesion masks, lesion crops, lesion-mask volume, or lesion-derived
  radiomics for binary csPCa detection.
- Do not use `case_isup_int`, Gleason, pathology-derived, diagnosis-derived, or
  target-derived variables as predictors.
- Do not split slices independently across train and validation.
- Use patient/case-level splits only.
- Use whole-gland masks only as anatomical support for cropping or
  whole-gland radiomics.
- Keep generated data, feature CSVs, model artifacts, outputs, logs, and
  reports outside Git.

## Main Commands

Build the all-fold image manifest on the cluster:

```bash
python scripts/classification/build_picai_image_manifest.py \
  --mask-manifest /home/degboh/prostate_mri_cancer_detection/data/features/picai_mask_manifest.csv \
  --images-root /home/degboh/prostate_mri_cancer_detection/data/raw/picai/images \
  --output /home/degboh/prostate_mri_cancer_detection/data/features/picai_all_folds_image_manifest.csv
```

Run the deep-learning readiness audit on the cluster:

```bash
python scripts/classification/audit_deep_learning_readiness.py \
  --manifest /home/degboh/prostate_mri_cancer_detection/data/features/picai_all_folds_image_manifest.csv \
  --output /home/degboh/prostate_mri_cancer_detection/outputs/deep_learning_readiness_all_folds.json \
  --limit 50
```

## Repository Layout

This scaffold follows [AGENTS.md](AGENTS.md):

```text
configs/      Experiment and preprocessing configs
data/         Raw, interim, processed, manifest, and external data folders
docs/         Classifier research design, guardrails, status, and workflow docs
notebooks/    Exploratory notebooks only
outputs/      Logs, metrics, predictions, figures, and reports
scripts/      Classifier-first command-line entrypoints
src/          Reusable Python package code
tests/        Unit tests and synthetic fixtures
```

## Data Policy

- Do not modify `data/raw` in place.
- Do not commit raw DICOM, NIfTI, masks, or large derived outputs.
- Do not invent labels. Unknown labels must remain unknown.
- Patient-level or case-level splits are required for medical imaging tasks.
