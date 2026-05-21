# Prostate MRI Cancer Detection Research

Research repository for prostate MRI data engineering, nnU-Net segmentation
baselines, and downstream experimental modeling.

This project is a research prototype. It is not clinically validated and must
not be used for real clinical decisions.

## Current Focus

The main training dataset is now:

```text
data/raw/public/prostate158_train/
```

Prostate158 is converted into two nnU-Net raw datasets:

- `Dataset502_Prostate158_Anatomy`: T2 anatomy segmentation.
- `Dataset503_Prostate158_Lesion`: T2 + ADC + DWI suspicious lesion
  segmentation.

Dataset501 from MSD Task05 is kept only as a completed bootstrap/baseline
artifact. It is not the primary training direction.

The current milestone is a reproducible Prostate158 nnU-Net foundation:

1. Preserve raw downloaded data under `data/raw`.
2. Build a Prostate158 manifest under `data/manifests`.
3. Convert Prostate158 into nnU-Net raw Dataset502 and Dataset503.
4. Keep anatomy segmentation and lesion segmentation workflows separate.
5. Add QC, tests, and documentation before nnU-Net training.

## Main Commands

```bash
python scripts/build_prostate158_manifest.py \
  --input-dir data/raw/public/prostate158_train \
  --output data/manifests/prostate158_manifest.csv \
  --split-output data/manifests/splits/prostate158_nnunet_split.json \
  --overwrite

python scripts/create_nnunet_dataset502_prostate158_anatomy.py \
  --manifest data/manifests/prostate158_manifest.csv \
  --output-dir data/nnunet/nnUNet_raw/Dataset502_Prostate158_Anatomy \
  --overwrite

python scripts/create_nnunet_dataset503_prostate158_lesion.py \
  --manifest data/manifests/prostate158_manifest.csv \
  --output-dir data/nnunet/nnUNet_raw/Dataset503_Prostate158_Lesion \
  --overwrite

python scripts/evaluate_prostate158_predictions.py \
  --overwrite \
  --qc-count 6

python scripts/prepare_kaggle_prostate_mri_t2.py --overwrite
bash scripts/predict_kaggle_prostate_mri_anatomy.sh
python scripts/visualize_kaggle_auto_segmentations.py --overwrite
```

## Repository Layout

This scaffold follows [AGENTS.md](AGENTS.md):

```text
configs/      Experiment and preprocessing configs
data/         Raw, interim, processed, manifest, and external data folders
docs/         Dataset, methodology, preprocessing, experiment, and runbook docs
notebooks/    Exploratory notebooks only
outputs/      Logs, metrics, predictions, figures, and reports
scripts/      Thin command-line entrypoints
src/          Reusable Python package code
tests/        Unit tests and synthetic fixtures
```

## Data Policy

- Do not modify `data/raw` in place.
- Do not commit raw DICOM, NIfTI, masks, or large derived outputs.
- Do not invent labels. Unknown labels must remain unknown.
- Patient-level or case-level splits are required for medical imaging tasks.
