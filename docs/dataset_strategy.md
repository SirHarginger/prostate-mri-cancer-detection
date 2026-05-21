# Dataset Strategy

This document records dataset roles, label availability, access method,
supported tasks, and known limitations.

## Current Local Data

- The main nnU-Net training data is `data/raw/public/prostate158_train/`.
- Prostate158 raw data must remain immutable.
- The current Prostate158 download has verified T2, ADC, DWI images, anatomy
  masks, and binary ADC reader1 suspicious lesion masks.
- MSD Task05/Dataset501 is retained as a bootstrap baseline artifact, not the
  main training direction.

## Dataset Roles

- `Dataset502_Prostate158_Anatomy`: primary nnU-Net anatomy segmentation
  dataset using T2 and anatomy labels.
- `Dataset503_Prostate158_Lesion`: primary nnU-Net suspicious lesion
  segmentation dataset using T2, ADC, DWI, and ADC reader1 lesion labels.
- Local unsegmented project data is for later inference/QC only after the
  Prostate158 models are trained and validated.

## Label Rules

- Anatomy masks are segmentation labels, not cancer diagnosis labels.
- Suspicious lesion masks are lesion-segmentation labels, not patient outcome
  labels unless explicitly documented.
- Anatomy labels and lesion labels must remain separate because they describe
  different concepts and may spatially overlap.
- PI-RADS, Gleason, histopathology, and cancer-status labels must remain
  separate fields.

## Citation Status

Use Prostate158 citations before publishing reports or claims:

- https://github.com/kbressem/prostate158
- https://www.sciencedirect.com/science/article/pii/S0010482522005789
