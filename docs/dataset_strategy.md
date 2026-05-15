# Dataset Strategy

This document records dataset roles, label availability, access method,
supported tasks, and known limitations.

## Current Local Data

- Local raw data lives under `data/raw`.
- Raw data must remain immutable.
- The available local prostate MRI data should be treated as image data until
  labels or masks are explicitly verified.

## Planned Dataset Roles

- Public labeled prostate MRI datasets may be used for segmentation training or
  bootstrapping.
- Local project data may be used for inference, QC, radiomics, and downstream
  experiments only when the needed labels are verified.

## Label Rules

- Gland masks are segmentation labels, not cancer diagnosis labels.
- Lesion masks are detection or lesion-segmentation labels, not patient outcome
  labels unless explicitly documented.
- PI-RADS, Gleason, histopathology, and cancer-status labels must remain
  separate fields.

## Citation Status

Add dataset citations before publishing reports or claims based on any dataset.

