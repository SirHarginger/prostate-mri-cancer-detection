# Methodology

This document describes the current research methodology for the Prostate158
nnU-Net pivot.

## Main Workflow

The main training data is `data/raw/public/prostate158_train/`. It is converted
into two nnU-Net datasets:

- `Dataset502_Prostate158_Anatomy`: T2-only anatomy segmentation.
- `Dataset503_Prostate158_Lesion`: T2, ADC, and DWI suspicious lesion
  segmentation.

These tasks are trained separately because anatomy zones and suspicious lesions
are different label concepts and may spatially overlap.

## Principles

- Start with the Prostate158 nnU-Net anatomy and lesion baselines.
- Use task-appropriate metrics.
- Keep anatomy segmentation, lesion segmentation, and later classification
  separate.
- Report limitations clearly.
- Avoid clinical claims beyond the available evidence.

## Dataset501 Status

The MSD-derived `Dataset501_ProstateROI_T2` remains a completed bootstrap
baseline for whole-prostate ROI localization. It is not the primary training
dataset for the current nnU-Net direction.
