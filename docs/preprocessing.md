# Preprocessing

This document will record DICOM/NIfTI conversion, orientation handling,
resampling, normalization, alignment checks, and QC outputs.

## Rules

- Never preprocess in `data/raw`.
- Write temporary conversions to `data/interim`.
- Write model-ready files to `data/processed`.
- Preserve metadata where practical.
- Log spacing, orientation, shape, and failed conversions.

## MSD Task05 Prostate Binary ROI Labels

MSD Task05 prostate labels contain separate nonzero labels for prostate zones.
For whole-prostate ROI segmentation, convert every nonzero label voxel to `1`
and keep background as `0`.

Example:

```bash
python scripts/convert_msd_prostate_labels_to_binary_roi.py \
  --input-dir data/raw/public/Task05_Prostate \
  --output-dir data/interim/public/msd_prostate_binary_roi
```

Outputs:

- `data/interim/public/msd_prostate_binary_roi/labelsTr/*.nii.gz`
- `data/interim/public/msd_prostate_binary_roi/manifest.csv`

The raw MSD files under `data/raw` are not modified.

## Archived Bootstrap: nnU-Net Dataset501 Prostate ROI T2

Dataset501 was created earlier as a bootstrap whole-prostate ROI baseline from
MSD Task05. It is kept for history and comparison only; Prostate158 is now the
main nnU-Net training data.

The archived conversion command was:

```bash
python scripts/create_nnunet_dataset501_prostate_roi_t2.py \
  --msd-dir data/raw/public/Task05_Prostate \
  --binary-roi-dir data/interim/public/msd_prostate_binary_roi \
  --output-dir data/nnunet/nnUNet_raw/Dataset501_ProstateROI_T2 \
  --overwrite
```

This writes:

- `data/nnunet/nnUNet_raw/Dataset501_ProstateROI_T2/dataset.json`
- `imagesTr/*_0000.nii.gz` with only MSD channel `0` / T2
- `labelsTr/*.nii.gz` with binary prostate ROI labels
- `imagesTs/*_0000.nii.gz` with only MSD channel `0` / T2

The original MSD images and prepared binary ROI labels are read-only inputs.
The labels are whole-prostate ROI masks, not tumor masks.

## Prostate158 Anatomy And Lesion Datasets

Prostate158 is the main nnU-Net training dataset for prostate anatomy and
suspicious lesion segmentation. The conversion creates two nnU-Net raw
datasets:

- `Dataset502_Prostate158_Anatomy`: T2 image with anatomy labels.
- `Dataset503_Prostate158_Lesion`: T2, ADC, and DWI images with binary ADC
  reader1 lesion labels.

The raw Prostate158 files are not modified. See `docs/prostate158_strategy.md`
for commands, assumptions, label cautions, and QC guidance.
