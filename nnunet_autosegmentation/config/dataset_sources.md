# Dataset Sources

## PROSTATEx

Use PROSTATEx as the first external unsegmented prostate MRI source.

Official TCIA collection page:

```text
https://www.cancerimagingarchive.net/collection/prostatex/
```

Download data on the cluster into:

```text
nnunet_autosegmentation/data/raw/prostatex/
```

Do not commit downloaded DICOM/NIfTI files.

## Training Segmentation Model

nnU-Net prediction needs a trained model. Potential labeled sources include:

- PI-CAI anatomical whole-gland masks already present in the main cluster data.
- Medical Segmentation Decathlon prostate task if a zone/gland segmentation
  experiment is planned separately.

Do not mix segmentation training outputs with the main classification outputs.
