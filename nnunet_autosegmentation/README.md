# nnU-Net Autosegmentation Workspace

This workspace is isolated from the main PI-CAI case-level classification
pipeline. It is for prostate MRI autosegmentation experiments, external data
download, nnU-Net inference, and downstream feature extraction from predicted
masks.

Do not commit downloaded imaging data, nnU-Net preprocessed data, checkpoints,
predicted masks, generated features, or QC images.

## Folder Policy

Generated and downloaded content belongs under ignored paths:

```text
nnunet_autosegmentation/data/
nnunet_autosegmentation/outputs/
```

Code, configs, and documentation can be committed.

## Intended Flow

1. Download an external prostate MRI dataset on the cluster.
2. Convert image data into the nnU-Net input layout.
3. Run nnU-Net prediction using a trained prostate segmentation model.
4. Store predicted masks under ignored data paths.
5. Extract features using predicted masks.
6. Keep feature outputs ignored unless a small schema/example is intentionally
   added as documentation.

## Candidate External Dataset

PROSTATEx from TCIA is a public prostate MRI dataset suitable for external
unsegmented-data experiments. Download it on the cluster only.

Official collection page:

```text
https://www.cancerimagingarchive.net/collection/prostatex/
```

## nnU-Net Requirement

nnU-Net prediction requires a trained model. This workspace assumes one of:

- a trained prostate segmentation model is available, or
- a prostate segmentation model will be trained separately using labeled data.

The current main classification pipeline should not depend on this workspace
until segmentation quality is validated.
