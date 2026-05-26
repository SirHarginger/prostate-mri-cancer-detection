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

1. Prepare PI-CAI T2W/ADC/HBV images and gland/lesion labels as
   `Dataset910_PI_CAIGlandLesion`.
2. Train nnU-Net on PI-CAI labels. Start with CPU-safe `2d`; use `3d_fullres`
   later when GPU runtime is available.
3. Convert the external PROSTATE-MRI dataset into the same three-channel
   layout. ADC is explicitly zero-filled if unavailable and reported as such.
4. Run prediction with the PI-CAI-trained model.
5. Extract label-specific features for label `1` prostate gland and label `2`
   csPCa lesion candidate. Do not merge labels with `mask > 0`.
6. Open the QC notebook and visually inspect overlays before making claims.

## PI-CAI Training Preparation

```bash
python nnunet_autosegmentation/scripts/prepare_picai_nnunet_training.py \
  --config nnunet_autosegmentation/config/picai_gland_lesion_nnunet_config.json \
  --limit 5
```

Then, on the cluster:

```bash
export nnUNet_raw="$PWD/nnunet_autosegmentation/data/nnunet_raw"
export nnUNet_preprocessed="$PWD/nnunet_autosegmentation/data/nnunet_preprocessed"
export nnUNet_results="$PWD/nnunet_autosegmentation/data/nnunet_results"

nnUNetv2_plan_and_preprocess -d 910 --verify_dataset_integrity
nnUNetv2_train 910 2d 0 -device cpu
```

## External Dataset

The downloaded Kaggle folder used in this project is named like a COVID
dataset, but its extracted DICOM metadata is the PROSTATE-MRI collection.

```text
https://doi.org/10.7937/K9/TCIA.2016.6046GUDv
```

Prepare external inference inputs:

```bash
python nnunet_autosegmentation/scripts/prepare_nnunet_dataset.py \
  --config nnunet_autosegmentation/config/picai_gland_lesion_nnunet_config.json
```

Predict and extract features:

```bash
bash nnunet_autosegmentation/scripts/run_nnunet_predict.sh \
  nnunet_autosegmentation/config/picai_gland_lesion_nnunet_config.json

python nnunet_autosegmentation/scripts/extract_features_from_masks.py \
  --config nnunet_autosegmentation/config/picai_gland_lesion_nnunet_config.json
```

## nnU-Net Requirement

nnU-Net prediction requires a trained model. The publication-grade path trains
on PI-CAI labels first; the pretrained model downloader remains only a quick
mechanical smoke-test path and should not be used for lesion claims.

The current main classification pipeline should not depend on this workspace
until segmentation quality is validated.
