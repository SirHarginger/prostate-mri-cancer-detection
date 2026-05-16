# nnU-Net Dataset501 Training Setup

This document describes the setup commands for the nnU-Net v2 baseline dataset:
`Dataset501_ProstateROI_T2`.

Dataset501 is a T2-only whole-prostate ROI segmentation dataset derived from
MSD Task05 Prostate. It uses MSD channel `0` only and binary prostate ROI masks
created by merging the original peripheral-zone and transition-zone labels.
These labels are not tumor masks.

## Expected Folder Structure

Before running nnU-Net planning or training, this structure should exist:

```text
data/nnunet/nnUNet_raw/Dataset501_ProstateROI_T2/
|-- dataset.json
|-- imagesTr/
|   |-- prostate_000_0000.nii.gz
|   `-- ...
|-- labelsTr/
|   |-- prostate_000.nii.gz
|   `-- ...
`-- imagesTs/
    |-- prostate_000_0000.nii.gz
    `-- ...
```

The generated data should contain:

- `32` training T2 images in `imagesTr`.
- `32` binary prostate ROI labels in `labelsTr`.
- `16` test T2 images in `imagesTs`.

## Python Environment

Activate the project environment before running nnU-Net commands. Use whichever
environment contains `nnunetv2`, `torch`, `nibabel`, and the CUDA/PyTorch stack
you intend to train with.

Example with the local virtual environment:

```bash
source .venv/bin/activate
```

Check command availability:

```bash
command -v nnUNetv2_plan_and_preprocess
command -v nnUNetv2_train
```

If either command is missing, install nnU-Net v2 into the active environment
before continuing.

## Environment Variables

nnU-Net v2 expects these variables:

```bash
export nnUNet_raw="$PWD/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/outputs/nnunet_results"
```

The helper script sets them and creates missing folders:

```bash
source scripts/setup_nnunet_env.sh
```

If you execute the script instead of sourcing it, the variables are only set
inside that process; the preprocess/train helper scripts source it internally.

## Planning and Preprocessing

Run dataset integrity verification, planning, and preprocessing:

```bash
bash scripts/run_nnunet_dataset501_preprocess.sh
```

This runs:

```bash
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
```

## Fold 0 Baseline Training

Run the first 3D full-resolution baseline fold:

```bash
bash scripts/train_nnunet_dataset501_fold0.sh
```

This runs:

```bash
nnUNetv2_train 501 3d_fullres 0 -p nnUNetPlans_lowvram
```

The helper script first creates `nnUNetPlans_lowvram.json` from the original
plans and reduces the 3D full-resolution batch size from `2` to `1`. It also
sets `nnUNet_compile=false` by default. This avoids `torch.compile`/Triton on
older GPUs such as the Quadro P1000, which has CUDA compute capability 6.1.
PyTorch CUDA can still use the GPU, but Triton requires newer hardware.

If 3D still runs out of memory on a 4 GB GPU, use the 2D fallback:

```bash
bash scripts/train_nnunet_dataset501_2d_fold0.sh
```

This runs:

```bash
nnUNetv2_train 501 2d 0 -p nnUNetPlans_lowvram
```

Do not treat the resulting model as clinically validated. This is a research
segmentation baseline for prostate ROI masks.

## Troubleshooting

- If `nnUNetv2_plan_and_preprocess` or `nnUNetv2_train` is not found, activate
  the Python environment where nnU-Net v2 is installed.
- If nnU-Net reports missing `nnUNet_raw`, `nnUNet_preprocessed`, or
  `nnUNet_results`, run `source scripts/setup_nnunet_env.sh` or use the helper
  scripts, which source it internally.
- If dataset integrity verification fails, confirm `Dataset501_ProstateROI_T2`
  exists under `data/nnunet/nnUNet_raw` and that file counts match the expected
  `32` train images, `32` train labels, and `16` test images.
- If CUDA/GPU errors occur during training, verify the active PyTorch
  installation, CUDA version, and `nvidia-smi` before restarting training.
- If training crashes with a Triton or `torch.compile` message saying the GPU
  needs CUDA capability `>= 7.0`, set `nnUNet_compile=false` before training:

  ```bash
  export nnUNet_compile=false
  bash scripts/train_nnunet_dataset501_fold0.sh
  ```

- If training crashes later with CUDA out-of-memory, the 4 GB Quadro P1000 may
  still be too small for `3d_fullres`. In that case, run
  `bash scripts/train_nnunet_dataset501_2d_fold0.sh` first or use a GPU with
  more VRAM for the 3D baseline.
