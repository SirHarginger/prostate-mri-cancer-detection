# nnU-Net Training Setup

This document describes nnU-Net v2 setup for the current main Prostate158
datasets.

The main nnU-Net training data is now Prostate158:

- `Dataset502_Prostate158_Anatomy`: T2-only anatomy segmentation.
- `Dataset503_Prostate158_Lesion`: T2 + ADC + DWI suspicious lesion
  segmentation.

`Dataset501_ProstateROI_T2` is retained as a bootstrap/baseline artifact only.

## Expected Folder Structure

Before running nnU-Net planning or training, these structures should exist:

```text
data/nnunet/nnUNet_raw/Dataset502_Prostate158_Anatomy/
|-- dataset.json
|-- imagesTr/
|   |-- prostate158_020_0000.nii.gz
|   `-- ...
|-- labelsTr/
|   |-- prostate158_020.nii.gz
|   `-- ...
`-- imagesTs/
```

```text
data/nnunet/nnUNet_raw/Dataset503_Prostate158_Lesion/
|-- dataset.json
|-- imagesTr/
|   |-- prostate158_020_0000.nii.gz  # T2
|   |-- prostate158_020_0001.nii.gz  # ADC
|   |-- prostate158_020_0002.nii.gz  # DWI
|   `-- ...
|-- labelsTr/
|   |-- prostate158_020.nii.gz
|   `-- ...
`-- imagesTs/
```

The generated data should contain:

- Dataset502: `139` T2 images and `139` anatomy labels.
- Dataset503: `417` image channels and `139` binary suspicious lesion labels.

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

## Planning And Preprocessing

Run dataset integrity verification, planning, and preprocessing:

```bash
bash scripts/run_nnunet_dataset502_preprocess.sh
bash scripts/run_nnunet_dataset503_preprocess.sh
```

These run:

```bash
nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 503 --verify_dataset_integrity
```

## Fold 0 Local Training

On the local Quadro P1000, start with 2D fold 0 smoke-training after
preprocessing:

```bash
bash scripts/train_nnunet_dataset502_2d_fold0.sh
bash scripts/train_nnunet_dataset503_2d_fold0.sh
```

These run low-VRAM 2D plans:

```bash
nnUNetv2_train 502 2d 0 -tr nnUNetTrainer_100epochs -p nnUNetPlans_lowvram
nnUNetv2_train 503 2d 0 -tr nnUNetTrainer_100epochs -p nnUNetPlans_lowvram
```

Both local training scripts intentionally use `nnUNetTrainer_100epochs` instead
of nnU-Net's default `1000` epochs.

The helper scripts set `nnUNet_compile=false` by default. This avoids
`torch.compile`/Triton on older GPUs such as the Quadro P1000, which has CUDA
compute capability 6.1. PyTorch CUDA can still use the GPU, but Triton requires
newer hardware.

Use a higher-VRAM GPU for serious 3D full-resolution Prostate158 training. Do
not treat any model output as clinically validated.

## Troubleshooting

- If `nnUNetv2_plan_and_preprocess` or `nnUNetv2_train` is not found, activate
  the Python environment where nnU-Net v2 is installed.
- If nnU-Net reports missing `nnUNet_raw`, `nnUNet_preprocessed`, or
  `nnUNet_results`, run `source scripts/setup_nnunet_env.sh` or use the helper
  scripts, which source it internally.
- If dataset integrity verification fails, confirm Dataset502 and Dataset503
  exist under `data/nnunet/nnUNet_raw` and that file counts match the expected
  Prostate158 counts.
- If CUDA/GPU errors occur during training, verify the active PyTorch
  installation, CUDA version, and `nvidia-smi` before restarting training.
- If training crashes with a Triton or `torch.compile` message saying the GPU
  needs CUDA capability `>= 7.0`, set `nnUNet_compile=false` before training.
- If training crashes later with CUDA out-of-memory, the 4 GB Quadro P1000 may
  be too small. Use the 2D scripts for local checks or move 3D training to a
  GPU with more VRAM.
