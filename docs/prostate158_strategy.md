# Prostate158 Strategy

Prostate158 is now the main dataset for nnU-Net training in this repository.
The downloaded training data is stored as immutable raw input under:

```text
data/raw/public/prostate158_train/
```

The dataset includes T2, ADC, and DWI NIfTI images, prostate anatomy masks,
and suspicious lesion masks. It is used here for research software only, not
for clinical diagnosis. Dataset details should be cited to the Prostate158
GitHub repository and paper:

- https://github.com/kbressem/prostate158
- https://www.sciencedirect.com/science/article/pii/S0010482522005789

## Main nnU-Net Datasets

The implementation creates two separate nnU-Net raw datasets instead of
collapsing anatomy and lesions into one single-label mask. These are the main
datasets for the project going forward.

```text
Dataset502_Prostate158_Anatomy
input:  T2
label:  t2_anatomy_reader1
labels: background, anatomy_label_1, anatomy_label_2
```

```text
Dataset503_Prostate158_Lesion
input:  T2, ADC, DWI
label:  adc_tumor_reader1
labels: background, suspicious_lesion
```

The anatomy label names are intentionally conservative until the official
numeric mapping is citation-confirmed. The ADC reader1 lesion mask is the first
lesion target because it exists for all 139 cases and is binary.

## Build Commands

Build the manifest and split file:

```bash
python scripts/build_prostate158_manifest.py \
  --input-dir data/raw/public/prostate158_train \
  --output data/manifests/prostate158_manifest.csv \
  --split-output data/manifests/splits/prostate158_nnunet_split.json \
  --overwrite
```

Create anatomy Dataset502:

```bash
python scripts/create_nnunet_dataset502_prostate158_anatomy.py \
  --manifest data/manifests/prostate158_manifest.csv \
  --output-dir data/nnunet/nnUNet_raw/Dataset502_Prostate158_Anatomy \
  --overwrite
```

Create lesion Dataset503:

```bash
python scripts/create_nnunet_dataset503_prostate158_lesion.py \
  --manifest data/manifests/prostate158_manifest.csv \
  --output-dir data/nnunet/nnUNet_raw/Dataset503_Prostate158_Lesion \
  --overwrite
```

## Outputs

```text
data/manifests/prostate158_manifest.csv
data/manifests/splits/prostate158_nnunet_split.json
data/nnunet/nnUNet_raw/Dataset502_Prostate158_Anatomy/
data/nnunet/nnUNet_raw/Dataset503_Prostate158_Lesion/
notebooks/exploratory/02_visualize_prostate158_samples.ipynb
```

The official Prostate158 train/valid split is preserved in the manifest and
split JSON. All 139 cases are placed in nnU-Net `imagesTr/labelsTr`; the split
can be copied into the nnU-Net preprocessed dataset later before training.

## Safety Notes

- Raw files under `data/raw/public/prostate158_train` are never modified.
- Dataset501 outputs, checkpoints, logs, and plans are not modified.
- Negative lesion cases are written with generated zero-valued masks using the
  ADC image geometry. This avoids copying raw `empty.nii.gz` masks that were
  observed to have affine mismatches.
- Dataset501 outputs can be kept as bootstrap history, but they are not the
  main nnU-Net training direction.

## nnU-Net Planning And Local Training

Set nnU-Net paths:

```bash
source scripts/setup_nnunet_env.sh
```

Plan and preprocess anatomy Dataset502:

```bash
bash scripts/run_nnunet_dataset502_preprocess.sh
```

Plan and preprocess lesion Dataset503:

```bash
bash scripts/run_nnunet_dataset503_preprocess.sh
```

On the local 4 GB Quadro P1000, start with 2D fold 0 smoke-training after
preprocessing. These scripts use `nnUNetTrainer_100epochs`, not the nnU-Net
default `1000` epochs:

```bash
bash scripts/train_nnunet_dataset502_2d_fold0.sh
bash scripts/train_nnunet_dataset503_2d_fold0.sh
```

Use a higher-VRAM GPU for serious 3D full-resolution training.

## Evaluation

After exporting Dataset502 predictions, evaluate anatomy segmentation against
the nnU-Net raw labels:

```bash
python scripts/evaluate_prostate158_predictions.py \
  --overwrite \
  --qc-count 6
```

The script writes:

```text
outputs/metrics/prostate158_dataset502_anatomy_metrics.csv
outputs/metrics/prostate158_dataset502_anatomy_summary.json
outputs/figures/qc/prostate158_predictions/
```

Use the validation split Dice values as the headline anatomy result. The
all-case summary is useful for debugging, but it includes training cases and
should not be presented as the main validation metric.
