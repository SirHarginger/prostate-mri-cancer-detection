# Runbook

Operational notes for running project workflows.

## Scaffold Commands

Some scaffold scripts define planned interfaces for later workflows; implemented
dataset-specific scripts are documented in their own sections.

Examples:

```bash
python scripts/build_manifest.py --dataset prostate_mri --input-dir data/raw --output data/manifests/prostate_mri_manifest.csv --dry-run
python scripts/verify_dataset.py --manifest data/manifests/prostate_mri_manifest.csv --dry-run
python scripts/preprocess_dataset.py --config configs/preprocess/example.yaml --dry-run
python scripts/train_segmentation.py --config configs/train/example.yaml --dry-run
python scripts/evaluate_segmentation.py --config configs/eval/example.yaml --dry-run
python scripts/visualize_qc.py --manifest data/manifests/prostate_mri_manifest.csv --output-dir outputs/figures/qc --dry-run
```

## Main Prostate158 nnU-Net Workflow

Build the Prostate158 manifest:

```bash
python scripts/build_prostate158_manifest.py \
  --input-dir data/raw/public/prostate158_train \
  --output data/manifests/prostate158_manifest.csv \
  --split-output data/manifests/splits/prostate158_nnunet_split.json \
  --overwrite
```

Create the anatomy nnU-Net raw dataset:

```bash
python scripts/create_nnunet_dataset502_prostate158_anatomy.py \
  --manifest data/manifests/prostate158_manifest.csv \
  --output-dir data/nnunet/nnUNet_raw/Dataset502_Prostate158_Anatomy \
  --overwrite
```

Create the lesion nnU-Net raw dataset:

```bash
python scripts/create_nnunet_dataset503_prostate158_lesion.py \
  --manifest data/manifests/prostate158_manifest.csv \
  --output-dir data/nnunet/nnUNet_raw/Dataset503_Prostate158_Lesion \
  --overwrite
```

Set nnU-Net environment variables:

```bash
source scripts/setup_nnunet_env.sh
```

Plan and preprocess the main Prostate158 datasets:

```bash
bash scripts/run_nnunet_dataset502_preprocess.sh
bash scripts/run_nnunet_dataset503_preprocess.sh
```

Start local 2D fold 0 training after preprocessing. These scripts use
`nnUNetTrainer_100epochs`:

```bash
bash scripts/train_nnunet_dataset502_2d_fold0.sh
bash scripts/train_nnunet_dataset503_2d_fold0.sh
```

Evaluate exported Dataset502 anatomy predictions against the Prostate158
labels. The validation split metrics are the main presentation numbers because
the full 139-case output also includes training cases.

```bash
python scripts/evaluate_prostate158_predictions.py \
  --overwrite \
  --qc-count 6
```

Use a higher-VRAM GPU for serious 3D full-resolution training.

Validate code and generated datasets:

```bash
python3 -m compileall src scripts tests
python3 -m pytest
```

Open and run the QC notebook:

```text
notebooks/exploratory/02_visualize_prostate158_samples.ipynb
```

## Kaggle PROSTATE_MRI Auto-Segmentation

The assignment target data is restored under:

```text
data/raw/world-wide-covid-dataset/PROSTATE_MRI/
```

Raw DICOM files remain immutable. Convert the axial T2 series to nnU-Net
inference images under `data/interim`:

```bash
python scripts/prepare_kaggle_prostate_mri_t2.py --overwrite
```

Run the trained Dataset502 anatomy model on those converted Kaggle T2 images:

```bash
bash scripts/predict_kaggle_prostate_mri_anatomy.sh
```

The auto-segmented anatomy masks are written to:

```text
outputs/predictions/kaggle_prostate_mri_anatomy_auto/
```

Create QC overlays for presentation:

```bash
python scripts/visualize_kaggle_auto_segmentations.py --overwrite
```

For interactive visual review in Jupyter, open:

```text
notebooks/exploratory/03_visualize_kaggle_auto_segmentations.ipynb
```
