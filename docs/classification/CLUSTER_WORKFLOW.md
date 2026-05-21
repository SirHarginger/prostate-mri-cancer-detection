# Cluster Workflow

This project uses a local/GitHub/cluster loop.

## Roles

- Local machine: code editing with VS Code and Codex.
- GitHub: source-of-truth sync point between machines.
- Cluster: heavy PI-CAI data processing, PyRadiomics extraction, training,
  evaluation, and artifact generation.

## Cluster Paths

Code repository:

```text
/home/degboh/projects/prostate-mri-cancer-detection
```

Storage root:

```text
/home/degboh/prostate_mri_cancer_detection
```

Large data and output locations:

```text
/home/degboh/prostate_mri_cancer_detection/data/raw/picai
/home/degboh/prostate_mri_cancer_detection/data/features
/home/degboh/prostate_mri_cancer_detection/outputs
/home/degboh/prostate_mri_cancer_detection/artifacts
/home/degboh/prostate_mri_cancer_detection/logs
/home/degboh/prostate_mri_cancer_detection/reports
```

## Environment

Activate the cluster conda environment before running classifier workflows:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate prostate-ml
```

## Local Coding Workflow

On the local machine:

```bash
git status -sb
# edit code or docs
python -m compileall src scripts tests
python -m pytest
git status -sb
git add <changed-files>
git commit -m "<message>"
git push
```

Only run commands that are appropriate for the local environment. Do not create
or commit generated PI-CAI data locally.

## Cluster Pull and Run Workflow

On the cluster:

```bash
cd /home/degboh/projects/prostate-mri-cancer-detection
git pull
source ~/miniforge3/etc/profile.d/conda.sh
conda activate prostate-ml
```

Current leakage-safe fold0 feature extraction command:

```bash
python scripts/classification/extract_picai_case_features.py \
  --manifest /home/degboh/prostate_mri_cancer_detection/data/features/picai_fold0_image_manifest.csv \
  --output /home/degboh/prostate_mri_cancer_detection/data/features/picai_fold0_case_features.csv
```

Expected current output:

- Shape: 300 rows x 118 columns.
- Label counts: 213 non-csPCa and 87 csPCa.
- Feature errors: 0.
- PyRadiomics GLCM symmetry warning may appear and is informational.

Baseline fold0 classifier training command:

```bash
python scripts/classification/train_picai_baseline_classifier.py \
  --features /home/degboh/prostate_mri_cancer_detection/data/features/picai_fold0_case_features.csv \
  --output-dir /home/degboh/prostate_mri_cancer_detection/artifacts/classifier_v1_fold0 \
  --overwrite
```

## Output Locations

Generated files should go under the cluster storage root:

- Feature CSVs: `/home/degboh/prostate_mri_cancer_detection/data/features`
- Model artifacts: `/home/degboh/prostate_mri_cancer_detection/artifacts`
- Metrics and predictions: `/home/degboh/prostate_mri_cancer_detection/outputs`
- Logs: `/home/degboh/prostate_mri_cancer_detection/logs`
- Reports: `/home/degboh/prostate_mri_cancer_detection/reports`

Example future classifier artifact directory:

```text
/home/degboh/prostate_mri_cancer_detection/artifacts/classifier_v1_fold0/
```

## Never Commit

Do not commit:

- PI-CAI raw data.
- `.mha`, `.nii`, `.nii.gz`, or `.dcm` files.
- ZIP files.
- Generated feature CSV files.
- Model artifacts.
- Outputs.
- Logs.
- Reports.
- `.env` files.
- `.joblib`, `.pkl`, `.pt`, or `.pth` files.

Keep generated data and artifacts outside the repository checkout whenever
possible.
