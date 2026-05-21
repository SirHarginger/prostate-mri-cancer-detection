# AGENTS.md

Project operating charter for the prostate MRI cancer detection research repo.

These instructions apply to the whole repository unless a more specific
`AGENTS.md` is added in a subdirectory.

## Project Context

This repository is research software for prostate MRI workflows, including:

- DICOM/NIfTI data handling.
- Prostate gland or zone segmentation.
- Possible suspicious lesion detection.
- Possible downstream classification only when labels support it.
- Radiomics and machine-learning experiments.

This is not clinical software and is not clinically validated.

## Current Classifier-First Direction

The project has pivoted from auto-segmentation-first work to a
classifier-first PI-CAI prostate MRI workflow. The current priority is a
leakage-safe, case-level classifier for clinically significant prostate cancer
(csPCa), starting with PI-CAI fold0.

Auto-segmentation work remains useful historical and supporting context, but it
is not the next implementation milestone.

### Current Next Milestone

The next major implementation task is:

```text
scripts/classification/train_picai_baseline_classifier.py
```

That script should train baseline fold0 binary csPCa classifiers from the
existing leakage-safe feature table. Do not start with advanced architectures.

### Local, GitHub, and Cluster Workflow

- Local machine: coding with VS Code and Codex.
- GitHub: sync point between local code changes and the cluster.
- Cluster: heavy data processing, PyRadiomics extraction, training,
  evaluation, and model artifact generation.

Cluster code repository:

```text
/home/degboh/projects/prostate-mri-cancer-detection
```

Cluster storage root:

```text
/home/degboh/prostate_mri_cancer_detection
```

Cluster conda setup:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate prostate-ml
```

Large cluster data and output paths:

```text
/home/degboh/prostate_mri_cancer_detection/data/raw/picai
/home/degboh/prostate_mri_cancer_detection/data/features
/home/degboh/prostate_mri_cancer_detection/outputs
/home/degboh/prostate_mri_cancer_detection/artifacts
/home/degboh/prostate_mri_cancer_detection/logs
/home/degboh/prostate_mri_cancer_detection/reports
```

### Current PI-CAI Status

- Full PI-CAI clinical manifest: 1500 cases, with 1075 non-csPCa and
  425 csPCa cases. The binary target is `case_cspca_binary`.
- PI-CAI fold0 image manifest: 300 cases with core bpMRI
  (`T2W + ADC + HBV`), with 213 non-csPCa and 87 csPCa cases.
- Current fold0 feature table:
  `/home/degboh/prostate_mri_cancer_detection/data/features/picai_fold0_case_features.csv`
- Current feature shape: 300 rows x 118 columns.
- Feature errors: 0.
- Current radiomics source: T2W whole-gland region only.

### Classifier Scientific Guardrails

Follow these hard rules for binary csPCa classification:

- Do not use lesion-mask radiomics for the v1 binary classifier.
- Do not use `case_ISUP`, `case_isup_int`, Gleason, pathology-derived, or
  diagnosis-derived variables as predictors.
- Do not blindly use every feature CSV column as a model input.
- Do not use `case_cspca_binary`, IDs, `feature_error`, lesion columns, or
  other target-derived columns as predictors.
- Use safe predictors only: `patient_age`, `psa`, `psad`,
  `prostate_volume`, encoded `center`, and `t2w_wholegland_*` radiomics.
- Do not use zonal features in v1. The current zonal mask geometry is not
  cleanly aligned.
- Do not make diagnostic, clinical validation, or patient-care claims.

Important leakage finding:

```text
In PI-CAI fold0, lesion masks are empty for all non-csPCa cases and non-empty
for all csPCa cases. Lesion-mask features would therefore leak the target label.
```

Important geometry finding from example case `10000_1000000`:

```text
T2W image:         640 x 640 x 31
ADC image:         116 x 114 x 31
HBV image:         116 x 114 x 31
Lesion mask:       116 x 114 x 31
Whole-gland mask:  640 x 640 x 31
Zonal mask:        640 x 640 x 25
```

Interpretation:

- Lesion masks align with ADC/HBV space.
- Whole-gland masks align with T2W space.
- Zonal masks are deferred for v1.

### Never Commit Generated or Private Artifacts

Do not commit:

- PI-CAI raw data.
- `.mha`, `.nii`, `.nii.gz`, or `.dcm` files.
- ZIP files.
- Generated feature CSV files.
- Model artifacts.
- Outputs, logs, or reports.
- `.env` files.
- `.joblib`, `.pkl`, `.pt`, or `.pth` files.

Generated classifier data, model artifacts, and reports belong under the
cluster storage root, not in Git.

## Engineering Principles

Follow these principles strictly.

### 1. Inspect Before Editing

Before modifying code:

- Inspect the repository tree.
- Read relevant README files.
- Check existing scripts.
- Check existing configs.
- Search for existing functions before creating new ones.
- Understand current naming conventions.

Do not add new files blindly.

### 2. Prefer Small, Safe Changes

Make focused changes that solve the specific task.

Avoid:

- Large rewrites.
- Unrequested framework migrations.
- Creating multiple competing pipelines.
- Adding heavy dependencies unnecessarily.
- Changing public interfaces without updating documentation.
- Touching unrelated files.

If a large change is necessary, break it into smaller steps.

### 3. Keep Raw Data Immutable

Never modify raw downloaded data in place.

Use a staged data layout:

```text
data/
|-- raw/                 # original downloaded files, never edited
|-- interim/             # temporary converted or extracted files
|-- processed/           # cleaned, aligned, model-ready files
|-- manifests/           # CSV/JSON manifests, splits, metadata
`-- external/            # externally supplied lookup tables or metadata
```

Rules:

- `data/raw` must remain read-only by convention.
- Preprocessing writes to `data/interim` or `data/processed`.
- Train/validation/test split files go into `data/manifests`.
- Never overwrite raw DICOM, NIfTI, masks, or labels.

### 4. Make Everything Reproducible

Prefer command-line workflows.

Every major script should be runnable with a clear command.

Examples:

```bash
python scripts/build_manifest.py --dataset nci_isbi --input-dir data/raw/nci_isbi --output data/manifests/nci_isbi_manifest.csv
python scripts/preprocess_dataset.py --config configs/preprocess/nci_isbi.yaml
python scripts/train_segmentation.py --config configs/train/unet_nci_isbi.yaml
python scripts/evaluate_segmentation.py --config configs/eval/unet_nci_isbi.yaml
```

Every script should:

- Accept input paths through arguments or config.
- Write outputs to explicit locations.
- Avoid hardcoded local paths.
- Log what it is doing.
- Fail clearly on missing files.
- Avoid silent skipping unless explicitly logged.

### 5. Separate Research Tasks Clearly

Keep segmentation, detection, and classification workflows distinct.

Use clear naming:

```text
segmentation/
detection/
classification/
preprocessing/
evaluation/
```

Do not mix:

- Gland segmentation labels.
- Lesion labels.
- Cancer diagnosis labels.
- PI-RADS labels.
- Histopathology labels.

Each dataset loader should clearly define:

- Input image.
- Target label.
- Task type.
- Expected file format.
- Output tensor shape.
- Metadata returned.

## Recommended Repository Structure

Prefer a structure similar to this unless the existing repo already has a better
one.

```text
prostate-cancer-detection/
|-- README.md
|-- AGENTS.md
|-- requirements.txt
|-- pyproject.toml
|-- .gitignore
|-- configs/
|   |-- data/
|   |-- preprocess/
|   |-- train/
|   `-- eval/
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|   |-- manifests/
|   `-- external/
|-- docs/
|   |-- dataset_strategy.md
|   |-- methodology.md
|   |-- preprocessing.md
|   |-- experiments.md
|   `-- runbook.md
|-- notebooks/
|   `-- exploratory/
|-- outputs/
|   |-- figures/
|   |-- logs/
|   |-- metrics/
|   |-- predictions/
|   `-- reports/
|-- scripts/
|   |-- build_manifest.py
|   |-- verify_dataset.py
|   |-- preprocess_dataset.py
|   |-- train_segmentation.py
|   |-- evaluate_segmentation.py
|   `-- visualize_qc.py
|-- src/
|   `-- prostate_detection/
|       |-- __init__.py
|       |-- data/
|       |-- preprocessing/
|       |-- models/
|       |-- training/
|       |-- evaluation/
|       |-- visualization/
|       `-- utils/
`-- tests/
    |-- test_manifest.py
    |-- test_preprocessing.py
    |-- test_dataset_loading.py
    `-- test_metrics.py
```

Do not force this structure if the user already has a working structure. Adapt
to the existing repo.

## Data Engineering Rules

### Dataset Manifest

Every dataset should have a manifest.

A manifest may include:

- `case_id`
- `patient_id`
- `dataset_name`
- `image_path`
- `mask_path`
- `label_path`
- `modality`
- `sequence`
- `task_type`
- `split`
- `spacing`
- `orientation`
- `source`
- `notes`

For prostate MRI, also consider:

- `t2w_path`
- `adc_path`
- `dwi_path`
- `high_b_dwi_path`
- `lesion_mask_path`
- `gland_mask_path`
- `zone_mask_path`
- `pirads_score`
- `gleason_score`
- `cs_pca_label`

Only include fields supported by the dataset.

Do not invent labels.

If a value is unknown, use `unknown` or leave it empty consistently.

### Dataset Verification

Before training, implement verification checks.

Check:

- Required files exist.
- Image and mask paths are valid.
- Image and mask shapes are compatible.
- Image and mask spacing are compatible where possible.
- Case IDs are unique.
- Splits are patient-level or case-level.
- No patient appears in multiple splits.
- Mask values are expected.
- Modalities are present as expected.
- Empty masks are flagged.

Recommended command:

```bash
python scripts/verify_dataset.py --manifest data/manifests/nci_isbi_manifest.csv
```

The verification output should include:

- Number of cases.
- Missing files.
- Duplicate cases.
- Split counts.
- Mask value summary.
- Shape mismatch warnings.
- Spacing mismatch warnings.
- Leakage warnings.

### DICOM and NIfTI Handling

Use appropriate libraries.

Possible tools:

- `pydicom`
- `SimpleITK`
- `nibabel`
- `dicom2nifti`
- MONAI transforms

General rules:

- Preserve metadata where practical.
- Do not assume orientation without checking.
- Do not assume masks align after conversion.
- Log original spacing, shape, and orientation.
- Save converted outputs under `data/interim` or `data/processed`.
- Keep conversion reproducible through scripts/configs.

When converting DICOM:

- Group slices by series.
- Preserve series identifiers.
- Avoid mixing sequences.
- Record conversion metadata.
- Validate output volume dimensions.

### Preprocessing

Preprocessing should be explicit and configurable.

Common preprocessing steps may include:

- Orientation normalization.
- Resampling to target spacing.
- Intensity normalization.
- Cropping around prostate region.
- Padding or resizing.
- Mask value normalization.
- Multi-sequence alignment checks.
- Quality-control visualization.

Do not apply preprocessing globally without documenting it.

Do not compute normalization statistics using test data.

Preprocessing configs should specify:

```yaml
dataset: nci_isbi
task: gland_segmentation
input_manifest: data/manifests/nci_isbi_manifest.csv
output_dir: data/processed/nci_isbi/
target_spacing: [0.5, 0.5, 3.0]
normalize: zscore_nonzero
crop: prostate_bbox
save_qc: true
```

Adapt values to evidence and dataset properties.

### Split Strategy

Use patient-level or case-level splits.

Avoid random slice-level splitting for medical imaging unless explicitly marked
as exploratory.

Recommended split files:

```text
data/manifests/splits/
|-- nci_isbi_train.csv
|-- nci_isbi_val.csv
`-- nci_isbi_test.csv
```

Split rules:

- Same patient must not appear across splits.
- Use deterministic random seed.
- Save split files.
- Do not regenerate splits silently.
- Document split strategy.

If official splits exist, prefer official splits and document them.

## Modeling Rules

### Start With Baselines

Start with simple, defensible baselines.

For segmentation:

- 2D U-Net baseline.
- 3D U-Net baseline where feasible.
- MONAI U-Net.
- nnU-Net-style baseline if appropriate.

For detection/classification:

- Only implement when labels support the task.
- Start with simple baselines before complex architectures.
- Use patient-level evaluation.

Do not jump directly to advanced models unless the user explicitly asks and the
data supports it.

Advanced models may include:

- SwinUNETR.
- UNETR.
- MedSAM-style workflows.
- Radiomics plus ML.
- Multimodal transformer models.

These require justification.

### Model Code Organization

Prefer structure such as:

```text
src/prostate_detection/models/
|-- unet.py
|-- losses.py
|-- metrics.py
`-- registry.py
```

Training structure:

```text
src/prostate_detection/training/
|-- train_loop.py
|-- losses.py
|-- callbacks.py
`-- logging.py
```

Evaluation structure:

```text
src/prostate_detection/evaluation/
|-- segmentation_metrics.py
|-- detection_metrics.py
|-- classification_metrics.py
`-- reports.py
```

Do not duplicate losses or metrics across scripts.

Reusable logic belongs under `src/`.

Scripts under `scripts/` should be thin CLI wrappers.

## Evaluation Rules

Choose metrics based on task.

### Segmentation Metrics

For gland, zone, or lesion segmentation:

- Dice similarity coefficient.
- Jaccard / IoU.
- Hausdorff distance.
- Average surface distance.
- Volume difference.
- Sensitivity.
- Precision.

### Detection Metrics

For lesion detection:

- Lesion-wise recall.
- False positives per patient.
- Free-response ROC.
- Sensitivity at fixed false-positive rates.
- Precision-recall curve.

### Classification Metrics

For cancer classification:

- AUROC.
- AUPRC.
- Sensitivity.
- Specificity.
- Balanced accuracy.
- F1 score.
- Calibration.
- Confusion matrix.

Rules:

- Do not use segmentation metrics to claim diagnostic accuracy.
- Report patient-level metrics for detection/classification when possible.
- Save metrics to `outputs/metrics`.
- Save predictions to `outputs/predictions`.
- Save figures to `outputs/figures`.

## Quality Control

Add QC outputs wherever useful.

Possible QC outputs:

- Image/mask overlay PNGs.
- Slice previews.
- Spacing and shape summaries.
- Mask value histograms.
- Missing-file reports.
- Dataset split summaries.
- Failed conversion logs.

Recommended command:

```bash
python scripts/visualize_qc.py --manifest data/manifests/nci_isbi_manifest.csv --output-dir outputs/figures/qc/nci_isbi
```

QC should help catch:

- Misaligned masks.
- Wrong orientation.
- Empty masks.
- Wrong mask labels.
- Broken paths.
- Unexpected shapes.
- Incorrect sequence pairing.

## Testing Rules

Add tests for reusable logic.

Test:

- Manifest parsing.
- Path validation.
- Split leakage checks.
- Mask value validation.
- Metric calculations.
- Dataset loader output shapes.
- Config loading.

Use small synthetic fixtures when real medical data cannot be included.

Recommended test structure:

```text
tests/
|-- fixtures/
|   |-- tiny_image.nii.gz
|   |-- tiny_mask.nii.gz
|   `-- tiny_manifest.csv
|-- test_manifest.py
|-- test_dataset_loading.py
|-- test_preprocessing.py
`-- test_metrics.py
```

Validation commands may include:

```bash
python -m pytest
python scripts/verify_dataset.py --manifest data/manifests/example_manifest.csv
python scripts/train_segmentation.py --config configs/train/example_unet.yaml --dry-run
```

If tests cannot be run, explain why.

## Documentation Rules

Every meaningful implementation should update documentation.

Relevant docs:

- `README.md`
- `docs/runbook.md`
- `docs/dataset_strategy.md`
- `docs/preprocessing.md`
- `docs/experiments.md`

Documentation should include:

- What the script does.
- Required inputs.
- Expected outputs.
- Example command.
- Folder locations.
- Assumptions.
- Known limitations.

For dataset documentation, include:

- Dataset role.
- Supported task.
- Label type.
- File format.
- Access method.
- Citation requirement.
- Known limitations.

Do not include unsupported clinical claims.

## CLI Design

Scripts should expose clear CLI arguments.

Example:

```python
def parse_args():
    parser = argparse.ArgumentParser(description="Build dataset manifest for prostate MRI data.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()
```

CLI rules:

- Use descriptive argument names.
- Validate required paths.
- Avoid hardcoded user-specific directories.
- Print or log summary output.
- Exit with clear error messages.
- Support dry-run mode for risky operations where useful.

## Config Rules

Prefer YAML or JSON configs for experiments.

Example:

```yaml
experiment_name: unet_nci_isbi_baseline
task: gland_segmentation
seed: 42

data:
  manifest: data/manifests/nci_isbi_manifest.csv
  image_key: image_path
  mask_key: mask_path

model:
  name: monai_unet
  spatial_dims: 3
  in_channels: 1
  out_channels: 2

training:
  epochs: 100
  batch_size: 2
  learning_rate: 0.0001

evaluation:
  metrics:
    - dice
    - hausdorff95

outputs:
  run_dir: outputs/runs/unet_nci_isbi_baseline
```

Rules:

- Keep configs readable.
- Avoid burying important settings in code.
- Save a copy of config with each run output.
- Use deterministic seeds where possible.

## Logging and Outputs

Every major run should produce a clear output folder.

Example:

```text
outputs/runs/unet_nci_isbi_baseline/
|-- config.yaml
|-- train.log
|-- metrics.json
|-- checkpoints/
|-- predictions/
`-- figures/
```

Log:

- Dataset used.
- Number of cases.
- Split counts.
- Model name.
- Loss function.
- Metrics.
- Random seed.
- Output directory.
- Warnings.

## Refactoring Rules

When refactoring:

- Preserve behavior unless the task asks to change it.
- Remove duplication.
- Keep public function names stable if already used.
- Update imports.
- Update tests.
- Update documentation.
- Run validation commands.

Do not refactor for style alone if it risks breaking functionality.

## Safety and Medical Claim Rules

This project is research software, not a clinical diagnostic device.

Do not write code or documentation that implies:

- The model diagnoses patients.
- The model is clinically validated.
- The system should be used for real clinical decisions.
- Segmentation accuracy equals cancer detection ability.

Use safer wording:

- Research prototype.
- Segmentation baseline.
- Suspicious lesion detection.
- Downstream classification.
- Experimental model.
- Not clinically validated.

## Output Format

When completing an implementation task, respond with:

```markdown
## What changed

## Files modified

## How to run

## Validation performed

## Notes / limitations
```

If no code was changed, respond with:

```markdown
## Recommendation

## Implementation plan

## Files to create or modify

## Validation commands

## Risks
```

## Codex Task Execution Pattern

When given a task, follow this internal execution pattern:

1. Inspect repo.
2. Identify existing conventions.
3. Plan minimal changes.
4. Implement.
5. Run formatting/lint/tests where available.
6. Update documentation if behavior changed.
7. Summarize clearly.

## Useful Validation Commands

Depending on the stack, use commands such as:

```bash
python -m pytest
python -m compileall src scripts
python scripts/verify_dataset.py --manifest data/manifests/example_manifest.csv
python scripts/train_segmentation.py --config configs/train/example.yaml --dry-run
python scripts/evaluate_segmentation.py --config configs/eval/example.yaml
ruff check .
black --check .
mypy src
```

Only run commands that are available in the repository.

If dependencies are missing, report the missing dependency and the command that
failed.

## Interaction With Other Skills

Recommended workflow:

```text
research-lead -> evidence-reviewer -> medical-ml-engineer -> citation-editor
```

### With `research-lead`

Use `research-lead` before major strategy or architecture decisions.

Example:

```text
Should we build segmentation first or detection first?
```

### With `evidence-reviewer`

Use `evidence-reviewer` before implementing anything dependent on dataset facts,
medical claims, model claims, or preprocessing standards.

Example:

```text
Does this dataset support lesion detection or only gland segmentation?
```

### With `medical-ml-engineer`

Use `medical-ml-engineer` for reproducible implementation of medical ML
pipelines after the strategy and evidence have been checked.

### With `citation-editor`

Use `citation-editor` after adding documentation that contains dataset, model,
preprocessing, evaluation, or clinical claims.

Example:

```text
Add citations to docs/dataset_strategy.md.
```

Expected skill set:

```text
skills/
|-- research-lead/
|   `-- SKILL.md
|-- evidence-reviewer/
|   `-- SKILL.md
|-- citation-editor/
|   `-- SKILL.md
`-- medical-ml-engineer/
    `-- SKILL.md
```

Role summary:

```text
research-lead       -> decides direction
evidence-reviewer   -> checks source/dataset/model evidence
medical-ml-engineer -> implements reproducible code
citation-editor     -> cites final documentation
```

## Do Not Do

Do not:

- Modify raw data.
- Hardcode local machine paths.
- Split medical images slice-by-slice across train and test.
- Treat segmentation labels as cancer labels.
- Mix dataset-specific assumptions into generic code.
- Duplicate existing functions.
- Add heavy dependencies without reason.
- Build advanced models before baselines.
- Claim clinical validation.
- Hide failed files during preprocessing.
- Silently overwrite outputs.
- Ignore tests or validation.
- Create undocumented scripts.
- Change unrelated files.
