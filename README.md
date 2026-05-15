# Prostate MRI Cancer Detection Research

Research repository for prostate MRI data engineering, segmentation baselines,
radiomics, and downstream experimental modeling.

This project is a research prototype. It is not clinically validated and must
not be used for real clinical decisions.

## Current Focus

The first milestone is a reproducible prostate MRI data foundation:

1. Preserve raw downloaded data under `data/raw`.
2. Build dataset manifests under `data/manifests`.
3. Convert and preprocess data into `data/interim` and `data/processed`.
4. Keep gland segmentation, lesion detection, and classification workflows
   clearly separated.
5. Add QC, tests, and documentation before model training.

## Repository Layout

This scaffold follows [AGENTS.md](AGENTS.md):

```text
configs/      Experiment and preprocessing configs
data/         Raw, interim, processed, manifest, and external data folders
docs/         Dataset, methodology, preprocessing, experiment, and runbook docs
notebooks/    Exploratory notebooks only
outputs/      Logs, metrics, predictions, figures, and reports
scripts/      Thin command-line entrypoints
src/          Reusable Python package code
tests/        Unit tests and synthetic fixtures
```

## Data Policy

- Do not modify `data/raw` in place.
- Do not commit raw DICOM, NIfTI, masks, or large derived outputs.
- Do not invent labels. Unknown labels must remain unknown.
- Patient-level or case-level splits are required for medical imaging tasks.

## Development Notes

The CLI files in `scripts/` are initial stubs. They define the intended command
surface and fail clearly until each workflow is implemented.

