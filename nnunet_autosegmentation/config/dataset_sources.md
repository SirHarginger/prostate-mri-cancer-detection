# Dataset Sources

This folder tracks external data sources used only for nnU-Net autosegmentation experiments.

## PROSTATE-MRI

- Source DOI in metadata: https://doi.org/10.7937/K9/TCIA.2016.6046GUDv
- Current cluster mirror path: `nnunet_autosegmentation/data/raw/world_wide_covid/PROSTATE_MRI`
- The Kaggle dataset name is misleading; the extracted metadata and DICOM folders are `PROSTATE-MRI`.
- Use only the axial T2 series (`T2 TSE ax hi`) for the first autosegmentation pass.

## Storage Rules

- Raw external data is not committed.
- nnU-Net preprocessed data is not committed.
- nnU-Net predictions are not committed.
- Extracted feature tables are not committed unless explicitly reviewed.
