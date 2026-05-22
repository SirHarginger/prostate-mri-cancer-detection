# PI-CAI Dataset Inventory

Stage 1 creates a case-level PI-CAI manifest before any preprocessing,
radiomics, CNN embedding extraction, modeling, or evaluation. The manifest is
an auditable inventory of local raw files and label correspondence only.

PI-CAI label sources are expected under `data/raw/picai/picai_labels`, following
the official label repository structure for clinical information, anatomical
delineations, and csPCa lesion delineations:
<https://github.com/DIAGNijmegen/picai_labels>.

## Command

Run from the repository root on the cluster where `data/raw/picai` is mounted:

```bash
PYTHONPATH=src python -m prostate_mri_cancer_detection.cli manifest \
  --raw-root data/raw/picai \
  --output data/interim/picai_manifest.csv \
  --report data/interim/picai_manifest_validation.json
```

Both outputs are written under `data/interim`, which is ignored by Git.

## Manifest Fields

- `case_id`: `patient_id_study_id`, parsed from image filenames after removing
  exact `_t2w`, `_adc`, or `_hbv` modality suffixes.
- `fold`: fold directory discovered from `data/raw/picai/images/fold*`.
- `path_t2w`, `path_adc`, `path_hbv`: relative paths to available bpMRI
  sequences.
- `available_sequences`: pipe-delimited sequence list.
- `clinical_row_found`: whether `clinical_information/marksheet.csv` contains
  a matching `patient_id` and `study_id`.
- `label_cspca`: value from `case_csPCa` only when that explicit column exists.
- `pirads_score`: value from explicit PI-RADS columns only when present.
- `path_gland_mask`: pipe-delimited anatomical delineation paths, if linked.
- `path_lesion_mask`: pipe-delimited csPCa lesion delineation paths, if linked.
- `has_gland_mask`, `has_lesion_mask`: mask availability flags.
- `missing_data_flags`: missing modalities, masks, clinical labels, duplicate
  modalities, or fold mismatch flags.

## Validation Report

The JSON report summarizes:

- case counts by fold
- T2W, ADC, and high b-value DWI availability
- linked clinical rows
- linked anatomical and lesion mask cases
- duplicate image modality records
- recognized non-manifest image planes such as `_cor` and `_sag`
- orphan clinical rows or masks not linked to discovered image cases
- missing-data flag counts
- skipped image-like files that match neither target bpMRI modalities nor
  recognized non-manifest planes

Coronal and sagittal image files are inventoried in the validation report as
non-manifest image files. They are not included as Stage 1 manifest columns
because the planned bpMRI workflow begins with the axial T2W, ADC, and high
b-value DWI trio.

## Claim Limits

This stage does not establish model performance, preprocessing validity,
lesion localization, clinical deployment readiness, or biopsy-reduction
potential. Those claims require later validated stages and must not be inferred
from the manifest alone.
