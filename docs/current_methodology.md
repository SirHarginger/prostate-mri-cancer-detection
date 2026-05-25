# Current Methodology And Limitations

This repository currently implements a staged prototype workflow for PI-CAI
bpMRI case-level clinically significant prostate cancer classification research.
The implemented stages are designed to validate data handling and pipeline
contracts before full preprocessing, final radiomics extraction, CNN training,
or publication-grade evaluation.

## Implemented Stages

- Stage 1: PI-CAI manifest generation linking case IDs, folds, T2W, ADC, high
  b-value DWI, clinical labels, anatomical masks, and csPCa lesion masks.
- Stage 2: header-only preprocessing validation, including modality/mask
  geometry checks and T2W-compatible mask candidate detection.
- Stage 3: dependency-light first-order T2W radiomics extraction for validated
  gland and lesion ROIs.
- Stage 4: split-safe deterministic T2W prototype embeddings with provenance.
- Stage 5: aligned prototype radiomics-only, prototype-embedding-only, and
  hybrid nearest-centroid baselines.
- Stage 6: prototype evaluation report generation with metrics, confusion
  matrices, false positives, false negatives, and fixed-sensitivity summaries.
- Stage 7: centroid-based prototype feature-importance reporting.

## Current Research Interpretation

Whole-gland T2W radiomics are currently the most complete case-level feature
representation, because they can be extracted for positive and negative cases.
Lesion radiomics are only available where non-empty lesion masks exist and
therefore are not a complete case-level feature table by themselves.

The prototype embedding table is useful for validating split-safe data flow and
fusion plumbing, but it is not a trained CNN representation.

## Current Limitations

- ADC and high b-value DWI are not yet resampled to the T2W grid.
- The radiomics extractor currently computes dependency-light first-order
  features, not full PyRadiomics texture features.
- The embedding pipeline is deterministic and untrained.
- Baseline metrics are prototype sanity checks from a small sampled workflow,
  not final scientific results.
- Validation and test conclusions are limited by the current sample size and
  split composition.
- PI-RADS comparison is not implemented.
- CNN visual explanations are not implemented.
- No external validation has been performed.

## Claim Guardrails

Use supported wording:

- "prototype pipeline"
- "case-level csPCa classification research workflow"
- "internal sample validation"
- "feature-importance inspection"
- "fixed-sensitivity exploratory summary"

Avoid unsupported wording:

- "clinical deployment"
- "radiologist replacement"
- "biopsy reduction"
- "tumor localization"
- "external validation"
- "CNN-only final baseline"
- "hybrid model performance improvement"
