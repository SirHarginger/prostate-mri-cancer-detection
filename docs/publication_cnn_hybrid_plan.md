# Publication-Grade CNN/Hybrid Optimization Plan

This note resets the project framing for publication-oriented experiments.
CNNs are no longer treated as automatically superior. They are candidate
representations that must prove additive value over radiomics.

## Research Objective

The current thesis path is case-level clinically significant prostate cancer
classification with three comparable representations:

- radiomics-only
- CNN-only embeddings
- hybrid radiomics + CNN embeddings

Lesion detection and PI-CAI-style nnU-Net detection are important future work,
but they are not the current main thesis path.

## Evidence Direction

PI-CAI's official baseline ecosystem targets 3D csPCa detection/diagnosis with
U-Net, nnU-Net, and nnDetection. That means the current `TinyMultisequenceCNN`
should be treated as a pipeline validator, not a publication-strength model:

```text
https://github.com/DIAGNijmegen/picai_baseline
```

For case-level hybrid classification, the serious CNN candidates should come
from medical-imaging classification families such as 2.5D/3D residual and
DenseNet-style models. MONAI can be added later for stronger ready-made
backbones if the cluster environment supports it:

```text
https://docs.monai.io/en/latest/networks.html
```

## Candidate Set

Current publication-candidate names:

- `cnn_candidate_25d_resnet`
- `cnn_candidate_3d_densenet`
- `hybrid_radiomics_cnn_candidate`

These are still provisional names. Final method names are reserved until model
selection is stable.

## Current Implementation Direction

The candidate CNN pipeline supports:

- T2W reference grid
- ADC and high b-value DWI resampled to T2W
- whole-gland centered 2.5D slice-window tensors
- whole-gland centered 3D tensors
- per-case/per-sequence normalization
- train-only augmentation
- validation-selected checkpoint and threshold reporting
- ignored tensor caches, feature tables, model checkpoints, and reports

The project should compare candidate CNN embeddings against radiomics on the
same case IDs and split policy before making any claim about CNN benefit.

## Evaluation Requirements

Publication-facing comparisons should include:

- ROC-AUC, sensitivity, specificity, precision, F1, and confusion matrix
- validation-selected fixed-sensitivity threshold analysis
- bootstrap confidence intervals
- paired bootstrap AUC deltas for radiomics vs CNN, radiomics vs hybrid, and
  CNN vs hybrid
- calibration diagnostics such as Brier score and calibration-bin summaries

Fixed-sensitivity outputs are threshold diagnostics only. They do not support
biopsy-reduction claims.

## Claim Limits

Allowed current framing:

- CNN embeddings are evaluated as a complementary representation.
- Hybrid radiomics + CNN is a hypothesis under internal PI-CAI fold evaluation.
- Current candidate models are not final publication architectures until model
  selection is complete.

Disallowed framing:

- CNN is the best model.
- The current Tiny CNN is publication-grade.
- Results are externally validated.
- Results support lesion localization, clinical deployment, radiologist
  replacement, or biopsy reduction.
