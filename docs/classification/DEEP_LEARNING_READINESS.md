# Deep-Learning Readiness for PI-CAI Classification

This report evaluates whether the project is ready for deep-learning
classification. It is intentionally conservative. Deep learning is part of the
long-term plan, but the current leakage-safe radiomics/classical ML baseline
remains the scientific anchor.

This project is research software only. It is not clinically validated and must
not be used as a standalone diagnostic system.

## A. Readiness Verdict

The project is ready for a small fold0 deep-learning prototype and readiness
audit. It is not ready for serious deep-learning training yet.

Current readiness:

- Ready for a pipeline prototype: yes, with PI-CAI fold0 and careful leakage
  controls.
- Ready for serious model training: no.
- Ready for model claims or backend deployment: no.

Why not serious training yet:

- Only fold0 images are downloaded, giving 300 core bpMRI cases.
- Folds 1-4 are not yet downloaded, so the full 1500-case PI-CAI image set is
  unavailable.
- The image preprocessing strategy for T2W/ADC/HBV alignment and prostate crop
  generation has not been validated.
- PyTorch, MONAI, CUDA, and dataloader readiness still need to be audited on
  the cluster.
- A leakage-safe classical baseline exists and should remain the comparison
  baseline for any deep-learning work.

## B. Safe Deep-Learning Task Options

### Option 1 - Whole-Prostate or Whole-Gland Crop Classifier

Recommended safe first direction.

- Input: T2W/ADC/HBV prostate-centered volume or 2.5D axial slices.
- Target: `case_cspca_binary`.
- Mask use: whole-gland mask only for anatomical cropping or localization.
- Safety: acceptable, because whole-gland masks are anatomical support masks and
  are available independently of binary csPCa status.

This should be the first deep-learning prototype if the audit confirms that
image loading, spacing, crop sizes, and GPU dependencies are ready.

### Option 2 - Full-Image Classifier

Possible but riskier.

- Input: full T2W/ADC/HBV images.
- Target: `case_cspca_binary`.
- Risk: the model may learn scanner, site, padding, field-of-view, background,
  or acquisition shortcuts rather than prostate cancer signal.

This can be useful as a quick engineering smoke test, but it is not the
preferred scientific baseline.

### Option 3 - Lesion-Crop Classifier

Not valid for binary csPCa detection.

- Input: lesion crop, lesion mask, lesion-mask channel, lesion volume, or
  lesion-mask-derived region.
- Target: `case_cspca_binary`.
- Problem: in PI-CAI fold0, lesion masks are empty for all non-csPCa cases and
  non-empty for all csPCa cases.

This leaks the binary label and must not be used for v1 binary detection.
Lesion masks can only be used later for a separate lesion-characterization task
among positive or suspicious cases.

### Option 4 - Hybrid Model

Valid later, not first.

- Input: deep image features plus safe clinical variables and/or leakage-safe
  radiomics.
- Target: `case_cspca_binary`.
- Requirement: a validated safe image pipeline must exist first.

This should come after a whole-gland crop image baseline is stable.

## C. Data Needed

Fold0 has 300 cases with T2W, ADC, and HBV:

- 213 non-csPCa.
- 87 csPCa.

Fold0 is enough for:

- Dataloader development.
- Image geometry audits.
- Prostate crop prototype.
- Small overfit/debug experiments.
- End-to-end smoke tests.

Fold0 is not enough for serious deep-learning claims.

For serious experiments:

- Download PI-CAI folds 1-4 on the cluster.
- Rebuild the image manifest for all downloaded folds.
- Confirm expected core bpMRI availability across all 1500 cases.
- Preserve patient/case-level splitting.
- Consider center-aware validation after the all-fold manifest is complete.
- Plan external validation later.

## D. Required Preprocessing

Minimum preprocessing work before training:

- Load `.mha` files with SimpleITK or MONAI.
- Read T2W, ADC, and HBV for each case.
- Handle the known resolution mismatch between T2W and ADC/HBV.
- Either resample ADC/HBV into a common image space or use modality-specific
  branches that respect native spacing.
- Use the whole-gland mask only to crop or localize the prostate region.
- Do not use lesion masks as binary-classification input.
- Do not use zonal masks for v1; current zonal geometry is not cleanly aligned.
- Normalize intensities per volume with a documented method.
- Record crop sizes, spacing, orientation, and failure cases.
- Save generated arrays, crops, manifests, logs, and outputs outside Git.

Known geometry example, case `10000_1000000`:

```text
T2W image:         640 x 640 x 31
ADC image:         116 x 114 x 31
HBV image:         116 x 114 x 31
Lesion mask:       116 x 114 x 31
Whole-gland mask:  640 x 640 x 31
Zonal mask:        640 x 640 x 25
```

Interpretation:

- Whole-gland mask aligns with T2W space.
- Lesion mask aligns with ADC/HBV space but must not be used for binary
  detection.
- Zonal mask is deferred.

## E. Architecture Recommendation

Compare candidate baselines:

- 2D CNN on axial slices: simplest engineering path, but needs a safe case-level
  aggregation strategy.
- 2.5D CNN using neighboring slices: still relatively simple and captures local
  through-plane context.
- 3D CNN on prostate crop: more anatomically natural but more sensitive to crop
  size, spacing, memory, and data scarcity.
- MONAI DenseNet/ResNet/EfficientNet-style medical image baseline: practical
  once crop tensors and dataloaders are reliable.

Recommended first prototype:

Start with a small 2.5D or 2D axial prototype on prostate-centered
whole-gland crops. Use it to validate data loading, cropping, augmentation,
case-level aggregation, and leakage controls. Defer 3D CNNs until crop geometry
and GPU memory are understood.

## F. Engineering Checks Needed

Before implementing training:

- Confirm GPU availability on the cluster.
- Confirm PyTorch installation.
- Confirm MONAI installation, or decide to start with PyTorch-only transforms.
- Confirm SimpleITK can read all needed `.mha` files.
- Audit T2W/ADC/HBV sizes and spacings over fold0.
- Confirm whole-gland mask availability and crop viability.
- Record candidate crop dimensions before generating arrays.
- Confirm class balance and stratified patient/case-level splits.
- Avoid slice-level train/validation splitting.
- Avoid lesion masks, lesion crops, lesion channels, and lesion-derived
  features.
- Exclude target-derived and diagnosis-derived columns.
- Set reproducibility seeds.
- Define model artifact format before training.
- Keep generated data and outputs under the cluster storage root.

## G. Recommended Next Implementation Task

Do not build a full trainer next.

Recommended next implementation:

```text
scripts/classification/audit_deep_learning_readiness.py
```

The audit should read the fold0 image manifest, count core bpMRI availability,
sample image geometry, and report torch/MONAI/CUDA readiness. After that audit
passes on the cluster, the next implementation should be a prototype dataset
class or prostate-crop preprocessing script, not a full training pipeline.
