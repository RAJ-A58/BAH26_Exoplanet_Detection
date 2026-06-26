# Exoplanet Detection Recovery Plan

This checklist replaces the earlier "all core phases complete" view. The current repository has a working synthetic prototype, but it does not yet deliver reliable real-world exoplanet detection on Kepler data.

## Phase 0: Stabilize the Baseline

- `[x]` Confirm the current real-data failure case on `Kepler-10b`.
- `[x]` Identify the main architectural gaps in the existing prototype.
- `[x]` Add a single evaluation entrypoint that records:
  - target name
  - known label
  - prediction score
  - predicted class
  - period source (`known` vs `searched`)
- `[x]` Save baseline outputs under `results/benchmarks/` so every later change can be compared against the same reference.

## Phase 1: Preprocessing Consistency

- `[x]` Implement sigma-clipping / flattening utilities for light-curve cleanup.
- `[x]` Remove `BatchNormalization` from the inference path in `scripts/train_model.py`.
- `[x]` Add one shared normalization helper used by:
  - synthetic dataset generation
  - model training
  - real Kepler inference
- `[x]` Use explicit per-sample standardization with epsilon protection to avoid train/inference scaling drift.
- `[x]` Verify that the same transformation is applied to both synthetic and real folded views.

## Phase 2: Realistic Training Data

- `[x]` Generate synthetic transit datasets with `batman` and `wotan`.
- `[x]` Expand the planet distribution beyond Jupiter-scale transits.
- `[x]` Include shallow rocky/super-Earth-like transits in the positive class.
- `[x]` Add structured negative examples:
  - eclipsing binary-like dips
  - stellar variability / starspot-dominated curves
  - non-transit noise curves
- `[x]` Store labels and metadata for each generated sample so evaluation can be sliced by signal type.
- `[ ]` Create holdout splits that are not reused during training.

## Phase 3: True Dual-View Model

- `[x]` Build a prototype dual-branch 1D CNN.
- `[x]` Replace the current shared 201-bin input with true dual-view inputs:
  - global view: full folded curve, approximately 2001 bins
  - local view: zoomed transit window, approximately 201 bins
- `[x]` Update dataset generation to emit both views for every sample.
- `[x]` Update training code to accept two `Input` tensors and merge learned features after separate branch encoding.
- `[x]` Retrain the model on the revised dataset after Phase 1 and Phase 2 are complete.

## Phase 4: Period Search

- `[x]` Implement Box Least Squares (BLS) search for real light-curve inference.
- `[x]` Produce candidate period and transit epoch from the searched light curve instead of hardcoding known values.
- `[x]` Keep a separate "known ephemeris" mode only for debugging and ablation comparisons.
- `[x]` Validate that the searched period produces sensible folded global and local views before model scoring.

## Phase 5: Evaluation and Metrics

- `[ ]` Add a dedicated evaluation script for synthetic holdout data.
- `[ ]` Add a dedicated evaluation script for labeled real Kepler targets.
- `[ ]` Report:
  - accuracy
  - precision
  - recall
  - F1
  - confusion matrix
  - ROC-AUC
  - PR-AUC
- `[ ]` Save plots and machine-readable summaries under `results/`.
- `[ ]` Compare results before and after each major model or data change.

## Phase 6: Real-World Benchmark Demo

- `[ ]` Build a small benchmark set of multiple confirmed Kepler planets.
- `[ ]` Include planets with a mix of:
  - shallow rocky transits
  - deeper giant-planet transits
  - different periods and signal-to-noise levels
- `[ ]` Add a matched set of false positives or non-planet targets.
- `[ ]` Run the full inference pipeline across the benchmark and summarize outcomes in one report.
- `[ ]` Do not claim successful real-data detection until this benchmark is passing consistently.

## Phase 7: Blend / Centroid Rejection

- `[ ]` Implement centroid-offset or blend-risk analysis as a post-classification validation step.
- `[ ]` Keep this module separate from the transit classifier so it can act as a veto or confidence modifier.
- `[ ]` Document what data products are required for this step and whether they are available in the chosen Kepler workflow.

## Immediate Order of Execution

1. Fix preprocessing consistency and remove the BatchNorm mismatch.
2. Upgrade the synthetic dataset to include realistic positive and negative cases.
3. Replace the fake dual-branch setup with true global/local inputs.
4. Add BLS-based period search.
5. Build real evaluation scripts and metrics.
6. Demonstrate performance on multiple confirmed Kepler targets.
7. Add centroid/blending rejection as a final realism layer.
