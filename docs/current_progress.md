# Current Project Progress

This repository currently contains a useful prototype, not a finished exoplanet detection pipeline.

## What Is Working

- Python environment and core dependencies are in place.
- Raw Kepler light curves can be downloaded and cleaned.
- Synthetic transit data can be generated with `batman` and detrended with `wotan`.
- Shared per-sample standardization now exists for dataset generation, training, and inference.
- The model code now uses true two-input global and local folded views instead of one shared 201-bin tensor.
- The real-data inference script now supports BLS-based period search as well as known-ephemeris debugging mode.

## What Is Not Yet Working

- The real-data test currently fails on a confirmed planet.
- The revised dataset and model still need to be retrained end to end and re-evaluated.
- Real benchmark evaluation on multiple labeled Kepler targets is not yet complete.
- Blend and centroid-based false-positive rejection is still missing.

## Current Evidence

- Synthetic training reaches strong validation metrics.
- Real `Kepler-10b` inference currently returns `17.56%` confidence.
- The pipeline code has now been upgraded, but fresh training artifacts and benchmark results still need to be generated.

## Current Diagnosis

The main failure mode is a combination of:

- simulation-to-reality gap in the training data
- need for retraining after the normalization and architecture changes
- need for real benchmark evidence after adding BLS and dual-view inputs

## Immediate Next Steps

1. Regenerate the synthetic dataset with the new multi-class signal mix.
2. Retrain the dual-view model on the regenerated dataset.
3. Run the updated Kepler inference script in both `known` and `searched` period modes.
4. Evaluate on multiple real Kepler targets with proper metrics.
5. Add centroid and blend rejection after classifier performance improves.

## Current Status Summary

The project has crossed the "prototype assembled" stage, but it has not yet crossed the "real exoplanet detection demonstrated" stage.
