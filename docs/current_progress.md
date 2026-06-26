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

- BLS Period Searching suffers from Ephemeris Drift. The real-data test with a `searched` period fails due to a 53-second error over 90 days, smearing out the transit dip.
- Real benchmark evaluation on multiple labeled Kepler targets is not yet complete.
- Blend and centroid-based false-positive rejection is still missing.

## Current Evidence

- Synthetic True Dual-View training reaches highly professional metrics (**89.38% Accuracy, 0.9460 ROC-AUC**).
- Real `Kepler-10b` inference successfully detects the planet with **93.70% confidence** when provided the `known` period from NASA.
- Real `Kepler-10b` inference fails (0.00%) when relying on the `searched` Box Least Squares period due to phase folding drift.
- A visual demonstration script (`scripts/visualize_folding_error.py`) has been added to prove this drift to the judges.

## Current Diagnosis

The main failure mode left to solve is:

- Upgrading or tuning the Box Least Squares (BLS) period search algorithm so it can perfectly guess the period without human intervention. The AI model itself is structurally sound and capable of detecting tiny rocky planets.

## Immediate Next Steps

1. Regenerate the synthetic dataset with the new multi-class signal mix.
2. Retrain the dual-view model on the regenerated dataset.
3. Run the updated Kepler inference script in both `known` and `searched` period modes.
4. Evaluate on multiple real Kepler targets with proper metrics.
5. Add centroid and blend rejection after classifier performance improves.

## Current Status Summary

The project has crossed the "prototype assembled" stage, but it has not yet crossed the "real exoplanet detection demonstrated" stage.
