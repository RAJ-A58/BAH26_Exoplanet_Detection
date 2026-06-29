# Known Challenges and Fixes

This document tracks the real blockers in the current repository and the concrete fixes required to move from a synthetic prototype to a credible Kepler detection pipeline.

## Challenge 1: Real-World Detection Currently Fails

The current model does not detect the confirmed planet `Kepler-10b` in the existing real-data test path.

- Current outcome: `17.56%` confidence on a confirmed planet.
- Impact: the project cannot yet claim successful real-data exoplanet detection.
- File involved: `scripts/test_kepler.py`

### Why this matters

High synthetic validation accuracy is not enough if the model fails on a confirmed Kepler target. Real benchmark performance must become the main success criterion.

### Required fix

- Treat `Kepler-10b` as the baseline failure case.
- Build a repeatable benchmark script so later changes can be measured against the same target set.

## Challenge 2: Synthetic Training Distribution Is Too Narrow

The current training data mostly teaches the model to separate large injected transits from simple noise.

### Current limitations

- Positive examples are generated with `rp` between `0.05` and `0.20`.
- Negative examples are mostly noise plus sinusoidal stellar wobble.
- There are no explicit eclipsing binary examples.
- There are no realistic blend or centroid-driven false positives.
- There is no class-aware metadata for later error analysis.

### Why this matters

The model has not been trained on the kinds of shallow planetary signals and structured false positives that appear in real Kepler data.

### Required fix

- Expand the positive distribution to include shallow rocky and super-Earth-like transits.
- Add structured negatives such as eclipsing-binary-like curves and starspot-dominated variability.
- Save metadata for each sample so failures can be traced to specific signal types.

## Challenge 3: BatchNorm Caused Train/Inference Drift

The earlier model used `BatchNormalization` inside the network even though inference was performed on one target at a time with a different distribution than the synthetic batches used in training.

### Why this matters

This creates a scaling mismatch between training and real-world inference. Even a good classifier can become unreliable if the input statistics are shifted at prediction time.

### Fix status

- `BatchNormalization` has been removed from the model code.
- A shared normalization helper now exists for:
  - dataset generation
  - training
  - real-data inference
- Per-sample standardization with epsilon protection is now applied consistently.
- Remaining work: retrain and re-benchmark to quantify the impact of the fix.

## Challenge 4: The Current "Dual-Branch" Model Is Not a True AstroNet-Style Dual View

Both branches currently consume the same 201-bin input and differ only by convolution kernel size.

### Why this matters

This is not the same as a true global-plus-local architecture. The model cannot simultaneously reason over the whole folded curve and a zoomed-in transit window the way AstroNet does.

### Fix status

- The model and preprocessing code now generate two separate views for each example:
  - global folded view, approximately 2001 bins
  - local transit-centered view, approximately 201 bins
- The training and inference paths now use two independent inputs.
- Remaining work: retrain and validate on real targets.

## Challenge 5: No Period Search in the Real Inference Pipeline

The current Kepler test path relies on known orbital parameters instead of discovering them from the light curve.

### Why this matters

A real detection pipeline must search for periodic transit-like structure before classification. Hardcoding the period and epoch is useful for debugging, but not for a deployable workflow.

### Fix status

- BLS-based period search has been added to the real-data inference script.
- A separate known-ephemeris path now exists for debugging and ablation testing.
- Remaining work: validate searched periods across multiple real targets.

## Challenge 6: Evaluation Is Incomplete

The current repository reports synthetic training accuracy and some Keras metrics, but it does not provide a complete evaluation layer.

### Missing items

- confusion matrix
- ROC-AUC
- precision-recall curve / PR-AUC
- labeled real-target benchmark report
- saved comparison artifacts across model versions

### Required fix

- Add dedicated evaluation scripts.
- Save both plots and machine-readable summaries under `results/`.
- Evaluate on synthetic holdout data and real labeled targets.

## Challenge 7: No Multi-Target Real Benchmark Yet

The repository currently centers on a single `Kepler-10b` test.

### Why this matters

A single target is not enough to support claims about robustness or real-world detection performance.

### Required fix

- Build a benchmark set of multiple confirmed planets.
- Add matched false-positive or non-planet examples.
- Report results across the whole benchmark, not just one star.

## Challenge 8: Blend / Centroid Analysis Is Only Conceptual

Centroid offset and pixel-level blending checks are mentioned in planning docs but are not implemented in code.

### Why this matters

This means the current pipeline has no dedicated mechanism for rejecting some important astrophysical false positives in crowded fields.

### Required fix

- Implement centroid/blend analysis as a post-classification validation module.
- Treat it as a later-stage realism upgrade after the classifier itself is working on real targets.

## Recommended Fix Order

1. Remove the preprocessing mismatch and standardize inputs consistently.
2. Rebuild the synthetic dataset with realistic positives and negatives.
3. Replace the fake dual-branch architecture with true global and local views.
4. Add BLS period search.
5. Build proper evaluation scripts and metrics.
6. Demonstrate performance on multiple real confirmed Kepler planets.
7. Add centroid/blend rejection.
