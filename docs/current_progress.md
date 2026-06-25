# Current Project Progress: Exoplanet Detection Pipeline

*This document summarizes the current state of our hackathon project, what is functioning, and what our immediate next steps are for improving the AI model.*

## 1. What We Have Built (The Pipeline)

We have successfully established an end-to-end data engineering and deep learning pipeline. The repository currently contains:

*   **`requirements.txt` & Environment:** A fully configured Python workspace with astrophysics libraries (`lightkurve`, `astropy`, `wotan`, `batman-package`) and AI libraries (`tensorflow`, `scikit-learn`).
*   **Data Acquisition (`download_sample.py`):** Code that successfully interfaces with NASA's MAST API to download raw `.FITS` light curve files.
*   **Data Processing (`preprocess_lightcurve.py`):** Algorithms to clean raw data, remove outliers (sigma-clipping), and perform **Phase-Folding** to amplify the transit signal.
*   **Synthetic Data Generation (`simulate_transit.py` & `generate_dataset.py`):** To solve the "imbalanced dataset" problem, we built a physics simulator using `batman` to inject fake planets into noisy timelines, and `wotan` to mathematically detrend the stellar noise. We can mass-produce balanced datasets (50% planets, 50% noise) exported directly to `.npy` arrays.
*   **The AI Model (`train_model.py`):** We have coded a **Dual-Branch 1D-CNN** (inspired by NASA's AstroNet). It has a "Local" branch (small kernels) to analyze transit dip shapes, and a "Global" branch (large kernels) to differentiate planets from binary stars.

## 2. Current Status & Results

**The Good News:** The pipeline is 100% functional. We can generate data, process it, and pass it through the complex Dual-Branch CNN without any crashes or mathematical shape errors.

**The Current Results:**
In our initial test run, the model achieved an accuracy of **~47%** with a loss stalled at `0.6931`. 
*Why?* A loss of 0.6931 is the exact mathematical baseline for random guessing in binary crossentropy. The CNN was unable to learn the planetary features and resorted to guessing.

## 3. Immediate Next Steps (The "Data Scientist Fix") — DONE

The failure to learn in the first run was expected due to two missing scaling factors, both of which have now been implemented:

1.  `[x]` **Increase Dataset Size:** Deep learning models require significantly more data to generalize.
    *   *Action taken:* Updated `generate_dataset.py` to produce **8,000 samples** (4,000 planets / 4,000 noise), up from 500.
2.  `[x]` **Implement Data Normalization:** Neural networks struggle to learn when input data is centered around `1.0` (which is the default baseline for relative stellar flux).
    *   *Action taken:* Injected a `BatchNormalization` layer (`Flux_Normalization`) into `train_model.py` immediately after the input, re-centering flux toward a mean of `0.0` before it feeds **both** CNN branches.

With these two fixes applied, we expect the validation accuracy to spike significantly on the next training run.

## 4. Future Improvements (If Accuracy Still Stalls)

*   **Focal Loss:** Swap `binary_crossentropy` for focal loss to focus the model on the rare, hard-to-classify planetary transits.
*   **True Dual-View Inputs:** Currently both branches share the same 201-bin curve and differ only by kernel size. Implement distinct global (e.g. 2001-bin) and local (e.g. 201-bin) views per the original AstroNet design in `implementation_plan.md`.
