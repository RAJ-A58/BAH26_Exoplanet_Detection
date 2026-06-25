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

## 3. Immediate Next Steps (The "Data Scientist Fix")

The failure to learn in the first run was expected due to two missing scaling factors that we will implement next:

1.  **Increase Dataset Size:** We only generated 500 samples for the test run. Deep learning models require significantly more data to generalize.
    *   *Action:* Update `generate_dataset.py` to produce **5,000 to 10,000 samples**.
2.  **Implement Data Normalization:** Neural networks struggle to learn when input data is centered around `1.0` (which is the default baseline for relative stellar flux). 
    *   *Action:* Inject a `StandardScaler` or `BatchNormalization` layer into `train_model.py` to center all light curve data around a mean of `0.0` before it enters the CNN.

Once these two fixes are applied, we expect the validation accuracy to spike significantly.
