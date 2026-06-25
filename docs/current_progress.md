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
After updating the dataset to **8,000 samples** and normalizing the data, our AI successfully trained to **~90% Validation Accuracy** with a **99% Precision** rate on synthetic data!

## 3. Real-World Testing & Current Roadblocks

To validate our 90% accurate model, we ran a zero-shot inference test against real NASA data for **Kepler-10b** (`test_kepler.py`).

**The Result:** The model failed to detect the planet, outputting only 17.56% confidence.

We have diagnosed this failure as a classic "Sim2Real" domain gap. The AI was trained to find massive Jupiter-sized gas giants, but Kepler-10b is a tiny, rocky terrestrial planet. The AI simply hasn't been trained to look for dips that small.

> Please read `docs/known_challenges_and_fixes.md` for a complete breakdown of this problem and exactly how we will reprogram the dataset to fix it.

## 4. Future Improvements (If Accuracy Still Stalls)

*   **Focal Loss:** Swap `binary_crossentropy` for focal loss to focus the model on the rare, hard-to-classify planetary transits.
*   **True Dual-View Inputs:** Currently both branches share the same 201-bin curve and differ only by kernel size. Implement distinct global (e.g. 2001-bin) and local (e.g. 201-bin) views per the original AstroNet design in `implementation_plan.md`.
