# Exoplanet Detection Pipeline Tasks

- `[x]` **Phase 1: Dataset Acquisition & Setup**
  - `[x]` Identify and select light curve datasets (Kepler/TESS).
  - `[x]` Setup Python environment and install astronomical libraries (e.g., `lightkurve`, `astropy`, `batman-package`, `wotan`).
  - `[x]` Download/simulate a sample of confirmed planets and false positives for initial testing.
- `[x]` **Phase 2: Data Preprocessing & Generation**
  - `[x]` Implement Sigma-clipping/Wotan for outlier removal and detrending.
  - `[x]` Build script to generate and phase-fold massive synthetic datasets using Batman.
  - `[x]` Implement Box-fitting Least Squares (BLS) or rely on known periods for phase-folding.
  - `[x]` Implement binning to generate 'Global' and 'Local' views.
- `[x]` **Phase 3: AI Model Implementation**
  - `[x]` Build the Dual-Branch 1D-CNN architecture (Global + Local branches).
  - `[x]` Setup training pipeline (loss function, optimizer).
- `[x]` **Phase 4: Training & Validation**
  - `[x]` Train the model on the prepared dataset.
  - `[x]` Generate evaluation metrics (Recall, Precision, Confusion Matrix).
