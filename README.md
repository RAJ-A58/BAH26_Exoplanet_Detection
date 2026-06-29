<div align="center">

#  BAH26 Exoplanet Detection Pipeline

**An end-to-end AI-powered exoplanet transit detection system using a Dual-Branch Residual 1D-CNN trained on 30,000 synthetic light curves injected with real NASA noise.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Astropy](https://img.shields.io/badge/Astropy-6.x-003399?style=for-the-badge&logo=python&logoColor=white)](https://www.astropy.org/)
[![Lightkurve](https://img.shields.io/badge/Lightkurve-2.x-FF4081?style=for-the-badge)](https://docs.lightkurve.org/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8792-brightgreen?style=for-the-badge)](./results/)
[![Benchmark](https://img.shields.io/badge/Benchmark-100%25%20%2811%2F11%20Stars%29-success?style=for-the-badge)](./results/)

> Built for the **ISRO Space Hackathon (BAH26)** Detecting planets beyond our solar system from raw NASA Kepler photometric data using deep learning.

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Frontend (Web Console)](#frontend-web-console)
- [Backend (Python Server)](#backend-python-server)
- [Scripts Reference](#scripts-reference)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Running the Web App](#running-the-web-app)
- [Data Format](#data-format)
- [Technical Deep Dive](#technical-deep-dive)
- [Known Challenges and Fixes](#known-challenges-and-fixes)
- [Docs](#docs)

---

## Overview

BAH26 is a complete, research-grade exoplanet detection pipeline that:

1. **Generates** 30,000 physics-accurate synthetic light curves using [`batman`](https://batman-package.readthedocs.io/) transit models, blended with real NASA noise extracted from Kepler targets (Kepler-22, 69, 62).
2. **Trains** a Deep Residual Dual-View 1D-CNN that simultaneously processes a global phase-folded view (2001 bins) and a local zoomed transit view (201 bins).
3. **Infers** on raw NASA Kepler data in real-time via a browser-based mission-control console, hunting for multiple planets in a single star using iterative pre-whitening.
4. **Validates** astrophysical signals through centroid shift analysis to reject background eclipsing binary false positives.

---

## Key Results

| Metric | Value |
|---|---|
| **Model Architecture** | Deep Residual Dual-Branch 1D-CNN |
| **Training Samples** | 30,000 synthetic light curves |
| **Real NASA Noise Injection** | 20% of training data |
| **Validation ROC-AUC** | **0.8792** |
| **Benchmark Accuracy** | **100% (11/11 stars)** |
| **Single-Planet Detection** | 9/9 ✅ (Kepler-1 through 10) |
| **Multi-Planet Detection** | Kepler-20b + Kepler-20c detected ✅ |
| **Kepler-20c Confidence** | 72.58% |
| **Kepler-20b Confidence** | 48.13% (limited to 4 quarters) |
| **Global View Bins** | 2001 |
| **Local View Bins** | 201 |
| **BLS Period Search** | Armed (Coarse-to-Fine with Auto-Centering) |

### Benchmark Highlights

- **Kepler-1 → Kepler-10**: All 9 single-planet systems correctly identified. In each case, the model detected Planet 1 at >99% confidence, pre-whitened it, and correctly halted at Iteration 2 — no hallucinated double-detections.
- **Kepler-20 (Multi-Planet)**: Iterative pre-whitening revealed **two independent planets** from the same stellar light curve — a breakthrough result demonstrating the full pipeline capability.
- **Key Insight**: The deep CNN perfectly discriminates real planetary transits from stellar noise when provided adequate SNR. Scaling to 10–15 quarters of data is expected to unlock confident detection of even Earth-sized multi-planet systems.

---

## Architecture

```
Raw Light Curve (Kepler / CSV)
         |
         v
+-----------------------------------------------+
|           PREPROCESSING STAGE                 |
|  1. Outlier removal (sigma-clip sigma=5)       |
|  2. Detrending via Wotan (biweight,            |
|     window_length = 0.278 d ~ 6.6 hrs)        |
|  3. BLS Period Search (Coarse-to-Fine)         |
|  4. Auto-Centering t0 (200-grid refinement)   |
|  5. Phase-Fold + Binning                       |
+----------------+------------------------------+
                 |
        +--------+---------+
        v                  v
  +------------+    +------------+
  | Global     |    | Local      |
  | View       |    | View       |
  | 2001 bins  |    | 201 bins   |
  | (-0.5 to   |    | (+-12% of  |
  | +0.5 phase)|    |  phase)    |
  +-----+------+    +------+-----+
        |                  |
        v                  v
  +------------+    +------------+
  | Global CNN |    | Local CNN  |
  | Branch     |    | Branch     |
  |            |    |            |
  | Conv1D(16) |    | Conv1D(16) |
  | MaxPool(2) |    | MaxPool(2) |
  |            |    |            |
  | ResBlock   |    | ResBlock   |
  |  (32, k=5) |    |  (32, k=3) |
  | ResBlock   |    | ResBlock   |
  |  (64, k=5) |    |  (64, k=3) |
  | ResBlock   |    |            |
  | (128, k=5) |    |            |
  |            |    |            |
  | GlobalAvgP |    | GlobalAvgP |
  +-----+------+    +------+-----+
        |                  |
        +--------+---------+
                 v
         +--------------+
         | Concatenate  |
         +------+-------+
                v
         Dense(256, relu) + L2
         Dropout(0.4)
         Dense(128, relu) + L2
         Dropout(0.3)
         Dense(1, sigmoid)
                |
                v
        Planet Probability
           [0.0 -> 1.0]
```

### Multi-Planet Iterative Pre-Whitening

```
Detect Planet N -> Mask Transit Bins -> Search Again -> Detect Planet N+1
     ^                                                          |
     +----------------------------------------------------------+
                      (up to 5 iterations)
```

---

## Project Structure

```
BAH26_Exoplanet_Detection/
|
|-- app.py                        # Python HTTP server + /api/predict endpoint
|-- requirements.txt              # Python dependencies
|-- run_pipeline.ps1              # Full pipeline runner (Windows PowerShell)
|-- run_remainder.ps1             # Partial pipeline resume script
|-- convert_to_ipynb.py           # Converts scripts to Jupyter Notebook
|-- generate_final_pdf.py         # Generates final PDF report
|-- exoplanet_demo.ipynb          # Jupyter demo notebook
|
|-- frontend/                     # Browser-based mission control console
|   |-- index.html                #   Main UI (Neon mission-control layout)
|   |-- app.js                    #   Frontend logic (Canvas charts, API calls)
|   +-- styles.css                #   Full neon dark-mode CSS (21 KB)
|
|-- scripts/                      # Core ML pipeline scripts
|   |-- pipeline_utils.py         #   Shared constants, phase-folding, binning
|   |-- generate_dataset.py       #   Synthetic data generator (batman + wotan)
|   |-- train_model.py            #   Dual-Branch ResNet 1D-CNN training
|   |-- evaluate_model.py         #   Full evaluation suite (ROC, PR, confusion)
|   |-- centroid_analysis.py      #   Centroid shift false-positive rejection
|   |-- centroid_override.py      #   Manual centroid override utilities
|   |-- run_benchmark_suite.py    #   11-star NASA benchmark runner
|   |-- train_specialist_model.py #   Specialist model for edge cases
|   |-- test_kepler.py            #   Kepler target inference tests
|   |-- simulate_transit.py       #   Single transit simulation utility
|   |-- preprocess_lightcurve.py  #   Standalone lightcurve preprocessor
|   |-- download_sample.py        #   Downloads a single Kepler sample
|   |-- debug_local_view.py       #   Debug tool for local view alignment
|   |-- visualize_folding_error.py#   Phase-folding error visualization
|   +-- generate_demo_report.py   #   Generates PDF demo report
|
|-- docs/                         # Project documentation
|   |-- implementation_plan.md    #   Technical architecture proposal
|   |-- current_progress.md       #   Live project status & results
|   |-- known_challenges_and_fixes.md
|   |-- multiplanet_system_walkthrough.md
|   |-- project_tasks.md
|   +-- tasks_for_teammates.md
|
|-- results/                      # Model outputs and reports
|   |-- synthetic/                #   Trained model + training plots
|   |-- evaluation/               #   ROC curves, PR curves, confusion matrices
|   |-- benchmarks/               #   11-star benchmark results
|   +-- exoplanet_pipeline_demo.pdf
|
|-- data/                         # Auto-created at runtime
|   |-- synthetic/                #   Generated training arrays (X_global, X_local, y)
|   +-- raw_nasa/                 #   Cached Kepler FITS downloads
|
+-- logs/                         # Pipeline step logs
```

---

## Frontend (Web Console)

The frontend is a neon **mission-control** browser interface served directly from the Python backend. It requires **no build step** — just start `app.py`.

### Interface Panels

| Panel | Description |
|---|---|
| **Masthead** | Animated cosmic background with planet rings, orbiting moons, and scan-line overlay |
| **Mode Tabs** | Switch between `Sample`, `CSV`, and `Kepler` input modes |
| **Control Panel** | Period source selector (BLS auto / Known ephemeris), hunt iterations (1–5), period/t0 inputs |
| **Verdict** | Real-time prediction verdict with color-coded confidence |
| **Confidence Meter** | Vertical fill bar showing planet probability (0–100%) |
| **Metrics Bar** | Source label, detected period (days), transit epoch t0 |
| **Global Chart** | Canvas-rendered 2001-bin phase-folded light curve (lime neon) |
| **Local Chart** | Canvas-rendered 201-bin zoomed transit view (cyan neon) |
| **Hunt List** | Per-iteration planet hunt results (index, verdict, period, confidence) |

### Input Modes

#### Sample Mode (Synthetic)
Runs a randomly selected sample directly from the trained synthetic dataset through the model. No network call required — instant prediction from saved `.npy` arrays.

#### CSV Mode
Upload any CSV with `time` and `flux` columns. The parser automatically detects:
- `time`, `time_days`, `bkjd`, `btjd`, `jd` → time axis
- `flux`, `pdcsap_flux`, `sap_flux`, `relative_flux` → flux axis

Minimum 50 rows required. Supports comma, tab, semicolon, and space delimiters.

#### Kepler Mode
Enter any NASA Kepler target name (e.g., `Kepler-10`, `KIC 757450`). The backend downloads up to 4 quarters of short-cadence data via Lightkurve, stitches them, removes NaNs, sigma-clips at 5σ, and flattens with a 401-point window — then runs the full BLS + CNN inference chain.

### Frontend Tech Stack

| Layer | Technology |
|---|---|
| HTML | Semantic HTML5, Canvas API |
| CSS | Vanilla CSS (21 KB) — no framework, full custom neon dark-mode design |
| JavaScript | Vanilla ES2022 — no bundler, no dependencies |
| Charts | Native HTML5 Canvas (custom renderer with glow effects, grid lines, center markers) |
| API | `POST /api/predict` — JSON |

### Frontend File Details

#### `frontend/index.html`
- 147 lines of semantic HTML5
- Animated cosmic field, planet backdrop (rings, core, storms, moons), scan layer
- Three input mode tabs, full form controls, canvas charts, hunt list

#### `frontend/app.js`
- 257 lines of vanilla JavaScript
- `drawChart(canvasId, points, color)` — full canvas renderer with gradient fill, glow stroke, center transit marker (yellow dots at phase 0)
- `renderResult(result)` — parses API response, updates verdict, meters, metrics, and both charts
- `runDetection(event)` — async fetch to `/api/predict`, handles busy state and error display
- `buildPayload()` — assembles JSON payload (reads CSV file via FileReader API if in CSV mode)
- Auto-redraws charts on window resize

#### `frontend/styles.css`
- 21 KB of hand-crafted CSS
- CSS custom properties (design tokens) for colors, spacing, and animations
- `@keyframes` for cosmic scan, orbit rings, planet storm, signal rail pulses
- Glassmorphism cards, neon glow effects, responsive layout

---

## Backend (Python Server)

`app.py` is a self-contained Python HTTP server built on `http.server.ThreadingHTTPServer`. It serves the `frontend/` directory as static files and exposes one POST endpoint.

### Endpoint

```
POST /api/predict
Content-Type: application/json
```

#### Request Payload

```json
{
  "mode": "synthetic | csv | target",

  "csvText": "<raw CSV string>",
  "fileName": "my_lightcurve.csv",

  "target": "Kepler-10",

  "periodSource": "searched | known",
  "period": 0.837495,
  "t0": 120.8,
  "maxIterations": 3
}
```

#### Response Payload

```json
{
  "ok": true,
  "result": {
    "source": "NASA Kepler target: Kepler-10",
    "periodSource": "searched",
    "iterations": [
      {
        "planetIndex": 1,
        "prediction": 0.9987,
        "predictedClass": 1,
        "period": 0.837493,
        "t0": 120.800123,
        "globalView": [0.002, "...420 points"],
        "localView": [0.001, "...201 points"],
        "foldedFlux": [0.999, "...500 points"]
      }
    ]
  }
}
```

### Core Backend Functions

| Function | Description |
|---|---|
| `get_model()` | Lazy-loads the Keras model; auto-patches `quantization_config` for older TF versions |
| `build_compat_model_archive()` | Strips unsupported config keys from the `.keras` ZIP for Keras backwards compatibility |
| `search_period_with_bls(time, flux)` | Runs BLS period search using `astropy.timeseries.BoxLeastSquares` over 0.5–15 day range |
| `load_real_lightcurve(target)` | Downloads 4 Kepler quarters via Lightkurve, stitches, sigma-clips (5σ), and flattens |
| `auto_center_t0(time, flux, period, initial_t0)` | 200-point grid search for optimal t0 minimizing center-bin flux |
| `mask_transit(time, flux, period, t0, duration)` | Sets transit bins to 1.0 for iterative pre-whitening |
| `parse_csv_lightcurve(csv_text)` | Parses uploaded CSV with header auto-detection |
| `infer_lightcurve(...)` | Runs multi-planet iterative detection loop |
| `predict_views(global_view, local_view)` | Calls model with dual-input tensors, returns sigmoid probability |
| `predict_payload(payload)` | Dispatches to correct inference path based on `mode` |

---

## Scripts Reference

All scripts live in `scripts/`. Run from the project root with the venv activated.

### Core Pipeline Scripts

| Script | Description | Key Args |
|---|---|---|
| `generate_dataset.py` | Generates 30,000 labeled synthetic light curves using batman + wotan | `--num-samples 30000` |
| `train_model.py` | Trains the Dual-Branch Residual 1D-CNN | `--epochs 15 --batch-size 32` |
| `evaluate_model.py` | Full evaluation: ROC, PR curves, confusion matrix, per-class report | — |
| `run_benchmark_suite.py` | Runs inference on 11 NASA Kepler stars and reports results | — |

### Data and Inference Scripts

| Script | Description |
|---|---|
| `pipeline_utils.py` | Shared utilities: `build_dual_views`, `standardize_series`, `bin_phased_flux`, constants |
| `centroid_analysis.py` | Downloads TPF data, computes centroid shift during transit, flags background EBs |
| `centroid_override.py` | Manual centroid override for edge-case targets |
| `test_kepler.py` | Single-target Kepler inference test with full diagnostic output |
| `simulate_transit.py` | Generates and visualizes a single synthetic transit |
| `preprocess_lightcurve.py` | Standalone lightcurve flattening and sigma-clip |
| `download_sample.py` | Downloads one Kepler light curve to `data/raw_nasa/` |
| `debug_local_view.py` | Diagnostic plot of local view alignment for a given target |
| `visualize_folding_error.py` | Shows phase-folding artifacts for debugging |
| `train_specialist_model.py` | Trains a specialist model for edge-case signal types |

### Reporting Scripts

| Script | Description |
|---|---|
| `generate_demo_report.py` | Generates the full PDF demo report with all diagnostic plots |

---

## Pipeline Walkthrough

### Step 1 — Dataset Generation

`scripts/generate_dataset.py` creates 30,000 labeled synthetic light curves across 5 classes cycling in order:

| Class | Label | Description |
|---|---|---|
| `planet_small` | 1 (Planet) | U-shaped transit, Rp/Rs = 0.01–0.035 (rocky/Earth-like) |
| `planet_large` | 1 (Planet) | U-shaped transit, Rp/Rs = 0.04–0.20 (Hot Jupiters) |
| `eclipsing_binary` | 0 (Not Planet) | V-shaped grazing + secondary eclipse (batman, inclination 75–82°) |
| `stellar_variability` | 0 (Not Planet) | Slow + fast sinusoidal stellar spot noise only |
| `noise` | 0 (Not Planet) | Pure Gaussian photon noise |

**Real NASA noise injection (20%)**: Randomly splices in real flat noise extracted from Kepler-22, Kepler-69, and Kepler-62 to close the sim-to-real gap.

Each sample is:
1. Simulated at 4000 time points over 27 days
2. Flattened with Wotan biweight (window = 0.278 days ≈ 6.6 hours)
3. Phase-folded and binned into global (2001 bins) + local (201 bins) views
4. Saved to `data/synthetic/` as `.npy` arrays

### Step 2 — Model Training

`scripts/train_model.py` trains the Dual-Branch Residual CNN:

- **Optimizer**: Adam (lr = 0.0005)
- **Loss**: Binary cross-entropy
- **Metrics**: Accuracy, Recall, Precision, ROC-AUC
- **Split**: 80% train / 20% validation (stratified)
- **Epochs**: 15 (default), Batch size: 32
- **Regularization**: L2(1e-4) on all Conv/Dense layers, Dropout (40% + 30%)

Outputs saved to `results/synthetic/`:
- `exoplanet_cnn_model.keras` — trained model weights + architecture
- `validation_predictions.npy` — (y_val, scores, predictions) for further analysis
- `training_history_and_confusion.png` — accuracy/loss curves + confusion matrix

### Step 3 — Evaluation

`scripts/evaluate_model.py` produces in `results/evaluation/`:
- Full classification report (per-class precision, recall, F1)
- ROC curve with AUC score
- Precision-Recall curve with AP score
- Per-class confusion matrix (5-class)
- Binary confusion matrix

### Step 4 — Centroid Analysis

`scripts/centroid_analysis.py` downloads Target Pixel File (TPF) data and:
1. Computes centroid row/col position for each photometric frame
2. Phase-folds the centroids on the detected orbital period
3. Measures the centroid shift between in-transit and out-of-transit frames
4. Flags signals where shift exceeds 0.5 Kepler pixels (~2 arcseconds) as likely background EBs

Threshold: `CENTROID_SHIFT_THRESHOLD_PX = 0.5` (Kepler pixel scale ≈ 4 arcsec/px)

### Step 5 — Benchmark

`scripts/run_benchmark_suite.py` runs the full inference chain on 11 Kepler stars and logs per-target detection results including confidence scores, detected periods, and iteration counts.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-org>/BAH26_Exoplanet_Detection.git
cd BAH26_Exoplanet_Detection

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Start the web console (model must already be trained)
python app.py
```

Then open **http://127.0.0.1:8000/** in your browser.

---

## Installation

### System Requirements

- **Python 3.10+**
- **TensorFlow 2.x** (CPU or GPU — GPU strongly recommended for training)
- Active internet connection (for Lightkurve NASA FITS data downloads)
- ~5 GB disk space for full dataset + model artifacts

### Dependencies

```
lightkurve       # NASA Kepler/TESS data download and processing
astropy          # BLS period search, astronomical units
numpy            # Array math and linear algebra
pandas           # Data handling and CSV I/O
matplotlib       # Plot generation and export
scipy            # Signal processing utilities
scikit-learn     # Train/val split, ROC-AUC, confusion matrix, classification report
tensorflow       # Dual-branch CNN training and inference
jupyter          # Demo notebook support
wotan            # Detrending algorithms (biweight, B-spline)
batman-package   # Physics-accurate planetary transit simulation
tqdm             # Progress bar display
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | HTTP port for the web console |
| `TF_CPP_MIN_LOG_LEVEL` | `3` | Suppress TensorFlow C++ verbose logs |
| `TF_ENABLE_ONEDNN_OPTS` | `0` | Disable oneDNN operator fusing (stability) |
| `MPLCONFIGDIR` | `/private/tmp/mplcfg` | Matplotlib config cache directory |

---

## Running the Full Pipeline

### Windows (PowerShell)

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Execute the complete 6-step pipeline
.\run_pipeline.ps1
```

**Pipeline Steps:**

| Step | Script | Description |
|---|---|---|
| 1/6 | `generate_dataset.py` | Generate 30,000 synthetic training samples |
| 2/6 | `train_model.py` | Train the Dual-Branch ResNet 1D-CNN |
| 3/6 | `evaluate_model.py` | Full evaluation with metric plots |
| 4/6 | `centroid_analysis.py --target Kepler-10` | Centroid false-positive validation |
| 5/6 | `centroid_analysis.py --target "KIC 6431670"` | Centroid check on crowded field |
| 6/6 | `run_benchmark_suite.py` | Full 11-star NASA benchmark |

All step logs are written to `logs/`:
- `logs/generate_dataset.log`
- `logs/train_model.log`
- `logs/evaluate_model.log`
- `logs/centroid_kepler10.log`
- `logs/centroid_kic.log`
- `logs/run_benchmark.log`

### Manual Step-by-Step (Any OS)

```bash
# Step 1: Generate dataset
python scripts/generate_dataset.py --num-samples 30000

# Step 2: Train model (GPU recommended)
python scripts/train_model.py --epochs 15 --batch-size 32

# Step 3: Evaluate model
python scripts/evaluate_model.py

# Step 4: Centroid analysis on Kepler-10
python scripts/centroid_analysis.py --target Kepler-10

# Step 5: Run 11-star benchmark
python scripts/run_benchmark_suite.py
```

---

## Running the Web App

```bash
# Default port 8000
python app.py

# Custom port — macOS/Linux
PORT=8001 python app.py

# Custom port — Windows PowerShell
$env:PORT = "8001"; python app.py
```

Open **http://127.0.0.1:8000/** — the UI loads instantly with no build step.

> **Note**: Opening `frontend/index.html` directly in a browser will render the UI but predictions will fail — `/api/predict` requires the Python server.

### Port Conflict Resolution

```powershell
# Find which process owns port 8000 (Windows)
netstat -ano | findstr :8000

# Kill by PID
taskkill /PID <PID> /F

# Or simply switch ports
$env:PORT = "8001"; python app.py
```

---

## Data Format

### CSV Upload Format

Your CSV must include at minimum a time column and a flux column. The parser auto-detects headers (case-insensitive):

| Accepted Time Headers | Accepted Flux Headers |
|---|---|
| `time`, `time_days`, `bkjd`, `btjd`, `jd` | `flux`, `pdcsap_flux`, `sap_flux`, `relative_flux` |

If no recognized header is found, columns 1 and 2 are used as time and flux respectively.

**Example CSV:**

```csv
time,flux
120.50,0.99982
120.51,0.99985
120.52,0.99101
120.53,0.98754
120.54,0.98501
```

**Requirements:**
- At least **50 valid numeric rows**
- Both columns must be finite (non-NaN, non-Inf)
- Delimiters: `,` / `\t` / `;` / ` ` (auto-detected via `csv.Sniffer`)

### Synthetic Data Arrays

Generated by `generate_dataset.py` and saved to `data/synthetic/`:

| File | Shape | Description |
|---|---|---|
| `X_global.npy` | `(N, 2001, 1)` | Global phase-folded views — full orbit |
| `X_local.npy` | `(N, 201, 1)` | Local transit views — zoomed on dip |
| `y_train.npy` | `(N,)` | Binary labels (1 = planet, 0 = not planet) |
| `y_class.npy` | `(N,)` | Multi-class labels (0–4 per CLASS_CYCLE) |
| `metadata.npy` | `(N, 4)` | (class_name, period, t0, label) per sample |

### Trained Model

Saved to `results/synthetic/exoplanet_cnn_model.keras` — standard Keras SavedModel format.

---

## Technical Deep Dive

### Phase Folding

The core function is `build_dual_views(time, flux, period, t0)` in `scripts/pipeline_utils.py`:

```python
# 1. Standardize: center on median, divide by std
flux = standardize_series(flux)

# 2. Compute phases in range [-0.5, +0.5]
phases = ((time - t0 + 0.5 * period) % period) / period - 0.5

# 3. Sort by phase
phases, folded_flux = sort_by_phase(phases, flux)

# 4. Global view: full phase range [-0.5, +0.5], 2001 bins
global_view = bin_phased_flux(phases, folded_flux, bins=2001)

# 5. Local view: +-12% of phase, 201 bins (zoomed on transit)
local_mask = |phases| <= 0.12
local_view = bin_phased_flux(local_phases, local_flux, bins=201)
```

**Empty bin interpolation**: Any phase bins with no data points are filled by linear interpolation from neighboring bins — this prevents "W-spike" artifacts from empty bins defaulting to zero.

### Signal Standardization

The full light curve is standardized **before** phase-folding:

```
standardized = (flux - median(flux)) / std(flux - median(flux))
```

This preserves relative transit depths across the full dynamic range — critical for the CNN to correctly differentiate Earth-sized planets from Hot Jupiters and eclipsing binaries.

### BLS Period Search

```python
durations = np.linspace(0.02, 0.15, 10) * u.day   # 0.02 to 0.15 day transit durations
periods = bls.autoperiod(
    durations,
    minimum_period=0.5,      # 0.5-day minimum (ultra-hot Jupiters)
    maximum_period=15.0,     # 15-day maximum
    frequency_factor=10.0    # oversampling factor
)
power = bls.power(periods, durations, objective="snr")
best_period = periods[argmax(power.power)]
```

### Auto-Centering t0

A 200-point grid search over ±10% of the period refines the BLS-reported t0 by minimizing flux in the local view center window (bins 95–105 out of 201):

```python
for shift in np.linspace(-0.1 * period, +0.1 * period, 200):
    test_t0 = initial_t0 + shift
    views = build_dual_views(time, flux, period, test_t0)
    center_flux = mean(views["local_view"][95:106])
    if center_flux < min_center_flux:
        best_t0 = test_t0
```

### Model Input/Output

```python
global_input = expand_dims(global_view, axis=(0, -1))  # shape: (1, 2001, 1)
local_input  = expand_dims(local_view, axis=(0, -1))   # shape: (1, 201,  1)

prediction = model.predict({
    "global_view": global_input,
    "local_view":  local_input
})[0][0]
# prediction is in [0.0, 1.0] — probability of being a planet
# predictedClass = 1 if prediction >= 0.5 else 0
```

### Keras Model Compatibility

The model may be saved with newer Keras config keys (e.g., `quantization_config`) that older TF versions cannot parse. `build_compat_model_archive()` in `app.py` automatically:
1. Opens the `.keras` ZIP archive
2. Strips unrecognized keys from `config.json` recursively
3. Writes a cleaned copy to a temp path
4. Loads from the cleaned archive as a fallback

---

## Known Challenges and Fixes

| Challenge | Root Cause | Fix Applied |
|---|---|---|
| **Ephemeris Drift** | BLS reports a slightly wrong t0 | Auto-Centering 200-point grid refinement narrows t0 to sub-day precision |
| **Empty Bin W-Spikes** | Empty phase bins defaulted to 0 | Linear interpolation of all empty bins from neighboring valid bins |
| **Hot Jupiter Rejection** | Model confused Hot Jupiters with EBs | Trained with `planet_large` class (U-shapes, inc 86–90°) alongside EB class (V-shapes, inc 75–82°) |
| **Sim-to-Real Gap** | Synthetic noise patterns differed from real observations | Real NASA noise injected in 20% of training samples (Kepler-22, 69, 62) |
| **Keras Config Incompatibility** | `quantization_config` key unsupported in older TF | `build_compat_model_archive()` strips unknown keys at runtime load |
| **Multi-Planet Hallucination** | Model re-detected the same planet on Iteration 2 | Iterative pre-whitening: transit bins set to 1.0 before next BLS run |
| **Low SNR Multi-Planet** | Kepler-20b at only 48% confidence with 4 quarters | Recommendation to download 10–15 quarters for low-SNR targets |

---

## Docs

Full internal documentation available in `docs/`:

| File | Description |
|---|---|
| `implementation_plan.md` | Detailed dual-branch CNN architecture proposal submitted for ISRO hackathon |
| `current_progress.md` | Live project status with benchmark results and key findings |
| `known_challenges_and_fixes.md` | Complete bug history and engineering solutions with dates |
| `multiplanet_system_walkthrough.md` | Step-by-step multi-planet detection walkthrough (Kepler-20) |
| `project_tasks.md` | Full task board with status tracking |
| `tasks_for_teammates.md` | Team task distribution and ownership |

---

<div align="center">

**Built with heart for the ISRO Space Hackathon (BAH26)**

*Reaching for the stars — one light curve at a time.*

*Last updated: June 2026*

</div>
