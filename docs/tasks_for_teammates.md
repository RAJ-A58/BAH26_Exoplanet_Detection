# Future Tasks for Hackathon Teammates

## Problem Statement
Our current CNN pipeline (with the Iterative Pre-Whitening algorithm) perfectly detects **100% of single-planet systems** in our 11-star benchmark suite. It also successfully isolated and discovered **Kepler-20c** (10.85 days, 72.58% confidence). 

However, it struggles to cross the 50% confidence threshold for extremely small, rocky/Neptune-sized multi-planets buried in stellar noise. 
- **Kepler-20b** (3.69 days): Period found, but confidence stalled at **48.13%**.
- **Kepler-62b** (5.71 days): Period found, but confidence stalled at **12.54%**.

## Action Items

### 1. The SNR Data Scaling Task (High Priority)
**Assigned to:** Data Engineer / Python Developer
- **The Issue:** We are currently only downloading 4 quarters (~400 days) of short-cadence NASA data to save execution time. For small planets like Kepler-20b, 400 days does not provide enough transits to overcome the noise.
- **The Task:** Update `test_kepler.py` to download `search_result[:15]` (all 15 quarters, ~1500 days). This will multiply the SNR and push the 48.13% confidence for Kepler-20b over the 50% finish line.
- **Note:** This requires running the Box Least Squares (BLS) algorithm on 1.5 million data points, which takes too long on a standard laptop CPU. This must be run overnight or on a fast cloud CPU instance.

### 2. The GPU Cloud Migration Task (Medium Priority)
**Assigned to:** Cloud / ML Engineer
- **The Issue:** TensorFlow GPU acceleration is not supported on native Windows for TF >= 2.11. The model training and inference are currently running on CPU.
- **The Task:** Migrate the codebase (`train_model.py` and `test_kepler.py`) to Kaggle or Google Colab. 
- **Goal:** Re-train the CNN on a GPU with a much larger dataset (e.g., 1,000,000 synthetic samples instead of 300,000) with even shallower transit depths injected to make the AI hyper-sensitive to Super-Earths like Kepler-62b.

### 3. The BLS Fine-Tuning Task (Low Priority)
**Assigned to:** Astronomer / Physicist
- **The Issue:** The BLS period-hunter sometimes zeroes in on a period that is off by a tiny fraction (e.g., 5.715 days instead of 5.713 days for Kepler-62). Over 400 days, this tiny fraction causes the folded transits to "smear" out, destroying the U-shape the CNN looks for.
- **The Task:** Improve the `auto_center_t0` function in `test_kepler.py` to not only sweep the transit epoch (`t0`), but also perform a micro-sweep of the `period` itself to perfectly align the transits before passing them to the CNN.
