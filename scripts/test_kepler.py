import os
import numpy as np
import lightkurve as lk
import tensorflow as tf
from tensorflow.keras.models import load_model

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw_nasa")
MODEL_PATH = os.path.join(BASE_DIR, "results", "synthetic", "exoplanet_cnn_model.keras")

# Configuration
BINS = 201
PERIOD = 0.837495   # Known period of Kepler-10b
T0 = 120.8          # Known transit epoch

print("1. Loading the trained AI Model...")
model = load_model(MODEL_PATH)

print("2. Downloading/Loading Real Kepler-10 Data from NASA...")
search_result = lk.search_lightcurve("Kepler-10", author="Kepler", cadence="short")
lc = search_result[0].download(download_dir=DATA_DIR)

print("3. Preprocessing the Real Data (Cleaning & Flattening)...")
lc_clean = lc.remove_nans().remove_outliers(sigma=5)
lc_flat = lc_clean.flatten(window_length=401)

# Extract time and flux as pure numbers
time = lc_flat.time.value
flux = lc_flat.flux.value

print("4. Phase-Folding and Binning the Data exactly like our Synthetic Training Data...")
# We must use the exact same mathematical binning technique we used in generate_dataset.py
phases = ((time - T0 + 0.5 * PERIOD) % PERIOD) - 0.5 * PERIOD
phases /= PERIOD # Normalize phase to [-0.5, 0.5]

# Sort by phase
sort_idx = np.argsort(phases)
phases = phases[sort_idx]
folded_flux = flux[sort_idx]

# Bin into exactly 201 points
bin_edges = np.linspace(-0.5, 0.5, BINS + 1)
bin_indices = np.digitize(phases, bin_edges)

binned_flux = np.zeros(BINS)
for i in range(1, BINS + 1):
    points_in_bin = folded_flux[bin_indices == i]
    if len(points_in_bin) > 0:
        binned_flux[i-1] = np.mean(points_in_bin)
    else:
        binned_flux[i-1] = 1.0 # Baseline if empty

# Shape the data for the CNN: [1 sample, 201 bins, 1 feature]
input_data = np.expand_dims(binned_flux, axis=0)
input_data = np.expand_dims(input_data, axis=-1)

print("\n--- THE MOMENT OF TRUTH ---")
print("Passing the Real NASA Data into our AI Model...")

# Make the prediction!
prediction = model.predict(input_data, verbose=0)[0][0]

print("\n=======================================================")
if prediction > 0.5:
    print(f"🌟 PLANET DETECTED! 🌟")
    print(f"Confidence Score: {prediction * 100:.2f}%")
    print("The AI successfully found Kepler-10b in the raw space noise!")
else:
    print(f"❌ NO PLANET FOUND.")
    print(f"Confidence Score: {prediction * 100:.2f}%")
    print("The AI thinks this is just an empty star or binary system.")
print("=======================================================\n")
