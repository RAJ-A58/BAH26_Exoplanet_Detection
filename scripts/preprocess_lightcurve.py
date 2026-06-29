import lightkurve as lk
import matplotlib.pyplot as plt
import os

# Define the root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw_nasa")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "nasa")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading Kepler-10 data...")
# Download the data again (lightkurve will use the cached version in the data/raw_nasa folder!)
search_result = lk.search_lightcurve("Kepler-10", author="Kepler", cadence="short")
lc = search_result[0].download(download_dir=DATA_DIR)

# Clean and flatten the data just like before
lc_clean = lc.remove_nans().remove_outliers(sigma=5)
lc_flat = lc_clean.flatten(window_length=401)

print("Data loaded and cleaned. Now searching for the planet...")

# Phase-Folding
period = 0.837495 
t0 = 120.8  

print(f"Folding the light curve over a period of {period} days...")
lc_folded = lc_flat.fold(period=period, epoch_time=t0)
lc_binned = lc_folded.bin(time_bin_size=0.005)

# Plotting the results
fig, ax = plt.subplots(figsize=(10, 5))
lc_folded.scatter(ax=ax, s=1, alpha=0.2, label='Raw Folded Data')
lc_binned.scatter(ax=ax, s=20, color='red', label='Binned Data (Signal)', zorder=10)

ax.set_title("Phase-Folded Light Curve of Kepler-10b (The Exoplanet Dip!)")
ax.set_xlim(-0.1, 0.1) 

save_path = os.path.join(RESULTS_DIR, "kepler10_exoplanet_dip.png")
plt.savefig(save_path)
print(f"Success! I saved the exoplanet transit dip as {save_path}")
