import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

print("Loading Kepler-10 data...")
# Download the data again (lightkurve will use the cached version in the /data folder!)
search_result = lk.search_lightcurve("Kepler-10", author="Kepler", cadence="short")
lc = search_result[0].download(download_dir="./data")

# Clean and flatten the data just like before
lc_clean = lc.remove_nans().remove_outliers(sigma=5)
lc_flat = lc_clean.flatten(window_length=401)

print("Data loaded and cleaned. Now searching for the planet...")

# THE MAGIC HAPPENS HERE: Phase-Folding
# Kepler-10b is a known planet with an orbital period of exactly 0.837495 days.
# We will "fold" the timeline on top of itself every 0.837495 days.
period = 0.837495 
t0 = 120.8  # The approximate start time of the first transit in this dataset

print(f"Folding the light curve over a period of {period} days...")
# Fold the light curve
lc_folded = lc_flat.fold(period=period, epoch_time=t0)

# Bin the data (Average nearby points together to reduce noise even more)
# This is crucial for Neural Networks!
lc_binned = lc_folded.bin(time_bin_size=0.005)

# Plotting the results
fig, ax = plt.subplots(figsize=(10, 5))
lc_folded.scatter(ax=ax, s=1, alpha=0.2, label='Raw Folded Data')
lc_binned.scatter(ax=ax, s=20, color='red', label='Binned Data (Signal)', zorder=10)

ax.set_title("Phase-Folded Light Curve of Kepler-10b (The Exoplanet Dip!)")
ax.set_xlim(-0.1, 0.1) # Zoom in exactly on the transit event

plt.savefig("kepler10_exoplanet_dip.png")
print("Success! I saved the exoplanet transit dip as kepler10_exoplanet_dip.png.")
