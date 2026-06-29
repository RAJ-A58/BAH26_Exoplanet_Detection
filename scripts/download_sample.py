import lightkurve as lk
import matplotlib.pyplot as plt
import os

# Define the root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw_nasa")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "nasa")

# Create directories to keep things organized
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Hello from the scripts folder! Connecting to NASA to download Kepler-10 data...")

# Download the data for a star called Kepler-10
search_result = lk.search_lightcurve("Kepler-10", author="Kepler", cadence="short")
lc = search_result[0].download(download_dir=DATA_DIR)

# Clean the data
lc_clean = lc.remove_nans().remove_outliers(sigma=5)
lc_flat = lc_clean.flatten(window_length=401)

# Plot the data and save the image
ax = lc_flat.plot(label="Kepler-10 Star Brightness")

save_path = os.path.join(RESULTS_DIR, "my_first_lightcurve.png")
plt.savefig(save_path)

print(f"Plot saved in the {RESULTS_DIR} folder!")
