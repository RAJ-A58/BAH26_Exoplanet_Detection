import lightkurve as lk
import matplotlib.pyplot as plt
import os

# Create a folder to save our space data
os.makedirs("data", exist_ok=True)

print("Hello from E drive! Connecting to NASA to download Kepler-10 data...")

# Download the data for a star called Kepler-10
search_result = lk.search_lightcurve("Kepler-10", author="Kepler", cadence="short")
lc = search_result[0].download(download_dir="./data")

# Clean the data
lc_clean = lc.remove_nans().remove_outliers(sigma=5)
lc_flat = lc_clean.flatten(window_length=401)

# Plot the data and save the image
ax = lc_flat.plot(label="Kepler-10 Star Brightness")
plt.savefig("my_first_lightcurve.png")

print("Success! I saved an image called my_first_lightcurve.png in your folder.")
