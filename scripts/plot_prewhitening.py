import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
import astropy.units as u
import wotan
import os

def run():
    print("Downloading Kepler-20 data...")
    search_result = lk.search_lightcurve("Kepler-20", author="Kepler", cadence="short")
    lc_collection = search_result[:4].download_all()
    lc = lc_collection.stitch()
    
    # Clean data
    lc = lc.remove_nans().remove_outliers(sigma=5)
    time = lc.time.value
    flux = lc.flux.value
    
    print("Detrending...")
    flattened_flux = wotan.flatten(time, flux, window_length=0.278, method='biweight')
    
    # BLS 1 (Before Pre-Whitening)
    print("Running BLS 1...")
    durations = np.linspace(0.02, 0.15, 5) * u.day
    bls1 = BoxLeastSquares(time * u.day, flattened_flux)
    periods1 = bls1.autoperiod(durations, minimum_period=1.0, maximum_period=15.0, frequency_factor=2.0)
    power1 = bls1.power(periods1, durations)
    best_period1 = periods1[np.argmax(power1.power)]
    best_t0_1 = power1.transit_time[np.argmax(power1.power)]
    
    # Pre-Whiten (Mask)
    print(f"Masking Planet 1 (Period: {best_period1.value:.2f} days)...")
    phase = (time - best_t0_1.value) % best_period1.value
    phase[phase > best_period1.value / 2] -= best_period1.value
    mask = np.abs(phase) < 0.1  # Mask transit window
    whitened_flux = flattened_flux.copy()
    whitened_flux[mask] = 1.0  # Erase the dip
    
    # BLS 2 (After Pre-Whitening)
    print("Running BLS 2...")
    bls2 = BoxLeastSquares(time * u.day, whitened_flux)
    periods2 = bls2.autoperiod(durations, minimum_period=1.0, maximum_period=15.0, frequency_factor=2.0)
    power2 = bls2.power(periods2, durations)
    best_period2 = periods2[np.argmax(power2.power)]
    
    # Plotting
    print("Generating Plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(periods1, power1.power, color='black', lw=1)
    ax1.set_title(f"Iteration 1: Strongest Signal at {best_period1.value:.2f} Days (Kepler-20c)", fontsize=14, color='darkblue')
    ax1.set_ylabel("BLS Power")
    ax1.axvline(best_period1.value, color='red', linestyle='--', alpha=0.7)
    
    ax2.plot(periods2, power2.power, color='black', lw=1)
    ax2.set_title(f"Iteration 2 (After Pre-Whitening): New Signal at {best_period2.value:.2f} Days (Kepler-20b)", fontsize=14, color='darkgreen')
    ax2.set_xlabel("Orbital Period (Days)")
    ax2.set_ylabel("BLS Power")
    ax2.axvline(best_period2.value, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("E:/ISRO/docs/prewhitening_kepler20.png", dpi=300)
    print("Plot saved to E:/ISRO/docs/prewhitening_kepler20.png")

if __name__ == "__main__":
    run()
