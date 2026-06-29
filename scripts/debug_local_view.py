import os
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt

from pipeline_utils import build_dual_views, RAW_NASA_DATA_DIR, RESULTS_DIR

def run_debug():
    print("Loading Kepler-10...")
    search_result = lk.search_lightcurve("Kepler-10", author="Kepler", cadence="short")
    lc = search_result[0].download(download_dir=RAW_NASA_DATA_DIR)
    lc_clean = lc.remove_nans().remove_outliers(sigma=5)
    lc_flat = lc_clean.flatten(window_length=401)
    
    time = lc_flat.time.value
    flux = lc_flat.flux.value
    
    # The output from the autoperiod search
    searched_period = 0.837516
    searched_t0 = 201.086085
    
    # NASA's known values
    known_period = 0.837495
    known_t0 = 120.8
    
    views_searched = build_dual_views(time, flux, searched_period, searched_t0)
    views_known = build_dual_views(time, flux, known_period, known_t0)
    
    # Plot Local Views (The 201 bins)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(views_searched['local_view'], marker='o', markersize=2, linestyle='-')
    ax1.set_title("AI Local View (Autoperiod BLS Search)")
    ax1.set_xlabel("Bin Index (0 to 200)")
    ax1.set_ylabel("Normalized Flux")
    
    ax2.plot(views_known['local_view'], marker='o', markersize=2, linestyle='-', color='green')
    ax2.set_title("AI Local View (Perfect Known Math)")
    ax2.set_xlabel("Bin Index (0 to 200)")
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "ai_local_view_debug.png")
    plt.savefig(save_path)
    print(f"Saved debug plot to {save_path}")

if __name__ == "__main__":
    run_debug()
