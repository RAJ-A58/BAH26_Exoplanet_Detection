import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import csv

import lightkurve as lk
import numpy as np
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
from tensorflow.keras.models import load_model

from pipeline_utils import (
    BENCHMARK_RESULTS_DIR,
    RAW_NASA_DATA_DIR,
    SYNTHETIC_RESULTS_DIR,
    build_dual_views,
    ensure_project_dirs,
)


MODEL_PATH = os.path.join(SYNTHETIC_RESULTS_DIR, "exoplanet_cnn_model.keras")
BENCHMARK_CSV = os.path.join(BENCHMARK_RESULTS_DIR, "kepler_benchmark_results.csv")


def search_period_with_bls(time: np.ndarray, flux: np.ndarray):
    # Use fewer durations to speed up the search on massive 10-quarter datasets
    durations = np.linspace(0.02, 0.15, 5) * u.day
    bls = BoxLeastSquares(time * u.day, flux)
    
    # frequency_factor=2.0 is perfectly fine for finding the spike, reducing grid size by 5x!
    periods = bls.autoperiod(durations, minimum_period=0.5, maximum_period=15.0, frequency_factor=2.0)
    
    # Use Signal-to-Noise Ratio (SNR) instead of raw power for shallow rocky planets
    power = bls.power(periods, durations, objective='snr')
    
    best_idx = int(np.argmax(power.power))
    
    return float(power.period[best_idx].value), float(power.transit_time[best_idx].value), power


def load_real_lightcurve(target: str):
    search_result = lk.search_lightcurve(target, author="Kepler", cadence="short")
    
    # Download up to 4 quarters of data to drastically boost the Signal-to-Noise Ratio (SNR)
    lc_collection = search_result[:4].download_all(download_dir=RAW_NASA_DATA_DIR)
    
    if lc_collection is None or len(lc_collection) == 0:
        raise ValueError(f"No short cadence data found for {target}.")
        
    lc = lc_collection.stitch()
    lc_clean = lc.remove_nans().remove_outliers(sigma=5)
    lc_flat = lc_clean.flatten(window_length=401)
    return lc_flat.time.value, lc_flat.flux.value


def append_benchmark_row(row):
    file_exists = os.path.exists(BENCHMARK_CSV)
    with open(BENCHMARK_CSV, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "known_label",
                "prediction_score",
                "predicted_class",
                "period_source",
                "period_days",
                "t0_days",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def auto_center_t0(time: np.ndarray, flux: np.ndarray, period: float, initial_t0: float) -> float:
    from pipeline_utils import build_dual_views
    best_t0 = initial_t0
    min_center_flux = float('inf')
    
    # Sweep t0 back and forth by up to 10% of the period
    shift_range = np.linspace(-0.1 * period, 0.1 * period, 200)
    
    for shift in shift_range:
        test_t0 = initial_t0 + shift
        views = build_dual_views(time, flux, period, test_t0)
        
        # The center of the local view is exactly bin 100. We look at bins 95 to 105.
        center_flux = np.mean(views["local_view"][95:106])
        if center_flux < min_center_flux:
            min_center_flux = center_flux
            best_t0 = test_t0
            
    return best_t0

def mask_transit(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float = 0.2) -> np.ndarray:
    masked_flux = flux.copy()
    # Calculate phase from -0.5*period to 0.5*period
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    # Find points inside the transit dip (give it a generous duration window)
    transit_mask = np.abs(phase) < (duration / 2)
    # Replace those points with 1.0 (the baseline flux)
    masked_flux[transit_mask] = 1.0
    return masked_flux

def main():
    parser = argparse.ArgumentParser(description="Run real Kepler inference with optional BLS period search.")
    parser.add_argument("--target", default="Kepler-10", help="Kepler target name to download.")
    parser.add_argument("--known-label", default="confirmed_planet", help="Human-readable label for benchmark logging.")
    parser.add_argument("--period", type=float, default=0.837495, help="Known orbital period in days.")
    parser.add_argument("--t0", type=float, default=120.8, help="Known transit epoch in days.")
    parser.add_argument(
        "--period-source",
        choices=("known", "searched"),
        default="searched",
        help="Use known ephemeris for debugging or BLS search for a real pipeline path.",
    )
    args = parser.parse_args()

    ensure_project_dirs()

    print("1. Loading the trained AI Model...")
    model = load_model(MODEL_PATH)

    print(f"2. Downloading/Loading Real {args.target} Data from NASA...")
    time, original_flux = load_real_lightcurve(args.target)

    current_flux = original_flux.copy()
    max_iterations = 3
    planets_found = 0

    for i in range(max_iterations):
        planet_idx = i + 1
        print(f"\n--- HUNTING FOR PLANET {planet_idx} ---")

        if args.period_source == "searched":
            print("3. Searching for the period with Box Least Squares...")
            period, initial_t0, _ = search_period_with_bls(time, current_flux)
            print("   Auto-centering the transit epoch (t0)...")
            t0 = auto_center_t0(time, current_flux, period, initial_t0)
        else:
            print("3. Using the provided orbital period and transit epoch...")
            period, t0 = args.period, args.t0

        print(f"Selected period: {period:.6f} days")
        print(f"Selected transit epoch: {t0:.6f} days")

        print("4. Building standardized global and local folded views...")
        dual_views = build_dual_views(time, current_flux, period=period, t0=t0)
        global_input = np.expand_dims(dual_views["global_view"], axis=(0, -1))
        local_input = np.expand_dims(dual_views["local_view"], axis=(0, -1))

        print("\n--- THE MOMENT OF TRUTH ---")
        print("Passing the real NASA data into the dual-view model...")
        prediction = model.predict({"global_view": global_input, "local_view": local_input}, verbose=0)[0][0]
        predicted_class = int(prediction >= 0.5)

        target_name_logged = f"{args.target} (Planet {planet_idx})"

        row = {
            "target": target_name_logged,
            "known_label": args.known_label,
            "prediction_score": f"{prediction:.6f}",
            "predicted_class": predicted_class,
            "period_source": args.period_source,
            "period_days": f"{period:.6f}",
            "t0_days": f"{t0:.6f}",
        }
        append_benchmark_row(row)

        print("\n=======================================================")
        print(f"Target: {target_name_logged}")
        print(f"Known label: {args.known_label}")
        print(f"Period source: {args.period_source}")
        if predicted_class == 1:
            print("PLANET DETECTED")
            print(f"Confidence Score: {prediction * 100:.2f}%")
            print(f"Benchmark row saved to {BENCHMARK_CSV}")
            print("=======================================================\n")
            
            planets_found += 1
            print(f"Pre-Whitening: Erasing Planet {planet_idx}'s transits from the light curve...")
            current_flux = mask_transit(time, current_flux, period, t0, duration=0.4)
            
            # If using hardcoded known period, we only run once since we don't have multiple hardcoded periods
            if args.period_source == "known":
                break
        else:
            print("NO PLANET DETECTED")
            print(f"Confidence Score: {prediction * 100:.2f}%")
            print(f"Benchmark row saved to {BENCHMARK_CSV}")
            print("=======================================================\n")
            
            if args.period_source == "known":
                break
                
            print(f"Pre-Whitening: Erasing the noise signal from the light curve...")
            current_flux = mask_transit(time, current_flux, period, t0, duration=0.4)
            # We do NOT break here. We masked the noise, so BLS can find the real planet next!


if __name__ == "__main__":
    main()
