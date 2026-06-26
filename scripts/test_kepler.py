import argparse
import csv
import os

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
    durations = np.linspace(0.05, 0.25, 7) * u.day
    periods = np.linspace(0.5, 20.0, 3000) * u.day
    bls = BoxLeastSquares(time * u.day, flux)
    power = bls.power(periods, durations)
    best_idx = int(np.argmax(power.power))
    return float(power.period[best_idx].value), float(power.transit_time[best_idx].value), power


def load_real_lightcurve(target: str):
    search_result = lk.search_lightcurve(target, author="Kepler", cadence="short")
    lc = search_result[0].download(download_dir=RAW_NASA_DATA_DIR)
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
    time, flux = load_real_lightcurve(args.target)

    if args.period_source == "searched":
        print("3. Searching for the period with Box Least Squares...")
        period, t0, _ = search_period_with_bls(time, flux)
    else:
        print("3. Using the provided orbital period and transit epoch...")
        period, t0 = args.period, args.t0

    print(f"Selected period: {period:.6f} days")
    print(f"Selected transit epoch: {t0:.6f} days")

    print("4. Building standardized global and local folded views...")
    dual_views = build_dual_views(time, flux, period=period, t0=t0)
    global_input = np.expand_dims(dual_views["global_view"], axis=(0, -1))
    local_input = np.expand_dims(dual_views["local_view"], axis=(0, -1))

    print("\n--- THE MOMENT OF TRUTH ---")
    print("Passing the real NASA data into the dual-view model...")
    prediction = model.predict({"global_view": global_input, "local_view": local_input}, verbose=0)[0][0]
    predicted_class = int(prediction >= 0.5)

    row = {
        "target": args.target,
        "known_label": args.known_label,
        "prediction_score": f"{prediction:.6f}",
        "predicted_class": predicted_class,
        "period_source": args.period_source,
        "period_days": f"{period:.6f}",
        "t0_days": f"{t0:.6f}",
    }
    append_benchmark_row(row)

    print("\n=======================================================")
    print(f"Target: {args.target}")
    print(f"Known label: {args.known_label}")
    print(f"Period source: {args.period_source}")
    if predicted_class == 1:
        print("PLANET DETECTED")
    else:
        print("NO PLANET DETECTED")
    print(f"Confidence Score: {prediction * 100:.2f}%")
    print(f"Benchmark row saved to {BENCHMARK_CSV}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
