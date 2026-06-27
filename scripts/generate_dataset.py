import argparse
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import batman
import numpy as np
from wotan import flatten
import lightkurve as lk
from sklearn.utils import shuffle

from pipeline_utils import SYNTHETIC_DATA_DIR, RAW_NASA_DATA_DIR, build_dual_views, ensure_project_dirs


DEFAULT_NUM_SAMPLES = 30000
DAYS_OBSERVED = 27.0
DATA_POINTS = 4000
CLASS_CYCLE = ("planet_small", "planet_large", "eclipsing_binary", "stellar_variability", "noise")

REAL_NOISE_POOL = []

def initialize_real_noise():
    print("Downloading Real NASA Noise Pool (this will take a minute)...", flush=True)
    targets = ["Kepler-22", "Kepler-69", "Kepler-62"]
    for target in targets:
        try:
            lc = lk.search_lightcurve(target, author="Kepler", cadence="short")[0].download(download_dir=RAW_NASA_DATA_DIR)
            lc = lc.remove_nans().remove_outliers(5)
            flat = flatten(lc.time.value, lc.flux.value, window_length=0.278, method="biweight")
            REAL_NOISE_POOL.append(flat)
            print(f"Successfully downloaded and processed {target} for noise pool.", flush=True)
        except Exception as e:
            print(f"Failed to download {target}: {e}", flush=True)


def add_stellar_variability(time: np.ndarray, amplitude_scale: float = 1.0) -> np.ndarray:
    slow_component = np.random.uniform(0.001, 0.006) * amplitude_scale * np.sin(
        2 * np.pi * time / np.random.uniform(6.0, 18.0)
    )
    fast_component = np.random.uniform(0.0003, 0.0015) * amplitude_scale * np.sin(
        2 * np.pi * time / np.random.uniform(1.5, 4.5) + np.random.uniform(0, np.pi)
    )
    return slow_component + fast_component


def create_transit_flux(
    time: np.ndarray,
    period: float,
    t0: float,
    radius_ratio: float,
    inclination_range: tuple[float, float],
) -> np.ndarray:
    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = radius_ratio
    params.a = np.random.uniform(8.0, 22.0)
    params.inc = np.random.uniform(*inclination_range)
    params.ecc = np.random.uniform(0.0, 0.1)
    params.w = np.random.uniform(80.0, 100.0)
    params.u = [0.1, 0.3]
    params.limb_dark = "quadratic"

    model = batman.TransitModel(params, time)
    return model.light_curve(params)


def inject_eclipsing_binary(
    time: np.ndarray,
    period: float,
    t0: float,
) -> np.ndarray:
    # Use Batman to generate realistic V-shaped eclipsing binaries (grazing transits)
    primary = create_transit_flux(
        time,
        period=period,
        t0=t0,
        radius_ratio=np.random.uniform(0.15, 0.35),
        inclination_range=(75.0, 82.0), # Grazing creates V-shapes
    )
    # Add a secondary eclipse offset by half a period
    secondary = create_transit_flux(
        time,
        period=period,
        t0=t0 + 0.5 * period,
        radius_ratio=np.random.uniform(0.05, 0.15),
        inclination_range=(75.0, 82.0),
    )
    return primary + secondary - 1.0


def generate_sample(sample_class: str):
    time = np.linspace(0, DAYS_OBSERVED, DATA_POINTS)
    period = np.random.uniform(2.0, 12.0)
    t0 = np.random.uniform(0.5, period)
    base_flux = np.ones(DATA_POINTS, dtype=np.float32)

    if sample_class == "planet_small":
        signal_flux = create_transit_flux(
            time,
            period=period,
            t0=t0,
            radius_ratio=np.random.uniform(0.01, 0.035),
            inclination_range=(87.0, 90.0),
        )
        label = 1
    elif sample_class == "planet_large":
        signal_flux = create_transit_flux(
            time,
            period=period,
            t0=t0,
            radius_ratio=np.random.uniform(0.04, 0.20), # Allow massive Hot Jupiters
            inclination_range=(86.0, 90.0), # Perfect U-shapes
        )
        label = 1
    elif sample_class == "eclipsing_binary":
        signal_flux = inject_eclipsing_binary(time, period=period, t0=t0)
        label = 0
    elif sample_class == "stellar_variability":
        signal_flux = base_flux
        label = 0
    elif sample_class == "noise":
        signal_flux = base_flux
        label = 0
    else:
        raise ValueError(f"Unsupported sample_class: {sample_class}")

    # Inject Real NASA Noise 20% of the time if available
    if len(REAL_NOISE_POOL) > 0 and np.random.rand() < 0.20:
        real_flat = REAL_NOISE_POOL[np.random.randint(0, len(REAL_NOISE_POOL))]
        if len(real_flat) > DATA_POINTS:
            start_idx = np.random.randint(0, len(real_flat) - DATA_POINTS)
            base_noise = real_flat[start_idx : start_idx + DATA_POINTS]
        else:
            base_noise = np.pad(real_flat, (0, DATA_POINTS - len(real_flat)), mode='wrap')
            
        noisy_flux = signal_flux + base_noise - 1.0
        flattened_flux = flatten(time, noisy_flux, window_length=0.278, method="biweight")
    else:
        variability_scale = 1.6 if sample_class in {"stellar_variability", "eclipsing_binary"} else 1.0
        stellar_wobble = add_stellar_variability(time, amplitude_scale=variability_scale)
        noise = np.random.normal(0, np.random.uniform(0.0004, 0.0015), DATA_POINTS)
        noisy_flux = signal_flux + stellar_wobble + noise
        flattened_flux = flatten(time, noisy_flux, window_length=0.278, method="biweight")

    dual_views = build_dual_views(time, flattened_flux, period=period, t0=t0)

    return {
        "global_view": dual_views["global_view"],
        "local_view": dual_views["local_view"],
        "label": label,
        "sample_class": sample_class,
        "period": period,
        "t0": t0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic exoplanet training data.")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of synthetic samples to generate.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N generated samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_dirs()
    os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)
    
    initialize_real_noise()

    print(f"Generating {args.num_samples} light curves...", flush=True)
    global_views = []
    local_views = []
    labels = []
    metadata_rows = []

    for index in range(args.num_samples):
        sample_class = CLASS_CYCLE[index % len(CLASS_CYCLE)]
        try:
            sample = generate_sample(sample_class)
        except Exception as exc:
            print(f"Skipping sample {index} due to error: {exc}", flush=True)
            continue

        global_views.append(sample["global_view"])
        local_views.append(sample["local_view"])
        labels.append(sample["label"])
        metadata_rows.append((sample["sample_class"], sample["period"], sample["t0"], sample["label"]))

        if (index + 1) % args.progress_every == 0:
            print(f"Generated {index + 1} / {args.num_samples} samples...", flush=True)

    # Shuffle the dataset completely before saving
    global_views, local_views, labels, metadata_rows = shuffle(
        global_views, local_views, labels, metadata_rows, random_state=42
    )

    X_global = np.expand_dims(np.asarray(global_views, dtype=np.float32), axis=-1)
    X_local = np.expand_dims(np.asarray(local_views, dtype=np.float32), axis=-1)
    y_array = np.asarray(labels, dtype=np.int32)

    np.save(os.path.join(SYNTHETIC_DATA_DIR, "X_global.npy"), X_global)
    np.save(os.path.join(SYNTHETIC_DATA_DIR, "X_local.npy"), X_local)
    np.save(os.path.join(SYNTHETIC_DATA_DIR, "y_train.npy"), y_array)
    np.save(
        os.path.join(SYNTHETIC_DATA_DIR, "metadata.npy"),
        np.asarray(metadata_rows, dtype=object),
        allow_pickle=True,
    )

    print("\n--- DATASET GENERATION COMPLETE ---", flush=True)
    print(f"X_global shape: {X_global.shape}", flush=True)
    print(f"X_local shape: {X_local.shape}", flush=True)
    print(f"y_train shape: {y_array.shape}", flush=True)
    print(f"Saved directly to: {SYNTHETIC_DATA_DIR}", flush=True)


if __name__ == "__main__":
    main()
