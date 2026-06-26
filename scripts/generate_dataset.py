import argparse
import os

import batman
import numpy as np
from wotan import flatten

from pipeline_utils import SYNTHETIC_DATA_DIR, build_dual_views, ensure_project_dirs


DEFAULT_NUM_SAMPLES = 8000
DAYS_OBSERVED = 27.0
DATA_POINTS = 4000
CLASS_CYCLE = ("planet_small", "planet_large", "eclipsing_binary", "stellar_variability", "noise")


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
    primary_depth = np.random.uniform(0.03, 0.12)
    secondary_depth = primary_depth * np.random.uniform(0.35, 0.8)
    width = np.random.uniform(0.03, 0.07) * period

    phase_primary = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    phase_secondary = ((time - (t0 + 0.5 * period) + 0.5 * period) % period) - 0.5 * period

    primary = primary_depth * np.exp(-0.5 * (phase_primary / width) ** 2)
    secondary = secondary_depth * np.exp(-0.5 * (phase_secondary / (width * 1.2)) ** 2)
    return 1.0 - primary - secondary


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
            radius_ratio=np.random.uniform(0.04, 0.12),
            inclination_range=(86.0, 90.0),
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

    variability_scale = 1.6 if sample_class in {"stellar_variability", "eclipsing_binary"} else 1.0
    stellar_wobble = add_stellar_variability(time, amplitude_scale=variability_scale)
    noise = np.random.normal(0, np.random.uniform(0.0004, 0.0015), DATA_POINTS)
    noisy_flux = signal_flux + stellar_wobble + noise

    flattened_flux = flatten(time, noisy_flux, window_length=1.5, method="biweight")
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
        default=50,
        help="Print progress every N generated samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_dirs()
    os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)

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
