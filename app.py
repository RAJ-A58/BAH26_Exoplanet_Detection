import csv
import io
import json
import os
import sys
import traceback
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplcfg")

import numpy as np
from astropy import units as u
from astropy.timeseries import BoxLeastSquares

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
SCRIPTS_DIR = BASE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_utils import SYNTHETIC_RESULTS_DIR, build_dual_views, ensure_project_dirs

MODEL_PATH = Path(SYNTHETIC_RESULTS_DIR) / "exoplanet_cnn_model.keras"
COMPAT_MODEL_PATH = Path("/private/tmp/exoplanet_cnn_model_compat.keras")
SYNTHETIC_GLOBAL_PATH = BASE_DIR / "data" / "synthetic" / "X_global.npy"
SYNTHETIC_LOCAL_PATH = BASE_DIR / "data" / "synthetic" / "X_local.npy"
SYNTHETIC_LABEL_PATH = BASE_DIR / "data" / "synthetic" / "y_train.npy"

MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        from tensorflow.keras.models import load_model

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        try:
            MODEL = load_model(MODEL_PATH, compile=False)
        except TypeError:
            MODEL = load_model(build_compat_model_archive(), compile=False)
    return MODEL


def build_compat_model_archive() -> Path:
    def strip_newer_config(value):
        if isinstance(value, dict):
            value.pop("quantization_config", None)
            for nested in value.values():
                strip_newer_config(nested)
        elif isinstance(value, list):
            for nested in value:
                strip_newer_config(nested)

    with zipfile.ZipFile(MODEL_PATH) as source, zipfile.ZipFile(COMPAT_MODEL_PATH, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "config.json":
                config = json.loads(data)
                strip_newer_config(config)
                data = json.dumps(config).encode("utf-8")
            target.writestr(item, data)

    return COMPAT_MODEL_PATH


def as_plot_points(values: np.ndarray, max_points: int = 420) -> List[float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) > max_points:
        indices = np.linspace(0, len(values) - 1, max_points).astype(int)
        values = values[indices]
    return [round(float(value), 6) for value in values]


def predict_views(global_view: np.ndarray, local_view: np.ndarray) -> float:
    model = get_model()
    global_input = np.expand_dims(np.asarray(global_view, dtype=np.float32), axis=(0, -1))
    local_input = np.expand_dims(np.asarray(local_view, dtype=np.float32), axis=(0, -1))
    prediction = model.predict({"global_view": global_input, "local_view": local_input}, verbose=0)[0][0]
    return float(prediction)


def search_period_with_bls(time: np.ndarray, flux: np.ndarray):
    durations = np.linspace(0.02, 0.15, 10) * u.day
    bls = BoxLeastSquares(time * u.day, flux)
    periods = bls.autoperiod(durations, minimum_period=0.5, maximum_period=15.0, frequency_factor=10.0)
    power = bls.power(periods, durations, objective="snr")
    best_idx = int(np.argmax(power.power))
    return float(power.period[best_idx].value), float(power.transit_time[best_idx].value), power


def load_real_lightcurve(target: str):
    import lightkurve as lk

    from pipeline_utils import RAW_NASA_DATA_DIR

    search_result = lk.search_lightcurve(target, author="Kepler", cadence="short")
    lc_collection = search_result[:4].download_all(download_dir=RAW_NASA_DATA_DIR)
    if lc_collection is None or len(lc_collection) == 0:
        raise ValueError(f"No short cadence data found for {target}.")

    lc = lc_collection.stitch()
    lc_clean = lc.remove_nans().remove_outliers(sigma=5)
    lc_flat = lc_clean.flatten(window_length=401)
    return lc_flat.time.value, lc_flat.flux.value


def auto_center_t0(time: np.ndarray, flux: np.ndarray, period: float, initial_t0: float) -> float:
    best_t0 = initial_t0
    min_center_flux = float("inf")

    for shift in np.linspace(-0.1 * period, 0.1 * period, 200):
        test_t0 = initial_t0 + shift
        views = build_dual_views(time, flux, period, test_t0)
        center_flux = np.mean(views["local_view"][95:106])
        if center_flux < min_center_flux:
            min_center_flux = center_flux
            best_t0 = test_t0

    return best_t0


def mask_transit(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float = 0.2) -> np.ndarray:
    masked_flux = flux.copy()
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    transit_mask = np.abs(phase) < (duration / 2)
    masked_flux[transit_mask] = 1.0
    return masked_flux


def synthetic_sample() -> Dict[str, Any]:
    if not SYNTHETIC_GLOBAL_PATH.exists() or not SYNTHETIC_LOCAL_PATH.exists():
        raise FileNotFoundError("Synthetic arrays are missing. Run scripts/generate_dataset.py first.")

    x_global = np.load(SYNTHETIC_GLOBAL_PATH, mmap_mode="r")
    x_local = np.load(SYNTHETIC_LOCAL_PATH, mmap_mode="r")
    labels = np.load(SYNTHETIC_LABEL_PATH, mmap_mode="r") if SYNTHETIC_LABEL_PATH.exists() else None

    index = int(np.random.default_rng().integers(0, len(x_global)))
    global_view = np.asarray(x_global[index]).reshape(-1)
    local_view = np.asarray(x_local[index]).reshape(-1)
    prediction = predict_views(global_view, local_view)
    known_label = int(labels[index]) if labels is not None else None

    return {
        "source": "Synthetic training sample",
        "iterations": [
            {
                "planetIndex": 1,
                "prediction": prediction,
                "predictedClass": int(prediction >= 0.5),
                "knownLabel": known_label,
                "period": None,
                "t0": None,
                "globalView": as_plot_points(global_view),
                "localView": as_plot_points(local_view, 201),
            }
        ],
    }


def parse_csv_lightcurve(csv_text: str) -> Tuple[np.ndarray, np.ndarray]:
    handle = io.StringIO(csv_text.strip())
    sample = handle.read(2048)
    handle.seek(0)
    dialect = csv.Sniffer().sniff(sample, delimiters=",\t; ")
    reader = csv.DictReader(handle, dialect=dialect)
    rows = list(reader)

    if not rows or reader.fieldnames is None:
        raise ValueError("CSV must include headers and at least one data row.")

    fields = {name.strip().lower(): name for name in reader.fieldnames if name}
    time_key = next((fields[key] for key in ("time", "time_days", "bkjd", "btjd", "jd") if key in fields), None)
    flux_key = next((fields[key] for key in ("flux", "pdcsap_flux", "sap_flux", "relative_flux") if key in fields), None)

    if time_key is None or flux_key is None:
        time_key, flux_key = reader.fieldnames[:2]

    time_values: List[float] = []
    flux_values: List[float] = []
    for row in rows:
        try:
            time_value = float(row[time_key])
            flux_value = float(row[flux_key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(time_value) and np.isfinite(flux_value):
            time_values.append(time_value)
            flux_values.append(flux_value)

    if len(time_values) < 50:
        raise ValueError("CSV needs at least 50 numeric time/flux rows.")

    return np.asarray(time_values, dtype=np.float64), np.asarray(flux_values, dtype=np.float64)


def infer_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    period_source: str,
    period: float,
    t0: float,
    max_iterations: int,
) -> List[Dict[str, Any]]:
    iterations = []
    current_flux = np.asarray(flux, dtype=np.float64).copy()

    for index in range(max_iterations):
        if period_source == "searched":
            found_period, initial_t0, _ = search_period_with_bls(time, current_flux)
            found_t0 = auto_center_t0(time, current_flux, found_period, initial_t0)
        else:
            found_period, found_t0 = period, t0

        views = build_dual_views(time, current_flux, period=found_period, t0=found_t0)
        prediction = predict_views(views["global_view"], views["local_view"])
        predicted_class = int(prediction >= 0.5)

        iterations.append(
            {
                "planetIndex": index + 1,
                "prediction": prediction,
                "predictedClass": predicted_class,
                "period": found_period,
                "t0": found_t0,
                "globalView": as_plot_points(views["global_view"]),
                "localView": as_plot_points(views["local_view"], 201),
                "foldedFlux": as_plot_points(views["folded_flux"], 500),
            }
        )

        current_flux = mask_transit(time, current_flux, found_period, found_t0, duration=0.4)
        if period_source == "known":
            break

    return iterations


def predict_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    ensure_project_dirs()
    mode = payload.get("mode", "synthetic")
    if mode == "synthetic":
        return synthetic_sample()

    period_source = payload.get("periodSource", "searched")
    period = float(payload.get("period") or 0.837495)
    t0 = float(payload.get("t0") or 120.8)
    max_iterations = max(1, min(5, int(payload.get("maxIterations") or 3)))

    if mode == "csv":
        time, flux = parse_csv_lightcurve(payload.get("csvText", ""))
        source = payload.get("fileName") or "Uploaded CSV"
    elif mode == "target":
        target = payload.get("target") or "Kepler-10"
        time, flux = load_real_lightcurve(target)
        source = f"NASA Kepler target: {target}"
    else:
        raise ValueError(f"Unsupported prediction mode: {mode}")

    return {
        "source": source,
        "periodSource": period_source,
        "iterations": infer_lightcurve(time, flux, period_source, period, t0, max_iterations),
    }


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8") or "{}")
            response = {"ok": True, "result": predict_payload(payload)}
            self.send_json(response)
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"ok": False, "error": str(exc)}, status=500)

    def send_json(self, payload: Dict[str, Any], status: int = 200):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main():
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), AppHandler)
    print(f"Exoplanet detector frontend running at http://0.0.0.0:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
