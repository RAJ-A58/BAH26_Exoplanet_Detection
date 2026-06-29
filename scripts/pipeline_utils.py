import os
from typing import Dict, Tuple

import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "synthetic")
RAW_NASA_DATA_DIR = os.path.join(DATA_DIR, "raw_nasa")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SYNTHETIC_RESULTS_DIR = os.path.join(RESULTS_DIR, "synthetic")
BENCHMARK_RESULTS_DIR = os.path.join(RESULTS_DIR, "benchmarks")

GLOBAL_BINS = 2001
LOCAL_BINS = 201
LOCAL_VIEW_HALF_WIDTH = 0.12
STANDARDIZATION_EPS = 1e-6


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        SYNTHETIC_DATA_DIR,
        RAW_NASA_DATA_DIR,
        RESULTS_DIR,
        SYNTHETIC_RESULTS_DIR,
        BENCHMARK_RESULTS_DIR,
    ):
        os.makedirs(path, exist_ok=True)


def standardize_series(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    median = np.median(values)
    centered = values - median
    scale = np.std(centered)
    if scale < STANDARDIZATION_EPS:
        scale = STANDARDIZATION_EPS
    return centered / scale


def compute_phases(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    phases = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    return phases / period


def sort_by_phase(phases: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(phases)
    return phases[order], flux[order]


def bin_phased_flux(
    phases: np.ndarray,
    flux: np.ndarray,
    bins: int,
    phase_min: float = -0.5,
    phase_max: float = 0.5,
    default_value: float = 0.0,
) -> np.ndarray:
    bin_edges = np.linspace(phase_min, phase_max, bins + 1)
    clipped_phases = np.clip(phases, phase_min, phase_max - 1e-12)
    bin_indices = np.digitize(clipped_phases, bin_edges)

    binned_flux = np.zeros(bins, dtype=np.float32)
    empty_mask = np.ones(bins, dtype=bool)
    
    for idx in range(1, bins + 1):
        bucket = flux[bin_indices == idx]
        if len(bucket) > 0:
            binned_flux[idx - 1] = np.mean(bucket)
            empty_mask[idx - 1] = False
            
    # Interpolate empty bins across the phase fold so they don't form "W" spikes
    if np.any(empty_mask):
        valid_indices = np.where(~empty_mask)[0]
        empty_indices = np.where(empty_mask)[0]
        
        # If we have at least one valid point, interpolate linearly
        if len(valid_indices) > 0:
            binned_flux[empty_indices] = np.interp(
                empty_indices, valid_indices, binned_flux[valid_indices]
            )
        else:
            # Fallback if literally every single bin is empty
            binned_flux.fill(default_value)

    return binned_flux


def build_dual_views(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    global_bins: int = GLOBAL_BINS,
    local_bins: int = LOCAL_BINS,
    local_half_width: float = LOCAL_VIEW_HALF_WIDTH,
) -> Dict[str, np.ndarray]:
    # Standardize the entire light curve once so relative depths are preserved
    flux = standardize_series(flux)
    
    phases = compute_phases(time, period=period, t0=t0)
    phases, folded_flux = sort_by_phase(phases, flux)

    global_view = bin_phased_flux(phases, folded_flux, bins=global_bins)
    local_mask = (phases >= -local_half_width) & (phases <= local_half_width)
    local_phases = phases[local_mask]
    local_flux = folded_flux[local_mask]

    if len(local_phases) < max(10, local_bins // 8):
        local_phases = phases
        local_flux = folded_flux
        phase_min = -0.5
        phase_max = 0.5
    else:
        phase_min = -local_half_width
        phase_max = local_half_width

    local_view = bin_phased_flux(
        local_phases,
        local_flux,
        bins=local_bins,
        phase_min=phase_min,
        phase_max=phase_max,
    )

    return {
        "phases": phases,
        "folded_flux": folded_flux,
        "global_view": global_view,
        "local_view": local_view,
    }

