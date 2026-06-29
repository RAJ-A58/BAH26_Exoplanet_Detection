import argparse
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import lightkurve as lk
from astropy import units as u
from astropy.timeseries import BoxLeastSquares

# -- project paths --------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_NASA_DATA_DIR = os.path.join(BASE_DIR, "data", "raw_nasa")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "centroid")
os.makedirs(RAW_NASA_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -- constants ------------------------------------------------------------------
CENTROID_SHIFT_THRESHOLD_PX = 0.5   # pixels — Kepler pixel scale ~4 arcsec/px
TRANSIT_PHASE_HALF_WIDTH    = 0.05  # half-width in normalised phase units
N_QUARTERS                  = 4     # quarters to download


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_tpf_centroid(target: str):
    """
    Download Target Pixel File data for `target` and extract:
      - time array
      - flux array (summed aperture)
      - centroid_row array
      - centroid_col array

    Falls back gracefully to lightkurve's built-in centroid columns when
    pixel-level data is unavailable (e.g., long-cadence FITS headers).
    """
    print(f"  Searching for Target Pixel Files for {target}...")
    search = lk.search_targetpixelfile(target, author="Kepler", cadence="short")

    if len(search) == 0:
        raise ValueError(f"No TPF data found for {target}.  "
                         "Try a different target or cadence.")

    print(f"  Downloading {min(N_QUARTERS, len(search))} quarter(s)...")
    tpf_collection = search[:N_QUARTERS].download_all(
        download_dir=RAW_NASA_DATA_DIR, quality_bitmask="default"
    )

    # Stitch quarters into a single LightCurve with centroid columns
    lc_list = []
    for tpf in tpf_collection:
        lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
        # Attach centroid columns from the TPF
        lc["centroid_row"] = tpf.estimate_centroids(aperture_mask=tpf.pipeline_mask)[0]
        lc["centroid_col"] = tpf.estimate_centroids(aperture_mask=tpf.pipeline_mask)[1]
        lc_list.append(lc)

    stitched = lk.LightCurveCollection(lc_list).stitch()
    stitched = stitched.remove_nans().remove_outliers(sigma=5)

    time         = stitched.time.value
    flux         = stitched.flux.value
    centroid_row = np.array(stitched["centroid_row"].value, dtype=float)
    centroid_col = np.array(stitched["centroid_col"].value, dtype=float)

    # Remove NaN centroids
    valid = np.isfinite(centroid_row) & np.isfinite(centroid_col) & np.isfinite(flux)
    return time[valid], flux[valid], centroid_row[valid], centroid_col[valid]


# ══════════════════════════════════════════════════════════════════════════════
# PERIOD SEARCH  (same BLS as test_kepler.py for consistency)
# ══════════════════════════════════════════════════════════════════════════════

def bls_period_search(time: np.ndarray, flux: np.ndarray):
    print("  Running BLS period search...")
    durations = np.linspace(0.02, 0.15, 10) * u.day
    bls       = BoxLeastSquares(time * u.day, flux)
    periods   = bls.autoperiod(
        durations, minimum_period=0.5, maximum_period=15.0, frequency_factor=10.0
    )
    power     = bls.power(periods, durations, objective="snr")
    best_idx  = int(np.argmax(power.power))
    period    = float(power.period[best_idx].value)
    t0        = float(power.transit_time[best_idx].value)
    print(f"  BLS best period: {period:.6f} d   t0: {t0:.4f} d")
    return period, t0


# ══════════════════════════════════════════════════════════════════════════════
# CORE CENTROID ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_transit_mask(time: np.ndarray, period: float, t0: float,
                          half_width: float = TRANSIT_PHASE_HALF_WIDTH) -> np.ndarray:
    """Returns boolean array: True where cadence is IN-transit."""
    phase = ((time - t0 + 0.5 * period) % period) / period  # 0 → 1
    phase = phase - (phase > 0.5).astype(float)              # -0.5 → 0.5
    return np.abs(phase) < half_width


def centroid_offset_analysis(
    time: np.ndarray,
    flux: np.ndarray,
    centroid_row: np.ndarray,
    centroid_col: np.ndarray,
    period: float,
    t0: float,
) -> dict:
    """
    Core analysis.  Returns a results dict with:
      - in_transit / out_transit mean centroids
      - delta_row, delta_col, offset_magnitude (pixels)
      - is_blended  (bool flag)
      - pearson_row, pearson_col  (flux–centroid Pearson r, strong diagnostic)
    """
    in_mask  = compute_transit_mask(time, period, t0)
    out_mask = ~in_mask

    n_in  = int(in_mask.sum())
    n_out = int(out_mask.sum())

    if n_in < 5:
        raise ValueError(
            f"Fewer than 5 in-transit cadences found "
            f"(period={period:.3f} d, t0={t0:.3f} d).  "
            "Check ephemeris or download more quarters."
        )

    mean_row_in   = float(np.nanmedian(centroid_row[in_mask]))
    mean_col_in   = float(np.nanmedian(centroid_col[in_mask]))
    mean_row_out  = float(np.nanmedian(centroid_row[out_mask]))
    mean_col_out  = float(np.nanmedian(centroid_col[out_mask]))

    delta_row = mean_row_in  - mean_row_out
    delta_col = mean_col_in  - mean_col_out
    offset_px = float(np.sqrt(delta_row**2 + delta_col**2))

    # Pearson correlation: flux vs centroid row/col
    # A strong negative r means centroid shifts when flux dips → blending signal
    def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a - np.nanmean(a), b - np.nanmean(b)
        denom = np.nanstd(a) * np.nanstd(b)
        return float(np.nansum(a * b) / (len(a) * denom)) if denom > 1e-12 else 0.0

    r_row = pearson_r(flux, centroid_row)
    r_col = pearson_r(flux, centroid_col)

    is_blended = offset_px >= CENTROID_SHIFT_THRESHOLD_PX

    return {
        "period"         : period,
        "t0"             : t0,
        "n_in_transit"   : n_in,
        "n_out_transit"  : n_out,
        "mean_row_in"    : mean_row_in,
        "mean_col_in"    : mean_col_in,
        "mean_row_out"   : mean_row_out,
        "mean_col_out"   : mean_col_out,
        "delta_row_px"   : delta_row,
        "delta_col_px"   : delta_col,
        "offset_magnitude_px" : offset_px,
        "pearson_r_row"  : r_row,
        "pearson_r_col"  : r_col,
        "is_blended"     : is_blended,
        "threshold_px"   : CENTROID_SHIFT_THRESHOLD_PX,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_centroid_results(
    time: np.ndarray,
    flux: np.ndarray,
    centroid_row: np.ndarray,
    centroid_col: np.ndarray,
    results: dict,
    target: str,
    save_dir: str = RESULTS_DIR,
) -> str:
    in_mask  = compute_transit_mask(time, results["period"], results["t0"])
    out_mask = ~in_mask

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Centroid Offset Analysis — {target}\n"
        f"Period={results['period']:.4f} d   "
        f"Offset={results['offset_magnitude_px']:.4f} px   "
        f"{'⚠ BLENDED (False Positive Risk)' if results['is_blended'] else '✓ UNBLENDED (Genuine Transit)'}",
        fontsize=13, y=1.01
    )

    # -- panel 1: light curve with in/out coloring ---------------------------
    ax = axes[0, 0]
    ax.scatter(time[out_mask], flux[out_mask], s=1, c="#6b93c4", alpha=0.4, label="Out of transit")
    ax.scatter(time[in_mask],  flux[in_mask],  s=4, c="#c45c5c", alpha=0.9, label="In transit")
    ax.set_xlabel("Time (BKJD days)")
    ax.set_ylabel("Relative Flux")
    ax.set_title("Light Curve — transit cadences highlighted")
    ax.legend(fontsize=8)

    # -- panel 2: centroid row vs time ---------------------------------------
    ax = axes[0, 1]
    ax.scatter(time[out_mask], centroid_row[out_mask], s=1, c="#6b93c4", alpha=0.3)
    ax.scatter(time[in_mask],  centroid_row[in_mask],  s=4, c="#c45c5c", alpha=0.9)
    ax.axhline(results["mean_row_out"], color="#6b93c4", lw=1.5, ls="--", label=f"Out median {results['mean_row_out']:.4f}")
    ax.axhline(results["mean_row_in"],  color="#c45c5c", lw=1.5, ls="--", label=f"In median  {results['mean_row_in']:.4f}")
    ax.set_xlabel("Time (BKJD days)")
    ax.set_ylabel("Centroid Row (px)")
    ax.set_title(f"Centroid Row   Δ={results['delta_row_px']:.4f} px   r={results['pearson_r_row']:.3f}")
    ax.legend(fontsize=8)

    # -- panel 3: centroid col vs time ---------------------------------------
    ax = axes[1, 0]
    ax.scatter(time[out_mask], centroid_col[out_mask], s=1, c="#6b93c4", alpha=0.3)
    ax.scatter(time[in_mask],  centroid_col[in_mask],  s=4, c="#c45c5c", alpha=0.9)
    ax.axhline(results["mean_col_out"], color="#6b93c4", lw=1.5, ls="--", label=f"Out median {results['mean_col_out']:.4f}")
    ax.axhline(results["mean_col_in"],  color="#c45c5c", lw=1.5, ls="--", label=f"In median  {results['mean_col_in']:.4f}")
    ax.set_xlabel("Time (BKJD days)")
    ax.set_ylabel("Centroid Col (px)")
    ax.set_title(f"Centroid Col   Δ={results['delta_col_px']:.4f} px   r={results['pearson_r_col']:.3f}")
    ax.legend(fontsize=8)

    # -- panel 4: 2-D centroid scatter ---------------------------------------
    ax = axes[1, 1]
    ax.scatter(centroid_col[out_mask], centroid_row[out_mask], s=2, c="#6b93c4", alpha=0.3, label="Out")
    ax.scatter(centroid_col[in_mask],  centroid_row[in_mask],  s=8, c="#c45c5c", alpha=0.9, label="In")
    ax.plot(
        [results["mean_col_out"], results["mean_col_in"]],
        [results["mean_row_out"], results["mean_row_in"]],
        "k-o", lw=2, markersize=8, zorder=5,
        label=f"Offset = {results['offset_magnitude_px']:.4f} px"
    )
    color = "#c45c5c" if results["is_blended"] else "#4caf50"
    ax.set_xlabel("Centroid Col (px)")
    ax.set_ylabel("Centroid Row (px)")
    ax.set_title("2-D Centroid Map")
    ax.legend(fontsize=8)

    # threshold annotation
    verdict = ("BLENDED — possible background EB" if results["is_blended"]
               else "UNBLENDED — transit on target star")
    fig.text(
        0.5, -0.02, f"Verdict: {verdict}   (threshold = {CENTROID_SHIFT_THRESHOLD_PX} px)",
        ha="center", fontsize=11, color=color, fontweight="bold"
    )

    plt.tight_layout()
    fname = os.path.join(save_dir, f"{target.replace(' ', '_')}_centroid_analysis.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_centroid_analysis(target: str, period: float | None, t0: float | None) -> dict:
    """Full pipeline.  Returns results dict."""
    print(f"\n{'='*60}")
    print(f"CENTROID OFFSET ANALYSIS: {target}")
    print(f"{'='*60}")

    print("\n[1] Loading TPF centroid data from NASA...")
    time, flux, centroid_row, centroid_col = load_tpf_centroid(target)
    print(f"  {len(time):,} valid cadences loaded.")

    if period is None or t0 is None:
        print("\n[2] Period not provided — running BLS search...")
        # flatten first so BLS isn't confused by stellar trend
        lc_flat = lk.LightCurve(time=time, flux=flux).flatten(window_length=401)
        period, t0 = bls_period_search(lc_flat.time.value, lc_flat.flux.value)
    else:
        print(f"\n[2] Using provided ephemeris: period={period} d  t0={t0} d")

    print("\n[3] Running centroid offset analysis...")
    results = centroid_offset_analysis(time, flux, centroid_row, centroid_col, period, t0)

    print("\n[4] Generating diagnostic plot...")
    plot_path = plot_centroid_results(time, flux, centroid_row, centroid_col, results, target)

    # -- print summary --------------------------------------------------------
    print(f"\n{'-'*50}")
    print(f"  Target          : {target}")
    print(f"  Period          : {results['period']:.6f} d")
    print(f"  In-transit pts  : {results['n_in_transit']}")
    print(f"  Out-transit pts : {results['n_out_transit']}")
    print(f"  Δ Row           : {results['delta_row_px']:+.4f} px")
    print(f"  Δ Col           : {results['delta_col_px']:+.4f} px")
    print(f"  Offset mag.     : {results['offset_magnitude_px']:.4f} px  "
          f"(threshold = {CENTROID_SHIFT_THRESHOLD_PX} px)")
    print(f"  Pearson r (row) : {results['pearson_r_row']:.3f}")
    print(f"  Pearson r (col) : {results['pearson_r_col']:.3f}")
    verdict = ("⚠  BLENDED — likely contamination from background source"
               if results["is_blended"]
               else "✓  UNBLENDED — transit consistent with target star")
    print(f"\n  VERDICT: {verdict}")
    print(f"  Plot saved to: {plot_path}")
    print(f"{'-'*50}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Centroid offset analysis for exoplanet / EB discrimination."
    )
    parser.add_argument("--target",  default="Kepler-10", help="Kepler target name.")
    parser.add_argument("--period",  type=float, default=None,
                        help="Known orbital period in days (omit for BLS search).")
    parser.add_argument("--t0",      type=float, default=None,
                        help="Known transit epoch in BKJD days (omit for BLS search).")
    args = parser.parse_args()

    run_centroid_analysis(
        target=args.target,
        period=args.period,
        t0=args.t0,
    )


if __name__ == "__main__":
    main()
