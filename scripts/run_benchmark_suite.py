import argparse
import csv
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# -- cross-platform python executable ------------------------------------------
# Works on Windows (venv), macOS, Linux, Kaggle/Colab — no hard-coded paths.
PYTHON = sys.executable

# -- project structure ---------------------------------------------------------
BASE_DIR        = Path(__file__).resolve().parent.parent
SCRIPTS_DIR     = BASE_DIR / "scripts"
BENCHMARK_DIR   = BASE_DIR / "results" / "benchmarks"
BENCHMARK_CSV   = BENCHMARK_DIR / "kepler_benchmark_results.csv"
SUMMARY_TXT     = BENCHMARK_DIR / "benchmark_summary.txt"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK TARGETS
# ══════════════════════════════════════════════════════════════════════════════

# Each entry:
#   name         — lightkurve search string
#   label        — human tag written to the CSV
#   ground_truth — "planet" or "false_positive"
#   note         — why this target is in the suite

TARGETS = [
    # -- TRUE POSITIVES: single-planet systems ---------------------------------
    {
        "name": "Kepler-10",
        "label": "confirmed_planet (Rocky, 0.84 days)",
        "ground_truth": "planet",
        "note": "Hardest single-planet: tiny rocky dip, very short period",
    },
    {
        "name": "Kepler-4",
        "label": "confirmed_planet (Neptune-size, 3.2 days)",
        "ground_truth": "planet",
        "note": "Mid-size Neptune-class planet",
    },
    {
        "name": "Kepler-8",
        "label": "confirmed_planet (Hot Jupiter, 3.5 days)",
        "ground_truth": "planet",
        "note": "Classic Hot Jupiter with deep transit",
    },
    {
        "name": "Kepler-7",
        "label": "confirmed_planet (Hot Jupiter, 4.9 days)",
        "ground_truth": "planet",
        "note": "Hot Jupiter, well-studied benchmark",
    },
    {
        "name": "Kepler-1",
        "label": "confirmed_planet (Hot Jupiter, 2.5 days)",
        "ground_truth": "planet",
        "note": "TrES-2b, first Kepler confirmed planet",
    },
    {
        "name": "Kepler-2",
        "label": "confirmed_planet (Hot Jupiter, 2.2 days)",
        "ground_truth": "planet",
        "note": "HAT-P-7b, very deep transit, easy detection",
    },
    {
        "name": "Kepler-5",
        "label": "confirmed_planet (Hot Jupiter, 3.5 days)",
        "ground_truth": "planet",
        "note": "Deep Hot Jupiter",
    },
    {
        "name": "Kepler-6",
        "label": "confirmed_planet (Hot Jupiter, 3.2 days)",
        "ground_truth": "planet",
        "note": "Deep Hot Jupiter",
    },
    {
        "name": "Kepler-3",
        "label": "confirmed_planet (Neptune, 4.9 days)",
        "ground_truth": "planet",
        "note": "Neptune-size planet",
    },
    # -- TRUE POSITIVES: multi-planet systems ----------------------------------
    {
        "name": "Kepler-20",
        "label": "confirmed_planet (Multi-planet: 20c @ 10.85d, 20b @ 3.69d)",
        "ground_truth": "planet",
        "note": "Multi-planet: tests iterative pre-whitening",
    },
    {
        "name": "Kepler-62",
        "label": "confirmed_planet (Multi-planet, Super-Earth @ 5.71 days)",
        "ground_truth": "planet",
        "note": "Hardest: tiny super-Earth — tests model sensitivity",
    },
    # -- FALSE POSITIVES: eclipsing binaries -----------------------------------
    {
        "name": "KIC 6431670",
        "label": "eclipsing_binary (Kepler EB catalog, deep V-shaped dip)",
        "ground_truth": "false_positive",
        "note": "Confirmed EB from Kepler EB catalog — model must output < 0.5",
    },
    {
        "name": "KIC 3544595",
        "label": "eclipsing_binary (Kepler EB catalog, secondary eclipse visible)",
        "ground_truth": "false_positive",
        "note": "EB with visible secondary eclipse — strong EB signature",
    },
    # -- FALSE POSITIVES: quiet / transit-free stars ---------------------------
    {
        "name": "KIC 4914923",
        "label": "quiet_star (no known transit or EB signal)",
        "ground_truth": "false_positive",
        "note": "Solar-type quiet star, no reported transit or EB in literature",
    },
    {
        "name": "KIC 11295426",
        "label": "quiet_star (no known transit or EB signal)",
        "ground_truth": "false_positive",
        "note": "Quiet star, used as noise-floor control",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_target(target: dict, quarters: int = 4) -> dict | None:
    """
    Invokes test_kepler.py as a subprocess for one target.
    Returns the last CSV row written by that script, or None on crash.
    """
    name  = target["name"]
    label = target["label"]

    cmd = [
        PYTHON,
        str(SCRIPTS_DIR / "test_kepler.py"),
        "--target",       name,
        "--known-label",  label,
        "--period-source", "searched",
    ]

    print(f"\n{'-'*60}")
    print(f"  Target : {name}")
    print(f"  Type   : {label}")
    print(f"  Truth  : {target['ground_truth'].upper()}")
    print(f"  Note   : {target['note']}")
    print(f"{'-'*60}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"  ⚠  Pipeline crashed for {name}: {exc}")
        return None

    # Read back the last row that was just written to the CSV
    if not BENCHMARK_CSV.exists():
        return None
    with open(BENCHMARK_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1] if rows else None


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: list[dict]) -> dict:
    """
    results: list of {ground_truth, predicted_class, prediction_score, ...}
    Returns precision, recall, F1, accuracy, and lists for the confusion matrix.
    """
    tp = fp = tn = fn = 0
    for r in results:
        pred  = int(r["predicted_class"])
        truth = 1 if r["ground_truth"] == "planet" else 0
        if truth == 1 and pred == 1: tp += 1
        elif truth == 0 and pred == 1: fp += 1
        elif truth == 0 and pred == 0: tn += 1
        else:                          fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    accuracy  = (tp + tn) / len(results) if results else 0.0

    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def print_summary(results: list[dict], metrics: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  EXOPLANET BENCHMARK SUITE — FULL RESULTS")
    lines.append("=" * 70)

    col_w = [22, 14, 8, 8, 8]
    header = (f"{'Target':<{col_w[0]}} {'Label':<{col_w[1]}} "
              f"{'Truth':<{col_w[2]}} {'Pred':<{col_w[3]}} {'Score':<{col_w[4]}}")
    lines.append(header)
    lines.append("-" * 70)

    for r in results:
        truth_str = "PLANET" if r["ground_truth"] == "planet" else "FP"
        pred_str  = "PLANET" if int(r["predicted_class"]) == 1 else "NOISE"
        score     = float(r["prediction_score"])
        correct   = "✓" if (
            (r["ground_truth"] == "planet") == (int(r["predicted_class"]) == 1)
        ) else "✗"
        name = r.get("target", r.get("name", "?"))[:col_w[0]-1]
        lines.append(
            f"{correct} {name:<{col_w[0]-2}} {truth_str:<{col_w[1]}} "
            f"{pred_str:<{col_w[2]}} {score:.4f}"
        )

    lines.append("-" * 70)
    lines.append(
        f"\n  Confusion matrix\n"
        f"                  Predicted PLANET   Predicted NOISE\n"
        f"  True PLANET          {metrics['tp']:>4}                {metrics['fn']:>4}\n"
        f"  True FP/Noise        {metrics['fp']:>4}                {metrics['tn']:>4}\n"
    )
    lines.append(f"  Precision : {metrics['precision']:.3f}")
    lines.append(f"  Recall    : {metrics['recall']:.3f}")
    lines.append(f"  F1 Score  : {metrics['f1']:.3f}")
    lines.append(f"  Accuracy  : {metrics['accuracy']:.3f}")
    lines.append("=" * 70)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Full benchmark suite with FP targets.")
    parser.add_argument("--quarters", type=int, default=4,
                        help="Number of Kepler quarters to download per target (default 4).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print target list without running inference.")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  EXOPLANET PIPELINE — MULTI-TARGET BENCHMARK SUITE")
    print(f"  {len(TARGETS)} targets  "
          f"({sum(1 for t in TARGETS if t['ground_truth']=='planet')} planets, "
          f"{sum(1 for t in TARGETS if t['ground_truth']=='false_positive')} false positives)")
    print("=" * 70)

    if args.dry_run:
        for i, t in enumerate(TARGETS, 1):
            print(f"  [{i:02d}] {t['name']:<20} {t['ground_truth']:<14} {t['note']}")
        print("\n(dry-run: no inference executed)")
        return

    # Clear old CSV so we get a fresh run
    if BENCHMARK_CSV.exists():
        BENCHMARK_CSV.unlink()

    collected = []
    for target in TARGETS:
        row = run_target(target, quarters=args.quarters)
        if row is not None:
            row["ground_truth"] = target["ground_truth"]
            collected.append(row)
        else:
            # Record a crash entry so the target appears in the summary
            collected.append({
                "target"           : target["name"],
                "ground_truth"     : target["ground_truth"],
                "predicted_class"  : -1,
                "prediction_score" : -1.0,
            })

    valid_results = [r for r in collected if int(r["predicted_class"]) != -1]
    metrics = compute_metrics(valid_results)
    summary = print_summary(collected, metrics)

    print("\n" + summary)

    with open(SUMMARY_TXT, "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")

    print(f"\n  Summary saved to: {SUMMARY_TXT}")
    print(f"  Raw CSV saved to: {BENCHMARK_CSV}\n")


if __name__ == "__main__":
    main()
