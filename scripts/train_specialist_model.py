"""
train_specialist_model.py
─────────────────────────
Stage-2 specialist binary classifier for super-Earth / rocky planet detection.

Why a specialist model?
───────────────────────
The main ResNet (stage 1) is trained on all 5 classes with a shared loss. For
Hot Jupiters (transit depth 1–2%) it achieves near-perfect detection. For
super-Earths like Kepler-62b (transit depth ~90 ppm = 0.009%) the signal is
100x shallower — the model learns to underweight it to minimise cross-class
loss. The specialist is trained ONLY on:
  positive class: planet_small  (Rp < 2.5 R_Earth, depth < 500 ppm)
  negative class: noise + stellar_variability

This narrow focus lets it learn very fine sensitivity to shallow U-shaped dips.

Two-stage cascade logic
───────────────────────
  s1 = main_model.predict(views)

  if s1 >= 0.50:          # confident planet  -> use stage-1 directly
      return s1

  elif s1 >= LOW_THRESH:  # uncertain zone    -> consult specialist
      s2 = specialist.predict(views)
      return max(s1, s2)  # any positive evidence wins

  else:                   # confident noise   -> skip specialist
      return s1

  LOW_THRESH = 0.15  (catches Kepler-62 at 8.74% and any other low-confidence
                       signals that would be silently dropped by stage-1 alone)

Usage
─────
  python scripts/train_specialist_model.py
  python scripts/train_specialist_model.py --epochs 30 --samples 20000
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (
    Add, Conv1D, Concatenate, Dense, Dropout,
    GlobalAveragePooling1D, Input, MaxPooling1D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_utils import (
    GLOBAL_BINS, LOCAL_BINS,
    SYNTHETIC_DATA_DIR, SYNTHETIC_RESULTS_DIR,
    ensure_project_dirs,
)

SPECIALIST_MODEL_PATH = os.path.join(
    SYNTHETIC_RESULTS_DIR, "specialist_small_planet_model.keras"
)

# ── Cascade thresholds ─────────────────────────────────────────────────────────
LOW_THRESH  = 0.15   # below this: confident noise, skip specialist
HIGH_THRESH = 0.50   # above this: confident planet, skip specialist

# CLASS_CYCLE in generate_dataset.py:
#   0 = planet_small, 1 = planet_large, 2 = eclipsing_binary,
#   3 = stellar_variability, 4 = noise
SMALL_PLANET_CLASS  = 0
NOISE_CLASS         = 4
VARIABILITY_CLASS   = 3


# ══════════════════════════════════════════════════════════════════════════════
# DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_specialist_dataset(
    X_global: np.ndarray,
    X_local:  np.ndarray,
    y_binary: np.ndarray,
    y_class:  np.ndarray | None,
) -> tuple:
    """
    Filter the 30k synthetic dataset to the classes relevant to the specialist.
    Returns (Xg, Xl, y_spec, sample_weights) all shuffled.
    """
    if y_class is not None:
        pos_mask  = (y_class == SMALL_PLANET_CLASS)
        neg_mask  = np.isin(y_class, [NOISE_CLASS, VARIABILITY_CLASS])
        keep_mask = pos_mask | neg_mask

        Xg = X_global[keep_mask]
        Xl = X_local[keep_mask]
        y  = pos_mask[keep_mask].astype(np.float32)

        print(f"  Specialist dataset: {int(y.sum())} small-planet positives, "
              f"{int((1-y).sum())} noise/variability negatives")

        # Upsample positives to balance classes (shallow dips are rare)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if len(neg_idx) > len(pos_idx):
            extra = np.random.choice(
                pos_idx, size=len(neg_idx) - len(pos_idx), replace=True
            )
            Xg = np.concatenate([Xg, Xg[extra]])
            Xl = np.concatenate([Xl, Xl[extra]])
            y  = np.concatenate([y,  y[extra]])
            print(f"  After upsampling: {int(y.sum())} positives, "
                  f"{int((1-y).sum())} negatives")
    else:
        print("  WARNING: y_class.npy not found — using raw binary labels.")
        Xg = X_global
        Xl = X_local
        y  = y_binary.astype(np.float32)

    # Slightly higher sample weight for positives
    weights = np.where(y == 1, 2.0, 1.0)

    # Shuffle
    perm = np.random.permutation(len(y))
    return Xg[perm], Xl[perm], y[perm], weights[perm]


# ══════════════════════════════════════════════════════════════════════════════
# SPECIALIST ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def residual_block(x, filters: int, kernel_size: int, l2_reg: float = 1e-4):
    shortcut = Conv1D(filters, 1, padding="same")(x)
    x = Conv1D(filters, kernel_size, activation="relu",
               padding="same", kernel_regularizer=l2(l2_reg))(x)
    x = Conv1D(filters, kernel_size, activation="relu",
               padding="same", kernel_regularizer=l2(l2_reg))(x)
    x = Add()([shortcut, x])
    x = MaxPooling1D(pool_size=2)(x)
    return x


def build_specialist_model() -> tf.keras.Model:
    """
    Key differences from the main ResNet:
    - Local branch uses kernel_size=7 (vs 3 in main) to capture wider
      U-shaped dips characteristic of long-duration super-Earth transits
    - Extra dense layer (64) for finer shallow-signal discrimination
    - Higher dropout (0.5) to avoid overfitting on the smaller subset
    - Lower learning rate (2e-4) for careful fine-tuned convergence
    """
    # Global branch (same depth as main model)
    global_input = Input(shape=(GLOBAL_BINS, 1), name="global_view")
    xg = Conv1D(16, 5, activation="relu", padding="same")(global_input)
    xg = MaxPooling1D(pool_size=2)(xg)
    xg = residual_block(xg, 32, 5)
    xg = residual_block(xg, 64, 5)
    xg = residual_block(xg, 128, 5)
    xg = GlobalAveragePooling1D()(xg)

    # Local branch — wider kernels for shallow, long-duration super-Earth dips
    local_input = Input(shape=(LOCAL_BINS, 1), name="local_view")
    xl = Conv1D(16, 7, activation="relu", padding="same")(local_input)
    xl = MaxPooling1D(pool_size=2)(xl)
    xl = residual_block(xl, 32, 7)
    xl = residual_block(xl, 64, 7)
    xl = GlobalAveragePooling1D()(xl)

    merged = Concatenate()([xg, xl])

    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(merged)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.4)(x)
    x = Dense(64,  activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid", name="small_planet_probability")(x)

    model = tf.keras.Model(inputs=[global_input, local_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def train(args) -> tf.keras.Model:
    ensure_project_dirs()

    print("\n" + "=" * 60)
    print("  SPECIALIST SMALL-PLANET MODEL — TRAINING")
    print("=" * 60)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\n[1] Loading synthetic dataset...")
    X_global = np.load(os.path.join(SYNTHETIC_DATA_DIR, "X_global.npy"))
    X_local  = np.load(os.path.join(SYNTHETIC_DATA_DIR, "X_local.npy"))
    y_binary = np.load(os.path.join(SYNTHETIC_DATA_DIR, "y_train.npy"))

    class_path = os.path.join(SYNTHETIC_DATA_DIR, "y_class.npy")
    y_class = np.load(class_path) if os.path.exists(class_path) else None
    if y_class is None:
        print("  WARNING: y_class.npy not found — specialist will be less precise.")

    # Strip channel dim if it was added by train_model.py
    if X_global.ndim == 3: X_global = X_global[:, :, 0]
    if X_local.ndim  == 3: X_local  = X_local[:,  :, 0]

    # Optional subsample for fast debug runs
    if args.samples and args.samples < len(y_binary):
        idx = np.random.choice(len(y_binary), args.samples, replace=False)
        X_global = X_global[idx]
        X_local  = X_local[idx]
        y_binary = y_binary[idx]
        if y_class is not None:
            y_class = y_class[idx]
        print(f"  Subsampled to {args.samples} samples.")

    print(f"  Full dataset: {len(y_binary):,} samples")

    # ── Build specialist subset ───────────────────────────────────────────────
    print("\n[2] Building specialist dataset (small planets vs noise/variability)...")
    Xg, Xl, y, weights = build_specialist_dataset(X_global, X_local, y_binary, y_class)

    # Train/validation split (stratified)
    (Xg_train, Xg_val, Xl_train, Xl_val,
     y_train,  y_val,  w_train,  _) = train_test_split(
        Xg, Xl, y, weights, test_size=0.2, random_state=42, stratify=y
    )

    # Add channel dim for Conv1D
    Xg_train = np.expand_dims(Xg_train, -1)
    Xg_val   = np.expand_dims(Xg_val,   -1)
    Xl_train = np.expand_dims(Xl_train, -1)
    Xl_val   = np.expand_dims(Xl_val,   -1)

    print(f"\n  Train: {len(y_train):,} samples  ({int(y_train.sum())} positives)")
    print(f"  Val:   {len(y_val):,} samples  ({int(y_val.sum())} positives)")

    # ── Build and train model ─────────────────────────────────────────────────
    print("\n[3] Building specialist model...")
    model = build_specialist_model()
    model.summary(print_fn=lambda x: print(f"  {x}"))

    callbacks = [
        EarlyStopping(
            monitor="val_roc_auc", patience=6, mode="max",
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_roc_auc", patience=3, factor=0.5,
            mode="max", min_lr=1e-6, verbose=1
        ),
    ]

    print(f"\n[4] Training for up to {args.epochs} epochs...")
    history = model.fit(
        {"global_view": Xg_train, "local_view": Xl_train},
        y_train,
        sample_weight=w_train,
        validation_data=(
            {"global_view": Xg_val, "local_view": Xl_val},
            y_val,
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[5] Evaluating specialist model...")
    y_scores = model.predict(
        {"global_view": Xg_val, "local_view": Xl_val},
        batch_size=256, verbose=0
    ).ravel()
    roc_auc = roc_auc_score(y_val, y_scores)
    y_pred  = (y_scores >= 0.5).astype(int)

    print(f"\n  Specialist ROC-AUC : {roc_auc:.4f}")
    print(classification_report(y_val, y_pred,
                                target_names=["noise/variability", "small_planet"]))

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n[6] Saving specialist model to {SPECIALIST_MODEL_PATH}...")
    model.save(SPECIALIST_MODEL_PATH)
    print("  Done.")

    _plot_history(history, roc_auc)
    return model


def _plot_history(history, final_roc_auc: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Specialist Small-Planet Model Training   "
        f"(best val ROC-AUC = {final_roc_auc:.4f})",
        fontsize=12
    )
    axes[0].plot(history.history["loss"],     label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history.get("roc_auc",     []), label="train AUC")
    axes[1].plot(history.history.get("val_roc_auc", []), label="val AUC")
    axes[1].set_title("ROC-AUC")
    axes[1].legend()
    plt.tight_layout()
    out = os.path.join(SYNTHETIC_RESULTS_DIR, "specialist_training_history.png")
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Training history plot saved to {out}")


# ══════════════════════════════════════════════════════════════════════════════
# CASCADE PREDICT  — drop-in replacement for model.predict in test_kepler.py
# ══════════════════════════════════════════════════════════════════════════════

def cascade_predict(
    main_model:        tf.keras.Model,
    specialist_model:  tf.keras.Model,
    global_view:       np.ndarray,
    local_view:        np.ndarray,
    low_thresh:  float = LOW_THRESH,
    high_thresh: float = HIGH_THRESH,
    verbose:     bool  = True,
) -> tuple:
    """
    Two-stage cascade inference.

    Parameters
    ----------
    main_model       : trained ResNet (stage 1)
    specialist_model : specialist small-planet model (stage 2)
    global_view      : (GLOBAL_BINS,) array — NOT batched, NOT channel-expanded
    local_view       : (LOCAL_BINS,)  array — NOT batched, NOT channel-expanded

    Returns
    -------
    final_score : float in [0, 1]
    stage_used  : "stage1" | "stage2"
    """
    def _prep(arr: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.expand_dims(arr, 0), -1)   # (1, N, 1)

    inputs = {
        "global_view": _prep(global_view),
        "local_view":  _prep(local_view),
    }

    # Stage 1
    s1 = float(main_model.predict(inputs, verbose=0)[0][0])
    if verbose:
        print(f"  [cascade] Stage-1 score: {s1:.4f}")

    if s1 >= high_thresh:
        if verbose:
            print(f"  [cascade] Confident planet (>={high_thresh}) — skip specialist.")
        return s1, "stage1"

    # 1. Fast path — confident noise or confident planet
    if s1 < 0.05:
        if verbose: print(f"  [cascade] Confident noise (<0.05) — skip specialist.")
        return s1, "stage1"

    # Stage 2 — uncertain range [low_thresh, high_thresh)
    s2 = float(specialist_model.predict(inputs, verbose=0)[0][0])
    if verbose:
        print(f"  [cascade] Specialist score: {s2:.4f}")

    final = max(s1, s2)   # any positive evidence wins
    if verbose:
        print(
            f"  [cascade] Final score = max({s1:.4f}, {s2:.4f}) = {final:.4f}  "
            f"-> {'PLANET' if final >= 0.5 else 'NOT PLANET'}"
        )
    return final, "stage2"


def load_specialist(path: str = SPECIALIST_MODEL_PATH) -> tf.keras.Model:
    """Convenience loader — use in test_kepler.py."""
    return tf.keras.models.load_model(path)


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION PATCH FOR test_kepler.py
# ══════════════════════════════════════════════════════════════════════════════
#
# At the top of test_kepler.py, add:
#
#   from train_specialist_model import cascade_predict, load_specialist
#
# Right after loading the main model, add:
#
#   specialist_model = None
#   if os.path.exists(SPECIALIST_MODEL_PATH):
#       specialist_model = load_specialist(SPECIALIST_MODEL_PATH)
#       print("  Specialist model loaded for cascade inference.")
#
# Replace the model.predict line (~line 162):
#
#   # OLD:
#   prediction = float(model.predict(
#       {"global_view": global_input, "local_view": local_input}, verbose=0)[0][0])
#
#   # NEW:
#   if specialist_model is not None:
#       prediction, stage_used = cascade_predict(
#           model, specialist_model,
#           dual_views["global_view"], dual_views["local_view"],
#       )
#   else:
#       prediction = float(model.predict(
#           {"global_view": global_input, "local_view": local_input}, verbose=0)[0][0])
#
#   predicted_class = int(prediction >= 0.5)
#
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Train specialist small-planet model.")
    parser.add_argument("--samples",    type=int, default=None,
                        help="Subsample N rows from dataset (default: use all).")
    parser.add_argument("--epochs",     type=int, default=30,
                        help="Maximum training epochs (default: 30).")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size (default: 64).")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
