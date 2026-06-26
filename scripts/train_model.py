import argparse
import os

import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Concatenate, Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Model

from pipeline_utils import GLOBAL_BINS, LOCAL_BINS, SYNTHETIC_DATA_DIR, SYNTHETIC_RESULTS_DIR, ensure_project_dirs

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Train the dual-view exoplanet classifier.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation for smoke tests.")
    return parser.parse_args()


def build_model() -> Model:
    global_input = Input(shape=(GLOBAL_BINS, 1), name="global_view")
    global_branch = Conv1D(filters=16, kernel_size=11, activation="relu", padding="same")(global_input)
    global_branch = MaxPooling1D(pool_size=5)(global_branch)
    global_branch = Conv1D(filters=32, kernel_size=7, activation="relu", padding="same")(global_branch)
    global_branch = MaxPooling1D(pool_size=5)(global_branch)
    global_branch = Flatten()(global_branch)

    local_input = Input(shape=(LOCAL_BINS, 1), name="local_view")
    local_branch = Conv1D(filters=16, kernel_size=5, activation="relu", padding="same")(local_input)
    local_branch = MaxPooling1D(pool_size=2)(local_branch)
    local_branch = Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(local_branch)
    local_branch = MaxPooling1D(pool_size=2)(local_branch)
    local_branch = Flatten()(local_branch)

    merged = Concatenate()([global_branch, local_branch])
    dense = Dense(128, activation="relu")(merged)
    dense = Dropout(0.35)(dense)
    dense = Dense(64, activation="relu")(dense)
    dense = Dropout(0.25)(dense)
    output = Dense(1, activation="sigmoid", name="planet_probability")(dense)

    model = Model(inputs=[global_input, local_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )
    return model


def main():
    args = parse_args()
    ensure_project_dirs()
    os.makedirs(SYNTHETIC_RESULTS_DIR, exist_ok=True)

    print("1. Loading Dataset...", flush=True)
    X_global = np.load(os.path.join(SYNTHETIC_DATA_DIR, "X_global.npy"))
    X_local = np.load(os.path.join(SYNTHETIC_DATA_DIR, "X_local.npy"))
    y = np.load(os.path.join(SYNTHETIC_DATA_DIR, "y_train.npy"))

    print(f"Total samples loaded: {len(y)}", flush=True)

    (
        X_global_train,
        X_global_val,
        X_local_train,
        X_local_val,
        y_train,
        y_val,
    ) = train_test_split(X_global, X_local, y, test_size=0.2, random_state=42, stratify=y)

    print("2. Building the True Dual-View 1D-CNN...", flush=True)
    model = build_model()
    model.summary()

    print("\n3. Training the Model...", flush=True)
    history = model.fit(
        {"global_view": X_global_train, "local_view": X_local_train},
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(
            {"global_view": X_global_val, "local_view": X_local_val},
            y_val,
        ),
    )

    print("\n4. Evaluating the Model...", flush=True)
    val_scores = model.predict({"global_view": X_global_val, "local_view": X_local_val}, verbose=0).ravel()
    val_predictions = (val_scores >= 0.5).astype(int)
    matrix = confusion_matrix(y_val, val_predictions)
    roc_auc = roc_auc_score(y_val, val_scores)

    print(f"Validation ROC-AUC: {roc_auc:.4f}", flush=True)
    print("Validation Confusion Matrix:", flush=True)
    print(matrix, flush=True)

    print("\n5. Saving Results...", flush=True)
    model_path = os.path.join(SYNTHETIC_RESULTS_DIR, "exoplanet_cnn_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}", flush=True)

    np.save(
        os.path.join(SYNTHETIC_RESULTS_DIR, "validation_predictions.npy"),
        np.column_stack((y_val, val_scores, val_predictions)),
    )

    if not args.skip_plots:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(history.history["accuracy"], label="Train Accuracy")
        axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
        axes[0].set_title("Model Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        axes[1].plot(history.history["loss"], label="Train Loss")
        axes[1].plot(history.history["val_loss"], label="Validation Loss")
        axes[1].set_title("Model Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        ConfusionMatrixDisplay(confusion_matrix=matrix).plot(ax=axes[2], colorbar=False)
        axes[2].set_title("Validation Confusion Matrix")

        plt.tight_layout()
        plot_path = os.path.join(SYNTHETIC_RESULTS_DIR, "training_history_and_confusion.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Training history plot saved to {plot_path}", flush=True)

    print("\n--- PIPELINE COMPLETE! ---", flush=True)


if __name__ == "__main__":
    main()
