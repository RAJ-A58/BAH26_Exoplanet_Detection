import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate, Dropout
from sklearn.model_selection import train_test_split

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_DATA_DIR = os.path.join(BASE_DIR, "data", "synthetic")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "synthetic")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("1. Loading Dataset...")
X = np.load(os.path.join(SYNTHETIC_DATA_DIR, "X_train.npy"))
y = np.load(os.path.join(SYNTHETIC_DATA_DIR, "y_train.npy"))

print(f"Total samples loaded: {len(X)}")

# Split into Training (80%) and Validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Building the Dual-Branch 1D-CNN (AstroNet-inspired)...")

# Input layer (Shape: 201 bins, 1 feature/flux)
input_layer = Input(shape=(201, 1), name="LightCurve_Input")

# --- BRANCH A: Local View (Small Kernels) ---
# Designed to look closely at the sharp edges of the transit dip
branch_a = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(input_layer)
branch_a = MaxPooling1D(pool_size=2)(branch_a)
branch_a = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(branch_a)
branch_a = MaxPooling1D(pool_size=2)(branch_a)
branch_a = Flatten()(branch_a)

# --- BRANCH B: Global View (Large Kernels) ---
# Designed to look at the overall shape (U-shape vs V-shape)
branch_b = Conv1D(filters=16, kernel_size=11, activation='relu', padding='same')(input_layer)
branch_b = MaxPooling1D(pool_size=2)(branch_b)
branch_b = Conv1D(filters=32, kernel_size=11, activation='relu', padding='same')(branch_b)
branch_b = MaxPooling1D(pool_size=2)(branch_b)
branch_b = Flatten()(branch_b)

# --- Merge Branches & Output ---
merged = Concatenate()([branch_a, branch_b])
dense = Dense(64, activation='relu')(merged)
dense = Dropout(0.3)(dense) # Prevent overfitting
output_layer = Dense(1, activation='sigmoid', name="Planet_Probability")(dense)

# Compile Model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')]
)

model.summary()

print("\n3. Training the Model...")
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=15,          # Passes over the data
    batch_size=32,
    validation_data=(X_val, y_val)
)

print("\n4. Saving Results...")
# Save the trained model
model_path = os.path.join(RESULTS_DIR, "exoplanet_cnn_model.keras")
model.save(model_path)
print(f"Model saved to {model_path}")

# Plot Training History
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy Plot
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Loss Plot
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plot_path = os.path.join(RESULTS_DIR, "training_history.png")
plt.savefig(plot_path)
print(f"Training history plot saved to {plot_path}")

print("\n--- PIPELINE COMPLETE! ---")
