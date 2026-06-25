import numpy as np
import os
import batman
from wotan import flatten
from tqdm import tqdm

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_DATA_DIR = os.path.join(BASE_DIR, "data", "synthetic")
os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)

# Configuration for our dataset
NUM_SAMPLES = 500         # Total light curves to generate (250 planets, 250 noise)
DAYS_OBSERVED = 27.0      # Standard observation window (similar to TESS)
DATA_POINTS = 2000        # Number of flux readings per curve
BINS = 201                # The fixed input size for our AI model

def generate_sample(has_planet):
    """Generates a single flattened, phase-folded, and binned light curve."""
    time = np.linspace(0, DAYS_OBSERVED, DATA_POINTS)
    
    # 1. Base Signal (Empty star)
    flux = np.ones(DATA_POINTS)
    
    # 2. Inject Planet if required
    period = 0
    t0 = 0
    if has_planet:
        params = batman.TransitParams()
        period = np.random.uniform(2.0, 10.0)    # Random period between 2-10 days
        t0 = np.random.uniform(1.0, period)      # Random start time
        params.t0 = t0
        params.per = period
        params.rp = np.random.uniform(0.05, 0.2) # Random planet size
        params.a = np.random.uniform(10., 20.)   # Random distance
        params.inc = np.random.uniform(85., 90.) # Random tilt (most transits are near 90)
        params.ecc = 0.
        params.w = 90.
        params.u = [0.1, 0.3]
        params.limb_dark = "quadratic"
        
        m = batman.TransitModel(params, time)
        flux = m.light_curve(params)
    else:
        # If no planet, we still need a "fake period" to fold the noise over 
        # so the AI learns what folded noise looks like.
        period = np.random.uniform(2.0, 10.0)
        t0 = np.random.uniform(1.0, period)
        
    # 3. Add Stellar Wobble & Noise
    stellar_wobble = np.random.uniform(0.001, 0.01) * np.sin(2 * np.pi * time / np.random.uniform(5, 15))
    noise = np.random.normal(0, 0.001, DATA_POINTS)
    noisy_flux = flux + stellar_wobble + noise
    
    # 4. Detrend with Wotan
    flattened_flux = flatten(time, noisy_flux, window_length=1.5, method='biweight')
    
    # 5. Phase Folding and Binning
    # We calculate the phase: a number from -0.5 to 0.5 where 0 is the transit dip
    phases = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    phases /= period # Normalize phase to [-0.5, 0.5]
    
    # Sort by phase
    sort_idx = np.argsort(phases)
    phases = phases[sort_idx]
    folded_flux = flattened_flux[sort_idx]
    
    # Bin the data into a fixed size (e.g., 201 bins) so our AI has a consistent input shape
    # We use numpy's histogram/digitize to group the phases into bins and take the mean flux of each bin
    bin_edges = np.linspace(-0.5, 0.5, BINS + 1)
    bin_indices = np.digitize(phases, bin_edges)
    
    binned_flux = np.zeros(BINS)
    for i in range(1, BINS + 1):
        points_in_bin = folded_flux[bin_indices == i]
        if len(points_in_bin) > 0:
            binned_flux[i-1] = np.mean(points_in_bin)
        else:
            binned_flux[i-1] = 1.0 # If no data in this bin, assume baseline flat flux
            
    return binned_flux

# Main Generation Loop
print(f"Generating {NUM_SAMPLES} light curves...")
X_data = []
y_labels = []

# To add a progress bar we'll use tqdm if it's installed, otherwise just a simple print
for i in range(NUM_SAMPLES):
    # 50% chance of being a planet (1) or just noise (0)
    is_planet = i % 2 == 0 
    
    try:
        binned_flux = generate_sample(has_planet=is_planet)
        X_data.append(binned_flux)
        y_labels.append(1 if is_planet else 0)
    except Exception as e:
        print(f"Skipping a sample due to error: {e}")
        continue
    
    if (i+1) % 50 == 0:
        print(f"Generated {i+1} / {NUM_SAMPLES} samples...")

# Convert to Numpy Arrays for Deep Learning
X_array = np.array(X_data)
y_array = np.array(y_labels)

# Expand dimensions so it works with 1D-CNNs (Shape: [Samples, 201, 1])
X_array = np.expand_dims(X_array, axis=-1)

# Save the dataset!
np.save(os.path.join(SYNTHETIC_DATA_DIR, "X_train.npy"), X_array)
np.save(os.path.join(SYNTHETIC_DATA_DIR, "y_train.npy"), y_array)

print("\n--- DATASET GENERATION COMPLETE ---")
print(f"X_train shape: {X_array.shape}")
print(f"y_train shape: {y_array.shape}")
print(f"Saved directly to: {SYNTHETIC_DATA_DIR}")
