import numpy as np
import matplotlib.pyplot as plt
import os
import batman
from wotan import flatten

# Define the root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "synthetic")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Simulating a 20-day observation of a star...")
time = np.linspace(0.0, 20.0, 2000)

# Create Fake Planet
params = batman.TransitParams()
params.t0 = 5.0                       
params.per = 7.0                      
params.rp = 0.1                       
params.a = 15.                        
params.inc = 90.                      
params.ecc = 0.                       
params.w = 90.                        
params.u = [0.1, 0.3]                 
params.limb_dark = "quadratic"        

m = batman.TransitModel(params, time)    
perfect_flux = m.light_curve(params)     

print("Injecting fake starspots and instrument noise...")
stellar_wobble = 0.005 * np.sin(2 * np.pi * time / 12.0) 
instrument_noise = np.random.normal(0, 0.002, len(time)) 

noisy_flux = perfect_flux + stellar_wobble + instrument_noise

print("Using WOTAN to flatten the light curve and remove the starspot wobble...")
flattened_flux, trend = flatten(
    time, 
    noisy_flux, 
    window_length=1.5, 
    method='biweight', 
    return_trend=True
)

print("Generating plot...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.scatter(time, noisy_flux, s=2, color='gray', label="Raw Noisy Data")
ax1.plot(time, trend, color='red', linewidth=2, label="Wotan Trendline (Starspot)")
ax1.set_ylabel("Relative Brightness")
ax1.set_title("1. Raw Data with Injected Planet, Noise, and Starspots")
ax1.legend(loc='lower right')

ax2.scatter(time, flattened_flux, s=2, color='blue', label="Flattened Data")
ax2.set_xlabel("Time (Days)")
ax2.set_ylabel("Relative Brightness")
ax2.set_title("2. Cleaned Data (Ready for AI)")
ax2.legend(loc='lower right')

plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, "synthetic_planet_wotan.png")
plt.savefig(save_path)
print(f"Success! Saved as {save_path}")
