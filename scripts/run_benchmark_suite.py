import subprocess
import os

print("\n=======================================================")
print("EXOPLANET PIPELINE - MULTI-TARGET BENCHMARK SUITE")
print("=======================================================\n")

targets = [
    {"name": "Kepler-10", "label": "confirmed_planet (Rocky, 0.8 days)"},
    {"name": "Kepler-4", "label": "confirmed_planet (Neptune, 3.2 days)"},
    {"name": "Kepler-8", "label": "confirmed_planet (Hot Jupiter, 3.5 days)"},
    {"name": "Kepler-7", "label": "confirmed_planet (Hot Jupiter, 4.9 days)"},
    {"name": "Kepler-1", "label": "confirmed_planet (Hot Jupiter, 2.5 days)"},
    {"name": "Kepler-2", "label": "confirmed_planet (Hot Jupiter, 2.2 days)"},
    {"name": "Kepler-3", "label": "confirmed_planet (Neptune, 4.9 days)"},
    {"name": "Kepler-5", "label": "confirmed_planet (Hot Jupiter, 3.5 days)"},
    {"name": "Kepler-6", "label": "confirmed_planet (Hot Jupiter, 3.2 days)"},
]

for target in targets:
    name = target["name"]
    label = target["label"]
    print(f"--- RUNNING BENCHMARK ON: {name} ---")
    print(f"Target Type: {label}")
    
    command = [
        ".\\isroenv\\Scripts\\python.exe",
        "scripts/test_kepler.py",
        "--target", name,
        "--known-label", label,
        "--period-source", "searched"
    ]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print(f"ERROR: Pipeline crashed while processing {name}.")
    
    print("\n")

print("Benchmark suite complete! Check E:\\ISRO\\results\\benchmarks\\kepler_benchmark_results.csv for a summary.")
