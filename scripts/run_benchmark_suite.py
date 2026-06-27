import subprocess
import os

print("\n=======================================================")
print("EXOPLANET PIPELINE - MULTI-TARGET BENCHMARK SUITE")
print("=======================================================\n")

targets = [
    {"name": "Kepler-10", "label": "confirmed_planet (Rocky, Fast Orbit)"},
    {"name": "Kepler-4", "label": "confirmed_planet (Neptune-size, 3.2 days)"},
    {"name": "Kepler-8", "label": "confirmed_planet (Hot Jupiter, 3.5 days)"},
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
