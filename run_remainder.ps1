$env:PYTHONIOENCODING = "utf-8"
.\venv\Scripts\Activate.ps1

echo "4/6 Centroid Analysis: Kepler-10..."
python scripts/centroid_analysis.py --target Kepler-10 > logs\centroid_kepler10.log 2>&1
if ($LASTEXITCODE -ne 0) { echo "centroid_analysis.py Kepler-10 failed"; exit 1 }

echo "5/6 Centroid Analysis: KIC 6431670..."
python scripts/centroid_analysis.py --target "KIC 6431670" > logs\centroid_kic.log 2>&1
if ($LASTEXITCODE -ne 0) { echo "centroid_analysis.py KIC failed"; exit 1 }

echo "6/6 Running Benchmark Suite..."
python scripts/run_benchmark_suite.py > logs\run_benchmark.log 2>&1
if ($LASTEXITCODE -ne 0) { echo "run_benchmark_suite.py failed"; exit 1 }

echo "Pipeline completed successfully!"
