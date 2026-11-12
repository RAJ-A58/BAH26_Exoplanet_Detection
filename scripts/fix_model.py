import zipfile
import json
import os
from pathlib import Path

MODEL_PATH = Path("E:/ISRO/results/synthetic/exoplanet_cnn_model.keras")
TEMP_PATH = Path("E:/ISRO/results/synthetic/temp_model.keras")

def strip_bad_keys(value):
    if isinstance(value, dict):
        value.pop("input_axes", None)
        value.pop("output_axes", None)
        value.pop("quantization_config", None)
        for nested in value.values():
            strip_bad_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            strip_bad_keys(nested)

def fix_model():
    print(f"Fixing {MODEL_PATH}...")
    with zipfile.ZipFile(MODEL_PATH, "r") as source, zipfile.ZipFile(TEMP_PATH, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "config.json":
                config = json.loads(data)
                strip_bad_keys(config)
                data = json.dumps(config).encode("utf-8")
            target.writestr(item, data)
    
    os.remove(MODEL_PATH)
    os.rename(TEMP_PATH, MODEL_PATH)
    print("Model fixed successfully! You can now run the benchmark.")

if __name__ == "__main__":
    if MODEL_PATH.exists():
        fix_model()
    else:
        print("Model file not found!")
