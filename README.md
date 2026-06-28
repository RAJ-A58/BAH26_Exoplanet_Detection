# BAH26 Exoplanet Detection

Deep-learning exoplanet detection pipeline with a local browser frontend for running the trained dual-view CNN model.

## Start the Frontend

From the repo root:

```bash
cd /Users/parthagrawal99/BAH26_Exoplanet_Detection
PYTHONPYCACHEPREFIX=/private/tmp/exoplanet_pycache MPLCONFIGDIR=/private/tmp/mplcfg PORT=8000 ./.venv/bin/python app.py
```

Then open:

```text
http://127.0.0.1:8000/
```

If port `8000` is already in use, either stop the old Python server or change the port:

```bash
PORT=8001 ./.venv/bin/python app.py
```

Then open `http://127.0.0.1:8001/`.

## Frontend Features

- Neon mission-control interface with animated planet background.
- Runs a saved synthetic sample through the trained model.
- Uploads a CSV light curve with `time` and `flux` columns.
- Runs Kepler target inference through the BLS period-search path.
- Supports known ephemeris mode using period and transit epoch inputs.
- Shows prediction verdict, confidence score, period, `t0`, global folded view, local transit view, and per-hunt results.
- Handles the current Keras model compatibility issue by creating a temporary sanitized model archive in `/private/tmp` when needed.

Opening `frontend/index.html` directly will show the UI, but predictions require the Python server because `/api/predict` is served by `app.py`.
