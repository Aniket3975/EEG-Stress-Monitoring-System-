# EEG Stress Detection System

Real-time simulated EEG stress detection built with FastAPI and a lightweight HTML dashboard.

## Features

- `GET /predict` for simulated stress classification
- `POST /predict` for future EEG feature input support
- Color-coded dashboard that refreshes every 2 seconds
- Modular structure for swapping in feature extraction, ML inference, and WebSockets

## Project Structure

```text
app/
  api/routes/       # HTTP endpoints
  models/           # Stress inference service
  schemas/          # Request/response models
  services/         # EEG simulation and future integrations
  utils/            # Shared scoring logic
static/             # Frontend dashboard
```

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Future Extensions

- Add EEG feature extraction in `app/services/`
- Replace simulated scoring in `app/models/stress_model.py` with a trained RandomForest model
- Add a WebSocket endpoint for true streaming updates


### HOW TO RUN
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
