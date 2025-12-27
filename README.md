# AI-Driven Predictive Maintenance System for Industrial Machines

AI-powered machine failure prediction with 99.95% accuracy. Analyzes sensor data and tells you which machines need maintenance.

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

Pick GUI mode (2), click "Train Models", wait a minute. Done.

## Features

- **XGBoost ML model** - 99.95% accuracy on failure prediction
- **3 dashboards** - Model performance, production monitoring, real-time animated
- **WebSocket streaming** - Live sensor data and predictions
- **Multiple interfaces** - CLI menu or web UI, your choice
- **Centralized config** - Change everything in one file

## Usage

### First Time
```bash
python run.py
# Choose GUI (2)
# Click "Train Models"
# Wait ~1 minute
```

This trains the models and generates dashboards. Only needed once.

### Daily Use
```bash
python run.py
# Choose GUI (2)
# Click whatever you need
```

Models stay saved, no need to retrain.

## Project Structure

```
APP/
├── run.py              # Start here
├── src/                # Source code
│   ├── config.py       # All settings
│   ├── web_dashboard.py    # FastAPI server + WebSocket
│   └── predictive_maintenance.py  # ML pipeline
├── dashboards/         # HTML dashboards (generated)
├── models/             # Trained models (generated)
├── data/               # Sensor data
└── results/            # Prediction CSVs
```

## Dashboards

**Model Performance** - Accuracy metrics, confusion matrix, ROC curves

**Production Monitoring** - Risk distribution, high-risk machines, priorities

**Real-Time Monitor** - Live animated charts, sensor readings, failure probability

First two generate when you train. Third one is always there.

## Configuration

Edit `src/config.py`:

```python
SERVER = {"port": 8000}  # Change port
THRESHOLDS = {"high_risk_min": 0.7}  # Risk level cutoff
WEBSOCKET = {"update_interval": 1.0}  # Update speed
```

## API Endpoints

When server is running:

```
GET  /                      Main UI
POST /api/train             Train models  
GET  /api/results           Prediction data
GET  /model_performance     Model dashboard
GET  /production_dashboard  Operations dashboard
GET  /realtime_dashboard    Live monitor
WS   /ws/realtime           WebSocket stream
```

## Tech Stack

**Backend:** Python 3.11+, FastAPI, XGBoost, scikit-learn

**Frontend:** HTML/CSS/JS, Chart.js, Plotly

**Data:** Pandas, NumPy

## Troubleshooting

**ModuleNotFoundError**: `pip install -r requirements.txt`

**Port 8000 in use**: Kill existing process or change port in config

**Dashboard 404**: Train models first - they're generated files

**WebSocket won't connect**: Models need to be trained

## For Developers

**Add route:**
```python
# In src/web_dashboard.py
@app.get("/api/custom")
async def custom():
    return {"data": "whatever"}
```

**Modify ML pipeline:** Edit `src/predictive_maintenance.py`

**Deploy:** Use Gunicorn:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.web_dashboard:app
```

## Performance

- Training: ~5 seconds
- Prediction: <10ms
- Memory: ~200MB
- WebSocket latency: ~50ms
