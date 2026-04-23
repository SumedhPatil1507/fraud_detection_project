# 🚨 Enterprise Fraud Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frauddetectionproject-ejx7okwuu6c8nrszvyzhhv.streamlit.app)
![CI](https://github.com/SumedhPatil1507/fraud_detection_project/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Live Demo →](https://frauddetectionproject-ejx7okwuu6c8nrszvyzhhv.streamlit.app)**

> End-to-end ML fraud detection — from raw transactions to real-time scoring, explainability, drift monitoring, and persistent audit logging.

---

## Dashboard

| Tab | What's inside |
|-----|--------------|
| 📊 Explorer | Interactive EDA — distributions, fraud-by-hour, channel analysis, velocity heatmap |
| 🏋️ Train | XGBoost + LightGBM ensemble, Optuna tuning, calibrated probabilities, 3-fold CV |
| 📈 Metrics | ROC, PR curve, confusion matrix, threshold analysis, business cost/savings |
| 🔍 Explainability | SHAP global bar, beeswarm, dependence plot, per-prediction waterfall |
| 📡 Drift | PSI-based feature drift with retraining recommendation |
| ⚡ Predict | Real-time scoring with fraud gauge chart, auto HITL queue |
| 🔴 Live Stream | Start/Stop live transaction feed with real-time Plotly chart |
| 👤 HITL | Human-in-the-loop analyst review queue |
| 🗂️ Audit | Supabase-backed prediction log with rolling fraud rate |

## Stack

`XGBoost` · `LightGBM` · `SHAP` · `Plotly` · `Streamlit` · `FastAPI` · `Supabase` · `scikit-learn` · `NetworkX` · `Docker`

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system diagram.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
uvicorn api:app --reload --port 8000   # API in separate terminal
```

## Docker

```bash
docker-compose up
# Streamlit → localhost:8501
# FastAPI   → localhost:8000
```

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: fraud-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"transaction_amount": 1500, "distance_from_home_km": 300, "hour": 2}'
```

Swagger docs at `localhost:8000/docs`

## Data

Upload CSV / Excel / JSON / Parquet via sidebar, or use the built-in dataset.
Required column: `label` (0 = legit, 1 = fraud).

## Secrets (Streamlit Cloud)

```toml
APP_PASSWORD = "your_password"
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your-anon-key"
```
