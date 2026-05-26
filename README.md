# 🚨 FraudGuard AI — Enterprise Edition v3.1

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frauddetectionproject-ejx7okwuu6c8nrszvyzhhv.streamlit.app)
![CI](https://github.com/SumedhPatil1507/fraud_detection_project/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Live Demo →](https://frauddetectionproject-ejx7okwuu6c8nrszvyzhhv.streamlit.app)**

Production-grade ML fraud detection — XGBoost/LightGBM ensemble with graph intelligence, async shadow mode, cryptographic tokenization, RBAC, SAR generation, and real-time savings tracking.

---

## Features

| Category | What's included |
|---|---|
| 🕸️ **Graph Intelligence** | Bipartite customer↔merchant graph (NetworkX), 3-strategy fraud ring detection (merchant hubs, connected components, shared devices), betweenness centrality features |
| ⚡ **Async Shadow Mode** | `asyncio.create_task()` parallel inference, `ThreadPoolExecutor` for CPU-bound scoring, `BackgroundTasks` logging — zero latency impact |
| 🔑 **Tokenization & Key Rotation** | HMAC-SHA256 format-preserving tokens, versioned `MultiFernet` key store, role-gated detokenization, Supabase `pgcrypto` SQL schema |
| 🔒 **Security** | RBAC (Admin/Analyst/Viewer), Fernet encryption at rest, per-role API rate limiting, data validation layer, PII masking |
| 🏦 **Financial Intelligence** | Dynamic threshold optimizer, fraud savings tracker, ROI projections, FP/FN cost modeling |
| 📋 **Compliance** | Automated FinCEN-style SAR generation, HITL analyst review queue, full audit log |
| 🚀 **MLOps** | PSI drift detection, automated retraining pipeline, model versioning, webhook notifications |
| 🧠 **Explainability** | SHAP global + per-prediction, LLM explanations via Groq (Llama-3), live transaction stream |

---

## Dashboard — 13 Tabs

`Explorer` · `Train` · `Metrics` · `Explainability` · `Drift` · `Predict` · `Live Stream` · `HITL` · `Audit` · `Graph Intel` · `Vault` · `Savings` · `SAR`

---

## Quick Start

```bash
git clone https://github.com/SumedhPatil1507/fraud_detection_project.git
cd fraud_detection_project
pip install -r requirements.txt
streamlit run app.py
```

API (separate terminal):
```bash
uvicorn api:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

Docker:
```bash
docker-compose up
# Streamlit → http://localhost:8501  |  API → http://localhost:8000
```

---

## API

All endpoints require `X-API-Key` header. Three role keys: `API_KEY_ADMIN`, `API_KEY_ANALYST`, `API_KEY_VIEWER`.

```bash
# Predict (analyst+) — async with parallel shadow scoring
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: analyst-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"transaction_amount": 1500, "distance_from_home_km": 300, "hour": 2}'

# Fraud rings (analyst+)
curl http://localhost:8000/graph/rings -H "X-API-Key: analyst-dev-key"

# Key rotation (admin only)
curl -X POST http://localhost:8000/vault/rotate -H "X-API-Key: admin-dev-key"

# SAR generation (analyst+)
curl -X POST http://localhost:8000/sar/generate \
  -H "X-API-Key: analyst-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"transaction_amount": 5000, "distance_from_home_km": 500}'
```

---

## Data Format

Upload CSV / Excel / JSON / Parquet. Required column: `label` (0 = legit, 1 = fraud).  
Graph features activate automatically when `customer_id` and `merchant_id` columns are present.

---

## Configuration

```bash
# Security
ENCRYPTION_KEY="your-fernet-key"
TOKEN_SECRET="your-hmac-secret"
API_KEY_ADMIN="admin-key"
API_KEY_ANALYST="analyst-key"
API_KEY_VIEWER="viewer-key"

# Optional
SUPABASE_URL="https://xxxx.supabase.co"
SUPABASE_KEY="your-anon-key"
GROQ_API_KEY="gsk_xxxx"
RETRAIN_WEBHOOK_URL="https://hooks.slack.com/xxx"
INSTITUTION_NAME="Your Bank Name"
```

Streamlit Cloud — add to **Settings → Secrets**:
```toml
SUPABASE_URL = "..."
SUPABASE_KEY = "..."
GROQ_API_KEY = "..."
```

---

## Stack

`XGBoost` · `LightGBM` · `SHAP` · `Optuna` · `NetworkX` · `FastAPI` · `asyncio` · `Streamlit` · `Plotly` · `cryptography` · `Supabase` · `Docker` · `pytest`

---

## Testing

```bash
pytest
pytest --cov=src --cov-report=html
```

---

## Architecture

```
Streamlit UI (13 tabs)
        │
FastAPI v3.1 (async endpoints, BackgroundTasks)
   ├── ML Engine: XGBoost + LightGBM + SHAP + Optuna
   ├── Graph Layer: NetworkX bipartite graph, ring detection
   └── Security: Fernet encryption, HMAC tokenization, RBAC
        │
Data Layer: Supabase · CSV fallback · Token Vault · SAR Reports
```

---

## License

MIT — see [LICENSE](LICENSE)

---

## Built by a College Student

Demonstrates senior ML engineer skills: graph intelligence, true async parallelism, cryptographic tokenization, FinCEN SAR compliance, production FastAPI, full-stack ML + security + DevOps.

**GitHub:** [@SumedhPatil1507](https://github.com/SumedhPatil1507) · **⭐ Star if useful!**
