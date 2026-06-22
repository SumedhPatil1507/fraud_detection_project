# 🚨 FraudGuard AI — Enterprise Edition v4.0

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frauddetectionproject-ejx7okwuu6c8nrszvyzhhv.streamlit.app)
![CI](https://github.com/SumedhPatil1507/fraud_detection_project/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Live Demo →](https://frauddetectionproject-ejx7okwuu6c8nrszvyzhhv.streamlit.app)**

Production-grade ML fraud detection platform. XGBoost/LightGBM ensemble with **Neo4j graph intelligence**, **async PostgreSQL + PgBouncer**, **Celery task queuing**, **S3 artifact storage**, **Prometheus observability**, and **DPDP Act 2023 / RBI IT Framework compliance mapping** — all with graceful fallbacks so it runs on Streamlit Cloud for free.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Streamlit UI  (15 tabs)  /  FastAPI v4.0        │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │       ML Engine         │
          │  XGBoost · LightGBM     │
          │  SHAP · Optuna · CV     │
          └────────────┬────────────┘
                       │
     ┌─────────────────┼─────────────────┐
     │                 │                 │
┌────▼────┐    ┌───────▼──────┐   ┌─────▼──────────┐
│  Graph  │    │  Data Layer  │   │   Security      │
│  Layer  │    │  asyncpg +   │   │  Fernet AES-128 │
│ Neo4j / │    │  PgBouncer → │   │  HMAC tokens    │
│ Neptune │    │  Supabase →  │   │  RBAC 3-tier    │
│ NetworkX│    │  CSV         │   │  Rate limiting  │
└─────────┘    └───────┬──────┘   └─────────────────┘
                       │
          ┌────────────┴────────────┐
          │    Background Services  │
          │  Celery + Redis         │
          │  S3 / MinIO artifacts   │
          │  Prometheus + Grafana   │
          └─────────────────────────┘
```

---

## Feature Matrix — 33 Modules

| Layer | Modules | Key capability |
|---|---|---|
| **ML** | `model.py`, `pipeline.py`, `shap_utils.py` | XGBoost+LightGBM ensemble, Optuna HPO, calibration |
| **Graph** | `graph_intelligence.py`, `graph_neo4j.py` | Neo4j/Neptune/NetworkX adapter, fraud ring detection (3 strategies), betweenness centrality |
| **Database** | `database.py`, `db_async.py` | SQLAlchemy asyncpg pool → PgBouncer → Supabase REST → CSV; MD5 record checksums |
| **Artifact Store** | `object_store.py` | S3/MinIO with MD5 integrity check; refuses corrupt model loads |
| **Task Queue** | `task_queue.py` | Celery + Redis; retrain, drift-check, SAR batch, beat schedule; sync fallback |
| **Observability** | `observability.py` | Prometheus p95/p99 latency, PSI gauge, TP/FP ratio, DB pool, queue depth; Grafana JSON |
| **Compliance** | `compliance.py` | DPDP Act 2023 (7 controls) + RBI IT Framework (7 controls); evidence + remediation; JSON report |
| **Security** | `encryption.py`, `rbac.py`, `tokenizer.py`, `pii.py`, `rate_limit.py` | Fernet encryption, RBAC, HMAC tokenization, key rotation, MultiFernet, pgcrypto SQL |
| **Shadow Mode** | `shadow.py` | `asyncio.create_task()` parallel inference, ThreadPoolExecutor, BackgroundTask logging |
| **MLOps** | `drift.py`, `retrain.py` | PSI drift detection, auto-retraining, model versioning, webhook notifications |
| **Finance** | `business.py`, `optimizer.py`, `savings_tracker.py` | Threshold optimizer, FP/FN cost model, ROI projections, savings accumulator |
| **Compliance/SAR** | `sar.py`, `audit.py` | FinCEN-style SAR generation, full prediction audit log |
| **UI** | `plots.py`, `hitl.py`, `llm_explain.py`, `simulator.py` | 30+ Plotly charts, HITL queue, Groq LLM explanations, live transaction stream |
| **API** | `validation.py`, `ingest.py`, `auth.py` | Pydantic validation, multi-format file ingestion, Streamlit auth |

---

## Dashboard — 15 Tabs

| # | Tab | Features |
|---|---|---|
| 1 | 📊 Explorer | EDA, distributions, fraud-by-hour, velocity heatmap, correlation matrix |
| 2 | 🏋️ Train | XGBoost + LightGBM, Optuna, 3-fold CV, hyperparameter tracking |
| 3 | 📈 Metrics | ROC/PR curves, confusion matrix, threshold analysis, cost/savings |
| 4 | 🔍 Explainability | SHAP global + per-prediction, LLM explanations (Groq) |
| 5 | 📡 Drift | PSI drift detection, retrain trigger, drift severity chart |
| 6 | ⚡ Predict | Real-time scoring, fraud gauge, validation warnings, auto-HITL |
| 7 | 🔴 Live Stream | Simulated transaction feed, live probability chart |
| 8 | 👤 HITL | Analyst review queue — confirm/reject fraud decisions |
| 9 | 🗂️ Audit | Full prediction log (Postgres → Supabase → CSV), rolling fraud rate |
| 10 | 🕸️ Graph Intel | Neo4j fraud ring detector, network viz, graph feature distributions |
| 11 | 🔑 Vault | Key rotation, tokenize/detokenize PII, pgcrypto SQL schema |
| 12 | 💰 Savings | Savings tracker, threshold optimizer chart, ROI calculator |
| 13 | 📋 SAR | FinCEN SAR management — review, file, dismiss |
| 14 | 📡 Observability | Live metrics (p95/p99), PSI chart, Celery queue, artifact registry, Grafana JSON |
| 15 | ⚖️ Compliance | DPDP Act 2023 + RBI IT Framework audit with score gauge |

---

## Quick Start

```bash
git clone https://github.com/SumedhPatil1507/fraud_detection_project.git
cd fraud_detection_project
pip install -r requirements.txt
streamlit run app.py
```

FastAPI (separate terminal):
```bash
uvicorn api:app --reload --port 8000
# Swagger → http://localhost:8000/docs
```

Full stack with Docker:
```bash
docker-compose up
# Streamlit   → http://localhost:8501
# FastAPI     → http://localhost:8000/docs
# Grafana     → http://localhost:3000  (admin/admin)
# MinIO       → http://localhost:9001  (minioadmin/minioadmin)
# Prometheus  → http://localhost:9090
```

---

## Deploy on Any Platform

| Platform | Config file | Start command |
|---|---|---|
| **Streamlit Cloud** | *(auto)* | Deploy `app.py` from GitHub |
| **Railway** | `railway.toml` | Auto-detected |
| **Render** | `render.yaml` | Auto-detected |
| **Heroku** | `Procfile` | `git push heroku main` |
| **Docker / VPS** | `docker-compose.yml` | `docker-compose up` |
| **Google Cloud Run** | `Dockerfile.streamlit` | `gcloud run deploy` |
| **AWS EC2** | `docker-compose.yml` | `docker-compose up -d` |
| **Azure App Service** | `Dockerfile.streamlit` | Set `WEBSITES_PORT=8501` |

### Streamlit Cloud (Free)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select this repo
3. Set main file: `app.py`
4. Add secrets under **Settings → Secrets**:

```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your-anon-key"
GROQ_API_KEY = "gsk_xxxx"
ENCRYPTION_KEY = "your-fernet-key"
TOKEN_SECRET = "your-hmac-secret"
```

### Railway (Paid, ~$5/mo)

```bash
npm install -g @railway/cli
railway login
railway init
railway up
```
Config in `railway.toml` — deploys automatically on push.

### Render (Free tier available)

Connect GitHub repo → Render detects `render.yaml` → deploys both Streamlit and API services.

### Heroku

```bash
heroku create fraudguard-ai
heroku config:set SUPABASE_URL=... GROQ_API_KEY=...
git push heroku main
```

---

## API Reference (v4.0)

All endpoints require `X-API-Key` header.

```bash
# Health + graph backend info
curl http://localhost:8000/

# Async predict (analyst+) — shadow scoring runs in parallel
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: analyst-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"transaction_amount": 1500, "distance_from_home_km": 300,
       "hour": 2, "customer_id": "C001", "merchant_id": "M042"}'

# Concurrent batch scoring (asyncio.gather)
curl -X POST http://localhost:8000/predict/batch \
  -H "X-API-Key: analyst-dev-key" \
  -d '{"transactions": [...]}'

# Fraud rings from Neo4j / NetworkX
curl http://localhost:8000/graph/rings -H "X-API-Key: analyst-dev-key"

# Trigger Celery retraining job
curl -X POST "http://localhost:8000/tasks/retrain?reason=drift" \
  -H "X-API-Key: admin-dev-key"

# Celery queue stats
curl http://localhost:8000/tasks/queue/stats -H "X-API-Key: admin-dev-key"

# Artifact registry (MD5 checksums)
curl http://localhost:8000/artifacts -H "X-API-Key: admin-dev-key"

# Key rotation
curl -X POST http://localhost:8000/vault/rotate -H "X-API-Key: admin-dev-key"

# DPDP + RBI compliance report
curl http://localhost:8000/compliance -H "X-API-Key: admin-dev-key"

# SAR generation
curl -X POST http://localhost:8000/sar/generate \
  -H "X-API-Key: analyst-dev-key" \
  -d '{"transaction_amount": 5000, "distance_from_home_km": 500}'
```

---

## Environment Variables

```bash
# ── Required in production ─────────────────────────────────────────────────────
ENCRYPTION_KEY="..."      # python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
TOKEN_SECRET="..."        # any strong random string
API_KEY_ADMIN="..."
API_KEY_ANALYST="..."
API_KEY_VIEWER="..."

# ── Database (asyncpg + PgBouncer) ─────────────────────────────────────────────
DATABASE_URL="postgresql+asyncpg://user:pass@pgbouncer:6432/fraudguard"

# ── Task queue ─────────────────────────────────────────────────────────────────
REDIS_URL="redis://localhost:6379/0"

# ── Artifact store ─────────────────────────────────────────────────────────────
AWS_S3_BUCKET="your-bucket"
# OR MinIO:
MINIO_ENDPOINT="http://minio:9000"
MINIO_BUCKET="fraudguard"
MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"

# ── Graph DB ───────────────────────────────────────────────────────────────────
NEO4J_URI="bolt://neo4j:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="password"
# OR Amazon Neptune:
NEPTUNE_ENDPOINT="your-cluster.neptune.amazonaws.com"

# ── Observability ──────────────────────────────────────────────────────────────
PROMETHEUS_PORT="9090"

# ── Optional ───────────────────────────────────────────────────────────────────
SUPABASE_URL="https://xxxx.supabase.co"
SUPABASE_KEY="your-anon-key"
GROQ_API_KEY="gsk_xxxx"
RETRAIN_WEBHOOK_URL="https://hooks.slack.com/xxx"
INSTITUTION_NAME="Your Bank Name"
```

> All infrastructure variables are **optional** — the app falls back gracefully:
> Neo4j → NetworkX · PostgreSQL → Supabase → CSV · S3 → local disk · Redis → sync mode

---

## Data Format

Upload CSV / Excel / JSON / Parquet via the sidebar. Required column: `label` (0 = legit, 1 = fraud).
Graph features auto-activate when `customer_id` and `merchant_id` columns are present.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **ML** | XGBoost · LightGBM · SHAP · scikit-learn · Optuna |
| **Graph** | Neo4j · Amazon Neptune · NetworkX (fallback) |
| **Database** | asyncpg · SQLAlchemy 2.0 · PgBouncer · PostgreSQL · Supabase |
| **Artifact Store** | AWS S3 · MinIO (S3-compatible) |
| **Task Queue** | Celery · Redis |
| **Observability** | Prometheus · Grafana |
| **Security** | Fernet · MultiFernet · HMAC-SHA256 · slowapi |
| **API** | FastAPI · asyncio · BackgroundTasks · Pydantic v2 |
| **Frontend** | Streamlit · Plotly · Seaborn |
| **LLM** | Groq (Llama-3) |
| **DevOps** | Docker Compose · GitHub Actions · pytest |

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

CI runs on every push: syntax check all 33 modules → pytest → import smoke test.

---

## License

MIT — see [LICENSE](LICENSE)

---

Built by **[@SumedhPatil1507](https://github.com/SumedhPatil1507)** — demonstrates senior ML engineering: async infrastructure, graph databases, cloud-native storage, regulatory compliance (DPDP/RBI), and production observability. **⭐ Star if useful!**
