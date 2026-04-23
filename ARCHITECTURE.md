# System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                           │
│  CSV / Excel / JSON / Parquet  ──►  src/ingest.py           │
│  Built-in dataset (sample_data.csv)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     ML Pipeline                             │
│  src/pipeline.py                                            │
│  ├── load_data()        — multi-format ingestion            │
│  ├── preprocess()       — dedup, encode categoricals        │
│  ├── feature_engineering() — log, cyclical, interactions    │
│  └── graph_features()   — NetworkX customer-merchant graph  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Model Layer                            │
│  src/model.py                                               │
│  ├── Optuna hyperparameter tuning (optional)                │
│  ├── XGBoost + LightGBM VotingClassifier                    │
│  ├── CalibratedClassifierCV (isotonic)                      │
│  ├── IsolationForest (anomaly detection)                    │
│  └── Model versioning (last 3 kept in outputs/model/)       │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌─────────────────────┐   ┌─────────────────────────────────┐
│   FastAPI (api.py)  │   │    Streamlit (app.py)           │
│                     │   │                                 │
│  POST /predict      │   │  📊 Explorer  — Plotly EDA      │
│  POST /predict/batch│   │  🏋️ Train    — CV + Optuna      │
│  GET  /model/info   │   │  📈 Metrics  — ROC, PR, CM      │
│  GET  /             │   │  🔍 SHAP     — Global + Local   │
│                     │   │  📡 Drift    — PSI detection    │
│  Auth: X-API-Key    │   │  ⚡ Predict  — Gauge chart      │
└─────────────────────┘   │  🔴 Live     — Real-time stream │
                          │  👤 HITL     — Review queue     │
                          │  🗂️ Audit    — Supabase log     │
                          │                                 │
                          │  Auth: Password gate            │
                          └────────────┬────────────────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │   Supabase (Postgres)  │
                          │   predictions table    │
                          │   — persistent audit   │
                          │   — rolling fraud rate │
                          └────────────────────────┘
```

## Key Design Decisions

- **VotingClassifier over Stacking** — faster on cloud, comparable accuracy for tabular fraud data
- **Isotonic calibration** — fraud systems need reliable probabilities, not just rankings
- **PSI drift detection** — industry standard for monitoring feature distribution shift
- **Supabase fallback to CSV** — app works without DB credentials, degrades gracefully
- **Feature alignment at inference** — all predictions use `{f: base.get(f, 0) for f in features}` to prevent feature mismatch errors
