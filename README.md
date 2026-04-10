# 🚨 Enterprise Fraud Detection System

An elite-level end-to-end ML fraud detection system with stacking ensemble, Optuna tuning, SHAP explainability, drift detection, FastAPI serving, MLflow tracking, and a Streamlit dashboard.

## Live App

🔗 **[frauddetectionproject-ejx7okwuu6c8nrszvyzhhv.streamlit.app](https://frauddetectionproject-ejx7okwuu6c8nrszvyzhhv.streamlit.app)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frauddetectionproject-ejx7okwuu6c8nrszvyzhhv.streamlit.app)
![CI](https://github.com/sumedhpatil1507/frauddetectionproject/actions/workflows/ci.yml/badge.svg)

## Features

| Feature | Details |
|---|---|
| Stacking Ensemble | XGBoost + LightGBM → Logistic Regression meta-learner |
| Hyperparameter Tuning | Optuna with configurable trials, F1-optimal threshold |
| Calibrated Probabilities | Isotonic regression calibration via `CalibratedClassifierCV` |
| 5-Fold Cross-Validation | Stratified CV with mean ± std AUC reporting |
| Graph Features | NetworkX customer-merchant bipartite graph (degree, risk score) |
| Anomaly Detection | Isolation Forest as unsupervised second signal |
| SHAP Explainability | Global beeswarm/bar + per-prediction waterfall chart |
| Drift Detection | PSI-based feature drift with retraining recommendation |
| Business Impact | Configurable FN/FP costs, savings vs cost, net impact |
| FastAPI REST API | `/predict`, `/predict/batch`, `/model/info` with Swagger docs |
| Prediction Audit Log | Every inference logged to CSV with timestamp |
| MLflow Tracking | Params, metrics, and model artifacts per run |
| CI/CD | GitHub Actions pytest on every push |

## Tech Stack

| Layer | Library |
|---|---|
| ML | XGBoost, LightGBM, scikit-learn |
| Tuning | Optuna |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Graph Features | NetworkX |
| App | Streamlit |
| API | FastAPI + Uvicorn |
| Testing | pytest |
| Data | pandas, numpy |
| Plots | matplotlib, seaborn |

## Project Structure

```
├── app.py                  # Streamlit dashboard (7 tabs)
├── api.py                  # FastAPI prediction service
├── main.py                 # CLI training script
├── requirements.txt
├── runtime.txt
├── data/
│   └── enterprise_fraud_dataset.csv
├── sample_data.csv
├── src/
│   ├── config.py           # All paths
│   ├── pipeline.py         # Load → graph features → preprocess → feature engineering
│   ├── model.py            # Optuna + stacking ensemble + calibration + CV + MLflow
│   ├── business.py         # Cost/savings metrics
│   ├── plots.py            # Chart functions
│   ├── shap_utils.py       # SHAP summary, beeswarm, waterfall
│   ├── drift.py            # PSI-based drift detection
│   └── audit.py            # Prediction audit log
├── tests/
│   ├── test_pipeline.py
│   └── test_business.py
├── .github/workflows/
│   └── ci.yml
└── outputs/
    └── model/              # Saved artifacts (auto-created)
```

## Running Locally

```bash
pip install -r requirements.txt

# Streamlit dashboard
streamlit run app.py

# FastAPI server
uvicorn api:app --reload --port 8000

# Tests
pytest tests/ -v

# MLflow UI (after training)
mlflow ui
```

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_amount": 1500, "distance_from_home_km": 300, "hour": 2, "threshold": 0.3}'
```

Swagger docs: [localhost:8000/docs](http://localhost:8000/docs)

## Dataset

Expects `data/enterprise_fraud_dataset.csv`. Falls back to `sample_data.csv`.
Key columns: `transaction_amount`, `distance_from_home_km`, `hour`, `label` (0=legit, 1=fraud).
