# 🚨 Enterprise Fraud Detection System

An end-to-end ML fraud detection system with XGBoost, SHAP explainability, FastAPI serving, MLflow tracking, and a Streamlit dashboard.

## Live App

🔗 **[frauddetectionproject-ejx7okwuu6c8nrszvyzhhv.streamlit.app](https://frauddetectionproject-ejx7okwuu6c8nrszvyzhhv.streamlit.app)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frauddetectionproject-ejx7okwuu6c8nrszvyzhhv.streamlit.app)
![CI](https://github.com/sumedhpatil1507/frauddetectionproject/actions/workflows/ci.yml/badge.svg)

## Features

- **Data Explorer** — class distribution, amount distributions, correlation heatmap
- **Model Training** — XGBoost with auto-balanced class weights + 5-fold cross-validation
- **Model Metrics** — ROC-AUC, PR curve, confusion matrix, threshold analysis, feature importance
- **SHAP Explainability** — global beeswarm/bar plots + per-prediction waterfall chart
- **Business Impact** — configurable FN/FP costs, savings vs cost, net impact
- **Real-time Prediction** — single transaction scoring with API curl snippet
- **FastAPI REST API** — `/predict` and `/predict/batch` endpoints with Swagger docs
- **MLflow Tracking** — logs params, metrics, and model artifacts per training run
- **CI with GitHub Actions** — pytest runs on every push

## Tech Stack

| Layer | Library |
|---|---|
| ML | XGBoost, scikit-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| App | Streamlit |
| API | FastAPI + Uvicorn |
| Testing | pytest |
| Data | pandas, numpy |
| Plots | matplotlib, seaborn |

## Project Structure

```
├── app.py                  # Streamlit dashboard
├── api.py                  # FastAPI prediction service
├── main.py                 # CLI training script
├── requirements.txt
├── runtime.txt
├── data/
│   └── enterprise_fraud_dataset.csv
├── sample_data.csv
├── src/
│   ├── config.py
│   ├── pipeline.py         # Load → preprocess → feature engineering
│   ├── model.py            # XGBoost + CV + MLflow logging
│   ├── business.py         # Cost/savings metrics
│   ├── plots.py            # Chart functions (return fig objects)
│   └── shap_utils.py       # SHAP summary, beeswarm, waterfall
├── tests/
│   ├── test_pipeline.py
│   └── test_business.py
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI
└── outputs/
    └── model/              # Saved model artifacts (auto-created)
```

## Running Locally

```bash
pip install -r requirements.txt

# Streamlit dashboard
streamlit run app.py

# FastAPI server (separate terminal)
uvicorn api:app --reload --port 8000

# Run tests
pytest tests/ -v

# MLflow UI (after training)
mlflow ui
```

## API Usage

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_amount": 1500, "distance_from_home_km": 300, "hour": 2, "threshold": 0.3}'

# Swagger docs
open http://localhost:8000/docs
```

## Dataset

Expects `data/enterprise_fraud_dataset.csv`. Falls back to `sample_data.csv`.

Key columns: `transaction_amount`, `distance_from_home_km`, `hour`, `label` (0=legit, 1=fraud).
