# 🚨 Enterprise Fraud Detection System

An end-to-end ML fraud detection dashboard built with XGBoost and Streamlit.

## Live App

🔗 **[your-app-url.streamlit.app](https://your-app-url.streamlit.app)** ← replace with your actual URL

Deploy directly to [Streamlit Cloud](https://streamlit.io/cloud) — no extra config needed.

## Features

- **Data Explorer** — dataset overview, class distribution, amount distributions, correlation heatmap
- **Model Training** — XGBoost with auto-balanced class weights, best threshold via F1 maximization
- **Model Metrics** — ROC curve, Precision-Recall curve, Confusion Matrix, Feature Importance, Threshold Analysis
- **Business Impact** — configurable FN/FP costs, estimated savings vs cost, net impact
- **Real-time Prediction** — predict fraud probability for a single transaction

## Tech Stack

| Layer | Library |
|---|---|
| ML | XGBoost, scikit-learn |
| App | Streamlit |
| Data | pandas, numpy |
| Plots | matplotlib, seaborn |

## Project Structure

```
├── app.py                  # Streamlit app entry point
├── main.py                 # CLI training script
├── requirements.txt
├── runtime.txt             # Python 3.11
├── data/
│   └── enterprise_fraud_dataset.csv
├── sample_data.csv         # Fallback dataset
├── src/
│   ├── config.py           # Paths
│   ├── pipeline.py         # Load → preprocess → feature engineering
│   ├── model.py            # XGBoost training + artifact saving
│   ├── business.py         # Cost/savings metrics
│   └── plots.py            # All chart functions
└── outputs/
    └── model/              # Saved model artifacts (auto-created)
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

The app expects `data/enterprise_fraud_dataset.csv` (tab or comma separated).  
Falls back to `sample_data.csv` if the main file is unavailable.

Key columns: `transaction_amount`, `distance_from_home_km`, `hour`, `label` (0=legit, 1=fraud).
