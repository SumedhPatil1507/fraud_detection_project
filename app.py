import streamlit as st
import pandas as pd
import pickle
import requests
import shap
import matplotlib.pyplot as plt

from src.pipeline import run_pipeline
from src.model import train_model
from src.business import compute_business_cost
from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH

st.set_page_config(layout="wide")
st.title("🚨 Fraud Detection System")

df = run_pipeline()

# FILTERS
st.sidebar.header("Filters")

if 'channel' in df.columns:
    ch = st.sidebar.selectbox("Channel", ["All"] + list(df['channel'].unique()))
    if ch != "All":
        df = df[df['channel'] == ch]

st.dataframe(df.head())

# TRAIN MODEL
if st.button("Train Model"):
    model, X_test, y_test, probs, threshold = train_model(df)

    st.session_state.model = model
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.probs = probs

# THRESHOLD TUNING
if "probs" in st.session_state:
    threshold = st.slider("Threshold", 0.0, 1.0, 0.3)

    metrics = compute_business_cost(
        st.session_state.y_test,
        st.session_state.probs,
        threshold
    )

    st.json(metrics)

# SHAP
st.subheader("🔍 SHAP Explainability")

explainer = shap.TreeExplainer(st.session_state.model)
shap_values = explainer.shap_values(st.session_state.X_test[:200])

plt.figure()
shap.summary_plot(shap_values, st.session_state.X_test[:200], show=False)
st.pyplot(plt.gcf())  
plt.clf()

# REAL-TIME PREDICTION (API)
if st.button("Predict via API"):
    res = requests.post("http://127.0.0.1:8000/predict", json={
        "transaction_amount": 10000,
        "distance_from_home_km": 200
    })
    st.write(res.json())