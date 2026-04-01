import streamlit as st
import pandas as pd
import pickle
import os

from src.pipeline import run_pipeline
from src.model import train_model
from src.business import compute_business_cost
from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH

st.set_page_config(layout="wide")

st.title("🚨 Fraud Detection System")

df = run_pipeline()

st.subheader("📊 Data Preview")
st.dataframe(df.head())

# =========================
# TRAIN MODEL
# =========================
if st.button("Train Model"):
    model, X_test, y_test, probs, threshold = train_model(df)

    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.probs = probs

    st.success("Model trained!")

# =========================
# THRESHOLD TUNING
# =========================
if "probs" in st.session_state:

    threshold = st.slider("🎯 Threshold", 0.0, 1.0, 0.3)

    metrics = compute_business_cost(
        st.session_state.y_test,
        st.session_state.probs,
        threshold
    )

    st.subheader("💰 Business Metrics")
    st.json(metrics)

# =========================
# REAL-TIME PREDICTION 
# =========================
if os.path.exists(MODEL_PATH):

    model = pickle.load(open(MODEL_PATH, "rb"))
    features = pickle.load(open(FEATURE_PATH, "rb"))
    means = pickle.load(open(MEAN_PATH, "rb"))

    st.subheader("⚡ Real-time Prediction")

    amount = st.number_input("Transaction Amount")
    distance = st.number_input("Distance from Home")

    if st.button("Predict Fraud"):

        input_dict = means.to_dict()
        input_dict["transaction_amount"] = amount
        input_dict["distance_from_home_km"] = distance

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[features]

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.write(f"Fraud Probability: {prob:.2f}")

        if pred == 1:
            st.error("🚨 Fraud")
        else:
            st.success("✅ Legitimate")