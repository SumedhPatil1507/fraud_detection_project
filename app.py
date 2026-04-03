import streamlit as st
import pandas as pd
import pickle
import os

from src.pipeline import run_pipeline
from src.model import train_model
from src.business import compute_business_cost
from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH
from src.plots import (
    plot_class_distribution,
    plot_amount_distribution,
    plot_correlation_heatmap,
    plot_roc_curve,
    plot_precision_recall,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_threshold_analysis,
)

st.set_page_config(page_title="Fraud Detection", layout="wide", page_icon="🚨")
st.title("🚨 Enterprise Fraud Detection System")

# ── Load & cache data ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def get_data():
    return run_pipeline()

df = get_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")
fn_cost = st.sidebar.number_input("False Negative Cost ($)", value=5000, step=500)
fp_cost = st.sidebar.number_input("False Positive Cost ($)", value=200, step=50)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_data, tab_train, tab_metrics, tab_predict = st.tabs([
    "📊 Data Explorer", "🏋️ Train Model", "📈 Model Metrics", "⚡ Real-time Prediction"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Dataset Overview")
    fraud_rate = df['label'].mean() * 100
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Fraud Cases", f"{df['label'].sum():,}")
    col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    col4.metric("Features", f"{df.shape[1] - 1}")

    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Visualizations")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_class_distribution(df))
    with c2:
        if 'transaction_amount' in df.columns:
            st.pyplot(plot_amount_distribution(df))

    st.pyplot(plot_correlation_heatmap(df))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.subheader("Train XGBoost Fraud Classifier")
    st.info("Training uses an auto-computed class weight based on the imbalance ratio.")

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training..."):
            model, X_test, y_test, probs, threshold, roc, report = train_model(df)
            st.session_state.update({
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "probs": probs,
                "threshold": threshold,
                "roc": roc,
                "report": report,
            })
        st.success(f"Model trained! ROC-AUC: **{roc:.4f}** | Best threshold: **{threshold:.3f}**")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    if "probs" not in st.session_state:
        st.info("Train the model first in the 'Train Model' tab.")
    else:
        y_test = st.session_state.y_test
        probs  = st.session_state.probs
        model  = st.session_state.model
        roc    = st.session_state.roc
        report = st.session_state.report

        threshold = st.slider("🎯 Decision Threshold", 0.01, 0.99,
                              float(st.session_state.threshold), 0.01)

        metrics = compute_business_cost(y_test, probs, threshold, fn_cost, fp_cost)

        st.subheader("Business Impact")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{roc:.4f}")
        c2.metric("Estimated Cost", f"${metrics['estimated_cost_usd']:,.0f}")
        c3.metric("Estimated Savings", f"${metrics['estimated_savings_usd']:,.0f}")
        c4.metric("Net Impact", f"${metrics['net_impact_usd']:,.0f}",
                  delta_color="normal")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("True Positives", metrics['true_positive'])
        c6.metric("False Positives", metrics['false_positive'])
        c7.metric("False Negatives", metrics['false_negative'])
        c8.metric("Precision / Recall",
                  f"{metrics['precision']:.2f} / {metrics['recall']:.2f}")

        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

        st.subheader("Plots")
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.pyplot(plot_roc_curve(y_test, probs))
        with r1c2:
            st.pyplot(plot_precision_recall(y_test, probs))

        st.pyplot(plot_confusion_matrix(y_test, probs, threshold))
        st.pyplot(plot_threshold_analysis(y_test, probs))

        feature_names = st.session_state.X_test.columns.tolist()
        st.pyplot(plot_feature_importance(model, feature_names))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REAL-TIME PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("⚡ Real-time Transaction Prediction")

    if not os.path.exists(MODEL_PATH):
        st.warning("No saved model found. Train and save a model first.")
    else:
        model   = pickle.load(open(MODEL_PATH, "rb"))
        features = pickle.load(open(FEATURE_PATH, "rb"))
        means   = pickle.load(open(MEAN_PATH, "rb"))

        pred_threshold = st.slider("Prediction Threshold", 0.01, 0.99, 0.30, 0.01,
                                   key="pred_thresh")

        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            amount   = c1.number_input("Transaction Amount ($)", min_value=0.0, value=500.0)
            distance = c2.number_input("Distance from Home (km)", min_value=0.0, value=10.0)
            hour     = c3.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=12)
            submitted = st.form_submit_button("🔍 Predict", type="primary")

        if submitted:
            input_dict = means.to_dict()
            input_dict["transaction_amount"] = amount
            input_dict["distance_from_home_km"] = distance
            if "hour" in input_dict:
                input_dict["hour"] = hour
            if "amount_log" in input_dict:
                import numpy as np
                input_dict["amount_log"] = np.log1p(amount)
            if "hour_sin" in input_dict:
                import numpy as np
                input_dict["hour_sin"] = np.sin(2 * np.pi * hour / 24)
                input_dict["hour_cos"] = np.cos(2 * np.pi * hour / 24)

            input_df = pd.DataFrame([input_dict])[features]
            prob = model.predict_proba(input_df)[0][1]
            pred = int(prob >= pred_threshold)

            st.metric("Fraud Probability", f"{prob:.2%}")
            if pred == 1:
                st.error("🚨 HIGH RISK — Likely Fraud")
            else:
                st.success("✅ LOW RISK — Likely Legitimate")
