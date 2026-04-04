import streamlit as st
import pandas as pd
import numpy as np
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
from src.shap_utils import plot_shap_summary, plot_shap_beeswarm, plot_waterfall

st.set_page_config(page_title="Fraud Detection", layout="wide", page_icon="🚨")
st.title("🚨 Enterprise Fraud Detection System")

@st.cache_data(show_spinner="Loading data...")
def get_data():
    return run_pipeline()

df = get_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")
fn_cost = st.sidebar.number_input("False Negative Cost ($)", value=5000, step=500)
fp_cost = st.sidebar.number_input("False Positive Cost ($)", value=200, step=50)
st.sidebar.markdown("---")
st.sidebar.markdown("**API**")
st.sidebar.code("uvicorn api:app --reload --port 8000", language="bash")
st.sidebar.markdown("[Swagger Docs → localhost:8000/docs](http://localhost:8000/docs)")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_data, tab_train, tab_metrics, tab_shap, tab_predict = st.tabs([
    "📊 Data Explorer", "🏋️ Train Model", "📈 Model Metrics",
    "🔍 SHAP Explainability", "⚡ Real-time Prediction"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Dataset Overview")
    fraud_rate = df['label'].mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Fraud Cases", f"{df['label'].sum():,}")
    c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    c4.metric("Features", f"{df.shape[1] - 1}")

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
    st.info("Uses auto-computed class weights + 5-fold stratified cross-validation.")

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training with cross-validation..."):
            model, X_test, y_test, probs, threshold, roc, report, cv_scores = train_model(df)
            st.session_state.update({
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "probs": probs,
                "threshold": threshold,
                "roc": roc,
                "report": report,
                "cv_scores": cv_scores,
            })

        st.success(f"Model trained! ROC-AUC: **{roc:.4f}** | Best threshold: **{threshold:.3f}**")

        st.subheader("5-Fold Cross-Validation ROC-AUC")
        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
            "ROC-AUC": cv_scores.round(4)
        })
        c1, c2 = st.columns([2, 1])
        with c1:
            st.bar_chart(cv_df.set_index("Fold"))
        with c2:
            st.metric("Mean CV AUC", f"{cv_scores.mean():.4f}")
            st.metric("Std CV AUC", f"± {cv_scores.std():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    if "probs" not in st.session_state:
        st.info("Train the model first.")
    else:
        y_test  = st.session_state.y_test
        probs   = st.session_state.probs
        model   = st.session_state.model
        roc     = st.session_state.roc
        report  = st.session_state.report

        threshold = st.slider("🎯 Decision Threshold", 0.01, 0.99,
                              float(st.session_state.threshold), 0.01)
        metrics = compute_business_cost(y_test, probs, threshold, fn_cost, fp_cost)

        st.subheader("Business Impact")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{roc:.4f}")
        c2.metric("Estimated Cost", f"${metrics['estimated_cost_usd']:,.0f}")
        c3.metric("Estimated Savings", f"${metrics['estimated_savings_usd']:,.0f}")
        c4.metric("Net Impact", f"${metrics['net_impact_usd']:,.0f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("True Positives", metrics['true_positive'])
        c6.metric("False Positives", metrics['false_positive'])
        c7.metric("False Negatives", metrics['false_negative'])
        c8.metric("Precision / Recall",
                  f"{metrics['precision']:.2f} / {metrics['recall']:.2f}")

        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        st.subheader("Plots")
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.pyplot(plot_roc_curve(y_test, probs))
        with r1c2:
            st.pyplot(plot_precision_recall(y_test, probs))

        st.pyplot(plot_confusion_matrix(y_test, probs, threshold))
        st.pyplot(plot_threshold_analysis(y_test, probs))
        st.pyplot(plot_feature_importance(model, st.session_state.X_test.columns.tolist()))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    if "model" not in st.session_state:
        st.info("Train the model first.")
    else:
        model  = st.session_state.model
        X_test = st.session_state.X_test

        st.subheader("Global Feature Importance (SHAP)")
        sample_size = min(300, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=42)

        with st.spinner("Computing SHAP values..."):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Mean |SHAP| — Bar**")
                st.pyplot(plot_shap_summary(model, X_sample))
            with c2:
                st.markdown("**Beeswarm — Direction & Magnitude**")
                st.pyplot(plot_shap_beeswarm(model, X_sample))

        st.subheader("Single Prediction Explanation")
        st.markdown("Adjust inputs below to see how each feature drives the fraud score.")

        with st.form("shap_form"):
            c1, c2, c3 = st.columns(3)
            s_amount   = c1.number_input("Amount ($)", value=1500.0)
            s_distance = c2.number_input("Distance (km)", value=300.0)
            s_hour     = c3.number_input("Hour", min_value=0, max_value=23, value=2)
            shap_submit = st.form_submit_button("Explain Prediction")

        if shap_submit:
            means = pickle.load(open(MEAN_PATH, "rb"))
            features = pickle.load(open(FEATURE_PATH, "rb"))
            input_dict = means.to_dict()
            input_dict["transaction_amount"] = s_amount
            input_dict["distance_from_home_km"] = s_distance
            input_dict["hour"] = s_hour
            input_dict["amount_log"] = np.log1p(s_amount)
            input_dict["hour_sin"] = np.sin(2 * np.pi * s_hour / 24)
            input_dict["hour_cos"] = np.cos(2 * np.pi * s_hour / 24)
            input_df = pd.DataFrame([input_dict])[features]

            prob = model.predict_proba(input_df)[0][1]
            st.metric("Fraud Probability", f"{prob:.2%}")
            with st.spinner("Generating waterfall..."):
                st.pyplot(plot_waterfall(model, input_df))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REAL-TIME PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("⚡ Real-time Transaction Prediction")

    if not os.path.exists(MODEL_PATH):
        st.warning("No saved model found. Train the model first.")
    else:
        model    = pickle.load(open(MODEL_PATH, "rb"))
        features = pickle.load(open(FEATURE_PATH, "rb"))
        means    = pickle.load(open(MEAN_PATH, "rb"))

        pred_threshold = st.slider("Prediction Threshold", 0.01, 0.99, 0.30, 0.01,
                                   key="pred_thresh")

        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            amount   = c1.number_input("Transaction Amount ($)", min_value=0.0, value=500.0)
            distance = c2.number_input("Distance from Home (km)", min_value=0.0, value=10.0)
            hour     = c3.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=12)
            c4, c5, c6 = st.columns(3)
            is_foreign  = c4.selectbox("Foreign Transaction", [0, 1])
            is_new_dev  = c5.selectbox("New Device", [0, 1])
            vpn         = c6.selectbox("VPN Detected", [0, 1])
            submitted = st.form_submit_button("🔍 Predict", type="primary")

        if submitted:
            input_dict = means.to_dict()
            input_dict.update({
                "transaction_amount": amount,
                "distance_from_home_km": distance,
                "hour": hour,
                "amount_log": np.log1p(amount),
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "is_foreign": is_foreign,
                "is_new_device": is_new_dev,
                "vpn_detected": vpn,
            })
            input_df = pd.DataFrame([input_dict])[features]
            prob = model.predict_proba(input_df)[0][1]
            pred = int(prob >= pred_threshold)

            st.metric("Fraud Probability", f"{prob:.2%}")
            if pred == 1:
                st.error("🚨 HIGH RISK — Likely Fraud")
            else:
                st.success("✅ LOW RISK — Likely Legitimate")

            st.markdown("**API equivalent call:**")
            st.code(f"""curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"transaction_amount": {amount}, "distance_from_home_km": {distance}, "hour": {hour}, "threshold": {pred_threshold}}}'""",
                    language="bash")
