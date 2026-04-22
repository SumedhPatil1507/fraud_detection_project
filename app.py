import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

from src.pipeline import run_pipeline
from src.model import train_model
from src.business import compute_business_cost
from src.drift import detect_drift, plot_drift
from src.audit import log_prediction, load_audit_log
from src.simulator import generate_batch, score_transaction
from src.llm_explain import explain_prediction, get_top_shap_factors, chat_with_analyst
from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH
from src.plots import (
    plot_class_distribution, plot_amount_distribution,
    plot_correlation_heatmap, plot_roc_curve, plot_precision_recall,
    plot_confusion_matrix, plot_feature_importance, plot_threshold_analysis,
)
from src.shap_utils import plot_shap_summary, plot_shap_beeswarm, plot_waterfall

st.set_page_config(page_title="Fraud Detection", layout="wide", page_icon="🚨")
st.title("🚨 Enterprise Fraud Detection System")

# ── Data source: upload or default ────────────────────────────────────────────
st.sidebar.header("📂 Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (optional)", type=["csv"],
    help="Upload your own transaction dataset. Must have a 'label' column (0=legit, 1=fraud)."
)

@st.cache_data(show_spinner="Loading data...")
def get_data(file_key):
    return run_pipeline(uploaded_file if file_key == "uploaded" else None)

file_key = "uploaded" if uploaded_file is not None else "default"
df, data_source = get_data(file_key)

if data_source == "uploaded":
    st.sidebar.success("✅ Using uploaded dataset")
else:
    st.sidebar.info("📁 Using built-in dataset")

# ── Sidebar controls ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")
fn_cost = st.sidebar.number_input("False Negative Cost ($)", value=5000, step=500)
fp_cost = st.sidebar.number_input("False Positive Cost ($)", value=200, step=50)
use_optuna = st.sidebar.checkbox("Use Optuna Tuning", value=False,
                                  help="Slower but finds better params.")
n_trials = st.sidebar.slider("Optuna Trials", 5, 20, 10) if use_optuna else 10
st.sidebar.markdown("---")
st.sidebar.markdown("**API**")
st.sidebar.code("uvicorn api:app --reload --port 8000", language="bash")
st.sidebar.markdown("[Swagger Docs → localhost:8000/docs](http://localhost:8000/docs)")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Data Explorer", "🏋️ Train Model", "📈 Model Metrics",
    "🔍 SHAP Explainability", "📡 Drift Detection",
    "⚡ Real-time Prediction", "🔴 Live Simulation", "🤖 AI Analyst", "🗂️ Audit Log"
])
tab_data, tab_train, tab_metrics, tab_shap, tab_drift, tab_predict, tab_live, tab_ai, tab_audit = tabs

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
    st.caption(f"Data source: **{data_source}**")
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
    st.subheader("Train Ensemble Fraud Classifier")
    st.info("XGBoost + LightGBM soft-vote ensemble with calibrated probabilities and 3-fold CV.")

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training..."):
            result = train_model(df, use_optuna=use_optuna, n_trials=n_trials)
            model, X_test, y_test, probs, threshold, roc, report, cv_scores, anomaly_scores, best_params = result
            st.session_state.update({
                "model": model, "X_test": X_test, "y_test": y_test,
                "probs": probs, "threshold": threshold, "roc": roc,
                "report": report, "cv_scores": cv_scores,
                "anomaly_scores": anomaly_scores, "best_params": best_params,
            })
        st.success(f"Done! ROC-AUC: **{roc:.4f}** | Best threshold: **{threshold:.3f}**")
        st.json(best_params)
        cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
                               "ROC-AUC": cv_scores.round(4)})
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
        y_test = st.session_state.y_test
        probs  = st.session_state.probs
        model  = st.session_state.model
        roc    = st.session_state.roc
        report = st.session_state.report
        anomaly_scores = st.session_state.anomaly_scores

        threshold = st.slider("🎯 Decision Threshold", 0.01, 0.99,
                              float(st.session_state.threshold), 0.01)
        metrics = compute_business_cost(y_test, probs, threshold, fn_cost, fp_cost)

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

        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        anomaly_df = pd.DataFrame({"anomaly_score": anomaly_scores,
                                   "fraud_prob": probs, "actual": y_test.values})
        st.scatter_chart(anomaly_df, x="anomaly_score", y="fraud_prob", color="actual")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.pyplot(plot_roc_curve(y_test, probs))
        with r1c2:
            st.pyplot(plot_precision_recall(y_test, probs))
        st.pyplot(plot_confusion_matrix(y_test, probs, threshold))
        st.pyplot(plot_threshold_analysis(y_test, probs))

        base = st.session_state.model
        fi_model = (base.estimators_[0] if hasattr(base, 'estimators_') else base)
        if hasattr(fi_model, 'feature_importances_'):
            st.pyplot(plot_feature_importance(fi_model, st.session_state.X_test.columns.tolist()))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SHAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    if "model" not in st.session_state:
        st.info("Train the model first.")
    else:
        model  = st.session_state.model
        X_test = st.session_state.X_test
        xgb_model = (model.estimators_[0] if hasattr(model, 'estimators_') else model)

        X_sample = X_test.sample(min(100, len(X_test)), random_state=42)

        @st.cache_resource(show_spinner="Computing SHAP values...")
        def get_shap_plots(_m, _X):
            return plot_shap_summary(_m, _X), plot_shap_beeswarm(_m, _X)

        fig_bar, fig_bee = get_shap_plots(xgb_model, X_sample)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Mean |SHAP| — Bar**")
            st.pyplot(fig_bar)
        with c2:
            st.markdown("**Beeswarm**")
            st.pyplot(fig_bee)

        with st.form("shap_form"):
            c1, c2, c3 = st.columns(3)
            s_amount   = c1.number_input("Amount ($)", value=1500.0)
            s_distance = c2.number_input("Distance (km)", value=300.0)
            s_hour     = c3.number_input("Hour", min_value=0, max_value=23, value=2)
            shap_submit = st.form_submit_button("Explain Prediction")

        if shap_submit:
            means    = pickle.load(open(MEAN_PATH, "rb"))
            features = pickle.load(open(FEATURE_PATH, "rb"))
            input_dict = means.to_dict()
            input_dict.update({
                "transaction_amount": s_amount, "distance_from_home_km": s_distance,
                "hour": s_hour, "amount_log": np.log1p(s_amount),
                "hour_sin": np.sin(2 * np.pi * s_hour / 24),
                "hour_cos": np.cos(2 * np.pi * s_hour / 24),
            })
            input_df = pd.DataFrame([input_dict])[features]
            prob = model.predict_proba(input_df)[0][1]
            st.metric("Fraud Probability", f"{prob:.2%}")
            st.pyplot(plot_waterfall(xgb_model, input_df))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DRIFT
# ══════════════════════════════════════════════════════════════════════════════
with tab_drift:
    st.subheader("📡 Feature Drift Detection (PSI)")
    if st.button("Run Drift Analysis"):
        X_numeric = df.select_dtypes(include='number').drop(
            columns=['label', 'financial_loss'], errors='ignore')
        drift_result = detect_drift(X_numeric)
        if "error" in drift_result:
            st.warning(drift_result["error"])
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Overall PSI", f"{drift_result['overall_psi']:.4f}")
            c2.metric("Drift Level", drift_result["drift_level"])
            c3.metric("Retrain Recommended",
                      "✅ Yes" if drift_result["retrain_recommended"] else "❌ No")
            if drift_result["drift_detected"]:
                st.error("🚨 Significant drift detected.")
            elif drift_result["overall_psi"] > 0.1:
                st.warning("⚠️ Moderate drift — monitor closely.")
            else:
                st.success("✅ No significant drift.")
            fig = plot_drift(drift_result)
            if fig:
                st.pyplot(fig)
            st.dataframe(
                pd.DataFrame.from_dict(drift_result["top_drifted_features"],
                                       orient='index', columns=["PSI"])
                .style.background_gradient(cmap="RdYlGn_r"),
                use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REAL-TIME PREDICTION
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
            is_foreign = c4.selectbox("Foreign Transaction", [0, 1])
            is_new_dev = c5.selectbox("New Device", [0, 1])
            vpn        = c6.selectbox("VPN Detected", [0, 1])
            submitted  = st.form_submit_button("🔍 Predict", type="primary")

        if submitted:
            input_dict = means.to_dict()
            input_dict.update({
                "transaction_amount": amount, "distance_from_home_km": distance,
                "hour": hour, "amount_log": np.log1p(amount),
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "is_foreign": is_foreign, "is_new_device": is_new_dev,
                "vpn_detected": vpn,
            })
            input_df = pd.DataFrame([input_dict])[features]
            prob = model.predict_proba(input_df)[0][1]
            pred = int(prob >= pred_threshold)
            log_prediction(amount, distance, hour, prob, bool(pred), pred_threshold)
            st.metric("Fraud Probability", f"{prob:.2%}")
            if pred == 1:
                st.error("🚨 HIGH RISK — Likely Fraud")
            else:
                st.success("✅ LOW RISK — Likely Legitimate")

            # ── LLM Explanation ────────────────────────────────────────────
            st.subheader("🤖 AI Explanation")
            with st.spinner("Generating explanation..."):
                xgb_base = (model.estimators_[0]
                            if hasattr(model, 'estimators_') else model)
                shap_factors = get_top_shap_factors(
                    xgb_base, input_df, features)
                explanation = explain_prediction(
                    prob, {"transaction_amount": amount,
                           "distance_from_home_km": distance,
                           "hour": hour, "is_foreign": is_foreign,
                           "is_new_device": is_new_dev, "vpn_detected": vpn},
                    shap_factors, pred_threshold
                )
            st.info(explanation)

            st.code(
                f'curl -X POST http://localhost:8000/predict \\\n'
                f'  -H "Content-Type: application/json" \\\n'
                f'  -d \'{{"transaction_amount": {amount}, '
                f'"distance_from_home_km": {distance}, "hour": {hour}, '
                f'"threshold": {pred_threshold}}}\'', language="bash")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — LIVE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader("🔴 Live Transaction Stream Simulation")
    st.markdown("Simulates a real-time feed of incoming transactions scored by the model.")

    if not os.path.exists(MODEL_PATH):
        st.warning("Train the model first to enable live simulation.")
    else:
        model    = pickle.load(open(MODEL_PATH, "rb"))
        features = pickle.load(open(FEATURE_PATH, "rb"))
        means    = pickle.load(open(MEAN_PATH, "rb"))

        c1, c2, c3 = st.columns(3)
        batch_size   = c1.slider("Transactions per batch", 5, 30, 10)
        fraud_rate   = c2.slider("Simulated fraud rate", 0.05, 0.5, 0.2)
        sim_threshold = c3.slider("Alert threshold", 0.1, 0.9, 0.3, key="sim_thresh")

        if st.button("▶️ Generate Live Batch", type="primary"):
            seed = int(time.time()) % 100000
            txns = generate_batch(n=batch_size, fraud_rate=fraud_rate, seed=seed)
            scored = [score_transaction(t, model, features, means, sim_threshold)
                      for t in txns]
            results_df = pd.DataFrame(scored)

            fraud_count = results_df['predicted_fraud'].sum()
            avg_prob    = results_df['fraud_probability'].mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("Transactions Processed", batch_size)
            c2.metric("Flagged as Fraud", int(fraud_count))
            c3.metric("Avg Fraud Probability", f"{avg_prob:.2%}")

            # Colour-coded table
            display_cols = ['transaction_amount', 'distance_from_home_km', 'hour',
                            'is_foreign', 'vpn_detected', 'fraud_probability',
                            'risk_level', 'true_label', 'predicted_fraud']
            display_cols = [c for c in display_cols if c in results_df.columns]

            def highlight_row(row):
                if row.get('predicted_fraud', 0) == 1:
                    return ['background-color: #ffe0e0'] * len(row)
                return [''] * len(row)

            st.dataframe(
                results_df[display_cols].style.apply(highlight_row, axis=1),
                use_container_width=True
            )

            # Probability bar chart
            st.bar_chart(results_df['fraud_probability'])

            # Confusion for this batch
            if 'true_label' in results_df.columns:
                correct = (results_df['predicted_fraud'] == results_df['true_label']).mean()
                st.metric("Batch Accuracy", f"{correct:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — AI ANALYST CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
with tab_ai:
    st.subheader("🤖 AI Fraud Analyst")
    st.markdown("Ask questions about the model, metrics, or fraud patterns in plain English.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Build metrics context if model is trained
    metrics_context = {}
    if "roc" in st.session_state:
        metrics_context = {
            "ROC-AUC": round(st.session_state.roc, 4),
            "Best threshold": round(st.session_state.threshold, 3),
            "CV mean AUC": round(float(st.session_state.cv_scores.mean()), 4),
            "Fraud precision": round(st.session_state.report["Fraud"]["precision"], 3),
            "Fraud recall": round(st.session_state.report["Fraud"]["recall"], 3),
            "Dataset size": len(df),
            "Fraud rate": f"{df['label'].mean():.2%}",
        }

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    question = st.chat_input("Ask about the model or fraud patterns...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chat_with_analyst(question, metrics_context)
            st.write(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer})

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.caption("Powered by Groq (llama3-8b). Set GROQ_API_KEY in Streamlit secrets.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab_audit:
    st.subheader("🗂️ Prediction Audit Log")
    audit_df = load_audit_log()
    if audit_df.empty:
        st.info("No predictions logged yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", len(audit_df))
        c2.metric("Flagged as Fraud", int(audit_df['is_fraud'].sum()))
        c3.metric("Avg Fraud Probability", f"{audit_df['fraud_probability'].mean():.2%}")
        st.dataframe(audit_df.sort_values("timestamp", ascending=False),
                     use_container_width=True)
        if len(audit_df) > 1:
            audit_df['timestamp'] = pd.to_datetime(audit_df['timestamp'])
            st.line_chart(audit_df.set_index("timestamp")["fraud_probability"])
