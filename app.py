import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from src.pipeline import run_pipeline
from src.model import train_model
from src.business import compute_business_cost
from src.drift import detect_drift, plot_drift
from src.audit import log_prediction, load_audit_log
from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH
from src.plots import (
    plot_class_distribution, plot_amount_distribution,
    plot_correlation_heatmap, plot_roc_curve, plot_precision_recall,
    plot_confusion_matrix, plot_feature_importance, plot_threshold_analysis,
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
use_optuna = st.sidebar.checkbox("Use Optuna Tuning", value=True)
n_trials = st.sidebar.slider("Optuna Trials", 5, 30, 10) if use_optuna else 10
st.sidebar.markdown("---")
st.sidebar.markdown("**API**")
st.sidebar.code("uvicorn api:app --reload --port 8000", language="bash")
st.sidebar.markdown("[Swagger Docs → localhost:8000/docs](http://localhost:8000/docs)")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Data Explorer", "🏋️ Train Model", "📈 Model Metrics",
    "🔍 SHAP Explainability", "📡 Drift Detection",
    "⚡ Real-time Prediction", "🗂️ Audit Log"
])
tab_data, tab_train, tab_metrics, tab_shap, tab_drift, tab_predict, tab_audit = tabs

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
    st.subheader("Train Stacking Ensemble (XGBoost + LightGBM → Logistic Regression)")
    st.info("Optuna tunes XGBoost hyperparameters → stacking ensemble → calibrated probabilities → 5-fold CV.")

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Running Optuna + training ensemble..."):
            result = train_model(df, use_optuna=use_optuna, n_trials=n_trials)
            model, X_test, y_test, probs, threshold, roc, report, cv_scores, anomaly_scores, best_params = result
            st.session_state.update({
                "model": model, "X_test": X_test, "y_test": y_test,
                "probs": probs, "threshold": threshold, "roc": roc,
                "report": report, "cv_scores": cv_scores,
                "anomaly_scores": anomaly_scores, "best_params": best_params,
            })

        st.success(f"Done! ROC-AUC: **{roc:.4f}** | Best threshold: **{threshold:.3f}**")

        st.subheader("Best Hyperparameters (Optuna)")
        st.json(best_params)

        st.subheader("5-Fold Cross-Validation ROC-AUC")
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

        st.subheader("Anomaly Detection (Isolation Forest)")
        st.caption("High anomaly score = unusual transaction pattern, independent of the supervised model.")
        anomaly_df = pd.DataFrame({"anomaly_score": anomaly_scores,
                                   "fraud_prob": probs,
                                   "actual": y_test.values})
        st.scatter_chart(anomaly_df, x="anomaly_score", y="fraud_prob", color="actual")

        st.subheader("Plots")
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.pyplot(plot_roc_curve(y_test, probs))
        with r1c2:
            st.pyplot(plot_precision_recall(y_test, probs))
        st.pyplot(plot_confusion_matrix(y_test, probs, threshold))
        st.pyplot(plot_threshold_analysis(y_test, probs))
        st.pyplot(plot_feature_importance(
            st.session_state.model.estimators_[0][1]
            if hasattr(st.session_state.model, 'estimators_') else model,
            st.session_state.X_test.columns.tolist()
        ))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    if "model" not in st.session_state:
        st.info("Train the model first.")
    else:
        model  = st.session_state.model
        X_test = st.session_state.X_test

        # Use the XGBoost base estimator for SHAP (TreeExplainer requirement)
        xgb_model = (model.estimators_[0][1]
                     if hasattr(model, 'estimators_') else model)

        st.subheader("Global Feature Importance (SHAP)")
        X_sample = X_test.sample(min(100, len(X_test)), random_state=42)

        @st.cache_data(show_spinner="Computing SHAP values...")
        def get_shap_plots(_xgb_model, _X_sample):
            return plot_shap_summary(_xgb_model, _X_sample), plot_shap_beeswarm(_xgb_model, _X_sample)

        fig_bar, fig_bee = get_shap_plots(xgb_model, X_sample)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Mean |SHAP| — Bar**")
            st.pyplot(fig_bar)
        with c2:
            st.markdown("**Beeswarm — Direction & Magnitude**")
            st.pyplot(fig_bee)

        st.subheader("Single Prediction Explanation")
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
                "transaction_amount": s_amount,
                "distance_from_home_km": s_distance,
                "hour": s_hour,
                "amount_log": np.log1p(s_amount),
                "hour_sin": np.sin(2 * np.pi * s_hour / 24),
                "hour_cos": np.cos(2 * np.pi * s_hour / 24),
            })
            input_df = pd.DataFrame([input_dict])[features]
            prob = model.predict_proba(input_df)[0][1]
            st.metric("Fraud Probability", f"{prob:.2%}")
            with st.spinner("Generating waterfall..."):
                st.pyplot(plot_waterfall(xgb_model, input_df))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_drift:
    st.subheader("📡 Feature Drift Detection (PSI)")
    st.markdown("""
    Population Stability Index measures how much the feature distribution has shifted
    from training. PSI > 0.2 triggers a retraining recommendation.
    """)

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
                st.error("🚨 Significant drift detected — model performance may have degraded.")
            elif drift_result["overall_psi"] > 0.1:
                st.warning("⚠️ Moderate drift — monitor closely.")
            else:
                st.success("✅ No significant drift detected.")

            fig = plot_drift(drift_result)
            if fig:
                st.pyplot(fig)

            st.subheader("Per-Feature PSI")
            st.dataframe(
                pd.DataFrame.from_dict(drift_result["top_drifted_features"],
                                       orient='index', columns=["PSI"])
                .style.background_gradient(cmap="RdYlGn_r"),
                use_container_width=True
            )

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

            log_prediction(amount, distance, hour, prob, bool(pred), pred_threshold)

            st.metric("Fraud Probability", f"{prob:.2%}")
            if pred == 1:
                st.error("🚨 HIGH RISK — Likely Fraud")
            else:
                st.success("✅ LOW RISK — Likely Legitimate")

            st.markdown("**API equivalent:**")
            st.code(
                f'curl -X POST http://localhost:8000/predict \\\n'
                f'  -H "Content-Type: application/json" \\\n'
                f'  -d \'{{"transaction_amount": {amount}, '
                f'"distance_from_home_km": {distance}, '
                f'"hour": {hour}, "threshold": {pred_threshold}}}\'',
                language="bash"
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab_audit:
    st.subheader("🗂️ Prediction Audit Log")
    st.caption("Every prediction made in the Real-time tab is logged here.")

    audit_df = load_audit_log()
    if audit_df.empty:
        st.info("No predictions logged yet. Make a prediction in the Real-time tab.")
    else:
        total = len(audit_df)
        fraud_count = audit_df['is_fraud'].sum()
        avg_prob = audit_df['fraud_probability'].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", total)
        c2.metric("Flagged as Fraud", int(fraud_count))
        c3.metric("Avg Fraud Probability", f"{avg_prob:.2%}")

        st.dataframe(audit_df.sort_values("timestamp", ascending=False),
                     use_container_width=True)

        if len(audit_df) > 1:
            st.subheader("Fraud Probability Over Time")
            audit_df['timestamp'] = pd.to_datetime(audit_df['timestamp'])
            st.line_chart(audit_df.set_index("timestamp")["fraud_probability"])
