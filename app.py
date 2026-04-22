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
from src.pii import mask_pii
from src.hitl import add_to_review_queue, render_hitl_tab
from src.ingest import read_uploaded_file, validate_dataframe
from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH
from src.plots import (
    plot_class_distribution, plot_amount_distribution,
    plot_correlation_heatmap, plot_roc_curve, plot_precision_recall,
    plot_confusion_matrix, plot_feature_importance, plot_threshold_analysis,
)
from src.shap_utils import plot_shap_summary, plot_shap_beeswarm, plot_waterfall

st.set_page_config(page_title="Fraud Detection", layout="wide", page_icon="🚨")
st.title("🚨 Enterprise Fraud Detection System")

# ── Sidebar: Data Source ───────────────────────────────────────────────────────
st.sidebar.header("📂 Data Source")
data_mode = st.sidebar.radio("Mode", ["Built-in Dataset", "Upload File", "Live Simulation"])

df = None
data_source = "built-in"

if data_mode == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload file", type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Supports CSV, Excel, JSON, Parquet. Must have a 'label' column."
    )
    if uploaded_file:
        try:
            raw_df = read_uploaded_file(uploaded_file)
            valid, msg = validate_dataframe(raw_df)
            if valid:
                df, data_source = run_pipeline(uploaded_file=None, raw_df=raw_df)
                st.sidebar.success(f"✅ Loaded {len(df):,} rows")
            else:
                st.sidebar.error(f"❌ {msg}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

if df is None:
    @st.cache_data(show_spinner="Loading data...")
    def get_default_data():
        return run_pipeline()
    df, data_source = get_default_data()

mask_data = st.sidebar.checkbox("🔒 Mask PII", value=True)
display_df = mask_pii(df) if mask_data else df

# ── Sidebar: Controls ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")
fn_cost = st.sidebar.number_input("False Negative Cost ($)", value=5000, step=500)
fp_cost = st.sidebar.number_input("False Positive Cost ($)", value=200, step=50)
use_optuna = st.sidebar.checkbox("Use Optuna Tuning", value=False)
n_trials = st.sidebar.slider("Optuna Trials", 5, 20, 10) if use_optuna else 10
st.sidebar.markdown("---")
st.sidebar.code("uvicorn api:app --reload --port 8000", language="bash")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Data Explorer", "🏋️ Train Model", "📈 Model Metrics",
    "🔍 Explainability", "📡 Drift Detection", "⚡ Predict",
    "🔴 Live Simulation", "👤 HITL Review", "🤖 AI Analyst", "🗂️ Audit Log"
])
(tab_data, tab_train, tab_metrics, tab_shap, tab_drift,
 tab_predict, tab_live, tab_hitl, tab_ai, tab_audit) = tabs

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
    st.caption(f"Source: **{data_source}** | PII masking: {'on' if mask_data else 'off'}")
    st.dataframe(display_df.head(200), use_container_width=True)

    st.subheader("Visualizations")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_class_distribution(df))
    with c2:
        if 'transaction_amount' in df.columns:
            st.pyplot(plot_amount_distribution(df))
    st.pyplot(plot_correlation_heatmap(df))

    # Velocity Heatmap
    if 'hour' in df.columns and 'transaction_velocity_1h' in df.columns:
        st.subheader("Velocity Heatmap")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        pivot = df.groupby(['hour', 'label'])['transaction_velocity_1h'].mean().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 3))
        sns.heatmap(pivot.T, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f")
        ax.set_title("Avg Transaction Velocity by Hour & Class")
        ax.set_yticklabels(['Legit', 'Fraud'])
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.subheader("Train Ensemble Fraud Classifier")
    st.info("XGBoost + LightGBM soft-vote ensemble → calibrated probabilities → 3-fold CV.")
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training..."):
            result = train_model(df, use_optuna=use_optuna, n_trials=n_trials)
            model, X_test, y_test, probs, threshold, roc, report, cv_scores, anomaly_scores, best_params = result
            st.session_state.update({
                "model": model, "X_test": X_test, "y_test": y_test,
                "probs": probs, "threshold": threshold, "roc": roc,
                "report": report, "cv_scores": cv_scores,
                "anomaly_scores": anomaly_scores, "best_params": best_params,
                "trained_features": X_test.columns.tolist(),
            })
        st.success(f"Done! ROC-AUC: **{roc:.4f}** | Threshold: **{threshold:.3f}**")
        st.json(best_params)
        cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
                               "ROC-AUC": cv_scores.round(4)})
        c1, c2 = st.columns([2, 1])
        with c1:
            st.bar_chart(cv_df.set_index("Fold"))
        with c2:
            st.metric("Mean CV AUC", f"{cv_scores.mean():.4f}")
            st.metric("Std", f"± {cv_scores.std():.4f}")

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

        threshold = st.slider("🎯 Threshold", 0.01, 0.99,
                              float(st.session_state.threshold), 0.01)
        metrics = compute_business_cost(y_test, probs, threshold, fn_cost, fp_cost)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{roc:.4f}")
        c2.metric("Cost", f"${metrics['estimated_cost_usd']:,.0f}")
        c3.metric("Savings", f"${metrics['estimated_savings_usd']:,.0f}")
        c4.metric("Net Impact", f"${metrics['net_impact_usd']:,.0f}")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("TP", metrics['true_positive'])
        c6.metric("FP", metrics['false_positive'])
        c7.metric("FN", metrics['false_negative'])
        c8.metric("P / R", f"{metrics['precision']:.2f} / {metrics['recall']:.2f}")

        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        # Anomaly scatter
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
            st.pyplot(plot_feature_importance(fi_model,
                      st.session_state.X_test.columns.tolist()))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLAINABILITY (SHAP + LIME)
# ══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    if "model" not in st.session_state:
        st.info("Train the model first.")
    else:
        model  = st.session_state.model
        X_test = st.session_state.X_test
        xgb_model = (model.estimators_[0] if hasattr(model, 'estimators_') else model)
        X_sample = X_test.sample(min(100, len(X_test)), random_state=42)

        explain_type = st.radio("Explainer", ["SHAP", "LIME"], horizontal=True)

        if explain_type == "SHAP":
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
        else:
            st.info("LIME explanation for a single prediction — fill the form below.")

        st.subheader("Single Prediction Explanation")
        with st.form("shap_form"):
            c1, c2, c3 = st.columns(3)
            s_amount   = c1.number_input("Amount ($)", value=1500.0)
            s_distance = c2.number_input("Distance (km)", value=300.0)
            s_hour     = c3.number_input("Hour", min_value=0, max_value=23, value=2)
            shap_submit = st.form_submit_button("Explain")

        if shap_submit and os.path.exists(MEAN_PATH):
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
            # Align to trained features exactly
            input_df = pd.DataFrame([{f: input_dict.get(f, 0) for f in features}])
            prob = model.predict_proba(input_df)[0][1]
            st.metric("Fraud Probability", f"{prob:.2%}")

            if explain_type == "SHAP":
                st.pyplot(plot_waterfall(xgb_model, input_df))
            else:
                try:
                    import lime.lime_tabular
                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        X_test.values, feature_names=features,
                        class_names=['Legit', 'Fraud'], mode='classification'
                    )
                    exp = explainer.explain_instance(
                        input_df.values[0], model.predict_proba, num_features=10)
                    fig = exp.as_pyplot_figure()
                    st.pyplot(fig)
                except ImportError:
                    st.warning("Install `lime` for LIME explanations: `pip install lime`")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DRIFT DETECTION
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
            c3.metric("Retrain?", "✅ Yes" if drift_result["retrain_recommended"] else "❌ No")
            if drift_result["drift_detected"]:
                st.error("🚨 Significant drift — retrain recommended.")
            elif drift_result["overall_psi"] > 0.1:
                st.warning("⚠️ Moderate drift.")
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
# TAB 6 — REAL-TIME PREDICTION (feature-safe)
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("⚡ Real-time Transaction Prediction")
    if not os.path.exists(MODEL_PATH):
        st.warning("Train the model first.")
    else:
        model    = pickle.load(open(MODEL_PATH, "rb"))
        features = pickle.load(open(FEATURE_PATH, "rb"))
        means    = pickle.load(open(MEAN_PATH, "rb"))
        pred_threshold = st.slider("Threshold", 0.01, 0.99, 0.30, 0.01, key="pred_thresh")

        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            amount     = c1.number_input("Amount ($)", min_value=0.0, value=500.0)
            distance   = c2.number_input("Distance (km)", min_value=0.0, value=10.0)
            hour       = c3.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
            c4, c5, c6 = st.columns(3)
            is_foreign = c4.selectbox("Foreign", [0, 1])
            is_new_dev = c5.selectbox("New Device", [0, 1])
            vpn        = c6.selectbox("VPN", [0, 1])
            submitted  = st.form_submit_button("🔍 Predict", type="primary")

        if submitted:
            # Build input aligned EXACTLY to trained features
            base = means.to_dict()
            base.update({
                "transaction_amount": amount,
                "distance_from_home_km": distance,
                "hour": hour,
                "amount_log": np.log1p(amount),
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "is_foreign": is_foreign,
                "is_new_device": is_new_dev,
                "vpn_detected": vpn,
                "amount_vs_avg": amount / (base.get("avg_amount_30d", amount) + 1),
                "amount_x_distance": amount * distance,
            })
            # Align to exact feature list — fill missing with 0
            input_df = pd.DataFrame([{f: base.get(f, 0) for f in features}])

            prob = model.predict_proba(input_df)[0][1]
            pred = int(prob >= pred_threshold)
            log_prediction(amount, distance, hour, prob, bool(pred), pred_threshold)

            st.metric("Fraud Probability", f"{prob:.2%}")
            if pred == 1:
                st.error("🚨 HIGH RISK — Likely Fraud")
                add_to_review_queue(
                    {"transaction_amount": amount, "distance_from_home_km": distance,
                     "hour": hour}, prob, "Auto-flagged by model")
            else:
                st.success("✅ LOW RISK — Likely Legitimate")

            # LLM Explanation
            st.subheader("🤖 AI Explanation")
            with st.spinner("Generating..."):
                xgb_base = (model.estimators_[0] if hasattr(model, 'estimators_') else model)
                shap_factors = get_top_shap_factors(xgb_base, input_df, features)
                explanation = explain_prediction(
                    prob, {"transaction_amount": amount,
                           "distance_from_home_km": distance, "hour": hour,
                           "is_foreign": is_foreign, "is_new_device": is_new_dev,
                           "vpn_detected": vpn},
                    shap_factors, pred_threshold)
            st.info(explanation)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — LIVE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader("🔴 Live Transaction Stream")
    if not os.path.exists(MODEL_PATH):
        st.warning("Train the model first.")
    else:
        model    = pickle.load(open(MODEL_PATH, "rb"))
        features = pickle.load(open(FEATURE_PATH, "rb"))
        means    = pickle.load(open(MEAN_PATH, "rb"))

        c1, c2, c3 = st.columns(3)
        batch_size    = c1.slider("Batch size", 5, 30, 10)
        sim_fraud_rate = c2.slider("Fraud rate", 0.05, 0.5, 0.2)
        sim_threshold  = c3.slider("Alert threshold", 0.1, 0.9, 0.3, key="sim_thresh")

        if st.button("▶️ Generate Batch", type="primary"):
            seed = int(time.time()) % 100000
            txns = generate_batch(n=batch_size, fraud_rate=sim_fraud_rate, seed=seed)
            scored = [score_transaction(t, model, features, means, sim_threshold)
                      for t in txns]
            results_df = pd.DataFrame(scored)

            c1, c2, c3 = st.columns(3)
            c1.metric("Processed", batch_size)
            c2.metric("Flagged", int(results_df['predicted_fraud'].sum()))
            c3.metric("Avg Prob", f"{results_df['fraud_probability'].mean():.2%}")

            display_cols = [c for c in ['transaction_amount', 'distance_from_home_km',
                            'hour', 'is_foreign', 'vpn_detected', 'fraud_probability',
                            'risk_level', 'true_label', 'predicted_fraud']
                            if c in results_df.columns]

            def highlight_row(row):
                return (['background-color: #ffe0e0'] * len(row)
                        if row.get('predicted_fraud', 0) == 1 else [''] * len(row))

            st.dataframe(results_df[display_cols].style.apply(highlight_row, axis=1),
                         use_container_width=True)
            st.bar_chart(results_df['fraud_probability'])

            if 'true_label' in results_df.columns:
                acc = (results_df['predicted_fraud'] == results_df['true_label']).mean()
                st.metric("Batch Accuracy", f"{acc:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — HITL REVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_hitl:
    render_hitl_tab()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — AI ANALYST CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
with tab_ai:
    st.subheader("🤖 AI Fraud Analyst")
    st.markdown("Ask anything about the model, metrics, or fraud patterns.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask about the model or fraud patterns...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chat_with_analyst(question, metrics_context)
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
    st.caption("Powered by Groq llama3-8b. Add GROQ_API_KEY to Streamlit secrets.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab_audit:
    st.subheader("🗂️ Prediction Audit Log")
    audit_df = load_audit_log()
    if audit_df.empty:
        st.info("No predictions logged yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(audit_df))
        c2.metric("Flagged", int(audit_df['is_fraud'].sum()))
        c3.metric("Avg Prob", f"{audit_df['fraud_probability'].mean():.2%}")
        st.dataframe(audit_df.sort_values("timestamp", ascending=False),
                     use_container_width=True)
        if len(audit_df) > 1:
            audit_df['timestamp'] = pd.to_datetime(audit_df['timestamp'])
            st.line_chart(audit_df.set_index("timestamp")["fraud_probability"])
