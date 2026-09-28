import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import plotly.express as px
import plotly.graph_objects as go

from src.pipeline import run_pipeline
from src.model import train_model, safe_predict_proba
from src.business import compute_business_cost
from src.drift import detect_drift
from src.simulator import generate_batch, score_transaction
from src.pii import mask_pii
from src.hitl import add_to_review_queue, render_hitl_tab
from src.ingest import read_uploaded_file, validate_dataframe
from src.database import log_prediction, load_predictions, get_stats
from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH
from src.graph_neo4j import detect_fraud_rings, plot_fraud_ring_network, load_ring_log
from src.tokenizer import (tokenize, detokenize, rotate_keys,
                            get_key_store_summary, tokenize_dataframe)
from src.optimizer import optimize_threshold, roi_projection
from src.savings_tracker import get_savings_summary, load_savings_log
from src.shadow import shadow_predict, shadow_divergence_stats, load_shadow_log
from src.sar import load_sar_reports, update_sar_status
from src.agent_graph import run_copilot
from src.vector_store import (
    index_audit_log, index_sar_reports, get_collection_stats,
)
from src.copilot_metrics import (
    COPILOT_AGENT_RUNS, COPILOT_SAR_DRAFTS,
    COPILOT_HITL_QUEUED, COPILOT_RETRIEVER_HITS,
)
from src.observability import get_live_metrics, compute_live_psi
from src.compliance import run_full_compliance, load_compliance_report
from src.task_queue import submit_retrain, submit_drift_check, get_queue_stats
from src.object_store import get_registry_summary

def stream_one(fraud_rate=0.2, seed=None):
    from src.simulator import generate_transaction
    rng = np.random.default_rng(seed)
    is_fraud = rng.random() < fraud_rate
    return generate_transaction(rng, fraud=is_fraud)

from src.plots import (
    plot_class_distribution, plot_amount_distribution, plot_amount_box,
    plot_correlation_heatmap, plot_roc_curve, plot_precision_recall,
    plot_confusion_matrix, plot_feature_importance, plot_threshold_analysis,
    plot_velocity_heatmap, plot_fraud_by_hour, plot_fraud_by_channel,
    plot_scatter_risk, plot_anomaly_scatter, plot_shap_bar_interactive,
)
from src.shap_utils import plot_shap_summary, plot_shap_beeswarm, plot_waterfall

st.set_page_config(
    page_title="FraudGuard AI — Enterprise Fraud Detection",
    layout="wide",
    page_icon="🚨",
    menu_items={
        "Get Help": "https://github.com/SumedhPatil1507/fraud_detection_project",
        "Report a bug": "https://github.com/SumedhPatil1507/fraud_detection_project/issues",
        "About": "**FraudGuard AI** — Enterprise Fraud Detection Platform\n\nLangGraph Copilot · XGBoost/LightGBM · Neo4j Graph · Prometheus · DPDP/RBI Compliance",
    },
)

st.markdown("""
<style>
[data-testid="metric-container"] { background:#1e1e2e; border-radius:8px; padding:12px; }
.stTabs [data-baseweb="tab"] { font-size:12px; font-weight:600; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("🚨 FraudGuard AI — Enterprise Fraud Detection Platform")
st.caption(
    "LangGraph Copilot · XGBoost/LightGBM · Neo4j Graph · "
    "Async PostgreSQL · Celery · S3 · ChromaDB · Prometheus · DPDP/RBI Compliance"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("📂 Data Source")
data_mode = st.sidebar.radio("Mode", ["Built-in Dataset", "Upload File"])

df = None
data_source = "built-in"

if data_mode == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload file", type=["csv", "xlsx", "xls", "json", "parquet"],
        help="CSV, Excel, JSON, Parquet — must have a 'label' column (0/1)."
    )
    if uploaded_file:
        try:
            raw_df = read_uploaded_file(uploaded_file)
            valid, msg = validate_dataframe(raw_df)
            if valid:
                df, data_source = run_pipeline(raw_df=raw_df)
                st.sidebar.success(f"✅ {len(df):,} rows loaded")
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

# ── Cached artifact loader — loaded once, shared across all tabs ──────────────
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_model_artifacts_cached():
    """Load model, features, means once and cache in memory for the session."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    _model    = pickle.load(open(MODEL_PATH, "rb"))
    _features = pickle.load(open(FEATURE_PATH, "rb"))


# ── Module-level SHAP helper — defined here so the cache key is stable ────────
@st.cache_data(show_spinner="Computing SHAP values…", ttl=3600)
def _compute_shap_cached(_model_key: str, X_values: "np.ndarray",
                          feat_cols: list) -> "tuple[np.ndarray | None, list, str]":
    """
    Compute SHAP values and return (shap_array, feature_names, error_msg).
    Uses st.cache_data (not cache_resource) so results are properly keyed.
    _model_key is just used as a hash discriminator — the actual model is
    retrieved from session_state inside the tab block.
    Returns error_msg='' on success, otherwise a descriptive string.
    """
    try:
        import shap as _shap
        import pickle, os
        from src.config import MODEL_PATH
        if not os.path.exists(MODEL_PATH):
            return None, [], "No model.pkl found — train the model first."
        model = pickle.load(open(MODEL_PATH, "rb"))
        # Unwrap CalibratedClassifierCV → VotingClassifier → XGBClassifier
        inner = model
        if hasattr(inner, "calibrated_classifiers_"):
            inner = inner.calibrated_classifiers_[0].estimator
        xgb = inner.estimators_[0] if hasattr(inner, "estimators_") else inner
        explainer = _shap.TreeExplainer(xgb)
        X_df = pd.DataFrame(X_values, columns=feat_cols)
        raw  = explainer.shap_values(X_df)
        if hasattr(raw, "values"):
            raw = raw.values
        if isinstance(raw, list):
            raw = raw[1]
        arr = np.array(raw)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr, feat_cols, ""
    except ValueError as e:
        if "base_score" in str(e) or "could not convert" in str(e):
            return None, [], (
                "The stored model was trained with an older XGBoost version that "
                "is incompatible with the current SHAP. "
                "**Re-train the model** (🏋️ Train tab) to fix this."
            )
        return None, [], f"SHAP ValueError: {e}"
    except Exception as e:
        return None, [], f"SHAP error: {e}"
    _means    = pickle.load(open(MEAN_PATH, "rb"))
    return _model, _features, _means

st.sidebar.header("⚙️ Model Controls")
fn_cost = st.sidebar.number_input("FN Cost ($)", value=5000, step=500)
fp_cost = st.sidebar.number_input("FP Cost ($)", value=200, step=50)
use_optuna = st.sidebar.checkbox("Optuna Tuning", value=False)
n_trials = st.sidebar.slider("Trials", 5, 20, 10) if use_optuna else 10

st.sidebar.header("📊 Dataset Info")
st.sidebar.metric("Rows", f"{len(df):,}")
st.sidebar.metric("Fraud Rate", f"{df['label'].mean():.2%}")
st.sidebar.metric("Features", df.shape[1] - 1)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Explorer", "🏋️ Train", "📈 Metrics",
    "🔍 Explainability", "📡 Drift", "⚡ Predict",
    "🔴 Live Stream", "👤 HITL", "🗂️ Audit",
    "🕸️ Graph Intel", "🔑 Vault", "💰 Savings", "📋 SAR",
    "📡 Observability", "⚖️ Compliance", "🤖 Copilot"
])
(tab_data, tab_train, tab_metrics, tab_shap, tab_drift, tab_predict,
 tab_live, tab_hitl, tab_audit, tab_graph, tab_vault, tab_savings,
 tab_sar, tab_obs, tab_compliance, tab_copilot) = tabs

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    fraud_rate = df['label'].mean() * 100
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Fraud Cases", f"{df['label'].sum():,}")
    c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    c4.metric("Features", f"{df.shape[1] - 1}")
    c5.metric("Source", data_source)

    with st.expander("📋 Raw Data", expanded=False):
        st.dataframe(display_df.head(500), use_container_width=True)

    st.subheader("Distribution Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_class_distribution(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_amount_distribution(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_amount_box(df), use_container_width=True)
    with c2:
        fig_scatter = plot_scatter_risk(df)
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Temporal & Channel Analysis")
    c1, c2 = st.columns(2)
    with c1:
        fig_hour = plot_fraud_by_hour(df)
        if fig_hour:
            st.plotly_chart(fig_hour, use_container_width=True)
    with c2:
        fig_ch = plot_fraud_by_channel(df)
        if fig_ch:
            st.plotly_chart(fig_ch, use_container_width=True)

    fig_vel = plot_velocity_heatmap(df)
    if fig_vel:
        st.plotly_chart(fig_vel, use_container_width=True)

    st.subheader("Correlation Heatmap")
    st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.subheader("Train Ensemble Fraud Classifier")
    st.info("XGBoost + LightGBM soft-vote → calibrated probabilities → 3-fold CV")

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
        st.success(f"✅ ROC-AUC: **{roc:.4f}** | Threshold: **{threshold:.3f}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC", f"{roc:.4f}")
        c2.metric("CV Mean AUC", f"{cv_scores.mean():.4f}")
        c3.metric("CV Std", f"± {cv_scores.std():.4f}")

        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
            "ROC-AUC": cv_scores.round(4)
        })
        fig_cv = px.bar(cv_df, x='Fold', y='ROC-AUC', color='ROC-AUC',
                        color_continuous_scale='Blues', title='Cross-Validation AUC per Fold',
                        text_auto='.4f')
        fig_cv.add_hline(y=cv_scores.mean(), line_dash='dash', line_color='red',
                         annotation_text=f'Mean={cv_scores.mean():.4f}')
        st.plotly_chart(fig_cv, use_container_width=True)

        with st.expander("Best Hyperparameters"):
            st.json(best_params)

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
        c2.metric("Net Impact", f"${metrics['net_impact_usd']:,.0f}")
        c3.metric("Savings", f"${metrics['estimated_savings_usd']:,.0f}")
        c4.metric("Cost", f"${metrics['estimated_cost_usd']:,.0f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("True Positives", metrics['true_positive'])
        c6.metric("False Positives", metrics['false_positive'])
        c7.metric("False Negatives", metrics['false_negative'])
        c8.metric("Precision / Recall",
                  f"{metrics['precision']:.2f} / {metrics['recall']:.2f}")

        with st.expander("Classification Report"):
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_roc_curve(y_test, probs), use_container_width=True)
        with c2:
            st.plotly_chart(plot_precision_recall(y_test, probs), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_confusion_matrix(y_test, probs, threshold),
                            use_container_width=True)
        with c2:
            st.plotly_chart(plot_anomaly_scatter(anomaly_scores, probs, y_test),
                            use_container_width=True)

        st.plotly_chart(plot_threshold_analysis(y_test, probs), use_container_width=True)

        base = st.session_state.model
        fi_model = (base.estimators_[0] if hasattr(base, 'estimators_') else base)
        X_test_fi = st.session_state.get("X_test")
        if hasattr(fi_model, 'feature_importances_') and X_test_fi is not None:
            st.plotly_chart(
                plot_feature_importance(fi_model, X_test_fi.columns.tolist()),
                use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    if "model" not in st.session_state:
        st.info("Train the model first (🏋️ Train tab).")
    else:
        _shap_model = st.session_state.model
        _shap_Xtest = st.session_state.X_test
        y_test      = st.session_state.y_test

        # Unwrap CalibratedClassifierCV → inner estimator for TreeExplainer
        _inner = _shap_model
        if hasattr(_inner, "calibrated_classifiers_"):
            _inner = _inner.calibrated_classifiers_[0].estimator
        xgb_model = (_inner.estimators_[0] if hasattr(_inner, "estimators_") else _inner)

        X_sample    = _shap_Xtest.sample(min(100, len(_shap_Xtest)), random_state=42)
        feat_cols   = X_sample.columns.tolist()
        model_key   = str(id(_shap_model))          # stable-ish cache discriminator

        # Use the module-level cached function (stable cache key)
        shap_vals, _, shap_err = _compute_shap_cached(
            model_key, X_sample.values, feat_cols
        )

        # ── Global importance ─────────────────────────────────────────────────
        st.subheader("Global SHAP Feature Importance")

        if shap_vals is None:
            st.error(shap_err or "SHAP unavailable.")
            st.info(
                "This usually means the saved model pkl was built with a different "
                "XGBoost/SHAP version. Re-train the model in the **🏋️ Train** tab, "
                "then return here."
            )
            if st.button("🔄 Clear SHAP cache & retry", key="refresh_shap"):
                st.cache_data.clear()
                st.rerun()
        else:
            c1, c2 = st.columns(2)
            with c1:
                # Bar chart — mean |SHAP| per feature
                mean_abs   = np.abs(shap_vals).mean(axis=0)
                top_n      = 15
                idx_top    = np.argsort(mean_abs)[-top_n:]
                bar_fig    = go.Figure(go.Bar(
                    x=mean_abs[idx_top],
                    y=[feat_cols[i] for i in idx_top],
                    orientation='h',
                    marker=dict(
                        color=mean_abs[idx_top],
                        colorscale='Purples',
                        showscale=True,
                        colorbar=dict(title="Mean |SHAP|"),
                    ),
                    hovertemplate="<b>%{y}</b><br>Mean |SHAP| = %{x:.4f}<extra></extra>",
                ))
                bar_fig.update_layout(
                    title=f"Top {top_n} Features — Mean |SHAP|",
                    xaxis_title="Mean |SHAP Value|",
                    height=480, margin=dict(l=160),
                )
                st.plotly_chart(bar_fig, use_container_width=True)

            with c2:
                # SHAP dependence scatter for the top feature
                top_feat_idx = int(np.abs(shap_vals).mean(axis=0).argmax())
                top_feat     = feat_cols[top_feat_idx]
                dep_df = pd.DataFrame({
                    "Feature Value": X_sample.iloc[:, top_feat_idx].values,
                    "SHAP Value":    shap_vals[:, top_feat_idx],
                    "Actual":        (y_test.iloc[:len(X_sample)].values
                                      if len(y_test) >= len(X_sample)
                                      else np.zeros(len(X_sample))),
                })
                dep_fig = px.scatter(
                    dep_df, x="Feature Value", y="SHAP Value", color="Actual",
                    title=f"SHAP Dependence — {top_feat}",
                    color_continuous_scale="RdYlGn_r", opacity=0.75,
                    hover_data=["Feature Value"],
                )
                dep_fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                st.plotly_chart(dep_fig, use_container_width=True)

            # ── Beeswarm ──────────────────────────────────────────────────────
            st.subheader("SHAP Beeswarm — Feature Impact Distribution")
            bees_top = 12
            bees_idx = np.argsort(np.abs(shap_vals).mean(axis=0))[-bees_top:][::-1]
            bees_rows = []
            for fi in bees_idx:
                fname = feat_cols[fi]
                for si in range(len(X_sample)):
                    bees_rows.append({
                        "Feature":       fname,
                        "SHAP Value":    float(shap_vals[si, fi]),
                        "Feature Value": float(X_sample.iloc[si, fi]),
                    })
            bees_df  = pd.DataFrame(bees_rows)
            bees_fig = px.strip(
                bees_df, x="SHAP Value", y="Feature", color="Feature Value",
                color_continuous_scale="RdBu_r", orientation="h",
                title=f"SHAP Beeswarm (Top {bees_top} Features)",
                hover_data=["Feature Value"],
            )
            bees_fig.update_traces(marker=dict(size=4, opacity=0.7))
            bees_fig.update_layout(
                height=max(460, bees_top * 34),
                margin=dict(l=180),
                coloraxis_colorbar=dict(title="Feature Value"),
            )
            bees_fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(bees_fig, use_container_width=True)

        # ── Single prediction waterfall ───────────────────────────────────────
        st.subheader("Single Prediction — SHAP Waterfall")
        with st.form("shap_form"):
            c1, c2, c3 = st.columns(3)
            s_amount   = c1.number_input("Amount ($)",   value=1500.0)
            s_distance = c2.number_input("Distance (km)", value=300.0)
            s_hour     = c3.number_input("Hour",          min_value=0, max_value=23, value=2)
            shap_submit = st.form_submit_button("Explain Prediction", type="primary")

        if shap_submit:
            _, features, means = load_model_artifacts_cached()
            if means is None:
                st.warning("No trained model found. Train the model first.")
            else:
                base_dict = means.to_dict()
                base_dict.update({
                    "transaction_amount":    s_amount,
                    "distance_from_home_km": s_distance,
                    "hour":                  s_hour,
                    "amount_log":            np.log1p(s_amount),
                    "hour_sin":              np.sin(2 * np.pi * s_hour / 24),
                    "hour_cos":              np.cos(2 * np.pi * s_hour / 24),
                })
                input_df = pd.DataFrame([{f: base_dict.get(f, 0) for f in features}])
                prob     = safe_predict_proba(_shap_model, input_df)[0][1]

                c_prob, c_verdict = st.columns(2)
                c_prob.metric("Fraud Probability", f"{prob:.2%}")
                c_verdict.metric("Verdict",
                                 "🔴 HIGH RISK" if prob >= 0.3 else "🟢 LOW RISK")

                # Interactive Plotly waterfall
                wf_fig = plot_waterfall(xgb_model, input_df)
                st.plotly_chart(wf_fig, use_container_width=True)

                # Signed-SHAP bar (shows direction, not just magnitude)
                try:
                    import shap as _shap_lib
                    _exp = _shap_lib.TreeExplainer(xgb_model)
                    _raw = _exp.shap_values(input_df)
                    if hasattr(_raw, "values"):
                        _raw = _raw.values
                    if isinstance(_raw, list):
                        _raw = _raw[1]
                    sv_row = np.array(_raw)
                    if sv_row.ndim > 1:
                        sv_row = sv_row[0]

                    single_df = pd.DataFrame({
                        "Feature":       features,
                        "SHAP Value":    sv_row,
                        "Feature Value": input_df.values[0],
                    }).sort_values("SHAP Value", key=abs, ascending=False).head(14)

                    signed_fig = px.bar(
                        single_df, x="SHAP Value", y="Feature",
                        orientation="h", color="SHAP Value",
                        color_continuous_scale="RdBu_r",
                        title="SHAP Signed Impact (Single Prediction)",
                        hover_data=["Feature Value"],
                    )
                    signed_fig.add_vline(x=0, line_dash="dash",
                                         line_color="gray", opacity=0.5)
                    signed_fig.update_layout(height=480, margin=dict(l=180))
                    st.plotly_chart(signed_fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Signed-SHAP bar unavailable: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_drift:
    st.subheader("📡 Feature Drift Detection (PSI)")
    st.markdown("Population Stability Index — PSI > 0.2 means significant drift, retrain recommended.")

    if st.button("Run Drift Analysis", type="primary"):
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

            # Interactive PSI bar chart
            psi_df = pd.DataFrame.from_dict(
                drift_result["top_drifted_features"], orient='index', columns=["PSI"]
            ).reset_index().rename(columns={"index": "Feature"})
            psi_df['Status'] = psi_df['PSI'].apply(
                lambda x: 'High' if x > 0.2 else 'Moderate' if x > 0.1 else 'Low')
            fig_psi = px.bar(psi_df, x='PSI', y='Feature', orientation='h',
                             color='Status',
                             color_discrete_map={'High': '#e74c3c',
                                                 'Moderate': '#e67e22', 'Low': '#2ecc71'},
                             title='Feature PSI — Drift Analysis')
            fig_psi.add_vline(x=0.1, line_dash='dash', line_color='orange')
            fig_psi.add_vline(x=0.2, line_dash='dash', line_color='red')
            st.plotly_chart(fig_psi, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REAL-TIME PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("⚡ Real-time Transaction Scoring")
    _pred_model, _pred_features, _pred_means = load_model_artifacts_cached()
    if _pred_model is None:
        st.warning("Train the model first.")
    else:
        model, features, means = _pred_model, _pred_features, _pred_means
        pred_threshold = st.slider("Threshold", 0.01, 0.99, 0.30, 0.01, key="pred_thresh")

        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            amount     = c1.number_input("Amount ($)", min_value=0.0, value=500.0)
            distance   = c2.number_input("Distance (km)", min_value=0.0, value=10.0)
            hour       = c3.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
            c4, c5, c6 = st.columns(3)
            is_foreign = c4.selectbox("Foreign Transaction", [0, 1])
            is_new_dev = c5.selectbox("New Device", [0, 1])
            vpn        = c6.selectbox("VPN Detected", [0, 1])
            submitted  = st.form_submit_button("🔍 Score Transaction", type="primary")

        if submitted:
            if means is None:
                st.error("Model artifacts not found. Train the model first.")
            else:
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
                input_df = pd.DataFrame([{f: base.get(f, 0) for f in features}])
                prob = safe_predict_proba(model, input_df)[0][1]
                pred = int(prob >= pred_threshold)
                log_prediction(amount, distance, hour, is_foreign,
                               is_new_dev, vpn, prob, bool(pred), pred_threshold)

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    title={'text': "Fraud Risk Score"},
                    delta={'reference': pred_threshold * 100},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': '#e74c3c' if pred else '#2ecc71'},
                        'steps': [
                            {'range': [0, 30], 'color': '#d5f5e3'},
                            {'range': [30, 60], 'color': '#fdebd0'},
                            {'range': [60, 100], 'color': '#fadbd8'},
                        ],
                        'threshold': {'line': {'color': 'red', 'width': 4},
                                      'thickness': 0.75, 'value': pred_threshold * 100}
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

                if pred == 1:
                    st.error("🚨 HIGH RISK — Likely Fraud")
                    add_to_review_queue(
                        {"transaction_amount": amount, "distance_from_home_km": distance,
                         "hour": hour}, prob, "Auto-flagged by model")
                    st.warning("⚠️ Added to HITL review queue.")
                else:
                    st.success("✅ LOW RISK — Likely Legitimate")

                st.code(
                    f'curl -X POST http://localhost:8000/predict \\\n'
                    f'  -H "Content-Type: application/json" \\\n'
                    f'  -d \'{{"transaction_amount": {amount}, '
                    f'"distance_from_home_km": {distance}, "hour": {hour}, '
                    f'"threshold": {pred_threshold}}}\'', language="bash")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — LIVE TRANSACTION STREAM
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader("🔴 Live Transaction Stream")
    st.caption("Simulates a real-time feed of incoming transactions scored by the model.")

    _live_model, _live_features, _live_means = load_model_artifacts_cached()
    if _live_model is None:
        st.warning("Train the model first.")
    else:
        model, features, means = _live_model, _live_features, _live_means

        # ── Controls ───────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        sim_fraud_rate = c1.slider("Fraud rate", 0.05, 0.5, 0.2, key="live_fr")
        sim_threshold  = c2.slider("Alert threshold", 0.1, 0.9, 0.3, key="live_th")
        stream_speed   = c3.selectbox("Speed", [0.5, 1, 2, 3], index=1,
                                       format_func=lambda x: f"{x}s/txn")
        max_history    = c4.slider("History size", 20, 200, 50)

        col_start, col_stop, col_clear = st.columns(3)
        start = col_start.button("▶️ Start Stream", type="primary")
        stop  = col_stop.button("⏹ Stop")
        clear = col_clear.button("🗑 Clear")

        if clear:
            st.session_state.live_feed = []
            st.rerun()
        if stop:
            st.session_state.streaming = False

        if "live_feed" not in st.session_state:
            st.session_state.live_feed = []
        if "streaming" not in st.session_state:
            st.session_state.streaming = False
        if start:
            st.session_state.streaming = True

        # ── Live metrics placeholders ──────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        ph_total   = m1.empty()
        ph_fraud   = m2.empty()
        ph_rate    = m3.empty()
        ph_avg     = m4.empty()

        ph_chart   = st.empty()
        ph_alerts  = st.empty()
        ph_table   = st.empty()

        def render_live(feed):
            if not feed:
                return
            fdf = pd.DataFrame(feed[-max_history:])
            total   = len(fdf)
            flagged = int(fdf['predicted_fraud'].sum())
            rate    = flagged / total * 100
            avg_p   = fdf['fraud_probability'].mean()

            ph_total.metric("Transactions", total)
            ph_fraud.metric("Flagged", flagged)
            ph_rate.metric("Fraud Rate", f"{rate:.1f}%")
            ph_avg.metric("Avg Prob", f"{avg_p:.2%}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(fdf))), y=fdf['fraud_probability'],
                mode='lines+markers',
                marker=dict(color=fdf['predicted_fraud'].map(
                    {1: '#e74c3c', 0: '#2ecc71'})),
                line=dict(color='#3498db', width=1.5),
                name='Fraud Prob'
            ))
            fig.add_hline(y=sim_threshold, line_dash='dash',
                          line_color='red', annotation_text='Threshold')
            fig.update_layout(
                title='Live Fraud Probability Feed',
                xaxis_title='Transaction #', yaxis_title='Fraud Probability',
                yaxis=dict(range=[0, 1]), height=300, margin=dict(t=40, b=30)
            )
            ph_chart.plotly_chart(fig, use_container_width=True)

            alerts = fdf[fdf['predicted_fraud'] == 1].tail(5)
            if not alerts.empty:
                alert_cols = [c for c in ['timestamp', 'transaction_amount',
                              'distance_from_home_km', 'fraud_probability',
                              'risk_level'] if c in alerts.columns]
                ph_alerts.error(
                    f"🚨 {len(alerts)} recent alert(s)\n" +
                    alerts[alert_cols].to_string(index=False))

            disp_cols = [c for c in ['timestamp', 'transaction_amount',
                         'distance_from_home_km', 'hour', 'fraud_probability',
                         'risk_level', 'predicted_fraud'] if c in fdf.columns]
            ph_table.dataframe(
                fdf[disp_cols].tail(20).sort_index(ascending=False),
                use_container_width=True)

        # ── Stream loop ────────────────────────────────────────────────────────
        if st.session_state.streaming:
            for _ in range(200):
                if not st.session_state.streaming:
                    break
                seed = int(time.time() * 1000) % 999999
                txn = stream_one(fraud_rate=sim_fraud_rate, seed=seed)
                scored = score_transaction(txn, model, features, means, sim_threshold)
                st.session_state.live_feed.append(scored)
                if len(st.session_state.live_feed) > max_history:
                    st.session_state.live_feed = st.session_state.live_feed[-max_history:]
                render_live(st.session_state.live_feed)
                time.sleep(stream_speed)
        else:
            render_live(st.session_state.live_feed)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — HITL REVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_hitl:
    render_hitl_tab()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — AUDIT LOG (with live refresh)
# ══════════════════════════════════════════════════════════════════════════════
with tab_audit:
    st.subheader("🗂️ Prediction Audit Log")

    # DB source indicator
    stats = get_stats()
    db_badge = "🟢 Supabase" if stats["source"] == "supabase" else "🟡 Local CSV"
    st.caption(f"Storage: {db_badge}")

    col1, col2 = st.columns([3, 1])
    with col2:
        auto_refresh = st.toggle("🔴 Live Refresh", value=False)
        refresh_interval = st.selectbox("Interval", [5, 10, 30], index=1,
                                        format_func=lambda x: f"{x}s")

    audit_df = load_predictions()
    if audit_df.empty:
        st.info("No predictions logged yet. Make predictions in the Predict tab.")
    else:
        audit_df['timestamp'] = pd.to_datetime(audit_df['timestamp'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", stats["total"])
        c2.metric("Flagged as Fraud", stats["fraud"])
        fraud_pct = stats["fraud"] / max(stats["total"], 1) * 100
        c3.metric("Fraud Rate", f"{fraud_pct:.1f}%")
        c4.metric("Last Updated", audit_df['timestamp'].max().strftime("%H:%M:%S"))

        fig_audit = px.scatter(
            audit_df, x='timestamp', y='fraud_probability',
            color='is_fraud',
            color_discrete_map={True: '#e74c3c', False: '#2ecc71'},
            title='Fraud Probability Over Time',
            labels={'fraud_probability': 'Fraud Probability',
                    'timestamp': 'Time', 'is_fraud': 'Flagged'},
            hover_data=[c for c in ['transaction_amount',
                        'distance_from_home_km', 'hour'] if c in audit_df.columns]
        )
        fig_audit.add_hline(y=0.3, line_dash='dash', line_color='orange',
                            annotation_text='Default threshold')
        st.plotly_chart(fig_audit, use_container_width=True)

        if len(audit_df) >= 5:
            audit_df_sorted = audit_df.sort_values('timestamp')
            audit_df_sorted['rolling_fraud_rate'] = (
                audit_df_sorted['is_fraud'].rolling(5, min_periods=1).mean() * 100
            )
            fig_trend = px.line(audit_df_sorted, x='timestamp',
                                y='rolling_fraud_rate',
                                title='Rolling Fraud Rate (5-prediction window)',
                                labels={'rolling_fraud_rate': 'Fraud Rate (%)'})
            fig_trend.add_hline(y=audit_df['is_fraud'].mean() * 100,
                                line_dash='dash', line_color='red',
                                annotation_text='Overall avg')
            st.plotly_chart(fig_trend, use_container_width=True)

        st.dataframe(audit_df.sort_values("timestamp", ascending=False),
                     use_container_width=True)

    if auto_refresh:
        st.caption(f"Auto-refreshing every {refresh_interval}s...")
        time.sleep(refresh_interval)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — GRAPH INTELLIGENCE & FRAUD RING DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_graph:
    st.subheader("🕸️ Graph Intelligence — Fraud Ring Detector")
    st.markdown(
        "Builds a **customer ↔ merchant bipartite graph** using NetworkX. "
        "Detects rings via **merchant hubs** (shared by multiple customers), "
        "**connected components** with elevated fraud rate, and **shared device** clusters."
    )

    c1, c2 = st.columns(2)
    min_fraud_rate = c1.slider("Min ring fraud rate", 0.05, 0.9, 0.1, 0.05,
                                help="Lower = more rings detected. Start at 0.1 for most datasets.")
    min_ring_size  = c2.slider("Min ring size (nodes)", 2, 10, 2,
                                help="Minimum nodes in a ring. 2 = merchant + 1 customer.")

    if st.button("🔍 Detect Fraud Rings", type="primary"):
        with st.spinner("Building transaction graph and detecting rings..."):
            rings = detect_fraud_rings(df, min_fraud_rate=min_fraud_rate,
                                        min_ring_size=min_ring_size)

        if not rings:
            st.warning(
                "No fraud rings detected. Try lowering the **Min ring fraud rate** slider, "
                "or ensure your dataset has `customer_id`, `merchant_id`, and `label` columns."
            )
        else:
            st.success(f"🚨 {len(rings)} fraud ring(s) detected!")

            total_amount = sum(r["total_amount"] for r in rings)
            avg_fraud_rate = np.mean([r["fraud_rate"] for r in rings])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rings Detected", len(rings))
            c2.metric("Total Exposure", f"${total_amount:,.0f}")
            c3.metric("Avg Fraud Rate", f"{avg_fraud_rate:.1%}")
            c4.metric("Largest Ring", max(r["size"] for r in rings))

            fig_net = plot_fraud_ring_network(rings, max_rings=5)
            if fig_net:
                st.plotly_chart(fig_net, use_container_width=True)

            ring_df = pd.DataFrame([{
                "Ring ID":      r["ring_id"],
                "Type":         r.get("ring_type", ""),
                "Size":         r["size"],
                "Customers":    r["customer_count"],
                "Merchants":    r["merchant_count"],
                "Fraud Rate":   f"{r['fraud_rate']:.1%}",
                "Total Amount": f"${r['total_amount']:,.0f}",
                "Fraud Txns":   r["total_fraud_txn"],
                "Risk Level":   r["risk_level"],
                "Hub Nodes":    ", ".join(str(h) for h in r["hub_nodes"][:2]),
            } for r in rings])
            st.dataframe(ring_df, use_container_width=True)

            graph_cols = [c for c in df.columns if c in
                          ["graph_risk_score", "customer_degree", "merchant_degree",
                           "customer_fraud_rate", "merchant_fraud_rate", "ring_member"]]
            if graph_cols:
                st.subheader("Graph Feature Distributions")
                c1, c2 = st.columns(2)
                if "graph_risk_score" in df.columns:
                    fig_gr = px.histogram(df, x="graph_risk_score", color="label",
                                          nbins=50, title="Graph Risk Score Distribution",
                                          color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                                          barmode="overlay", opacity=0.7)
                    c1.plotly_chart(fig_gr, use_container_width=True)
                if "ring_member" in df.columns:
                    ring_fraud = df.groupby("ring_member")["label"].mean().reset_index()
                    ring_fraud.columns = ["Ring Member", "Fraud Rate"]
                    ring_fraud["Ring Member"] = ring_fraud["Ring Member"].map(
                        {0: "Non-Ring", 1: "Ring Member"})
                    fig_rm = px.bar(ring_fraud, x="Ring Member", y="Fraud Rate",
                                    color="Ring Member", title="Fraud Rate: Ring vs Non-Ring",
                                    color_discrete_map={"Ring Member": "#e74c3c",
                                                        "Non-Ring": "#2ecc71"})
                    c2.plotly_chart(fig_rm, use_container_width=True)

    existing_rings = load_ring_log()
    if existing_rings:
        with st.expander(f"📂 Last Detected Rings ({len(existing_rings)} rings)", expanded=False):
            st.json(existing_rings[:3])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — VAULT & TOKENIZATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_vault:
    st.subheader("🔑 Cryptographic Vault & Tokenization Engine")
    st.markdown(
        "Format-preserving tokenization for PII fields. "
        "Versioned key rotation with MultiFernet. "
        "Role-gated detokenization — only admin/analyst can reverse tokens."
    )

    # Key store status
    key_summary = get_key_store_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Key Versions", key_summary["total_keys"])
    c2.metric("Active Version", key_summary["active_version"] or "None")
    c3.metric("Status", "🟢 Active" if key_summary["active_version"] else "🔴 No Key")

    if key_summary["keys"]:
        key_df = pd.DataFrame(key_summary["keys"])
        st.dataframe(key_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Rotate Keys", type="primary"):
            result = rotate_keys()
            st.success(f"✅ Key rotated → Version {result['new_version']}")
            st.json(result)

    with col2:
        if st.button("🔑 Initialize Keys (first run)"):
            from src.tokenizer import generate_key
            k = generate_key(purpose="init")
            st.success(f"✅ Key v{k['version']} generated")

    st.divider()
    st.subheader("🔐 Tokenize / Detokenize PII")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Tokenize a value**")
        tok_field = st.selectbox("Field", ["customer_id", "merchant_id", "device_id"])
        tok_value = st.text_input("Plaintext value", placeholder="e.g. CUST_12345")
        if st.button("Tokenize"):
            if tok_value:
                token = tokenize(tok_value, tok_field)
                st.code(token)
                st.caption("Token stored in vault. Original encrypted with active key.")

    with col2:
        st.markdown("**Detokenize (role-gated)**")
        detok_token = st.text_input("Token", placeholder="TKN_CU_XXXX")
        detok_role  = st.selectbox("Your role", ["admin", "analyst", "viewer"])
        if st.button("Detokenize"):
            if detok_token:
                result = detokenize(detok_token, detok_role)
                if "REDACTED" in result or "FAILED" in result or "NOT_FOUND" in result:
                    st.error(result)
                else:
                    st.success(f"Plaintext: `{result}`")

    st.divider()
    st.subheader("📋 Supabase pgcrypto Schema")
    st.caption("Copy this SQL into your Supabase SQL editor to enable server-side encryption.")
    from src.tokenizer import get_pgcrypto_schema
    st.code(get_pgcrypto_schema(), language="sql")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 12 — SAVINGS & ROI OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════
with tab_savings:
    st.subheader("💰 Fraud Savings Tracker & ROI Optimizer")

    # Live savings summary
    summary = get_savings_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Fraud Caught", summary["total_caught"])
    c2.metric("Total Savings", f"${summary['total_savings_usd']:,.0f}")
    c3.metric("Confirmed Savings", f"${summary['confirmed_savings_usd']:,.0f}")
    c4.metric("Avg Fraud Amount", f"${summary['avg_fraud_amount']:,.0f}")

    savings_df = load_savings_log()
    if not savings_df.empty:
        savings_df["timestamp"] = pd.to_datetime(savings_df["timestamp"])
        savings_df["cumulative_savings"] = savings_df["estimated_saving_usd"].cumsum()
        fig_sav = px.area(savings_df, x="timestamp", y="cumulative_savings",
                          title="Cumulative Fraud Savings Over Time",
                          labels={"cumulative_savings": "Cumulative Savings ($)"},
                          color_discrete_sequence=["#2ecc71"])
        st.plotly_chart(fig_sav, use_container_width=True)

    st.divider()
    st.subheader("🎯 Dynamic Threshold Optimizer")
    if "probs" in st.session_state and "y_test" in st.session_state:
        y_test = st.session_state.y_test
        probs  = st.session_state.probs
        opt_result = optimize_threshold(
            np.array(y_test), np.array(probs), fn_cost=fn_cost, fp_cost=fp_cost
        )
        sweep = opt_result["sweep_df"]
        opt_t = opt_result["optimal_threshold"]

        c1, c2 = st.columns(2)
        c1.metric("Optimal Threshold", f"{opt_t:.3f}")
        c2.metric("Max Net Impact", f"${opt_result['max_net_impact_usd']:,.0f}")

        fig_opt = px.line(sweep, x="threshold", y="net_impact_usd",
                          title="Net Impact vs Threshold",
                          labels={"net_impact_usd": "Net Impact ($)", "threshold": "Threshold"})
        fig_opt.add_vline(x=opt_t, line_dash="dash", line_color="gold",
                          annotation_text=f"Optimal: {opt_t:.3f}")
        fig_opt.add_hline(y=0, line_dash="dot", line_color="red")
        st.plotly_chart(fig_opt, use_container_width=True)

        c1, c2 = st.columns(2)
        fig_pr = px.line(sweep, x="threshold", y=["precision", "recall", "f1"],
                         title="Precision / Recall / F1 vs Threshold")
        c1.plotly_chart(fig_pr, use_container_width=True)
        fig_cost = px.line(sweep, x="threshold", y=["savings_usd", "cost_usd"],
                           title="Savings vs Cost vs Threshold",
                           color_discrete_map={"savings_usd": "#2ecc71", "cost_usd": "#e74c3c"})
        c2.plotly_chart(fig_cost, use_container_width=True)
    else:
        st.info("Train the model first to run threshold optimization.")

    st.divider()
    st.subheader("📈 ROI Projection Calculator")
    c1, c2, c3 = st.columns(3)
    daily_txns    = c1.number_input("Daily Transactions", value=10000, step=1000)
    fraud_rate_pct = c2.slider("Fraud Rate (%)", 0.1, 10.0, 2.0, 0.1)
    avg_fraud_amt = c3.number_input("Avg Fraud Amount ($)", value=2000, step=100)
    c4, c5, c6 = st.columns(3)
    det_rate  = c4.slider("Detection Rate", 0.5, 1.0, 0.85, 0.01)
    fp_rate_p = c5.slider("False Positive Rate", 0.001, 0.05, 0.01, 0.001)
    proj_months = c6.slider("Projection (months)", 1, 24, 12)

    roi = roi_projection(
        daily_transactions=int(daily_txns),
        fraud_rate=fraud_rate_pct / 100,
        avg_fraud_amount=avg_fraud_amt,
        detection_rate=det_rate,
        fp_rate=fp_rate_p,
        fp_cost=fp_cost,
        months=proj_months,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly Net Savings", f"${roi['monthly_net_usd']:,.0f}")
    c2.metric(f"{proj_months}-Month ROI", f"${roi['annual_net_usd']:,.0f}")
    c3.metric("Daily Fraud Caught", f"{roi['daily_fraud_caught']:.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 13 — SAR MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab_sar:
    st.subheader("📋 SAR Report Management")
    st.markdown("Automated Suspicious Activity Reports — FinCEN-style with ML justification.")

    reports = load_sar_reports()
    if not reports:
        st.info("No SAR reports generated yet. Flag a transaction in the Predict tab.")
    else:
        # Summary
        statuses = [r.get("status", "DRAFT") for r in reports]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total SARs", len(reports))
        c2.metric("Draft", statuses.count("DRAFT"))
        c3.metric("Filed", statuses.count("FILED"))
        c4.metric("Reviewed", statuses.count("REVIEWED"))

        # Status filter
        status_filter = st.selectbox("Filter by status", ["All", "DRAFT", "REVIEWED", "FILED"])
        filtered = reports if status_filter == "All" else [
            r for r in reports if r.get("status") == status_filter
        ]

        for sar in filtered[:20]:
            txn = sar.get("transaction", {})
            ml  = sar.get("ml_assessment", {})
            with st.expander(
                f"📄 {sar['sar_id']} | ${txn.get('amount_usd', 0):,.0f} | "
                f"Prob: {ml.get('fraud_probability', 0):.1%} | {sar.get('status', 'DRAFT')}"
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Amount", f"${txn.get('amount_usd', 0):,.0f}")
                c2.metric("Fraud Prob", f"{ml.get('fraud_probability', 0):.1%}")
                c3.metric("Risk Level", ml.get("risk_level", "N/A"))

                st.markdown(f"**Narrative:** {sar.get('narrative', '')}")
                st.markdown(f"**Recommended Action:** {sar.get('recommended_action', '')}")

                if sar.get("risk_indicators"):
                    st.markdown("**Risk Indicators:** " + " · ".join(sar["risk_indicators"]))

                col1, col2, col3 = st.columns(3)
                notes = st.text_input("Analyst notes", key=f"notes_{sar['sar_id']}")
                if col1.button("✅ Mark Reviewed", key=f"rev_{sar['sar_id']}"):
                    update_sar_status(sar["sar_id"], "REVIEWED", notes)
                    st.rerun()
                if col2.button("📤 Mark Filed", key=f"file_{sar['sar_id']}"):
                    update_sar_status(sar["sar_id"], "FILED", notes)
                    st.rerun()
                if col3.button("🗑 Dismiss", key=f"dis_{sar['sar_id']}"):
                    update_sar_status(sar["sar_id"], "DISMISSED", notes)
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 14 — OBSERVABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_obs:
    st.subheader("📡 Infrastructure Observability")
    st.caption("Prometheus-ready metrics · Live PSI · TP/FP monitoring · Queue depth · DB pool")

    # ── Live model metrics ─────────────────────────────────────────────────────
    audit_df_obs = load_predictions(500)
    metrics_live = get_live_metrics(audit_df_obs)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Predictions", f"{metrics_live['total_predictions']:,}")
    c2.metric("Live Fraud Rate",   f"{metrics_live['fraud_rate']:.2%}")
    c3.metric("Avg Probability",   f"{metrics_live['avg_probability']:.3f}")
    c4.metric("p95 Score",         f"{metrics_live['p95_probability']:.3f}")
    c5.metric("p99 Score",         f"{metrics_live['p99_probability']:.3f}")

    col1, col2 = st.columns(2)

    # ── Probability distribution ───────────────────────────────────────────────
    if not audit_df_obs.empty and "fraud_probability" in audit_df_obs.columns:
        with col1:
            fig_dist = px.histogram(
                audit_df_obs, x="fraud_probability",
                color="is_fraud" if "is_fraud" in audit_df_obs.columns else None,
                nbins=50, title="Live Fraud Score Distribution",
                color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
                barmode="overlay", opacity=0.75,
                labels={"fraud_probability": "Fraud Probability"}
            )
            fig_dist.add_vline(x=0.95, line_dash="dash", line_color="red",
                               annotation_text="p95")
            fig_dist.add_vline(x=0.99, line_dash="dot", line_color="orange",
                               annotation_text="p99")
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            if "timestamp" in audit_df_obs.columns:
                audit_df_obs["timestamp"] = pd.to_datetime(audit_df_obs["timestamp"])
                audit_df_obs_s = audit_df_obs.sort_values("timestamp")
                audit_df_obs_s["rolling_fr"] = (
                    audit_df_obs_s["is_fraud"].astype(float)
                    .rolling(10, min_periods=1).mean() * 100
                )
                fig_roll = px.line(audit_df_obs_s, x="timestamp", y="rolling_fr",
                                   title="Rolling Fraud Rate (10-prediction window)",
                                   labels={"rolling_fr": "Fraud Rate (%)"})
                st.plotly_chart(fig_roll, use_container_width=True)

    st.divider()

    # ── Drift (PSI) live ───────────────────────────────────────────────────────
    st.subheader("Live PSI Drift Tracking")
    if st.button("Refresh PSI", key="obs_psi"):
        X_num = df.select_dtypes(include="number").drop(
            columns=["label", "financial_loss"], errors="ignore")
        from src.drift import detect_drift as _detect
        dr = _detect(X_num)
        from src.observability import update_psi_metrics
        update_psi_metrics(dr)
        if "top_drifted_features" in dr:
            psi_data = compute_live_psi(audit_df_obs, dr)
            fig_psi = px.bar(
                x=psi_data["features"], y=psi_data["psi_values"],
                title=f"PSI per Feature — Overall: {psi_data['overall_psi']:.4f} ({psi_data['drift_level']})",
                labels={"x": "Feature", "y": "PSI"},
                color=psi_data["psi_values"],
                color_continuous_scale=["#2ecc71", "#e67e22", "#e74c3c"],
            )
            fig_psi.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Moderate")
            fig_psi.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="High")
            st.plotly_chart(fig_psi, use_container_width=True)

    st.divider()

    # ── Task queue stats ───────────────────────────────────────────────────────
    st.subheader("Task Queue (Celery + Redis)")
    q_stats = get_queue_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Celery Available", "✅" if q_stats.get("celery_available") else "❌ (sync mode)")
    c2.metric("Active Tasks",  q_stats.get("active_tasks", "—"))
    c3.metric("Queued Tasks",  q_stats.get("queued_tasks", "—"))
    st.caption(f"Redis: `{q_stats.get('redis_url', 'not configured')}`")

    col1, col2 = st.columns(2)
    if col1.button("🔄 Trigger Retraining", key="obs_retrain"):
        result = submit_retrain(reason="manual_ui")
        st.json(result)
    if col2.button("📊 Trigger Drift Check", key="obs_drift"):
        result = submit_drift_check()
        st.json(result)

    st.divider()

    # ── Object store registry ──────────────────────────────────────────────────
    st.subheader("Object Store — Artifact Registry")
    reg = get_registry_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Artifacts",  reg.get("total_artifacts", 0))
    c2.metric("Backend",    reg.get("backend", "local").upper())
    c3.metric("Bucket",     reg.get("bucket", "local"))
    if reg.get("artifacts"):
        st.dataframe(pd.DataFrame(reg["artifacts"]), use_container_width=True)

    st.divider()

    # ── Grafana dashboard snippet ──────────────────────────────────────────────
    with st.expander("📋 Grafana Dashboard JSON (copy to Grafana Import)"):
        from src.observability import GRAFANA_DASHBOARD_JSON
        st.json(GRAFANA_DASHBOARD_JSON)

    st.caption(
        "To enable Prometheus metrics: `pip install prometheus-client` and set "
        "`PROMETHEUS_PORT=9090`. Scrape endpoint: `http://api-host:9090/metrics`"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 15 — COMPLIANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_compliance:
    st.subheader("⚖️ Compliance Mapping & Regulatory Verification")
    st.markdown(
        "Automated checks against **DPDP Act 2023** (India) and **RBI IT Framework**. "
        "Each control is evaluated against current deployment configuration."
    )

    if st.button("🔍 Run Compliance Audit", type="primary"):
        with st.spinner("Running compliance checks..."):
            report = run_full_compliance()
        st.session_state["compliance_report"] = report

    report = st.session_state.get("compliance_report") or load_compliance_report()

    if report:
        summary = report.get("summary", {})
        score = summary.get("score", 0)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Compliance Score", f"{score:.0f}%")
        c2.metric("Total Controls", summary.get("total", 0))
        c3.metric("✅ Pass", summary.get("pass", 0))
        c4.metric("⚠️ Warn", summary.get("warn", 0))
        c5.metric("❌ Fail", summary.get("fail", 0))

        # Score gauge
        fig_score = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Compliance Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ecc71" if score >= 70 else "#e67e22" if score >= 50 else "#e74c3c"},
                "steps": [
                    {"range": [0, 50],  "color": "#fadbd8"},
                    {"range": [50, 75], "color": "#fdebd0"},
                    {"range": [75, 100],"color": "#d5f5e3"},
                ],
            }
        ))
        fig_score.update_layout(height=250)
        st.plotly_chart(fig_score, use_container_width=True)

        # Framework breakdown
        for framework, controls in report.get("frameworks", {}).items():
            st.subheader(f"📋 {framework}")
            for ctrl in controls:
                status = ctrl["status"]
                icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "NA": "ℹ️"}.get(status, "")
                color = {"PASS": "success", "WARN": "warning", "FAIL": "error", "NA": "info"}.get(status, "info")
                with st.expander(f"{icon} {ctrl['control_id']} — {ctrl['title']} [{status}]"):
                    st.markdown(f"**Evidence:** {ctrl['evidence']}")
                    if ctrl.get("remediation"):
                        st.markdown(f"**Remediation:** {ctrl['remediation']}")
                    st.caption(f"Checked: {ctrl.get('checked_at', '')[:19]}")
    else:
        st.info("Click 'Run Compliance Audit' to generate the report.")

    with st.expander("📖 Regulatory References"):
        st.markdown("""
**DPDP Act 2023** — Digital Personal Data Protection Act, India
- Ministry of Electronics & IT: https://www.meity.gov.in/data-protection-framework
- Applies to: Any entity processing digital personal data of Indian residents

**RBI IT Framework** — Reserve Bank of India IT Framework for Banks
- Circular: RBI/2023-24/112 CEPD No.S1130/13-01-003/2023-24
- Applies to: Banks, NBFCs, payment aggregators regulated by RBI

**FinCEN SAR** — Financial Crimes Enforcement Network
- Suspicious Activity Report filing obligations (US)
- Automated SAR generation in `/outputs/sar_reports/`
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 16 — FRAUD INVESTIGATION COPILOT
# ══════════════════════════════════════════════════════════════════════════════
with tab_copilot:
    st.subheader("🤖 Fraud Investigation Copilot")
    st.markdown(
        "LangGraph multi-agent pipeline: **Retriever** (semantic search over "
        "audit log + SARs) → **Tool** (graph-ring membership + SHAP drivers) → "
        "**Writer** (AI SAR narrative) → **HITL queue** (analyst approval)."
    )

    # ── Vector-store status & indexing ────────────────────────────────────────
    with st.expander("📚 Vector Store — Index Management", expanded=False):
        stats = get_collection_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Audit-log docs",  stats["audit_log_docs"])
        c2.metric("SAR docs",        stats["sar_docs"])
        c3.metric("Embed model",     stats["embed_model"])
        c4.metric(
            "Backend",
            "ChromaDB ✅" if stats["chroma_available"] else "ChromaDB ❌ (install)",
        )
        if not stats["chroma_available"] or not stats["st_available"]:
            st.warning(
                "Vector search is disabled. Run: "
                "`pip install chromadb sentence-transformers` then restart."
            )

        col_idx1, col_idx2, col_idx3 = st.columns(3)
        if col_idx1.button("🔄 Index Audit Log"):
            with st.spinner("Embedding audit log…"):
                n = index_audit_log()
            st.success(f"Upserted {n} audit-log documents.")
        if col_idx2.button("🔄 Index SAR Reports"):
            with st.spinner("Embedding SAR reports…"):
                n = index_sar_reports()
            st.success(f"Upserted {n} SAR documents.")
        if col_idx3.button("🔄 Index Both"):
            with st.spinner("Indexing…"):
                na = index_audit_log()
                ns = index_sar_reports()
            st.success(f"Audit: {na} docs  |  SARs: {ns} docs")

    st.divider()

    # ── Prometheus counter snapshot ────────────────────────────────────────────
    with st.expander("📊 Copilot Metrics Snapshot", expanded=False):
        st.caption(
            "Live counters from `fraudguard_copilot_*` — scraped by Prometheus "
            "on the `/metrics` endpoint."
        )
        try:
            from prometheus_client import REGISTRY
            # Pull the sample values we care about
            _prom_rows = []
            for metric in REGISTRY.collect():
                if metric.name.startswith("fraudguard_copilot"):
                    for sample in metric.samples:
                        _prom_rows.append({
                            "metric": sample.name,
                            "labels": str(sample.labels),
                            "value":  sample.value,
                        })
            if _prom_rows:
                st.dataframe(
                    pd.DataFrame(_prom_rows).sort_values("metric"),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No copilot metrics recorded yet — run the pipeline first.")
        except Exception:
            st.info("prometheus_client not installed — metrics are no-ops.")

    st.divider()

    # ── Transaction input form ────────────────────────────────────────────────
    st.subheader("🔎 Investigate a Transaction")
    st.caption(
        "Fill in the transaction details below. If a trained model is loaded "
        "the copilot will also compute SHAP feature drivers."
    )

    with st.form("copilot_form"):
        fc1, fc2, fc3 = st.columns(3)
        cp_amount   = fc1.number_input("Amount ($)",         min_value=0.0, value=3500.0, step=100.0)
        cp_distance = fc2.number_input("Distance (km)",      min_value=0.0, value=450.0,  step=10.0)
        cp_hour     = fc3.number_input("Hour (0–23)",        min_value=0,   max_value=23, value=3)

        fc4, fc5, fc6 = st.columns(3)
        cp_foreign  = fc4.selectbox("Foreign transaction",  [0, 1], format_func=lambda x: "Yes" if x else "No")
        cp_new_dev  = fc5.selectbox("New / unknown device", [0, 1], format_func=lambda x: "Yes" if x else "No")
        cp_vpn      = fc6.selectbox("VPN detected",         [0, 1], format_func=lambda x: "Yes" if x else "No")

        fc7, fc8 = st.columns(2)
        cp_prob   = fc7.slider(
            "Fraud probability (from model)", 0.0, 1.0, 0.75, 0.01,
            help="Use the Predict tab to obtain this value, or enter manually."
        )
        cp_cust_id = fc8.text_input("Customer ID (optional)", placeholder="e.g. 1042")
        cp_merch_id = st.text_input("Merchant ID (optional)", placeholder="e.g. 77")

        cp_submit = st.form_submit_button("🚀 Run Copilot Investigation", type="primary")

    if cp_submit:
        # Build the transaction dict
        cp_txn = {
            "transaction_amount":     cp_amount,
            "distance_from_home_km":  cp_distance,
            "hour":                   cp_hour,
            "is_foreign":             bool(cp_foreign),
            "is_new_device":          bool(cp_new_dev),
            "vpn_detected":           bool(cp_vpn),
            "timestamp":              pd.Timestamp.utcnow().isoformat(),
            "customer_id":            cp_cust_id or "unknown",
            "merchant_id":            cp_merch_id or "unknown",
        }

        # Optionally pass the trained model + aligned input_df for SHAP
        cp_model, cp_features, cp_means = load_model_artifacts_cached()
        cp_input_df = None
        if cp_model is not None and cp_means is not None:
            base = cp_means.to_dict()
            base.update({
                "transaction_amount":    cp_amount,
                "distance_from_home_km": cp_distance,
                "hour":                  cp_hour,
                "amount_log":            np.log1p(cp_amount),
                "hour_sin":              np.sin(2 * np.pi * cp_hour / 24),
                "hour_cos":              np.cos(2 * np.pi * cp_hour / 24),
                "is_foreign":            cp_foreign,
                "is_new_device":         cp_new_dev,
                "vpn_detected":          cp_vpn,
            })
            cp_input_df = pd.DataFrame([{f: base.get(f, 0) for f in cp_features}])

        # ── Run the pipeline ──────────────────────────────────────────────────
        with st.spinner("🤖 Copilot investigating… (retriever → tool → writer → HITL)"):
            try:
                result = run_copilot(
                    transaction=cp_txn,
                    fraud_probability=float(cp_prob),
                    model=(
                        cp_model.estimators_[0]
                        if cp_model is not None and hasattr(cp_model, "estimators_")
                        else cp_model
                    ),
                    input_df=cp_input_df,
                    feature_names=list(cp_features) if cp_features is not None else [],
                )
                st.session_state["copilot_result"] = result
            except Exception as cp_err:
                st.error(f"Copilot pipeline error: {cp_err}")
                result = None

    # ── Display results ───────────────────────────────────────────────────────
    result = st.session_state.get("copilot_result")
    if result:
        sar   = result.get("sar_draft", {})
        graph = result.get("graph_intel", {})
        cases = result.get("retrieved_cases", [])
        shap  = result.get("shap_factors", [])
        rings = result.get("fraud_rings", [])

        # ── Top-level summary metrics ──────────────────────────────────────────
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("SAR ID",            sar.get("sar_id", "—"))
        r2.metric("Status",            sar.get("status", "DRAFT"))
        r3.metric("Precedents found",  len(cases))
        r4.metric("Graph risk score",  f"{graph.get('graph_risk_score', 0):.4f}")
        r5.metric("Ring member",       "⚠️ Yes" if graph.get("ring_member") else "✅ No")

        # ── AI narrative ──────────────────────────────────────────────────────
        st.subheader("📝 AI-Drafted SAR Narrative")
        narrative = result.get("narrative", "")
        st.markdown(
            f'<div style="background:#1e1e2e; padding:16px; border-radius:8px; '
            f'border-left:4px solid #e74c3c; font-size:14px; line-height:1.7;">'
            f'{narrative.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Three-column details ───────────────────────────────────────────────
        col_shap, col_graph, col_prec = st.columns(3)

        with col_shap:
            st.markdown("**🔬 SHAP Feature Drivers**")
            if shap:
                shap_df = pd.DataFrame(shap)[["feature", "value", "shap_impact", "direction"]]
                shap_df["shap_impact"] = shap_df["shap_impact"].round(4)
                # Colour-code direction
                def _colour_direction(val):
                    return "color: #e74c3c" if val == "increases" else "color: #2ecc71"
                st.dataframe(
                    shap_df.style.map(_colour_direction, subset=["direction"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("SHAP not available — train a model first.")

        with col_graph:
            st.markdown("**🕸️ Graph Intelligence**")
            if graph:
                gi_items = {
                    "Graph risk score":    f"{graph.get('graph_risk_score', 0):.4f}",
                    "Ring member":         "Yes" if graph.get("ring_member") else "No",
                    "Rings matched":       graph.get("ring_count", 0),
                    "Max ring fraud rate": f"{graph.get('max_ring_fraud_rate', 0):.1%}",
                    "Customer degree":     graph.get("customer_degree", 0),
                    "Merchant degree":     graph.get("merchant_degree", 0),
                }
                st.table(pd.DataFrame.from_dict(gi_items, orient="index", columns=["Value"]))
                if rings:
                    with st.expander(f"Ring details ({len(rings)} rings)"):
                        for r in rings[:3]:
                            st.json({
                                "ring_id":    r.get("ring_id"),
                                "ring_type":  r.get("ring_type"),
                                "fraud_rate": r.get("fraud_rate"),
                                "risk_level": r.get("risk_level"),
                                "size":       r.get("size"),
                            })
            else:
                st.info("No graph data — run ring detection in the Graph Intel tab first.")

        with col_prec:
            st.markdown("**📂 Historical Precedents**")
            if cases:
                for i, hit in enumerate(cases[:5]):
                    src_icon = "📋" if hit["source"] == "sar_reports" else "🗂️"
                    with st.expander(
                        f"{src_icon} [{i+1}] {hit['id']}  ·  score={hit['score']:.3f}",
                        expanded=(i == 0),
                    ):
                        st.caption(f"Source: {hit['source']}")
                        st.markdown(hit["text"][:350] + ("…" if len(hit["text"]) > 350 else ""))
                        meta = hit.get("metadata", {})
                        if "fraud_probability" in meta:
                            st.caption(
                                f"Fraud prob: {meta['fraud_probability']:.4f}  |  "
                                f"Amount: ${meta.get('transaction_amount', meta.get('amount_usd', 0)):.2f}"
                            )
            else:
                st.info("No precedents found — index the audit log / SARs first.")

        st.divider()

        # ── HITL confirmation banner ───────────────────────────────────────────
        st.success(
            f"✅ SAR draft **{sar.get('sar_id', '—')}** has been routed to the "
            f"**HITL Review Queue** (see the 👤 HITL tab). "
            f"An analyst must approve or reject it before filing."
        )

        # ── Full SAR JSON ──────────────────────────────────────────────────────
        with st.expander("🗃️ Full SAR JSON (read-only draft)", expanded=False):
            # Strip non-serialisable keys before rendering
            safe_sar = {k: v for k, v in sar.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
            st.json(safe_sar)
