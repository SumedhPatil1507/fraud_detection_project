"""
SHAP Utility Plots — fully interactive via Plotly.

All three public functions return a plotly.graph_objects.Figure so they can be
rendered with st.plotly_chart() without requiring matplotlib.  The functions
handle both the old ndarray API (shap < 0.46) and the new Explanation-object
API (shap >= 0.46) transparently.

Public API
----------
plot_shap_summary(model, X_sample)       -> go.Figure  (bar: mean |SHAP|)
plot_shap_beeswarm(model, X_sample)      -> go.Figure  (scatter beeswarm)
plot_waterfall(model, input_df)          -> go.Figure  (waterfall single row)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    import shap as _shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False

try:
    import streamlit as st
    _ST = True
except ImportError:
    _ST = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unavailable_fig(msg: str = "SHAP unavailable — install shap>=0.46.0") -> go.Figure:
    """Return a simple Plotly figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#e74c3c"),
    )
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _get_explainer(model):
    """Build a TreeExplainer, cached when running inside Streamlit."""
    return _shap.TreeExplainer(model)


if _ST:
    _get_explainer = st.cache_resource(show_spinner=False)(_get_explainer)


def _extract_shap_array(raw) -> np.ndarray:
    """
    Normalise shap output to a plain 2-D numpy array (samples × features).

    Handles:
    - shap.Explanation objects  (shap >= 0.46)
    - list [neg_class, pos_class]  (binary TreeExplainer old API)
    - plain ndarray
    """
    if hasattr(raw, 'values'):          # Explanation object
        raw = raw.values
    if isinstance(raw, list):           # binary classifier list
        raw = raw[1]
    raw = np.array(raw)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    return raw


# ── Public plot functions ─────────────────────────────────────────────────────

def plot_shap_summary(model, X_sample: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Interactive bar chart of mean |SHAP| per feature (global importance).
    Replaces the old matplotlib summary_plot.
    """
    if not _SHAP_OK:
        return _unavailable_fig()
    try:
        explainer  = _get_explainer(model)
        raw        = explainer.shap_values(X_sample)
        sv         = _extract_shap_array(raw)
        feat_names = list(X_sample.columns)

        mean_abs = np.abs(sv).mean(axis=0)
        idx      = np.argsort(mean_abs)[-top_n:]

        fig = go.Figure(go.Bar(
            x=mean_abs[idx],
            y=[feat_names[i] for i in idx],
            orientation='h',
            marker=dict(
                color=mean_abs[idx],
                colorscale='Purples',
                showscale=True,
                colorbar=dict(title="Mean |SHAP|"),
            ),
            hovertemplate=(
                "<b>%{y}</b><br>Mean |SHAP| = %{x:.4f}<extra></extra>"
            ),
        ))
        fig.update_layout(
            title=f"Global SHAP Feature Importance (Top {top_n})",
            xaxis_title="Mean |SHAP Value|",
            height=max(400, top_n * 28),
            margin=dict(l=160),
        )
        return fig
    except Exception as e:
        return _unavailable_fig(f"SHAP error: {e}")


def plot_shap_beeswarm(model, X_sample: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Interactive scatter beeswarm: one point per sample per feature,
    x-axis = SHAP value, colour = feature value.
    Replaces the old matplotlib beeswarm summary_plot.
    """
    if not _SHAP_OK:
        return _unavailable_fig()
    try:
        explainer  = _get_explainer(model)
        raw        = explainer.shap_values(X_sample)
        sv         = _extract_shap_array(raw)
        feat_names = list(X_sample.columns)

        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[-top_n:][::-1]

        rows = []
        for fi in top_idx:
            fname = feat_names[fi]
            for si in range(len(X_sample)):
                rows.append({
                    "Feature":        fname,
                    "SHAP Value":     float(sv[si, fi]),
                    "Feature Value":  float(X_sample.iloc[si, fi]),
                })
        plot_df = pd.DataFrame(rows)

        fig = px.strip(
            plot_df,
            x="SHAP Value",
            y="Feature",
            color="Feature Value",
            color_continuous_scale="RdBu_r",
            orientation="h",
            title=f"SHAP Beeswarm (Top {top_n} Features)",
            hover_data=["Feature Value"],
        )
        fig.update_traces(marker=dict(size=4, opacity=0.7))
        fig.update_layout(
            height=max(450, top_n * 32),
            margin=dict(l=160),
            coloraxis_colorbar=dict(title="Feature Value"),
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        return fig
    except Exception as e:
        return _unavailable_fig(f"SHAP error: {e}")


def plot_waterfall(model, input_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Interactive Plotly waterfall chart for a single prediction row.
    Shows how each feature pushes the score above or below the base value.
    Replaces the old matplotlib waterfall.
    """
    if not _SHAP_OK:
        return _unavailable_fig()
    try:
        explainer   = _get_explainer(model)
        raw         = explainer.shap_values(input_df)
        sv          = _extract_shap_array(raw)[0]   # single row → 1-D
        feat_names  = list(input_df.columns)

        # Base value (expected model output)
        try:
            base_val = float(explainer.expected_value)
        except Exception:
            base_val = 0.0

        # Pick top_n features by |SHAP|
        idx = np.argsort(np.abs(sv))[-top_n:][::-1]

        labels   = [f"{feat_names[i]}\n= {input_df.iloc[0, i]:.3g}" for i in idx]
        values   = [float(sv[i]) for i in idx]
        colours  = ['#e74c3c' if v > 0 else '#2ecc71' for v in values]

        fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"] * len(values) + ["total"],
            x=values + [sum(values)],
            y=labels + ["Prediction"],
            connector=dict(line=dict(color="rgb(63,63,63)", width=0.5)),
            increasing=dict(marker_color="#e74c3c"),
            decreasing=dict(marker_color="#2ecc71"),
            totals=dict(marker_color="#3498db"),
            hovertemplate="<b>%{y}</b><br>SHAP = %{x:.4f}<extra></extra>",
        ))
        fig.add_vline(x=base_val, line_dash="dot", line_color="gray",
                      annotation_text=f"Base = {base_val:.3f}",
                      annotation_position="top right")
        fig.update_layout(
            title=f"SHAP Waterfall — Single Prediction (Top {top_n} Features)",
            xaxis_title="SHAP Value",
            height=max(450, top_n * 32),
            margin=dict(l=220),
            waterfallgap=0.4,
        )
        return fig
    except Exception as e:
        return _unavailable_fig(f"SHAP error: {e}")
