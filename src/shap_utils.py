import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    _SHAP_OK = True
except Exception:
    _SHAP_OK = False


def _no_shap_fig(msg="SHAP unavailable"):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12)
    ax.axis('off')
    return fig


def plot_shap_summary(model, X_sample):
    if not _SHAP_OK:
        return _no_shap_fig()
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        fig = plt.gcf()
        fig.tight_layout()
        return fig
    except Exception as e:
        return _no_shap_fig(f"SHAP error: {e}")


def plot_shap_beeswarm(model, X_sample):
    if not _SHAP_OK:
        return _no_shap_fig()
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        return fig
    except Exception as e:
        return _no_shap_fig(f"SHAP error: {e}")


def plot_waterfall(model, input_df):
    if not _SHAP_OK:
        return _no_shap_fig()
    try:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(input_df)
        shap.plots.waterfall(explanation[0], show=False)
        fig = plt.gcf()
        fig.tight_layout()
        return fig
    except Exception as e:
        return _no_shap_fig(f"SHAP error: {e}")
