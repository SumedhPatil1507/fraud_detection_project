import shap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_explainer(model, X_train_sample):
    """Build a SHAP TreeExplainer from the trained XGBoost model."""
    return shap.TreeExplainer(model)


def plot_shap_summary(model, X_sample):
    """SHAP summary bar plot — top feature importances by mean |SHAP|."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_shap_beeswarm(model, X_sample):
    """SHAP beeswarm — shows direction and magnitude of each feature."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def explain_single(model, input_df):
    """Return SHAP values + base value for a single prediction."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    return explainer.expected_value, shap_values[0]


def plot_waterfall(model, input_df):
    """Waterfall plot explaining a single prediction."""
    explainer = shap.TreeExplainer(model)
    explanation = explainer(input_df)
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(explanation[0], show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig
