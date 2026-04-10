"""
Model drift detection using Population Stability Index (PSI).
PSI < 0.1  → no drift
PSI 0.1–0.2 → moderate drift, monitor
PSI > 0.2  → significant drift, consider retraining
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import TRAIN_DIST_PATH


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Compute PSI between expected (training) and actual (new) distributions."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    def bucket_counts(arr):
        counts, _ = np.histogram(arr, bins=breakpoints)
        counts = np.where(counts == 0, 1e-6, counts)
        return counts / counts.sum()

    exp_pct = bucket_counts(expected)
    act_pct = bucket_counts(actual)
    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct + 1e-10))
    return float(psi)


def detect_drift(X_new: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Compare X_new feature distributions against saved training distributions.
    Returns per-feature PSI and an overall drift flag.
    """
    if not __import__('os').path.exists(TRAIN_DIST_PATH):
        return {"error": "No training distribution saved. Train the model first."}

    train_dist = pickle.load(open(TRAIN_DIST_PATH, "rb"))
    results = {}

    for col in X_new.select_dtypes(include='number').columns:
        if col not in train_dist:
            continue
        mean = train_dist[col]["mean"]
        std  = train_dist[col]["std"]
        # Reconstruct approximate training distribution via normal sampling
        rng = np.random.default_rng(42)
        expected = rng.normal(mean, std, size=1000)
        actual = X_new[col].dropna().values
        if len(actual) < 5:
            continue
        results[col] = round(compute_psi(expected, actual), 4)

    if not results:
        return {"error": "No numeric features to compare."}

    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    top_features = dict(list(sorted_results.items())[:top_n])
    overall_psi = float(np.mean(list(results.values())))

    return {
        "overall_psi": round(overall_psi, 4),
        "drift_detected": overall_psi > 0.2,
        "drift_level": "HIGH" if overall_psi > 0.2 else "MODERATE" if overall_psi > 0.1 else "LOW",
        "top_drifted_features": top_features,
        "retrain_recommended": overall_psi > 0.2,
    }


def plot_drift(drift_result: dict):
    """Bar chart of PSI per feature."""
    features = drift_result.get("top_drifted_features", {})
    if not features:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e74c3c" if v > 0.2 else "#e67e22" if v > 0.1 else "#2ecc71"
              for v in features.values()]
    ax.barh(list(features.keys()), list(features.values()), color=colors)
    ax.axvline(0.1, color='orange', linestyle='--', lw=1.5, label='Moderate (0.1)')
    ax.axvline(0.2, color='red', linestyle='--', lw=1.5, label='High (0.2)')
    ax.set_xlabel("PSI")
    ax.set_title("Feature Drift — Population Stability Index")
    ax.legend()
    fig.tight_layout()
    return fig
