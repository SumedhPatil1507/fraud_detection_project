"""
Shadow Mode Deployment
Runs a challenger model alongside the champion silently,
logs divergences without affecting live decisions.
"""
from __future__ import annotations
import os, pickle, json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.config import MODEL_DIR

SHADOW_LOG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "shadow_log.csv")
SHADOW_MODEL_PATH = os.path.join(MODEL_DIR, "shadow_model.pkl")


def load_shadow_model():
    if os.path.exists(SHADOW_MODEL_PATH):
        return pickle.load(open(SHADOW_MODEL_PATH, "rb"))
    return None


def shadow_predict(champion_model, input_df: pd.DataFrame, features: list,
                   champion_prob: float, threshold: float = 0.3) -> dict:
    """Score with shadow model and log divergence."""
    shadow = load_shadow_model()
    if shadow is None:
        return {"shadow_available": False}

    try:
        shadow_prob = float(shadow.predict_proba(input_df)[0][1])
    except Exception:
        return {"shadow_available": False, "error": "shadow predict failed"}

    champion_pred = int(champion_prob >= threshold)
    shadow_pred   = int(shadow_prob >= threshold)
    diverged      = champion_pred != shadow_pred

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "champion_prob": round(champion_prob, 4),
        "shadow_prob":   round(shadow_prob, 4),
        "champion_pred": champion_pred,
        "shadow_pred":   shadow_pred,
        "diverged":      diverged,
        "prob_delta":    round(abs(champion_prob - shadow_prob), 4),
    }

    os.makedirs(os.path.dirname(SHADOW_LOG), exist_ok=True)
    df = pd.DataFrame([record])
    write_header = not os.path.exists(SHADOW_LOG)
    df.to_csv(SHADOW_LOG, mode="a", header=write_header, index=False)

    return {**record, "shadow_available": True}


def load_shadow_log() -> pd.DataFrame:
    if not os.path.exists(SHADOW_LOG):
        return pd.DataFrame()
    return pd.read_csv(SHADOW_LOG)


def shadow_divergence_stats() -> dict:
    df = load_shadow_log()
    if df.empty:
        return {"total": 0, "diverged": 0, "divergence_rate": 0.0}
    return {
        "total": len(df),
        "diverged": int(df["diverged"].sum()),
        "divergence_rate": round(float(df["diverged"].mean()), 4),
        "mean_prob_delta": round(float(df["prob_delta"].mean()), 4),
        "shadow_higher_count": int((df["shadow_prob"] > df["champion_prob"]).sum()),
    }
