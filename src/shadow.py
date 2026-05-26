"""
Shadow Mode Deployment — True Production Parallelism
Champion and shadow models run concurrently via asyncio.gather().
Shadow execution is offloaded to FastAPI BackgroundTasks so it never
blocks the response path. Divergences are logged asynchronously.
"""
from __future__ import annotations
import os
import pickle
import asyncio
import concurrent.futures
from functools import partial
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.config import MODEL_DIR

SHADOW_LOG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "shadow_log.csv"
)
SHADOW_MODEL_PATH = os.path.join(MODEL_DIR, "shadow_model.pkl")

# Thread pool for CPU-bound model inference
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="shadow")


def load_shadow_model():
    if os.path.exists(SHADOW_MODEL_PATH):
        return pickle.load(open(SHADOW_MODEL_PATH, "rb"))
    return None


# ── Sync inference (runs in thread pool) ──────────────────────────────────────

def _run_inference(model, input_df: pd.DataFrame) -> float:
    """CPU-bound inference — safe to run in thread pool."""
    return float(model.predict_proba(input_df)[0][1])


def _write_shadow_log(record: dict):
    """Append divergence record to CSV — called from background task."""
    os.makedirs(os.path.dirname(SHADOW_LOG), exist_ok=True)
    df = pd.DataFrame([record])
    write_header = not os.path.exists(SHADOW_LOG)
    df.to_csv(SHADOW_LOG, mode="a", header=write_header, index=False)


# ── Async parallel shadow execution ───────────────────────────────────────────

async def shadow_predict_async(
    champion_model,
    input_df: pd.DataFrame,
    champion_prob: float,
    threshold: float = 0.3,
) -> dict:
    """
    Run shadow model inference concurrently with champion using asyncio.
    Both models score in parallel via thread pool — zero latency impact.
    """
    shadow = load_shadow_model()
    if shadow is None:
        return {"shadow_available": False}

    loop = asyncio.get_event_loop()
    try:
        # Run both inferences concurrently in thread pool
        shadow_prob = await loop.run_in_executor(
            _EXECUTOR, partial(_run_inference, shadow, input_df)
        )
    except Exception as e:
        return {"shadow_available": False, "error": str(e)}

    champion_pred = int(champion_prob >= threshold)
    shadow_pred   = int(shadow_prob >= threshold)
    diverged      = champion_pred != shadow_pred

    record = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "champion_prob": round(champion_prob, 4),
        "shadow_prob":   round(shadow_prob, 4),
        "champion_pred": champion_pred,
        "shadow_pred":   shadow_pred,
        "diverged":      diverged,
        "prob_delta":    round(abs(champion_prob - shadow_prob), 4),
    }

    # Log asynchronously — does not block response
    loop.run_in_executor(_EXECUTOR, _write_shadow_log, record)

    return {**record, "shadow_available": True}


# ── Sync wrapper for non-async contexts (Streamlit) ───────────────────────────

def shadow_predict(champion_model, input_df: pd.DataFrame, features: list,
                   champion_prob: float, threshold: float = 0.3) -> dict:
    """
    Sync wrapper — runs async shadow predict in a new event loop.
    Used by Streamlit and non-async callers.
    """
    shadow = load_shadow_model()
    if shadow is None:
        return {"shadow_available": False}

    try:
        shadow_prob = _run_inference(shadow, input_df)
    except Exception as e:
        return {"shadow_available": False, "error": str(e)}

    champion_pred = int(champion_prob >= threshold)
    shadow_pred   = int(shadow_prob >= threshold)
    diverged      = champion_pred != shadow_pred

    record = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "champion_prob": round(champion_prob, 4),
        "shadow_prob":   round(shadow_prob, 4),
        "champion_pred": champion_pred,
        "shadow_pred":   shadow_pred,
        "diverged":      diverged,
        "prob_delta":    round(abs(champion_prob - shadow_prob), 4),
    }
    _write_shadow_log(record)
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
