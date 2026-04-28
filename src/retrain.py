"""
Automated Retraining Pipeline
Detects drift, triggers retraining, versions models, notifies via webhook.
"""
from __future__ import annotations
import os, pickle, json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.config import MODEL_DIR, TRAIN_DIST_PATH
from src.drift import detect_drift
from src.pipeline import run_pipeline
from src.model import train_model

RETRAIN_LOG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "retrain_log.json")


def should_retrain(X_new: pd.DataFrame, psi_threshold: float = 0.2) -> tuple[bool, dict]:
    """Check drift and return (should_retrain, drift_result)."""
    drift = detect_drift(X_new)
    if "error" in drift:
        return False, drift
    return drift.get("retrain_recommended", False), drift


def run_retraining(df: pd.DataFrame, reason: str = "scheduled",
                   use_optuna: bool = False) -> dict:
    """Full retrain cycle — train, save, log."""
    start = datetime.now(timezone.utc)
    try:
        result = train_model(df, use_optuna=use_optuna)
        model, X_test, y_test, probs, threshold, roc, report, cv_scores, _, best_params = result
        status = "success"
        metrics = {
            "roc_auc": round(roc, 4),
            "cv_mean_auc": round(float(cv_scores.mean()), 4),
            "threshold": round(threshold, 4),
            "fraud_precision": round(report.get("Fraud", {}).get("precision", 0), 4),
            "fraud_recall": round(report.get("Fraud", {}).get("recall", 0), 4),
        }
        error = None
    except Exception as e:
        status = "failed"
        metrics = {}
        error = str(e)

    end = datetime.now(timezone.utc)
    log_entry = {
        "run_id": f"retrain-{start.strftime('%Y%m%d-%H%M%S')}",
        "triggered_at": start.isoformat(),
        "completed_at": end.isoformat(),
        "duration_seconds": round((end - start).total_seconds(), 1),
        "reason": reason,
        "status": status,
        "metrics": metrics,
        "error": error,
    }

    _append_retrain_log(log_entry)
    _notify_webhook(log_entry)
    return log_entry


def _append_retrain_log(entry: dict):
    os.makedirs(os.path.dirname(RETRAIN_LOG), exist_ok=True)
    history = []
    if os.path.exists(RETRAIN_LOG):
        try:
            with open(RETRAIN_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(entry)
    with open(RETRAIN_LOG, "w") as f:
        json.dump(history[-50:], f, indent=2)  # keep last 50


def load_retrain_log() -> list:
    if not os.path.exists(RETRAIN_LOG):
        return []
    try:
        with open(RETRAIN_LOG) as f:
            return json.load(f)
    except Exception:
        return []


def _notify_webhook(entry: dict):
    webhook_url = os.environ.get("RETRAIN_WEBHOOK_URL", "")
    if not webhook_url:
        return
    try:
        import requests
        requests.post(webhook_url, json=entry, timeout=5)
    except Exception:
        pass
