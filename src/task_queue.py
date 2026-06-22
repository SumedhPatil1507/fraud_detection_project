"""
Async Task Queue — Celery + Redis with graceful fallback.
Offloads heavy workloads: retraining, SAR batch generation,
drift analysis, and bulk predictions from the API request path.

Setup (production):
  1. Install: pip install celery[redis] redis
  2. Start worker: celery -A src.task_queue worker --loglevel=info -Q fraud_tasks
  3. Start beat:   celery -A src.task_queue beat --loglevel=info
  4. Set env:      REDIS_URL=redis://localhost:6379/0

Fallback: If Redis is unavailable, tasks run synchronously inline.
"""
from __future__ import annotations
import os
import logging
from typing import Any, Callable
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# ── Celery app ─────────────────────────────────────────────────────────────────
try:
    from celery import Celery
    from celery.utils.log import get_task_logger

    celery_app = Celery(
        "fraudguard",
        broker=REDIS_URL,
        backend=REDIS_URL,
        include=["src.task_queue"],
    )

    celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,  # fair dispatch
        task_routes={
            "src.task_queue.retrain_task":      {"queue": "fraud_tasks"},
            "src.task_queue.drift_check_task":  {"queue": "fraud_tasks"},
            "src.task_queue.sar_batch_task":    {"queue": "fraud_tasks"},
            "src.task_queue.batch_score_task":  {"queue": "fraud_tasks"},
        },
        beat_schedule={
            "drift-check-hourly": {
                "task": "src.task_queue.drift_check_task",
                "schedule": 3600.0,
            },
        },
    )
    _CELERY = True
    task_logger = get_task_logger(__name__)

except ImportError:
    celery_app = None
    _CELERY = False
    task_logger = logger
    logger.warning("[TaskQueue] Celery/Redis not installed — tasks run synchronously")


# ── Task definitions ───────────────────────────────────────────────────────────

def _define_tasks():
    if not _CELERY or celery_app is None:
        return

    @celery_app.task(bind=True, name="src.task_queue.retrain_task",
                     max_retries=2, default_retry_delay=60)
    def retrain_task(self, reason: str = "scheduled"):
        """Background model retraining — triggered by drift or schedule."""
        task_logger.info(f"[retrain_task] Starting: reason={reason}")
        try:
            from src.pipeline import run_pipeline
            from src.retrain import run_retraining
            df, _ = run_pipeline()
            result = run_retraining(df, reason=reason)
            task_logger.info(f"[retrain_task] Done: {result}")
            return result
        except Exception as exc:
            task_logger.error(f"[retrain_task] Failed: {exc}")
            raise self.retry(exc=exc)

    @celery_app.task(name="src.task_queue.drift_check_task")
    def drift_check_task():
        """Periodic drift check — auto-triggers retraining if PSI > 0.2."""
        task_logger.info("[drift_check_task] Running PSI drift check")
        try:
            from src.pipeline import run_pipeline
            from src.drift import detect_drift
            df, _ = run_pipeline()
            X = df.select_dtypes(include="number").drop(
                columns=["label", "financial_loss"], errors="ignore"
            )
            result = detect_drift(X)
            task_logger.info(f"[drift_check_task] PSI={result.get('overall_psi')}")
            if result.get("retrain_recommended"):
                retrain_task.delay(reason="auto_drift_psi")
            return result
        except Exception as e:
            task_logger.error(f"[drift_check_task] Failed: {e}")

    @celery_app.task(name="src.task_queue.sar_batch_task")
    def sar_batch_task(transactions: list[dict]):
        """Generate SAR reports for a batch of flagged transactions."""
        task_logger.info(f"[sar_batch_task] Processing {len(transactions)} SARs")
        results = []
        try:
            from src.sar import generate_sar
            for txn in transactions:
                sar = generate_sar(
                    txn, txn.get("fraud_probability", 0.5), [], threshold=0.3
                )
                results.append(sar["sar_id"])
        except Exception as e:
            task_logger.error(f"[sar_batch_task] Failed: {e}")
        return results

    @celery_app.task(name="src.task_queue.batch_score_task")
    def batch_score_task(transactions: list[dict]) -> list[dict]:
        """Score a large batch of transactions outside the API request path."""
        task_logger.info(f"[batch_score_task] Scoring {len(transactions)} transactions")
        import pickle
        from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH
        import numpy as np
        import pandas as pd

        try:
            model    = pickle.load(open(MODEL_PATH, "rb"))
            features = pickle.load(open(FEATURE_PATH, "rb"))
            means    = pickle.load(open(MEAN_PATH, "rb"))
        except Exception as e:
            return [{"error": f"Model load failed: {e}"}]

        results = []
        for txn in transactions:
            try:
                d = means.to_dict()
                d.update({k: v for k, v in txn.items()})
                d["amount_log"] = np.log1p(d.get("transaction_amount", 0))
                input_df = pd.DataFrame([d])[features]
                prob = float(model.predict_proba(input_df)[0][1])
                results.append({
                    "fraud_probability": round(prob, 4),
                    "is_fraud": prob >= 0.3,
                    "risk_level": "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW",
                })
            except Exception as e:
                results.append({"error": str(e)})
        return results


_define_tasks()


# ── Public API — works with or without Celery ─────────────────────────────────

def submit_retrain(reason: str = "manual") -> dict:
    if _CELERY and celery_app:
        try:
            from src.task_queue import retrain_task
            task = retrain_task.delay(reason=reason)
            return {"task_id": task.id, "status": "queued", "backend": "celery"}
        except Exception as e:
            logger.warning(f"[TaskQueue] Celery submit failed ({e}) — running sync")

    # Sync fallback
    from src.pipeline import run_pipeline
    from src.retrain import run_retraining
    df, _ = run_pipeline()
    result = run_retraining(df, reason=reason)
    return {**result, "backend": "sync"}


def submit_drift_check() -> dict:
    if _CELERY and celery_app:
        try:
            from src.task_queue import drift_check_task
            task = drift_check_task.delay()
            return {"task_id": task.id, "status": "queued", "backend": "celery"}
        except Exception as e:
            logger.warning(f"[TaskQueue] Celery submit failed ({e}) — running sync")

    from src.pipeline import run_pipeline
    from src.drift import detect_drift
    df, _ = run_pipeline()
    X = df.select_dtypes(include="number").drop(
        columns=["label", "financial_loss"], errors="ignore"
    )
    return {**detect_drift(X), "backend": "sync"}


def get_task_status(task_id: str) -> dict:
    if not _CELERY or celery_app is None:
        return {"status": "celery_unavailable"}
    try:
        from celery.result import AsyncResult
        result = AsyncResult(task_id, app=celery_app)
        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def get_queue_stats() -> dict:
    if not _CELERY or celery_app is None:
        return {"celery_available": False, "redis_url": REDIS_URL}
    try:
        inspect = celery_app.control.inspect()
        active  = inspect.active() or {}
        reserved = inspect.reserved() or {}
        return {
            "celery_available": True,
            "redis_url":        REDIS_URL,
            "active_tasks":     sum(len(v) for v in active.values()),
            "queued_tasks":     sum(len(v) for v in reserved.values()),
            "workers":          list(active.keys()),
        }
    except Exception as e:
        return {"celery_available": False, "error": str(e)}
