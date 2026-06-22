"""
Real-Time Infrastructure Observability — Prometheus + Grafana
Instruments:
  - API p95/p99 latency histograms
  - DB pool connection gauge
  - Celery queue depth gauge
  - Live PSI drift tracking (concept drift)
  - TP/FP ratio gauge (model performance)
  - Prediction throughput counter

Graceful fallback: if prometheus_client is not installed,
all metrics are no-ops that don't break the app.
"""
from __future__ import annotations
import time
import os
import functools
from typing import Callable, Any
import numpy as np
import pandas as pd

# ── Optional prometheus_client ─────────────────────────────────────────────────
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server, multiprocess,
    )
    _PROM = True
except ImportError:
    _PROM = False


# ── Metric definitions ─────────────────────────────────────────────────────────

class _NoOpMetric:
    """Silent no-op for when prometheus_client is not installed."""
    def labels(self, **_):   return self
    def observe(self, *_):   pass
    def inc(self, *_):       pass
    def set(self, *_):       pass
    def time(self):
        import contextlib
        return contextlib.nullcontext()


def _metric(metric_class, *args, **kwargs):
    if _PROM:
        try:
            return metric_class(*args, **kwargs)
        except Exception:
            return _NoOpMetric()
    return _NoOpMetric()


# API metrics
API_LATENCY = _metric(
    Histogram if _PROM else None,
    "fraudguard_api_latency_seconds",
    "API endpoint latency",
    ["endpoint", "method", "status"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
) if _PROM else _NoOpMetric()

API_REQUESTS = _metric(
    Counter if _PROM else None,
    "fraudguard_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
) if _PROM else _NoOpMetric()

# Database metrics
DB_POOL_ACTIVE = _metric(
    Gauge if _PROM else None,
    "fraudguard_db_pool_active_connections",
    "Active database pool connections",
) if _PROM else _NoOpMetric()

DB_POOL_IDLE = _metric(
    Gauge if _PROM else None,
    "fraudguard_db_pool_idle_connections",
    "Idle database pool connections",
) if _PROM else _NoOpMetric()

DB_QUERY_LATENCY = _metric(
    Histogram if _PROM else None,
    "fraudguard_db_query_latency_seconds",
    "Database query latency",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
) if _PROM else _NoOpMetric()

# Celery queue metrics
CELERY_QUEUE_DEPTH = _metric(
    Gauge if _PROM else None,
    "fraudguard_celery_queue_depth",
    "Celery task queue depth",
    ["queue"],
) if _PROM else _NoOpMetric()

# Model performance metrics
FRAUD_PREDICTIONS = _metric(
    Counter if _PROM else None,
    "fraudguard_predictions_total",
    "Total predictions made",
    ["verdict"],
) if _PROM else _NoOpMetric()

MODEL_SCORE = _metric(
    Histogram if _PROM else None,
    "fraudguard_fraud_probability",
    "Distribution of fraud probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
) if _PROM else _NoOpMetric()

PSI_GAUGE = _metric(
    Gauge if _PROM else None,
    "fraudguard_psi_score",
    "Population Stability Index per feature",
    ["feature"],
) if _PROM else _NoOpMetric()

TP_RATE = _metric(
    Gauge if _PROM else None,
    "fraudguard_true_positive_rate",
    "Live true positive rate",
) if _PROM else _NoOpMetric()

FP_RATE_GAUGE = _metric(
    Gauge if _PROM else None,
    "fraudguard_false_positive_rate",
    "Live false positive rate",
) if _PROM else _NoOpMetric()


# ── Decorators ─────────────────────────────────────────────────────────────────

def track_latency(endpoint: str, method: str = "POST"):
    """Decorator to track API endpoint latency."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            status = "200"
            try:
                result = await fn(*args, **kwargs)
                return result
            except Exception as e:
                status = "500"
                raise
            finally:
                elapsed = time.perf_counter() - start
                try:
                    API_LATENCY.labels(endpoint=endpoint, method=method, status=status).observe(elapsed)
                    API_REQUESTS.labels(endpoint=endpoint, method=method, status=status).inc()
                except Exception:
                    pass
        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            status = "200"
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception:
                status = "500"
                raise
            finally:
                elapsed = time.perf_counter() - start
                try:
                    API_LATENCY.labels(endpoint=endpoint, method=method, status=status).observe(elapsed)
                    API_REQUESTS.labels(endpoint=endpoint, method=method, status=status).inc()
                except Exception:
                    pass
        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper
    return decorator


def record_prediction(prob: float, is_fraud: bool):
    """Record a prediction event to Prometheus."""
    try:
        MODEL_SCORE.observe(prob)
        FRAUD_PREDICTIONS.labels(verdict="fraud" if is_fraud else "legit").inc()
    except Exception:
        pass


def update_psi_metrics(drift_result: dict):
    """Push PSI scores for top drifted features to Prometheus."""
    try:
        for feat, psi in drift_result.get("top_drifted_features", {}).items():
            PSI_GAUGE.labels(feature=feat).set(psi)
    except Exception:
        pass


def update_model_performance(tp_rate: float, fp_rate: float):
    try:
        TP_RATE.set(tp_rate)
        FP_RATE_GAUGE.set(fp_rate)
    except Exception:
        pass


def update_db_pool(active: int, idle: int):
    try:
        DB_POOL_ACTIVE.set(active)
        DB_POOL_IDLE.set(idle)
    except Exception:
        pass


def start_metrics_server(port: int = 9090):
    """Start Prometheus metrics HTTP server (production use)."""
    if not _PROM:
        print("[Observability] prometheus_client not installed — metrics disabled")
        return
    try:
        start_http_server(port)
        print(f"[Observability] Prometheus metrics at http://0.0.0.0:{port}/metrics")
    except Exception as e:
        print(f"[Observability] Failed to start metrics server: {e}")


# ── Streamlit Dashboard Metrics (no Prometheus needed) ────────────────────────

def get_live_metrics(audit_df: pd.DataFrame) -> dict:
    """
    Compute live model performance metrics from audit log.
    Used in Streamlit observability tab — no Prometheus required.
    """
    if audit_df.empty:
        return {
            "total_predictions": 0,
            "fraud_rate": 0.0,
            "avg_probability": 0.0,
            "p95_probability": 0.0,
            "p99_probability": 0.0,
        }

    probs = audit_df["fraud_probability"].dropna()
    frauds = audit_df["is_fraud"].astype(bool)

    return {
        "total_predictions": len(audit_df),
        "fraud_rate":        round(float(frauds.mean()), 4),
        "avg_probability":   round(float(probs.mean()), 4),
        "p95_probability":   round(float(probs.quantile(0.95)), 4),
        "p99_probability":   round(float(probs.quantile(0.99)), 4),
        "last_hour_count":   _last_n_minutes(audit_df, 60),
        "last_day_count":    _last_n_minutes(audit_df, 1440),
    }


def _last_n_minutes(df: pd.DataFrame, minutes: int) -> int:
    try:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes)
        return int((ts >= cutoff).sum())
    except Exception:
        return 0


def compute_live_psi(audit_df: pd.DataFrame, drift_result: dict) -> dict:
    """Returns PSI data formatted for Streamlit charts."""
    features = drift_result.get("top_drifted_features", {})
    return {
        "features": list(features.keys()),
        "psi_values": list(features.values()),
        "overall_psi": drift_result.get("overall_psi", 0),
        "drift_level": drift_result.get("drift_level", "UNKNOWN"),
    }


GRAFANA_DASHBOARD_JSON = {
    "title": "FraudGuard AI — Operations",
    "panels": [
        {"title": "API p95 Latency", "type": "graph",
         "targets": [{"expr": 'histogram_quantile(0.95, rate(fraudguard_api_latency_seconds_bucket[5m]))'}]},
        {"title": "API p99 Latency", "type": "graph",
         "targets": [{"expr": 'histogram_quantile(0.99, rate(fraudguard_api_latency_seconds_bucket[5m]))'}]},
        {"title": "Fraud Rate (5m)", "type": "stat",
         "targets": [{"expr": 'rate(fraudguard_predictions_total{verdict="fraud"}[5m])'}]},
        {"title": "DB Pool Active", "type": "gauge",
         "targets": [{"expr": "fraudguard_db_pool_active_connections"}]},
        {"title": "PSI Drift Score", "type": "graph",
         "targets": [{"expr": 'fraudguard_psi_score'}]},
        {"title": "Celery Queue Depth", "type": "graph",
         "targets": [{"expr": 'fraudguard_celery_queue_depth'}]},
    ]
}
