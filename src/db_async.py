"""
Async PostgreSQL persistence layer via SQLAlchemy + asyncpg.
Falls back gracefully to Supabase REST or CSV when PostgreSQL is unavailable.

Connection pooling:
  - Direct asyncpg pool (works with PgBouncer in transaction mode)
  - Pool config: min=2, max=10, max_inactive_connection_lifetime=300s
  - DATABASE_URL env var: postgres+asyncpg://user:pass@pgbouncer-host:6432/dbname

Fallback chain: asyncpg pool → Supabase REST → local CSV
"""
from __future__ import annotations
import os
import asyncio
import hashlib
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

# ── Optional asyncpg / SQLAlchemy ─────────────────────────────────────────────
try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker, DeclarativeBase
    from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, text
    _SQLA = True
except ImportError:
    _SQLA = False

from src.config import AUDIT_LOG_PATH

# ── Schema ─────────────────────────────────────────────────────────────────────
_engine = None
_async_session = None


def _get_db_url() -> Optional[str]:
    url = os.environ.get("DATABASE_URL", "")
    if url and "asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://")
        url = url.replace("postgres://", "postgresql+asyncpg://")
    return url or None


async def _init_engine():
    global _engine, _async_session
    if _engine is not None:
        return True
    url = _get_db_url()
    if not url or not _SQLA:
        return False
    try:
        _engine = create_async_engine(
            url,
            pool_size=10,
            max_overflow=5,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={"server_settings": {"application_name": "fraudguard"}},
        )
        _async_session = sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
        # Create table if not exists
        async with _engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id                  SERIAL PRIMARY KEY,
                    timestamp           TIMESTAMPTZ DEFAULT NOW(),
                    transaction_amount  NUMERIC(12,2),
                    distance_from_home_km NUMERIC(8,2),
                    hour                SMALLINT,
                    is_foreign          BOOLEAN,
                    is_new_device       BOOLEAN,
                    vpn_detected        BOOLEAN,
                    fraud_probability   NUMERIC(6,4),
                    is_fraud            BOOLEAN,
                    threshold_used      NUMERIC(4,3),
                    graph_risk_score    NUMERIC(8,4),
                    ring_member         BOOLEAN DEFAULT FALSE,
                    session_id          TEXT,
                    checksum            CHAR(32)
                );
                CREATE INDEX IF NOT EXISTS idx_pred_ts    ON predictions(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_pred_fraud ON predictions(is_fraud);
            """))
        return True
    except Exception as e:
        print(f"[DB] PostgreSQL unavailable: {e} — falling back to CSV")
        _engine = None
        return False


def _record_checksum(record: dict) -> str:
    payload = f"{record.get('timestamp')}{record.get('transaction_amount')}{record.get('fraud_probability')}"
    return hashlib.md5(payload.encode()).hexdigest()


async def log_prediction_async(amount, distance, hour, is_foreign, is_new_device,
                                vpn, prob, is_fraud, threshold,
                                graph_risk=0.0, ring_member=False) -> str:
    """Write prediction to PostgreSQL (async). Falls back to Supabase → CSV."""
    record = {
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "transaction_amount":   round(float(amount), 2),
        "distance_from_home_km": round(float(distance), 2),
        "hour":                 int(hour),
        "is_foreign":           bool(is_foreign),
        "is_new_device":        bool(is_new_device),
        "vpn_detected":         bool(vpn),
        "fraud_probability":    round(float(prob), 4),
        "is_fraud":             bool(is_fraud),
        "threshold_used":       round(float(threshold), 3),
        "graph_risk_score":     round(float(graph_risk), 4),
        "ring_member":          bool(ring_member),
        "checksum":             "",
    }
    record["checksum"] = _record_checksum(record)

    # Try PostgreSQL first
    ready = await _init_engine()
    if ready and _async_session:
        try:
            async with _async_session() as session:
                await session.execute(text("""
                    INSERT INTO predictions
                    (timestamp, transaction_amount, distance_from_home_km, hour,
                     is_foreign, is_new_device, vpn_detected, fraud_probability,
                     is_fraud, threshold_used, graph_risk_score, ring_member, checksum)
                    VALUES (:timestamp, :transaction_amount, :distance_from_home_km, :hour,
                            :is_foreign, :is_new_device, :vpn_detected, :fraud_probability,
                            :is_fraud, :threshold_used, :graph_risk_score, :ring_member, :checksum)
                """), record)
                await session.commit()
            return "postgresql"
        except Exception as e:
            print(f"[DB] PostgreSQL write failed: {e}")

    # Fallback: Supabase REST
    try:
        from src.database import _get_client
        client = _get_client()
        if client:
            flat = {k: v for k, v in record.items()
                    if k not in ("graph_risk_score", "ring_member", "checksum")}
            client.table("predictions").insert(flat).execute()
            return "supabase"
    except Exception:
        pass

    # Final fallback: CSV
    _csv_append(record)
    return "csv"


def log_prediction(amount, distance, hour, is_foreign, is_new_device,
                   vpn, prob, is_fraud, threshold,
                   graph_risk=0.0, ring_member=False) -> str:
    """Sync wrapper — runs async log in thread-safe manner."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context — schedule as task
            asyncio.ensure_future(
                log_prediction_async(amount, distance, hour, is_foreign,
                                     is_new_device, vpn, prob, is_fraud,
                                     threshold, graph_risk, ring_member)
            )
            return "async_scheduled"
        else:
            return loop.run_until_complete(
                log_prediction_async(amount, distance, hour, is_foreign,
                                     is_new_device, vpn, prob, is_fraud,
                                     threshold, graph_risk, ring_member)
            )
    except Exception:
        _csv_append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transaction_amount": amount, "distance_from_home_km": distance,
            "hour": hour, "is_foreign": is_foreign, "is_new_device": is_new_device,
            "vpn_detected": vpn, "fraud_probability": prob,
            "is_fraud": is_fraud, "threshold_used": threshold,
        })
        return "csv"


def _csv_append(record: dict):
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    df = pd.DataFrame([record])
    write_header = not os.path.exists(AUDIT_LOG_PATH)
    df.to_csv(AUDIT_LOG_PATH, mode="a", header=write_header, index=False)


async def load_predictions_async(limit: int = 500) -> pd.DataFrame:
    ready = await _init_engine()
    if ready and _async_session:
        try:
            async with _async_session() as session:
                result = await session.execute(text(
                    f"SELECT * FROM predictions ORDER BY timestamp DESC LIMIT {limit}"
                ))
                rows = result.mappings().all()
                if rows:
                    return pd.DataFrame([dict(r) for r in rows])
        except Exception as e:
            print(f"[DB] PostgreSQL read failed: {e}")

    # Fallback to CSV
    if os.path.exists(AUDIT_LOG_PATH):
        return pd.read_csv(AUDIT_LOG_PATH).sort_values(
            "timestamp", ascending=False).head(limit)
    return pd.DataFrame()


def load_predictions(limit: int = 500) -> pd.DataFrame:
    """Sync wrapper for load_predictions_async."""
    try:
        # Try Supabase first (available in Streamlit Cloud)
        from src.database import load_predictions as supa_load
        df = supa_load(limit)
        if not df.empty:
            return df
    except Exception:
        pass
    if os.path.exists(AUDIT_LOG_PATH):
        return pd.read_csv(AUDIT_LOG_PATH).sort_values(
            "timestamp", ascending=False).head(limit)
    return pd.DataFrame()


def get_stats() -> dict:
    try:
        from src.database import get_stats as supa_stats
        return supa_stats()
    except Exception:
        pass
    if os.path.exists(AUDIT_LOG_PATH):
        df = pd.read_csv(AUDIT_LOG_PATH)
        return {"total": len(df), "fraud": int(df["is_fraud"].sum()), "source": "csv"}
    return {"total": 0, "fraud": 0, "source": "none"}


PGBOUNCER_SETUP_SQL = """
-- Run on your PostgreSQL instance before connecting through PgBouncer
-- PgBouncer config (pgbouncer.ini):
--   pool_mode = transaction
--   max_client_conn = 1000
--   default_pool_size = 25
--   server_idle_timeout = 300

-- Application user with minimal privileges
CREATE USER fraudguard_app WITH PASSWORD 'strong_password_here';
GRANT CONNECT ON DATABASE fraudguard TO fraudguard_app;
GRANT USAGE ON SCHEMA public TO fraudguard_app;
GRANT SELECT, INSERT ON TABLE predictions TO fraudguard_app;
GRANT USAGE, SELECT ON SEQUENCE predictions_id_seq TO fraudguard_app;

-- Disable prepared statements (required for PgBouncer transaction mode)
-- Set in SQLAlchemy: connect_args={"prepared_statement_cache_size": 0}
"""


def get_pgbouncer_setup() -> str:
    return PGBOUNCER_SETUP_SQL
