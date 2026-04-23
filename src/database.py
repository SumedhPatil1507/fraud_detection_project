"""
Supabase persistence layer.
Falls back to local CSV if Supabase is not configured.
"""
import os
import pandas as pd
from datetime import datetime
from src.config import AUDIT_LOG_PATH


def _get_client():
    try:
        from supabase import create_client
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None


def log_prediction(amount, distance, hour, is_foreign, is_new_device,
                   vpn, prob, is_fraud, threshold):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "transaction_amount": round(float(amount), 2),
        "distance_from_home_km": round(float(distance), 2),
        "hour": int(hour),
        "is_foreign": int(is_foreign),
        "is_new_device": int(is_new_device),
        "vpn_detected": int(vpn),
        "fraud_probability": round(float(prob), 4),
        "is_fraud": bool(is_fraud),
        "threshold_used": round(float(threshold), 3),
    }

    client = _get_client()
    if client:
        try:
            client.table("predictions").insert(record).execute()
            return "supabase"
        except Exception:
            pass

    # Fallback to CSV
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    df = pd.DataFrame([record])
    write_header = not os.path.exists(AUDIT_LOG_PATH)
    df.to_csv(AUDIT_LOG_PATH, mode='a', header=write_header, index=False)
    return "csv"


def load_predictions(limit=500) -> pd.DataFrame:
    client = _get_client()
    if client:
        try:
            res = (client.table("predictions")
                   .select("*")
                   .order("timestamp", desc=True)
                   .limit(limit)
                   .execute())
            if res.data:
                return pd.DataFrame(res.data)
        except Exception:
            pass

    # Fallback to CSV
    if os.path.exists(AUDIT_LOG_PATH):
        return pd.read_csv(AUDIT_LOG_PATH).sort_values(
            "timestamp", ascending=False).head(limit)
    return pd.DataFrame()


def get_stats() -> dict:
    client = _get_client()
    if client:
        try:
            total = client.table("predictions").select("id", count="exact").execute()
            fraud = client.table("predictions").select("id", count="exact")\
                          .eq("is_fraud", True).execute()
            return {
                "total": total.count or 0,
                "fraud": fraud.count or 0,
                "source": "supabase"
            }
        except Exception:
            pass

    if os.path.exists(AUDIT_LOG_PATH):
        df = pd.read_csv(AUDIT_LOG_PATH)
        return {"total": len(df), "fraud": int(df['is_fraud'].sum()), "source": "csv"}
    return {"total": 0, "fraud": 0, "source": "none"}
