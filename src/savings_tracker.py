"""
Fraud Savings Tracker
Accumulates real-time savings from caught fraud across sessions.
Persists to CSV so totals survive restarts.
"""
from __future__ import annotations
import os, json
import pandas as pd
from datetime import datetime, timezone

SAVINGS_LOG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "savings_log.csv"
)


def record_catch(
    transaction_amount: float,
    fraud_probability: float,
    analyst_confirmed: bool = False,
    fn_cost: float = 5000,
) -> dict:
    """Log a caught fraud event and return running totals."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transaction_amount": round(float(transaction_amount), 2),
        "fraud_probability": round(float(fraud_probability), 4),
        "analyst_confirmed": analyst_confirmed,
        "estimated_saving_usd": round(float(fn_cost), 2),
    }
    os.makedirs(os.path.dirname(SAVINGS_LOG), exist_ok=True)
    df = pd.DataFrame([record])
    write_header = not os.path.exists(SAVINGS_LOG)
    df.to_csv(SAVINGS_LOG, mode="a", header=write_header, index=False)
    return record


def get_savings_summary() -> dict:
    if not os.path.exists(SAVINGS_LOG):
        return {"total_caught": 0, "total_savings_usd": 0.0,
                "confirmed_savings_usd": 0.0, "avg_fraud_amount": 0.0}
    df = pd.read_csv(SAVINGS_LOG)
    confirmed = df[df["analyst_confirmed"] == True]
    return {
        "total_caught": len(df),
        "total_savings_usd": round(float(df["estimated_saving_usd"].sum()), 2),
        "confirmed_savings_usd": round(float(confirmed["estimated_saving_usd"].sum()), 2),
        "avg_fraud_amount": round(float(df["transaction_amount"].mean()), 2),
        "highest_catch_usd": round(float(df["transaction_amount"].max()), 2),
        "last_catch": df["timestamp"].iloc[-1] if len(df) else None,
    }


def load_savings_log() -> pd.DataFrame:
    if not os.path.exists(SAVINGS_LOG):
        return pd.DataFrame()
    return pd.read_csv(SAVINGS_LOG)
