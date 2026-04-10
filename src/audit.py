"""
Prediction audit log — appends every prediction to a CSV for monitoring.
"""
import os
import pandas as pd
from datetime import datetime
from src.config import AUDIT_LOG_PATH


def log_prediction(amount: float, distance: float, hour: int,
                   prob: float, is_fraud: bool, threshold: float):
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "transaction_amount": amount,
        "distance_from_home_km": distance,
        "hour": hour,
        "fraud_probability": round(prob, 4),
        "is_fraud": is_fraud,
        "threshold_used": threshold,
    }
    df = pd.DataFrame([record])
    write_header = not os.path.exists(AUDIT_LOG_PATH)
    df.to_csv(AUDIT_LOG_PATH, mode='a', header=write_header, index=False)


def load_audit_log() -> pd.DataFrame:
    if not os.path.exists(AUDIT_LOG_PATH):
        return pd.DataFrame()
    return pd.read_csv(AUDIT_LOG_PATH)
