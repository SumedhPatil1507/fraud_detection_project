"""
Live transaction simulator.
Generates synthetic transactions that mimic real fraud patterns,
scored in real-time using the trained model.
"""
import numpy as np
import pandas as pd
from datetime import datetime


def generate_transaction(rng: np.random.Generator, fraud: bool = False) -> dict:
    hour = rng.integers(0, 24)
    is_night = int(hour < 6 or hour >= 22)

    if fraud:
        amount = float(rng.exponential(1500) + 500)
        distance = float(rng.exponential(400) + 100)
        is_foreign = int(rng.random() > 0.3)
        is_new_device = int(rng.random() > 0.4)
        vpn = int(rng.random() > 0.5)
        velocity_1h = int(rng.integers(3, 8))
        deviation = float(rng.uniform(3.0, 8.0))
    else:
        amount = float(rng.exponential(200) + 20)
        distance = float(rng.exponential(30) + 1)
        is_foreign = int(rng.random() > 0.85)
        is_new_device = int(rng.random() > 0.8)
        vpn = int(rng.random() > 0.95)
        velocity_1h = int(rng.integers(0, 3))
        deviation = float(rng.uniform(0.5, 2.0))

    avg_30d = amount / max(deviation, 0.1)
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "transaction_amount": round(amount, 2),
        "distance_from_home_km": round(distance, 2),
        "hour": int(hour),
        "is_night": is_night,
        "is_foreign": is_foreign,
        "is_new_device": is_new_device,
        "vpn_detected": vpn,
        "transaction_velocity_1h": velocity_1h,
        "amount_deviation_ratio": round(deviation, 4),
        "avg_amount_30d": round(avg_30d, 2),
        "true_label": int(fraud),
    }


def score_transaction(txn: dict, model, features: list,
                      means: pd.Series, threshold: float) -> dict:
    input_dict = means.to_dict()
    input_dict.update({k: v for k, v in txn.items()
                       if k not in ("true_label", "timestamp")})
    input_dict["amount_log"] = np.log1p(txn["transaction_amount"])
    input_dict["hour_sin"] = np.sin(2 * np.pi * txn["hour"] / 24)
    input_dict["hour_cos"] = np.cos(2 * np.pi * txn["hour"] / 24)
    input_dict["amount_vs_avg"] = txn["transaction_amount"] / (txn["avg_amount_30d"] + 1)
    input_dict["amount_x_distance"] = txn["transaction_amount"] * txn["distance_from_home_km"]
    try:
        input_df = pd.DataFrame([input_dict])[features]
        prob = float(model.predict_proba(input_df)[0][1])
    except Exception:
        prob = 0.0
    risk = "🔴 HIGH" if prob >= 0.7 else "🟡 MEDIUM" if prob >= threshold else "🟢 LOW"
    return {**txn, "fraud_probability": round(prob, 4),
            "predicted_fraud": int(prob >= threshold), "risk_level": risk}


def generate_batch(n: int = 10, fraud_rate: float = 0.2,
                   seed: int = None) -> list[dict]:
    rng = np.random.default_rng(seed)
    txns = []
    for _ in range(n):
        is_fraud = rng.random() < fraud_rate
        txns.append(generate_transaction(rng, fraud=is_fraud))
    return txns


def stream_one(fraud_rate: float = 0.2, seed: int = None) -> dict:
    """Generate and return a single transaction (for live streaming)."""
    rng = np.random.default_rng(seed)
    is_fraud = rng.random() < fraud_rate
    return generate_transaction(rng, fraud=is_fraud)

    """Generate a single synthetic transaction."""
    hour = rng.integers(0, 24)
    is_night = int(hour < 6 or hour >= 22)

    if fraud:
        amount = float(rng.exponential(1500) + 500)
        distance = float(rng.exponential(400) + 100)
        is_foreign = int(rng.random() > 0.3)
        is_new_device = int(rng.random() > 0.4)
        vpn = int(rng.random() > 0.5)
        velocity_1h = int(rng.integers(3, 8))
        deviation = float(rng.uniform(3.0, 8.0))
    else:
        amount = float(rng.exponential(200) + 20)
        distance = float(rng.exponential(30) + 1)
        is_foreign = int(rng.random() > 0.85)
        is_new_device = int(rng.random() > 0.8)
        vpn = int(rng.random() > 0.95)
        velocity_1h = int(rng.integers(0, 3))
        deviation = float(rng.uniform(0.5, 2.0))

    avg_30d = amount / max(deviation, 0.1)

    return {
        "transaction_amount": round(amount, 2),
        "distance_from_home_km": round(distance, 2),
        "hour": int(hour),
        "is_night": is_night,
        "is_foreign": is_foreign,
        "is_new_device": is_new_device,
        "vpn_detected": vpn,
        "transaction_velocity_1h": velocity_1h,
        "amount_deviation_ratio": round(deviation, 4),
        "avg_amount_30d": round(avg_30d, 2),
        "true_label": int(fraud),
    }


def score_transaction(txn: dict, model, features: list,
                      means: pd.Series, threshold: float) -> dict:
    """Score a single transaction dict using the loaded model."""
    input_dict = means.to_dict()
    input_dict.update({k: v for k, v in txn.items() if k != "true_label"})

    # Derived features
    input_dict["amount_log"] = np.log1p(txn["transaction_amount"])
    input_dict["hour_sin"] = np.sin(2 * np.pi * txn["hour"] / 24)
    input_dict["hour_cos"] = np.cos(2 * np.pi * txn["hour"] / 24)
    input_dict["amount_vs_avg"] = txn["transaction_amount"] / (txn["avg_amount_30d"] + 1)
    input_dict["amount_x_distance"] = txn["transaction_amount"] * txn["distance_from_home_km"]

    try:
        input_df = pd.DataFrame([input_dict])[features]
        prob = float(model.predict_proba(input_df)[0][1])
    except Exception:
        prob = 0.0

    risk = "🔴 HIGH" if prob >= 0.7 else "🟡 MEDIUM" if prob >= threshold else "🟢 LOW"
    return {**txn, "fraud_probability": round(prob, 4),
            "predicted_fraud": int(prob >= threshold), "risk_level": risk}


def generate_batch(n: int = 10, fraud_rate: float = 0.2,
                   seed: int = None) -> list[dict]:
    """Generate a batch of mixed transactions."""
    rng = np.random.default_rng(seed)
    txns = []
    for _ in range(n):
        is_fraud = rng.random() < fraud_rate
        txns.append(generate_transaction(rng, fraud=is_fraud))
    return txns
