"""
Automated SAR (Suspicious Activity Report) Generation
Generates FinCEN-style SAR reports as structured JSON.
"""
from __future__ import annotations
import os
import json
from datetime import datetime, timezone
from typing import Optional
import pandas as pd

SAR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "sar_reports")


def _sar_id() -> str:
    return "SAR-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")[:20]


def generate_sar(
    transaction: dict,
    fraud_probability: float,
    shap_factors: list,
    analyst_notes: str = "",
    threshold: float = 0.3,
) -> dict:
    sar_id = _sar_id()
    now = datetime.now(timezone.utc).isoformat()

    risk_indicators = []
    if transaction.get("is_foreign"):
        risk_indicators.append("Foreign transaction detected")
    if transaction.get("is_new_device"):
        risk_indicators.append("New/unrecognized device used")
    if transaction.get("vpn_detected"):
        risk_indicators.append("VPN/proxy detected")
    if transaction.get("transaction_velocity_1h", 0) >= 3:
        risk_indicators.append(f"High velocity: {transaction.get('transaction_velocity_1h')} txns/hr")
    if transaction.get("amount_deviation_ratio", 0) >= 3:
        risk_indicators.append(f"Amount {transaction.get('amount_deviation_ratio', 0):.1f}x above baseline")
    if transaction.get("distance_from_home_km", 0) >= 200:
        risk_indicators.append(f"Unusual distance: {transaction.get('distance_from_home_km', 0):.0f} km from home")

    top_factors = [
        f"{f['feature']} = {f['value']} ({f['direction']} risk by {abs(f['shap_impact']):.3f})"
        for f in (shap_factors or [])[:5]
    ]

    sar = {
        "sar_id": sar_id,
        "generated_at": now,
        "status": "DRAFT",
        "filing_institution": os.environ.get("INSTITUTION_NAME", "FraudGuard AI Platform"),
        "transaction": {
            "amount_usd": round(float(transaction.get("transaction_amount", 0)), 2),
            "timestamp": transaction.get("timestamp", now),
            "hour_of_day": transaction.get("hour"),
            "distance_from_home_km": transaction.get("distance_from_home_km"),
            "is_foreign": bool(transaction.get("is_foreign", False)),
            "is_new_device": bool(transaction.get("is_new_device", False)),
            "vpn_detected": bool(transaction.get("vpn_detected", False)),
        },
        "ml_assessment": {
            "fraud_probability": round(fraud_probability, 4),
            "threshold_used": threshold,
            "verdict": "SUSPICIOUS" if fraud_probability >= threshold else "REVIEW",
            "risk_level": "HIGH" if fraud_probability >= 0.7 else "MEDIUM",
            "top_shap_factors": top_factors,
        },
        "risk_indicators": risk_indicators,
        "analyst_notes": analyst_notes,
        "narrative": _build_narrative(transaction, fraud_probability, risk_indicators, top_factors),
        "recommended_action": _recommend_action(fraud_probability, risk_indicators),
    }

    os.makedirs(SAR_DIR, exist_ok=True)
    path = os.path.join(SAR_DIR, f"{sar_id}.json")
    with open(path, "w") as f:
        json.dump(sar, f, indent=2)

    return sar


def _build_narrative(txn: dict, prob: float, indicators: list, factors: list) -> str:
    amount = txn.get("transaction_amount", 0)
    ts = txn.get("timestamp", "unknown time")
    ind_text = "; ".join(indicators) if indicators else "standard risk profile"
    factor_text = "; ".join(factors[:3]) if factors else "N/A"
    return (
        f"On {ts}, a transaction of ${amount:,.2f} was flagged by the automated fraud "
        f"detection system with a fraud probability of {prob:.1%}. "
        f"Risk indicators identified: {ind_text}. "
        f"Primary model drivers: {factor_text}. "
        f"This report is generated automatically and requires analyst review before filing."
    )


def _recommend_action(prob: float, indicators: list) -> str:
    if prob >= 0.8 or len(indicators) >= 4:
        return "BLOCK_AND_FILE — Block transaction and file SAR immediately."
    if prob >= 0.5 or len(indicators) >= 2:
        return "HOLD_AND_REVIEW — Place transaction on hold, analyst review required."
    return "MONITOR — Flag for enhanced monitoring, no immediate action required."


def load_sar_reports() -> list:
    if not os.path.exists(SAR_DIR):
        return []
    reports = []
    for fname in sorted(os.listdir(SAR_DIR), reverse=True):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(SAR_DIR, fname)) as f:
                    reports.append(json.load(f))
            except Exception:
                pass
    return reports


def update_sar_status(sar_id: str, status: str, notes: str = "") -> bool:
    path = os.path.join(SAR_DIR, f"{sar_id}.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        sar = json.load(f)
    sar["status"] = status
    if notes:
        sar["analyst_notes"] = notes
    sar["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w") as f:
        json.dump(sar, f, indent=2)
    return True
