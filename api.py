"""
FastAPI prediction service for the Fraud Detection model.

Run locally:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /           — health check
    POST /predict    — single transaction fraud prediction
    POST /predict/batch — batch predictions
    GET  /model/info — model metadata
"""

import pickle
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Optional

from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH

app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost-based fraud detection with SHAP explainability",
    version="1.0.0",
)

# ── API Key Auth ───────────────────────────────────────────────────────────────
API_KEY = os.environ.get("API_KEY", "fraud-dev-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return key

_model = None
_features = None
_means = None


def load_artifacts():
    global _model, _features, _means
    if not os.path.exists(MODEL_PATH):
        return False
    _model = pickle.load(open(MODEL_PATH, "rb"))
    _features = pickle.load(open(FEATURE_PATH, "rb"))
    _means = pickle.load(open(MEAN_PATH, "rb"))
    return True


load_artifacts()


# ── Schemas ────────────────────────────────────────────────────────────────────
class Transaction(BaseModel):
    transaction_amount: float = Field(..., example=1500.0)
    distance_from_home_km: float = Field(..., example=250.0)
    hour: Optional[int] = Field(None, ge=0, le=23, example=2)
    is_foreign: Optional[int] = Field(None, example=1)
    is_new_device: Optional[int] = Field(None, example=1)
    vpn_detected: Optional[int] = Field(None, example=0)
    transaction_velocity_1h: Optional[float] = Field(None, example=3.0)
    amount_deviation_ratio: Optional[float] = Field(None, example=4.5)
    threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0, example=0.3)


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    threshold_used: float


class BatchRequest(BaseModel):
    transactions: List[Transaction]


# ── Helpers ────────────────────────────────────────────────────────────────────
def build_input(txn: Transaction) -> pd.DataFrame:
    input_dict = _means.to_dict()
    input_dict["transaction_amount"] = txn.transaction_amount
    input_dict["distance_from_home_km"] = txn.distance_from_home_km
    if txn.hour is not None:
        input_dict["hour"] = txn.hour
        input_dict["hour_sin"] = np.sin(2 * np.pi * txn.hour / 24)
        input_dict["hour_cos"] = np.cos(2 * np.pi * txn.hour / 24)
    if txn.is_foreign is not None:
        input_dict["is_foreign"] = txn.is_foreign
    if txn.is_new_device is not None:
        input_dict["is_new_device"] = txn.is_new_device
    if txn.vpn_detected is not None:
        input_dict["vpn_detected"] = txn.vpn_detected
    if txn.transaction_velocity_1h is not None:
        input_dict["transaction_velocity_1h"] = txn.transaction_velocity_1h
    if txn.amount_deviation_ratio is not None:
        input_dict["amount_deviation_ratio"] = txn.amount_deviation_ratio
    input_dict["amount_log"] = np.log1p(txn.transaction_amount)
    return pd.DataFrame([input_dict])[_features]


def risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    if prob >= 0.4:
        return "MEDIUM"
    return "LOW"


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info", tags=["Model"])
def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    return {
        "model_type": type(_model).__name__,
        "n_features": len(_features),
        "features": _features,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(txn: Transaction):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    try:
        input_df = build_input(txn)
        prob = float(_model.predict_proba(input_df)[0][1])
        threshold = txn.threshold or 0.3
        return PredictionResponse(
            fraud_probability=round(prob, 4),
            is_fraud=prob >= threshold,
            risk_level=risk_label(prob),
            threshold_used=threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(req: BatchRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    results = []
    for txn in req.transactions:
        try:
            input_df = build_input(txn)
            prob = float(_model.predict_proba(input_df)[0][1])
            threshold = txn.threshold or 0.3
            results.append({
                "fraud_probability": round(prob, 4),
                "is_fraud": prob >= threshold,
                "risk_level": risk_label(prob),
            })
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results, "count": len(results)}
