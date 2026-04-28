"""
FastAPI prediction service — Enterprise Edition
Features: RBAC, rate limiting, data validation, shadow mode, SAR trigger, audit logging

Run locally:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /                  health check
    POST /predict           single transaction (analyst+)
    POST /predict/batch     batch predictions (analyst+)
    GET  /model/info        model metadata (viewer+)
    GET  /audit             recent predictions (analyst+)
    GET  /shadow/stats      shadow model divergence (admin)
    POST /sar/generate      generate SAR for a transaction (analyst+)
    GET  /savings           fraud savings summary (analyst+)
    GET  /retrain/log       retraining history (admin)
"""

import pickle
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH
from src.rbac import Role, get_role, require_permission
from src.validation import validate_transaction
from src.database import log_prediction, load_predictions, get_stats
from src.shadow import shadow_predict, shadow_divergence_stats
from src.sar import generate_sar, load_sar_reports
from src.savings_tracker import record_catch, get_savings_summary
from src.retrain import load_retrain_log

# ── Rate limiting (optional dep) ──────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    limiter = Limiter(key_func=get_remote_address)
    _RATE_LIMIT = True
except ImportError:
    limiter = None
    _RATE_LIMIT = False

app = FastAPI(
    title="FraudGuard AI — Enterprise API",
    description="XGBoost fraud detection with RBAC, shadow mode, SAR generation, and audit logging.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if _RATE_LIMIT:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    validation_warnings: List[str] = []
    shadow: Optional[dict] = None


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
    return {"status": "ok", "model_loaded": _model is not None, "version": "2.0.0"}


@app.get("/model/info", tags=["Model"])
def model_info(role: Role = Depends(require_permission("model_info"))):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_type": type(_model).__name__,
        "n_features": len(_features),
        "features": _features,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(txn: Transaction, role: Role = Depends(require_permission("predict"))):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    vr = validate_transaction(txn.model_dump())
    if not vr.valid:
        raise HTTPException(status_code=422, detail={"validation_errors": vr.errors})
    try:
        input_df = build_input(txn)
        prob = float(_model.predict_proba(input_df)[0][1])
        threshold = txn.threshold or 0.3
        is_fraud = prob >= threshold
        shadow = shadow_predict(_model, input_df, _features, prob, threshold)
        log_prediction(txn.transaction_amount, txn.distance_from_home_km,
                       txn.hour or 0, txn.is_foreign or 0, txn.is_new_device or 0,
                       txn.vpn_detected or 0, prob, is_fraud, threshold)
        if is_fraud:
            record_catch(txn.transaction_amount, prob)
        return PredictionResponse(
            fraud_probability=round(prob, 4),
            is_fraud=is_fraud,
            risk_level=risk_label(prob),
            threshold_used=threshold,
            validation_warnings=vr.warnings,
            shadow=shadow if shadow.get("shadow_available") else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(req: BatchRequest, role: Role = Depends(require_permission("batch_predict"))):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    results = []
    for txn in req.transactions:
        vr = validate_transaction(txn.model_dump())
        if not vr.valid:
            results.append({"error": vr.errors})
            continue
        try:
            input_df = build_input(txn)
            prob = float(_model.predict_proba(input_df)[0][1])
            threshold = txn.threshold or 0.3
            is_fraud = prob >= threshold
            if is_fraud:
                record_catch(txn.transaction_amount, prob)
            results.append({
                "fraud_probability": round(prob, 4),
                "is_fraud": is_fraud,
                "risk_level": risk_label(prob),
                "warnings": vr.warnings,
            })
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results, "count": len(results)}


@app.get("/audit", tags=["Audit"])
def audit(limit: int = 100, role: Role = Depends(require_permission("audit"))):
    df = load_predictions(limit)
    return {"records": df.to_dict("records") if not df.empty else [], "total": len(df)}


@app.get("/shadow/stats", tags=["Shadow"])
def shadow_stats(role: Role = Depends(require_permission("shadow_deploy"))):
    return shadow_divergence_stats()


@app.post("/sar/generate", tags=["SAR"])
def create_sar(txn: Transaction, role: Role = Depends(require_permission("sar_generate"))):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    input_df = build_input(txn)
    prob = float(_model.predict_proba(input_df)[0][1])
    sar = generate_sar(txn.model_dump(), prob, [], threshold=txn.threshold or 0.3)
    return sar


@app.get("/sar/list", tags=["SAR"])
def list_sars(role: Role = Depends(require_permission("sar_generate"))):
    return {"reports": load_sar_reports()}


@app.get("/savings", tags=["Business"])
def savings(role: Role = Depends(require_permission("predict"))):
    return get_savings_summary()


@app.get("/retrain/log", tags=["MLOps"])
def retrain_log(role: Role = Depends(require_permission("retrain"))):
    return {"history": load_retrain_log()}
