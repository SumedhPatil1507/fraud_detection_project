"""
FastAPI prediction service - Enterprise Edition v3.0
Features: RBAC, async shadow mode, graph intelligence, tokenization,
          rate limiting, data validation, SAR trigger, audit logging

Run locally:
    uvicorn api:app --reload --port 8000
"""

import pickle
import os
import asyncio
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH
from src.rbac import Role, require_permission
from src.validation import validate_transaction
from src.database import log_prediction, load_predictions
from src.shadow import shadow_predict_async, shadow_divergence_stats
from src.sar import generate_sar, load_sar_reports
from src.savings_tracker import record_catch, get_savings_summary
from src.retrain import load_retrain_log
from src.graph_intelligence import load_ring_log
from src.tokenizer import tokenize, rotate_keys, get_key_store_summary

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
    title="FraudGuard AI - Enterprise API",
    description="XGBoost fraud detection with RBAC, async shadow mode, graph intelligence, tokenization.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if _RATE_LIMIT:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_model = _features = _means = None


def load_artifacts():
    global _model, _features, _means
    if not os.path.exists(MODEL_PATH):
        return False
    _model    = pickle.load(open(MODEL_PATH, "rb"))
    _features = pickle.load(open(FEATURE_PATH, "rb"))
    _means    = pickle.load(open(MEAN_PATH, "rb"))
    return True


load_artifacts()


class Transaction(BaseModel):
    transaction_amount:      float         = Field(..., example=1500.0)
    distance_from_home_km:   float         = Field(..., example=250.0)
    hour:                    Optional[int] = Field(None, ge=0, le=23)
    is_foreign:              Optional[int] = Field(None)
    is_new_device:           Optional[int] = Field(None)
    vpn_detected:            Optional[int] = Field(None)
    transaction_velocity_1h: Optional[float] = Field(None)
    amount_deviation_ratio:  Optional[float] = Field(None)
    threshold:               Optional[float] = Field(0.3, ge=0.0, le=1.0)
    customer_id:             Optional[str] = Field(None)
    merchant_id:             Optional[str] = Field(None)


class PredictionResponse(BaseModel):
    fraud_probability:   float
    is_fraud:            bool
    risk_level:          str
    threshold_used:      float
    validation_warnings: List[str] = []
    shadow:              Optional[dict] = None
    graph:               Optional[dict] = None


class BatchRequest(BaseModel):
    transactions: List[Transaction]


def build_input(txn: Transaction) -> pd.DataFrame:
    d = _means.to_dict()
    d["transaction_amount"]    = txn.transaction_amount
    d["distance_from_home_km"] = txn.distance_from_home_km
    if txn.hour is not None:
        d["hour"]     = txn.hour
        d["hour_sin"] = np.sin(2 * np.pi * txn.hour / 24)
        d["hour_cos"] = np.cos(2 * np.pi * txn.hour / 24)
    for attr in ("is_foreign","is_new_device","vpn_detected",
                 "transaction_velocity_1h","amount_deviation_ratio"):
        v = getattr(txn, attr)
        if v is not None:
            d[attr] = v
    d["amount_log"] = np.log1p(txn.transaction_amount)
    return pd.DataFrame([d])[_features]


def risk_label(p: float) -> str:
    return "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.4 else "LOW"


def _bg_log(amount, distance, hour, is_foreign, is_new_device, vpn, prob, is_fraud, threshold):
    try:
        log_prediction(amount, distance, hour, is_foreign, is_new_device, vpn, prob, is_fraud, threshold)
        if is_fraud:
            record_catch(amount, prob)
    except Exception:
        pass


@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": _model is not None, "version": "3.0.0"}


@app.get("/model/info", tags=["Model"])
def model_info(role: Role = Depends(require_permission("model_info"))):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")
    return {"model_type": type(_model).__name__, "n_features": len(_features), "features": _features}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(txn: Transaction, background_tasks: BackgroundTasks,
                  role: Role = Depends(require_permission("predict"))):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")
    vr = validate_transaction(txn.model_dump())
    if not vr.valid:
        raise HTTPException(422, detail={"validation_errors": vr.errors})
    try:
        input_df  = build_input(txn)
        threshold = txn.threshold or 0.3
        prob      = float(_model.predict_proba(input_df)[0][1])
        is_fraud  = prob >= threshold
        shadow_task = asyncio.create_task(shadow_predict_async(_model, input_df, prob, threshold))
        graph_ctx = None
        if txn.customer_id and txn.merchant_id:
            c_token = tokenize(txn.customer_id, "customer_id")
            m_token = tokenize(txn.merchant_id, "merchant_id")
            graph_ctx = {"customer_token": c_token, "merchant_token": m_token}
        shadow = await shadow_task
        background_tasks.add_task(_bg_log, txn.transaction_amount, txn.distance_from_home_km,
                                   txn.hour or 0, txn.is_foreign or 0, txn.is_new_device or 0,
                                   txn.vpn_detected or 0, prob, is_fraud, threshold)
        return PredictionResponse(
            fraud_probability=round(prob, 4), is_fraud=is_fraud,
            risk_level=risk_label(prob), threshold_used=threshold,
            validation_warnings=vr.warnings,
            shadow=shadow if shadow.get("shadow_available") else None,
            graph=graph_ctx,
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(req: BatchRequest, background_tasks: BackgroundTasks,
                        role: Role = Depends(require_permission("batch_predict"))):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    async def _score(txn):
        vr = validate_transaction(txn.model_dump())
        if not vr.valid:
            return {"error": vr.errors}
        try:
            input_df  = build_input(txn)
            prob      = float(_model.predict_proba(input_df)[0][1])
            threshold = txn.threshold or 0.3
            is_fraud  = prob >= threshold
            background_tasks.add_task(_bg_log, txn.transaction_amount, txn.distance_from_home_km,
                                       txn.hour or 0, txn.is_foreign or 0, txn.is_new_device or 0,
                                       txn.vpn_detected or 0, prob, is_fraud, threshold)
            return {"fraud_probability": round(prob,4), "is_fraud": is_fraud,
                    "risk_level": risk_label(prob), "warnings": vr.warnings}
        except Exception as e:
            return {"error": str(e)}

    results = await asyncio.gather(*[_score(t) for t in req.transactions])
    return {"predictions": list(results), "count": len(results)}


@app.get("/audit", tags=["Audit"])
def audit(limit: int = 100, role: Role = Depends(require_permission("audit"))):
    df = load_predictions(limit)
    return {"records": df.to_dict("records") if not df.empty else [], "total": len(df)}


@app.get("/shadow/stats", tags=["Shadow"])
def shadow_stats(role: Role = Depends(require_permission("shadow_deploy"))):
    return shadow_divergence_stats()


@app.post("/sar/generate", tags=["SAR"])
async def create_sar(txn: Transaction, background_tasks: BackgroundTasks,
                     role: Role = Depends(require_permission("sar_generate"))):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")
    input_df = build_input(txn)
    prob     = float(_model.predict_proba(input_df)[0][1])
    return generate_sar(txn.model_dump(), prob, [], threshold=txn.threshold or 0.3)


@app.get("/sar/list", tags=["SAR"])
def list_sars(role: Role = Depends(require_permission("sar_generate"))):
    return {"reports": load_sar_reports()}


@app.get("/savings", tags=["Business"])
def savings(role: Role = Depends(require_permission("predict"))):
    return get_savings_summary()


@app.get("/retrain/log", tags=["MLOps"])
def retrain_log(role: Role = Depends(require_permission("retrain"))):
    return {"history": load_retrain_log()}


@app.get("/graph/rings", tags=["Graph Intelligence"])
def fraud_rings(role: Role = Depends(require_permission("audit"))):
    return {"rings": load_ring_log()}


@app.get("/vault/keys", tags=["Security"])
def vault_keys(role: Role = Depends(require_permission("manage_users"))):
    return get_key_store_summary()


@app.post("/vault/rotate", tags=["Security"])
def vault_rotate(role: Role = Depends(require_permission("manage_users"))):
    return rotate_keys()
