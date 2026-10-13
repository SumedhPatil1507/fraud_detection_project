"""
FastAPI prediction service — Enterprise Edition v4.0
Features: RBAC, async shadow mode, Neo4j graph intelligence,
          cryptographic tokenization, async PostgreSQL, Celery tasks,
          S3 artifact store, Prometheus observability, DPDP/RBI compliance

Run locally:
    uvicorn api:app --reload --port 8000
Swagger UI: http://localhost:8000/docs
"""

import pickle
import os
import asyncio
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from src.config import MODEL_PATH, FEATURE_PATH, MEAN_PATH
from src.model import safe_predict_proba
from src.rbac import Role, require_permission
from src.validation import validate_transaction
from src.db_async import log_prediction, load_predictions
from src.shadow import shadow_predict_async, shadow_divergence_stats
from src.sar import generate_sar, load_sar_reports
from src.savings_tracker import record_catch, get_savings_summary
from src.retrain import load_retrain_log
from src.graph_neo4j import load_ring_log, detect_fraud_rings, get_graph_adapter
from src.tokenizer import tokenize, rotate_keys, get_key_store_summary
from src.object_store import get_registry_summary, load_model_artifacts
from src.task_queue import submit_retrain, submit_drift_check, get_queue_stats, get_task_status
from src.compliance import run_full_compliance, load_compliance_report
from src.observability import record_prediction, start_metrics_server

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
    description=(
        "XGBoost/LightGBM fraud detection · Neo4j graph intelligence · "
        "Async PostgreSQL · Celery tasks · S3 artifacts · Prometheus · DPDP/RBI compliance"
    ),
    version="4.0.0",
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
    # Try object store with checksum verification first
    try:
        model, features, means = load_model_artifacts()
        if model is not None:
            _model, _features, _means = model, features, means
            return True
    except Exception as e:
        print(f"[API] Object store load failed ({e}) — using local pickle")
    # Local fallback
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
    verdict: str
    reasoning: str
    recommendation: str
    next_steps: List[str]


class BatchRequest(BaseModel):
    transactions: List[Transaction]


RESPONSE_KEYS = {"verdict", "reasoning", "recommendation", "next_steps"}


def _standard_response(verdict: str, reasoning: str, recommendation: str,
                       next_steps: List[str]) -> dict:
    """Build the API's single public JSON contract, with no extra properties."""
    return {"verdict": verdict, "reasoning": reasoning,
            "recommendation": recommendation, "next_steps": next_steps}


def _fraud_explanation(txn: Transaction, probability: float,
                       input_df: pd.DataFrame, threshold: float) -> dict:
    """Combine model probability, SHAP contributions, graph centrality and GNN signals."""
    from src.llm_explain import get_top_shap_factors
    from src.graph_intelligence import build_transaction_graph, score_transaction_graph

    factors = get_top_shap_factors(_model, input_df, _features or [], top_n=5)
    shap_text = "; ".join(
        f"{item['feature']} {item['direction']} risk by {item['shap_impact']:+.4f}"
        for item in factors
    ) or "SHAP contributions unavailable"
    graph_score = {}
    gnn_scores = {}
    ring_detected = False
    if txn.customer_id and txn.merchant_id:
        try:
            graph = build_transaction_graph(pd.DataFrame([{
                "customer_id": txn.customer_id,
                "merchant_id": txn.merchant_id,
                "transaction_amount": txn.transaction_amount,
                "label": int(probability >= threshold),
            }]))
            graph_score = score_transaction_graph(txn.customer_id, txn.merchant_id, graph)
            ring_members = {
                member for ring in load_ring_log()
                for member in ring.get("members", [])
            }
            ring_detected = (f"C_{txn.customer_id}" in ring_members
                             or f"M_{txn.merchant_id}" in ring_members)
            adapter = get_graph_adapter()
            if adapter.backend == "neo4j":
                gnn_scores = adapter.get_gnn_scores(txn.customer_id, txn.merchant_id)
        except Exception:
            # A failed optional graph source is reported as unavailable, never as a zero-risk signal.
            graph_score = {}

    graph_text = (
        f"graph centrality/risk={graph_score['graph_risk_score']:.4f}"
        if graph_score and "graph_risk_score" in graph_score
        else "graph centrality unavailable"
    )
    if gnn_scores:
        graph_text += f", R-GCN collusion={gnn_scores.get('gnn_collusion_risk', 0):.4f}"
        ring_detected = ring_detected or gnn_scores.get("gnn_collusion_risk", 0) >= 0.7

    high_risk = probability >= max(threshold, 0.7) or ring_detected
    medium_risk = probability >= threshold
    if ring_detected:
        verdict = "HIGH_RISK_COLLUSION_RING_DETECTED"
        recommendation = "Freeze payout and escalate to FIU/compliance review"
        next_steps = [
            "Step 1: Enqueue to HITL review queue",
            "Step 2: Cross-reference PAN/SSN in fraud database where authorized",
            "Step 3: Dispatch field audit if required by the investigator",
        ]
    elif high_risk:
        verdict = "HIGH_RISK_FRAUD_DETECTED"
        recommendation = "Freeze payout and escalate for investigator review"
        next_steps = [
            "Step 1: Enqueue to HITL review queue",
            "Step 2: Validate identity and transaction evidence",
            "Step 3: Escalate to FIU when statutory reporting criteria are met",
        ]
    elif medium_risk:
        verdict = "REVIEW_FRAUD_RISK"
        recommendation = "Hold for analyst review"
        next_steps = [
            "Step 1: Enqueue to HITL review queue",
            "Step 2: Review SHAP and graph evidence",
        ]
    else:
        verdict = "CLEAR_AUTO_APPROVE"
        recommendation = "Auto-approve under the configured risk threshold"
        next_steps = ["Step 1: Record the decision and continue monitoring"]

    reasoning = (
        f"Fraud probability={probability:.4f} at threshold={threshold:.4f}; "
        f"R-GCN/graph {graph_text}; XGBoost SHAP contributions: {shap_text}."
    )
    return _standard_response(verdict, reasoning, recommendation, next_steps)


@app.middleware("http")
async def standardize_json_responses(request: Request, call_next):
    """Normalize every JSON route response, including framework error bodies."""
    response = await call_next(request)
    if "application/json" not in response.headers.get("content-type", ""):
        return response
    body = b"".join([chunk async for chunk in response.body_iterator])
    try:
        payload = json.loads(body or b"null")
    except (TypeError, ValueError):
        payload = {"message": "Response could not be decoded as JSON"}
    if isinstance(payload, dict) and set(payload) == RESPONSE_KEYS:
        standardized = payload
    else:
        status = response.status_code
        standardized = _standard_response(
            "REQUEST_REJECTED" if status >= 400 else "API_OPERATION_COMPLETED",
            json.dumps(payload, default=str, ensure_ascii=False),
            "Correct the request or review the operation result",
            ["Step 1: Review the reasoning details", "Step 2: Retry or escalate as appropriate"],
        )
    if response.background is not None:
        await response.background()
    return JSONResponse(standardized, status_code=response.status_code,
                        headers={key: value for key, value in response.headers.items()
                                 if key.lower() not in {"content-length", "content-type"}})


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
    return {"status": "ok", "model_loaded": _model is not None, "version": "4.0.0",
            "graph_backend": get_graph_adapter().backend}


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
        prob      = float(safe_predict_proba(_model, input_df)[0][1])
        is_fraud  = prob >= threshold
        shadow_task = asyncio.create_task(shadow_predict_async(_model, input_df, prob, threshold))
        graph_ctx = None
        if txn.customer_id and txn.merchant_id:
            c_token = tokenize(txn.customer_id, "customer_id")
            m_token = tokenize(txn.merchant_id, "merchant_id")
            graph_ctx = {"customer_token": c_token, "merchant_token": m_token}
        shadow = await shadow_task
        # Record to Prometheus
        record_prediction(prob, is_fraud)
        background_tasks.add_task(_bg_log, txn.transaction_amount, txn.distance_from_home_km,
                                   txn.hour or 0, txn.is_foreign or 0, txn.is_new_device or 0,
                                   txn.vpn_detected or 0, prob, is_fraud, threshold)
        return PredictionResponse(**_fraud_explanation(txn, prob, input_df, threshold))
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
            return _standard_response("REQUEST_REJECTED", "; ".join(vr.errors),
                                      "Correct the invalid transaction fields",
                                      ["Step 1: Correct the request", "Step 2: Submit it again"])
        try:
            input_df  = build_input(txn)
            prob      = float(safe_predict_proba(_model, input_df)[0][1])
            threshold = txn.threshold or 0.3
            is_fraud  = prob >= threshold
            background_tasks.add_task(_bg_log, txn.transaction_amount, txn.distance_from_home_km,
                                       txn.hour or 0, txn.is_foreign or 0, txn.is_new_device or 0,
                                       txn.vpn_detected or 0, prob, is_fraud, threshold)
            return _fraud_explanation(txn, prob, input_df, threshold)
        except Exception as e:
            return _standard_response("SCORING_ERROR", type(e).__name__,
                                     "Escalate the scoring failure for investigation",
                                     ["Step 1: Record the failed transaction", "Step 2: Retry after service recovery"])

    results = await asyncio.gather(*[_score(t) for t in req.transactions])
    priority = {"HIGH_RISK_COLLUSION_RING_DETECTED": 4, "HIGH_RISK_FRAUD_DETECTED": 3,
                "REVIEW_FRAUD_RISK": 2, "CLEAR_AUTO_APPROVE": 1, "REQUEST_REJECTED": 0,
                "SCORING_ERROR": 0}
    worst = max(results, key=lambda result: priority.get(result["verdict"], 0),
                 default=_standard_response("CLEAR_AUTO_APPROVE", "Empty batch",
                                            "No transactions to score", []))
    return _standard_response(
        worst["verdict"],
        f"Batch size={len(results)}. " + " | ".join(item["reasoning"] for item in results),
        worst["recommendation"], worst["next_steps"],
    )


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
    prob     = float(safe_predict_proba(_model, input_df)[0][1])
    sar = generate_sar(txn.model_dump(), prob, [], threshold=txn.threshold or 0.3)
    explanation = _fraud_explanation(txn, prob, input_df, txn.threshold or 0.3)
    explanation["reasoning"] += f" SAR draft {sar['sar_id']} created with status {sar['status']}."
    return explanation


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

# ── v4.0 New Endpoints ────────────────────────────────────────────────────────

@app.get("/artifacts", tags=["MLOps"])
def artifact_registry(role: Role = Depends(require_permission("manage_users"))):
    """S3/MinIO artifact registry with MD5 checksums."""
    return get_registry_summary()


@app.post("/tasks/retrain", tags=["MLOps"])
def trigger_retrain(reason: str = "api", role: Role = Depends(require_permission("retrain"))):
    """Submit retraining job to Celery queue."""
    return submit_retrain(reason=reason)


@app.post("/tasks/drift-check", tags=["MLOps"])
def trigger_drift(role: Role = Depends(require_permission("drift"))):
    """Submit drift check to Celery queue."""
    return submit_drift_check()


@app.get("/tasks/{task_id}", tags=["MLOps"])
def task_status(task_id: str, role: Role = Depends(require_permission("audit"))):
    """Get Celery task status by ID."""
    return get_task_status(task_id)


@app.get("/tasks/queue/stats", tags=["MLOps"])
def queue_stats(role: Role = Depends(require_permission("audit"))):
    """Celery queue depth and worker stats."""
    return get_queue_stats()


@app.get("/compliance", tags=["Compliance"])
def compliance_report(role: Role = Depends(require_permission("manage_users"))):
    """DPDP Act 2023 + RBI IT Framework compliance report."""
    report = load_compliance_report()
    if not report:
        report = run_full_compliance()
    return report


@app.post("/compliance/run", tags=["Compliance"])
def run_compliance(role: Role = Depends(require_permission("manage_users"))):
    """Re-run full compliance audit."""
    return run_full_compliance()


@app.get("/graph/rings", tags=["Graph Intelligence"])
def fraud_rings(role: Role = Depends(require_permission("audit"))):
    """Detected fraud rings from Neo4j / NetworkX."""
    return {"rings": load_ring_log(), "backend": get_graph_adapter().backend}


@app.get("/vault/keys", tags=["Security"])
def vault_keys(role: Role = Depends(require_permission("manage_users"))):
    return get_key_store_summary()


@app.post("/vault/rotate", tags=["Security"])
def vault_rotate(role: Role = Depends(require_permission("manage_users"))):
    return rotate_keys()


@app.on_event("startup")
async def startup():
    """Start Prometheus metrics server on startup (if configured)."""
    port = int(os.environ.get("PROMETHEUS_PORT", 0))
    if port:
        import threading
        threading.Thread(target=start_metrics_server, args=(port,), daemon=True).start()
