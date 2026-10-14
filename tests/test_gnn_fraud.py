"""Unit tests for heterogeneous R-GCN node and transaction outputs."""
import pytest
import asyncio
import json
from starlette.requests import Request
from starlette.responses import JSONResponse

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
import pandas as pd

from src.graph_intelligence import HETERO_NODE_TYPES, build_heterogeneous_graph
from src.models.gnn_fraud import HeteroFraudRGCN, score_transaction_edges


def test_api_json_contract_has_exact_standardized_keys():
    from api import RESPONSE_KEYS, standardize_json_responses, _standard_response

    steps = [
        "Step 1: Enqueue to HITL review queue",
        "Step 2: Cross-reference PAN/SSN in fraud database",
        "Step 3: Dispatch field audit",
    ]
    risk_response = _standard_response(
        "HIGH_RISK_COLLUSION_RING_DETECTED",
        "R-GCN graph centrality/risk: 0.8; XGBoost SHAP feature contributions: velocity.",
        "Freeze payout and escalate to FIU/compliance review", steps,
    )
    assert set(risk_response) == RESPONSE_KEYS
    assert risk_response["verdict"] == "HIGH_RISK_COLLUSION_RING_DETECTED"
    assert "R-GCN graph centrality" in risk_response["reasoning"]
    assert "XGBoost SHAP" in risk_response["reasoning"]
    assert risk_response["next_steps"] == steps

    scope = {"type": "http", "method": "GET", "path": "/health",
             "headers": [], "query_string": b"", "server": ("test", 80),
             "scheme": "http", "http_version": "1.1", "client": ("test", 1)}
    request = Request(scope)

    async def call_next(_request):
        return JSONResponse({"status": "ok", "model_loaded": True})

    response = asyncio.run(standardize_json_responses(request, call_next))
    payload = json.loads(response.body)

    assert set(payload) == RESPONSE_KEYS
    assert isinstance(payload["verdict"], str)
    assert isinstance(payload["reasoning"], str)
    assert isinstance(payload["recommendation"], str)
    assert isinstance(payload["next_steps"], list)


def _sample_graph():
    frame = pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3"],
        "customer_id": ["c1", "c2", "c1"],
        "merchant_id": ["m1", "m1", "m2"],
        "device_id": ["d1", "d1", "d2"],
        "ip_address": ["ip1", "ip1", "ip2"],
        "transaction_amount": [10.0, 20.0, 30.0],
        "label": [0, 1, 0],
    })
    return build_heterogeneous_graph(frame)


def test_rgcn_emits_embeddings_and_probabilities_for_all_node_types():
    graph = _sample_graph()
    model = HeteroFraudRGCN(hidden_dim=16, embedding_dim=8)

    embeddings, fraud_probabilities, rto_probabilities, node_probabilities = (
        score_transaction_edges(model, graph)
    )

    assert set(embeddings) == set(HETERO_NODE_TYPES)
    assert set(node_probabilities) == set(HETERO_NODE_TYPES)
    assert all(embeddings[kind].shape[1] == 8 for kind in HETERO_NODE_TYPES)
    assert fraud_probabilities.shape == (3,)
    assert rto_probabilities.shape == (3,)
    assert torch.all((fraud_probabilities >= 0) & (fraud_probabilities <= 1))
    assert torch.all((rto_probabilities >= 0) & (rto_probabilities <= 1))


def test_rgcn_forward_preserves_empty_optional_node_stores():
    frame = pd.DataFrame({
        "customer_id": ["c1"], "merchant_id": ["m1"],
        "transaction_amount": [12.0], "label": [0],
    })
    graph = build_heterogeneous_graph(frame)
    model = HeteroFraudRGCN(hidden_dim=8, embedding_dim=4)

    embeddings, fraud_logits, rto_logits, node_logits = model(
        graph.x_dict, graph.edge_index_dict,
        graph["customer", "filed_transaction", "merchant"].edge_index,
    )

    assert embeddings["device_id"].shape == (0, 4)
    assert embeddings["ip_address"].shape == (0, 4)
    assert fraud_logits.shape == rto_logits.shape == (1,)
    assert node_logits["device_id"].shape == node_logits["ip_address"].shape == (0,)
