"""Writer guardrail tests: rejected content must route to human review."""
from src import agent_graph
from src.copilot_guardrails import validate_sar_narrative


def test_unverified_writer_output_routes_to_review_fallback(monkeypatch):
    state = {
        "transaction": {"transaction_id": "txn-test", "transaction_amount": 120.0,
                        "timestamp": "2026-01-01T12:00:00Z", "jurisdiction": "US"},
        "fraud_probability": 0.8,
        "retrieved_cases": [{"id": "chunk-1", "text": "Historical case about repeated card fraud."}],
        "graph_intel": {}, "fraud_rings": [], "shap_factors": [],
    }
    monkeypatch.setattr(
        agent_graph, "_llm_narrative",
        lambda _prompt: "The customer is part of a criminal ring [chunk_id:chunk-1].",
    )

    rejected = agent_graph.writer_node(state)

    assert rejected["writer_validation_status"] == "rejected"
    assert rejected["sar_draft"] == {}
    assert agent_graph._writer_route(rejected) == "fallback"
    review_state = agent_graph.writer_fallback_node(rejected)
    assert review_state["writer_validation_status"] == "fallback_review_required"
    assert review_state["sar_draft"]["status"] == "REVIEW_REQUIRED"
    assert review_state["sar_draft"]["requires_manual_review"] is True


def test_writer_citations_must_exist_and_support_the_claim():
    hits = [{"id": "chunk-device", "text": "Multiple customer accounts used the same device for transactions."}]
    unsupported = "The account laundered funds through an offshore shell [chunk_id:chunk-device]."
    result = validate_sar_narrative(unsupported, hits, {"jurisdiction": "US"})
    assert result["valid"] is False
    assert any("not grounded" in error for error in result["errors"])


def test_valid_grounded_citation_passes():
    hits = [{"id": "chunk-device", "text": "Multiple customer accounts used the same device for transactions."}]
    narrative = "Multiple customer accounts used the same device for transactions [chunk_id:chunk-device]."
    result = validate_sar_narrative(narrative, hits, {"jurisdiction": "US"})
    assert result["valid"] is True
