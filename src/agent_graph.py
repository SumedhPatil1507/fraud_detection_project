"""
Fraud Investigation Copilot — LangGraph Agent Graph
=====================================================
Three-node pipeline wired with LangGraph's StateGraph:

  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  RETRIEVER   │───▶│     TOOL     │───▶│    WRITER    │
  │    agent     │    │    agent     │    │    agent     │
  └──────────────┘    └──────────────┘    └──────────────┘
        │                    │                    │
  semantic search      graph_intel +         SAR narrative
  audit log +          shap_factors           via Groq LLM
  SAR reports          tool calls             (or fallback)
                                                   │
                                           ┌───────▼────────┐
                                           │  HITL queue    │
                                           │ (add_to_review  │
                                           │  _queue)        │
                                           └────────────────┘

State schema  (CopilotState, TypedDict)
---------------------------------------
  transaction        dict    — raw transaction fields
  fraud_probability  float
  model              Any     — loaded sklearn/XGB model (optional)
  input_df           Any     — single-row pd.DataFrame (optional)
  feature_names      list[str]

  # filled by RetrieverAgent
  retrieved_cases    list[dict]   — hits from vector_store.search_all()

  # filled by ToolAgent
  graph_intel        dict    — from graph_intelligence.score_transaction_graph
  fraud_rings        list[dict]  — rings the customer/merchant belongs to
  shap_factors       list[dict]  — from llm_explain.get_top_shap_factors

  # filled by WriterAgent
  sar_draft          dict    — full SAR dict (status="DRAFT")
  narrative          str     — AI-enriched narrative paragraph

  # set by HITL router
  hitl_item_index    int | None  — index into st.session_state.review_queue

Observability
-------------
Every node is wrapped with @instrument_agent (copilot_metrics.py) which
emits fraudguard_copilot_agent_runs_total and
fraudguard_copilot_agent_latency_seconds automatically.
Additional fine-grained events (tool calls, retriever hits, SAR drafts,
HITL queues) are recorded inline.

Graceful degradation
--------------------
- LangGraph not installed  → run_copilot() falls back to a straight pipeline
  (same logic, no graph overhead) and still returns the same output dict.
- SHAP / graph-intel unavailable → those fields are empty dicts / lists.
- Groq unavailable → writer uses sar.py's _build_narrative() fallback.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, TypedDict

import pandas as pd

from src.config import COPILOT_TOP_K, COPILOT_LLM_MODEL
from src.copilot_metrics import (
    instrument_agent,
    record_tool_call,
    record_retriever_hits,
    record_sar_draft,
    record_hitl_queued,
    record_agent_error,
)
from src.vector_store import search_all
from src.graph_intelligence import (
    score_transaction_graph,
    build_transaction_graph,
    detect_fraud_rings,
    load_ring_log,
)
from src.llm_explain import get_top_shap_factors
from src.sar import generate_sar, SAR_DIR
from src.hitl import add_to_review_queue

log = logging.getLogger(__name__)

# ── Optional LangGraph import ─────────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    _LG_OK = True
except ImportError:
    _LG_OK = False
    log.warning(
        "[agent_graph] langgraph not installed — will use linear fallback pipeline"
    )

# ── Optional Groq import (reuse existing pattern from llm_explain.py) ─────────
try:
    import groq as _groq_mod
    _GROQ_OK = True
except ImportError:
    _GROQ_OK = False


def _get_groq_client():
    if not _GROQ_OK:
        return None
    api_key = os.environ.get("GROQ_API_KEY", "")
    # Also try Streamlit secrets when running inside the app
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        return None
    try:
        return _groq_mod.Groq(api_key=api_key)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# State definition
# ══════════════════════════════════════════════════════════════════════════════

class CopilotState(TypedDict, total=False):
    # ── inputs ────────────────────────────────────────────────────────────────
    transaction:        dict           # raw transaction dict
    fraud_probability:  float
    model:              Any            # sklearn / XGB model, optional
    input_df:           Any            # pd.DataFrame single row, optional
    feature_names:      list           # list[str]

    # ── retriever output ──────────────────────────────────────────────────────
    retrieved_cases:    list           # list[dict] from vector_store

    # ── tool-agent output ─────────────────────────────────────────────────────
    graph_intel:        dict           # score_transaction_graph result
    fraud_rings:        list           # rings this transaction belongs to
    shap_factors:       list           # list[dict] from get_top_shap_factors

    # ── writer output ─────────────────────────────────────────────────────────
    sar_draft:          dict           # full SAR JSON dict, status=DRAFT
    narrative:          str            # AI-enriched narrative

    # ── HITL bookkeeping ──────────────────────────────────────────────────────
    hitl_item_index:    int | None


# ══════════════════════════════════════════════════════════════════════════════
# Node 1 — RetrieverAgent
# ══════════════════════════════════════════════════════════════════════════════

@instrument_agent("retriever")
def retriever_node(state: CopilotState) -> CopilotState:
    """
    Semantic search over audit log + past SARs to surface the most similar
    historical cases.

    Builds a natural-language query from the transaction fields, then calls
    vector_store.search_all().  The top-k hits are stored in
    state["retrieved_cases"].
    """
    txn  = state.get("transaction", {})
    prob = state.get("fraud_probability", 0.0)

    amount   = txn.get("transaction_amount", 0)
    distance = txn.get("distance_from_home_km", 0)
    hour     = txn.get("hour", 0)
    foreign  = txn.get("is_foreign", False)
    new_dev  = txn.get("is_new_device", False)
    vpn      = txn.get("vpn_detected", False)

    query = (
        f"Fraud transaction: amount=${float(amount):.2f}, "
        f"distance={float(distance):.1f}km, hour={hour}, "
        f"foreign={'yes' if foreign else 'no'}, "
        f"new_device={'yes' if new_dev else 'no'}, "
        f"vpn={'yes' if vpn else 'no'}, "
        f"fraud_probability={float(prob):.4f}."
    )

    log.info("[retriever] query: %s", query)
    hits = search_all(query, top_k=COPILOT_TOP_K)
    record_retriever_hits(len(hits))
    log.info("[retriever] retrieved %d cases", len(hits))

    return {**state, "retrieved_cases": hits}


# ══════════════════════════════════════════════════════════════════════════════
# Node 2 — ToolAgent
# ══════════════════════════════════════════════════════════════════════════════

def _query_graph_intel(txn: dict) -> tuple[dict, list]:
    """
    Query graph_intelligence for:
      1. Real-time graph score for this (customer, merchant) pair.
      2. Any fraud rings the transaction participants belong to.

    Returns (graph_score_dict, matching_rings).
    """
    customer_id = str(txn.get("customer_id", ""))
    merchant_id = str(txn.get("merchant_id", ""))

    # Re-use the persisted ring log rather than re-running full ring detection
    # (which requires the full dataset).  If no ring log exists, returns [].
    rings = load_ring_log()

    matching_rings: list[dict] = []
    for ring in rings:
        members = ring.get("members", [])
        if (
            f"C_{customer_id}" in members
            or f"M_{merchant_id}" in members
        ):
            matching_rings.append(ring)

    # Build a minimal single-transaction DataFrame for the graph scorer
    try:
        df_single = pd.DataFrame([{
            "customer_id":        customer_id,
            "merchant_id":        merchant_id,
            "transaction_amount": float(txn.get("transaction_amount", 0)),
            "label":              1,  # we're investigating a flagged transaction
        }])
        G = build_transaction_graph(df_single)
        graph_score = score_transaction_graph(customer_id, merchant_id, G)
    except Exception as exc:
        log.warning("[tool_agent] graph scoring failed: %s", exc)
        graph_score = {
            "graph_risk_score": 0.0,
            "ring_member": int(bool(matching_rings)),
            "customer_degree": 0,
            "merchant_degree": 0,
        }

    # Augment with ring membership details
    graph_score["ring_count"]     = len(matching_rings)
    graph_score["ring_ids"]       = [r.get("ring_id") for r in matching_rings]
    graph_score["max_ring_fraud_rate"] = max(
        (r.get("fraud_rate", 0) for r in matching_rings), default=0.0
    )

    return graph_score, matching_rings


def _query_shap(state: CopilotState) -> list[dict]:
    """
    Extract top SHAP feature drivers for this transaction.
    Requires model and input_df to be present in state.
    """
    model      = state.get("model")
    input_df   = state.get("input_df")
    feat_names = state.get("feature_names", [])

    if model is None or input_df is None or not feat_names:
        log.info("[tool_agent] SHAP skipped — model/input_df not provided")
        return []

    try:
        factors = get_top_shap_factors(model, input_df, feat_names, top_n=6)
        return factors
    except Exception as exc:
        log.warning("[tool_agent] SHAP query failed: %s", exc)
        return []


@instrument_agent("tool")
def tool_node(state: CopilotState) -> CopilotState:
    """
    Tool-calling agent — dispatches two tools in sequence:
      1. graph_intel  — fraud-ring membership + graph risk score
      2. shap_factors — top SHAP feature drivers
    Each call is individually metered via record_tool_call().
    """
    txn = state.get("transaction", {})

    # ── Tool 1: graph intelligence ────────────────────────────────────────────
    try:
        graph_intel, fraud_rings = _query_graph_intel(txn)
        record_tool_call("graph_intel", "success")
        log.info(
            "[tool_agent] graph_intel: risk=%.4f rings=%d",
            graph_intel.get("graph_risk_score", 0),
            len(fraud_rings),
        )
    except Exception as exc:
        log.error("[tool_agent] graph_intel error: %s", exc)
        record_tool_call("graph_intel", "error")
        graph_intel  = {}
        fraud_rings  = []

    # ── Tool 2: SHAP factors ──────────────────────────────────────────────────
    try:
        shap_factors = _query_shap(state)
        record_tool_call("shap_factors", "success")
        log.info("[tool_agent] shap_factors: %d drivers", len(shap_factors))
    except Exception as exc:
        log.error("[tool_agent] shap_factors error: %s", exc)
        record_tool_call("shap_factors", "error")
        shap_factors = []

    return {
        **state,
        "graph_intel":  graph_intel,
        "fraud_rings":  fraud_rings,
        "shap_factors": shap_factors,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node 3 — WriterAgent
# ══════════════════════════════════════════════════════════════════════════════

_WRITER_SYSTEM = """\
You are a specialist fraud analyst and SAR (Suspicious Activity Report) writer.
Your task is to draft the narrative section of a FinCEN-style SAR, citing both
the ML evidence and any relevant historical precedents.

Guidelines:
- Be concise, factual, and professional (3-5 paragraphs).
- Open with the transaction verdict and fraud probability.
- Paragraph 2: summarise the top SHAP feature drivers.
- Paragraph 3: summarise the graph-intelligence findings (ring membership,
  graph risk score).
- Paragraph 4: cite up to 3 historical precedent cases from the retrieved
  evidence, referencing their IDs and similarity scores.
- Close with the recommended action.
- Do NOT fabricate facts. Only use the data provided.
"""


def _build_writer_prompt(state: CopilotState) -> str:
    txn   = state.get("transaction", {})
    prob  = state.get("fraud_probability", 0.0)
    shap  = state.get("shap_factors", [])
    graph = state.get("graph_intel", {})
    rings = state.get("fraud_rings", [])
    cases = state.get("retrieved_cases", [])

    shap_text = "\n".join(
        f"  • {f['feature']} = {f['value']} "
        f"({f['direction']} risk by {abs(f['shap_impact']):.4f})"
        for f in shap[:6]
    ) or "  No SHAP data available."

    ring_text = (
        f"  Ring member: YES — {len(rings)} ring(s) matched, "
        f"max ring fraud rate = {graph.get('max_ring_fraud_rate', 0):.1%}. "
        f"Ring IDs: {', '.join(graph.get('ring_ids', [])[:3]) or 'N/A'}."
        if graph.get("ring_member")
        else "  Not a known ring member."
    )

    precedent_text = "\n".join(
        f"  [{i+1}] {h['source'].upper()} id={h['id']} "
        f"(similarity={h['score']:.3f}): {h['text'][:200]}..."
        for i, h in enumerate(cases[:3])
    ) or "  No historical precedents found in vector store."

    return f"""\
=== TRANSACTION UNDER INVESTIGATION ===
Amount:           ${float(txn.get('transaction_amount', 0)):,.2f}
Fraud probability: {float(prob):.1%}
Hour of day:      {txn.get('hour', 'N/A')}
Distance (km):    {txn.get('distance_from_home_km', 'N/A')}
Foreign txn:      {'Yes' if txn.get('is_foreign') else 'No'}
New device:       {'Yes' if txn.get('is_new_device') else 'No'}
VPN detected:     {'Yes' if txn.get('vpn_detected') else 'No'}

=== TOP SHAP FEATURE DRIVERS ===
{shap_text}

=== GRAPH INTELLIGENCE ===
  Graph risk score: {graph.get('graph_risk_score', 0):.4f}
  Customer degree:  {graph.get('customer_degree', 0)}
  Merchant degree:  {graph.get('merchant_degree', 0)}
{ring_text}

=== HISTORICAL PRECEDENT CASES ===
{precedent_text}
"""


def _llm_narrative(prompt: str) -> str | None:
    """Call Groq and return the narrative text, or None on any failure."""
    client = _get_groq_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=COPILOT_LLM_MODEL,
            messages=[
                {"role": "system", "content": _WRITER_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=600,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        log.warning("[writer_agent] Groq call failed: %s", exc)
        return None


def _fallback_narrative(state: CopilotState) -> str:
    """Rule-based narrative when Groq is unavailable."""
    txn   = state.get("transaction", {})
    prob  = state.get("fraud_probability", 0.0)
    shap  = state.get("shap_factors", [])
    graph = state.get("graph_intel", {})
    cases = state.get("retrieved_cases", [])

    amount = txn.get("transaction_amount", 0)
    ts     = txn.get("timestamp", "unknown time")

    top_drivers = ", ".join(
        f"{f['feature']} ({f['direction']} risk)" for f in shap[:3]
    ) or "no SHAP data"

    ring_note = (
        f"The entity is a member of {graph.get('ring_count', 0)} known fraud "
        f"ring(s) with a peak fraud rate of "
        f"{graph.get('max_ring_fraud_rate', 0):.1%}."
        if graph.get("ring_member")
        else "No known fraud-ring membership was detected."
    )

    precedent_note = (
        f"Retrieved {len(cases)} similar historical case(s) from the audit "
        f"log and SAR database (top similarity: "
        f"{cases[0]['score']:.3f})."
        if cases
        else "No similar historical cases were found."
    )

    return (
        f"On {ts}, a transaction of ${float(amount):,.2f} was automatically "
        f"flagged with a fraud probability of {float(prob):.1%}. "
        f"The primary model drivers were: {top_drivers}. "
        f"{ring_note} "
        f"{precedent_note} "
        f"This draft SAR was generated by the Fraud Investigation Copilot and "
        f"requires analyst review before filing. "
        f"(Set GROQ_API_KEY for AI-enriched narratives.)"
    )


@instrument_agent("writer")
def writer_node(state: CopilotState) -> CopilotState:
    """
    Writer agent — drafts the SAR narrative, enriches it with LLM prose
    (via Groq), then calls generate_sar() to produce the full SAR dict.
    The SAR status is forced to DRAFT so it is never auto-filed.
    """
    txn   = state.get("transaction", {})
    prob  = state.get("fraud_probability", 0.0)
    shap  = state.get("shap_factors", [])

    # ── Build AI-enriched narrative ───────────────────────────────────────────
    prompt    = _build_writer_prompt(state)
    narrative = _llm_narrative(prompt)
    if narrative is None:
        narrative = _fallback_narrative(state)
        log.info("[writer_agent] used fallback narrative")
    else:
        log.info("[writer_agent] LLM narrative generated (%d chars)", len(narrative))

    # ── Generate the SAR (calls sar.generate_sar, saves to disk) ─────────────
    try:
        # Build analyst_notes that summarise the graph + precedent evidence
        graph      = state.get("graph_intel", {})
        rings      = state.get("fraud_rings", [])
        cases      = state.get("retrieved_cases", [])
        ring_note  = (
            f"Ring member: {len(rings)} ring(s). "
            f"Max ring fraud rate: {graph.get('max_ring_fraud_rate', 0):.1%}. "
            if rings else "No ring membership. "
        )
        prec_note  = (
            f"Precedents: {len(cases)} similar case(s) retrieved "
            f"(best match score {cases[0]['score']:.3f})."
            if cases else "No historical precedents found."
        )
        analyst_notes = ring_note + prec_note

        sar_draft = generate_sar(
            transaction=txn,
            fraud_probability=prob,
            shap_factors=shap,
            analyst_notes=analyst_notes,
            threshold=0.3,
        )
        # Override the auto-generated narrative with our enriched version
        # and ensure the status stays DRAFT (never auto-filed)
        sar_draft["narrative"] = narrative
        sar_draft["status"]    = "DRAFT"
        sar_draft["copilot_generated"] = True

        # Persist the enriched version back to disk
        import json
        path = os.path.join(SAR_DIR, f"{sar_draft['sar_id']}.json")
        with open(path, "w") as fh:
            json.dump(sar_draft, fh, indent=2)

        record_sar_draft("drafted")
        log.info("[writer_agent] SAR draft saved: %s", sar_draft["sar_id"])

    except Exception as exc:
        log.error("[writer_agent] generate_sar failed: %s", exc)
        record_sar_draft("error")
        # Minimal stub so downstream HITL enqueue doesn't crash
        sar_draft = {
            "sar_id":    "SAR-ERROR",
            "status":    "DRAFT",
            "narrative": narrative,
            "error":     str(exc),
        }

    return {**state, "sar_draft": sar_draft, "narrative": narrative}


# ══════════════════════════════════════════════════════════════════════════════
# HITL router — runs after writer_node, enqueues into add_to_review_queue()
# ══════════════════════════════════════════════════════════════════════════════

@instrument_agent("hitl_router")
def hitl_router_node(state: CopilotState) -> CopilotState:
    """
    Routes the SAR draft to the existing HITL review queue instead of
    auto-filing.  Enriches the queue item's 'reason' field with the copilot
    investigation summary so the analyst sees everything in one place.
    """
    sar   = state.get("sar_draft", {})
    txn   = state.get("transaction", {})
    prob  = state.get("fraud_probability", 0.0)
    graph = state.get("graph_intel", {})
    cases = state.get("retrieved_cases", [])

    sar_id     = sar.get("sar_id", "SAR-unknown")
    ring_note  = (
        f"Ring member (×{graph.get('ring_count', 0)} rings, "
        f"max rate {graph.get('max_ring_fraud_rate', 0):.1%}). "
        if graph.get("ring_member") else ""
    )
    prec_note  = (
        f"{len(cases)} precedent(s) retrieved. "
        if cases else ""
    )
    reason = (
        f"Copilot draft {sar_id}. "
        f"{ring_note}{prec_note}"
        f"Graph risk={graph.get('graph_risk_score', 0):.4f}."
    )

    try:
        add_to_review_queue(txn, prob, reason)
        record_hitl_queued()
        log.info("[hitl_router] enqueued SAR %s for analyst review", sar_id)
    except Exception as exc:
        # add_to_review_queue uses st.session_state; may fail outside Streamlit
        log.warning(
            "[hitl_router] Could not enqueue to HITL queue "
            "(running outside Streamlit?): %s", exc
        )

    return {**state, "hitl_item_index": None}


# ══════════════════════════════════════════════════════════════════════════════
# Graph assembly
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph() -> "StateGraph | None":
    """
    Compile the LangGraph StateGraph.
    Returns None (with a warning) if langgraph is not installed.
    """
    if not _LG_OK:
        return None

    builder = StateGraph(CopilotState)

    builder.add_node("retriever",    retriever_node)
    builder.add_node("tool",         tool_node)
    builder.add_node("writer",       writer_node)
    builder.add_node("hitl_router",  hitl_router_node)

    # Linear edges: retriever → tool → writer → hitl_router → END
    builder.set_entry_point("retriever")
    builder.add_edge("retriever",   "tool")
    builder.add_edge("tool",        "writer")
    builder.add_edge("writer",      "hitl_router")
    builder.add_edge("hitl_router", END)

    return builder.compile()


# Module-level compiled graph (lazy initialisation — compiled on first call)
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ══════════════════════════════════════════════════════════════════════════════
# Linear fallback (no LangGraph dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _run_linear(initial_state: CopilotState) -> CopilotState:
    """
    Execute the same three-node pipeline without LangGraph, used when
    langgraph is not installed.  Shares all node implementations so
    behaviour is identical.
    """
    state = initial_state.copy()
    for node_fn in (retriever_node, tool_node, writer_node, hitl_router_node):
        try:
            state = node_fn(state)
        except Exception as exc:
            log.error("[linear_fallback] node %s raised: %s", node_fn.__name__, exc)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_copilot(
    transaction: dict,
    fraud_probability: float,
    model=None,
    input_df: "pd.DataFrame | None" = None,
    feature_names: "list[str] | None" = None,
) -> dict:
    """
    Run the full Fraud Investigation Copilot pipeline for one transaction.

    Parameters
    ----------
    transaction : dict
        Raw transaction fields (same schema as hitl.add_to_review_queue).
    fraud_probability : float
        Model output score in [0, 1].
    model : sklearn / XGB estimator, optional
        If provided, SHAP factors will be computed.
    input_df : pd.DataFrame, optional
        Single-row DataFrame aligned to the model's feature set.
    feature_names : list[str], optional
        Feature names matching input_df columns.

    Returns
    -------
    dict
        Final CopilotState with all agent outputs populated:
        {
          "transaction":       dict,
          "fraud_probability": float,
          "retrieved_cases":   list[dict],
          "graph_intel":       dict,
          "fraud_rings":       list[dict],
          "shap_factors":      list[dict],
          "sar_draft":         dict,    # status always "DRAFT"
          "narrative":         str,
          "hitl_item_index":   None,
        }
    """
    initial: CopilotState = {
        "transaction":       transaction,
        "fraud_probability": float(fraud_probability),
        "model":             model,
        "input_df":          input_df,
        "feature_names":     feature_names or [],
        "retrieved_cases":   [],
        "graph_intel":       {},
        "fraud_rings":       [],
        "shap_factors":      [],
        "sar_draft":         {},
        "narrative":         "",
        "hitl_item_index":   None,
    }

    graph = _get_graph()
    if graph is not None:
        log.info("[copilot] running via LangGraph StateGraph")
        try:
            final_state: CopilotState = graph.invoke(initial)
        except Exception as exc:
            log.error("[copilot] LangGraph invocation failed, falling back to linear: %s", exc)
            final_state = _run_linear(initial)
    else:
        log.info("[copilot] running via linear fallback (langgraph not installed)")
        final_state = _run_linear(initial)

    # Strip non-serialisable fields before returning
    return {
        k: v for k, v in final_state.items()
        if k not in ("model", "input_df")
    }
