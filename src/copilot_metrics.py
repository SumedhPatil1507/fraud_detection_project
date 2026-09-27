"""
Fraud Investigation Copilot — Prometheus Metrics
=================================================
Follows the exact same conventions as src/observability.py:
  - Prefix:  fraudguard_
  - No-op fallback via _NoOpMetric when prometheus_client is absent
  - All metric objects are module-level singletons
  - Helper functions mirror record_prediction / update_psi_metrics style

Metrics defined here
--------------------
  fraudguard_copilot_agent_runs_total          Counter   agent, status
  fraudguard_copilot_agent_latency_seconds     Histogram agent
  fraudguard_copilot_tool_calls_total          Counter   tool, status
  fraudguard_copilot_retriever_hits            Histogram (number of docs returned)
  fraudguard_copilot_sar_drafts_total          Counter   status (drafted|error)
  fraudguard_copilot_hitl_queued_total         Counter   (no labels)
  fraudguard_copilot_graph_queries_total       Counter   status
  fraudguard_copilot_shap_queries_total        Counter   status
  fraudguard_copilot_graph_errors_total        Counter   agent (unexpected exceptions)
"""
from __future__ import annotations

import time
import functools
from typing import Callable, Any

# ── Reuse the graceful-fallback pattern from observability.py ─────────────────
try:
    from prometheus_client import Counter, Histogram, Gauge
    _PROM = True
except ImportError:
    _PROM = False
    Counter = Histogram = Gauge = None  # type: ignore[assignment,misc]


class _NoOpMetric:
    """Silent no-op — identical interface to the real prometheus_client types."""

    def labels(self, **_) -> "_NoOpMetric":
        return self

    def observe(self, *_) -> None:
        pass

    def inc(self, *_) -> None:
        pass

    def set(self, *_) -> None:
        pass

    def time(self):
        import contextlib
        return contextlib.nullcontext()


def _metric(metric_class, name: str, doc: str, labels=None, buckets=None):
    """Create a real metric or silently return a no-op."""
    if not _PROM:
        return _NoOpMetric()
    try:
        kwargs: dict = {}
        if labels:
            kwargs["labelnames"] = labels
        if buckets:
            kwargs["buckets"] = buckets
        return metric_class(name, doc, **kwargs)
    except Exception:
        return _NoOpMetric()


# ── Metric definitions ────────────────────────────────────────────────────────

# Total agent node invocations, broken down by agent name and outcome
COPILOT_AGENT_RUNS = _metric(
    Counter,
    "fraudguard_copilot_agent_runs_total",
    "Total copilot agent node invocations",
    labels=["agent", "status"],           # status: success | error
)

# Wall-clock time each agent node spends (includes LLM / tool round-trips)
COPILOT_AGENT_LATENCY = _metric(
    Histogram,
    "fraudguard_copilot_agent_latency_seconds",
    "Copilot agent node wall-clock latency",
    labels=["agent"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# Individual tool calls made by the ToolAgent
COPILOT_TOOL_CALLS = _metric(
    Counter,
    "fraudguard_copilot_tool_calls_total",
    "Total tool calls dispatched by the copilot ToolAgent",
    labels=["tool", "status"],            # tool: graph_intel | shap_factors
)                                         # status: success | error

# Distribution of how many documents the RetrieverAgent returns per query
COPILOT_RETRIEVER_HITS = _metric(
    Histogram,
    "fraudguard_copilot_retriever_hits",
    "Number of historical cases retrieved per copilot query",
    buckets=[0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
)

# SAR drafts produced by the WriterAgent
COPILOT_SAR_DRAFTS = _metric(
    Counter,
    "fraudguard_copilot_sar_drafts_total",
    "SAR drafts produced by the copilot WriterAgent",
    labels=["status"],                    # status: drafted | error
)

# Items pushed into the HITL queue by the copilot
COPILOT_HITL_QUEUED = _metric(
    Counter,
    "fraudguard_copilot_hitl_queued_total",
    "Number of copilot SAR drafts routed to the HITL review queue",
)

# Graph intelligence queries (score_transaction_graph / detect_fraud_rings)
COPILOT_GRAPH_QUERIES = _metric(
    Counter,
    "fraudguard_copilot_graph_queries_total",
    "Graph intelligence tool calls from the copilot ToolAgent",
    labels=["status"],
)

# SHAP factor queries
COPILOT_SHAP_QUERIES = _metric(
    Counter,
    "fraudguard_copilot_shap_queries_total",
    "SHAP top-driver queries from the copilot ToolAgent",
    labels=["status"],
)

# Unexpected exceptions bubbling out of an agent node
COPILOT_ERRORS = _metric(
    Counter,
    "fraudguard_copilot_graph_errors_total",
    "Unhandled exceptions inside copilot agent nodes",
    labels=["agent"],
)


# ── Helper functions (mirror observability.py style) ─────────────────────────

def record_agent_run(agent: str, status: str, elapsed: float) -> None:
    """Record a completed agent node execution."""
    try:
        COPILOT_AGENT_RUNS.labels(agent=agent, status=status).inc()
        COPILOT_AGENT_LATENCY.labels(agent=agent).observe(elapsed)
    except Exception:
        pass


def record_tool_call(tool: str, status: str) -> None:
    """Record a single tool call from the ToolAgent."""
    try:
        COPILOT_TOOL_CALLS.labels(tool=tool, status=status).inc()
        if tool == "graph_intel":
            COPILOT_GRAPH_QUERIES.labels(status=status).inc()
        elif tool == "shap_factors":
            COPILOT_SHAP_QUERIES.labels(status=status).inc()
    except Exception:
        pass


def record_retriever_hits(n_docs: int) -> None:
    """Record how many documents the retriever returned."""
    try:
        COPILOT_RETRIEVER_HITS.observe(n_docs)
    except Exception:
        pass


def record_sar_draft(status: str) -> None:
    """Record a SAR draft attempt (drafted | error)."""
    try:
        COPILOT_SAR_DRAFTS.labels(status=status).inc()
    except Exception:
        pass


def record_hitl_queued() -> None:
    """Record that one copilot draft was pushed to the HITL queue."""
    try:
        COPILOT_HITL_QUEUED.inc()
    except Exception:
        pass


def record_agent_error(agent: str) -> None:
    """Record an unhandled exception in an agent node."""
    try:
        COPILOT_ERRORS.labels(agent=agent).inc()
    except Exception:
        pass


# ── Decorator ─────────────────────────────────────────────────────────────────

def instrument_agent(agent_name: str):
    """
    Decorator that wraps a synchronous agent-node function and automatically
    records run count, latency, and errors — matching the @track_latency
    pattern in observability.py.

    Usage::

        @instrument_agent("retriever")
        def retriever_node(state: CopilotState) -> CopilotState:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            status = "success"
            try:
                return fn(*args, **kwargs)
            except Exception:
                status = "error"
                record_agent_error(agent_name)
                raise
            finally:
                elapsed = time.perf_counter() - start
                record_agent_run(agent_name, status, elapsed)
        return wrapper
    return decorator
