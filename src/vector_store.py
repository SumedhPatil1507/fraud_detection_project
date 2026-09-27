"""
Fraud Investigation Copilot — Vector Store
==========================================
Embeds and indexes two document collections in ChromaDB:

  audit_log    — rows from outputs/predictions_audit.csv
                 (one document per prediction event)
  sar_reports  — filed SAR JSONs from outputs/sar_reports/
                 (one document per SAR, indexed on narrative + risk indicators)

Public API
----------
  index_audit_log()                        -> int   (docs upserted)
  index_sar_reports()                      -> int   (docs upserted)
  search_audit_log(query, top_k)           -> list[dict]
  search_sar_reports(query, top_k)         -> list[dict]
  search_all(query, top_k)                 -> list[dict]   (merged, deduplicated)
  get_collection_stats()                   -> dict

Design notes
------------
- ChromaDB is used with its built-in cosine-similarity search.
- Embeddings are produced locally by sentence-transformers (no API key needed).
- The embedding function is wrapped in a thin ChromaDB EmbeddingFunction so
  ChromaDB handles batching and caching transparently.
- Documents are upserted (not re-inserted) so calling index_* repeatedly is safe.
- All public functions degrade gracefully: if chromadb or
  sentence-transformers are not installed they return empty results and log a
  warning instead of raising.
"""
from __future__ import annotations

import os
import json
import logging
from typing import Any

import pandas as pd

from src.config import (
    AUDIT_LOG_PATH,
    CHROMA_PERSIST_DIR,
    CHROMA_AUDIT_COLLECTION,
    CHROMA_SAR_COLLECTION,
    EMBEDDING_MODEL,
    COPILOT_TOP_K,
)

log = logging.getLogger(__name__)

# ── Optional heavy imports ─────────────────────────────────────────────────────
try:
    import chromadb
    from chromadb import EmbeddingFunction, Documents, Embeddings
    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False
    log.warning("[vector_store] chromadb not installed — vector search disabled")

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except ImportError:
    _ST_OK = False
    log.warning("[vector_store] sentence-transformers not installed — vector search disabled")

# SAR reports directory (mirrors sar.py's SAR_DIR)
_SAR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "sar_reports"
)

_client: "chromadb.PersistentClient | None" = None
_embedder: "SentenceTransformer | None" = None


# ── ChromaDB embedding function wrapper ───────────────────────────────────────

class _STEmbeddingFunction:
    """Wraps SentenceTransformer to satisfy ChromaDB's EmbeddingFunction protocol."""

    def __init__(self, model: "SentenceTransformer") -> None:
        self._model = model

    def __call__(self, input: "Documents") -> "Embeddings":  # noqa: A002
        vecs = self._model.encode(list(input), convert_to_numpy=True, show_progress_bar=False)
        return vecs.tolist()


# ── Lazy singletons ───────────────────────────────────────────────────────────

def _get_embedder() -> "SentenceTransformer | None":
    global _embedder
    if _embedder is None and _ST_OK:
        try:
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
            log.info("[vector_store] Loaded embedding model: %s", EMBEDDING_MODEL)
        except Exception as exc:
            log.error("[vector_store] Failed to load embedding model: %s", exc)
    return _embedder


def _get_client() -> "chromadb.PersistentClient | None":
    global _client
    if _client is None and _CHROMA_OK:
        try:
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            log.info("[vector_store] ChromaDB client at %s", CHROMA_PERSIST_DIR)
        except Exception as exc:
            log.error("[vector_store] Failed to init ChromaDB client: %s", exc)
    return _client


def _get_collection(name: str) -> "chromadb.Collection | None":
    """Return (creating if necessary) a collection that uses our local embedder."""
    client = _get_client()
    embedder = _get_embedder()
    if client is None or embedder is None:
        return None
    try:
        ef = _STEmbeddingFunction(embedder)
        return client.get_or_create_collection(
            name=name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        log.error("[vector_store] Failed to get collection '%s': %s", name, exc)
        return None


# ── Document builders ─────────────────────────────────────────────────────────

def _audit_row_to_doc(row: "pd.Series", idx: int) -> tuple[str, str, dict]:
    """Convert one audit-log row to (doc_id, text, metadata)."""
    doc_id = f"audit_{idx}"
    amount  = row.get("transaction_amount", 0)
    dist    = row.get("distance_from_home_km", 0)
    hour    = row.get("hour", 0)
    prob    = row.get("fraud_probability", 0)
    is_frau = row.get("is_fraud", False)
    ts      = row.get("timestamp", "")

    text = (
        f"Transaction at {ts}: amount=${float(amount):.2f}, "
        f"distance={float(dist):.1f}km, hour={hour}, "
        f"fraud_probability={float(prob):.4f}, "
        f"verdict={'FRAUD' if is_frau else 'LEGITIMATE'}."
    )
    meta = {
        "source":               "audit_log",
        "timestamp":            str(ts),
        "transaction_amount":   float(amount),
        "distance_from_home_km": float(dist),
        "hour":                 int(hour),
        "fraud_probability":    float(prob),
        "is_fraud":             bool(is_frau),
    }
    return doc_id, text, meta


def _sar_to_doc(sar: dict) -> tuple[str, str, dict]:
    """Convert one SAR dict to (doc_id, text, metadata)."""
    sar_id    = sar.get("sar_id", "SAR-unknown")
    narrative = sar.get("narrative", "")
    indicators = "; ".join(sar.get("risk_indicators", []))
    rec_action = sar.get("recommended_action", "")
    ml         = sar.get("ml_assessment", {})
    prob       = ml.get("fraud_probability", 0)
    risk_level = ml.get("risk_level", "MEDIUM")
    shap_top   = "; ".join(ml.get("top_shap_factors", [])[:3])
    status     = sar.get("status", "DRAFT")
    generated  = sar.get("generated_at", "")
    txn        = sar.get("transaction", {})
    amount     = txn.get("amount_usd", 0)

    text = (
        f"SAR {sar_id} ({status}) generated {generated}: "
        f"amount=${float(amount):.2f}, fraud_probability={float(prob):.4f}, "
        f"risk={risk_level}. "
        f"Narrative: {narrative} "
        f"Indicators: {indicators}. "
        f"Top drivers: {shap_top}. "
        f"Action: {rec_action}."
    )
    meta = {
        "source":             "sar_reports",
        "sar_id":             sar_id,
        "status":             status,
        "generated_at":       generated,
        "fraud_probability":  float(prob),
        "risk_level":         risk_level,
        "amount_usd":         float(amount),
        "recommended_action": rec_action,
    }
    return sar_id, text, meta


# ── Indexing functions ────────────────────────────────────────────────────────

def index_audit_log() -> int:
    """
    Embed every row of the audit-log CSV and upsert into the audit collection.
    Safe to call repeatedly — existing documents are overwritten in place.

    Returns
    -------
    int
        Number of documents upserted (0 on error or empty log).
    """
    col = _get_collection(CHROMA_AUDIT_COLLECTION)
    if col is None:
        return 0

    if not os.path.exists(AUDIT_LOG_PATH):
        log.info("[vector_store] Audit log not found at %s — nothing to index", AUDIT_LOG_PATH)
        return 0

    try:
        df = pd.read_csv(AUDIT_LOG_PATH)
    except Exception as exc:
        log.error("[vector_store] Failed to read audit log: %s", exc)
        return 0

    if df.empty:
        return 0

    ids, docs, metas = [], [], []
    for idx, row in df.iterrows():
        doc_id, text, meta = _audit_row_to_doc(row, int(idx))
        ids.append(doc_id)
        docs.append(text)
        metas.append(meta)

    try:
        # Upsert in batches of 500 to avoid ChromaDB request-size limits
        batch = 500
        for start in range(0, len(ids), batch):
            col.upsert(
                ids=ids[start : start + batch],
                documents=docs[start : start + batch],
                metadatas=metas[start : start + batch],
            )
        log.info("[vector_store] Upserted %d audit-log docs", len(ids))
        return len(ids)
    except Exception as exc:
        log.error("[vector_store] Upsert failed for audit log: %s", exc)
        return 0


def index_sar_reports() -> int:
    """
    Embed every SAR JSON file and upsert into the SAR collection.
    Safe to call repeatedly.

    Returns
    -------
    int
        Number of documents upserted (0 on error or empty directory).
    """
    col = _get_collection(CHROMA_SAR_COLLECTION)
    if col is None:
        return 0

    if not os.path.exists(_SAR_DIR):
        log.info("[vector_store] SAR directory not found — nothing to index")
        return 0

    ids, docs, metas = [], [], []
    for fname in os.listdir(_SAR_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(_SAR_DIR, fname)) as fh:
                sar = json.load(fh)
            doc_id, text, meta = _sar_to_doc(sar)
            ids.append(doc_id)
            docs.append(text)
            metas.append(meta)
        except Exception as exc:
            log.warning("[vector_store] Skipping %s: %s", fname, exc)

    if not ids:
        return 0

    try:
        batch = 500
        for start in range(0, len(ids), batch):
            col.upsert(
                ids=ids[start : start + batch],
                documents=docs[start : start + batch],
                metadatas=metas[start : start + batch],
            )
        log.info("[vector_store] Upserted %d SAR docs", len(ids))
        return len(ids)
    except Exception as exc:
        log.error("[vector_store] Upsert failed for SAR reports: %s", exc)
        return 0


# ── Search functions ──────────────────────────────────────────────────────────

def _result_to_dicts(result: dict, source_tag: str) -> list[dict]:
    """Flatten a ChromaDB query result into a list of plain dicts."""
    hits = []
    docs      = (result.get("documents")  or [[]])[0]
    metas     = (result.get("metadatas")  or [[]])[0]
    distances = (result.get("distances")  or [[]])[0]
    ids_      = (result.get("ids")        or [[]])[0]

    for doc_id, text, meta, dist in zip(ids_, docs, metas, distances):
        hits.append({
            "id":           doc_id,
            "source":       source_tag,
            "text":         text,
            "metadata":     meta or {},
            # Convert cosine distance → similarity score in [0, 1]
            "score":        round(1.0 - float(dist), 4),
        })
    return hits


def search_audit_log(query: str, top_k: int = COPILOT_TOP_K) -> list[dict]:
    """
    Semantic search over the audit-log collection.

    Parameters
    ----------
    query : str
        Natural-language description of the case being investigated.
    top_k : int
        Maximum number of results to return.

    Returns
    -------
    list[dict]
        Each dict has keys: id, source, text, metadata, score.
        Empty list on error or when the collection is empty.
    """
    col = _get_collection(CHROMA_AUDIT_COLLECTION)
    if col is None:
        return []
    try:
        n = col.count()
        if n == 0:
            return []
        result = col.query(
            query_texts=[query],
            n_results=min(top_k, n),
            include=["documents", "metadatas", "distances"],
        )
        return _result_to_dicts(result, "audit_log")
    except Exception as exc:
        log.error("[vector_store] audit_log search failed: %s", exc)
        return []


def search_sar_reports(query: str, top_k: int = COPILOT_TOP_K) -> list[dict]:
    """
    Semantic search over the SAR-report collection.

    Returns
    -------
    list[dict]
        Each dict has keys: id, source, text, metadata, score.
    """
    col = _get_collection(CHROMA_SAR_COLLECTION)
    if col is None:
        return []
    try:
        n = col.count()
        if n == 0:
            return []
        result = col.query(
            query_texts=[query],
            n_results=min(top_k, n),
            include=["documents", "metadatas", "distances"],
        )
        return _result_to_dicts(result, "sar_reports")
    except Exception as exc:
        log.error("[vector_store] sar_reports search failed: %s", exc)
        return []


def search_all(query: str, top_k: int = COPILOT_TOP_K) -> list[dict]:
    """
    Search both collections and return the top-k results merged by score.

    Duplicates (same id from both collections) are deduplicated, keeping the
    higher-scoring hit.

    Returns
    -------
    list[dict]
        Sorted by score descending, at most top_k items.
    """
    audit_hits = search_audit_log(query, top_k=top_k)
    sar_hits   = search_sar_reports(query, top_k=top_k)

    seen: dict[str, dict] = {}
    for hit in audit_hits + sar_hits:
        hid = hit["id"]
        if hid not in seen or hit["score"] > seen[hid]["score"]:
            seen[hid] = hit

    return sorted(seen.values(), key=lambda h: h["score"], reverse=True)[:top_k]


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_collection_stats() -> dict[str, Any]:
    """
    Return counts for both collections plus backend metadata.

    Returns
    -------
    dict
        {
          "audit_log_docs": int,
          "sar_docs": int,
          "embed_model": str,
          "persist_dir": str,
          "chroma_available": bool,
          "st_available": bool,
        }
    """
    audit_count = 0
    sar_count   = 0
    if _CHROMA_OK and _ST_OK:
        try:
            c_audit = _get_collection(CHROMA_AUDIT_COLLECTION)
            if c_audit is not None:
                audit_count = c_audit.count()
        except Exception:
            pass
        try:
            c_sar = _get_collection(CHROMA_SAR_COLLECTION)
            if c_sar is not None:
                sar_count = c_sar.count()
        except Exception:
            pass

    return {
        "audit_log_docs":   audit_count,
        "sar_docs":         sar_count,
        "embed_model":      EMBEDDING_MODEL,
        "persist_dir":      CHROMA_PERSIST_DIR,
        "chroma_available": _CHROMA_OK,
        "st_available":     _ST_OK,
    }
