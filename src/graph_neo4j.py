"""
Enterprise Graph Infrastructure — Neo4j / Amazon Neptune adapter.
Provides the same interface as graph_intelligence.py but backed by
a property graph database instead of in-memory NetworkX.

Adapter pattern:
  - If NEO4J_URI is set → uses Neo4j Bolt driver
  - If NEPTUNE_ENDPOINT is set → uses Gremlin (openCypher via boto3)
  - Falls back to NetworkX (graph_intelligence.py) automatically

Usage:
    from src.graph_neo4j import GraphAdapter
    g = GraphAdapter()
    rings = g.detect_fraud_rings(df)
    features = g.extract_features(df)
"""
from __future__ import annotations
import os
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional

# ── Neo4j driver (optional) ────────────────────────────────────────────────────
try:
    from neo4j import GraphDatabase, basic_auth
    _NEO4J_DRIVER = True
except ImportError:
    _NEO4J_DRIVER = False

# ── NetworkX fallback ──────────────────────────────────────────────────────────
from src.graph_intelligence import (
    detect_fraud_rings as _nx_rings,
    extract_graph_features as _nx_features,
    load_ring_log, _save_rings,
)


class GraphAdapter:
    """
    Unified graph adapter. Auto-selects backend:
      1. Neo4j (if NEO4J_URI configured)
      2. Amazon Neptune via openCypher (if NEPTUNE_ENDPOINT configured)
      3. NetworkX in-process fallback
    """

    def __init__(self):
        self._backend = "networkx"
        self._driver = None
        self._init_backend()

    def _init_backend(self):
        neo4j_uri = os.environ.get("NEO4J_URI", "")
        neptune_ep = os.environ.get("NEPTUNE_ENDPOINT", "")

        if neo4j_uri and _NEO4J_DRIVER:
            try:
                user = os.environ.get("NEO4J_USER", "neo4j")
                pwd  = os.environ.get("NEO4J_PASSWORD", "")
                self._driver = GraphDatabase.driver(
                    neo4j_uri, auth=basic_auth(user, pwd)
                )
                self._driver.verify_connectivity()
                self._backend = "neo4j"
                print(f"[Graph] Connected to Neo4j at {neo4j_uri}")
            except Exception as e:
                print(f"[Graph] Neo4j unavailable ({e}) — falling back to NetworkX")

        elif neptune_ep:
            self._backend = "neptune"
            print(f"[Graph] Neptune endpoint configured: {neptune_ep}")

    @property
    def backend(self) -> str:
        return self._backend

    # ── Public interface ───────────────────────────────────────────────────────

    def detect_fraud_rings(self, df: pd.DataFrame,
                            min_fraud_rate: float = 0.1,
                            min_ring_size: int = 2) -> list[dict]:
        if self._backend == "neo4j" and self._driver:
            return self._neo4j_detect_rings(df, min_fraud_rate, min_ring_size)
        elif self._backend == "neptune":
            return self._neptune_detect_rings(df, min_fraud_rate, min_ring_size)
        return _nx_rings(df, min_fraud_rate=min_fraud_rate,
                         min_ring_size=min_ring_size)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._backend == "neo4j" and self._driver:
            return self._neo4j_features(df)
        return _nx_features(df)

    def upsert_transaction(self, customer_id: str, merchant_id: str,
                            amount: float, is_fraud: int = 0):
        """Write a single transaction edge to the graph DB."""
        if self._backend == "neo4j" and self._driver:
            self._neo4j_upsert(customer_id, merchant_id, amount, is_fraud)

    def close(self):
        if self._driver:
            self._driver.close()

    # ── Neo4j implementation ───────────────────────────────────────────────────

    def _neo4j_upsert(self, customer_id: str, merchant_id: str,
                       amount: float, is_fraud: int):
        with self._driver.session() as session:
            session.run("""
                MERGE (c:Customer {id: $cid})
                MERGE (m:Merchant {id: $mid})
                MERGE (c)-[t:TRANSACTED_WITH]->(m)
                ON CREATE SET t.count = 1, t.total_amount = $amount,
                              t.fraud_count = $is_fraud
                ON MATCH SET  t.count = t.count + 1,
                              t.total_amount = t.total_amount + $amount,
                              t.fraud_count = t.fraud_count + $is_fraud
                SET c.last_seen = $now, m.last_seen = $now
            """, cid=str(customer_id), mid=str(merchant_id),
                 amount=float(amount), is_fraud=int(is_fraud),
                 now=datetime.now(timezone.utc).isoformat())

    def _neo4j_bulk_load(self, df: pd.DataFrame):
        """Bulk load a DataFrame into Neo4j."""
        label_col = "label" if "label" in df.columns else None
        with self._driver.session() as session:
            for _, row in df.iterrows():
                session.run("""
                    MERGE (c:Customer {id: $cid})
                    MERGE (m:Merchant {id: $mid})
                    MERGE (c)-[t:TRANSACTED_WITH]->(m)
                    ON CREATE SET t.count = 1, t.total_amount = $amount,
                                  t.fraud_count = $fraud
                    ON MATCH  SET t.count = t.count + 1,
                                  t.total_amount = t.total_amount + $amount,
                                  t.fraud_count  = t.fraud_count + $fraud
                """, cid=str(row["customer_id"]), mid=str(row["merchant_id"]),
                     amount=float(row.get("transaction_amount", 0)),
                     fraud=int(row[label_col]) if label_col else 0)

    def _neo4j_detect_rings(self, df: pd.DataFrame,
                              min_fraud_rate: float,
                              min_ring_size: int) -> list[dict]:
        """
        Use Neo4j Cypher to find merchant hubs with elevated fraud rates.
        Falls back to NetworkX if query fails.
        """
        try:
            self._neo4j_bulk_load(df)
            with self._driver.session() as session:
                result = session.run("""
                    MATCH (c:Customer)-[t:TRANSACTED_WITH]->(m:Merchant)
                    WITH m,
                         count(DISTINCT c) AS customer_count,
                         sum(t.fraud_count) AS total_fraud,
                         sum(t.count) AS total_tx,
                         sum(t.total_amount) AS total_amount
                    WHERE customer_count >= $min_size
                      AND total_tx > 0
                      AND toFloat(total_fraud) / total_tx >= $min_rate
                    RETURN m.id AS merchant_id,
                           customer_count, total_fraud, total_tx, total_amount,
                           toFloat(total_fraud) / total_tx AS fraud_rate
                    ORDER BY fraud_rate DESC
                    LIMIT 50
                """, min_size=min_ring_size, min_rate=min_fraud_rate)

                rings = []
                for i, rec in enumerate(result):
                    rings.append({
                        "ring_id":         f"NEO4J-RING-{i:04d}",
                        "ring_type":       "neo4j_merchant_hub",
                        "detected_at":     datetime.now(timezone.utc).isoformat(),
                        "size":            int(rec["customer_count"]) + 1,
                        "customer_count":  int(rec["customer_count"]),
                        "merchant_count":  1,
                        "fraud_rate":      round(float(rec["fraud_rate"]), 4),
                        "total_amount":    round(float(rec["total_amount"]), 2),
                        "total_fraud_txn": int(rec["total_fraud"]),
                        "hub_nodes":       [f"M_{rec['merchant_id']}"],
                        "risk_level":      "CRITICAL" if rec["fraud_rate"] >= 0.7 else "HIGH",
                        "members":         [f"M_{rec['merchant_id']}"],
                    })
                _save_rings(rings)
                return rings
        except Exception as e:
            print(f"[Graph] Neo4j ring detection failed ({e}) — using NetworkX")
            return _nx_rings(df, min_fraud_rate=min_fraud_rate,
                             min_ring_size=min_ring_size)

    def _neo4j_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract per-transaction features from Neo4j graph."""
        try:
            self._neo4j_bulk_load(df)
            with self._driver.session() as session:
                rows = []
                for _, row in df.iterrows():
                    cid = str(row["customer_id"])
                    mid = str(row["merchant_id"])
                    result = session.run("""
                        MATCH (c:Customer {id: $cid})-[t:TRANSACTED_WITH]->(m:Merchant {id: $mid})
                        OPTIONAL MATCH (c)-[:TRANSACTED_WITH]->(other_m:Merchant)
                        OPTIONAL MATCH (other_c:Customer)-[:TRANSACTED_WITH]->(m)
                        RETURN
                            count(DISTINCT other_m) AS c_degree,
                            count(DISTINCT other_c) AS m_degree,
                            COALESCE(t.fraud_count, 0) AS edge_fraud,
                            COALESCE(t.count, 1) AS edge_count
                    """, cid=cid, mid=mid)
                    rec = result.single()
                    if rec:
                        c_deg = int(rec["c_degree"])
                        m_deg = int(rec["m_degree"])
                        edge_fr = int(rec["edge_fraud"]) / max(int(rec["edge_count"]), 1)
                    else:
                        c_deg, m_deg, edge_fr = 0, 0, 0.0

                    rows.append({
                        "customer_degree":     c_deg,
                        "merchant_degree":     m_deg,
                        "customer_fraud_rate": round(edge_fr, 4),
                        "merchant_fraud_rate": round(edge_fr, 4),
                        "customer_betweenness": 0.0,
                        "merchant_betweenness": 0.0,
                        "graph_risk_score":    round(
                            0.3 * np.log1p(c_deg) + 0.3 * np.log1p(m_deg) + 0.4 * edge_fr, 4
                        ),
                        "ring_member":         0,
                    })
                return pd.DataFrame(rows, index=df.index)
        except Exception as e:
            print(f"[Graph] Neo4j feature extraction failed ({e}) — using NetworkX")
            return _nx_features(df)

    # ── Neptune stub ───────────────────────────────────────────────────────────

    def _neptune_detect_rings(self, df: pd.DataFrame,
                               min_fraud_rate: float,
                               min_ring_size: int) -> list[dict]:
        """
        Neptune via openCypher (boto3 + requests).
        Stub — falls back to NetworkX. Extend with:
          import requests
          endpoint = os.environ['NEPTUNE_ENDPOINT']
          response = requests.post(f"https://{endpoint}:8182/openCypher",
                                   json={"query": cypher_query})
        """
        print("[Graph] Neptune backend — falling back to NetworkX (extend with boto3)")
        return _nx_rings(df, min_fraud_rate=min_fraud_rate,
                         min_ring_size=min_ring_size)


# ── Module-level singleton ─────────────────────────────────────────────────────
_ADAPTER: Optional[GraphAdapter] = None


def get_graph_adapter() -> GraphAdapter:
    global _ADAPTER
    if _ADAPTER is None:
        _ADAPTER = GraphAdapter()
    return _ADAPTER


# ── Drop-in replacements for graph_intelligence functions ─────────────────────

def detect_fraud_rings(df: pd.DataFrame,
                        min_fraud_rate: float = 0.1,
                        min_ring_size: int = 2) -> list[dict]:
    return get_graph_adapter().detect_fraud_rings(df, min_fraud_rate, min_ring_size)


def extract_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    return get_graph_adapter().extract_features(df)


def plot_fraud_ring_network(rings: list[dict], max_rings: int = 3):
    from src.graph_intelligence import plot_fraud_ring_network as _plot
    return _plot(rings, max_rings)
