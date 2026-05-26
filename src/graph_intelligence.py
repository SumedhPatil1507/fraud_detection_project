"""
Graph Intelligence — Real-Time Fraud Ring Detector
Builds a bipartite customer-merchant graph using NetworkX.
Detects fraud rings via community detection, centrality analysis,
and suspicious subgraph extraction.

Ring detection strategy:
  - A "ring" is a connected component where ≥1 customer shares ≥1 merchant
    with another customer (merchant_degree >= 2), AND the component has
    elevated fraud rate.
  - Falls back to merchant-centric rings when customer IDs are sparse.
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

try:
    import networkx as nx
    _NX = True
except ImportError:
    _NX = False

RING_LOG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "fraud_rings.json"
)


# ── Graph Construction ─────────────────────────────────────────────────────────

def build_transaction_graph(df: pd.DataFrame) -> Optional["nx.Graph"]:
    """
    Build a weighted bipartite graph: customers ↔ merchants.
    Edge weight = number of shared transactions.
    Node attributes: fraud_rate, total_amount, degree.
    """
    if not _NX:
        return None
    if "customer_id" not in df.columns or "merchant_id" not in df.columns:
        return None

    G = nx.Graph()
    label_col = "label" if "label" in df.columns else None

    for _, row in df.iterrows():
        cid = f"C_{row['customer_id']}"
        mid = f"M_{row['merchant_id']}"
        amount = float(row.get("transaction_amount", 0))
        is_fraud = int(row[label_col]) if label_col else 0

        for nid, ntype in [(cid, "customer"), (mid, "merchant")]:
            if not G.has_node(nid):
                G.add_node(nid, node_type=ntype, total_amount=0,
                           fraud_count=0, tx_count=0)

        G.nodes[cid]["total_amount"] += amount
        G.nodes[cid]["tx_count"] += 1
        G.nodes[cid]["fraud_count"] += is_fraud
        G.nodes[mid]["total_amount"] += amount
        G.nodes[mid]["tx_count"] += 1
        G.nodes[mid]["fraud_count"] += is_fraud

        if G.has_edge(cid, mid):
            G[cid][mid]["weight"] += 1
            G[cid][mid]["total_amount"] += amount
            G[cid][mid]["fraud_count"] += is_fraud
        else:
            G.add_edge(cid, mid, weight=1, total_amount=amount,
                       fraud_count=is_fraud)

    for node in G.nodes:
        tx = G.nodes[node]["tx_count"]
        fc = G.nodes[node]["fraud_count"]
        G.nodes[node]["fraud_rate"] = round(fc / tx, 4) if tx > 0 else 0.0

    return G


# ── Rich Feature Extraction ────────────────────────────────────────────────────

def extract_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-transaction graph features:
    customer_degree, merchant_degree, customer_fraud_rate,
    merchant_fraud_rate, customer_betweenness, merchant_betweenness,
    graph_risk_score, ring_member
    """
    if not _NX:
        return pd.DataFrame(index=df.index)
    if "customer_id" not in df.columns or "merchant_id" not in df.columns:
        return pd.DataFrame(index=df.index)

    G = build_transaction_graph(df)
    if G is None or G.number_of_nodes() == 0:
        return pd.DataFrame(index=df.index)

    degree = dict(G.degree())

    sample_k = min(500, G.number_of_nodes())
    try:
        betweenness = nx.betweenness_centrality(G, k=sample_k, normalized=True)
    except Exception:
        betweenness = {n: 0.0 for n in G.nodes}

    ring_members = _detect_ring_members(G)

    rows = []
    for _, row in df.iterrows():
        cid = f"C_{row['customer_id']}"
        mid = f"M_{row['merchant_id']}"

        c_deg = degree.get(cid, 0)
        m_deg = degree.get(mid, 0)
        c_fr  = G.nodes[cid]["fraud_rate"] if G.has_node(cid) else 0.0
        m_fr  = G.nodes[mid]["fraud_rate"] if G.has_node(mid) else 0.0
        c_bet = betweenness.get(cid, 0.0)
        m_bet = betweenness.get(mid, 0.0)

        graph_risk = (
            0.3 * np.log1p(c_deg) +
            0.3 * np.log1p(m_deg) +
            0.2 * c_fr +
            0.2 * m_fr +
            0.5 * c_bet +
            0.5 * m_bet
        )

        rows.append({
            "customer_degree":      c_deg,
            "merchant_degree":      m_deg,
            "customer_fraud_rate":  round(c_fr, 4),
            "merchant_fraud_rate":  round(m_fr, 4),
            "customer_betweenness": round(c_bet, 6),
            "merchant_betweenness": round(m_bet, 6),
            "graph_risk_score":     round(graph_risk, 4),
            "ring_member":          int(cid in ring_members or mid in ring_members),
        })

    return pd.DataFrame(rows, index=df.index)


# ── Fraud Ring Detection ───────────────────────────────────────────────────────

def _detect_ring_members(G: "nx.Graph", min_fraud_rate: float = 0.3,
                          min_size: int = 2) -> set:
    """
    Identify ring members. Uses merchant-centric detection:
    a merchant node with degree >= 2 (shared by multiple customers)
    and high fraud rate is a ring hub.
    """
    ring_members = set()
    # Merchant-hub approach: merchants connected to multiple customers
    for node in G.nodes:
        if G.nodes[node].get("node_type") != "merchant":
            continue
        if G.degree(node) < 2:
            continue
        if G.nodes[node].get("fraud_rate", 0) >= min_fraud_rate:
            ring_members.add(node)
            ring_members.update(G.neighbors(node))

    # Also check connected components
    for component in nx.connected_components(G):
        if len(component) < min_size:
            continue
        subgraph = G.subgraph(component)
        total_fraud = sum(subgraph.nodes[n]["fraud_count"] for n in subgraph.nodes)
        total_tx    = sum(subgraph.nodes[n]["tx_count"]    for n in subgraph.nodes)
        if total_tx == 0:
            continue
        if total_fraud / total_tx >= min_fraud_rate:
            ring_members.update(component)

    return ring_members


def detect_fraud_rings(df: pd.DataFrame,
                        min_fraud_rate: float = 0.3,
                        min_ring_size: int = 2) -> list[dict]:
    """
    Full fraud ring detection pipeline.
    Detects rings via:
    1. Merchant hubs (merchant shared by ≥2 customers with high fraud rate)
    2. Connected components with elevated fraud rate
    3. Shared device rings (if device_id column present)
    """
    if not _NX:
        return []
    G = build_transaction_graph(df)
    if G is None:
        return []

    rings = []
    seen_components = set()

    # ── Strategy 1: Merchant-hub rings ────────────────────────────────────────
    for node in list(G.nodes):
        if G.nodes[node].get("node_type") != "merchant":
            continue
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            continue
        merchant_fraud_rate = G.nodes[node].get("fraud_rate", 0)
        if merchant_fraud_rate < min_fraud_rate:
            continue

        component = frozenset([node] + neighbors)
        if component in seen_components:
            continue
        seen_components.add(component)

        total_amount = sum(G.nodes[n]["total_amount"] for n in component)
        total_fraud  = sum(G.nodes[n]["fraud_count"]  for n in component)
        total_tx     = sum(G.nodes[n]["tx_count"]     for n in component)
        fraud_rate   = total_fraud / total_tx if total_tx > 0 else 0

        rings.append({
            "ring_id":         f"RING-HUB-{len(rings):04d}",
            "ring_type":       "merchant_hub",
            "detected_at":     datetime.now(timezone.utc).isoformat(),
            "size":            len(component),
            "customer_count":  len(neighbors),
            "merchant_count":  1,
            "fraud_rate":      round(fraud_rate, 4),
            "total_amount":    round(total_amount, 2),
            "total_fraud_txn": total_fraud,
            "hub_nodes":       [node],
            "risk_level":      "CRITICAL" if fraud_rate >= 0.7 else "HIGH",
            "members":         list(component)[:20],
        })

    # ── Strategy 2: Connected component rings ─────────────────────────────────
    for i, component in enumerate(nx.connected_components(G)):
        if len(component) < min_ring_size + 1:
            continue
        fc = frozenset(component)
        if fc in seen_components:
            continue

        subgraph = G.subgraph(component)
        total_fraud  = sum(subgraph.nodes[n]["fraud_count"] for n in component)
        total_tx     = sum(subgraph.nodes[n]["tx_count"]    for n in component)
        total_amount = sum(subgraph.nodes[n]["total_amount"] for n in component)
        if total_tx == 0:
            continue
        fraud_rate = total_fraud / total_tx
        if fraud_rate < min_fraud_rate:
            continue

        seen_components.add(fc)
        customers = [n for n in component if n.startswith("C_")]
        merchants = [n for n in component if n.startswith("M_")]
        sub_degree = dict(subgraph.degree())
        hub_nodes  = sorted(sub_degree, key=sub_degree.get, reverse=True)[:3]

        rings.append({
            "ring_id":         f"RING-CC-{i:04d}",
            "ring_type":       "connected_component",
            "detected_at":     datetime.now(timezone.utc).isoformat(),
            "size":            len(component),
            "customer_count":  len(customers),
            "merchant_count":  len(merchants),
            "fraud_rate":      round(fraud_rate, 4),
            "total_amount":    round(total_amount, 2),
            "total_fraud_txn": total_fraud,
            "hub_nodes":       hub_nodes,
            "risk_level":      "CRITICAL" if fraud_rate >= 0.7 else "HIGH",
            "members":         list(component)[:20],
        })

    # ── Strategy 3: Shared device rings ───────────────────────────────────────
    if "device_id" in df.columns and "label" in df.columns:
        device_rings = _detect_device_rings(df, min_fraud_rate)
        rings.extend(device_rings)

    rings.sort(key=lambda r: r["fraud_rate"], reverse=True)
    _save_rings(rings)
    return rings


def _detect_device_rings(df: pd.DataFrame, min_fraud_rate: float = 0.3) -> list[dict]:
    """Detect rings where multiple customers share the same device."""
    rings = []
    if "device_id" not in df.columns:
        return rings

    device_groups = df.groupby("device_id")
    for device_id, group in device_groups:
        if group["customer_id"].nunique() < 2:
            continue
        fraud_rate = group["label"].mean() if "label" in group.columns else 0
        if fraud_rate < min_fraud_rate:
            continue
        rings.append({
            "ring_id":         f"RING-DEV-{device_id}",
            "ring_type":       "shared_device",
            "detected_at":     datetime.now(timezone.utc).isoformat(),
            "size":            len(group),
            "customer_count":  group["customer_id"].nunique(),
            "merchant_count":  group["merchant_id"].nunique() if "merchant_id" in group.columns else 0,
            "fraud_rate":      round(float(fraud_rate), 4),
            "total_amount":    round(float(group["transaction_amount"].sum()), 2) if "transaction_amount" in group.columns else 0,
            "total_fraud_txn": int(group["label"].sum()) if "label" in group.columns else 0,
            "hub_nodes":       [str(device_id)],
            "risk_level":      "CRITICAL" if fraud_rate >= 0.7 else "HIGH",
            "members":         [f"C_{c}" for c in group["customer_id"].unique()[:10]],
        })
    return rings


def _save_rings(rings: list[dict]):
    os.makedirs(os.path.dirname(RING_LOG), exist_ok=True)
    with open(RING_LOG, "w") as f:
        json.dump(rings, f, indent=2)


def load_ring_log() -> list[dict]:
    if not os.path.exists(RING_LOG):
        return []
    try:
        with open(RING_LOG) as f:
            return json.load(f)
    except Exception:
        return []


# ── Real-Time Single Transaction Graph Score ───────────────────────────────────

def score_transaction_graph(customer_id: str, merchant_id: str,
                              G: "nx.Graph") -> dict:
    if not _NX or G is None:
        return {"graph_risk_score": 0.0, "ring_member": 0,
                "customer_degree": 0, "merchant_degree": 0}

    cid = f"C_{customer_id}"
    mid = f"M_{merchant_id}"

    c_deg = G.degree(cid) if G.has_node(cid) else 0
    m_deg = G.degree(mid) if G.has_node(mid) else 0
    c_fr  = G.nodes[cid].get("fraud_rate", 0.0) if G.has_node(cid) else 0.0
    m_fr  = G.nodes[mid].get("fraud_rate", 0.0) if G.has_node(mid) else 0.0

    ring_members = _detect_ring_members(G)
    is_ring = int(cid in ring_members or mid in ring_members)

    graph_risk = (
        0.3 * np.log1p(c_deg) +
        0.3 * np.log1p(m_deg) +
        0.2 * c_fr +
        0.2 * m_fr
    )

    return {
        "graph_risk_score":    round(graph_risk, 4),
        "ring_member":         is_ring,
        "customer_degree":     c_deg,
        "merchant_degree":     m_deg,
        "customer_fraud_rate": round(c_fr, 4),
        "merchant_fraud_rate": round(m_fr, 4),
    }


# ── Plotly Visualization ───────────────────────────────────────────────────────

def plot_fraud_ring_network(rings: list[dict], max_rings: int = 3):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not rings:
        return None

    fig = go.Figure()
    colors = ["#e74c3c", "#e67e22", "#9b59b6", "#3498db", "#1abc9c"]

    for ri, ring in enumerate(rings[:max_rings]):
        members = ring.get("members", [])
        if len(members) < 2:
            continue

        n = len(members)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        offset_x = ri * 3.5
        xs = np.cos(angles) + offset_x
        ys = np.sin(angles)
        pos = {m: (xs[i], ys[i]) for i, m in enumerate(members)}
        color = colors[ri % len(colors)]

        node_x = [pos[m][0] for m in members]
        node_y = [pos[m][1] for m in members]
        node_text = [m[:12] for m in members]
        node_colors = ["#c0392b" if m.startswith("C_") else "#2980b9"
                       for m in members]

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=14, color=node_colors,
                        line=dict(width=2, color="white")),
            text=node_text, textposition="top center",
            textfont=dict(size=9),
            name=f"Ring {ri+1} ({ring.get('ring_type','')}, fraud={ring['fraud_rate']:.0%})",
            hovertemplate=(
                f"<b>Ring {ri+1}</b><br>"
                f"Type: {ring.get('ring_type','')}<br>"
                f"Fraud Rate: {ring['fraud_rate']:.1%}<br>"
                f"Total Amount: ${ring['total_amount']:,.0f}<br>"
                f"Members: {ring['size']}<extra></extra>"
            ),
        ))

        for i in range(len(members)):
            for j in range(i + 1, min(i + 3, len(members))):
                x0, y0 = pos[members[i]]
                x1, y1 = pos[members[j]]
                fig.add_trace(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=1, color=color),
                    showlegend=False, hoverinfo="skip",
                ))

    fig.update_layout(
        title="Fraud Ring Network — Detected Rings",
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(bgcolor="#1e1e2e", bordercolor="#444"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
