#!/usr/bin/env python
"""Train the FraudGuard heterogeneous R-GCN and publish scores to Neo4j.

Example: python scripts/train_gnn_fraud.py data/transactions.csv --epochs 40
Required environment: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import torch.nn.functional as F

from src.graph_intelligence import build_heterogeneous_graph, HETERO_NODE_TYPES
from src.graph_neo4j import GraphAdapter
from src.models.gnn_fraud import HeteroFraudRGCN, score_transaction_edges


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    frame = pd.read_csv(args.input)
    if args.label_column != "label":
        frame = frame.rename(columns={args.label_column: "label"})
    graph = build_heterogeneous_graph(frame)
    edge_index = graph["customer", "filed_transaction", "merchant"].edge_index
    edge_labels = graph["customer", "filed_transaction", "merchant"].edge_label
    if edge_labels.numel() == 0:
        raise ValueError("No transaction edges found in the input data")

    model = HeteroFraudRGCN(hidden_dim=args.hidden_dim, embedding_dim=args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        _, edge_logits, node_logits = model(graph.x_dict, graph.edge_index_dict, edge_index)
        edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)
        node_losses = [
            F.binary_cross_entropy_with_logits(node_logits[kind], graph[kind].y)
            for kind in HETERO_NODE_TYPES if graph[kind].y.numel()
        ]
        loss = edge_loss + args.node_loss_weight * torch.stack(node_losses).mean()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch}/{args.epochs} loss={loss.item():.6f} "
                  f"edge_loss={edge_loss.item():.6f}")

    embeddings, edge_probabilities, node_probabilities = score_transaction_edges(model, graph)
    node_outputs = {}
    for kind in HETERO_NODE_TYPES:
        node_outputs[kind] = [
            {"id": node_id,
             "embedding": embeddings[kind][index].tolist(),
             "collusion_risk": float(node_probabilities[kind][index])}
            for node_id, index in graph.node_id_maps[kind].items()
        ]
    edge_outputs = []
    for position, (_, row) in enumerate(frame.iterrows()):
        transaction_id = row.get("transaction_id")
        edge_outputs.append({
            "transaction_id": str(transaction_id) if pd.notna(transaction_id)
                              else f"row-{position}",
            "customer_id": str(row["customer_id"]),
            "merchant_id": str(row["merchant_id"]),
            "amount": float(row["transaction_amount"]),
            "label": float(row["label"]),
            "fraud_probability": float(edge_probabilities[position]),
        })

    adapter = GraphAdapter()
    try:
        if adapter.backend != "neo4j":
            raise RuntimeError("Set NEO4J_URI and install the Neo4j driver before training")
        adapter.write_gnn_outputs(frame, node_outputs, edge_outputs)
    finally:
        adapter.close()

    output_path = Path(args.model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "node_types": list(HETERO_NODE_TYPES),
                "hidden_dim": args.hidden_dim,
                "embedding_dim": args.embedding_dim}, output_path)
    return {"loss": float(loss.detach()), "model_path": str(output_path),
            "nodes_written": sum(map(len, node_outputs.values())),
            "transaction_edges_written": len(edge_outputs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="CSV with customer_id, merchant_id, transaction_amount, label")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--node-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--model-out", default="outputs/gnn_fraud.pt")
    args = parser.parse_args()
    if args.epochs < 1 or args.log_every < 1:
        parser.error("--epochs and --log-every must be positive")
    result = train(args)
    print("Training complete:", result)


if __name__ == "__main__":
    main()
