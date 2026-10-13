"""Two-layer heterogeneous R-GCN for transaction, RTO, and collusion risk."""
from __future__ import annotations

from typing import Dict, Tuple

try:
    import torch
    from torch import Tensor, nn
    from torch_geometric.nn import HeteroConv, MessagePassing
except ImportError as exc:  # Keep non-GNN application paths importable.
    torch = None
    _PYG_IMPORT_ERROR = exc
else:
    _PYG_IMPORT_ERROR = None

from src.graph_intelligence import HETERO_EDGE_TYPES, HETERO_NODE_TYPES


if torch is not None:
    class RelationRGCNConv(MessagePassing):
        """One typed relation transform with mean aggregation and root update."""
        def __init__(self, hidden_dim: int):
            super().__init__(aggr="mean", flow="source_to_target")
            self.message_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.root_linear = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x: Tuple[Tensor, Tensor], edge_index: Tensor) -> Tensor:
            source, target = x
            messages = self.propagate(
                edge_index, x=(source, target), size=(source.size(0), target.size(0))
            )
            return self.root_linear(target) + messages

        def message(self, x_j: Tensor) -> Tensor:
            return self.message_linear(x_j)


    class HeteroFraudRGCN(nn.Module):
        """Predict transaction-edge fraud and per-node collusion probability."""
        def __init__(self, input_dim: int = 2, hidden_dim: int = 64,
                     embedding_dim: int = 32):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.embedding_dim = embedding_dim
            self.input_projection = nn.ModuleDict({
                node_type: nn.Linear(input_dim, hidden_dim)
                for node_type in HETERO_NODE_TYPES
            })
            self.conv1 = HeteroConv(
                {edge_type: RelationRGCNConv(hidden_dim) for edge_type in HETERO_EDGE_TYPES},
                aggr="sum",
            )
            self.conv2 = HeteroConv(
                {edge_type: RelationRGCNConv(hidden_dim) for edge_type in HETERO_EDGE_TYPES},
                aggr="sum",
            )
            self.embedding_projection = nn.ModuleDict({
                node_type: nn.Linear(hidden_dim, embedding_dim)
                for node_type in HETERO_NODE_TYPES
            })
            self.node_collusion_head = nn.ModuleDict({
                node_type: nn.Linear(embedding_dim, 1)
                for node_type in HETERO_NODE_TYPES
            })
            self.edge_fraud_head = nn.Sequential(
                nn.Linear(embedding_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
            )
            self.edge_rto_head = nn.Sequential(
                nn.Linear(embedding_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
            )

        def forward(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict,
                    transaction_edge_index: Tensor):
            hidden = {key: torch.relu(self.input_projection[key](value))
                      for key, value in x_dict.items()}
            first = self.conv1(hidden, edge_index_dict)
            hidden = {key: torch.relu(first.get(key, value)) for key, value in hidden.items()}
            second = self.conv2(hidden, edge_index_dict)
            hidden = {key: torch.relu(second.get(key, value)) for key, value in hidden.items()}
            embeddings = {key: self.embedding_projection[key](value)
                          for key, value in hidden.items()}
            customers, merchants = embeddings["customer"], embeddings["merchant"]
            source, target = transaction_edge_index
            customer_embedding, merchant_embedding = customers[source], merchants[target]
            edge_features = torch.cat((customer_embedding, merchant_embedding,
                                       customer_embedding * merchant_embedding), dim=-1)
            edge_logits = self.edge_fraud_head(edge_features).squeeze(-1)
            rto_logits = self.edge_rto_head(edge_features).squeeze(-1)
            node_logits = {key: self.node_collusion_head[key](value).squeeze(-1)
                           for key, value in embeddings.items()}
            return embeddings, edge_logits, rto_logits, node_logits
else:
    class HeteroFraudRGCN:  # pragma: no cover - used only without optional GNN deps
        def __init__(self, *args, **kwargs):
            raise ImportError("Install torch and torch-geometric to use HeteroFraudRGCN") from _PYG_IMPORT_ERROR


def score_transaction_edges(model, graph):
    """Return edge fraud and node collusion probabilities without gradients."""
    if torch is None:
        raise ImportError("Install torch and torch-geometric to score graph data") from _PYG_IMPORT_ERROR
    model.eval()
    edge_index = graph["customer", "filed_transaction", "merchant"].edge_index
    with torch.no_grad():
        embeddings, edge_logits, rto_logits, node_logits = model(
            graph.x_dict, graph.edge_index_dict, edge_index
        )
        return ({key: value.cpu() for key, value in embeddings.items()},
                torch.sigmoid(edge_logits).cpu(),
                torch.sigmoid(rto_logits).cpu(),
                {key: torch.sigmoid(value).cpu() for key, value in node_logits.items()})
