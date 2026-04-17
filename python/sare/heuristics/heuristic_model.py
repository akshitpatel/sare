"""
SARE Heuristic Model — Energy Predictor

H_θ(G) = f_θ(φ(G))

Small MLP on top of graph embedding. Predicts the energy reduction
potential of a graph state. Used to guide search.

Loss: L = (H_θ(G) - (-ΔE_actual))²

Training data comes from the system's own solve traces (episodic store).
This model does NOT replace verification — it only accelerates search.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from sare.heuristics.graph_embedding import GraphEmbedding, _DEVICE


class HeuristicModel(nn.Module):
    """
    f_θ : R^d → R

    Predicts energy reduction potential.
    Higher output = more promising graph state.

    Architecture:
        GraphEmbedding → MLP(256, 128, 64) → scalar
    """

    def __init__(self, embed_dim: int = 64, num_layers: int = 3,
                 num_types: int = 64):
        super().__init__()
        self.embedding = GraphEmbedding(embed_dim, num_layers, num_types)

        input_dim = self.embedding.output_dim  # 2 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.to(_DEVICE)

    def forward(self, type_indices: torch.Tensor,
                adjacency: list) -> torch.Tensor:
        """
        Returns scalar prediction of energy reduction potential.
        GraphEmbedding.forward always returns a CPU tensor, so we move it
        to _DEVICE before the MLP (which lives on _DEVICE after predict() is called).
        """
        graph_embed = self.embedding(type_indices, adjacency)  # CPU tensor
        graph_embed = graph_embed.to(_DEVICE)
        return self.mlp(graph_embed).squeeze(-1)

    def predict(self, type_indices: torch.Tensor,
                adjacency: list) -> float:
        """Convenience method: runs on device set in __init__, returns Python float."""
        type_indices = type_indices.to(_DEVICE)
        with torch.no_grad():
            # GraphEmbedding.forward already returns CPU tensor; result is scalar CPU float
            val = self.forward(type_indices, adjacency)
            return float(val.cpu().item())

    def predict_graph(self, graph) -> float:
        """Predict energy reduction potential for a graph. Returns 0.0 on error."""
        try:
            nodes = list(graph.nodes) if hasattr(graph, 'nodes') else []
            if not nodes:
                return 0.0
            type_map = {}
            type_indices_list = []
            for n in nodes:
                t = getattr(n, 'node_type', getattr(n, 'type', 'unknown'))
                if t not in type_map:
                    type_map[t] = len(type_map)
                type_indices_list.append(type_map[t])
            type_indices_tensor = torch.tensor(type_indices_list, dtype=torch.long)
            # Build adjacency list
            edges = list(graph.edges) if hasattr(graph, 'edges') else []
            node_ids = {id(n): i for i, n in enumerate(nodes)}
            adjacency = []
            for e in edges:
                s = node_ids.get(id(getattr(e, 'source', None)), -1)
                t2 = node_ids.get(id(getattr(e, 'target', None)), -1)
                if s >= 0 and t2 >= 0:
                    adjacency.append((s, t2))
            return float(self.predict(type_indices_tensor, adjacency))
        except Exception:
            return 0.0


class HeuristicLoss(nn.Module):
    """
    L = (H_θ(G) - (-ΔE_actual))²

    MSE between predicted and actual energy reduction.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predicted: torch.Tensor,
                actual_delta_e: torch.Tensor) -> torch.Tensor:
        """
        predicted: model output (predicted -ΔE)
        actual_delta_e: actual energy change (negative = improvement)
        target = -actual_delta_e (so positive = good)
        """
        target = -actual_delta_e
        return self.mse(predicted, target)
