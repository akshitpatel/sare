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

from sare.heuristics.graph_embedding import GraphEmbedding


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

    def forward(self, type_indices: torch.Tensor,
                adjacency: list) -> torch.Tensor:
        """
        Returns scalar prediction of energy reduction potential.
        """
        graph_embed = self.embedding(type_indices, adjacency)
        return self.mlp(graph_embed).squeeze(-1)

    def predict(self, type_indices: torch.Tensor,
                adjacency: list) -> float:
        """Convenience method: returns Python float."""
        with torch.no_grad():
            return self.forward(type_indices, adjacency).item()


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
