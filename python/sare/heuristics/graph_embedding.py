"""
SARE Heuristic Model — Graph Embedding Layer

Implements φ(G) → R^d via message-passing over the typed graph.
The embedding captures structural features for the heuristic predictor.

Architecture:
    1. Node type embedding (learnable lookup table)
    2. Multi-layer message passing (neighbor aggregation)
    3. Global readout (mean + max pooling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


# Use best available device: CUDA > MPS (Apple Metal) > CPU
if torch.cuda.is_available():
    _DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    _DEVICE = torch.device("mps")
else:
    _DEVICE = torch.device("cpu")


class NodeEncoder(nn.Module):
    """Encodes node types + attributes into initial feature vectors."""

    def __init__(self, num_types: int = 64, embed_dim: int = 64):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, embed_dim)
        self.type_to_idx: Dict[str, int] = {}
        self.next_idx = 0

    def get_type_idx(self, type_str: str) -> int:
        if type_str not in self.type_to_idx:
            self.type_to_idx[type_str] = self.next_idx
            self.next_idx = min(self.next_idx + 1,
                                self.type_embedding.num_embeddings - 1)
        return self.type_to_idx[type_str]

    def forward(self, type_indices: torch.Tensor) -> torch.Tensor:
        """type_indices: [num_nodes] → [num_nodes, embed_dim]"""
        return self.type_embedding(type_indices)


class MessagePassingLayer(nn.Module):
    """Single message passing layer: aggregate neighbor features."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.message_fn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_fn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features: torch.Tensor,
                adjacency: List[Tuple[int, int]]) -> torch.Tensor:
        """
        node_features: [N, D]
        adjacency: list of (src, tgt) pairs
        Returns updated node_features: [N, D]
        """
        N, D = node_features.shape
        device = node_features.device

        messages = torch.zeros_like(node_features)
        counts = torch.zeros(N, 1, device=device)

        # Vectorized edge processing for speed (avoid Python loops over edges).
        # Filter invalid edges and build tensors.
        if adjacency:
            src_list = []
            tgt_list = []
            for src, tgt in adjacency:
                if 0 <= src < N and 0 <= tgt < N:
                    src_list.append(src)
                    tgt_list.append(tgt)

            if src_list:
                src_idx = torch.tensor(src_list, device=device, dtype=torch.long)
                tgt_idx = torch.tensor(tgt_list, device=device, dtype=torch.long)

                # Build edge-wise messages: message_fn([x_src, x_tgt])
                msg_input = torch.cat([node_features[src_idx], node_features[tgt_idx]], dim=-1)  # [E, 2D]
                msg = F.relu(self.message_fn(msg_input))  # [E, D]

                # Aggregate by target: sum messages into messages[tgt]
                messages.index_add_(0, tgt_idx, msg)

                # Degree counts per target
                ones = torch.ones((tgt_idx.shape[0], 1), device=device, dtype=node_features.dtype)
                counts.index_add_(0, tgt_idx, ones)

        # Normalize by degree
        counts = counts.clamp(min=1)
        messages = messages / counts

        # Update with residual connection
        update_input = torch.cat([node_features, messages], dim=-1)
        updated = F.relu(self.update_fn(update_input))
        return self.norm(updated + node_features)  # residual


class GraphEmbedding(nn.Module):
    """
    φ(G) → R^d

    Full graph embedding via message-passing over typed graph.
    Uses mean + max pooling for global readout.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3,
                 num_types: int = 64):
        super().__init__()
        self.encoder = NodeEncoder(num_types, hidden_dim)
        self.layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.output_dim = hidden_dim * 2  # mean + max pooling

        # Keep module on the target device; avoid per-forward .to() calls.
        self.to(_DEVICE)

    def forward(self, type_indices: torch.Tensor,
                adjacency: List[Tuple[int, int]]) -> torch.Tensor:
        """
        type_indices: [N] - node type indices
        adjacency: list of (src, tgt) edge pairs
        Returns: [output_dim] - graph embedding vector (always on CPU)
        """
        type_indices = type_indices.to(_DEVICE)
        x = self.encoder(type_indices)  # [N, D]

        for layer in self.layers:
            x = layer(x, adjacency)

        # Global readout: mean + max pooling
        mean_pool = x.mean(dim=0)   # [D]
        max_pool = x.max(dim=0)[0]  # [D]
        result = torch.cat([mean_pool, max_pool])  # [2D]
        # Always return on CPU so downstream code (numpy, engine) doesn't need MPS
        return result.cpu()