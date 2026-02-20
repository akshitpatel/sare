"""
SARE Heuristic Trainer

Trains the heuristic model H_θ from episodic store solve traces.
The trainer:
1. Reads solve episodes from the episodic store
2. Extracts (graph_state, energy_change) pairs
3. Trains the MLP to predict ΔE from graph embeddings

Training loop:
    for epoch in range(epochs):
        for (G_t, ΔE_t) in dataset:
            predicted = H_θ(G_t)
            loss = (predicted - (-ΔE_t))²
            loss.backward()
            optimizer.step()
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
from pathlib import Path

from sare.heuristics.heuristic_model import HeuristicModel, HeuristicLoss


@dataclass
class TrainingSample:
    """One training sample: graph features + actual energy change."""
    type_indices: List[int]
    adjacency: List[Tuple[int, int]]
    delta_energy: float


class SolveTraceDataset(Dataset):
    """Dataset constructed from solve trace files."""

    def __init__(self, samples: List[TrainingSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "type_indices": torch.tensor(s.type_indices, dtype=torch.long),
            "adjacency": s.adjacency,
            "delta_energy": torch.tensor(s.delta_energy, dtype=torch.float32),
        }


class HeuristicTrainer:
    """
    Trains the heuristic model from solve traces.

    Usage:
        trainer = HeuristicTrainer(model)
        trainer.load_traces("logs/solves.jsonl")
        trainer.train(epochs=50, lr=1e-3)
        trainer.save_model("models/heuristic_v1.pt")
    """

    def __init__(self, model: Optional[HeuristicModel] = None):
        self.model = model or HeuristicModel()
        self.loss_fn = HeuristicLoss()
        self.samples: List[TrainingSample] = []
        self.training_history: List[float] = []

    def add_sample(self, type_indices: List[int],
                   adjacency: List[Tuple[int, int]],
                   delta_energy: float):
        """Add a single training sample."""
        self.samples.append(TrainingSample(type_indices, adjacency, delta_energy))

    def load_traces(self, path: str):
        """Load training data from JSONL solve trace file."""
        trace_path = Path(path)
        if not trace_path.exists():
            return

        with open(trace_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                if "energy_trajectory" in entry:
                    trajectory = entry["energy_trajectory"]
                    node_types = entry.get("node_types", [])
                    adjacency_raw = entry.get("adjacency", [])
                    if node_types:
                        type_indices = [
                            self.model.embedding.encoder.get_type_idx(str(t))
                            for t in node_types
                        ]
                        adjacency = []
                        for pair in adjacency_raw:
                            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                                try:
                                    adjacency.append((int(pair[0]), int(pair[1])))
                                except (TypeError, ValueError):
                                    continue
                    else:
                        type_indices = [0]
                        adjacency = []

                    for i in range(len(trajectory) - 1):
                        delta_e = trajectory[i + 1] - trajectory[i]
                        self.samples.append(TrainingSample(
                            type_indices=type_indices,
                            adjacency=adjacency,
                            delta_energy=delta_e
                        ))

    def train(self, epochs: int = 50, lr: float = 1e-3,
              batch_size: int = 32) -> List[float]:
        """
        Train the heuristic model.
        Returns: list of per-epoch average losses.
        """
        if len(self.samples) == 0:
            return []

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            count = 0

            # Simple training loop (no DataLoader for variable-size graphs)
            for sample in self.samples:
                type_indices = torch.tensor(sample.type_indices, dtype=torch.long)
                delta_e = torch.tensor(sample.delta_energy, dtype=torch.float32)

                predicted = self.model(type_indices, sample.adjacency)
                loss = self.loss_fn(predicted, delta_e)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                count += 1

            avg_loss = epoch_loss / max(count, 1)
            losses.append(avg_loss)
            self.training_history.append(avg_loss)

        self.model.eval()
        return losses

    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
