"""
SARE Credit Assignment

Per-module reward propagation with baseline subtraction.
Implements the mathematical formulation:

    R = -(E_final - E_initial)
    Δθ_k = η(R - b)
    U_k^{t+1} = (1 - α)U_k^t + α(-ΔE_k)

This module:
1. Takes a solve trace (transform sequence + energy trajectory)
2. Computes per-transform credit using the reward signal
3. Updates module utility scores with exponential moving average
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CreditResult:
    """Credit assignment result for a single transform application."""
    transform_name: str
    delta_energy: float       # actual ΔE for this step
    credit: float             # R - baseline
    updated_utility: float    # new U_k


class CreditAssigner:
    """
    Per-module reward propagation.

    Maintains utility scores U_k for each transform module and
    updates them based on solve trace energy trajectories.
    """

    def __init__(self, alpha: float = 0.1, eta: float = 0.01):
        """
        Args:
            alpha: EMA smoothing factor for utility updates
            eta: learning rate for parameter updates
        """
        self.alpha = alpha
        self.eta = eta
        self.utilities: Dict[str, float] = {}  # U_k per module
        self.baseline: float = 0.0             # running baseline b
        self.baseline_count: int = 0

    def assign_credit(self, transform_sequence: List[str],
                      energy_trajectory: List[float]) -> List[CreditResult]:
        """
        Assign credit to each transform in a solve trace.

        Args:
            transform_sequence: list of transform names applied
            energy_trajectory: E at each step (len = len(transforms) + 1)

        Returns:
            List of CreditResult for each transform application
        """
        if len(energy_trajectory) < 2:
            return []

        # Total reward: R = -(E_final - E_initial)
        total_reward = -(energy_trajectory[-1] - energy_trajectory[0])

        # Update baseline with running average
        self.baseline_count += 1
        alpha_b = 1.0 / self.baseline_count
        self.baseline = (1 - alpha_b) * self.baseline + alpha_b * total_reward

        results = []
        for i, transform_name in enumerate(transform_sequence):
            if i + 1 >= len(energy_trajectory):
                break

            # Per-step ΔE
            delta_e = energy_trajectory[i + 1] - energy_trajectory[i]

            # Credit: R - b (baseline subtracted)
            credit = total_reward - self.baseline

            # Update utility: U_k^{t+1} = (1-α)U_k^t + α(-ΔE_k)
            current_u = self.utilities.get(transform_name, 0.0)
            new_u = (1 - self.alpha) * current_u + self.alpha * (-delta_e)
            self.utilities[transform_name] = new_u

            results.append(CreditResult(
                transform_name=transform_name,
                delta_energy=delta_e,
                credit=credit,
                updated_utility=new_u,
            ))

        return results

    def get_utility(self, transform_name: str) -> float:
        """Get current utility score for a transform."""
        return self.utilities.get(transform_name, 0.0)

    def get_all_utilities(self) -> Dict[str, float]:
        """Get all utility scores."""
        return dict(self.utilities)

    def get_top_transforms(self, n: int = 10) -> List[tuple]:
        """Get top-N transforms by utility."""
        sorted_utils = sorted(self.utilities.items(),
                              key=lambda x: x[1], reverse=True)
        return sorted_utils[:n]

    def get_underperformers(self, threshold: float = 0.0) -> List[str]:
        """Get transforms with utility below threshold (candidates for pruning)."""
        return [name for name, u in self.utilities.items() if u < threshold]
