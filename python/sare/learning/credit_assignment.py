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
import json
from pathlib import Path
from typing import Dict, List, Optional


_DEFAULT_PERSIST_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "credit_assigner.json"


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
        self.baseline: float = 0.0             # global baseline (kept for backward compat)
        self.baseline_count: int = 0
        self._baselines: Dict[str, float] = {}  # per-domain EMA baselines

    def _get_baseline(self, domain: str) -> float:
        """Return EMA baseline for the given domain (falls back to global)."""
        return self._baselines.get(domain, self.baseline)

    def _update_baseline(self, domain: str, reward: float, alpha_b: float = 0.05) -> None:
        """Update per-domain and global EMA baselines."""
        b = self._baselines.get(domain, self.baseline)
        self._baselines[domain] = (1 - alpha_b) * b + alpha_b * reward
        # Also keep global baseline in sync
        self.baseline = (1 - alpha_b) * self.baseline + alpha_b * reward

    def assign_credit(self, transform_sequence: List[str],
                      energy_trajectory: List[float],
                      domain: str = "general") -> List[CreditResult]:
        """
        Assign credit to each transform in a solve trace.

        Args:
            transform_sequence: list of transform names applied
            energy_trajectory: E at each step (len = len(transforms) + 1)
            domain: problem domain for per-domain baseline tracking

        Returns:
            List of CreditResult for each transform application
        """
        if len(energy_trajectory) < 2:
            return []

        # Total reward: R = -(E_final - E_initial)
        total_reward = -(energy_trajectory[-1] - energy_trajectory[0])

        # Update per-domain baseline (replaces single global baseline)
        self._update_baseline(domain, total_reward)

        results = []
        for i, transform_name in enumerate(transform_sequence):
            if i + 1 >= len(energy_trajectory):
                break

            # Per-step ΔE
            delta_e = energy_trajectory[i + 1] - energy_trajectory[i]

            # Credit: R - b using per-domain baseline
            credit = total_reward - self._get_baseline(domain)

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

    def save(self, path: Optional[Path] = None) -> None:
        """Persist utility scores and baseline statistics to disk."""
        target = Path(path or _DEFAULT_PERSIST_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "eta": self.eta,
            "utilities": self.utilities,
            "baseline": self.baseline,
            "baseline_count": self.baseline_count,
            "baselines": self._baselines,
        }
        import os as _os, threading as _thr
        _tmp = target.parent / f"{target.stem}.{_os.getpid()}.{_thr.get_ident()}.tmp"
        _tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _os.replace(_tmp, target)

    def load(self, path: Optional[Path] = None) -> None:
        """Restore utility scores and baseline statistics from disk."""
        target = Path(path or _DEFAULT_PERSIST_PATH)
        if not target.exists():
            return

        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupted file (e.g. from non-atomic concurrent write) — reset
            target.unlink(missing_ok=True)
            return
        self.alpha = float(payload.get("alpha", self.alpha))
        self.eta = float(payload.get("eta", self.eta))
        self.utilities = {
            str(name): float(score)
            for name, score in payload.get("utilities", {}).items()
        }
        self.baseline = float(payload.get("baseline", 0.0))
        self.baseline_count = int(payload.get("baseline_count", 0))
        self._baselines = {
            str(k): float(v)
            for k, v in payload.get("baselines", {}).items()
        }
