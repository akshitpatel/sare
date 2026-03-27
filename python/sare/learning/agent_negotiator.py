import logging
import threading
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class NegotiationArena:
    """
    Central clearinghouse where multiple agents submit discovered proofs for
    the same problem. The Arena picks the lowest-energy (most elegant) solution
    and broadcasts it as the 'Network Truth' to the Concept Graph.
    """

    def __init__(self, energy_engine=None):
        # energy_engine is optional; we fall back to path-length heuristic
        self.energy_engine = energy_engine
        self.active_debates: Dict[str, List[Dict[str, Any]]] = {}
        self.negotiated_truths: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def submit_discovery(self, problem_signature: str, agent_id: str,
                         transform: Any, derivation_path: List[str]):
        """Agents call this when they solve a problem or induce a rule."""
        with self._lock:
            if problem_signature not in self.active_debates:
                self.active_debates[problem_signature] = []
            cost = self._evaluate_path_energy(transform, derivation_path)
            self.active_debates[problem_signature].append({
                "agent_id": agent_id,
                "transform": transform,
                "path": derivation_path,
                "energy_cost": cost,
                "timestamp": __import__("time").time(),
            })
            logger.debug("Agent %s submitted proof for '%s' (cost=%.2f)",
                         agent_id, problem_signature, cost)

    def _evaluate_path_energy(self, transform: Any, path: List[str]) -> float:
        """Shorter paths = lower cost. Use real energy engine if available."""
        base_cost = float(len(path))
        if self.energy_engine and hasattr(self.energy_engine, "costs"):
            base_cost *= self.energy_engine.costs.get("apply_transform", 1.0)
        name = getattr(transform, "name", "") or str(transform)
        complexity_cost = len(name) * 0.05
        return base_cost + complexity_cost

    def deliberate(self) -> List[Dict[str, Any]]:
        """
        Close all active debates, pick winners (lowest energy), add to
        negotiated_truths. Returns newly minted truths to broadcast.
        """
        new_truths = []
        with self._lock:
            for signature, submissions in self.active_debates.items():
                if not submissions:
                    continue
                submissions.sort(key=lambda x: x["energy_cost"])
                winner = submissions[0]
                truth = {
                    "signature": signature,
                    "winner": winner["agent_id"],
                    "cost": winner["energy_cost"],
                    "competing": len(submissions),
                    "transform_name": getattr(winner["transform"], "name",
                                              str(winner["transform"]))[:40],
                }
                self.negotiated_truths.insert(0, truth)
                new_truths.append(truth)
                logger.info("Arena settled '%s'. Winner: %s (cost=%.2f, rivals=%d)",
                            signature, winner["agent_id"],
                            winner["energy_cost"], len(submissions) - 1)
            self.active_debates.clear()
            if len(self.negotiated_truths) > 100:
                self.negotiated_truths = self.negotiated_truths[:100]
        return new_truths

    def get_recent_truths(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return self.negotiated_truths[:limit]
