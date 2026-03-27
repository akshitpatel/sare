"""
P4.2 — Working Memory + Attention Mechanism

Lightweight working memory that the Brain maintains *per solve*.
It guides transform selection by keeping track of:
  - Current domain and graph state
  - Active goal (what form we're trying to reach)
  - Relevant transforms for this domain (ranked by past success)
  - Recent failures (transforms to de-prioritize)
  - Hypothesis being tested
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class WorkingMemoryState:
    """Snapshot of working memory at one point in solving."""
    domain: str = "general"
    goal: str = ""                         # Natural language goal
    relevant_transforms: List[str] = field(default_factory=list)
    recently_failed: List[str] = field(default_factory=list)
    hypothesis: str = ""                   # Current conjecture
    context_tags: List[str] = field(default_factory=list)


class WorkingMemory:
    """
    Lightweight per-solve working memory with domain attention.

    Usage::

        wm = WorkingMemory()
        # Before search:
        priority_ts = wm.get_prioritized_transforms(transforms, domain)
        # After search:
        wm.record_outcome(domain, transforms_used, success, delta)
    """

    CAPACITY = 500  # max episodes to remember

    def __init__(self):
        self._state = WorkingMemoryState()
        # Domain→transform success stats: {domain: {t_name: (successes, attempts)}}
        self._domain_stats: Dict[str, Dict[str, list]] = {}
        self._history: Deque[dict] = deque(maxlen=self.CAPACITY)

    # ── Attention: Prioritize transforms for the current domain ──

    def get_prioritized_transforms(self, transforms: list, domain: str) -> list:
        """
        Re-rank transforms by domain-specific success rate.
        Domain-relevant transforms come first; consistently-failing ones last.
        """
        stats = self._domain_stats.get(domain, {})
        if not stats:
            return transforms  # No data yet: use default order

        def score(t) -> float:
            name = t.name()
            if name in stats:
                s, a = stats[name]
                if a == 0:
                    return 0.5
                success_rate = s / a
                # Penalize transforms we recently failed with
                if name in self._state.recently_failed:
                    success_rate *= 0.3
                return success_rate
            return 0.4  # Unseen in this domain: neutral score

        return sorted(transforms, key=score, reverse=True)

    def get_domain_top_transforms(self, domain: str, n: int = 5) -> List[str]:
        """Get the n best-performing transforms for this domain."""
        stats = self._domain_stats.get(domain, {})
        ranked = sorted(
            stats.items(),
            key=lambda x: (x[1][0] / max(x[1][1], 1)),
            reverse=True,
        )
        return [name for name, _ in ranked[:n]]

    # ── State management ──────────────────────────────────────

    def focus(self, domain: str, goal: str = "", hypothesis: str = ""):
        """Set working memory focus for a new solve."""
        top = self.get_domain_top_transforms(domain)
        self._state = WorkingMemoryState(
            domain=domain,
            goal=goal,
            relevant_transforms=top,
            recently_failed=[],
            hypothesis=hypothesis,
        )

    def record_failure(self, transform_name: str):
        """Mark a transform as recently failed (de-prioritize)."""
        if transform_name not in self._state.recently_failed:
            self._state.recently_failed.append(transform_name)
            if len(self._state.recently_failed) > 10:
                self._state.recently_failed.pop(0)

    def record_outcome(self, domain: str, transforms_used: List[str],
                       success: bool, delta: float):
        """Update domain stats based on solve outcome."""
        if domain not in self._domain_stats:
            self._domain_stats[domain] = {}

        for t_name in transforms_used:
            if t_name not in self._domain_stats[domain]:
                self._domain_stats[domain][t_name] = [0, 0]
            self._domain_stats[domain][t_name][1] += 1
            if success:
                self._domain_stats[domain][t_name][0] += 1

        self._history.append({
            "domain": domain, "transforms": transforms_used,
            "success": success, "delta": delta,
        })

    # ── Causal reasoning hint for search ──────────────────────

    def predict_useful_transforms(self, domain: str, graph,
                                   all_transforms: list) -> List[str]:
        """
        P4.2 causal: predict which transforms are likely useful
        based on graph structure + domain success history.
        Returns ordered list of transform names.
        """
        predictions = []

        # Historical success in this domain
        domain_best = self.get_domain_top_transforms(domain, n=10)
        predictions.extend(domain_best)

        # Structural: look at operators in graph
        if graph is not None:
            labels = {n.label for n in graph.nodes if n.type == "operator"}
            for t in all_transforms:
                name = t.name()
                if name not in predictions:
                    # Heuristic: if transform name shares keyword with graph operators
                    if any(op in name or name in op for op in labels):
                        predictions.append(name)

        return predictions

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> dict:
        domain_summary = {}
        for domain, ts in self._domain_stats.items():
            total = sum(s for s, _ in ts.values())
            attempts = sum(a for _, a in ts.values())
            rate = total / max(attempts, 1)
            domain_summary[domain] = {
                "solve_rate": round(rate, 3),
                "transforms_tried": len(ts),
                "best": self.get_domain_top_transforms(domain, n=3),
            }
        return {
            "state_domain": self._state.domain,
            "domains_tracked": len(self._domain_stats),
            "history_length": len(self._history),
            "domain_stats": domain_summary,
        }

    @property
    def current_state(self) -> WorkingMemoryState:
        return self._state
