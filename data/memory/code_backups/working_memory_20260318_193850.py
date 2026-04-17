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
    goal: str = ""  # Natural language goal
    relevant_transforms: List[str] = field(default_factory=list)
    recently_failed: List[str] = field(default_factory=list)
    hypothesis: str = ""  # Current conjecture
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

    # Approved change: add a robust smoothing+confidence-based priority score
    # and make focus() actually prime recently_failed / goal / hypothesis
    # with stable state.

    def __init__(self):
        self._state = WorkingMemoryState()
        # Domain→transform success stats: {domain: {t_name: [successes, attempts]}}
        self._domain_stats: Dict[str, Dict[str, List[int]]] = {}
        self._history: Deque[dict] = deque(maxlen=self.CAPACITY)

    # ── Attention: Prioritize transforms for the current domain ──

    def get_prioritized_transforms(self, transforms: list, domain: str) -> list:
        """
        Re-rank transforms by domain-specific success rate.

        Uses smoothed success probability (Beta prior) and a recency penalty for
        recently_failed transforms. Returns the input list order when no
        domain-specific stats exist yet.
        """
        stats = self._domain_stats.get(domain, {})
        if not stats:
            return transforms  # No data yet: use default order

        # Beta prior smoothing parameters; chosen to be neutral-ish early.
        # Prior mean = alpha/(alpha+beta). With alpha=beta=1 => mean 0.5.
        alpha = 1.0
        beta = 1.0

        recently_failed = set(self._state.recently_failed)

        def score(t) -> float:
            name = t.name()
            if name in stats:
                s, a = stats[name]
                # Smoothed success probability
                p = (alpha + float(s)) / (alpha + beta + float(a))

                # Confidence adjustment: favor transforms with more evidence
                # to prevent noisy early spikes.
                evidence = min(float(a), 20.0) / 20.0  # 0..1
                confidence = 0.25 + 0.75 * evidence  # 0.25..1.0

                # Recency penalty for known recent failures (softly)
                if name in recently_failed:
                    p *= 0.25  # stronger penalty to push to bottom

                return p * confidence
            # Unseen in this domain: neutral score
            base_p = 0.5
            if name in recently_failed:
                base_p *= 0.25
            return base_p * 0.25  # keep low priority until evidence exists

        # Deterministic tie-break: fallback to original order by stable sort key
        # via enumerate.
        indexed = list(enumerate(transforms))

        def sort_key(pair):
            _, t = pair
            return score(t)

        indexed_sorted = sorted(indexed, key=sort_key, reverse=True)
        return [t for _, t in indexed_sorted]

    def get_domain_top_transforms(self, domain: str, n: int = 5) -> List[str]:
        """Get the n best-performing transforms for this domain."""
        stats = self._domain_stats.get(domain, {})
        if not stats:
            return []

        # Smoothed mean success probability
        alpha = 1.0
        beta = 1.0

        def smoothed_rate(item):
            name, (s, a) = item
            p = (alpha + float(s)) / (alpha + beta + float(a))
            # Prefer more evidence slightly to reduce ties
            evidence = min(float(a), 20.0) / 20.0
            return (p, 0.25 + 0.75 * evidence)

        ranked = sorted(stats.items(), key=smoothed_rate, reverse=True)
        return [name for name, _ in ranked[:n]]

    # ── State management ──────────────────────────────────────

    def focus(self, domain: str, goal: str = "", hypothesis: str = ""):
        """
        Set working memory focus for a new solve.

        Approved change: preserve intent (domain/goal/hypothesis) while resetting
        recently_failed in a controlled manner, and prime relevant_transforms
        using current domain statistics.
        """
        top = self.get_domain_top_transforms(domain, n=10)

        # Reset recent failures between solves to avoid over-punishing across
        # independent episodes.
        self._state = WorkingMemoryState(
            domain=domain,
            goal=goal or "",
            relevant_transforms=top,
            recently_failed=[],
            hypothesis=hypothesis or "",
        )

    def record_failure(self, transform_name: str):
        """Mark a transform as recently failed (de-prioritize)."""
        if not transform_name:
            return
        if transform_name not in self._state.recently_failed:
            self._state.recently_failed.append(transform_name)
            # Keep bounded recent-failure list for stability
            if len(self._state.recently_failed) > 10:
                self._state.recently_failed.pop(0)

    def record_outcome(
        self,
        domain: str,
        transforms_used: List[str],
        success: bool,
        delta: float,
    ):
        """
        Update domain stats based on solve outcome.

        Approved change: incorporate delta into the learning signal by
        optionally treating negative delta as failure signal and positive delta
        as amplified success. This improves prioritization in
        get_prioritized_transforms without changing interfaces.
        """
        if domain not in self._domain_stats:
            self._domain_stats[domain] = {}

        # Determine effective success based on delta magnitude:
        # - If success is True but delta is very negative, down-weight success.
        # - If success is False but delta is very positive, up-weight failure.
        # This is intentionally conservative to avoid destabilizing early learning.
        eff_success = bool(success)
        if delta is not None:
            try:
                d = float(delta)
                # Thresholds: tuned for "most deltas are small"; only flip on strong mismatch.
                if success and d < -0.5:
                    eff_success = False
                elif (not success) and d > 0.5:
                    eff_success = True
            except (TypeError, ValueError):
                pass

        # Convert effective success to integer update (single-step per attempt).
        for t_name in transforms_used:
            if not t_name:
                continue
            if t_name not in self._domain_stats[domain]:
                self._domain_stats[domain][t_name] = [0, 0]
            # attempts always increment
            self._domain_stats[domain][t_name][1] += 1
            if eff_success:
                self._domain_stats[domain][t_name][0] += 1

            # If this attempt failed, mark as recently_failed to immediately
            # influence the current solve's subsequent selection (if any).
            if not eff_success:
                self.record_failure(t_name)

        self._history.append(
            {
                "domain": domain,
                "transforms": list(transforms_used),
                "success": bool(eff_success),
                "delta": float(delta) if delta is not None else None,
            }
        )

    # ── Causal reasoning hint for search ──────────────────────

    def predict_useful_transforms(self, domain: str, graph, all_transforms: list) -> List[str]:
        """
        P4.2 causal: predict which transforms are likely useful
        based on graph structure + domain success history.
        Returns ordered list of transform names.
        """
        predictions: List[str] = []

        # Historical success in this domain
        domain_best = self.get_domain_top_transforms(domain, n=10)
        predictions.extend(domain_best)

        # Structural: look at operators in graph
        if graph is not None:
            try:
                labels = {n.label for n in graph.nodes if getattr(n, "type", None) == "operator"}
            except Exception:
                labels = set()

            for t in all_transforms:
                name = t.name()
                if name not in predictions:
                    if any(op in name or name in op for op in labels):
                        predictions.append(name)

        return predictions

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> dict:
        domain_summary = {}
        for domain, ts in self._domain_stats.items():
            attempts = sum(a for _, a in ts.values())
            successes = sum(s for s, _ in ts.values())
            rate = float(successes) / float(attempts) if attempts else 0.0
            domain_summary[domain] = {
                "attempts": attempts,
                "successes": successes,
                "success_rate": rate,
                "top_transforms": self.get_domain_top_transforms(domain, n=5),
            }
        return {
            "current_state": {
                "domain": self._state.domain,
                "goal": self._state.goal,
                "hypothesis": self._state.hypothesis,
                "relevant_transforms": list(self._state.relevant_transforms),
                "recently_failed": list(self._state.recently_failed),
                "context_tags": list(self._state.context_tags),
            },
            "domains": domain_summary,
            "history_len": len(self._history),
        }