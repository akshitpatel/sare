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

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List

_WM_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "working_memory.json"

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

    def __init__(self):
        self._state = WorkingMemoryState()
        # Domain→transform success stats: {domain: {t_name: [successes, attempts]}}
        self._domain_stats: Dict[str, Dict[str, List[int]]] = {}
        self._history: Deque[dict] = deque(maxlen=self.CAPACITY)
        self._session_count: int = 0
        self._update_count: int = 0
        self.load()

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

            # Unseen in this domain: neutral score, but deprioritize if recently failed
            base_p = 0.5
            if name in recently_failed:
                base_p *= 0.25
            return base_p * 0.25  # keep low priority until evidence exists

        # Stable deterministic tie-break: stable sort with enumerate index
        indexed = list(enumerate(transforms))
        indexed_sorted = sorted(indexed, key=lambda pair: (score(pair[1]), -pair[0]), reverse=True)
        return [t for _, t in indexed_sorted]

    def get_domain_top_transforms(self, domain: str, n: int = 5) -> List[str]:
        """Get the n best-performing transforms for this domain."""
        stats = self._domain_stats.get(domain, {})
        if not stats:
            return []

        alpha = 1.0
        beta = 1.0

        def smoothed_rate(item):
            name, (s, a) = item
            p = (alpha + float(s)) / (alpha + beta + float(a))
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
            if len(self._state.recently_failed) > 10:
                self._state.recently_failed.pop(0)

    def record_outcome(
        self,
        domain: str,
        transforms_used: List[str],
        success: bool,
        delta: float = 0.0,
    ):
        """
        Update transform success statistics after a solve attempt.
        """
        dom_stats = self._domain_stats.setdefault(domain, {})
        s_add = 1 if success else 0

        for t_name in transforms_used:
            if not t_name:
                continue
            if t_name not in dom_stats:
                dom_stats[t_name] = [0, 0]  # successes, attempts
            dom_stats[t_name][1] += 1  # attempts
            dom_stats[t_name][0] += s_add  # successes

        episode = {
            "domain": domain,
            "transforms_used": list(transforms_used),
            "success": bool(success),
            "delta": float(delta),
        }
        self._history.append(episode)

        self._update_count += 1
        if self._update_count % 50 == 0:
            self.save()

        return episode

    # ── Persistence ───────────────────────────────────────────

    def save(self) -> None:
        try:
            _WM_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "domain_stats": self._domain_stats,
                "session_count": self._session_count,
            }
            tmp = _WM_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, _WM_PATH)
        except Exception as e:
            log.debug("[WorkingMemory] save failed: %s", e)

    def load(self) -> None:
        try:
            if _WM_PATH.exists():
                data = json.loads(_WM_PATH.read_text())
                self._domain_stats  = data.get("domain_stats", {})
                self._session_count = data.get("session_count", 0)
                self._session_count += 1   # increment on each new session
        except Exception as e:
            log.debug("[WorkingMemory] load failed: %s", e)

    # ── Introspection (debug / UI) ───────────────────────────

    def record_transform(self, domain: str, transform_name: str, success: bool) -> None:
        """Convenience wrapper: record a single transform outcome."""
        self.record_outcome(domain, [transform_name], success)

    def get_state(self) -> WorkingMemoryState:
        return self._state

    def get_stats(self, domain: str) -> Dict[str, List[int]]:
        return self._domain_stats.get(domain, {})