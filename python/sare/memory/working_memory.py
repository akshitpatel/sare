import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List

# Optional imports – if modules are unavailable we degrade gracefully
try:
    from memory.credit_assigner import get_credit_assigner  # type: ignore
except Exception:  # pragma: no cover
    get_credit_assigner = None

try:
    from neuro.dopamine import get_dopamine_system  # type: ignore
except Exception:  # pragma: no cover
    get_dopamine_system = None

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
    Lightweight per‑solve working memory with domain‑specific attention.

    The class tracks per‑domain transform success statistics, recent failures,
    and integrates signals from the CreditAssigner and DopamineSystem to close
    the feedback loops between learning, motivation, and attention.
    """

    CAPACITY = 500  # max episodes to remember

    def __init__(self):
        self._state = WorkingMemoryState()
        # Domain → transform → [successes, attempts]
        self._domain_stats: Dict[str, Dict[str, List[int]]] = {}
        self._history: Deque[dict] = deque(maxlen=self.CAPACITY)
        self._session_count: int = 0
        self._update_count: int = 0
        self.load()

    # ── Attention: Prioritize transforms for the current domain ──

    def get_prioritized_transforms(self, transforms: list, domain: str) -> list:
        """
        Re‑rank ``transforms`` using a blend of:
          * Domain‑specific smoothed success probability (Beta prior)
          * Recency penalty for recently failed transforms
          * Utility scores from ``CreditAssigner`` (if available)
          * Exploration boost when dopamine tonic level signals “explore”

        The function returns a new list ordered from highest to lowest priority.
        """
        # Fast‑path: no statistics → preserve original order
        stats = self._domain_stats.get(domain, {})
        if not stats:
            return transforms

        # Hyper‑parameters
        alpha = 1.0
        beta = 1.0
        recent_failed = set(self._state.recently_failed)

        # Optional external signals
        credit_assigner = get_credit_assigner() if callable(get_credit_assigner) else None
        dopamine = get_dopamine_system() if callable(get_dopamine_system) else None

        # Determine exploration factor from dopamine (0.5 = neutral, >0.5 = more exploratory)
        exploration_factor = 1.0
        if dopamine is not None:
            # tonic in [0,1]; map to [0.8,1.2] where higher tonic → more exploration
            tonic = getattr(dopamine, "tonic", 0.5)
            exploration_factor = 0.8 + 0.4 * tonic

        def score(transform) -> float:
            name = transform.name()
            # Base smoothed success probability
            if name in stats:
                successes, attempts = stats[name]
                p = (alpha + successes) / (alpha + beta + attempts)

                # Confidence weighting (more evidence → higher confidence)
                evidence = min(attempts, 20) / 20.0  # 0..1
                confidence = 0.25 + 0.75 * evidence
                base_score = p * confidence
            else:
                # Unseen transforms get a neutral baseline
                base_score = 0.5 * 0.25

            # Recency penalty for recent failures
            if name in recent_failed:
                base_score *= 0.25

            # CreditAssigner utility (if available)
            if credit_assigner is not None:
                try:
                    util = credit_assigner.get_utility(name)  # expects float in [0,1]
                    # Blend utility gently – weight 0.3
                    base_score = 0.7 * base_score + 0.3 * util
                except Exception:  # pragma: no cover
                    pass

            # Exploration boost from dopamine
            base_score *= exploration_factor

            return base_score

        # Stable deterministic tie‑break: preserve original order when scores equal
        indexed = list(enumerate(transforms))
        indexed.sort(key=lambda pair: (score(pair[1]), -pair[0]), reverse=True)
        return [t for _, t in indexed]

    def get_domain_top_transforms(self, domain: str, n: int = 5) -> List[str]:
        """Return the ``n`` best‑performing transform names for ``domain``."""
        stats = self._domain_stats.get(domain, {})
        if not stats:
            return []

        alpha = 1.0
        beta = 1.0

        def smoothed(item):
            name, (s, a) = item
            p = (alpha + s) / (alpha + beta + a)
            evidence = min(a, 20) / 20.0
            confidence = 0.25 + 0.75 * evidence
            return (p * confidence, name)

        ranked = sorted(stats.items(), key=smoothed, reverse=True)
        return [name for name, _ in ranked[:n]]

    # ── State management ──────────────────────────────────────

    def focus(self, domain: str, goal: str = "", hypothesis: str = ""):
        """
        Initialise a new solving episode.  The method:
          * Updates the domain/goal/hypothesis fields.
          * Clears recent‑failure history.
          * Primes ``relevant_transforms`` with the top‑10 transforms for the domain.
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
        """Mark a transform as recently failed so it is de‑prioritized."""
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
        Update statistics after a solve attempt.

        * ``success`` increments the success counter for each used transform.
        * ``attempts`` are always incremented.
        * The episode is stored in the internal history buffer.
        * Periodically persists the updated statistics to disk.
        """
        dom_stats = self._domain_stats.setdefault(domain, {})
        s_add = 1 if success else 0

        for t_name in transforms_used:
            if not t_name:
                continue
            if t_name not in dom_stats:
                dom_stats[t_name] = [0, 0]  # [successes, attempts]
            successes, attempts = dom_stats[t_name]
            dom_stats[t_name] = [successes + s_add, attempts + 1]

        # Record episode history
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "domain": domain,
            "transforms_used": transforms_used,
            "success": success,
            "delta": delta,
        }
        self._history.append(entry)
        self._session_count += 1
        self._update_count += 1

        # Persist every 50 updates to avoid excessive I/O
        if self._update_count >= 50:
            self.save()
            self._update_count = 0

    # ── Persistence ─────────────────────────────────────────────

    def load(self):
        """Load working‑memory state from disk if the file exists."""
        if not _WM_PATH.is_file():
            log.debug("WorkingMemory file not found; starting fresh.")
            return
        try:
            with open(_WM_PATH, "r", encoding="utf8") as f:
                data = json.load(f)
            self._domain_stats = data.get("domain_stats", {})
            # History is optional – keep existing buffer if missing
            hist = data.get("history", [])
            for item in hist:
                self._history.append(item)
            self._session_count = data.get("session_count", 0)
            self._update_count = data.get("update_count", 0)
            log.debug("WorkingMemory loaded from %s", _WM_PATH)
        except Exception as e:  # pragma: no cover
            log.error("Failed to load WorkingMemory from %s: %s", _WM_PATH, e)

    def save(self):
        """Serialise the current working‑memory statistics to disk."""
        try:
            data = {
                "domain_stats": self._domain_stats,
                "history": list(self._history),
                "session_count": self._session_count,
                "update_count": self._update_count,
            }
            os.makedirs(_WM_PATH.parent, exist_ok=True)
            with open(_WM_PATH, "w", encoding="utf8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            log.debug("WorkingMemory saved to %s", _WM_PATH)
        except Exception as e:  # pragma: no cover
            log.error("Failed to save WorkingMemory to %s: %s", _WM_PATH, e)

    # ── Utility ────────────────────────────────────────────────

    def snapshot(self) -> WorkingMemoryState:
        """Return a shallow copy of the current working‑memory state."""
        return WorkingMemoryState(
            domain=self._state.domain,
            goal=self._state.goal,
            relevant_transforms=list(self._state.relevant_transforms),
            recently_failed=list(self._state.recently_failed),
            hypothesis=self._state.hypothesis,
            context_tags=list(self._state.context_tags),
        )