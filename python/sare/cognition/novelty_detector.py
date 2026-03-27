"""
NoveltyDetector — Penalizes unoriginal problems, rewards genuine novelty.

A problem is "novel" if its structural hash doesn't exist in the schema cache
AND its domain pattern hasn't been seen recently. This drives the system toward
genuinely new territory rather than repeating easy wins.

Used by CurriculumGenerator to boost novel problems' selection probability.
"""
from __future__ import annotations
import hashlib, json, logging, time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional
import numpy as np

log = logging.getLogger(__name__)
HISTORY_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "novelty_history.json"

class NoveltyDetector:
    def __init__(self, window: int = 200):
        self._window = window
        self._recent: deque = deque(maxlen=window)   # structural hashes seen recently
        self._domain_counts: Dict[str, int] = defaultdict(int)
        self._total = 0
        self._load()

    def _hash(self, expression: str) -> str:
        return hashlib.md5(expression.lower().strip().encode()).hexdigest()[:12]

    def score(self, expression: str, domain: str = "general") -> float:
        """
        Return a novelty score 0.0-1.0.
        1.0 = completely new (never seen this structure or domain recently)
        0.0 = exact repeat seen many times
        """
        h = self._hash(expression)
        # Structural novelty: how recently was this hash seen?
        recent_list = list(self._recent)
        structural_recency = recent_list.count(h) / max(len(recent_list), 1)
        structural_novelty = 1.0 - min(1.0, structural_recency * 10)

        # Domain novelty: is this domain over-represented recently?
        domain_freq = self._domain_counts.get(domain, 0) / max(self._total, 1)
        domain_novelty = 1.0 - min(1.0, domain_freq * 5)

        # Combined novelty (geometric mean weights structural more)
        novelty = 0.7 * structural_novelty + 0.3 * domain_novelty
        return round(max(0.05, novelty), 3)  # floor at 0.05 so nothing is completely suppressed

    def record(self, expression: str, domain: str = "general"):
        """Record that this expression was attempted."""
        h = self._hash(expression)
        self._recent.append(h)
        self._domain_counts[domain] = self._domain_counts.get(domain, 0) + 1
        self._total += 1
        if self._total % 100 == 0:
            self.save()

    def _load(self):
        try:
            if HISTORY_PATH.exists():
                d = json.loads(HISTORY_PATH.read_text())
                self._domain_counts = defaultdict(int, d.get("domain_counts", {}))
                self._total = d.get("total", 0)
                for h in d.get("recent", []):
                    self._recent.append(h)
        except Exception as e:
            log.debug("NoveltyDetector load: %s", e)

    def save(self):
        try:
            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            import os
            tmp = str(HISTORY_PATH) + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"domain_counts": dict(self._domain_counts),
                           "total": self._total,
                           "recent": list(self._recent)[-100:]}, f)
            os.replace(tmp, HISTORY_PATH)
        except Exception as e:
            log.debug("NoveltyDetector save: %s", e)

    @property
    def stats(self) -> dict:
        return {"total_seen": self._total, "window": self._window,
                "unique_domains": len(self._domain_counts),
                "recent_buffer": len(self._recent)}

_instance: Optional[NoveltyDetector] = None
def get_novelty_detector() -> NoveltyDetector:
    global _instance
    if _instance is None: _instance = NoveltyDetector()
    return _instance
