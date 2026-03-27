"""
ForgettingCurve — Ebbinghaus Memory Decay + Leitner Spaced Repetition
======================================================================

Biological basis:
  - Memories decay exponentially over time unless rehearsed.
  - Each successful review increases the memory's "stability" — how long
    before it decays to forgetting threshold again.
  - This is the Ebbinghaus forgetting curve:
      strength(t) = S0 * exp(-decay_rate * t / stability)

Leitner boxes (0..5):
  - Items start in box 0 (daily review)
  - Correct recall → advance to next box (longer interval)
  - Wrong recall → reset to box 0
  - Box intervals: [1, 2, 4, 8, 16, 32] days

Integration points:
  - DopamineSystem.encoding_strength → strength_0 when registering new memories
  - HippocampusDaemon sleep cycle → decay_all() + get_due_reviews()
  - DevelopmentalCurriculum.next_problem() → checks due reviews before ZPD
  - MemoryManager.store() → registers each solve episode
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_MEMORY = Path(__file__).resolve().parents[4] / "data" / "memory"
LEITNER_INTERVALS = [1, 2, 4, 8, 16, 32]   # days


@dataclass
class MemoryItem:
    """A single memory item tracked by the forgetting curve."""
    item_id:         str
    item_type:       str   # "rule", "episode", "concept", "transform", "domain"
    domain:          str
    strength:        float = 1.0       # current memory strength [0, 1]
    stability:       float = 1.0       # how resistant to decay (grows with reviews)
    leitner_box:     int   = 0         # [0..5]
    review_count:    int   = 0
    last_reviewed:   float = field(default_factory=time.time)
    registered_at:   float = field(default_factory=time.time)
    due_at:          float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "item_id":      self.item_id,
            "item_type":    self.item_type,
            "domain":       self.domain,
            "strength":     round(self.strength, 4),
            "stability":    round(self.stability, 4),
            "leitner_box":  self.leitner_box,
            "review_count": self.review_count,
            "last_reviewed": self.last_reviewed,
            "registered_at": self.registered_at,
            "due_at":        self.due_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryItem":
        return cls(
            item_id=d.get("item_id", ""),
            item_type=d.get("item_type", "episode"),
            domain=d.get("domain", "general"),
            strength=float(d.get("strength", 1.0)),
            stability=float(d.get("stability", 1.0)),
            leitner_box=int(d.get("leitner_box", 0)),
            review_count=int(d.get("review_count", 0)),
            last_reviewed=float(d.get("last_reviewed", time.time())),
            registered_at=float(d.get("registered_at", time.time())),
            due_at=float(d.get("due_at", time.time())),
        )


class ForgettingCurve:
    """
    Ebbinghaus decay model with Leitner spaced repetition scheduling.

    Key formula:
        strength(t) = S0 * exp(-decay_rate * Δt_days / stability)

    where:
        S0         = initial encoding strength (from DopamineSystem.encoding_strength)
        decay_rate = STAGE_CAPABILITY_GATES[stage]["decay_rate"]
        stability  = grows +0.3 per successful review (better drilled → slower decay)
    """

    PERSIST_PATH_ITEMS = _MEMORY / "forgetting_curve.json"
    FORGETTING_THRESHOLD = 0.3    # below this → flagged for immediate review
    MAX_ITEMS = 5000              # cap to prevent unbounded growth

    def __init__(self):
        self._items: Dict[str, MemoryItem] = {}
        self._decay_rate: float = 0.01   # updated by brain via set_decay_rate()
        self._last_decay_time: float = time.time()
        self._ops_since_save: int = 0
        self._load()

    def set_decay_rate(self, rate: float):
        """Called by brain when stage advances (stage gates the decay rate)."""
        self._decay_rate = max(0.0001, min(0.1, rate))

    def register(
        self,
        item_id: str,
        item_type: str,
        domain: str = "general",
        encoding_strength: float = 1.0,
    ) -> MemoryItem:
        """
        Register a new memory item.

        encoding_strength comes from DopamineSystem.encoding_strength —
        high surprise → stronger initial encoding.
        """
        # Clamp encoding strength to [0.5, 2.0]
        s0 = max(0.5, min(2.0, encoding_strength))
        now = time.time()
        item = MemoryItem(
            item_id=item_id,
            item_type=item_type,
            domain=domain,
            strength=min(1.0, s0 / 2.0),  # normalise to [0,1]
            stability=s0,                   # higher encoding → higher initial stability
            leitner_box=0,
            registered_at=now,
            last_reviewed=now,
            due_at=now + LEITNER_INTERVALS[0] * 86400,  # review in 1 day
        )
        self._items[item_id] = item
        self._maybe_prune()
        self._ops_since_save += 1
        if self._ops_since_save >= 50:
            self._save()
            self._ops_since_save = 0
        return item

    def decay_all(self, current_time: Optional[float] = None) -> int:
        """
        Apply exponential decay to all items.
        Returns number of items that crossed below FORGETTING_THRESHOLD.
        """
        now = current_time or time.time()
        at_risk = 0
        for item in self._items.values():
            elapsed_days = (now - item.last_reviewed) / 86400.0
            if elapsed_days <= 0:
                continue
            # Ebbinghaus: strength(t) = strength * exp(-k * t / stability)
            decay = math.exp(-self._decay_rate * elapsed_days / max(item.stability, 0.1))
            item.strength = max(0.0, item.strength * decay)
            if item.strength < self.FORGETTING_THRESHOLD:
                at_risk += 1
        self._last_decay_time = now
        self._save()
        return at_risk

    def get_due_reviews(self, limit: int = 20, current_time: Optional[float] = None) -> List[MemoryItem]:
        """
        Return items that are due for review, sorted by lowest strength first.
        Items below FORGETTING_THRESHOLD are always included regardless of due_at.
        """
        now = current_time or time.time()
        due = [
            item for item in self._items.values()
            if item.due_at <= now or item.strength < self.FORGETTING_THRESHOLD
        ]
        return sorted(due, key=lambda x: x.strength)[:limit]

    def record_review(self, item_id: str, recalled: bool) -> Optional[MemoryItem]:
        """
        Record the outcome of a review session.
        recalled=True  → advance Leitner box, increase stability
        recalled=False → reset to box 0
        """
        item = self._items.get(item_id)
        if item is None:
            return None

        now = time.time()
        item.review_count += 1
        item.last_reviewed = now

        if recalled:
            # Advance Leitner box
            item.leitner_box = min(5, item.leitner_box + 1)
            # Increase stability (well-drilled items decay slower)
            item.stability = min(10.0, item.stability + 0.3)
            # Restore strength
            item.strength = min(1.0, item.strength + 0.4)
        else:
            # Reset to box 0
            item.leitner_box = 0
            item.stability = max(0.1, item.stability * 0.8)
            item.strength = max(0.1, item.strength * 0.5)

        # Schedule next review based on Leitner box
        interval_days = LEITNER_INTERVALS[item.leitner_box]
        item.due_at = now + interval_days * 86400

        self._ops_since_save += 1
        if self._ops_since_save >= 10:
            self._save()
            self._ops_since_save = 0
        return item

    def get_item(self, item_id: str) -> Optional[MemoryItem]:
        return self._items.get(item_id)

    def get_stats(self) -> dict:
        items = list(self._items.values())
        if not items:
            return {
                "total_items": 0, "at_risk": 0, "avg_strength": 1.0,
                "avg_stability": 1.0, "leitner_distribution": {},
                "decay_rate": self._decay_rate,
            }
        avg_strength = sum(i.strength for i in items) / len(items)
        avg_stability = sum(i.stability for i in items) / len(items)
        at_risk = sum(1 for i in items if i.strength < self.FORGETTING_THRESHOLD)
        leitner_dist: Dict[int, int] = {}
        for i in items:
            leitner_dist[i.leitner_box] = leitner_dist.get(i.leitner_box, 0) + 1
        due_now = len(self.get_due_reviews(limit=1000))
        return {
            "total_items": len(items),
            "at_risk": at_risk,
            "due_now": due_now,
            "avg_strength": round(avg_strength, 3),
            "avg_stability": round(avg_stability, 3),
            "leitner_distribution": {str(k): v for k, v in sorted(leitner_dist.items())},
            "decay_rate": self._decay_rate,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _maybe_prune(self):
        if len(self._items) > self.MAX_ITEMS:
            # Remove strongest (least at-risk) items beyond cap
            sorted_items = sorted(self._items.values(), key=lambda x: x.strength, reverse=True)
            to_remove = sorted_items[self.MAX_ITEMS:]
            for item in to_remove:
                del self._items[item.item_id]

    def _save(self):
        try:
            _MEMORY.mkdir(parents=True, exist_ok=True)
            tmp = self.PERSIST_PATH_ITEMS.with_suffix(".tmp")
            data = {
                "decay_rate": self._decay_rate,
                "last_decay_time": self._last_decay_time,
                "items": [item.to_dict() for item in list(self._items.values())[-3000:]],
            }
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.PERSIST_PATH_ITEMS)
        except OSError as e:
            log.debug("[ForgettingCurve] Save error: %s", e)

    def _load(self):
        if not self.PERSIST_PATH_ITEMS.exists():
            return
        try:
            d = json.loads(self.PERSIST_PATH_ITEMS.read_text())
            self._decay_rate = float(d.get("decay_rate", 0.01))
            self._last_decay_time = float(d.get("last_decay_time", time.time()))
            for item_dict in d.get("items", []):
                item = MemoryItem.from_dict(item_dict)
                self._items[item.item_id] = item
            log.debug("[ForgettingCurve] Loaded %d items", len(self._items))
        except Exception as e:
            log.debug("[ForgettingCurve] Load error: %s", e)


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: Optional[ForgettingCurve] = None


def get_forgetting_curve() -> ForgettingCurve:
    global _instance
    if _instance is None:
        _instance = ForgettingCurve()
    return _instance
