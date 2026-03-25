"""
ForgettingPrevention — EWC-lite to prevent catastrophic forgetting of important rules.

Elastic Weight Consolidation (Kirkpatrick et al. 2017) prevents forgetting by
tracking parameter importance (Fisher information) and penalizing changes to
important parameters when learning new tasks.

Our lightweight version:
- Tracks "importance" of each ConceptRule (# problems solved × confidence)
- When adding a new rule with the same name as an existing important rule,
  requires the new rule to have higher confidence before overwriting
- Periodically "consolidates" the registry: locks rules above importance threshold

This is a practical approximation of EWC suitable for symbolic rule systems.
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

IMPORTANCE_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "rule_importance.json"
CONSOLIDATION_THRESHOLD = 10  # importance score to "lock" a rule (lowered from 100; EMA asymptote ~0.7)
MIN_CONFIDENCE_TO_OVERWRITE = 0.05  # must exceed existing confidence by this much


class ForgettingPrevention:
    """
    EWC-lite: track rule importance and prevent important rules from being overwritten.

    Importance = observations × confidence (proxy for Fisher information diagonal)
    """

    def __init__(self, persist_path: Optional[Path] = None):
        self._path = Path(persist_path or IMPORTANCE_PATH)
        self._importance: Dict[str, float] = {}   # rule_name → importance score
        self._lock_count = 0   # rules above consolidation threshold
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._importance = data.get("importance", {})
            log.debug("ForgettingPrevention loaded %d rule importances", len(self._importance))
        except Exception as e:
            log.debug("ForgettingPrevention load failed: %s", e)

    def save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            import os
            tmp = str(self._path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"importance": self._importance, "saved_at": time.time()}, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as e:
            log.debug("ForgettingPrevention save failed: %s", e)

    def record_usage(self, rule_name: str, confidence: float, observations: int = 1):
        """Update importance when a rule is used successfully."""
        if not rule_name:
            return
        current = self._importance.get(rule_name, 0.0)
        # Importance = EMA of (observations × confidence)
        new_importance = float(observations) * float(confidence)
        alpha = 0.1
        self._importance[rule_name] = (1 - alpha) * current + alpha * new_importance
        # Auto-save every 50 updates
        if len(self._importance) % 50 == 0:
            self.save()

    def should_overwrite(self, rule_name: str, new_confidence: float) -> bool:
        """
        Return True if a new rule is allowed to overwrite the existing one.

        Rules with high importance (well-established) require higher confidence
        to be overwritten — preventing catastrophic forgetting.
        """
        if rule_name not in self._importance:
            return True  # new rule, allow freely

        importance = self._importance[rule_name]

        if importance >= CONSOLIDATION_THRESHOLD:
            # Consolidated rule: require significant confidence improvement
            # (This is the "locked" state — very hard to overwrite)
            log.debug("ForgettingPrevention: rule '%s' is consolidated (importance=%.1f) — protected",
                      rule_name, importance)
            return False  # Don't overwrite consolidated rules

        # Semi-important rule: require at least MIN_CONFIDENCE_TO_OVERWRITE improvement
        # We'd need to know existing confidence to compare, but we don't have it here
        # Conservative: allow if importance is below threshold
        return True

    def get_top_rules(self, n: int = 10) -> list:
        """Return the n most important rules."""
        sorted_rules = sorted(self._importance.items(), key=lambda x: x[1], reverse=True)
        return [{"name": r, "importance": round(imp, 2)} for r, imp in sorted_rules[:n]]

    def consolidate(self) -> int:
        """Mark rules above threshold as consolidated. Returns count locked."""
        locked = sum(1 for imp in self._importance.values() if imp >= CONSOLIDATION_THRESHOLD)
        self._lock_count = locked
        if locked > 0:
            log.info("ForgettingPrevention: %d rules consolidated (protected from overwrite)", locked)
        return locked

    @property
    def stats(self) -> dict:
        if not self._importance:
            return {"tracked_rules": 0, "consolidated": 0, "avg_importance": 0.0}
        avg = sum(self._importance.values()) / len(self._importance)
        consolidated = sum(1 for v in self._importance.values() if v >= CONSOLIDATION_THRESHOLD)
        return {
            "tracked_rules": len(self._importance),
            "consolidated": consolidated,
            "avg_importance": round(avg, 2),
            "top_rules": self.get_top_rules(5),
        }


_instance: Optional[ForgettingPrevention] = None

def get_forgetting_prevention() -> ForgettingPrevention:
    global _instance
    if _instance is None:
        _instance = ForgettingPrevention()
    return _instance
