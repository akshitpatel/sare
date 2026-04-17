"""
Homeostatic System — Drive-based behavior regulation.

Biological analogy: internal drives (hunger, curiosity, social needs) push
the system toward balanced, healthy behavior. When curiosity is high, explore.
When mastery drive peaks, deepen a weak domain. When social drive rises,
seek human input. When consolidation is overdue, run memory sleep cycles.

Drives:
  curiosity      — need to encounter new patterns / explore
  mastery        — need to achieve competence in a domain
  social         — need for human interaction / teaching
  exploration    — need to try new domains
  consolidation  — need to consolidate/sleep-compress memories

Each drive has:
  - level: 0.0 (satisfied) to 1.0 (urgent)
  - decay_rate: how fast it builds when unsatisfied (per tick)
  - last_satisfied: timestamp

Data stored in: data/memory/homeostasis.json
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).resolve().parents[3] / "data" / "memory"
HOMEOSTASIS_PATH = MEMORY_DIR / "homeostasis.json"


@dataclass
class Drive:
    """One internal drive of the homeostatic system."""
    name: str
    level: float = 0.5          # 0.0 (satisfied) to 1.0 (urgent)
    decay_rate: float = 0.01    # how fast level grows per tick (toward 1.0)
    last_satisfied: float = field(default_factory=time.time)

    def tick(self, elapsed_seconds: float = 60.0):
        """Increase drive level with elapsed time (unsatisfied drives grow)."""
        growth = self.decay_rate * (elapsed_seconds / 60.0)
        self.level = min(1.0, self.level + growth)

    def satisfy(self, amount: float = 0.3):
        """Reduce drive level by amount (clamped to [0, 1])."""
        self.level = max(0.0, self.level - amount)
        self.last_satisfied = time.time()

    def urgency(self) -> str:
        if self.level >= 0.85:
            return "urgent"
        elif self.level >= 0.6:
            return "elevated"
        elif self.level >= 0.35:
            return "moderate"
        else:
            return "satisfied"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level": round(self.level, 4),
            "decay_rate": self.decay_rate,
            "last_satisfied": self.last_satisfied,
            "urgency": self.urgency(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Drive":
        dr = cls(
            name=d["name"],
            level=d.get("level", 0.5),
            decay_rate=d.get("decay_rate", 0.01),
        )
        dr.last_satisfied = d.get("last_satisfied", time.time())
        return dr


# Default drive configuration — decay rates reduced 40% to prevent drives
# from maxing out within minutes (they should build over hours).
_DEFAULT_DRIVES = {
    "curiosity":     Drive("curiosity",     level=0.7,  decay_rate=0.005),
    "mastery":       Drive("mastery",       level=0.5,  decay_rate=0.004),
    "social":        Drive("social",        level=0.6,  decay_rate=0.006),
    "exploration":   Drive("exploration",   level=0.4,  decay_rate=0.004),
    "consolidation": Drive("consolidation", level=0.3,  decay_rate=0.003),
}

# Behavior recommendation thresholds
_BEHAVIOR_MAP = [
    # (drive_name, min_level) → behavior recommendation
    ("social",        0.75, "seek_human_input"),
    ("consolidation", 0.70, "consolidate_memory"),
    ("curiosity",     0.75, "explore_new_domain"),
    ("mastery",       0.70, "deepen_weak_domain"),
    ("exploration",   0.65, "generate_analogies"),
    ("curiosity",     0.50, "explore_new_domain"),
    ("mastery",       0.50, "deepen_weak_domain"),
]


class HomeostaticSystem:
    """
    Regulates SARE-HX's internal drives, steering behavior toward balance.

    The system "wants" different things at different times — driven not by
    explicit programming but by the buildup of unsatisfied internal needs.

    Callers can:
      - Call tick() periodically to age drives
      - Call satisfy(drive, amount) when an activity satisfies a drive
      - Call on_* convenience methods after specific events
      - Query get_behavior_recommendation() for the highest-priority action
    """

    DEFAULT_PATH = HOMEOSTASIS_PATH

    def __init__(self, persist_path: Optional[Path] = None):
        self._path = Path(persist_path or self.DEFAULT_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self.drives: Dict[str, Drive] = {
            name: Drive(d.name, d.level, d.decay_rate)
            for name, d in _DEFAULT_DRIVES.items()
        }
        self._last_tick: float = time.time()
        self._tick_count: int = 0

        self.load()

    # ── Tick & satisfy ────────────────────────────────────────────────────────

    def tick(self):
        """
        Advance time: all drives decay (build up) based on elapsed real time.
        Should be called periodically (e.g., every 60 seconds).
        """
        now = time.time()
        elapsed = now - self._last_tick
        self._last_tick = now
        self._tick_count += 1

        for drive in self.drives.values():
            drive.tick(elapsed_seconds=elapsed)

        if self._tick_count % 10 == 0:
            self.save()
            # Inner monologue: narrate drive state every 10 ticks
            try:
                dominant = self.get_dominant_drive()
                level = self.drives[dominant].level
                if level > 0.75:
                    from sare.meta.inner_monologue import get_inner_monologue
                    get_inner_monologue().think(
                        f"{dominant.capitalize()} drive spiking (level={level:.2f}) — "
                        f"recommend: {self.get_behavior_recommendation()}",
                        context="homeostasis",
                        emotion="curious" if dominant == "curiosity" else "neutral",
                    )
            except Exception:
                pass

    def satisfy(self, drive_name: str, amount: float = 0.3):
        """Reduce a specific drive (satisfy it)."""
        if drive_name in self.drives:
            self.drives[drive_name].satisfy(amount)
        else:
            log.debug("HomeostaticSystem: unknown drive '%s'", drive_name)

    # ── Event-driven satisfaction ─────────────────────────────────────────────

    def on_rule_discovered(self):
        """Discovering a new rule satisfies curiosity strongly."""
        self.satisfy("curiosity", 0.6)
        self.satisfy("mastery", 0.2)

    def on_domain_mastered(self):
        """Mastering a domain satisfies the mastery drive strongly."""
        self.satisfy("mastery", 0.8)
        self.satisfy("curiosity", 0.2)

    def on_social_interaction(self):
        """Talking with a human satisfies social and curiosity drives."""
        self.satisfy("social", 0.4)
        self.satisfy("curiosity", 0.15)

    def on_exploration(self):
        """Exploring a new domain or problem satisfies exploration drive."""
        self.satisfy("exploration", 0.3)
        self.satisfy("curiosity", 0.2)

    def on_sleep_cycle(self):
        """Memory consolidation sleep satisfies the consolidation drive."""
        self.satisfy("consolidation", 0.6)

    def on_analogy_found(self):
        """Finding a cross-domain analogy satisfies exploration and curiosity."""
        self.satisfy("exploration", 0.35)
        self.satisfy("curiosity", 0.25)

    def on_problem_solved(self):
        """Solving a problem satisfies mastery (small amount)."""
        self.satisfy("mastery", 0.12)
        self.satisfy("curiosity", 0.05)

    def on_batch_completed(self, solved_count: int, total: int):
        """Called after each learning batch; satisfaction scales with solve rate."""
        rate = solved_count / max(total, 1)
        self.satisfy("mastery", rate * 0.15)
        self.satisfy("curiosity", 0.08)
        if rate > 0.8:
            self.satisfy("exploration", 0.1)

    # ── Behavior recommendation ───────────────────────────────────────────────

    def get_behavior_recommendation(self) -> str:
        """
        Returns the highest-priority recommended behavior based on current drives.

        Returns one of:
          "explore_new_domain"   — curiosity is high
          "deepen_weak_domain"   — mastery drive is high
          "seek_human_input"     — social drive is high
          "consolidate_memory"   — consolidation drive is high
          "generate_analogies"   — exploration drive is high
          "continue_working"     — all drives are satisfied
        """
        for drive_name, min_level, behavior in _BEHAVIOR_MAP:
            if self.drives.get(drive_name, Drive("x", 0)).level >= min_level:
                return behavior
        return "continue_working"

    def get_dominant_drive(self) -> str:
        """Return the name of the drive with highest urgency."""
        return max(self.drives.keys(), key=lambda d: self.drives[d].level)

    def get_search_modulation(self) -> dict:
        """
        Return search parameter deltas based on current drive levels.

        Returns:
          beam_delta:    int   — add to beam_width (+2 if curious, -2 if fatigued)
          budget_delta:  float — add to budget_seconds (+2.0 if confident, -1.0 if fatigued)
          domain_switch: bool  — True if frustration is high (many failures)
          mode:          str   — "explore" | "deepen" | "consolidate" | "normal"
        """
        curiosity = self.drives.get("curiosity", Drive("curiosity", 0.5)).level
        mastery   = self.drives.get("mastery",   Drive("mastery",   0.5)).level
        social    = self.drives.get("social",    Drive("social",    0.5)).level
        consol    = self.drives.get("consolidation", Drive("consolidation", 0.3)).level

        # Fatigue proxy: consolidation urgency is high → system is overloaded
        fatigued = consol > 0.75

        if fatigued:
            return {
                "beam_delta": -2,
                "budget_delta": -1.0,
                "domain_switch": False,
                "mode": "consolidate",
            }
        if curiosity > 0.75:
            return {
                "beam_delta": 2,
                "budget_delta": 0.0,
                "domain_switch": False,
                "mode": "explore",
            }
        if mastery > 0.70:
            return {
                "beam_delta": 0,
                "budget_delta": 2.0,
                "domain_switch": False,
                "mode": "deepen",
            }
        # Frustration: high social + low mastery suggests repeated failure
        if social > 0.80 and mastery < 0.3:
            return {
                "beam_delta": 0,
                "budget_delta": 0.0,
                "domain_switch": True,
                "mode": "explore",
            }
        return {
            "beam_delta": 0,
            "budget_delta": 0.0,
            "domain_switch": False,
            "mode": "normal",
        }

    # ── State reporting ───────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Full drive state for API responses."""
        # Ensure drives are up to date
        now = time.time()
        elapsed = now - self._last_tick
        if elapsed > 30:
            self.tick()

        return {
            "drives": {name: d.to_dict() for name, d in self.drives.items()},
            "behavior_recommendation": self.get_behavior_recommendation(),
            "dominant_drive": self.get_dominant_drive(),
            "last_tick": self._last_tick,
            "tick_count": self._tick_count,
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self):
        try:
            data = {
                "drives": {name: d.to_dict() for name, d in self.drives.items()},
                "last_tick": self._last_tick,
                "tick_count": self._tick_count,
                "saved_at": time.time(),
            }
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            log.warning("HomeostaticSystem save error: %s", e)

    def load(self):
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)

            # Support legacy homeostasis.json format (has "value" instead of "level")
            drives_data = data.get("drives", {})
            for name, dd in drives_data.items():
                if isinstance(dd, dict):
                    # Remap legacy "value" → "level"
                    if "level" not in dd and "value" in dd:
                        dd["level"] = dd["value"]
                    if "name" not in dd:
                        dd["name"] = name
                    # Remap legacy "decay_rate" (negative meant building up → flip sign)
                    if dd.get("decay_rate", 0) < 0:
                        dd["decay_rate"] = abs(dd["decay_rate"])
                    if name in self.drives:
                        self.drives[name] = Drive.from_dict(dd)
                    else:
                        self.drives[name] = Drive.from_dict(dd)

            self._last_tick = data.get("last_tick", time.time())
            self._tick_count = data.get("tick_count", 0)

            log.info("HomeostaticSystem loaded: %d drives", len(self.drives))
        except Exception as e:
            log.warning("HomeostaticSystem load error: %s", e)


# ── Singleton ──────────────────────────────────────────────────────────────────

_HOMEOSTATIC_SYSTEM: Optional[HomeostaticSystem] = None


def get_homeostatic_system() -> HomeostaticSystem:
    global _HOMEOSTATIC_SYSTEM
    if _HOMEOSTATIC_SYSTEM is None:
        _HOMEOSTATIC_SYSTEM = HomeostaticSystem()
    return _HOMEOSTATIC_SYSTEM
