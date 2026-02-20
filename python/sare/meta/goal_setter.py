"""
GoalSetter — Phase 8: Metacognition
====================================
Manages high-level autonomous objectives for SARE-HX.

Responsibilities:
  - Maintain a prioritised goal stack
  - Auto-generate sub-goals from SelfModel's weak domains
  - Retire goals once competence reaches the mastered threshold
  - Expose suggest_next_goal() for the UI / ExperimentRunner
  - Persist state to data/memory/goals.json
"""
from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GOALS_PATH = Path("data/memory/goals.json")


class GoalStatus(str, Enum):
    ACTIVE   = "active"
    ACHIEVED = "achieved"
    PAUSED   = "paused"


class GoalType(str, Enum):
    DOMAIN_MASTERY   = "domain_mastery"       # Reach N% solve rate on a domain
    RULE_DISCOVERY   = "rule_discovery"        # Learn k new rules in a domain
    CALIBRATION      = "calibration"           # Reduce calibration error below ε
    FRONTIER_SHRINK  = "frontier_shrink"       # Reduce unsolved count by N
    CUSTOM           = "custom"


@dataclass
class Goal:
    id:          str
    type:        GoalType
    description: str
    domain:      Optional[str]
    target:      float                    # e.g. 0.8 = 80% solve rate
    current:     float    = 0.0
    status:      GoalStatus = GoalStatus.ACTIVE
    priority:    int       = 5           # 1 (highest) – 10 (lowest)
    created_at:  float     = field(default_factory=time.time)
    updated_at:  float     = field(default_factory=time.time)
    achieved_at: Optional[float] = None

    # ── progress ──────────────────────────────────────────────────────────
    @property
    def progress(self) -> float:
        """0.0 – 1.0 fraction towards target."""
        if self.target <= 0:
            return 0.0
        return min(self.current / self.target, 1.0)

    def update_progress(self, current: float) -> bool:
        """Returns True if goal was just achieved."""
        self.current   = current
        self.updated_at = time.time()
        if self.status == GoalStatus.ACTIVE and self.current >= self.target:
            self.status      = GoalStatus.ACHIEVED
            self.achieved_at = time.time()
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"]   = self.status.value
        d["type"]     = self.type.value
        d["progress"] = self.progress
        return d


class GoalSetter:
    """
    Autonomous goal management for self-directed learning.

    Usage::

        gs = GoalSetter()
        gs.load()
        gs.refresh_from_self_model(self_model.self_report())
        next_goal = gs.suggest_next_goal()
        print(next_goal)
    """

    # Thresholds
    DOMAIN_MASTERY_TARGET = 0.80   # 80% solve rate = mastered
    CALIBRATION_TARGET    = 0.10   # 10% calibration error
    MIN_RULES_PER_DOMAIN  = 5      # rule discovery goal
    FRONTIER_SHRINK_BY    = 10     # # unsolved problems to eliminate

    def __init__(self) -> None:
        self._goals:      Dict[str, Goal] = {}
        self._goal_count: int = 0

    # ── Goal creation helpers ──────────────────────────────────────────────
    def _new_id(self) -> str:
        self._goal_count += 1
        return f"g{self._goal_count:04d}"

    def add_goal(
        self,
        type:        GoalType,
        description: str,
        target:      float,
        domain:      Optional[str] = None,
        priority:    int = 5,
    ) -> Goal:
        goal = Goal(
            id=self._new_id(),
            type=type,
            description=description,
            domain=domain,
            target=target,
            priority=priority,
        )
        self._goals[goal.id] = goal
        logger.info("[GoalSetter] New goal: %s (%s)", goal.id, description)
        return goal

    # ── Auto-generation from SelfModel report ─────────────────────────────
    def refresh_from_self_model(self, report: Dict[str, Any]) -> List[Goal]:
        """
        Inspect a SelfModel self_report() and generate goals for weak domains.
        Returns list of newly created goals.
        """
        domains   = report.get("domains", {})
        calib_err = report.get("calibration_error", 0.0) or 0.0
        new_goals: List[Goal] = []

        # -- Domain mastery goals -----------------------------------------
        for domain, info in domains.items():
            solve_rate = info.get("solve_rate", 0.0) or 0.0
            mastery    = info.get("mastery_level", "novice")
            if mastery in ("novice", "learning"):
                key = f"mastery:{domain}"
                if not self._goal_exists(key):
                    g = self.add_goal(
                        type=GoalType.DOMAIN_MASTERY,
                        description=f"Achieve ≥80% solve rate on domain '{domain}' (currently {solve_rate*100:.0f}%)",
                        target=self.DOMAIN_MASTERY_TARGET,
                        domain=domain,
                        priority=3 if mastery == "novice" else 5,
                    )
                    g.current = solve_rate
                    g.id = key          # use stable key for dedup
                    self._goals[key] = g
                    new_goals.append(g)
                else:
                    # Update progress
                    self._goals[key].update_progress(solve_rate)

        # -- Calibration goal ---------------------------------------------
        if calib_err > self.CALIBRATION_TARGET:
            if not self._goal_exists("calibration"):
                g = self.add_goal(
                    type=GoalType.CALIBRATION,
                    description=f"Reduce calibration error to <10% (currently {calib_err*100:.1f}%)",
                    target=self.CALIBRATION_TARGET,
                    priority=4,
                )
                g.id = "calibration"
                # Invert: current progress is how far below threshold we are
                g.current = max(0.0, self.CALIBRATION_TARGET * 2 - calib_err)
                g.target  = self.CALIBRATION_TARGET * 2
                self._goals["calibration"] = g
                new_goals.append(g)
            else:
                # Lower calib_err = better → remap progress
                g = self._goals["calibration"]
                g.update_progress(max(0.0, g.target - calib_err))

        return new_goals

    def _goal_exists(self, goal_id: str) -> bool:
        return goal_id in self._goals and self._goals[goal_id].status == GoalStatus.ACTIVE

    # ── Goal selection ─────────────────────────────────────────────────────
    def suggest_next_goal(self) -> Optional[Goal]:
        """
        Return the highest-priority active goal that hasn't been achieved.
        Tie-break: lowest progress (most work needed).
        """
        candidates = [
            g for g in self._goals.values()
            if g.status == GoalStatus.ACTIVE
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda g: (g.priority, g.progress))

    def active_goals(self) -> List[Goal]:
        return sorted(
            [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE],
            key=lambda g: (g.priority, -g.progress),
        )

    def achieved_goals(self) -> List[Goal]:
        return [g for g in self._goals.values() if g.status == GoalStatus.ACHIEVED]

    # ── Reporting ──────────────────────────────────────────────────────────
    def report(self) -> Dict[str, Any]:
        active   = self.active_goals()
        achieved = self.achieved_goals()
        next_g   = self.suggest_next_goal()
        return {
            "active_count":   len(active),
            "achieved_count": len(achieved),
            "next_goal":      next_g.to_dict() if next_g else None,
            "active_goals":   [g.to_dict() for g in active],
            "achieved_goals": [g.to_dict() for g in achieved[-10:]],  # last 10
        }

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self) -> None:
        GOALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "goal_count": self._goal_count,
            "goals": {k: v.to_dict() for k, v in self._goals.items()},
        }
        with open(GOALS_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        logger.debug("[GoalSetter] Saved %d goals", len(self._goals))

    def load(self) -> None:
        if not GOALS_PATH.exists():
            return
        try:
            with open(GOALS_PATH) as f:
                payload = json.load(f)
            self._goal_count = payload.get("goal_count", 0)
            self._goals = {}
            for k, d in payload.get("goals", {}).items():
                g = Goal(
                    id=d["id"],
                    type=GoalType(d["type"]),
                    description=d["description"],
                    domain=d.get("domain"),
                    target=d["target"],
                    current=d.get("current", 0.0),
                    status=GoalStatus(d.get("status", "active")),
                    priority=d.get("priority", 5),
                    created_at=d.get("created_at", time.time()),
                    updated_at=d.get("updated_at", time.time()),
                    achieved_at=d.get("achieved_at"),
                )
                self._goals[k] = g
            logger.info("[GoalSetter] Restored %d goals", len(self._goals))
        except Exception as e:
            logger.warning("[GoalSetter] Load failed: %s", e)
