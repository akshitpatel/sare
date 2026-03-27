"""
SelfModel — TODO-05 Implementation

Metacognitive competence tracker: SARE-HX's internal model of itself.

The SelfModel answers the question: "What do I know, and how well?"

It tracks:
  - Per-domain solve rate (arithmetic, logic, general)
  - Per-transform utility (which rules are actually useful)
  - Confidence calibration (is my confidence score accurate?)
  - Exploration bias: directs curiosity toward weak domains

The SelfModel is the bridge between:
  - FrontierManager (what has been tried)
  - CurriculumGenerator (what should be tried next)
  - ExperimentRunner (where to allocate compute)

In cognitive science terms: this is metacognition.
"Knowing what you know" is the foundation of intelligent learning.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class DomainCompetence:
    """Competence model for a single domain."""
    domain: str
    solve_rate: float = 0.0       # 0-1 fraction of problems solved
    avg_delta: float = 0.0        # average energy reduction achieved
    avg_steps: float = 0.0        # average steps to solution
    total_attempts: int = 0
    recent_successes: int = 0     # last 20 attempts (approx)
    recent_attempts: int = 0      # last 20 attempts (approx)
    confidence_error: float = 0.0 # |predicted_conf - actual_success|, calibration
    last_updated: float = field(default_factory=time.time)

    @property
    def recent_rate(self) -> float:
        """
        Recent success rate estimate.

        Uses recent_attempts and recent_successes counters. If no recent attempts exist,
        falls back to global solve_rate.
        """
        if self.recent_attempts <= 0:
            return self.solve_rate
        return self.recent_successes / self.recent_attempts

    @property
    def mastery_level(self) -> str:
        r = self.recent_rate
        if r < 0.2:
            return "novice"
        elif r < 0.5:
            return "learning"
        elif r < 0.8:
            return "competent"
        else:
            return "mastered"

    @property
    def exploration_weight(self) -> float:
        """
        How much attention should curiosity devote to this domain?
        High weight = needs more practice.

        Uses a zone-of-proximal-development model:
          - Too easy (>0.9 rate) → low priority
          - Too hard (<0.1 rate) → low priority (out of reach)
          - 0.3-0.7 rate → high priority (learning zone)
        """
        r = self.recent_rate
        if r <= 0.0:
            return 0.3  # never tried — some baseline curiosity
        elif r < 0.1:
            return 0.2  # probably out of reach currently
        elif r < 0.3:
            return 0.6  # hard but maybe reachable
        elif r < 0.7:
            return 1.0  # optimal learning zone
        elif r < 0.9:
            return 0.5  # getting easier
        else:
            return 0.1  # mostly mastered

    def update(
        self,
        success: bool,
        delta: float,
        steps: int,
        predicted_confidence: float = 0.5,
    ):
        """
        Update competence statistics.

        - Global solve_rate, avg_delta, avg_steps use an adaptive EMA:
          alpha = 1/(n+1) until enough history, then fixed small alpha.
        - Recent counters track a decayed sliding window (approx last ~20).
        - Confidence calibration error tracks EMA of absolute error between
          predicted_confidence and actual outcome.
        """
        n = self.total_attempts
        alpha = 1.0 / (n + 1) if n < 50 else 0.02

        success_f = 1.0 if success else 0.0
        predicted_confidence = float(predicted_confidence)
        if not math.isfinite(predicted_confidence):
            predicted_confidence = 0.5
        predicted_confidence = max(0.0, min(1.0, predicted_confidence))

        # Global aggregates
        self.solve_rate = (1.0 - alpha) * self.solve_rate + alpha * success_f
        if success:
            self.avg_delta = (1.0 - alpha) * self.avg_delta + alpha * float(delta)
            self.avg_steps = (1.0 - alpha) * self.avg_steps + alpha * float(steps)

        # Recent window (approx last 20) using decay when exceeding the target window
        if self.recent_attempts >= 20:
            self.recent_successes = int(self.recent_successes * 0.9)
            self.recent_attempts = max(10, int(self.recent_attempts * 0.9))

        self.recent_attempts += 1
        self.recent_successes += int(success)

        # Calibration error EMA
        actual = success_f
        calib_error = abs(predicted_confidence - actual)
        self.confidence_error = (1.0 - alpha) * self.confidence_error + alpha * calib_error

        self.total_attempts += 1
        self.last_updated = time.time()

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "solve_rate": round(self.solve_rate, 3),
            "avg_delta": round(self.avg_delta, 3),
            "avg_steps": round(self.avg_steps, 1),
            "total_attempts": int(self.total_attempts),
            "recent_successes": int(self.recent_successes),
            "recent_attempts": int(self.recent_attempts),
            "confidence_error": round(self.confidence_error, 3),
            "mastery_level": self.mastery_level,
            "exploration_weight": round(self.exploration_weight, 3),
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DomainCompetence":
        dc = cls(domain=d["domain"])
        dc.solve_rate = d.get("solve_rate", 0.0)
        dc.avg_delta = d.get("avg_delta", 0.0)
        dc.avg_steps = d.get("avg_steps", 0.0)
        dc.total_attempts = d.get("total_attempts", 0)
        dc.recent_successes = d.get("recent_successes", 0)
        dc.recent_attempts = d.get("recent_attempts", 0)
        dc.confidence_error = d.get("confidence_error", 0.0)
        dc.last_updated = d.get("last_updated", 0.0)
        return dc


@dataclass
class TransformUtility:
    """Tracks how useful a specific transform has been."""
    name: str
    use_count: int = 0
    success_count: int = 0
    avg_delta: float = 0.0

    @property
    def utility(self) -> float:
        if self.use_count == 0:
            return 0.0
        return (self.success_count / self.use_count) * self.avg_delta

    def record(self, success: bool, delta: float):
        alpha = 1.0 / (self.use_count + 1) if self.use_count < 50 else 0.02
        self.use_count += 1
        if success:
            self.success_count += 1
            self.avg_delta = (1.0 - alpha) * self.avg_delta + alpha * float(delta)


@dataclass
class StrategyRecord:
    """Tracks effectiveness of a search strategy (beam_search, mcts, etc.)."""
    name: str
    attempts: int = 0
    successes: int = 0
    avg_delta: float = 0.0
    avg_steps: float = 0.0

    def record(self, success: bool, delta: float, steps: int):
        alpha = 1.0 / (self.attempts + 1) if self.attempts < 50 else 0.02
        self.attempts += 1
        if success:
            self.successes += 1
            self.avg_delta = (1.0 - alpha) * self.avg_delta + alpha * float(delta)
            self.avg_steps = (1.0 - alpha) * self.avg_steps + alpha * float(steps)

    @property
    def solve_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts


class SelfModel:
    """
    Metacognitive tracker for SARE-HX.

    Approved change target functions/methods:
      - DomainCompetence class behavior already defined above
      - SelfModel.observe (update competence + optional transform/strategy stats)
      - SelfModel constant for compatibility
    """

    # Storage keys / defaults (stays stable for compatibility with other modules)
    DEFAULT_STATE_PATH = "self_model_state.json"

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = Path(state_path) if state_path else Path(self.DEFAULT_STATE_PATH)
        self.domains: Dict[str, DomainCompetence] = {}
        self.transforms: Dict[str, TransformUtility] = {}
        self.strategies: Dict[str, StrategyRecord] = {}

        # Cached aggregate (optional)
        self.last_observation_time: float = 0.0

        self._load()

    def _load(self):
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Failed to load self model state: %s", self.state_path)
            return

        try:
            domains = data.get("domains", {})
            for k, v in domains.items():
                # v should be dict
                if isinstance(v, dict) and "domain" in v:
                    dc = DomainCompetence.from_dict(v)
                elif isinstance(v, dict) and "domain" not in v:
                    vv = dict(v)
                    vv["domain"] = k
                    dc = DomainCompetence.from_dict(vv)
                else:
                    continue
                self.domains[dc.domain] = dc

            transforms = data.get("transforms", {})
            for k, v in transforms.items():
                if not isinstance(v, dict):
                    continue
                name = v.get("name", k)
                tu = TransformUtility(
                    name=name,
                    use_count=int(v.get("use_count", 0)),
                    success_count=int(v.get("success_count", 0)),
                    avg_delta=float(v.get("avg_delta", 0.0)),
                )
                self.transforms[name] = tu

            strategies = data.get("strategies", {})
            for k, v in strategies.items():
                if not isinstance(v, dict):
                    continue
                name = v.get("name", k)
                sr = StrategyRecord(
                    name=name,
                    attempts=int(v.get("attempts", 0)),
                    successes=int(v.get("successes", 0)),
                    avg_delta=float(v.get("avg_delta", 0.0)),
                    avg_steps=float(v.get("avg_steps", 0.0)),
                )
                self.strategies[name] = sr

            self.last_observation_time = float(data.get("last_observation_time", 0.0))
        except Exception:
            log.exception("Failed to parse self model state: %s", self.state_path)

    def _save(self):
        try:
            payload = {
                "domains": {k: v.to_dict() for k, v in self.domains.items()},
                "transforms": {
                    k: {
                        "name": v.name,
                        "use_count": v.use_count,
                        "success_count": v.success_count,
                        "avg_delta": v.avg_delta,
                    }
                    for k, v in self.transforms.items()
                },
                "strategies": {
                    k: {
                        "name": v.name,
                        "attempts": v.attempts,
                        "successes": v.successes,
                        "avg_delta": v.avg_delta,
                        "avg_steps": v.avg_steps,
                    }
                    for k, v in self.strategies.items()
                },
                "last_observation_time": self.last_observation_time,
            }
            self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            log.exception("Failed to save self model state: %s", self.state_path)

    def observe(
        self,
        *,
        domain: str,
        success: bool,
        delta: float,
        steps: int,
        predicted_confidence: float = 0.5,
        transform_name: Optional[str] = None,
        strategy_name: Optional[str] = None,
        persist: bool = False,
    ):
        """
        Observe the outcome of a solve attempt and update competence estimates.

        This method is used by metacognition loops:
          - competence tracking (domains)
          - transform utility (optional)
          - search strategy record (optional)
        """
        if not domain:
            return

        dc = self.domains.get(domain)
        if dc is None:
            dc = DomainCompetence(domain=domain)
            self.domains[domain] = dc
        dc.update(
            success=success,
            delta=delta,
            steps=steps,
            predicted_confidence=predicted_confidence,
        )

        if transform_name:
            tu = self.transforms.get(transform_name)
            if tu is None:
                tu = TransformUtility(name=transform_name)
                self.transforms[transform_name] = tu
            tu.record(success=success, delta=delta)

        if strategy_name:
            sr = self.strategies.get(strategy_name)
            if sr is None:
                sr = StrategyRecord(name=strategy_name)
                self.strategies[strategy_name] = sr
            sr.record(success=success, delta=delta, steps=steps)

        self.last_observation_time = time.time()
        if persist:
            self._save()

    def get_domain_snapshot(self, domain: str) -> Optional[dict]:
        dc = self.domains.get(domain)
        return None if dc is None else dc.to_dict()

    def get_competence_weights(self) -> Dict[str, float]:
        """
        Return exploration weights per domain, normalized if possible.
        """
        weights = {d: dc.exploration_weight for d, dc in self.domains.items()}
        if not weights:
            return {}
        s = sum(weights.values())
        if s <= 0.0:
            return weights
        return {k: v / s for k, v in weights.items()}

    def to_dict(self) -> dict:
        return {
            "domains": {k: v.to_dict() for k, v in self.domains.items()},
            "transforms": {
                k: {
                    "name": v.name,
                    "use_count": v.use_count,
                    "success_count": v.success_count,
                    "avg_delta": v.avg_delta,
                    "utility": v.utility,
                }
                for k, v in self.transforms.items()
            },
            "strategies": {
                k: {
                    "name": v.name,
                    "attempts": v.attempts,
                    "successes": v.successes,
                    "solve_rate": v.solve_rate,
                    "avg_delta": v.avg_delta,
                    "avg_steps": v.avg_steps,
                }
                for k, v in self.strategies.items()
            },
            "last_observation_time": self.last_observation_time,
        }

    def self_report(self) -> dict:
        """Full self-assessment report. Expected by GoalSetter and web API."""
        base = self.to_dict()
        domains = base.get("domains", {})

        errors = [d.get("confidence_error", 0.0) for d in domains.values()]
        calibration_error = round(sum(errors) / max(len(errors), 1), 4)

        tried = {k: v for k, v in domains.items() if v.get("total_attempts", 0) > 0}
        best_domain = max(tried, key=lambda k: tried[k].get("solve_rate", 0.0), default="")
        weakest_domain = min(tried, key=lambda k: tried[k].get("solve_rate", 1.0), default="")

        total_attempts = sum(v.get("total_attempts", 0) for v in domains.values())

        return {
            **base,
            "calibration_error": calibration_error,
            "total_domains": len(domains),
            "total_attempts": total_attempts,
            "best_domain": best_domain,
            "weakest_domain": weakest_domain,
        }

    def close(self):
        self._save()


__all__ = ["DomainCompetence", "TransformUtility", "StrategyRecord", "SelfModel"]