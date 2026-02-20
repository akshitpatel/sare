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
    recent_successes: int = 0     # last 20 attempts
    recent_attempts: int = 0      # last 20 attempts
    confidence_error: float = 0.0 # |predicted_conf - actual_success|, calibration
    last_updated: float = field(default_factory=time.time)

    @property
    def recent_rate(self) -> float:
        if self.recent_attempts == 0:
            return self.solve_rate
        return self.recent_successes / self.recent_attempts

    @property
    def mastery_level(self) -> str:
        r = self.recent_rate
        if r < 0.2:   return "novice"
        elif r < 0.5: return "learning"
        elif r < 0.8: return "competent"
        else:          return "mastered"

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
        if r <= 0.0:   return 0.3   # never tried — some baseline curiosity
        elif r < 0.1:  return 0.2   # probably out of reach currently
        elif r < 0.3:  return 0.6   # hard but maybe reachable
        elif r < 0.7:  return 1.0   # optimal learning zone
        elif r < 0.9:  return 0.5   # getting easier
        else:          return 0.1   # mostly mastered

    def update(self, success: bool, delta: float, steps: int, predicted_confidence: float = 0.5):
        n = self.total_attempts
        alpha = 1.0 / (n + 1) if n < 50 else 0.02  # EMA for older data

        self.solve_rate = (1 - alpha) * self.solve_rate + alpha * float(success)
        if success:
            self.avg_delta = (1 - alpha) * self.avg_delta + alpha * delta
            self.avg_steps = (1 - alpha) * self.avg_steps + alpha * steps

        # Recent window (last 20)
        if self.recent_attempts >= 20:
            # Slide window — approximate by decaying
            self.recent_successes = int(self.recent_successes * 0.9)
            self.recent_attempts  = max(10, int(self.recent_attempts * 0.9))
        self.recent_attempts  += 1
        self.recent_successes += int(success)

        # Calibration error (EMA)
        actual = 1.0 if success else 0.0
        calib_error = abs(predicted_confidence - actual)
        self.confidence_error = (1 - alpha) * self.confidence_error + alpha * calib_error

        self.total_attempts += 1
        self.last_updated = time.time()

    def to_dict(self) -> dict:
        return {
            "domain":            self.domain,
            "solve_rate":        round(self.solve_rate, 3),
            "avg_delta":         round(self.avg_delta, 3),
            "avg_steps":         round(self.avg_steps, 1),
            "total_attempts":    self.total_attempts,
            "recent_successes":  self.recent_successes,
            "recent_attempts":   self.recent_attempts,
            "confidence_error":  round(self.confidence_error, 3),
            "mastery_level":     self.mastery_level,
            "exploration_weight": round(self.exploration_weight, 3),
            "last_updated":      self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DomainCompetence":
        dc = cls(domain=d["domain"])
        dc.solve_rate       = d.get("solve_rate", 0.0)
        dc.avg_delta        = d.get("avg_delta", 0.0)
        dc.avg_steps        = d.get("avg_steps", 0.0)
        dc.total_attempts   = d.get("total_attempts", 0)
        dc.recent_successes = d.get("recent_successes", 0)
        dc.recent_attempts  = d.get("recent_attempts", 0)
        dc.confidence_error  = d.get("confidence_error", 0.0)
        dc.last_updated     = d.get("last_updated", 0.0)
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
            self.avg_delta = (1 - alpha) * self.avg_delta + alpha * delta


class SelfModel:
    """
    SARE-HX's model of its own capabilities.

    Provides:
      - Per-domain competence scores
      - Exploration priority weights (zone of proximal development)
      - Transform utility rankings
      - Confidence calibration error
      - Self-assessment report

    Updated after every solve attempt via observe().
    Queried by ExperimentRunner to bias curriculum generation.
    """

    DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "self_model.json"

    def __init__(self, persist_path: Optional[Path] = None):
        self._path = Path(persist_path or self.DEFAULT_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._domains: Dict[str, DomainCompetence] = {}
        self._transforms: Dict[str, TransformUtility] = {}
        self._total_solves: int = 0
        self._created_at: float = time.time()

    # ── Core API ───────────────────────────────────────────────

    def observe(self, domain: str, success: bool, delta: float,
                steps: int, transforms_used: List[str],
                predicted_confidence: float = 0.5):
        """
        Record the outcome of one solve attempt.
        This is the main update function — call it after every solve.
        """
        # Update domain competence
        if domain not in self._domains:
            self._domains[domain] = DomainCompetence(domain=domain)
        self._domains[domain].update(success, delta, steps, predicted_confidence)

        # Update transform utilities
        for t_name in transforms_used:
            if t_name not in self._transforms:
                self._transforms[t_name] = TransformUtility(name=t_name)
            self._transforms[t_name].record(success, delta)

        self._total_solves += 1

        # Auto-save every 50 observations
        if self._total_solves % 50 == 0:
            self.save()

    def infer_domain(self, expression: str) -> str:
        """Infer the domain from an expression string."""
        expr = expression.lower()
        if any(tok in expr for tok in ("not", "and", "or", "true", "false", "¬", "∧", "∨", "⊤", "⊥")):
            return "logic"
        if any(tok in expr for tok in ("+", "-", "*", "/", "^")):
            return "arithmetic"
        return "general"

    # ── Queries ────────────────────────────────────────────────

    def competence(self, domain: str) -> DomainCompetence:
        """Get competence model for a domain, creating if needed."""
        if domain not in self._domains:
            self._domains[domain] = DomainCompetence(domain=domain)
        return self._domains[domain]

    def curiosity_weights(self) -> Dict[str, float]:
        """
        Returns a probability distribution over domains for curriculum focus.
        Implements zone-of-proximal-development: more weight to domains
        where the system is currently learning (0.3-0.7 solve rate).
        """
        weights = {}
        for domain, dc in self._domains.items():
            weights[domain] = dc.exploration_weight

        # Normalize to sum = 1
        total = sum(weights.values()) or 1.0
        return {d: w / total for d, w in weights.items()}

    def prioritized_domain(self) -> str:
        """Return the domain that most needs attention right now."""
        weights = self.curiosity_weights()
        if not weights:
            return "arithmetic"  # default starting domain
        return max(weights, key=weights.get)

    def top_transforms(self, n: int = 10) -> List[TransformUtility]:
        """Return the most useful transforms by utility score."""
        return sorted(
            self._transforms.values(),
            key=lambda t: t.utility,
            reverse=True,
        )[:n]

    def low_utility_transforms(self, threshold: float = 0.1) -> List[str]:
        """Return transform names with very low utility (candidates for pruning)."""
        return [
            t.name for t in self._transforms.values()
            if t.use_count >= 10 and t.utility < threshold
        ]

    def calibration_error(self) -> float:
        """Global average confidence calibration error."""
        if not self._domains:
            return 0.0
        return sum(dc.confidence_error for dc in self._domains.values()) / len(self._domains)

    # ── Self-assessment ─────────────────────────────────────────

    def self_report(self) -> dict:
        """Full structured self-assessment."""
        return {
            "total_solves":         self._total_solves,
            "domains":              {d: dc.to_dict() for d, dc in self._domains.items()},
            "curiosity_weights":    self.curiosity_weights(),
            "prioritized_domain":   self.prioritized_domain(),
            "calibration_error":    round(self.calibration_error(), 3),
            "top_transforms":       [
                {"name": t.name, "utility": round(t.utility, 3), "uses": t.use_count}
                for t in self.top_transforms(5)
            ],
            "low_utility":          self.low_utility_transforms(),
        }

    # ── Persistence ────────────────────────────────────────────

    def save(self):
        try:
            data = {
                "total_solves": self._total_solves,
                "created_at":   self._created_at,
                "domains":      {d: dc.to_dict() for d, dc in self._domains.items()},
                "transforms":   {
                    n: {"name": t.name, "use_count": t.use_count,
                        "success_count": t.success_count, "avg_delta": t.avg_delta}
                    for n, t in self._transforms.items()
                },
            }
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            log.warning("SelfModel save error: %s", e)

    def load(self):
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            self._total_solves = data.get("total_solves", 0)
            self._created_at   = data.get("created_at", time.time())
            for domain, d in data.get("domains", {}).items():
                self._domains[domain] = DomainCompetence.from_dict(d)
            for name, t in data.get("transforms", {}).items():
                tu = TransformUtility(name=name)
                tu.use_count     = t.get("use_count", 0)
                tu.success_count = t.get("success_count", 0)
                tu.avg_delta     = t.get("avg_delta", 0.0)
                self._transforms[name] = tu
            log.info("SelfModel loaded: %d domains, %d transforms",
                     len(self._domains), len(self._transforms))
        except Exception as e:
            log.warning("SelfModel load error: %s", e)
