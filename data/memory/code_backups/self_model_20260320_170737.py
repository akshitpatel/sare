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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
        dc.last_updated = d.get("last_updated", time.time())
        return dc


class SelfModel:
    """Metacognitive competence tracker for SARE-HX."""

    def __init__(self, data_dir: Path = Path("data/memory")):
        self.data_dir = data_dir
        self.data_file = data_dir / "self_model.json"
        self._domains: Dict[str, DomainCompetence] = {}
        self._load()

    def _load(self):
        """Load self model from persistent storage."""
        if not self.data_file.exists():
            log.info("No self model found, starting fresh")
            return

        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)
                for d in data:
                    self._domains[d["domain"]] = DomainCompetence.from_dict(d)
        except Exception as e:
            log.error(f"Failed to load self model: {e}")

    def save(self):
        """Persist self model to storage."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.data_file, "w") as f:
                json.dump([d.to_dict() for d in self._domains.values()], f, indent=2)
        except Exception as e:
            log.error(f"Failed to save self model: {e}")

    def get_domain(self, domain: str) -> DomainCompetence:
        """Get competence object for a domain, creating if needed."""
        if domain not in self._domains:
            self._domains[domain] = DomainCompetence(domain=domain)
        return self._domains[domain]

    def update(
        self,
        domain: str,
        success: bool,
        delta: float,
        steps: int,
        predicted_confidence: float = 0.5,
    ):
        """Update competence statistics for a domain."""
        dc = self.get_domain(domain)
        dc.update(success, delta, steps, predicted_confidence)
        self.save()

    def get_confidence(self, domain: str) -> float:
        """
        Get calibrated confidence for a domain.

        Returns a confidence score between 0.0 and 1.0 that accounts for:
        - Recent success rate (primary signal)
        - Confidence calibration error (how accurate past predictions were)
        - Domain mastery level (novice vs competent vs mastered)

        The calibration adjusts the raw success rate based on how well
        the system has predicted its own performance in the past.
        """
        dc = self.get_domain(domain)
        recent_rate = dc.recent_rate

        # If we have no data, return neutral confidence
        if dc.total_attempts == 0:
            return 0.5

        # Confidence calibration: if we're usually overconfident, reduce confidence
        # If we're usually underconfident, increase confidence
        calib_adjustment = 0.0
        if dc.confidence_error > 0.0:
            # The sign here is tricky: if error is high, our predictions are bad
            # So we should be more uncertain
            calib_adjustment = -dc.confidence_error * 0.5

        # Mastery level adjustment: as we master a domain, we become more confident
        # even if success rate is high (because we understand why we succeed)
        mastery_adjustment = 0.0
        if dc.mastery_level == "novice":
            mastery_adjustment = -0.2
        elif dc.mastery_level == "learning":
            mastery_adjustment = 0.0
        elif dc.mastery_level == "competent":
            mastery_adjustment = 0.1
        elif dc.mastery_level == "mastered":
            mastery_adjustment = 0.2

        # Combine factors with weights
        confidence = (
            recent_rate * 0.7 +  # Primary signal: recent performance
            calib_adjustment * 0.2 +  # Calibration quality
            mastery_adjustment * 0.1  # Mastery level
        )

        # Clamp to [0, 1] range
        return max(0.0, min(1.0, confidence))

    def get_weak_domains(self, min_attempts: int = 5, threshold: float = 0.3) -> List[str]:
        """
        Get domains that need more practice.

        Returns domains where:
        - We've attempted at least min_attempts problems
        - Recent success rate is below threshold
        - Exploration weight is high (indicating learning zone)
        """
        weak = []
        for domain, dc in self._domains.items():
            if dc.total_attempts < min_attempts:
                continue
            if dc.recent_rate < threshold and dc.exploration_weight > 0.5:
                weak.append(domain)
        return weak

    def get_all_domains(self) -> List[str]:
        """Get all tracked domains."""
        return list(self._domains.keys())

    def get_domain_stats(self, domain: str) -> dict:
        """Get detailed statistics for a domain."""
        dc = self.get_domain(domain)
        return {
            "domain": dc.domain,
            "solve_rate": dc.solve_rate,
            "recent_rate": dc.recent_rate,
            "mastery_level": dc.mastery_level,
            "exploration_weight": dc.exploration_weight,
            "confidence": self.get_confidence(domain),
            "total_attempts": dc.total_attempts,
            "avg_steps": dc.avg_steps,
            "avg_delta": dc.avg_delta,
            "confidence_error": dc.confidence_error,
        }

    def get_summary(self) -> dict:
        """Get summary statistics across all domains."""
        domains = list(self._domains.values())
        if not domains:
            return {
                "total_domains": 0,
                "total_attempts": 0,
                "average_solve_rate": 0.0,
                "average_confidence": 0.5,
            }

        total_attempts = sum(d.total_attempts for d in domains)
        avg_solve_rate = sum(d.solve_rate for d in domains) / len(domains)
        avg_confidence = sum(self.get_confidence(d.domain) for d in domains) / len(domains)

        return {
            "total_domains": len(domains),
            "total_attempts": total_attempts,
            "average_solve_rate": round(avg_solve_rate, 3),
            "average_confidence": round(avg_confidence, 3),
            "weak_domains": self.get_weak_domains(),
        }


# Singleton instance for convenience
_self_model_instance = None


def get_self_model() -> SelfModel:
    """Get the global SelfModel instance."""
    global _self_model_instance
    if _self_model_instance is None:
        _self_model_instance = SelfModel()
    return _self_model_instance
