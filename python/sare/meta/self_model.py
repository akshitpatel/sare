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
        dc.last_updated = d.get("last_updated", time.time())
        return dc


class SelfModel:
    """
    Metacognitive model of the system's own competence.

    This is the system's "sense of self" in terms of what it knows.
    It is used by:
      - CurriculumGenerator to pick domains to study
      - GoalSetter to set mastery goals
      - HomeostaticSystem to adjust curiosity drive
      - ExperimentRunner to allocate beam width
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/memory")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / "self_model.json"

        self.domains: Dict[str, DomainCompetence] = {}
        self.last_save = time.time()
        self.load()

    def load(self) -> None:
        """Load from JSON file."""
        if not self.file_path.exists():
            log.info("No self_model.json found; starting fresh.")
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            domains = data.get("domains", {})
            for domain_str, d in domains.items():
                dc = DomainCompetence.from_dict(d)
                self.domains[domain_str] = dc
            log.info(f"Loaded self-model with {len(self.domains)} domains.")
        except Exception as e:
            log.warning(f"Failed to load self_model.json: {e}")

    def save(self) -> None:
        """Save to JSON file."""
        try:
            data = {
                "domains": {d.domain: d.to_dict() for d in self.domains.values()},
                "last_save": time.time(),
            }
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.last_save = time.time()
        except Exception as e:
            log.warning(f"Failed to save self_model.json: {e}")

    def get_domain(self, domain: str) -> DomainCompetence:
        """Get or create a DomainCompetence for the given domain."""
        if domain not in self.domains:
            self.domains[domain] = DomainCompetence(domain=domain)
        return self.domains[domain]

    def update(
        self,
        domain: str,
        solved: bool,
        energy_delta: float,
        steps: int,
        predicted_confidence: float = 0.5,
    ) -> None:
        """
        Update the self-model after a problem attempt.

        Args:
            domain: The domain of the problem (e.g., "arithmetic", "logic").
            solved: Whether the problem was solved.
            energy_delta: Energy reduction achieved (positive = improvement).
            steps: Number of steps taken in the proof.
            predicted_confidence: The system's confidence before solving.
        """
        dc = self.get_domain(domain)
        dc.update(solved, energy_delta, steps, predicted_confidence)

        # Every 25 attempts, reflect on weak domains via LLM
        if dc.total_attempts % 25 == 0:
            import threading
            threading.Thread(
                target=self.reflect_on_weak_domain,
                args=(domain,),
                daemon=True,
            ).start()

        # Auto-save every 10 updates or after 60 seconds
        if dc.total_attempts % 10 == 0 or time.time() - self.last_save > 60:
            self.save()

    def get_confidence(self, domain: str) -> float:
        """
        Return predicted solve rate for a domain.

        This is the system's confidence that it can solve a random problem
        from this domain. It is used by the GoalSetter and CurriculumGenerator
        to decide what to study next.

        Returns:
            Float between 0 and 1.
        """
        dc = self.get_domain(domain)
        return dc.recent_rate

    def predicted_success(self, domain: str) -> float:
        """
        Alias for get_confidence, for compatibility with external callers.
        """
        return self.get_confidence(domain)

    def get_weak_domains(self, min_attempts: int = 5, threshold: float = 0.3) -> List[str]:
        """
        Return domains where the system is weak (low solve rate).

        Args:
            min_attempts: Minimum number of attempts to consider a domain.
            threshold: Solve rate below which a domain is considered weak.

        Returns:
            List of domain names sorted by exploration weight (highest first).
        """
        candidates = []
        for dc in self.domains.values():
            if dc.total_attempts < min_attempts:
                continue
            if dc.recent_rate < threshold:
                candidates.append((dc.exploration_weight, dc.domain))
        # Sort by exploration weight descending (highest priority first)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [domain for _, domain in candidates]

    def suggest_focus_domain(self) -> str:
        """
        Suggest which domain to focus on next.

        Uses exploration_weight to pick the domain that is in the optimal
        learning zone (ZPD). If no domain has enough attempts, returns "general".
        """
        if not self.domains:
            return "general"

        # Build list of (weight, domain) for domains with at least 2 attempts
        weighted = []
        for dc in self.domains.values():
            if dc.total_attempts >= 2:
                weighted.append((dc.exploration_weight, dc.domain))

        if not weighted:
            # Pick the domain with the fewest attempts
            return min(self.domains.values(), key=lambda d: d.total_attempts).domain

        # Pick the highest weight
        weighted.sort(key=lambda x: x[0], reverse=True)
        return weighted[0][1]

    def get_all_domains(self) -> List[str]:
        """Return list of all known domains."""
        return list(self.domains.keys())

    def get_mastered_domains(self, threshold: float = 0.95, min_attempts: int = 10) -> List[str]:
        """Return domains where solve_rate >= threshold (mastered — ready for harder problems)."""
        return [
            d for d, dc in self.domains.items()
            if dc.recent_rate >= threshold and dc.total_attempts >= min_attempts
        ]

    def get_domain_stats(self, domain: str) -> Optional[Dict]:
        """Return full statistics for a domain."""
        dc = self.domains.get(domain)
        if dc is None:
            return None
        return dc.to_dict()

    def get_overall_stats(self) -> Dict:
        """Return aggregated statistics across all domains."""
        total_attempts = sum(dc.total_attempts for dc in self.domains.values())
        if total_attempts == 0:
            return {"total_attempts": 0, "avg_solve_rate": 0.0}

        weighted_sum = sum(dc.solve_rate * dc.total_attempts for dc in self.domains.values())
        avg_solve_rate = weighted_sum / total_attempts

        return {
            "total_attempts": total_attempts,
            "avg_solve_rate": round(avg_solve_rate, 3),
            "domain_count": len(self.domains),
        }

    def reflect_on_weak_domain(self, domain: str) -> Optional[str]:
        """Call LLM to reason about why we're stuck and suggest strategies.

        Returns the reflection text, or None if LLM unavailable / domain not weak.
        Only fires when: solve_rate < 0.3 AND total_attempts >= 15.
        """
        dc = self.domains.get(domain)
        if dc is None or dc.total_attempts < 15 or dc.recent_rate >= 0.3:
            return None
        try:
            from sare.interface.llm_bridge import _call_llm
            prompt = (
                f"I am an AI reasoning system. Here are my performance stats for the '{domain}' domain:\n"
                f"  - Solve rate: {dc.recent_rate:.1%}\n"
                f"  - Average steps to solution: {dc.avg_steps:.1f}\n"
                f"  - Confidence calibration error: {dc.confidence_error:.2f} (0=perfect)\n"
                f"  - Total attempts: {dc.total_attempts}\n"
                f"  - Mastery level: {dc.mastery_level}\n\n"
                f"Suggest 2-3 concrete learning strategies I should try to improve. "
                f"Be specific to {domain} math. Reply in 3 sentences max."
            )
            reflection = _call_llm(prompt).strip()
            if reflection:
                log.info("[SelfModel] LLM reflection for %s: %s", domain, reflection[:80])
            return reflection or None
        except Exception as exc:
            log.debug("SelfModel LLM reflection failed: %s", exc)
            return None

    def to_dict(self) -> Dict:
        """Return a JSON-serializable representation."""
        return {
            "domains": {d.domain: d.to_dict() for d in self.domains.values()},
            "overall": self.get_overall_stats(),
        }

    def self_report(self) -> Dict:
        """Alias for to_dict() — called by web.py /api/self endpoint."""
        return self.to_dict()


# Singleton pattern
_self_model_instance: Optional[SelfModel] = None


def get_self_model(data_dir: Optional[Path] = None) -> SelfModel:
    """Return the singleton SelfModel instance."""
    global _self_model_instance
    if _self_model_instance is None:
        _self_model_instance = SelfModel(data_dir)
    return _self_model_instance