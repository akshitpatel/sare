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
from typing import Dict, List

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
    Prefrontal self-referential competency tracker.
    Persists to data/memory/self_model.json.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._domains: Dict[str, DomainCompetence] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Failed to load self_model from %s", self.path)
            return

        domains = data.get("domains", {})
        if isinstance(domains, dict):
            # Backwards compatible format: map domain -> competence dict
            for dom, comp in domains.items():
                if isinstance(comp, dict):
                    comp = dict(comp)
                    comp["domain"] = comp.get("domain", dom)
                    dc = DomainCompetence.from_dict(comp)
                    self._domains[dc.domain] = dc
                else:
                    dc = DomainCompetence(domain=dom)
                    self._domains[dom] = dc
        elif isinstance(domains, list):
            # Format: list of competence dicts
            for comp in domains:
                if isinstance(comp, dict) and "domain" in comp:
                    dc = DomainCompetence.from_dict(comp)
                    self._domains[dc.domain] = dc

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        out = {"domains": {dom: dc.to_dict() for dom, dc in self._domains.items()}}
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)

    def get_domain(self, domain: str) -> DomainCompetence:
        if domain not in self._domains:
            self._domains[domain] = DomainCompetence(domain=domain)
        return self._domains[domain]

    def update(self, domain: str, solved: bool, energy_delta: float, steps: int, predicted_confidence: float = 0.5):
        dc = self.get_domain(domain)
        dc.update(success=bool(solved), delta=float(energy_delta), steps=int(steps), predicted_confidence=float(predicted_confidence))
        self.save()

    def get_confidence(self, domain: str) -> float:
        """
        Confidence proxy for expected solve likelihood in [0,1].
        Uses recent_rate when available; otherwise solve_rate.
        """
        dc = self.get_domain(domain)
        return float(dc.recent_rate)

    def get_weak_domains(
        self,
        min_total_attempts: int = 10,
        weak_solve_rate: float = 0.55,
        max_domains: int = 5,
    ) -> List[str]:
        """
        Return weak domains to focus on next.

        APPROVED CHANGE:
          - Use a combined "weakness" score that prefers domains with low recent success rate
            and insufficient mastery evidence, rather than simply sorting by solve_rate alone.
          - This ensures CurriculumGenerator gets a better ZPD target signal and avoids
            repeatedly selecting "barely tried" domains with unstable estimates.

        The weakness score:
          weakness = (weak_solve_rate - recent_rate) clipped at >=0
                    * evidence_factor
                    * uncertainty_factor

        evidence_factor:
          - close to 1 once recent_attempts >= min_total_attempts
          - rises smoothly with evidence to prevent noisy rankings.

        uncertainty_factor:
          - increases weight when confidence_error is high and/or attempts are low.
        """
        if max_domains <= 0:
            return []

        if not self._domains:
            return []

        scored: List[tuple[str, float]] = []
        for dom, dc in self._domains.items():
            recent = float(dc.recent_rate)
            if recent >= weak_solve_rate:
                continue

            # Evidence factor in [0,1]
            # If min_total_attempts=10, then recent_attempts=0 -> ~0.0, >=10 -> ~1.0
            ra = max(0, int(dc.recent_attempts))
            evidence_factor = 1.0 - math.exp(-ra / max(1.0, float(min_total_attempts)))
            evidence_factor = max(0.0, min(1.0, evidence_factor))

            # Uncertainty factor based on calibration error and attempts.
            # confidence_error is in [0,1] typically; map to [0.5, 1.5]
            ce = float(dc.confidence_error)
            ce = 0.0 if not math.isfinite(ce) else ce
            ce = max(0.0, min(1.0, ce))
            # When attempts are low, increase uncertainty effect
            attempts_norm = max(0.0, min(1.0, float(ra) / max(1.0, float(min_total_attempts))))
            # If attempts_norm is low, we amplify. If high, we reduce.
            uncertainty_factor = (0.75 + 0.75 * ce) * (0.8 + 0.4 * (1.0 - attempts_norm))
            weakness = (weak_solve_rate - recent)
            weakness = max(0.0, weakness) * evidence_factor * uncertainty_factor

            # If a domain has almost no evidence, allow some exploration but low priority.
            # This prevents zeroed scores from blocking new domains.
            if dc.total_attempts <= 0 and recent <= 0.0:
                weakness *= 0.1

            if weakness > 0:
                scored.append((dom, float(weakness)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [dom for dom, _ in scored[:max_domains]]

    def suggest_focus_domain(self) -> str:
        """
        Pick a single best domain for focused study using weak-domain ranking.
        """
        weak = self.get_weak_domains()
        return weak[0] if weak else "general"


# Singleton-like loader for the rest of the system
_SELF_MODEL_SINGLETON = None


def get_self_model() -> SelfModel:
    global _SELF_MODEL_SINGLETON
    if _SELF_MODEL_SINGLETON is None:
        _SELF_MODEL_SINGLETON = SelfModel(path=Path("data/memory/self_model.json"))
    return _SELF_MODEL_SINGLETON