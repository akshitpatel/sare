"""
MetaCurriculumEngine — S29-1
Meta-level curriculum management: domain discovery, learning-progress measurement,
cross-domain transfer testing, and adaptive weighting.

Core loop (called from learn_cycle):
  1. Compute learning_progress per domain  (2nd derivative of skill EMA)
  2. detect_unsolved_domains()             (stalled or never attempted)
  3. generate_transfer_task(src, dst)      (bridge problem for analogy transfer)
  4. promote / demote domains              (adjust curriculum weights)

Integrates with CurriculumGenerator and ContinuousStreamLearner.
"""
from __future__ import annotations

import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_SKILL_ALPHA     = 0.12   # EMA for skill level
_PROGRESS_ALPHA  = 0.15   # EMA for learning-progress (derivative)
_STALL_THRESHOLD = 0.005  # progress < this → domain stalled
_MIN_WEIGHT      = 0.05
_MAX_WEIGHT      = 0.80
_WEIGHT_STEP     = 0.06

# ── transfer task templates ────────────────────────────────────────────────────

_TRANSFER_TEMPLATES: Dict[Tuple[str, str], List[str]] = {
    ("arithmetic", "algebra"):       ["x + {a} = {b}", "{a} * x = {b}"],
    ("algebra",    "calculus"):      ["d/dx {a}*x^{b}", "d/dx ({a}*x + {b})"],
    ("arithmetic", "physics"):       ["F = {a} * {b}", "v = {a} + {b} * t"],
    ("calculus",   "physics"):       ["integrate {a}*t dt", "d/dt ({a}*t^2)"],
    ("algebra",    "logic"):         ["if x={a} and y={b} then x*y", "x > {a} implies x != 0"],
    ("logic",      "algebra"):       ["(A or B) and not C → solve for C", "A → B, B = {a}"],
    ("trig",       "calculus"):      ["d/dx sin({a}*x)", "d/dx cos(x^{b})"],
    ("arithmetic", "trig"):          ["sin({a} * pi)", "cos({b} * pi / {a})"],
    ("computer_science", "logic"):   ["if T then halt else loop"],
    ("biology",    "chemistry"):     ["cell = H2O + ATP"],
    ("physics",    "chemistry"):     ["E = m * c^2", "PV = n * R * T"],
    ("economics",  "algebra"):       ["revenue = price * quantity"],
}

_ALL_DOMAINS = [
    "arithmetic", "algebra", "calculus", "physics", "logic", "trig",
    # Phase B additions
    "biology", "chemistry", "computer_science", "psychology",
    "economics", "geography", "history", "linguistics",
    "code", "qa", "planning",
]


def _transfer_task(src: str, dst: str) -> Optional[str]:
    key = (src, dst)
    rev = (dst, src)
    templates = _TRANSFER_TEMPLATES.get(key) or _TRANSFER_TEMPLATES.get(rev)
    if not templates:
        return f"{src}_to_{dst}: apply {src} rule in {dst} context"
    tpl = random.choice(templates)
    return tpl.format(a=random.randint(2, 9), b=random.randint(2, 9))


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class DomainStatus:
    domain:          str
    skill:           float = 0.0    # EMA of solve rate
    progress:        float = 0.0    # EMA of skill delta (2nd order)
    weight:          float = 0.2    # curriculum sampling weight
    attempts:        int   = 0
    promotions:      int   = 0
    demotions:       int   = 0
    last_skill:      float = 0.0
    stall_streak:    int   = 0
    transfer_tested: int   = 0

    def update(self, success: bool) -> None:
        val = 1.0 if success else 0.0
        new_skill = _SKILL_ALPHA * val + (1 - _SKILL_ALPHA) * self.skill
        delta     = new_skill - self.skill
        self.progress  = _PROGRESS_ALPHA * delta + (1 - _PROGRESS_ALPHA) * self.progress
        self.last_skill = self.skill
        self.skill      = new_skill
        self.attempts  += 1

        if abs(self.progress) < _STALL_THRESHOLD:
            self.stall_streak += 1
        else:
            self.stall_streak = 0

    @property
    def is_stalled(self) -> bool:
        return self.stall_streak >= 3 and self.attempts >= 5

    @property
    def is_unsolved(self) -> bool:
        return self.skill < 0.35 and self.attempts >= 3

    def to_dict(self) -> dict:
        return {
            "domain":       self.domain,
            "skill":        round(self.skill, 3),
            "progress":     round(self.progress, 4),
            "weight":       round(self.weight, 3),
            "attempts":     self.attempts,
            "is_stalled":   self.is_stalled,
            "is_unsolved":  self.is_unsolved,
            "stall_streak": self.stall_streak,
            "promotions":   self.promotions,
            "demotions":    self.demotions,
        }


@dataclass
class TransferResult:
    src:       str
    dst:       str
    task:      str
    solved:    bool
    skill_before: float
    skill_after:  float
    ts:        float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "src": self.src, "dst": self.dst,
            "task": self.task[:60], "solved": self.solved,
            "skill_before": round(self.skill_before, 3),
            "skill_after":  round(self.skill_after, 3),
        }


# ── MetaCurriculumEngine ──────────────────────────────────────────────────────

class MetaCurriculumEngine:
    """
    Observes per-domain skill EMAs, measures learning progress (derivative),
    adaptively reweights curriculum, generates cross-domain transfer tasks.
    """

    def __init__(self) -> None:
        self._statuses: Dict[str, DomainStatus] = {
            d: DomainStatus(d) for d in _ALL_DOMAINS
        }
        self._curriculum  = None
        self._engine      = None
        self._stream      = None

        self._transfer_results: List[TransferResult] = []
        self._transfer_limit   = 50

        self._total_promotions = 0
        self._total_demotions  = 0
        self._total_transfers  = 0
        self._tick_count       = 0

    # ── wiring ────────────────────────────────────────────────────────────────

    def wire(self, curriculum=None, engine=None, stream=None) -> None:
        self._curriculum = curriculum
        self._engine     = engine
        self._stream     = stream

    # ── observe outcome ───────────────────────────────────────────────────────

    def observe(self, domain: str, success: bool) -> None:
        """Record a solve result for domain. Called from learn_cycle or stream."""
        ds = self._statuses.setdefault(domain, DomainStatus(domain))
        ds.update(success)

    # ── core tick ─────────────────────────────────────────────────────────────

    def tick(self) -> dict:
        """
        Run one meta-curriculum cycle:
          1. Identify stalled/unsolved domains
          2. For each stalled domain, generate and attempt a transfer task
          3. Promote domains with positive progress; demote saturated
          4. Push updated weights to CurriculumGenerator
        Returns summary dict.
        """
        self._tick_count += 1
        unsolved  = self.detect_unsolved_domains()
        stalled   = [d for d, s in self._statuses.items() if s.is_stalled]
        promoted  = []
        demoted   = []
        transfers = []

        for domain in stalled[:2]:   # cap to 2 per tick to avoid overload
            src  = self._best_transfer_source(domain)
            if src:
                result = self.run_transfer_test(src, domain)
                transfers.append(result.to_dict())
                if result.solved:
                    self._promote(domain, reason="transfer success")
                    promoted.append(domain)

        for domain, ds in self._statuses.items():
            if ds.is_unsolved and domain not in promoted:
                self._promote(domain, reason="unsolved → needs practice")
                promoted.append(domain)
            elif ds.skill > 0.85 and ds.progress < _STALL_THRESHOLD:
                self._demote(domain)
                demoted.append(domain)

        self._push_weights()

        return {
            "tick":       self._tick_count,
            "unsolved":   unsolved,
            "stalled":    stalled,
            "promoted":   promoted,
            "demoted":    demoted,
            "transfers":  transfers,
        }

    # ── domain discovery ──────────────────────────────────────────────────────

    def detect_unsolved_domains(self) -> List[str]:
        return [d for d, s in self._statuses.items() if s.is_unsolved]

    def _best_transfer_source(self, target: str) -> Optional[str]:
        """Find domain with highest skill that is not target."""
        candidates = [(d, s.skill) for d, s in self._statuses.items()
                      if d != target and s.skill > 0.5]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]

    # ── transfer testing ──────────────────────────────────────────────────────

    def run_transfer_test(self, src: str, dst: str) -> TransferResult:
        """Generate a bridge task from src→dst and attempt to solve it."""
        task         = _transfer_task(src, dst)
        before       = self._statuses.get(dst, DomainStatus(dst)).skill
        solved       = False

        if self._engine:
            try:
                eng = self._engine
                if hasattr(eng, '_engine'):
                    eng = eng._engine
                if hasattr(eng, 'solve'):
                    result = eng.solve(task)
                    if isinstance(result, dict):
                        solved = bool(result.get("success", result.get("solved", False)))
                    else:
                        solved = result is not None and getattr(result, 'energy', 1.0) < 0.5
            except Exception as exc:
                log.debug(f"MetaCurriculum transfer test failed for {src}->{dst}: {exc}")
                solved = False

        if solved:
            self.observe(dst, True)

        after  = self._statuses.get(dst, DomainStatus(dst)).skill
        ds_dst = self._statuses.setdefault(dst, DomainStatus(dst))
        ds_dst.transfer_tested += 1
        self._total_transfers  += 1

        result = TransferResult(src, dst, task or "", solved, before, after)
        self._transfer_results.append(result)
        if len(self._transfer_results) > self._transfer_limit:
            self._transfer_results.pop(0)
        return result

    # ── weight management ─────────────────────────────────────────────────────

    def _promote(self, domain: str, reason: str = "") -> None:
        ds = self._statuses.setdefault(domain, DomainStatus(domain))
        ds.weight = min(_MAX_WEIGHT, ds.weight + _WEIGHT_STEP)
        ds.promotions      += 1
        self._total_promotions += 1
        log.debug(f"MetaCurriculum promote {domain}: {reason}")

    def _demote(self, domain: str) -> None:
        ds = self._statuses.setdefault(domain, DomainStatus(domain))
        ds.weight = max(_MIN_WEIGHT, ds.weight - _WEIGHT_STEP)
        ds.demotions     += 1
        self._total_demotions += 1
        log.debug(f"MetaCurriculum demote {domain}: saturated")

    def _push_weights(self) -> None:
        """Push updated weights to CurriculumGenerator."""
        if not self._curriculum:
            return
        try:
            total = sum(ds.weight for ds in self._statuses.values())
            for domain, ds in self._statuses.items():
                norm = ds.weight / total if total > 0 else 1 / len(self._statuses)
                if hasattr(self._curriculum, 'set_domain_weight'):
                    self._curriculum.set_domain_weight(domain, norm)
        except Exception as e:
            log.debug(f"MetaCurriculum push_weights: {e}")

    # ── summary ───────────────────────────────────────────────────────────────

    def learning_progress_score(self) -> float:
        """Overall learning progress = avg positive progress across domains."""
        vals = [max(0.0, ds.progress) for ds in self._statuses.values()
                if ds.attempts > 0]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def cross_domain_transfer_rate(self) -> float:
        if not self._transfer_results:
            return 0.0
        return round(
            sum(1 for r in self._transfer_results if r.solved)
            / len(self._transfer_results), 3
        )

    def summary(self) -> dict:
        return {
            "domains":              {d: s.to_dict() for d, s in self._statuses.items()},
            "unsolved_domains":     self.detect_unsolved_domains(),
            "stalled_domains":      [d for d, s in self._statuses.items() if s.is_stalled],
            "learning_progress":    self.learning_progress_score(),
            "transfer_rate":        self.cross_domain_transfer_rate(),
            "total_promotions":     self._total_promotions,
            "total_demotions":      self._total_demotions,
            "total_transfers":      self._total_transfers,
            "tick_count":           self._tick_count,
            "recent_transfers":     [r.to_dict() for r in self._transfer_results[-10:]],
            "curriculum_wired":     self._curriculum is not None,
        }
