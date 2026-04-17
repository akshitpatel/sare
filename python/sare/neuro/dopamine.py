"""
DopamineSystem — Reward Prediction Error (RPE) Engine
======================================================
Models the brain's dopaminergic reward system.

Biological basis:
  - Dopamine neurons fire on *unexpected* reward (RPE > 0)
  - They pause on *unexpected lack* of reward (RPE < 0)
  - They don't fire when reward is perfectly predicted (RPE = 0)
  - This is how the brain knows WHAT to reinforce

In SARE:
  RPE = actual_reward - predicted_reward

  actual_reward   = weighted value of what just happened (solve, promote, etc.)
  predicted_reward = rolling average expectation from WorldModel

Behavioral effects of dopamine level:
  ┌─────────────────────────────────────────────────────┐
  │ tonic > 0.7 → EXPLORE   (novelty seeking, risk-on) │
  │ tonic 0.3-0.7 → LEARN   (balanced exploit/explore) │
  │ tonic < 0.3 → CONSOLIDATE (deepen mastery, safe)   │
  └─────────────────────────────────────────────────────┘

  Higher dopamine also raises exploration_temperature —
  the system becomes more "creative" and willing to try
  transforms it hasn't used before.

Usage::
    ds = get_dopamine_system()
    rpe = ds.receive_reward("rule_promoted", delta=8.0)
    mode = ds.behavior_mode     # "explore" | "learn" | "consolidate"
    temp = ds.exploration_temperature  # 0.2–1.0

    # Time advance (call every ~60s)
    ds.tick(elapsed_seconds=60.0)
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_MEMORY = Path(__file__).resolve().parents[4] / "data" / "memory"

# ── Reward catalogue ──────────────────────────────────────────────────────────
# Maps event type → intrinsic value (before RPE adjustment)
REWARD_WEIGHTS: Dict[str, float] = {
    # Learning events
    "solve_known":          0.20,   # solve a familiar-type problem
    "solve_novel":          0.55,   # solve a type not seen before
    "solve_domain_first":   0.80,   # first solve in a new domain
    # Knowledge events
    "rule_promoted":        0.60,   # new rule extracted + promoted
    "transform_synthesized":1.10,   # LLM wrote a new Transform class
    "symbol_created":       1.30,   # invented a brand-new symbol/primitive
    "algorithm_invented":   1.50,   # invented a new search algorithm
    # Cross-domain events
    "analogy_found":        0.70,   # cross-domain transfer discovered
    "cross_domain_solve":   0.85,   # applied rule from domain A to domain B
    # Creative events
    "creative_hypothesis":  0.50,   # creativity engine found novel connection
    "world_model_surprise": 0.40,   # something violated worldmodel prediction
    # Negative events
    "stuck":               -0.15,   # failed to solve (same problem repeatedly)
    "rule_rejected":       -0.10,   # causal induction rejected a candidate
    "contradiction":       -0.25,   # new belief contradicts existing knowledge
}

# ── RPE smoothing window (rolling mean of expected rewards) ───────────────────
_RPE_WINDOW = 50


@dataclass
class RewardEvent:
    event_type:  str
    actual:      float      # actual reward received
    predicted:   float      # what we expected
    rpe:         float      # actual - predicted
    domain:      str
    timestamp:   float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "event_type": self.event_type,
            "actual":     round(self.actual, 3),
            "predicted":  round(self.predicted, 3),
            "rpe":        round(self.rpe, 3),
            "domain":     self.domain,
            "timestamp":  self.timestamp,
        }


class DopamineSystem:
    """
    Reward Prediction Error engine that drives SARE's intrinsic motivation.

    Two dopamine pools:
      tonic_level   — slow-moving baseline (0-1). Reflects overall
                       engagement and long-term reward history.
      phasic_burst  — fast spike on surprise (decays within minutes).
                       Captures the "aha!" moment.
    """

    PERSIST_PATH = _MEMORY / "dopamine.json"

    # Tonic homeostatic setpoint (rests here when no reward/punishment)
    SETPOINT = 0.50

    # Learning rates
    LR_POSITIVE = 0.06    # tonic rise on positive RPE
    LR_NEGATIVE = 0.03    # tonic fall on negative RPE (asymmetric — loss aversion)

    # Phasic decay: half-life ~5 minutes
    PHASIC_DECAY = 0.97   # per 60-second tick

    def __init__(self):
        self.tonic_level:  float = self.SETPOINT
        self.phasic_burst: float = 0.0
        self._rpe_history:    deque = deque(maxlen=_RPE_WINDOW)
        self._events:         deque = deque(maxlen=200)
        self._domain_rewards: Dict[str, List[float]] = {}  # domain → recent RPEs
        self._last_tick:  float = time.time()
        self._total_rewards: int = 0
        self._load()

    # ── Core API ─────────────────────────────────────────────────────────────

    def receive_reward(
        self,
        event_type: str,
        domain:     str = "general",
        delta:      float = 0.0,
        context:    Optional[dict] = None,
    ) -> float:
        """
        Process a reward event. Returns the RPE (reward prediction error).

        Parameters
        ----------
        event_type : key from REWARD_WEIGHTS
        domain     : learning domain (for domain-specific dopamine tracking)
        delta      : additional numeric signal (e.g. energy reduction)
        context    : optional metadata

        Returns
        -------
        rpe : float — positive = better than expected, negative = worse
        """
        base    = REWARD_WEIGHTS.get(event_type, 0.0)
        # Scale by energy delta for continuous signals
        actual  = base + math.tanh(delta / 10.0) * abs(base) * 0.5
        predicted = self._predict_expected()

        rpe = actual - predicted

        # Update dopamine levels
        if rpe > 0:
            self.phasic_burst = min(1.0, self.phasic_burst + rpe * 0.4)
            self.tonic_level  = min(1.0, self.tonic_level + rpe * self.LR_POSITIVE)
        elif rpe < 0:
            self.phasic_burst = max(0.0, self.phasic_burst + rpe * 0.2)
            self.tonic_level  = max(0.0, self.tonic_level + rpe * self.LR_NEGATIVE)

        # Record
        self._rpe_history.append(rpe)
        event = RewardEvent(event_type=event_type, actual=actual,
                            predicted=predicted, rpe=rpe, domain=domain)
        self._events.append(event)
        self._domain_rewards.setdefault(domain, []).append(rpe)
        if len(self._domain_rewards[domain]) > 50:
            self._domain_rewards[domain] = self._domain_rewards[domain][-30:]

        self._total_rewards += 1
        if self._total_rewards % 20 == 0:
            self._save()

        log.debug("[Dopamine] %s domain=%s actual=%.2f pred=%.2f RPE=%+.2f tonic=%.2f",
                  event_type, domain, actual, predicted, rpe, self.tonic_level)
        return rpe

    def tick(self, elapsed_seconds: float = 60.0):
        """Advance time: decay phasic burst, drift tonic toward setpoint."""
        ticks = elapsed_seconds / 60.0
        self.phasic_burst *= (self.PHASIC_DECAY ** ticks)
        # Gentle homeostatic pull toward SETPOINT
        drift = (self.SETPOINT - self.tonic_level) * 0.001 * elapsed_seconds
        self.tonic_level = max(0.0, min(1.0, self.tonic_level + drift))
        self._last_tick = time.time()

    # ── Behavioral outputs ────────────────────────────────────────────────────

    @property
    def combined_level(self) -> float:
        """Effective dopamine level (tonic + phasic contribution)."""
        return min(1.0, self.tonic_level + self.phasic_burst * 0.35)

    @property
    def behavior_mode(self) -> str:
        """How should the system behave right now?"""
        lvl = self.combined_level
        if lvl >= 0.72:   return "explore"       # novelty seeking, risky
        elif lvl >= 0.45: return "learn"          # balanced
        elif lvl >= 0.25: return "consolidate"    # deepen known domains
        else:              return "rest"           # minimal output, sleep-mode

    @property
    def exploration_temperature(self) -> float:
        """
        Temperature for stochastic decisions (0.2–1.0).
        High dopamine = high temperature = more creative / exploratory.
        """
        return 0.20 + self.combined_level * 0.80

    @property
    def curiosity_bonus(self) -> float:
        """
        Bonus for novel states in beam search (0.0–0.5).
        Flows into AttentionBeamScorer.gamma.
        """
        return self.combined_level * 0.50

    @property
    def learning_rate_multiplier(self) -> float:
        """
        How fast to update beliefs right now?
        High dopamine → faster updates (more plasticity).
        """
        return 0.5 + self.combined_level * 1.0  # 0.5–1.5

    @property
    def search_temperature(self) -> float:
        """
        Beam search temperature scaled from dopamine level.
        Maps tonic_level [0, 1] → temperature [0.1, 1.0].
        High dopamine = high temperature = more exploratory beam search.
        Used by: experiment_runner effective_beam_width = stage_max * search_temperature
        """
        return 0.10 + self.tonic_level * 0.90

    @property
    def encoding_strength(self) -> float:
        """
        Memory encoding weight based on prediction error magnitude.
        Maps |last_rpe| → encoding weight [0.5, 2.0].
        Surprising events (high |RPE|) are encoded more strongly.
        Used by: memory_manager.store(encoding_strength=...)
        """
        if not self._rpe_history:
            return 1.0
        last_rpe = abs(list(self._rpe_history)[-1]) if self._rpe_history else 0.0
        # Linear scale: rpe=0 → 0.5, rpe=1.0 → 2.0
        return max(0.5, min(2.0, 0.5 + last_rpe * 1.5))

    def domain_engagement(self, domain: str) -> float:
        """
        How 'engaged' (dopaminergically) is the system with this domain?
        Recent positive RPEs = high engagement.
        """
        rpes = self._domain_rewards.get(domain, [])
        if not rpes:
            return self.tonic_level   # baseline
        recent = rpes[-10:]
        return float(max(0.0, min(1.0, self.tonic_level + sum(recent) * 0.1)))

    def most_rewarding_domain(self) -> Optional[str]:
        """Return the domain that produced the highest recent RPE sum."""
        if not self._domain_rewards:
            return None
        return max(
            self._domain_rewards,
            key=lambda d: sum(self._domain_rewards[d][-5:])
        )

    # ── Self-assessment ───────────────────────────────────────────────────────

    def get_state(self) -> dict:
        recent_rpes = list(self._rpe_history)[-10:]
        recent_events = [e.to_dict() for e in list(self._events)[-8:]]
        avg_rpe = sum(recent_rpes) / len(recent_rpes) if recent_rpes else 0.0
        return {
            "tonic_level":            round(self.tonic_level, 3),
            "phasic_burst":           round(self.phasic_burst, 3),
            "combined_level":         round(self.combined_level, 3),
            "behavior_mode":          self.behavior_mode,
            "exploration_temperature":round(self.exploration_temperature, 3),
            "curiosity_bonus":        round(self.curiosity_bonus, 3),
            "learning_rate_mult":     round(self.learning_rate_multiplier, 3),
            "avg_rpe_recent":         round(avg_rpe, 3),
            "total_rewards":          self._total_rewards,
            "domain_engagement":      {
                d: round(self.domain_engagement(d), 3)
                for d in self._domain_rewards
            },
            "most_rewarding_domain":  self.most_rewarding_domain(),
            "recent_events":          recent_events,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _predict_expected(self) -> float:
        """Rolling mean of recent rewards (the brain's 'baseline expectation')."""
        if not self._rpe_history:
            return 0.0
        return sum(self._rpe_history) / len(self._rpe_history)

    def _save(self):
        try:
            _MEMORY.mkdir(parents=True, exist_ok=True)
            self.PERSIST_PATH.write_text(json.dumps({
                "tonic_level":   self.tonic_level,
                "phasic_burst":  self.phasic_burst,
                "total_rewards": self._total_rewards,
                "domain_rewards": {d: v[-20:] for d, v in self._domain_rewards.items()},
                "rpe_history":   list(self._rpe_history)[-50:],
                "saved_at":      time.time(),
            }, indent=2), encoding="utf-8")
        except OSError as e:
            log.debug("[Dopamine] Save error: %s", e)

    def _load(self):
        if not self.PERSIST_PATH.exists():
            return
        try:
            d = json.loads(self.PERSIST_PATH.read_text())
            self.tonic_level   = float(d.get("tonic_level",  self.SETPOINT))
            self.phasic_burst  = float(d.get("phasic_burst", 0.0))
            self._total_rewards = int(d.get("total_rewards", 0))
            for dom, vals in d.get("domain_rewards", {}).items():
                self._domain_rewards[dom] = list(vals)
            for v in d.get("rpe_history", []):
                self._rpe_history.append(float(v))
        except Exception as e:
            log.debug("[Dopamine] Load error: %s", e)

    def save(self):
        self._save()


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: Optional[DopamineSystem] = None

def get_dopamine_system() -> DopamineSystem:
    global _instance
    if _instance is None:
        _instance = DopamineSystem()
    return _instance
