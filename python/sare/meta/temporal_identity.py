"""
TemporalIdentity — S26-6
Persistent self across sessions: accumulates a compressed identity vector
(domain strengths, strategy history, personality traits, key discoveries)
that loads at boot and biases all future decisions.
"""
from __future__ import annotations
import json
import os
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

log = logging.getLogger(__name__)

_SAVE_PATH = os.path.join(
    os.path.dirname(__file__), "../../../data/memory/temporal_identity.json"
)


@dataclass
class PersonalityTrait:
    name:     str
    value:    float   # 0–1
    evidence: str

    def to_dict(self) -> dict:
        return {"name": self.name, "value": round(self.value, 3),
                "evidence": self.evidence}


@dataclass
class KeyDiscovery:
    session:    int
    description: str
    domain:     str
    timestamp:  float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"session": self.session, "description": self.description[:80],
                "domain": self.domain, "timestamp": self.timestamp}


class TemporalIdentity:
    """
    Persists a rich identity record across sessions.
    Loads automatically at init; saves on update() and periodically.
    """

    def __init__(self) -> None:
        self._session_count        = 0
        self._total_solves         = 0
        self._domain_strengths:    Dict[str, float]        = {}
        self._best_strategy_history: List[Dict]            = []
        self._key_discoveries:     List[KeyDiscovery]      = []
        self._personality:         Dict[str, PersonalityTrait] = {}
        self._birth_timestamp      = time.time()
        self._last_save            = 0.0
        self._tick_count           = 0

        self._load()

    # ── persistence ───────────────────────────────────────────────────────────
    def _load(self) -> None:
        path = os.path.abspath(_SAVE_PATH)
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._session_count      = data.get("session_count", 0)
            self._total_solves       = data.get("total_solves", 0)
            self._domain_strengths   = data.get("domain_strengths", {})
            self._best_strategy_history = data.get("best_strategy_history", [])
            self._birth_timestamp    = data.get("birth_timestamp", self._birth_timestamp)
            for d in data.get("key_discoveries", []):
                self._key_discoveries.append(
                    KeyDiscovery(d["session"], d["description"],
                                 d.get("domain", "general"), d.get("timestamp", 0)))
            for name, pt in data.get("personality", {}).items():
                self._personality[name] = PersonalityTrait(
                    name, pt.get("value", 0.5), pt.get("evidence", ""))
            log.info(f"[TemporalIdentity] loaded: session {self._session_count}, "
                     f"{self._total_solves} total solves")
        except Exception as e:
            log.debug(f"TemporalIdentity load: {e}")

    def _save(self) -> None:
        path = os.path.abspath(_SAVE_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            data = {
                "session_count":         self._session_count,
                "total_solves":          self._total_solves,
                "domain_strengths":      self._domain_strengths,
                "best_strategy_history": self._best_strategy_history[-20:],
                "birth_timestamp":       self._birth_timestamp,
                "key_discoveries":       [d.to_dict() for d in
                                          self._key_discoveries[-50:]],
                "personality":           {n: p.to_dict()
                                          for n, p in self._personality.items()},
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self._last_save = time.time()
        except Exception as e:
            log.debug(f"TemporalIdentity save: {e}")

    # ── update ────────────────────────────────────────────────────────────────
    def update(self, session_stats: dict) -> None:
        """Called at the end of each learn session with Brain stats."""
        self._session_count += 1
        solves = session_stats.get("solves_this_session", 0)
        self._total_solves += solves

        # Update domain strengths (EMA)
        domain_rates = session_stats.get("domain_rates", {})
        for domain, rate in domain_rates.items():
            prev = self._domain_strengths.get(domain, 0.5)
            self._domain_strengths[domain] = 0.8 * prev + 0.2 * rate

        # Strategy history
        best_strategy = session_stats.get("best_strategy")
        if best_strategy:
            self._best_strategy_history.append({
                "session": self._session_count,
                "strategy": best_strategy,
                "timestamp": time.time(),
            })

        # Key discoveries
        discoveries = session_stats.get("new_discoveries", [])
        for desc in discoveries[:3]:
            domain = session_stats.get("primary_domain", "general")
            self._key_discoveries.append(
                KeyDiscovery(self._session_count, str(desc), domain))

        # Recompute personality
        self._compute_personality(session_stats)
        self._save()

    def tick(self) -> None:
        """Called every N cycles; auto-saves every 60 s."""
        self._tick_count += 1
        if time.time() - self._last_save > 60:
            self._save()

    # ── personality computation ───────────────────────────────────────────────
    def _compute_personality(self, stats: dict) -> None:
        solves  = stats.get("solves_this_session", 0)
        fails   = stats.get("fails_this_session", 0)
        rate    = solves / max(solves + fails, 1)

        # Persistence — keeps trying despite failure
        pers_val = min(1.0, (fails / max(solves + fails, 1)) * 2)
        self._personality["persistence"] = PersonalityTrait(
            "persistence", pers_val,
            f"session {self._session_count}: {fails} fails out of {solves+fails}")

        # Curiosity — breadth of domains explored
        n_domains = len(stats.get("domain_rates", {}))
        curiosity_val = min(1.0, n_domains / 8.0)
        self._personality["curiosity"] = PersonalityTrait(
            "curiosity", curiosity_val,
            f"{n_domains} domains in session {self._session_count}")

        # Mastery — depth in best domain
        dom_rates = stats.get("domain_rates", {})
        best_rate = max(dom_rates.values(), default=0.0)
        self._personality["mastery"] = PersonalityTrait(
            "mastery", best_rate,
            f"best domain rate {best_rate:.0%}")

        # Confidence — overall solve rate
        self._personality["confidence"] = PersonalityTrait(
            "confidence", rate,
            f"overall rate {rate:.0%} this session")

    # ── context API ───────────────────────────────────────────────────────────
    def get_identity_context(self) -> dict:
        """Returns a context dict the Brain can inject into decisions."""
        strongest = sorted(self._domain_strengths.items(),
                           key=lambda x: x[1], reverse=True)[:3]
        weakest   = sorted(self._domain_strengths.items(),
                           key=lambda x: x[1])[:3]
        recent_strategy = (self._best_strategy_history[-1]["strategy"]
                           if self._best_strategy_history else None)
        return {
            "session_count":   self._session_count,
            "total_solves":    self._total_solves,
            "strongest_domains": [d for d, _ in strongest],
            "weakest_domains":   [d for d, _ in weakest],
            "preferred_strategy": recent_strategy,
            "personality":     {n: round(p.value, 2)
                                 for n, p in self._personality.items()},
        }

    # ── summary ───────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        age_days = (time.time() - self._birth_timestamp) / 86400
        recent_discoveries = [d.to_dict() for d in self._key_discoveries[-5:]]
        recent_strategies  = self._best_strategy_history[-5:]
        return {
            "session_count":     self._session_count,
            "total_solves":      self._total_solves,
            "age_days":          round(age_days, 2),
            "tick_count":        self._tick_count,
            "domain_strengths":  {d: round(v, 3)
                                   for d, v in sorted(
                                       self._domain_strengths.items(),
                                       key=lambda x: -x[1])},
            "personality":       {n: p.to_dict()
                                   for n, p in self._personality.items()},
            "best_strategy_history": recent_strategies,
            "recent_discoveries": recent_discoveries,
            "identity_context":  self.get_identity_context(),
        }
