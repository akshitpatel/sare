"""
AgentMemoryBank — S28-4
Persistent cross-session memory for each agent in AgentSociety.

Per-agent stores:
  episodic_memory  — recent interaction events (capped, time-stamped)
  semantic_memory  — distilled facts/beliefs with confidence
  skill_profile    — per-domain solve-rate EMA
  trust_score      — 0-1 trust based on debate/consensus history
  personality      — {curiosity, stubbornness, collaboration} traits

Persists to data/memory/agent_memories.json.
"""
from __future__ import annotations

import json
import logging
import math
import pathlib
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_SAVE_PATH        = pathlib.Path("data/memory/agent_memories.json")
_EPISODIC_LIMIT   = 30
_SEMANTIC_LIMIT   = 50
_TRUST_DECAY      = 0.995   # per tick
_SKILL_EMA_ALPHA  = 0.1


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class EpisodicEvent:
    event_type:  str
    description: str
    domain:      str  = "general"
    outcome:     str  = "unknown"   # "success" | "failure" | "draw"
    ts:          float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "event_type":  self.event_type,
            "description": self.description[:80],
            "domain":      self.domain,
            "outcome":     self.outcome,
            "age_s":       round(time.time() - self.ts, 1),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodicEvent":
        obj = cls(d["event_type"], d["description"],
                  d.get("domain", "general"), d.get("outcome", "unknown"))
        obj.ts = time.time() - d.get("age_s", 0)
        return obj


@dataclass
class SemanticFact:
    claim:      str
    confidence: float
    domain:     str   = "general"
    source:     str   = "debate"
    ts:         float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"claim": self.claim[:80], "confidence": round(self.confidence, 3),
                "domain": self.domain, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticFact":
        return cls(d["claim"], d["confidence"], d.get("domain", "general"),
                   d.get("source", "debate"))


@dataclass
class AgentProfile:
    agent_id:        str
    skill_profile:   Dict[str, float]        = field(default_factory=dict)
    trust_score:     float                   = 0.5
    episodic_memory: List[EpisodicEvent]     = field(default_factory=list)
    semantic_memory: List[SemanticFact]      = field(default_factory=list)
    personality:     Dict[str, float]        = field(default_factory=lambda: {
        "curiosity":     0.5,
        "stubbornness":  0.5,
        "collaboration": 0.5,
    })
    total_interactions: int = 0
    sessions_seen:      int = 1
    created_at:         float = field(default_factory=time.time)

    # ── episodic ──────────────────────────────────────────────────────────────

    def remember(self, event_type: str, description: str,
                 domain: str = "general", outcome: str = "unknown") -> None:
        evt = EpisodicEvent(event_type, description, domain, outcome)
        self.episodic_memory.append(evt)
        if len(self.episodic_memory) > _EPISODIC_LIMIT:
            self.episodic_memory.pop(0)
        self.total_interactions += 1

        # Update skill profile
        if outcome in ("success", "failure"):
            alpha  = _SKILL_EMA_ALPHA
            cur    = self.skill_profile.get(domain, 0.5)
            val    = 1.0 if outcome == "success" else 0.0
            self.skill_profile[domain] = round(alpha * val + (1 - alpha) * cur, 3)

    # ── semantic ──────────────────────────────────────────────────────────────

    def learn(self, claim: str, confidence: float,
              domain: str = "general", source: str = "debate") -> None:
        for fact in self.semantic_memory:
            if fact.claim == claim:
                fact.confidence = min(1.0, (fact.confidence + confidence) / 1.5)
                return
        self.semantic_memory.append(SemanticFact(claim, confidence, domain, source))
        if len(self.semantic_memory) > _SEMANTIC_LIMIT:
            self.semantic_memory.sort(key=lambda f: f.confidence)
            self.semantic_memory.pop(0)

    def recall(self, query: str, n: int = 5) -> List[SemanticFact]:
        """Simple token-overlap retrieval from semantic memory."""
        q_tokens = set(query.lower().split())
        scored = []
        for fact in self.semantic_memory:
            f_tokens = set(fact.claim.lower().split())
            score    = len(q_tokens & f_tokens) / max(1, len(q_tokens | f_tokens))
            scored.append((score * fact.confidence, fact))
        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:n]]

    # ── trust ─────────────────────────────────────────────────────────────────

    def update_trust(self, outcome: str) -> None:
        delta = {"success": 0.05, "failure": -0.08, "draw": 0.01}.get(outcome, 0)
        self.trust_score = max(0.0, min(1.0, self.trust_score + delta))

    def tick(self) -> None:
        self.trust_score *= _TRUST_DECAY
        # Drift personality from episodic evidence
        if self.episodic_memory:
            recent = self.episodic_memory[-10:]
            successes = sum(1 for e in recent if e.outcome == "success")
            self.personality["curiosity"] = min(1.0,
                len(set(e.domain for e in recent)) / 5.0)
            fails = sum(1 for e in recent if e.outcome == "failure")
            if len(recent) > 0:
                self.personality["stubbornness"] = round(
                    fails / len(recent) * 0.5 + self.personality["stubbornness"] * 0.5, 3)
                self.personality["collaboration"] = round(
                    successes / len(recent) * 0.5 + self.personality["collaboration"] * 0.5, 3)

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "agent_id":           self.agent_id,
            "skill_profile":      self.skill_profile,
            "trust_score":        round(self.trust_score, 3),
            "personality":        {k: round(v, 3) for k, v in self.personality.items()},
            "total_interactions": self.total_interactions,
            "sessions_seen":      self.sessions_seen,
            "episodic_count":     len(self.episodic_memory),
            "semantic_count":     len(self.semantic_memory),
            "top_skills":         sorted(self.skill_profile.items(),
                                         key=lambda x: -x[1])[:3],
            "recent_episodic":    [e.to_dict() for e in self.episodic_memory[-5:]],
            "top_facts":          [f.to_dict() for f in
                                   sorted(self.semantic_memory,
                                          key=lambda f: -f.confidence)[:3]],
        }

    def to_save_dict(self) -> dict:
        d = self.to_dict()
        d["episodic_memory"] = [e.to_dict() for e in self.episodic_memory]
        d["semantic_memory"] = [f.to_dict() for f in self.semantic_memory]
        d["created_at"]      = self.created_at
        return d

    @classmethod
    def from_save_dict(cls, d: dict) -> "AgentProfile":
        p = cls(d["agent_id"])
        p.skill_profile      = d.get("skill_profile", {})
        p.trust_score        = d.get("trust_score", 0.5)
        p.personality        = d.get("personality", p.personality)
        p.total_interactions = d.get("total_interactions", 0)
        p.sessions_seen      = d.get("sessions_seen", 1) + 1
        p.created_at         = d.get("created_at", time.time())
        p.episodic_memory    = [EpisodicEvent.from_dict(e)
                                 for e in d.get("episodic_memory", [])]
        p.semantic_memory    = [SemanticFact.from_dict(f)
                                 for f in d.get("semantic_memory", [])]
        return p


# ── AgentMemoryBank ───────────────────────────────────────────────────────────

class AgentMemoryBank:
    """
    Manages AgentProfile objects for all known agents.
    Auto-saves to JSON; loads on construction.
    Wires into AgentSociety to intercept debate/consensus outcomes.
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, AgentProfile] = {}
        self._agent_society = None
        self._tick_count    = 0
        self._last_save     = 0.0
        self._save_interval = 60.0
        self._load()

    # ── wiring ────────────────────────────────────────────────────────────────

    def wire(self, agent_society=None) -> None:
        self._agent_society = agent_society
        if agent_society:
            self._seed_from_society()

    def _seed_from_society(self) -> None:
        try:
            agents = getattr(self._agent_society, '_agents', {})
            for agent_id in agents:
                self._get_or_create(agent_id)
        except Exception as e:
            log.debug(f"AgentMemoryBank seed: {e}")

    # ── profile access ────────────────────────────────────────────────────────

    def _get_or_create(self, agent_id: str) -> AgentProfile:
        if agent_id not in self._profiles:
            self._profiles[agent_id] = AgentProfile(agent_id)
        return self._profiles[agent_id]

    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        return self._profiles.get(agent_id)

    # ── public interface ──────────────────────────────────────────────────────

    def remember(self, agent_id: str, event_type: str, description: str,
                 domain: str = "general", outcome: str = "unknown") -> None:
        self._get_or_create(agent_id).remember(event_type, description, domain, outcome)

    def learn(self, agent_id: str, claim: str, confidence: float,
              domain: str = "general", source: str = "debate") -> None:
        self._get_or_create(agent_id).learn(claim, confidence, domain, source)

    def recall(self, agent_id: str, query: str, n: int = 5) -> List[SemanticFact]:
        prof = self._profiles.get(agent_id)
        return prof.recall(query, n) if prof else []

    def update_trust(self, agent_id: str, outcome: str) -> None:
        self._get_or_create(agent_id).update_trust(outcome)

    # ── tick ──────────────────────────────────────────────────────────────────

    def tick(self) -> None:
        self._tick_count += 1
        for prof in self._profiles.values():
            prof.tick()
        if time.time() - self._last_save > self._save_interval:
            self._save()

    # ── persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            _SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {aid: prof.to_save_dict()
                    for aid, prof in self._profiles.items()}
            _SAVE_PATH.write_text(json.dumps(data, indent=2))
            self._last_save = time.time()
        except Exception as e:
            log.debug(f"AgentMemoryBank save: {e}")

    def _load(self) -> None:
        try:
            if _SAVE_PATH.exists():
                data = json.loads(_SAVE_PATH.read_text())
                for aid, d in data.items():
                    self._profiles[aid] = AgentProfile.from_save_dict(d)
                log.debug(f"AgentMemoryBank loaded {len(self._profiles)} agents")
        except Exception as e:
            log.debug(f"AgentMemoryBank load: {e}")

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        profiles = {aid: p.to_dict() for aid, p in self._profiles.items()}
        return {
            "n_agents":     len(self._profiles),
            "tick_count":   self._tick_count,
            "save_path":    str(_SAVE_PATH),
            "agents":       profiles,
            "society_wired": self._agent_society is not None,
        }
