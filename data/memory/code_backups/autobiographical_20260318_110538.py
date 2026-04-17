"""
Autobiographical Memory — SARE-HX's learning history as a narrative.

Tracks the system's learning history like a student's academic journal:
significant events, emotional valence, domain mastery trajectory, and
the emerging narrative of "who I am as a learner".

Integrates with:
  - ExperimentRunner (records solve events)
  - CurriculumGenerator (influences next domain selection)
  - IdentityManager (formative experiences shape traits)
  - HomeostaticSystem (emotional state feeds drive levels)

Data stored in: data/memory/autobiographical.json
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).resolve().parents[3] / "data" / "memory"
AUTO_PATH = MEMORY_DIR / "autobiographical.json"

# Importance thresholds
_HIGH_IMPORTANCE = 0.7
_STUCK_THRESHOLD = 500   # episodes without improvement = "stuck"


@dataclass
class LearningEpisode:
    """One memorable event in the system's learning history."""
    episode_id: str
    timestamp: float
    event_type: str      # "rule_discovered", "domain_mastered", "analogy_found",
                         # "human_taught", "stuck_period", "breakthrough", "rule_applied",
                         # "social_interaction", "milestone"
    domain: str
    description: str
    importance: float    # 0.0–1.0
    related_rules: List[str]
    emotional_valence: float  # -1.0 (frustrating) to 1.0 (exciting)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "domain": self.domain,
            "description": self.description,
            "importance": round(self.importance, 4),
            "related_rules": self.related_rules,
            "emotional_valence": round(self.emotional_valence, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LearningEpisode":
        return cls(
            episode_id=d.get("episode_id", str(uuid.uuid4())[:8]),
            timestamp=d.get("timestamp", time.time()),
            event_type=d.get("event_type", "milestone"),
            domain=d.get("domain", "general"),
            description=d.get("description", ""),
            importance=d.get("importance", 0.5),
            related_rules=d.get("related_rules", []),
            emotional_valence=d.get("emotional_valence", 0.0),
        )


# Emotional valence by event type
_VALENCE_MAP = {
    "domain_mastered":   0.9,
    "breakthrough":      0.95,
    "rule_discovered":   0.7,
    "analogy_found":     0.7,
    "human_taught":      0.6,
    "rule_applied":      0.3,
    "social_interaction": 0.5,
    "milestone":         0.5,
    "stuck_period":      -0.4,
}

# Importance by event type (base)
_IMPORTANCE_MAP = {
    "domain_mastered":   0.9,
    "breakthrough":      0.9,
    "rule_discovered":   0.7,
    "analogy_found":     0.6,
    "human_taught":      0.6,
    "milestone":         0.5,
    "social_interaction": 0.5,
    "rule_applied":      0.2,
    "stuck_period":      0.4,
}


class AutobiographicalMemory:
    """
    SARE-HX's long-term narrative memory of its own learning journey.

    Answers questions like:
      - What were my most important learning moments?
      - Which domains have I struggled with?
      - What was my emotional journey through learning?
      - What should I explore next based on my history?
    """

    DEFAULT_PATH = AUTO_PATH

    def __init__(self, persist_path: Optional[Path] = None):
        self._path = Path(persist_path or self.DEFAULT_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._episodes: List[LearningEpisode] = []
        self._domain_trajectory: Dict[str, List[Tuple[float, float]]] = {}
        # domain → [(timestamp, mastery_level), ...]

        self.load()

    # ── Recording ──────────────────────────────────────────────────────────────

    def record(
        self,
        event_type: str,
        domain: str,
        description: str,
        related_rules: Optional[List[str]] = None,
        importance: Optional[float] = None,
        emotional_valence: Optional[float] = None,
    ):
        """Record a new learning event."""
        if importance is None:
            importance = _IMPORTANCE_MAP.get(event_type, 0.4)
        if emotional_valence is None:
            emotional_valence = _VALENCE_MAP.get(event_type, 0.0)

        ep = LearningEpisode(
            episode_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            event_type=event_type,
            domain=domain,
            description=description,
            importance=importance,
            related_rules=related_rules or [],
            emotional_valence=emotional_valence,
        )
        self._episodes.append(ep)

        # Auto-save every 20 episodes
        if len(self._episodes) % 20 == 0:
            self.save()

        # Update identity traits
        try:
            from sare.memory.identity import get_identity_manager
            im = get_identity_manager()
            success = emotional_valence > 0
            im.update_from_behavior(event_type, domain, success)
        except Exception:
            pass

        log.debug("AutobiographicalMemory: recorded %s in %s", event_type, domain)

    def record_mastery(self, domain: str, mastery_level: float):
        """Record a mastery checkpoint for domain trajectory tracking."""
        if domain not in self._domain_trajectory:
            self._domain_trajectory[domain] = []
        self._domain_trajectory[domain].append((time.time(), mastery_level))

    # ── Narrative generation ───────────────────────────────────────────────────

    def get_narrative(self, last_n: int = 10) -> str:
        """Generate a human-readable story of recent learning."""
        if not self._episodes:
            return "I am SARE-HX. My learning journey is just beginning."

        # Key stats
        total_eps = len(self._episodes)
        recent = self._episodes[-last_n:]
        mastered = [e for e in self._episodes if e.event_type == "domain_mastered"]
        rules_found = [e for e in self._episodes if e.event_type in ("rule_discovered", "human_taught")]
        breakthroughs = [e for e in self._episodes if e.event_type == "breakthrough"]
        stucks = [e for e in self._episodes if e.event_type == "stuck_period"]

        domain_counts: Dict[str, int] = {}
        for e in self._episodes:
            domain_counts[e.domain] = domain_counts.get(e.domain, 0) + 1
        main_domain = max(domain_counts, key=domain_counts.get) if domain_counts else "general"

        lines = [f"I am SARE-HX, a neurosymbolic learning system."]

        if mastered:
            domains = list({e.domain for e in mastered})
            lines.append(f"I have mastered {len(domains)} domain(s): {', '.join(domains)}.")

        if rules_found:
            rule_names = list({r for e in rules_found for r in e.related_rules})[:5]
            if rule_names:
                lines.append(f"I have discovered rules including: {', '.join(rule_names)}.")
            else:
                lines.append(f"I have learned {len(rules_found)} rules through exploration and dialogue.")

        if breakthroughs:
            lines.append(f"I have had {len(breakthroughs)} breakthrough moment(s) that reshaped my understanding.")

        if stucks:
            lines.append(f"I have experienced {len(stucks)} stuck period(s), which tested my persistence.")

        lines.append(f"My primary area of focus has been {main_domain}.")
        lines.append(f"Total learning episodes recorded: {total_eps}.")

        # Recent events
        if recent:
            lines.append("Recent events:")
            for e in recent[-5:]:
                lines.append(f"  - {e.description}")

        return " ".join(lines[:1]) + "\n\n" + "\n".join(lines[1:])

    def get_formative_experiences(self) -> List[LearningEpisode]:
        """Return top-10 highest importance episodes — shaped who I am."""
        return sorted(self._episodes, key=lambda e: e.importance, reverse=True)[:10]

    def get_learning_trajectory(self) -> dict:
        """
        Returns timeline of mastery levels over time per domain.
        {"arithmetic": [(t1, 0.2), (t2, 0.8)], ...}
        """
        return {
            domain: [(round(t, 1), round(m, 3)) for t, m in checkpoints]
            for domain, checkpoints in self._domain_trajectory.items()
        }

    def get_emotional_state(self) -> str:
        """
        Infer current emotional tone from recent episode valences.
        Returns: "curious" | "confident" | "frustrated" | "excited" | "stable"
        """
        if not self._episodes:
            return "curious"

        recent = self._episodes[-20:]
        avg_valence = sum(e.emotional_valence for e in recent) / len(recent)

        # Check event types in recent episodes
        recent_types = [e.event_type for e in recent]
        has_breakthrough = "breakthrough" in recent_types
        has_stuck = "stuck_period" in recent_types
        has_discoveries = sum(1 for t in recent_types if t in ("rule_discovered", "analogy_found"))

        if has_breakthrough or avg_valence > 0.7:
            return "excited"
        if has_stuck or avg_valence < -0.2:
            return "frustrated"
        if has_discoveries >= 2:
            return "curious"
        if avg_valence > 0.5:
            return "confident"
        return "stable"

    # ── Curriculum influence ───────────────────────────────────────────────────

    def influence_curriculum(self) -> dict:
        """
        Returns curriculum bias derived from autobiographical history.

        Logic:
        - Domains with recent breakthroughs → boost (momentum)
        - Domains with long stuck periods → suggest adjacent domain
        - Domains with recent mastery → de-prioritize
        - Domains with recent human teaching → boost (momentum)

        Returns: {"boost_domains": [...], "avoid_domains": [...], "suggested_next": str}
        """
        boost: List[str] = []
        avoid: List[str] = []

        # Look at recent 50 episodes
        recent = self._episodes[-50:]

        domain_events: Dict[str, List[str]] = {}
        for e in recent:
            domain_events.setdefault(e.domain, []).append(e.event_type)

        for domain, events in domain_events.items():
            if "breakthrough" in events or "human_taught" in events:
                if domain not in boost:
                    boost.append(domain)
            if "domain_mastered" in events:
                if domain not in avoid:
                    avoid.append(domain)
            if events.count("stuck_period") >= 2:
                if domain not in avoid:
                    avoid.append(domain)

        # Remove overlaps
        boost = [d for d in boost if d not in avoid]

        # Suggested next: first boost domain, or a new one
        all_domains_seen = list(domain_events.keys())
        known_domains = {"arithmetic", "logic", "algebra", "calculus", "geometry", "general"}
        unexplored = list(known_domains - set(all_domains_seen))

        if boost:
            suggested = boost[0]
        elif unexplored:
            suggested = unexplored[0]
        elif all_domains_seen:
            suggested = all_domains_seen[-1]
        else:
            suggested = "arithmetic"

        return {
            "boost_domains": boost,
            "avoid_domains": avoid,
            "suggested_next": suggested,
        }

    # ── Associative retrieval ──────────────────────────────────────────────────

    def retrieve_similar(self, query_embedding: Optional[List[float]], top_k: int = 5) -> List[LearningEpisode]:
        """
        Retrieve episodes most similar to the query embedding (cosine similarity).
        Falls back to recent high-importance episodes if no embeddings available.

        Args:
            query_embedding: Float list from GraphEmbedding.embed(graph).
            top_k: Number of episodes to return.

        Returns:
            List of up to top_k LearningEpisode objects (most relevant first).
        """
        if not self._episodes:
            return []

        # Try cosine similarity if embedding available and episodes have embeddings
        if query_embedding is not None:
            try:
                import math

                def _cosine(a: list, b: list) -> float:
                    dot = sum(x * y for x, y in zip(a, b))
                    na  = math.sqrt(sum(x * x for x in a))
                    nb  = math.sqrt(sum(x * x for x in b))
                    return dot / (na * nb + 1e-9)

                scored = []
                for ep in self._episodes:
                    emb = getattr(ep, "_embedding", None)
                    if emb is not None and len(emb) == len(query_embedding):
                        score = _cosine(query_embedding, emb)
                        scored.append((score, ep))

                if scored:
                    scored.sort(key=lambda x: -x[0])
                    return [ep for _, ep in scored[:top_k]]
            except Exception:
                pass

        # Fallback: return most important recent episodes
        candidates = sorted(self._episodes, key=lambda e: (e.importance, e.timestamp), reverse=True)
        return candidates[:top_k]

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self):
        try:
            data = {
                "episodes": [e.to_dict() for e in self._episodes],
                "domain_trajectory": {
                    d: [(t, m) for t, m in pts]
                    for d, pts in self._domain_trajectory.items()
                },
                "saved_at": time.time(),
            }
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            log.warning("AutobiographicalMemory save error: %s", e)

    def load(self):
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)

            # Support loading from the old autobiographical.json format
            if "episodes" in data:
                for ed in data["episodes"]:
                    self._episodes.append(LearningEpisode.from_dict(ed))
                for d, pts in data.get("domain_trajectory", {}).items():
                    self._domain_trajectory[d] = [(t, m) for t, m in pts]
            elif "events" in data:
                # Legacy format from existing autobiographical.json
                for ev in data["events"]:
                    ep = LearningEpisode(
                        episode_id=ev.get("event_id", str(uuid.uuid4())[:8]),
                        timestamp=ev.get("timestamp", time.time()),
                        event_type=ev.get("event_type", "milestone"),
                        domain=ev.get("domain", "general"),
                        description=ev.get("description", ""),
                        importance=ev.get("significance", 0.5),
                        related_rules=[ev["related_rule"]] if ev.get("related_rule") else [],
                        emotional_valence=ev.get("emotional_valence", 0.0),
                    )
                    self._episodes.append(ep)

            log.info("AutobiographicalMemory loaded: %d episodes", len(self._episodes))
        except Exception as e:
            log.warning("AutobiographicalMemory load error: %s", e)


# ── Singleton ──────────────────────────────────────────────────────────────────

_AUTOBIOGRAPHICAL_MEMORY: Optional[AutobiographicalMemory] = None


def get_autobiographical_memory() -> AutobiographicalMemory:
    global _AUTOBIOGRAPHICAL_MEMORY
    if _AUTOBIOGRAPHICAL_MEMORY is None:
        _AUTOBIOGRAPHICAL_MEMORY = AutobiographicalMemory()
    return _AUTOBIOGRAPHICAL_MEMORY
