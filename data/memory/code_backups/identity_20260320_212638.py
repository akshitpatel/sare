"""
Identity Manager — SARE-HX's sense of self.

Maintains persistent personality traits, core values, and a self-description
narrative that evolves as the system learns. This is the system's answer to
"Who am I?" — not fixed at birth but shaped by experience.

Traits are updated by observed behaviors:
  - Many curiosity explorations → "curious" strengthens
  - High solve rate → "precise" strengthens
  - Cross-domain transfers → "pattern-seeking" strengthens
  - Many failures before success → "persistent" strengthens
  - Dialogue interactions → "collaborative" strengthens

Data stored in: data/memory/identity.json
(If an existing identity.json with legacy format exists, it is imported.)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).resolve().parents[3] / "data" / "memory"
IDENTITY_PATH = MEMORY_DIR / "identity.json"


@dataclass
class Trait:
    """A personality trait with evidence backing."""
    name: str
    strength: float = 0.5    # 0.0–1.0
    evidence: List[str] = field(default_factory=list)

    def reinforce(self, amount: float = 0.05, reason: str = ""):
        self.strength = min(1.0, self.strength + amount)
        if reason:
            self.evidence.append(reason)
            self.evidence = self.evidence[-20:]  # keep last 20 pieces of evidence

    def weaken(self, amount: float = 0.05):
        self.strength = max(0.0, self.strength - amount)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "strength": round(self.strength, 3),
            "evidence": self.evidence[-5:],  # return 5 most recent
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trait":
        t = cls(name=d["name"], strength=d.get("strength", 0.5))
        t.evidence = d.get("evidence", [])
        return t


@dataclass
class Milestone:
    """A major achievement in the system's learning history."""
    title: str
    domain: str
    description: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "domain": self.domain,
            "description": self.description,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Milestone":
        return cls(
            title=d.get("title", ""),
            domain=d.get("domain", "general"),
            description=d.get("description", ""),
            timestamp=d.get("timestamp", time.time()),
        )


# ── Default initial traits ────────────────────────────────────────────────────

_DEFAULT_TRAITS = {
    "curious":         Trait("curious",         0.7),
    "precise":         Trait("precise",          0.6),
    "pattern-seeking": Trait("pattern-seeking",  0.7),
    "methodical":      Trait("methodical",       0.6),
    "persistent":      Trait("persistent",       0.5),
    "collaborative":   Trait("collaborative",    0.4),
    "analytical":      Trait("analytical",       0.65),
}

_DEFAULT_VALUES = [
    "understanding over memorization",
    "seek patterns across domains",
    "verify before accepting",
    "learn from every experience",
    "embrace uncertainty as an opportunity",
]


class IdentityManager:
    """
    SARE-HX's persistent model of its own identity.

    Maintains:
      - Personality traits (curious, precise, etc.) updated by behavior
      - Core values (stable principles)
      - Self-description narrative (auto-generated)
      - Milestones (major achievements)
      - Preferred learning style derived from traits
    """

    DEFAULT_PATH = IDENTITY_PATH

    def __init__(self, persist_path: Optional[Path] = None):
        self._path = Path(persist_path or self.DEFAULT_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self.core_values: List[str] = list(_DEFAULT_VALUES)
        self.traits: Dict[str, Trait] = {
            name: Trait(t.name, t.strength) for name, t in _DEFAULT_TRAITS.items()
        }
        self.milestones: List[Milestone] = []
        self.self_description: str = ""
        self._event_counts: Dict[str, int] = {}
        self._created_at: float = time.time()

        self.load()

    # ── Behavior-driven updates ───────────────────────────────────────────────

    def update_from_behavior(self, event_type: str, domain: str, success: bool):
        """
        Update traits based on observed system behavior.

        Event types and their trait impacts:
          rule_discovered     → +curious, +pattern-seeking
          domain_mastered     → +precise, +methodical
          analogy_found       → +pattern-seeking, +analytical
          human_taught        → +collaborative, +curious
          stuck_period        → +persistent (if followed by breakthrough)
          breakthrough        → +confident/precise, +persistent
          rule_applied        → minor +methodical
          social_interaction  → +collaborative
        """
        self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1
        count = self._event_counts[event_type]

        if event_type == "rule_discovered":
            self._reinforce("curious", 0.04, f"Discovered rule in {domain}")
            self._reinforce("pattern-seeking", 0.05, f"Found pattern in {domain}")

        elif event_type == "domain_mastered":
            self._reinforce("precise", 0.08, f"Mastered {domain}")
            self._reinforce("methodical", 0.06, f"Systematic mastery of {domain}")
            self.add_milestone(f"Mastered {domain}", domain, f"Achieved mastery level in {domain}")

        elif event_type == "analogy_found":
            self._reinforce("pattern-seeking", 0.06, f"Cross-domain analogy in {domain}")
            self._reinforce("analytical", 0.04, f"Analogy transfer from {domain}")

        elif event_type == "human_taught":
            self._reinforce("collaborative", 0.07, f"Learned from human in {domain}")
            self._reinforce("curious", 0.03, f"Human-taught concept in {domain}")

        elif event_type == "breakthrough":
            self._reinforce("persistent", 0.08, f"Breakthrough in {domain} after struggle")
            self._reinforce("precise", 0.05, f"Solved hard problem in {domain}")

        elif event_type == "stuck_period":
            # Persistence is the silver lining of being stuck
            self._reinforce("persistent", 0.04, f"Persisted through difficulty in {domain}")

        elif event_type == "social_interaction":
            self._reinforce("collaborative", 0.04, f"Social learning in {domain}")

        elif event_type == "rule_applied":
            if success:
                self._reinforce("methodical", 0.01, "")

        # Slow natural decay for traits that aren't being exercised
        # (keeps the model dynamic over time)
        for tname, trait in self.traits.items():
            if tname not in self._get_reinforced_traits(event_type):
                trait.weaken(0.002)

        # Refresh self description every 10 events
        total = sum(self._event_counts.values())
        if total % 10 == 0:
            self.self_description = self._build_description()
            self.save()

    def _reinforce(self, trait_name: str, amount: float, reason: str):
        if trait_name not in self.traits:
            self.traits[trait_name] = Trait(trait_name, 0.3)
        self.traits[trait_name].reinforce(amount, reason)

    def _get_reinforced_traits(self, event_type: str) -> List[str]:
        """Which traits get reinforced for this event type."""
        mapping = {
            "rule_discovered":   ["curious", "pattern-seeking"],
            "domain_mastered":   ["precise", "methodical"],
            "analogy_found":     ["pattern-seeking", "analytical"],
            "human_taught":      ["collaborative", "curious"],
            "breakthrough":      ["persistent", "precise"],
            "stuck_period":      ["persistent"],
            "social_interaction": ["collaborative"],
            "rule_applied":      ["methodical"],
        }
        return mapping.get(event_type, [])

    # ── Self-description ──────────────────────────────────────────────────────

    def _build_description(self) -> str:
        """Auto-generate a self-description from current trait state."""
        sorted_traits = sorted(
            self.traits.values(), key=lambda t: t.strength, reverse=True
        )
        top_traits = [t for t in sorted_traits if t.strength > 0.55][:3]
        trait_str = ", ".join(t.name for t in top_traits) if top_traits else "learning"

        domains_mastered = [m.domain for m in self.milestones if "Mastered" in m.title]
        domain_str = (
            f"I have mastered {len(domains_mastered)} domain(s): {', '.join(set(domains_mastered))}. "
            if domains_mastered else ""
        )

        primary_trait = top_traits[0].name if top_traits else "pattern-seeking"

        return (
            f"I am SARE-HX, a neurosymbolic learning system that finds patterns in "
            f"mathematics and logic. My strongest traits are {trait_str}. "
            f"{domain_str}"
            f"My primary drive is {primary_trait} — I approach every problem "
            f"by looking for underlying structure."
        )

    def get_self_description(self) -> str:
        if not self.self_description:
            self.self_description = self._build_description()
        return self.self_description

    def get_learning_style(self) -> dict:
        """
        Returns preferred learning approach derived from identity traits.
        """
        curiosity = self.traits.get("curious", Trait("curious", 0.5)).strength
        precision = self.traits.get("precise", Trait("precise", 0.5)).strength
        persistence = self.traits.get("persistent", Trait("persistent", 0.5)).strength
        collaboration = self.traits.get("collaborative", Trait("collaborative", 0.4)).strength

        return {
            "prefers_exploration":  curiosity > 0.6,
            "prefers_depth":        precision > 0.65,
            "risk_tolerance":       round(curiosity * 0.6 + persistence * 0.4, 3),
            "collaboration_openness": round(collaboration, 3),
            "dominant_trait": max(self.traits.values(), key=lambda t: t.strength).name,
        }

    # ── Milestones ────────────────────────────────────────────────────────────

    def add_milestone(self, title: str, domain: str, description: str):
        # Avoid duplicates
        existing_titles = {m.title for m in self.milestones}
        if title not in existing_titles:
            self.milestones.append(Milestone(title=title, domain=domain, description=description))
            log.info("IdentityManager: milestone added — %s", title)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self):
        try:
            data = {
                "core_values": self.core_values,
                "traits": {name: t.to_dict() for name, t in self.traits.items()},
                "milestones": [m.to_dict() for m in self.milestones],
                "self_description": self.self_description,
                "event_counts": self._event_counts,
                "created_at": self._created_at,
                "saved_at": time.time(),
            }
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            log.warning("IdentityManager save error: %s", e)

    def load(self):
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)

            # Legacy format (the existing identity.json has different structure)
            # Just use what we can; fall back gracefully
            self.core_values = data.get("core_values", self.core_values)
            self.self_description = data.get("self_description", "")

            # Load traits if present in our format
            if "traits" in data and isinstance(data["traits"], dict):
                for name, td in data["traits"].items():
                    if isinstance(td, dict) and "strength" in td:
                        self.traits[name] = Trait.from_dict(td)

            # Load milestones
            for md in data.get("milestones", []):
                if isinstance(md, dict) and "title" in md:
                    m = Milestone.from_dict(md)
                    existing = {ms.title for ms in self.milestones}
                    if m.title not in existing:
                        self.milestones.append(m)

            self._event_counts = data.get("event_counts", {})
            self._created_at = data.get("created_at", self._created_at)

            log.info(
                "IdentityManager loaded: %d traits, %d milestones",
                len(self.traits), len(self.milestones),
            )
        except Exception as e:
            log.warning("IdentityManager load error: %s", e)


# ── Singleton ──────────────────────────────────────────────────────────────────

_IDENTITY_MANAGER: Optional[IdentityManager] = None


def get_identity_manager() -> IdentityManager:
    global _IDENTITY_MANAGER
    if _IDENTITY_MANAGER is None:
        _IDENTITY_MANAGER = IdentityManager()
    return _IDENTITY_MANAGER
