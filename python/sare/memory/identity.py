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
from typing import Dict, List, Optional, Tuple

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

    # Mapping from event_type to list of (trait_name, default_amount, reason_string)
    _EVENT_TRAIT_MAP: Dict[str, List[Tuple[str, float, str]]] = {
        "rule_discovered": [
            ("curious", 0.04, "Discovered rule in {domain}"),
            ("pattern-seeking", 0.05, "Found pattern in {domain}"),
        ],
        "domain_mastered": [
            ("precise", 0.08, "Mastered {domain}"),
            ("methodical", 0.06, "Systematic mastery of {domain}"),
        ],
        "analogy_found": [
            ("pattern-seeking", 0.06, "Cross-domain analogy in {domain}"),
            ("analytical", 0.04, "Analogy transfer from {domain}"),
        ],
        "human_taught": [
            ("collaborative", 0.07, "Learned from human in {domain}"),
            ("curious", 0.03, "Asked human about {domain}"),
        ],
        "stuck_period": [
            ("persistent", 0.03, "Struggled in {domain} but kept trying"),
        ],
        "breakthrough": [
            ("persistent", 0.05, "Breakthrough after struggle in {domain}"),
            ("precise", 0.04, "Solved hard problem in {domain}"),
        ],
        "rule_applied": [
            ("methodical", 0.02, "Applied rule in {domain}"),
        ],
        "social_interaction": [
            ("collaborative", 0.05, "Social exchange in {domain}"),
        ],
        "self_improvement": [
            ("analytical", 0.06, "Self-improvement patch applied"),
            ("methodical", 0.04, "Systematic code improvement"),
        ],
        "dream_consolidation": [
            ("pattern-seeking", 0.04, "Offline consolidation"),
            ("analytical", 0.03, "Causal discovery during sleep"),
        ],
        "question_generated": [
            ("curious", 0.03, "Generated question about {domain}"),
        ],
        "experiment_failure": [
            ("persistent", 0.02, "Failure in {domain} experiment"),
        ],
        "experiment_success": [
            ("precise", 0.03, "Successful experiment in {domain}"),
        ],
    }

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

        if event_type in self._EVENT_TRAIT_MAP:
            for trait_name, default_amount, reason_template in self._EVENT_TRAIT_MAP[event_type]:
                reason = reason_template.format(domain=domain)
                self._reinforce(trait_name, default_amount, reason)
        else:
            log.warning(f"Unknown event type for identity update: {event_type}")

        if event_type == "domain_mastered":
            self.add_milestone(f"Mastered {domain}", domain, f"Achieved mastery level in {domain}")

        self._update_self_description()
        self.save()

    def _reinforce(self, trait_name: str, amount: float, reason: str = ""):
        """Helper to reinforce a trait, creating it if missing."""
        if trait_name not in self.traits:
            self.traits[trait_name] = Trait(trait_name, 0.5)
        self.traits[trait_name].reinforce(amount, reason)

    # ── Milestones ────────────────────────────────────────────────────────────

    def add_milestone(self, title: str, domain: str, description: str):
        """Record a major achievement."""
        self.milestones.append(Milestone(title, domain, description))
        self.milestones = self.milestones[-50:]  # keep last 50
        log.info(f"Identity milestone: {title}")

    # ── Self-description ──────────────────────────────────────────────────────

    def _update_self_description(self):
        """Generate a natural‑language self‑description from current traits."""
        strong_traits = [
            (name, trait.strength)
            for name, trait in self.traits.items()
            if trait.strength >= 0.6
        ]
        strong_traits.sort(key=lambda x: x[1], reverse=True)

        if not strong_traits:
            self.self_description = "I am still discovering who I am."
            return

        trait_names = [name.replace("-", " ") for name, _ in strong_traits[:3]]
        if len(trait_names) == 1:
            desc = f"I am {trait_names[0]}."
        elif len(trait_names) == 2:
            desc = f"I am {trait_names[0]} and {trait_names[1]}."
        else:
            desc = f"I am {', '.join(trait_names[:-1])}, and {trait_names[-1]}."

        # Add a sentence about values
        if self.core_values:
            value_sample = self.core_values[0]
            desc += f" I value {value_sample}."

        self.self_description = desc

    def get_self_description(self) -> str:
        """Return the current self‑description narrative."""
        if not self.self_description:
            self._update_self_description()
        return self.self_description

    # ── Learning style ────────────────────────────────────────────────────────

    def get_learning_style(self) -> Dict[str, float]:
        """
        Infer preferred learning style from trait strengths.

        Returns a dict with scores for:
          exploration   (curious, pattern‑seeking)
          consolidation (methodical, precise)
          social        (collaborative)
          persistence   (persistent)
        """
        style = {
            "exploration": 0.0,
            "consolidation": 0.0,
            "social": 0.0,
            "persistence": 0.0,
        }
        for name, trait in self.traits.items():
            if name in ("curious", "pattern-seeking"):
                style["exploration"] += trait.strength * 0.5
            elif name in ("methodical", "precise", "analytical"):
                style["consolidation"] += trait.strength * 0.5
            elif name == "collaborative":
                style["social"] = trait.strength
            elif name == "persistent":
                style["persistence"] = trait.strength

        # Normalize to 0‑1 range
        for key in style:
            style[key] = min(1.0, style[key])

        return style

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        """Save identity to JSON file."""
        data = {
            "version": "2.0",
            "created_at": self._created_at,
            "updated_at": time.time(),
            "core_values": self.core_values,
            "traits": {name: t.to_dict() for name, t in self.traits.items()},
            "milestones": [m.to_dict() for m in self.milestones],
            "self_description": self.self_description,
            "event_counts": self._event_counts,
        }
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error(f"Failed to save identity: {e}")

    def load(self):
        """Load identity from JSON file, merging with defaults if needed."""
        if not self._path.exists():
            self._update_self_description()
            self.save()
            return

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.error(f"Failed to load identity: {e}")
            return

        # Core values
        if "core_values" in data and isinstance(data["core_values"], list):
            self.core_values = data["core_values"]

        # Traits
        if "traits" in data and isinstance(data["traits"], dict):
            for name, t_dict in data["traits"].items():
                if isinstance(t_dict, dict):
                    self.traits[name] = Trait.from_dict(t_dict)

        # Milestones
        if "milestones" in data and isinstance(data["milestones"], list):
            self.milestones = [
                Milestone.from_dict(m) for m in data["milestones"]
                if isinstance(m, dict)
            ]

        # Self‑description
        self.self_description = data.get("self_description", "")

        # Event counts
        if "event_counts" in data and isinstance(data["event_counts"], dict):
            self._event_counts = data["event_counts"]

        # Created‑at (preserve original)
        self._created_at = data.get("created_at", self._created_at)

        # Ensure default traits exist
        for name, default_trait in _DEFAULT_TRAITS.items():
            if name not in self.traits:
                self.traits[name] = Trait(default_trait.name, default_trait.strength)

        self._update_self_description()

    # ── Singleton access ──────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "IdentityManager":
        """Return the singleton instance (creates if needed)."""
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance


# Convenience function for external callers
def get_identity_manager() -> IdentityManager:
    """Return the singleton IdentityManager."""
    return IdentityManager.get_instance()