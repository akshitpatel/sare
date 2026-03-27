"""
EnvironmentSimulator — Grounded learning from concrete object-world observations.

Biological intelligence learns concepts from observation:
  child sees: 🍎 + 🍎 = 2 apples → abstracts: "addition combines quantities"

This module provides a simple discrete world where SARE-HX can:
  1. Run experiments: place objects, apply operations, observe results
  2. Form concepts from repeated observations
  3. Ground symbolic rules in concrete experience

World model: objects with quantities, typed properties, and transformation rules.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ── World Object ───────────────────────────────────────────────────────────────

@dataclass
class WorldObject:
    """A concrete object in the environment."""
    name: str           # e.g. "apple"
    quantity: float     # e.g. 3
    unit: str = ""      # e.g. "pieces"
    properties: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        if self.quantity == int(self.quantity):
            q = int(self.quantity)
        else:
            q = self.quantity
        return f"{q} {self.name}{'s' if q != 1 else ''}"

    def to_dict(self):
        return {"name": self.name, "quantity": self.quantity, "unit": self.unit}


@dataclass
class Observation:
    """Result of running an experiment in the environment."""
    operation: str
    inputs: List[WorldObject]
    result: WorldObject
    description: str        # natural language: "3 apples + 0 apples = 3 apples"
    symbolic: str           # abstract form: "x + 0 = x"
    concept_hint: str       # what concept this demonstrates: "identity_addition"
    energy_delta: float = 0.0

    def to_dict(self):
        return {
            "operation": self.operation,
            "inputs": [o.to_dict() for o in self.inputs],
            "result": self.result.to_dict(),
            "description": self.description,
            "symbolic": self.symbolic,
            "concept_hint": self.concept_hint,
        }


# ── Environment Simulator ─────────────────────────────────────────────────────

class EnvironmentSimulator:
    """
    A discrete object world for grounded concept learning.

    Supports:
      - Arithmetic experiments (add, remove, multiply groups)
      - Physics experiments (simple: gravity, collision)
      - Pattern observation and concept abstraction
    """

    OBJECT_TYPES = ["apple", "block", "coin", "ball", "step", "unit"]

    def __init__(self):
        self._observations: List[Observation] = []
        self._concept_counts: Dict[str, int] = {}

    # ── Core experiment runners ────────────────────────────────────────────────

    def experiment_add(
        self, a: float, b: float, obj: str = "apple"
    ) -> Observation:
        """Run: a objects + b objects → observe result."""
        result = a + b
        a_obj = WorldObject(obj, a)
        b_obj = WorldObject(obj, b)
        r_obj = WorldObject(obj, result)

        # What concept does this demonstrate?
        if b == 0:
            hint = "identity_addition"
            sym = "x + 0 = x"
        elif a == b:
            hint = "doubling"
            sym = "x + x = 2x"
        else:
            hint = "addition"
            sym = "x + y = z"

        obs = Observation(
            operation="add",
            inputs=[a_obj, b_obj],
            result=r_obj,
            description=f"{a_obj} + {b_obj} = {r_obj}",
            symbolic=sym,
            concept_hint=hint,
        )
        self._observations.append(obs)
        self._concept_counts[hint] = self._concept_counts.get(hint, 0) + 1
        return obs

    def experiment_remove(
        self, a: float, b: float, obj: str = "apple"
    ) -> Observation:
        """Run: a objects - b objects → observe result."""
        result = max(0.0, a - b)
        a_obj = WorldObject(obj, a)
        b_obj = WorldObject(obj, b)
        r_obj = WorldObject(obj, result)

        if b == 0:
            hint = "identity_subtraction"
            sym = "x - 0 = x"
        elif a == b:
            hint = "self_cancellation"
            sym = "x - x = 0"
        else:
            hint = "subtraction"
            sym = "x - y = z"

        obs = Observation(
            operation="remove",
            inputs=[a_obj, b_obj],
            result=r_obj,
            description=f"{a_obj} - {b_obj} = {r_obj}",
            symbolic=sym,
            concept_hint=hint,
        )
        self._observations.append(obs)
        self._concept_counts[hint] = self._concept_counts.get(hint, 0) + 1
        return obs

    def experiment_multiply(
        self, a: float, b: float, obj: str = "apple"
    ) -> Observation:
        """Run: a groups of b objects → observe total."""
        result = a * b
        a_obj = WorldObject(f"group-of-{obj}", a)
        b_obj = WorldObject(obj, b)
        r_obj = WorldObject(obj, result)

        if b == 1:
            hint = "identity_multiplication"
            sym = "x * 1 = x"
        elif b == 0 or a == 0:
            hint = "annihilation"
            sym = "x * 0 = 0"
        else:
            hint = "multiplication"
            sym = "x * y = z"

        obs = Observation(
            operation="multiply",
            inputs=[a_obj, b_obj],
            result=r_obj,
            description=f"{int(a)} group(s) of {b_obj} = {r_obj}",
            symbolic=sym,
            concept_hint=hint,
        )
        self._observations.append(obs)
        self._concept_counts[hint] = self._concept_counts.get(hint, 0) + 1
        return obs

    def experiment_negate_twice(self, statement: str = "it is raining") -> Observation:
        """Double negation: ¬¬P = P."""
        once = f"it is NOT the case that {statement}"
        twice = statement  # ¬¬P = P

        obs = Observation(
            operation="negate_twice",
            inputs=[WorldObject("statement", 1, properties={"text": statement})],
            result=WorldObject("statement", 1, properties={"text": twice}),
            description=f"NOT (NOT '{statement}') = '{twice}'",
            symbolic="¬¬x = x",
            concept_hint="double_negation",
        )
        self._observations.append(obs)
        self._concept_counts["double_negation"] = self._concept_counts.get("double_negation", 0) + 1
        return obs

    # ── Batch experiment runners ────────────────────────────────────────────────

    def run_identity_discovery(self, n: int = 5) -> List[Observation]:
        """Run n experiments designed to discover identity elements."""
        obs = []
        objects = random.sample(self.OBJECT_TYPES, min(3, len(self.OBJECT_TYPES)))
        for obj in objects:
            for _ in range(n // len(objects) + 1):
                a = random.randint(1, 10)
                obs.append(self.experiment_add(a, 0, obj))   # identity addition
                obs.append(self.experiment_multiply(a, 1, obj))  # identity multiplication
                obs.append(self.experiment_remove(a, 0, obj))    # identity subtraction
        return obs

    def run_annihilation_discovery(self, n: int = 5) -> List[Observation]:
        """Run experiments discovering annihilation (x*0=0)."""
        obs = []
        for obj in random.sample(self.OBJECT_TYPES, min(3, len(self.OBJECT_TYPES))):
            for a in [1, 2, 5, 10, 100]:
                obs.append(self.experiment_multiply(a, 0, obj))
        return obs

    def run_commutativity_discovery(self, n: int = 4) -> List[Observation]:
        """Discover that a+b = b+a by observing both orders."""
        obs = []
        pairs = [(2, 3), (1, 4), (5, 2)]
        for a, b in pairs:
            obj = random.choice(self.OBJECT_TYPES)
            obs_ab = self.experiment_add(a, b, obj)
            obs_ba = self.experiment_add(b, a, obj)
            # Both have same result: demonstrates commutativity
            obs.extend([obs_ab, obs_ba])
        return obs

    def run_full_discovery_session(self) -> List[Observation]:
        """Run a complete concept discovery session."""
        all_obs = []
        all_obs.extend(self.run_identity_discovery(6))
        all_obs.extend(self.run_annihilation_discovery(4))
        all_obs.extend(self.run_commutativity_discovery(4))
        # Double negation in logic
        statements = ["it is sunny", "the number is positive", "x is greater than 0"]
        for s in statements:
            all_obs.append(self.experiment_negate_twice(s))
        return all_obs

    # ── Concept extraction ─────────────────────────────────────────────────────

    def extract_concepts(self) -> Dict[str, List[Observation]]:
        """Group observations by the concept they demonstrate."""
        groups: Dict[str, List[Observation]] = {}
        for obs in self._observations:
            groups.setdefault(obs.concept_hint, []).append(obs)
        return groups

    def get_concept_evidence(self, concept_hint: str) -> List[Observation]:
        """Get all observations supporting a concept."""
        return [o for o in self._observations if o.concept_hint == concept_hint]

    def generate_symbolic_rules(self) -> List[Tuple[str, str, int]]:
        """
        From repeated observations, generate candidate symbolic rules.
        Returns list of (concept, symbolic_form, evidence_count).
        """
        rules = []
        by_concept = self.extract_concepts()
        for concept, obs_list in by_concept.items():
            if len(obs_list) < 2:
                continue
            # All observations for this concept share the same symbolic form
            sym_forms = set(o.symbolic for o in obs_list)
            for sym in sym_forms:
                rules.append((concept, sym, len(obs_list)))
        rules.sort(key=lambda x: -x[2])
        return rules

    def summary(self) -> dict:
        return {
            "total_observations": len(self._observations),
            "concepts_discovered": len(self._concept_counts),
            "concept_evidence": dict(self._concept_counts),
            "symbolic_rules": [
                {"concept": c, "rule": r, "evidence": n}
                for c, r, n in self.generate_symbolic_rules()
            ],
        }
