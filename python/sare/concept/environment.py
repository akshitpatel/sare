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

        if b == 0:
            hint = "identity_addition"
            sym = "x + 0 = x"
        elif a == 0:
            hint = "zero_addition"
            sym = "0 + x = x"
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
        elif b > a:
            hint = "floor_subtraction"
            sym = "x - y = 0 when y > x"
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
        elif a == 1:
            hint = "single_group"
            sym = "1 * x = x"
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
            description=f"{int(a) if a == int(a) else a} group(s) of {b_obj} = {r_obj}",
            symbolic=sym,
            concept_hint=hint,
        )
        self._observations.append(obs)
        self._concept_counts[hint] = self._concept_counts.get(hint, 0) + 1
        return obs

    def experiment_negate_twice(self, statement: str = "it is raining") -> Observation:
        """Double negation: ¬¬P = P."""
        obs = Observation(
            operation="negate_twice",
            inputs=[WorldObject("statement", 1, properties={"text": statement})],
            result=WorldObject("statement", 1, properties={"text": statement}),
            description=f'not(not("{statement}")) = "{statement}"',
            symbolic="¬¬P = P",
            concept_hint="double_negation",
        )
        self._observations.append(obs)
        self._concept_counts["double_negation"] = self._concept_counts.get("double_negation", 0) + 1
        return obs

    def experiment_gravity(
        self, obj: str = "ball", height: float = 1.0
    ) -> Observation:
        """Simple gravity: object at height falls to ground."""
        start = WorldObject(obj, 1, unit="item", properties={"height": height})
        end = WorldObject(obj, 1, unit="item", properties={"height": 0.0, "state": "grounded"})
        obs = Observation(
            operation="gravity",
            inputs=[start],
            result=end,
            description=f"{obj} at height {height} falls to ground",
            symbolic="height > 0 -> height = 0",
            concept_hint="gravity",
            energy_delta=-abs(height),
        )
        self._observations.append(obs)
        self._concept_counts["gravity"] = self._concept_counts.get("gravity", 0) + 1
        return obs

    def experiment_collision(
        self, obj_a: str = "ball", obj_b: str = "block", speed: float = 1.0
    ) -> Observation:
        """Simple collision: moving object hits another object and stops."""
        a = WorldObject(obj_a, 1, properties={"speed": speed})
        b = WorldObject(obj_b, 1, properties={"speed": 0.0})
        result = WorldObject(obj_a, 1, properties={"speed": 0.0, "collided_with": obj_b})
        obs = Observation(
            operation="collision",
            inputs=[a, b],
            result=result,
            description=f"{obj_a} moving at {speed} hits {obj_b} and stops",
            symbolic="moving(A) & hit(A,B) -> stopped(A)",
            concept_hint="collision_stop",
            energy_delta=-abs(speed),
        )
        self._observations.append(obs)
        self._concept_counts["collision_stop"] = self._concept_counts.get("collision_stop", 0) + 1
        return obs

    # ── Sampling / curriculum helpers ─────────────────────────────────────────

    def random_experiment(self) -> Observation:
        """Run a random experiment for exploration."""
        choice = random.choice(
            ["add", "remove", "multiply", "negate_twice", "gravity", "collision"]
        )

        if choice == "add":
            a = random.randint(0, 5)
            b = random.randint(0, 5)
            obj = random.choice(self.OBJECT_TYPES)
            return self.experiment_add(a, b, obj=obj)

        if choice == "remove":
            a = random.randint(0, 5)
            b = random.randint(0, 5)
            obj = random.choice(self.OBJECT_TYPES)
            return self.experiment_remove(a, b, obj=obj)

        if choice == "multiply":
            a = random.randint(0, 5)
            b = random.randint(0, 5)
            obj = random.choice(self.OBJECT_TYPES)
            return self.experiment_multiply(a, b, obj=obj)

        if choice == "negate_twice":
            stmt = random.choice(
                [
                    "it is raining",
                    "the light is on",
                    "the door is open",
                    "the block is red",
                ]
            )
            return self.experiment_negate_twice(stmt)

        if choice == "gravity":
            obj = random.choice(["ball", "apple", "coin", "block"])
            height = random.randint(1, 10)
            return self.experiment_gravity(obj=obj, height=float(height))

        obj_a = random.choice(["ball", "coin"])
        obj_b = random.choice(["block", "wall"])
        speed = random.randint(1, 5)
        return self.experiment_collision(obj_a=obj_a, obj_b=obj_b, speed=float(speed))

    def run_curriculum(self, n: int = 10) -> List[Observation]:
        """Run a small mixed curriculum of experiments."""
        return [self.random_experiment() for _ in range(max(0, n))]

    # ── Observation / concept utilities ───────────────────────────────────────

    def observations(self) -> List[Observation]:
        return list(self._observations)

    def concept_counts(self) -> Dict[str, int]:
        return dict(self._concept_counts)

    def recent_observations(self, limit: int = 10) -> List[Observation]:
        if limit <= 0:
            return []
        return self._observations[-limit:]

    def clear(self) -> None:
        self._observations.clear()
        self._concept_counts.clear()

    def summarize_concepts(self) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for obs in self._observations:
            entry = summary.setdefault(
                obs.concept_hint,
                {"count": 0, "symbolic_forms": set(), "operations": set()},
            )
            entry["count"] += 1
            entry["symbolic_forms"].add(obs.symbolic)
            entry["operations"].add(obs.operation)

        for concept, entry in summary.items():
            entry["symbolic_forms"] = sorted(entry["symbolic_forms"])
            entry["operations"] = sorted(entry["operations"])
        return summary

    def generate_symbolic_rules(self, min_support: int = 2) -> List[Dict[str, Any]]:
        """
        Promote frequently observed grounded regularities into symbolic rules.

        Simplified: each concept maps to exactly one symbolic form, so we avoid
        iterating over a set of forms.
        """
        concept_to_symbolic: Dict[str, str] = {}
        concept_to_operations: Dict[str, set] = {}

        for obs in self._observations:
            if obs.concept_hint not in concept_to_symbolic:
                concept_to_symbolic[obs.concept_hint] = obs.symbolic
            concept_to_operations.setdefault(obs.concept_hint, set()).add(obs.operation)

        rules: List[Dict[str, Any]] = []
        for concept, count in self._concept_counts.items():
            if count < min_support:
                continue
            symbolic = concept_to_symbolic.get(concept)
            if not symbolic:
                continue
            rules.append(
                {
                    "concept": concept,
                    "rule": symbolic,
                    "support": count,
                    "operations": sorted(concept_to_operations.get(concept, set())),
                    "grounded": True,
                }
            )

        rules.sort(key=lambda r: (-r["support"], r["concept"], r["rule"]))
        return rules

    # ── Backwards-compatible aliases / exports ────────────────────────────────

    def get_observations(self) -> List[Observation]:
        return self.observations()

    def get_concept_counts(self) -> Dict[str, int]:
        return self.concept_counts()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observations": [o.to_dict() for o in self._observations],
            "concept_counts": dict(self._concept_counts),
        }