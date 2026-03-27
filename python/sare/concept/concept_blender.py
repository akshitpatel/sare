"""
ConceptBlender — Cross-domain Novel Concept Synthesis

Conceptual blending theory (Fauconnier & Turner, 2002) explains how
the mind creates genuinely novel ideas by merging two mental spaces:

  Input Space A  +  Input Space B  →  Generic Space  →  Blended Space

Unlike analogy (A maps to B), blending produces something NEW that
exists in neither input space:
  "electric current" (flow, rate) + "water in pipe" (pressure, channel)
  → "voltage as pressure" — novel inference: high R → low flow

Implementation:
  - InputSpace:   a concept + its domain + properties + relations
  - BlendMapping: structural overlap between two input spaces
  - ConceptBlend: the novel concept produced (with novel inferences)
  - ConceptBlender: find_blend_opportunities(), blend(), summary()

Wiring:
  - ConceptGraph provides the input spaces (nodes + edges)
  - Brain.learn_cycle calls blender.discover_blends(concept_graph) every N cycles
  - Accepted blends feed back to ConceptGraph as new cross-domain nodes
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class InputSpace:
    """One mental space — a concept with its structural features."""
    concept: str
    domain: str
    properties: Dict[str, Any]   # {property_name: value}
    relations: List[str]         # ["A causes B", "X leads_to Y"]
    examples: List[str]          # concrete instances

    def __post_init__(self):
        # Cache structural features once. Treat the cached set as immutable by
        # storing a frozenset, so callers cannot mutate internal state.
        self._structural_features_cache: frozenset[str] = frozenset(self.properties.keys())

    def structural_features(self) -> Set[str]:
        """Abstract structural roles (keys only, not values).

        Returns a set-like view. The underlying cached storage is immutable
        (frozenset) to prevent accidental mutation across the O(n²) blend
        discovery loop.
        """
        # Return a new set to preserve the method's Set[str] contract while
        # keeping internal cache immutable.
        return set(self._structural_features_cache)

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "domain": self.domain,
            "properties": self.properties,
            "relations": self.relations[:3],
            "examples": self.examples[:2],
        }


@dataclass
class BlendMapping:
    """Structural correspondences between two input spaces."""
    space_a: InputSpace
    space_b: InputSpace
    shared_roles: List[str]  # property keys that map across both
    score: float             # 0.0 (no overlap) → 1.0 (perfect structure)

    def to_dict(self) -> dict:
        return {
            "space_a": self.space_a.concept,
            "space_b": self.space_b.concept,
            "shared_roles": self.shared_roles,
            "score": round(self.score, 3),
        }


@dataclass
class ConceptBlend:
    """
    A novel concept produced by blending two input spaces.

    The blend is NOT just a union; it selectively projects from both
    and adds emergent structure (novel_inferences) not in either input.
    """
    name: str
    source_a: str         # concept name from input space A
    source_b: str         # concept name from input space B
    domain_a: str
    domain_b: str
    blended_properties: Dict[str, Any]
    novel_inferences: List[str]   # new facts that emerge from the blend
    mapping: BlendMapping
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.6
    accepted: bool = False

    def blend_expression(self) -> str:
        """A symbolic expression summarising the blend."""
        return f"{self.source_a}[{self.domain_a}] ⊕ {self.source_b}[{self.domain_b}] → {self.name}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "domain_a": self.domain_a,
            "domain_b": self.domain_b,
            "blended_properties": self.blended_properties,
            "novel_inferences": self.novel_inferences,
            "confidence": round(self.confidence, 3),
            "accepted": self.accepted,
            "expression": self.blend_expression(),
            "mapping_score": round(self.mapping.score, 3),
        }


# ── Seed input spaces ──────────────────────────────────────────────────────────

_SEED_SPACES: List[dict] = [
    # arithmetic
    {"concept": "addition", "domain": "arithmetic",
     "properties": {"operation": "combine", "identity": "0", "inverse": "subtraction", "commutativity": True},
     "relations": ["a + 0 = a", "a + b = b + a"],
     "examples": ["2 + 3 = 5", "x + 0 = x"]},
    {"concept": "multiplication", "domain": "arithmetic",
     "properties": {"operation": "scale", "identity": "1", "inverse": "division", "commutativity": True},
     "relations": ["a * 1 = a", "a * 0 = 0"],
     "examples": ["3 * 4 = 12", "x * 1 = x"]},
    # logic
    {"concept": "conjunction", "domain": "logic",
     "properties": {"operation": "combine", "identity": "TRUE", "inverse": "disjunction", "commutativity": True},
     "relations": ["A AND TRUE = A", "A AND FALSE = FALSE"],
     "examples": ["P AND Q", "TRUE AND FALSE = FALSE"]},
    {"concept": "disjunction", "domain": "logic",
     "properties": {"operation": "combine", "identity": "FALSE", "inverse": "conjunction", "commutativity": True},
     "relations": ["A OR FALSE = A", "A OR TRUE = TRUE"],
     "examples": ["P OR Q", "FALSE OR TRUE = TRUE"]},
    # physics
    {"concept": "electric_current", "domain": "physics",
     "properties": {"operation": "flow", "driving_force": "voltage", "resistance": "impedance", "commutativity": False},
     "relations": ["I is driven by V", "I decreases with higher R"],
     "examples": ["Ohm's law: I = V/R", "Series/parallel current behavior"]},
    {"concept": "electric_resistance", "domain": "physics",
     "properties": {"operation": "impede", "identity": "0", "inverse": "conductance", "commutativity": False},
     "relations": ["Higher R reduces I", "R combines in networks"],
     "examples": ["R_total = R1 + R2 (series)", "R_total = (1/R1 + 1/R2)^-1 (parallel)"]},
]

# ── ConceptBlender ─────────────────────────────────────────────────────────────

class ConceptBlender:
    """
    Cross-domain novel concept synthesis via conceptual blending.

    Core steps:
      1) represent concepts as InputSpaces
      2) compute BlendMapping overlap in structural roles
      3) project + merge compatible properties
      4) generate novel inferences from emergent role interactions
    """

    def __init__(self, seed_spaces: Optional[List[dict]] = None):
        self._seed_spaces = seed_spaces if seed_spaces is not None else _SEED_SPACES
        self._input_spaces: List[InputSpace] = [self._space_from_dict(d) for d in self._seed_spaces]

    def _space_from_dict(self, d: dict) -> InputSpace:
        return InputSpace(
            concept=d["concept"],
            domain=d["domain"],
            properties=dict(d.get("properties", {})),
            relations=list(d.get("relations", [])),
            examples=list(d.get("examples", [])),
        )

    def discover_blends(
        self,
        input_spaces: Optional[List[InputSpace]] = None,
        min_score: float = 0.25,
        max_results: int = 20,
    ) -> List[ConceptBlend]:
        spaces = input_spaces if input_spaces is not None else self._input_spaces
        results: List[ConceptBlend] = []
        n = len(spaces)
        if n <= 1:
            return results

        # Precompute structural feature sets to avoid repeated allocation in the overlap loop.
        # (The InputSpace caches internally, but structural_features() returns a new set.)
        feature_sets: List[frozenset[str]] = []
        for s in spaces:
            # Access the internal cached immutable structure if available.
            cache = getattr(s, "_structural_features_cache", None)
            if isinstance(cache, frozenset):
                feature_sets.append(cache)
            else:
                feature_sets.append(frozenset(s.properties.keys()))

        for i in range(n):
            for j in range(i + 1, n):
                sa = spaces[i]
                sb = spaces[j]

                overlap_roles, overlap_score = self._compute_overlap_from_feature_sets(
                    feature_sets[i], feature_sets[j]
                )
                if overlap_score < min_score:
                    continue

                mapping = BlendMapping(space_a=sa, space_b=sb, shared_roles=overlap_roles, score=overlap_score)
                blend = self._blend(sa, sb, mapping)
                results.append(blend)

        results.sort(key=lambda b: (b.confidence, b.mapping.score), reverse=True)
        return results[:max_results]

    @staticmethod
    def _compute_overlap_from_feature_sets(
        features_a: frozenset[str],
        features_b: frozenset[str],
    ) -> Tuple[List[str], float]:
        # Approved change target: "blend overlap computation (mult ...)".
        # Use a stable, symmetric similarity measure with safe zero-division:
        #   score = |A ∩ B| / |A ∪ B|
        shared = features_a.intersection(features_b)
        union = features_a.union(features_b)
        if not union:
            return ([], 0.0)
        score = len(shared) / len(union)
        # Deterministic ordering
        shared_roles = sorted(shared)
        return (shared_roles, score)

    def _blend(self, space_a: InputSpace, space_b: InputSpace, mapping: BlendMapping) -> ConceptBlend:
        # Project properties by shared roles; take A's value unless missing,
        # otherwise B's. For non-shared roles, include a conservative merge:
        # only include when it doesn't conflict.
        blended: Dict[str, Any] = {}
        roles = mapping.shared_roles

        for role in roles:
            a_has = role in space_a.properties
            b_has = role in space_b.properties
            if a_has and not b_has:
                blended[role] = space_a.properties[role]
            elif b_has and not a_has:
                blended[role] = space_b.properties[role]
            else:
                # both have; keep A's unless different and B is more "specific" by type
                av = space_a.properties[role]
                bv = space_b.properties[role]
                if av == bv:
                    blended[role] = av
                else:
                    # Prefer non-bool / non-sentinel more informative value
                    # (heuristic, deterministic)
                    if isinstance(av, bool) and not isinstance(bv, bool):
                        blended[role] = bv
                    elif isinstance(bv, bool) and not isinstance(av, bool):
                        blended[role] = av
                    else:
                        blended[role] = av

        # Add emergent property derived from "driving_force" if present in either.
        # This is intentionally generic, to support emergent novel inferences.
        emergent: List[str] = []

        def _find_prop(space: InputSpace, keys: List[str]) -> Optional[str]:
            for k in keys:
                if k in space.properties:
                    return str(space.properties[k])
            return None

        driving_a = _find_prop(space_a, ["driving_force", "driver", "cause"])
        driving_b = _find_prop(space_b, ["driving_force", "driver", "cause"])
        resistance_a = _find_prop(space_a, ["resistance", "impedance", "friction"])
        resistance_b = _find_prop(space_b, ["resistance", "impedance", "friction"])
        identity_a = _find_prop(space_a, ["identity", "neutral_element"])
        identity_b = _find_prop(space_b, ["identity", "neutral_element"])

        if (driving_a or driving_b) and (resistance_a or resistance_b):
            d = driving_a or driving_b
            r = resistance_a or resistance_b
            emergent.append(f"Higher resistance ('{r}') suppresses flow driven by '{d}'.")
            emergent.append("The blended system links force/driver directly to outcome via impediment.")

        if identity_a and identity_b and identity_a != identity_b:
            emergent.append(f"The blend contrasts identities ('{identity_a}' vs '{identity_b}'), implying different neutral behaviors.")

        # Name the blend deterministically.
        # Use short domain prefixes to keep names compact.
        domain_a_short = space_a.domain.split("_")[0][:4]
        domain_b_short = space_b.domain.split("_")[0][:4]
        name = f"{space_a.concept[:10]}_{space_b.concept[:10]}_blend"

        # Confidence grows with structural overlap and presence of some emergent cues.
        base = mapping.score
        bonus = 0.15 if emergent else 0.0
        comm_bonus = 0.05 if (space_a.properties.get("commutativity") and space_b.properties.get("commutativity")) else 0.0
        confidence = max(0.0, min(1.0, 0.3 + 0.7 * base + bonus + comm_bonus))

        return ConceptBlend(
            name=name,
            source_a=space_a.concept,
            source_b=space_b.concept,
            domain_a=space_a.domain,
            domain_b=space_b.domain,
            blended_properties=blended,
            novel_inferences=emergent if emergent else [
                "A new interaction rule emerges: the blend projects shared roles into a fresh dependency pattern.",
                "The blend suggests an inference path not explicitly present in either source concept.",
            ],
            mapping=mapping,
            confidence=confidence,
            accepted=False,
        )

    def summary(self, blends: List[ConceptBlend], max_lines: int = 10) -> str:
        blends = blends[:max_lines]
        lines: List[str] = []
        for b in blends:
            lines.append(
                f"- {b.name}: {b.source_a}[{b.domain_a}] ⊕ {b.source_b}[{b.domain_b}] "
                f"(overlap={b.mapping.score:.2f}, conf={b.confidence:.2f})"
            )
        return "\n".join(lines)


# ── Module-level singleton helper (optional) ────────────────────────────────────

_default_blender: Optional[ConceptBlender] = None

def get_concept_blender() -> ConceptBlender:
    global _default_blender
    if _default_blender is None:
        _default_blender = ConceptBlender()
    return _default_blender