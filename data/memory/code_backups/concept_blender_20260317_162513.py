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
    concept:     str
    domain:      str
    properties:  Dict[str, Any]   # {property_name: value}
    relations:   List[str]        # ["A causes B", "X leads_to Y"]
    examples:    List[str]        # concrete instances

    def structural_features(self) -> Set[str]:
        """Abstract structural roles (keys only, not values)."""
        return set(self.properties.keys())

    def to_dict(self) -> dict:
        return {
            "concept":    self.concept,
            "domain":     self.domain,
            "properties": self.properties,
            "relations":  self.relations[:3],
            "examples":   self.examples[:2],
        }


@dataclass
class BlendMapping:
    """Structural correspondences between two input spaces."""
    space_a:      InputSpace
    space_b:      InputSpace
    shared_roles: List[str]       # property keys that map across both
    score:        float           # 0.0 (no overlap) → 1.0 (perfect structure)

    def to_dict(self) -> dict:
        return {
            "space_a":     self.space_a.concept,
            "space_b":     self.space_b.concept,
            "shared_roles": self.shared_roles,
            "score":       round(self.score, 3),
        }


@dataclass
class ConceptBlend:
    """
    A novel concept produced by blending two input spaces.

    The blend is NOT just a union; it selectively projects from both
    and adds emergent structure (novel_inferences) not in either input.
    """
    name:             str
    source_a:         str         # concept name from input space A
    source_b:         str         # concept name from input space B
    domain_a:         str
    domain_b:         str
    blended_properties: Dict[str, Any]
    novel_inferences: List[str]   # new facts that emerge from the blend
    mapping:          BlendMapping
    timestamp:        float = field(default_factory=time.time)
    confidence:       float = 0.6
    accepted:         bool  = False

    def blend_expression(self) -> str:
        """A symbolic expression summarising the blend."""
        return f"{self.source_a}[{self.domain_a}] ⊕ {self.source_b}[{self.domain_b}] → {self.name}"

    def to_dict(self) -> dict:
        return {
            "name":               self.name,
            "source_a":           self.source_a,
            "source_b":           self.source_b,
            "domain_a":           self.domain_a,
            "domain_b":           self.domain_b,
            "blended_properties": self.blended_properties,
            "novel_inferences":   self.novel_inferences,
            "confidence":         round(self.confidence, 3),
            "accepted":           self.accepted,
            "expression":         self.blend_expression(),
            "mapping_score":      round(self.mapping.score, 3),
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
     "properties": {"operation": "flow", "driving_force": "voltage", "resistance": "impedance", "rate": True},
     "relations": ["I = V/R", "high_resistance → low_current"],
     "examples": ["current through wire", "I = 5 A"]},
    {"concept": "water_flow", "domain": "physics",
     "properties": {"operation": "flow", "driving_force": "pressure", "resistance": "pipe_friction", "rate": True},
     "relations": ["Q = P/R", "high_resistance → low_flow"],
     "examples": ["water in pipe", "flow rate = 2 L/s"]},
    # calculus
    {"concept": "derivative", "domain": "calculus",
     "properties": {"operation": "rate_of_change", "identity": "constant→0", "inverse": "integral", "linearity": True},
     "relations": ["d/dx(constant) = 0", "d/dx(x^n) = n*x^(n-1)"],
     "examples": ["d/dx(x^2) = 2x", "d/dx(sin x) = cos x"]},
    {"concept": "integral", "domain": "calculus",
     "properties": {"operation": "accumulation", "identity": "zero_function", "inverse": "derivative", "linearity": True},
     "relations": ["∫0 dx = C", "∫x dx = x^2/2"],
     "examples": ["∫x^2 dx = x^3/3", "area under curve"]},
    # algebra
    {"concept": "group", "domain": "algebra",
     "properties": {"operation": "binary_op", "identity": "e", "inverse": "exists", "associativity": True},
     "relations": ["a * e = a", "(a*b)*c = a*(b*c)"],
     "examples": ["integers under +", "symmetry rotations"]},
    {"concept": "ring", "domain": "algebra",
     "properties": {"operation": "two_ops", "identity": "0_and_1", "inverse": "additive", "distributivity": True},
     "relations": ["a*(b+c) = a*b + a*c", "a+0 = a"],
     "examples": ["integers", "polynomials"]},
]


def _make_spaces() -> List[InputSpace]:
    spaces = []
    for s in _SEED_SPACES:
        spaces.append(InputSpace(
            concept    = s["concept"],
            domain     = s["domain"],
            properties = s["properties"],
            relations  = s["relations"],
            examples   = s["examples"],
        ))
    return spaces


# ── Novel inference templates ──────────────────────────────────────────────────

def _generate_novel_inferences(blend: dict, shared_roles: List[str],
                                 space_a: InputSpace, space_b: InputSpace) -> List[str]:
    """
    Generate novel inferences that emerge from the blend.
    These are statements true in the blended space but not in either input.
    """
    inferences = []
    # Pattern 1: identity cross-transfer
    id_a = space_a.properties.get("identity")
    id_b = space_b.properties.get("identity")
    if id_a and id_b and id_a != id_b:
        inferences.append(
            f"In the blend, {space_a.concept}({id_b}) behaves like "
            f"{space_b.concept}({id_a}) — cross-domain identity equivalence"
        )
    # Pattern 2: inverse cross-transfer
    inv_a = space_a.properties.get("inverse")
    inv_b = space_b.properties.get("inverse")
    if inv_a and inv_b and inv_a != inv_b:
        inferences.append(
            f"The inverse of {space_a.concept} in domain {space_b.domain} "
            f"corresponds to {inv_b}"
        )
    # Pattern 3: shared structural role → common law
    if "commutativity" in shared_roles:
        inferences.append(
            f"Both {space_a.concept} and {space_b.concept} are commutative — "
            f"the blend inherits order-independence"
        )
    if "operation" in shared_roles:
        op_a = space_a.properties.get("operation", "op")
        op_b = space_b.properties.get("operation", "op")
        if op_a == op_b:
            inferences.append(
                f"Both spaces share the '{op_a}' operation — "
                f"laws governing {op_a} in {space_a.domain} transfer to {space_b.domain}"
            )
    # Pattern 4: example cross-application
    if space_a.examples and space_b.examples:
        inferences.append(
            f"Example from {space_a.domain} '{space_a.examples[0]}' "
            f"maps structurally to '{space_b.examples[0]}' in {space_b.domain}"
        )
    return inferences[:4]


# ── ConceptBlender ─────────────────────────────────────────────────────────────

class ConceptBlender:
    """
    Discovers and creates novel concepts by blending input spaces.

    Usage::
        blender = ConceptBlender()
        blends  = blender.discover_blends()            # seed spaces
        blends  = blender.discover_blends(cg_spaces)   # from ConceptGraph
        blender.feed_to_concept_graph(cg)
    """

    MIN_MAPPING_SCORE = 0.25   # minimum structural overlap to attempt blend
    MAX_BLENDS        = 50     # total blends to keep

    def __init__(self):
        self._seed_spaces: List[InputSpace] = _make_spaces()
        self._extra_spaces: List[InputSpace] = []
        self._blends: List[ConceptBlend] = []
        self._total_attempts = 0
        self._total_accepted = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_space(self, concept: str, domain: str, properties: dict,
                  relations: List[str], examples: List[str]) -> InputSpace:
        """Add an external input space (e.g., from ConceptGraph node)."""
        sp = InputSpace(concept=concept, domain=domain,
                        properties=properties, relations=relations,
                        examples=examples)
        self._extra_spaces.append(sp)
        return sp

    def discover_blends(self, max_new: int = 5) -> List[ConceptBlend]:
        """
        Scan all pairs of input spaces from different domains.
        Return list of newly created blends.
        """
        all_spaces = self._seed_spaces + self._extra_spaces
        new_blends: List[ConceptBlend] = []
        existing_pairs: Set[Tuple[str, str]] = {
            (b.source_a, b.source_b) for b in self._blends
        }

        for i, spa in enumerate(all_spaces):
            for spb in all_spaces[i + 1:]:
                if spa.domain == spb.domain:
                    continue    # same domain → analogy, not blend
                pair = (spa.concept, spb.concept)
                if pair in existing_pairs:
                    continue
                mapping = self._compute_mapping(spa, spb)
                if mapping.score < self.MIN_MAPPING_SCORE:
                    continue
                blend = self._create_blend(spa, spb, mapping)
                self._blends.append(blend)
                new_blends.append(blend)
                existing_pairs.add(pair)
                if len(new_blends) >= max_new:
                    break
            if len(new_blends) >= max_new:
                break

        # Trim total blends to MAX
        if len(self._blends) > self.MAX_BLENDS:
            self._blends = sorted(
                self._blends, key=lambda b: b.mapping.score, reverse=True
            )[:self.MAX_BLENDS]
        return new_blends

    def blend_pair(self, concept_a: str, concept_b: str) -> Optional[ConceptBlend]:
        """Blend a specific pair by concept name."""
        all_spaces = self._seed_spaces + self._extra_spaces
        spa = next((s for s in all_spaces if s.concept == concept_a), None)
        spb = next((s for s in all_spaces if s.concept == concept_b), None)
        if not spa or not spb:
            return None
        mapping = self._compute_mapping(spa, spb)
        blend = self._create_blend(spa, spb, mapping)
        self._blends.append(blend)
        return blend

    def feed_to_concept_graph(self, concept_graph) -> int:
        """Push accepted blends into the ConceptGraph as cross-domain nodes."""
        fed = 0
        for blend in self._blends:
            if blend.accepted:
                continue
            try:
                concept_graph.add_concept(
                    name    = blend.name,
                    domain  = f"{blend.domain_a}+{blend.domain_b}",
                    meaning = "; ".join(blend.novel_inferences[:2]),
                    examples= [blend.blend_expression()],
                )
                blend.accepted = True
                self._total_accepted += 1
                fed += 1
            except Exception:
                pass
        return fed

    def summary(self) -> dict:
        accepted = [b for b in self._blends if b.accepted]
        domains_covered: Set[str] = set()
        for b in self._blends:
            domains_covered.add(b.domain_a)
            domains_covered.add(b.domain_b)
        return {
            "total_blends":    len(self._blends),
            "accepted_blends": len(accepted),
            "domains_covered": list(domains_covered),
            "seed_spaces":     len(self._seed_spaces),
            "extra_spaces":    len(self._extra_spaces),
            "recent_blends":   [b.to_dict() for b in self._blends[-6:]],
            "top_blends":      [b.to_dict() for b in
                                sorted(self._blends,
                                       key=lambda x: x.mapping.score,
                                       reverse=True)[:4]],
        }

    # ── Internal mechanics ────────────────────────────────────────────────────

    def _compute_mapping(self, spa: InputSpace, spb: InputSpace) -> BlendMapping:
        """Compute structural overlap between two input spaces."""
        roles_a = spa.structural_features()
        roles_b = spb.structural_features()
        shared  = sorted(roles_a & roles_b)
        score   = len(shared) / max(len(roles_a | roles_b), 1)
        # Bonus if both have same operation type
        if (spa.properties.get("operation") == spb.properties.get("operation")
                and spa.properties.get("operation")):
            score = min(1.0, score + 0.15)
        # Bonus if both are commutative
        if spa.properties.get("commutativity") and spb.properties.get("commutativity"):
            score = min(1.0, score + 0.10)
        return BlendMapping(space_a=spa, space_b=spb,
                            shared_roles=shared, score=score)

    def _create_blend(self, spa: InputSpace, spb: InputSpace,
                      mapping: BlendMapping) -> ConceptBlend:
        """Create a novel concept from two input spaces."""
        self._total_attempts += 1
        # Blend name: combine key terms
        name = f"{spa.concept}_{spb.domain}_blend"
        # Blended properties: union with cross-annotations
        blended = dict(spa.properties)
        for k, v in spb.properties.items():
            if k not in blended:
                blended[k] = v
            elif blended[k] != v:
                blended[f"{k}_({spb.domain})"] = v   # keep both
        # Novel inferences
        novel = _generate_novel_inferences(blended, mapping.shared_roles, spa, spb)
        confidence = 0.4 + mapping.score * 0.5   # higher mapping → more confident
        return ConceptBlend(
            name               = name,
            source_a           = spa.concept,
            source_b           = spb.concept,
            domain_a           = spa.domain,
            domain_b           = spb.domain,
            blended_properties = blended,
            novel_inferences   = novel,
            mapping            = mapping,
            confidence         = round(confidence, 3),
        )
