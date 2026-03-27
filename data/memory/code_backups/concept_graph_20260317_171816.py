"""
ConceptGraph — The missing biological intelligence layer for SARE-HX.

Biological intelligence architecture:
  Perception → Concept Formation → Symbolic Reasoning → Planning → Action

SARE previously started at Symbolic Reasoning.
This module adds the Concept Formation layer between Perception and Symbols.

A concept is:
  - A grounded abstraction learned from concrete observations
  - "addition" = {meaning: "combine quantities", examples: ["3+2=5", "apples+apples"]}
  - Linked to symbolic rules (x + 0 = x) AND to grounded examples (3 apples + 0 apples = 3 apples)

Usage:
    cg = ConceptGraph()
    cg.ground_example("addition", "3 apples + 2 apples = 5 apples", {"domain": "arithmetic"})
    cg.abstract_from_examples("addition")   # → generates symbolic rule candidates
    cg.to_symbol("addition")                # → "+"
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

_PERSIST_PATH = Path(__file__).parent.parent.parent.parent / "data" / "memory" / "concept_graph.json"


# ── Concept dataclass ──────────────────────────────────────────────────────────

@dataclass
class ConceptExample:
    """A grounded, concrete instance of a concept."""
    text: str               # e.g. "3 apples + 0 apples = 3 apples"
    objects: List[str]      # e.g. ["apple"]
    operation: str          # e.g. "add"
    inputs: List[Any]       # e.g. [3, 0]
    result: Any             # e.g. 3
    domain: str             # e.g. "arithmetic"
    symbolic: str           # e.g. "x + 0 = x"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "text": self.text, "objects": self.objects, "operation": self.operation,
            "inputs": self.inputs, "result": self.result, "domain": self.domain,
            "symbolic": self.symbolic,
        }


@dataclass
class Concept:
    """
    A concept node in the ConceptGraph.

    Bridges the gap between grounded experience and symbolic rules.
    A child learns "addition" by seeing 3+2=5 many times before the symbol '+'.
    """
    name: str                               # e.g. "addition"
    meaning: str                            # e.g. "combine quantities"
    symbol: str                             # e.g. "+"
    domain: str                             # e.g. "arithmetic"
    examples: List[ConceptExample] = field(default_factory=list)
    related: List[str] = field(default_factory=list)  # other concept names
    symbolic_rules: List[str] = field(default_factory=list)  # e.g. ["x + 0 = x"]
    confidence: float = 0.5
    use_count: int = 0
    created_at: float = field(default_factory=time.time)

    def add_example(self, ex: ConceptExample):
        self.examples.append(ex)
        self.use_count += 1
        self.confidence = min(0.99, 0.3 + 0.05 * len(self.examples))

    def ground_count(self) -> int:
        return len(self.examples)

    def is_well_grounded(self) -> bool:
        return len(self.examples) >= 3 and self.confidence >= 0.4

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "meaning": self.meaning,
            "symbol": self.symbol,
            "domain": self.domain,
            "examples": [e.to_dict() for e in self.examples[-5:]],  # last 5
            "related": self.related,
            "symbolic_rules": self.symbolic_rules,
            "confidence": round(self.confidence, 3),
            "use_count": self.use_count,
            "well_grounded": self.is_well_grounded(),
        }


# ── Seed concept library ───────────────────────────────────────────────────────

_SEED_CONCEPTS: List[dict] = [
    {
        "name": "addition", "meaning": "combine quantities to get a larger total",
        "symbol": "+", "domain": "arithmetic",
        "related": ["subtraction", "identity_addition", "commutativity"],
        "symbolic_rules": ["x + 0 = x", "x + y = y + x", "(x + y) + z = x + (y + z)"],
    },
    {
        "name": "subtraction", "meaning": "remove a quantity from another",
        "symbol": "-", "domain": "arithmetic",
        "related": ["addition", "negation"],
        "symbolic_rules": ["x - 0 = x", "x - x = 0"],
    },
    {
        "name": "multiplication", "meaning": "repeated addition of equal groups",
        "symbol": "*", "domain": "arithmetic",
        "related": ["addition", "identity_multiplication", "distribution"],
        "symbolic_rules": ["x * 1 = x", "x * 0 = 0", "x * y = y * x"],
    },
    {
        "name": "identity_addition", "meaning": "adding zero changes nothing",
        "symbol": "0", "domain": "arithmetic",
        "related": ["addition", "identity_multiplication"],
        "symbolic_rules": ["x + 0 = x"],
    },
    {
        "name": "identity_multiplication", "meaning": "multiplying by one changes nothing",
        "symbol": "1", "domain": "arithmetic",
        "related": ["multiplication", "identity_addition"],
        "symbolic_rules": ["x * 1 = x"],
    },
    {
        "name": "annihilation", "meaning": "multiplying by zero collapses to zero",
        "symbol": "0", "domain": "arithmetic",
        "related": ["multiplication", "identity_multiplication"],
        "symbolic_rules": ["x * 0 = 0"],
    },
    {
        "name": "negation", "meaning": "flip the truth value of a statement",
        "symbol": "¬", "domain": "logic",
        "related": ["double_negation", "conjunction"],
        "symbolic_rules": ["¬¬x = x"],
    },
    {
        "name": "double_negation", "meaning": "negating twice returns the original",
        "symbol": "¬¬", "domain": "logic",
        "related": ["negation", "involution"],
        "symbolic_rules": ["¬¬x = x"],
    },
    {
        "name": "conjunction", "meaning": "both conditions must hold",
        "symbol": "∧", "domain": "logic",
        "related": ["negation", "identity_conjunction"],
        "symbolic_rules": ["x ∧ true = x", "x ∧ false = false"],
    },
    {
        "name": "differentiation", "meaning": "rate of change of a function",
        "symbol": "d/dx", "domain": "calculus",
        "related": ["integration", "chain_rule"],
        "symbolic_rules": ["d/dx(c) = 0", "d/dx(x) = 1", "d/dx(x^n) = n*x^(n-1)"],
    },
    {
        "name": "integration", "meaning": "accumulated area under a function",
        "symbol": "∫", "domain": "calculus",
        "related": ["differentiation", "sum_rule_integration"],
        "symbolic_rules": ["∫c dx = cx", "∫x dx = x²/2"],
    },
]


# ── ConceptGraph ───────────────────────────────────────────────────────────────

class ConceptGraph:
    """
    The Concept Formation layer of SARE-HX.

    Sits between Perception (raw input) and Symbolic Reasoning (transforms).
    Bridges grounded experience ("3 apples + 0 apples") to abstract rules ("x + 0 = x").
    """

    def __init__(self, seed: bool = True):
        self._concepts: Dict[str, Concept] = {}
        self._symbol_to_concept: Dict[str, List[str]] = defaultdict(list)
        self._domain_concepts: Dict[str, Set[str]] = defaultdict(set)
        self._abstraction_count = 0

        if seed:
            self._seed_library()

        self._try_load()

    def _seed_library(self):
        """Seed the concept graph with known mathematical concepts."""
        for spec in _SEED_CONCEPTS:
            c = Concept(
                name=spec["name"],
                meaning=spec["meaning"],
                symbol=spec["symbol"],
                domain=spec["domain"],
                related=spec.get("related", []),
                symbolic_rules=spec.get("symbolic_rules", []),
                confidence=0.9,
                use_count=0,
            )
            self._concepts[c.name] = c
            self._symbol_to_concept[c.symbol].append(c.name)
            self._domain_concepts[c.domain].add(c.name)

    # ── Public API ─────────────────────────────────────────────────────────────

    def ground_example(
        self,
        concept_name: str,
        text: str,
        objects: Optional[List[str]] = None,
        operation: Optional[str] = None,
        inputs: Optional[List[Any]] = None,
        result: Any = None,
        domain: str = "arithmetic",
        symbolic: str = "",
    ) -> ConceptExample:
        """
        Add a grounded concrete example to a concept.

        Example:
            cg.ground_example("addition",
                text="3 apples + 0 apples = 3 apples",
                objects=["apple"], operation="add",
                inputs=[3, 0], result=3,
                symbolic="x + 0 = x")
        """
        ex = ConceptExample(
            text=text,
            objects=objects or [],
            operation=operation or concept_name,
            inputs=inputs or [],
            result=result,
            domain=domain,
            symbolic=symbolic,
        )

        if concept_name not in self._concepts:
            # Auto-create concept if not seeded
            self._concepts[concept_name] = Concept(
                name=concept_name, meaning=f"learned from observation: {text[:40]}",
                symbol="?", domain=domain,
            )

        self._concepts[concept_name].add_example(ex)
        log.debug(f"Grounded '{concept_name}': {text[:50]}")
        return ex

    def abstract_from_examples(self, concept_name: str) -> List[str]:
        """
        Given accumulated examples for a concept, generate symbolic rule candidates.
        Returns list of candidate rule strings.
        """
        concept = self._concepts.get(concept_name)
        if not concept or not concept.examples:
            return []

        candidates = []
        # Detect identity pattern: same inputs, one is 0, result == non-zero input
        zero_identity = [
            e for e in concept.examples
            if len(e.inputs) == 2 and 0 in e.inputs and e.result == max(e.inputs)
        ]
        if zero_identity:
            candidates.append(f"x {concept.symbol} 0 = x")
            if f"x {concept.symbol} 0 = x" not in concept.symbolic_rules:
                concept.symbolic_rules.append(f"x {concept.symbol} 0 = x")

        # Detect commutativity: multiple examples where swapping inputs gives same result
        commute = [
            e for e in concept.examples
            if len(e.inputs) == 2 and e.inputs[0] != e.inputs[1]
        ]
        if len(commute) >= 2:
            a_then_b = [e for e in commute]
            b_then_a = [(e.inputs[1], e.inputs[0], e.result) for e in commute]
            # check if a+b == b+a in examples
            if any(
                (e.inputs[1], e.inputs[0], e.result) in [(ex.inputs[0], ex.inputs[1], ex.result) for ex in commute]
                for e in commute
            ):
                rule = f"x {concept.symbol} y = y {concept.symbol} x"
                candidates.append(rule)
                if rule not in concept.symbolic_rules:
                    concept.symbolic_rules.append(rule)

        self._abstraction_count += 1
        return candidates

    def ground_solve_episode(
        self,
        expression: str,
        result: str,
        transforms_used: List[str],
        domain: str,
        delta: float,
    ):
        """
        Called after each successful solve to auto-ground concepts.
        Maps transform names → concept names → adds examples.
        """
        _TRANSFORM_TO_CONCEPT = {
            "add_zero_elim": "identity_addition",
            "mul_one_elim": "identity_multiplication",
            "mul_zero_annihilate": "annihilation",
            "double_negation": "double_negation",
            "double_negation_logic": "double_negation",
            "bool_and_true": "conjunction",
            "bool_or_false": "conjunction",
            "trig_zero": "differentiation",
            "derivative": "differentiation",
            "integral": "integration",
        }
        for t_name in transforms_used:
            concept_name = _TRANSFORM_TO_CONCEPT.get(t_name)
            if not concept_name:
                # Try partial match
                for key, val in _TRANSFORM_TO_CONCEPT.items():
                    if key in t_name or t_name in key:
                        concept_name = val
                        break
            if concept_name:
                self.ground_example(
                    concept_name=concept_name,
                    text=f"{expression} → {result} (via {t_name})",
                    operation=t_name,
                    inputs=[expression],
                    result=result,
                    domain=domain,
                    symbolic=f"{expression} simplifies to {result}",
                )

    def to_symbol(self, concept_name: str) -> Optional[str]:
        """Translate a concept name to its primary symbol."""
        c = self._concepts.get(concept_name)
        return c.symbol if c else None

    def from_symbol(self, symbol: str) -> List[str]:
        """Find all concepts associated with a symbol."""
        return self._symbol_to_concept.get(symbol, [])

    def explain_symbol(self, symbol: str) -> Optional[str]:
        """Return human-readable meaning for a symbol."""
        names = self.from_symbol(symbol)
        if names and names[0] in self._concepts:
            return self._concepts[names[0]].meaning
        return None

    def get_domain_concepts(self, domain: str) -> List[Concept]:
        """Get all concepts for a domain."""
        names = self._domain_concepts.get(domain, set())
        return [self._concepts[n] for n in names if n in self._concepts]

    def concept_for_transform(self, transform_name: str) -> Optional[Concept]:
        """Find the concept most relevant to a transform name."""
        for name, concept in self._concepts.items():
            if (transform_name.lower() in name.lower() or
                    name.lower() in transform_name.lower() or
                    any(transform_name.lower() in r.lower() for r in concept.symbolic_rules)):
                return concept
        return None

    def get_well_grounded(self) -> List[Concept]:
        """Return all concepts that have at least 3 grounded examples."""
        return [c for c in self._concepts.values() if c.is_well_grounded()]

    # ── Cross-Domain Concept Formation ─────────────────────────────────────────

    # Structural roles shared across domains
    _STRUCTURAL_ROLES: List[dict] = [
        {
            "role": "identity",
            "description": "doing X changes nothing",
            "matches": [
                ("arithmetic", ["x + 0 = x", "x * 1 = x"]),
                ("logic",      ["x ∧ true = x", "x ∨ false = x", "p and true"]),
                ("calculus",   ["d/dx(c) = 0"]),
            ],
        },
        {
            "role": "annihilation",
            "description": "one element collapses everything to a fixed point",
            "matches": [
                ("arithmetic", ["x * 0 = 0"]),
                ("logic",      ["x ∧ false = false", "x ∨ true = true"]),
            ],
        },
        {
            "role": "involution",
            "description": "applying twice returns to start",
            "matches": [
                ("logic",      ["¬¬x = x"]),
                ("arithmetic", ["--x = x", "neg neg x"]),
                ("calculus",   ["integral(derivative(f)) = f"]),
            ],
        },
        {
            "role": "commutativity",
            "description": "order does not matter",
            "matches": [
                ("arithmetic", ["x + y = y + x", "x * y = y * x"]),
                ("logic",      ["x ∧ y = y ∧ x", "x ∨ y = y ∨ x"]),
            ],
        },
        {
            "role": "distributivity",
            "description": "operation distributes over another",
            "matches": [
                ("arithmetic", ["x * (y + z) = x*y + x*z"]),
                ("logic",      ["x ∧ (y ∨ z) = (x∧y) ∨ (x∧z)"]),
            ],
        },
    ]

    def cross_domain_analogies(self) -> List[dict]:
        """
        Detect cross-domain structural analogies between concepts.

        Example: 'x + 0 = x' (arithmetic identity) is structurally
        analogous to 'p ∧ true = p' (logic identity).

        Returns list of analogy dicts with source, target, role, shared_rule.
        """
        analogies: List[dict] = []

        for role_spec in self._STRUCTURAL_ROLES:
            role = role_spec["role"]
            # Find concepts in each domain that match this role
            domain_matches: List[Tuple[str, str, str]] = []  # (domain, concept_name, rule)
            for domain, rule_fragments in role_spec["matches"]:
                for name, concept in self._concepts.items():
                    if concept.domain != domain:
                        continue
                    for rule in concept.symbolic_rules:
                        for frag in rule_fragments:
                            if frag in rule or rule in frag:
                                domain_matches.append((domain, name, rule))
                                break

            # Generate cross-domain analogy pairs
            for i, (d1, n1, r1) in enumerate(domain_matches):
                for d2, n2, r2 in domain_matches[i+1:]:
                    if d1 != d2:  # different domains = cross-domain analogy
                        analogies.append({
                            "role": role,
                            "description": role_spec["description"],
                            "source_domain": d1,
                            "source_concept": n1,
                            "source_rule": r1,
                            "target_domain": d2,
                            "target_concept": n2,
                            "target_rule": r2,
                        })

        return analogies

    def merge_concepts(self, c1_name: str, c2_name: str,
                       merged_name: str = None,
                       role: str = "cross_domain") -> Optional[Concept]:
        """
        Merge two concepts from different domains into a shared abstract concept.

        Example: identity_addition (arithmetic) + identity_conjunction (logic)
                 → abstract_identity: "doing X changes nothing — across all domains"

        Returns the merged Concept (added to the graph) or None if already exists.
        """
        c1 = self._concepts.get(c1_name)
        c2 = self._concepts.get(c2_name)
        if not c1 or not c2:
            return None

        merged_name = merged_name or f"abstract_{role}_{c1_name[:6]}_{c2_name[:6]}"
        if merged_name in self._concepts:
            # Enrich existing merged concept
            merged = self._concepts[merged_name]
        else:
            # Create new abstract merged concept
            merged = Concept(
                name=merged_name,
                meaning=f"Abstract {role}: {c1.meaning} (like {c2.meaning})",
                symbol=f"{c1.symbol}/{c2.symbol}",
                domain="cross_domain",
                related=[c1_name, c2_name],
                symbolic_rules=[],
                confidence=min(c1.confidence, c2.confidence) * 0.9,
            )
            self._concepts[merged_name] = merged
            self._domain_concepts["cross_domain"].add(merged_name)
            log.info(f"ConceptGraph: merged '{c1_name}' + '{c2_name}' → '{merged_name}'")

        # Merge symbolic rules (deduplicate)
        for rule in c1.symbolic_rules + c2.symbolic_rules:
            if rule not in merged.symbolic_rules:
                merged.symbolic_rules.append(rule)

        # Cross-link both source concepts to the merged one
        for c in (c1, c2):
            if merged_name not in c.related:
                c.related.append(merged_name)

        return merged

    def run_cross_domain_merge(self) -> int:
        """
        Auto-detect cross-domain analogies and create merged abstract concepts.
        Called by Brain._consolidate_concepts every N cycles.
        Returns number of new merges performed.
        """
        analogies = self.cross_domain_analogies()
        merged = set()
        count = 0

        for analogy in analogies:
            n1 = analogy["source_concept"]
            n2 = analogy["target_concept"]
            role = analogy["role"]
            pair_key = tuple(sorted([n1, n2]))
            if pair_key in merged:
                continue
            merged.add(pair_key)
            result = self.merge_concepts(n1, n2, role=role)
            if result:
                count += 1

        return count

    def summary(self) -> dict:
        cross_domain = len(self._domain_concepts.get("cross_domain", set()))
        return {
            "total_concepts": len(self._concepts),
            "well_grounded": len(self.get_well_grounded()),
            "total_examples": sum(c.ground_count() for c in self._concepts.values()),
            "domains": {d: len(names) for d, names in self._domain_concepts.items()},
            "abstractions_performed": self._abstraction_count,
            "cross_domain_concepts": cross_domain,
            "analogies_detected": len(self.cross_domain_analogies()),
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self):
        try:
            _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {name: c.to_dict() for name, c in self._concepts.items()}
            _PERSIST_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.debug(f"ConceptGraph save failed: {e}")

    def _try_load(self):
        try:
            if _PERSIST_PATH.exists():
                data = json.loads(_PERSIST_PATH.read_text())
                for name, spec in data.items():
                    if name in self._concepts:
                        # Merge examples only
                        c = self._concepts[name]
                        c.use_count = spec.get("use_count", 0)
                        c.confidence = spec.get("confidence", c.confidence)
                        for ex_dict in spec.get("examples", []):
                            c.examples.append(ConceptExample(
                                text=ex_dict.get("text", ""),
                                objects=ex_dict.get("objects", []),
                                operation=ex_dict.get("operation", ""),
                                inputs=ex_dict.get("inputs", []),
                                result=ex_dict.get("result"),
                                domain=ex_dict.get("domain", "general"),
                                symbolic=ex_dict.get("symbolic", ""),
                            ))
        except Exception:
            pass
