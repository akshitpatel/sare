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


@dataclass
class ConceptExample:
    """A grounded, concrete instance of a concept."""
    text: str
    objects: List[str]
    operation: str
    inputs: List[Any]
    result: Any
    domain: str
    symbolic: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "objects": self.objects,
            "operation": self.operation,
            "inputs": self.inputs,
            "result": self.result,
            "domain": self.domain,
            "symbolic": self.symbolic,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptExample":
        return cls(
            text=data.get("text", ""),
            objects=list(data.get("objects", [])),
            operation=data.get("operation", ""),
            inputs=list(data.get("inputs", [])),
            result=data.get("result"),
            domain=data.get("domain", ""),
            symbolic=data.get("symbolic", ""),
            timestamp=float(data.get("timestamp", time.time())),
        )


@dataclass
class Concept:
    """
    A concept node in the ConceptGraph.

    Bridges the gap between grounded experience and symbolic rules.
    A child learns "addition" by seeing 3+2=5 many times before the symbol '+'.
    """
    name: str
    meaning: str
    symbol: str
    domain: str
    examples: List[ConceptExample] = field(default_factory=list)
    related: List[str] = field(default_factory=list)
    symbolic_rules: List[str] = field(default_factory=list)
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
            "examples": [e.to_dict() for e in self.examples[-5:]],
            "related": self.related,
            "symbolic_rules": self.symbolic_rules,
            "confidence": round(self.confidence, 3),
            "use_count": self.use_count,
            "created_at": self.created_at,
            "well_grounded": self.is_well_grounded(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Concept":
        examples = [ConceptExample.from_dict(e) for e in data.get("examples", [])]
        return cls(
            name=data.get("name", ""),
            meaning=data.get("meaning", ""),
            symbol=data.get("symbol", ""),
            domain=data.get("domain", ""),
            examples=examples,
            related=list(data.get("related", [])),
            symbolic_rules=list(data.get("symbolic_rules", [])),
            confidence=float(data.get("confidence", 0.5)),
            use_count=int(data.get("use_count", len(examples))),
            created_at=float(data.get("created_at", time.time())),
        )


_SEED_CONCEPTS: List[dict] = [
    {
        "name": "addition",
        "meaning": "combine quantities to get a larger total",
        "symbol": "+",
        "domain": "arithmetic",
        "related": ["subtraction", "identity_addition", "commutativity"],
        "symbolic_rules": ["x + 0 = x", "x + y = y + x", "(x + y) + z = x + (y + z)"],
    },
    {
        "name": "subtraction",
        "meaning": "remove a quantity from another",
        "symbol": "-",
        "domain": "arithmetic",
        "related": ["addition", "negation"],
        "symbolic_rules": ["x - 0 = x", "x - x = 0"],
    },
    {
        "name": "multiplication",
        "meaning": "repeated addition of equal groups",
        "symbol": "*",
        "domain": "arithmetic",
        "related": ["addition", "identity_multiplication", "distribution"],
        "symbolic_rules": ["x * 1 = x", "x * 0 = 0", "x * y = y * x"],
    },
    {
        "name": "identity_addition",
        "meaning": "adding zero changes nothing",
        "symbol": "0",
        "domain": "arithmetic",
        "related": ["addition", "identity_multiplication"],
        "symbolic_rules": ["x + 0 = x"],
    },
    {
        "name": "identity_multiplication",
        "meaning": "multiplying by one changes nothing",
        "symbol": "1",
        "domain": "arithmetic",
        "related": ["multiplication", "identity_addition"],
        "symbolic_rules": ["x * 1 = x"],
    },
    {
        "name": "annihilation",
        "meaning": "multiplying by zero collapses to zero",
        "symbol": "0",
        "domain": "arithmetic",
        "related": ["multiplication", "identity_multiplication"],
        "symbolic_rules": ["x * 0 = 0"],
    },
    {
        "name": "negation",
        "meaning": "flip the truth value of a statement",
        "symbol": "¬",
        "domain": "logic",
        "related": ["double_negation", "conjunction"],
        "symbolic_rules": ["¬¬x = x"],
    },
    {
        "name": "double_negation",
        "meaning": "negating twice returns the original",
        "symbol": "¬¬",
        "domain": "logic",
        "related": ["negation", "involution"],
        "symbolic_rules": ["¬¬x = x"],
    },
    {
        "name": "conjunction",
        "meaning": "both conditions must hold",
        "symbol": "∧",
        "domain": "logic",
        "related": ["negation", "disjunction"],
        "symbolic_rules": ["x ∧ x = x"],
    },
    {
        "name": "disjunction",
        "meaning": "at least one condition must hold",
        "symbol": "∨",
        "domain": "logic",
        "related": ["conjunction", "negation"],
        "symbolic_rules": ["x ∨ x = x"],
    },
    {
        "name": "commutativity",
        "meaning": "order does not affect the result",
        "symbol": "↔",
        "domain": "algebra",
        "related": ["addition", "multiplication"],
        "symbolic_rules": ["x + y = y + x", "x * y = y * x"],
    },
    {
        "name": "distribution",
        "meaning": "one operation distributes over another",
        "symbol": "⊗",
        "domain": "algebra",
        "related": ["multiplication", "addition"],
        "symbolic_rules": ["x * (y + z) = x*y + x*z"],
    },
    {
        "name": "involution",
        "meaning": "applying an operation twice returns the original",
        "symbol": "⟲",
        "domain": "abstract",
        "related": ["double_negation"],
        "symbolic_rules": ["f(f(x)) = x"],
    },
]


class ConceptGraph:
    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = persist_path or _PERSIST_PATH
        self.concepts: Dict[str, Concept] = {}
        self.by_domain: Dict[str, Set[str]] = defaultdict(set)
        self._load_or_seed()

    def _load_or_seed(self) -> None:
        if self.persist_path.exists():
            try:
                with self.persist_path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                for entry in raw.get("concepts", []):
                    concept = Concept.from_dict(entry)
                    if concept.name:
                        self.concepts[concept.name] = concept
                        self.by_domain[concept.domain].add(concept.name)
                if self.concepts:
                    return
            except Exception as e:
                log.warning("Failed loading concept graph: %s", e)
        for entry in _SEED_CONCEPTS:
            concept = Concept(
                name=entry["name"],
                meaning=entry["meaning"],
                symbol=entry["symbol"],
                domain=entry["domain"],
                related=list(entry.get("related", [])),
                symbolic_rules=list(entry.get("symbolic_rules", [])),
            )
            self.concepts[concept.name] = concept
            self.by_domain[concept.domain].add(concept.name)
        self._persist()

    def _persist(self) -> None:
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self.persist_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {"concepts": [c.to_dict() for c in self.concepts.values()]},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            log.warning("Failed persisting concept graph: %s", e)

    def get(self, name: str) -> Optional[Concept]:
        return self.concepts.get(name)

    def all_concepts(self) -> List[Concept]:
        return list(self.concepts.values())

    def ensure_concept(
        self,
        name: str,
        meaning: str = "",
        symbol: str = "",
        domain: str = "general",
    ) -> Concept:
        if name in self.concepts:
            return self.concepts[name]
        concept = Concept(name=name, meaning=meaning or name, symbol=symbol, domain=domain)
        self.concepts[name] = concept
        self.by_domain[domain].add(name)
        self._persist()
        return concept

    def ground_example(self, concept_name: str, text: str, metadata: Dict[str, Any]) -> None:
        concept = self.ensure_concept(
            concept_name,
            meaning=metadata.get("meaning", concept_name),
            symbol=metadata.get("symbol", ""),
            domain=metadata.get("domain", "general"),
        )
        ex = ConceptExample(
            text=text,
            objects=list(metadata.get("objects", [])),
            operation=metadata.get("operation", concept.symbol or concept.name),
            inputs=list(metadata.get("inputs", [])),
            result=metadata.get("result"),
            domain=metadata.get("domain", concept.domain),
            symbolic=metadata.get("symbolic", ""),
        )
        concept.add_example(ex)
        for rel in metadata.get("related", []):
            if rel not in concept.related:
                concept.related.append(rel)
        self._persist()

    def abstract_from_examples(self, concept_name: str) -> List[str]:
        concept = self.get(concept_name)
        if not concept:
            return []
        candidates: Set[str] = set(concept.symbolic_rules)
        for ex in concept.examples:
            if ex.symbolic:
                candidates.add(ex.symbolic)
        concept.symbolic_rules = sorted(candidates)
        self._persist()
        return concept.symbolic_rules

    def to_symbol(self, concept_name: str) -> str:
        concept = self.get(concept_name)
        return concept.symbol if concept else ""

    def related_concepts(self, concept_name: str) -> List[str]:
        concept = self.get(concept_name)
        return list(concept.related) if concept else []

    def concepts_in_domain(self, domain: str) -> List[str]:
        return sorted(self.by_domain.get(domain, set()))

    def stats(self) -> Dict[str, Any]:
        return {
            "concept_count": len(self.concepts),
            "domains": {k: len(v) for k, v in self.by_domain.items()},
            "well_grounded": sum(1 for c in self.concepts.values() if c.is_well_grounded()),
            "examples": sum(len(c.examples) for c in self.concepts.values()),
        }

    def _rule_roles(self, rule: str) -> Set[str]:
        roles: Set[str] = set()
        r = rule.lower()
        if "=" in r:
            roles.add("equality")
        if "+" in r:
            roles.add("commutative_op")
        if "*" in r:
            roles.add("multiplicative_op")
        if "0" in r:
            roles.add("zero")
        if "1" in r:
            roles.add("one")
        if "x" in r and "y" in r:
            roles.add("binary_vars")
        if "x" in r and "z" in r:
            roles.add("ternary_pattern")
        if "¬" in rule or "not" in r:
            roles.add("negation")
        if "∧" in rule or "and" in r:
            roles.add("conjunction")
        if "∨" in rule or "or" in r:
            roles.add("disjunction")
        if "f(f(x))" in r or "¬¬" in rule:
            roles.add("involution")
        if "(" in r and ")" in r:
            roles.add("grouping")
        return roles

    def cross_domain_analogies(self) -> List[Dict[str, Any]]:
        concepts = list(self.concepts.values())
        role_to_concepts: Dict[str, List[Tuple[str, str, Set[str]]]] = defaultdict(list)

        for concept in concepts:
            concept_roles: Set[str] = set()
            for rule in concept.symbolic_rules:
                concept_roles.update(self._rule_roles(rule))
            if not concept_roles:
                continue
            for role in concept_roles:
                role_to_concepts[role].append((concept.name, concept.domain, concept_roles))

        pair_roles: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for role, entries in role_to_concepts.items():
            n = len(entries)
            for i in range(n):
                name_a, domain_a, _roles_a = entries[i]
                for j in range(i + 1, n):
                    name_b, domain_b, _roles_b = entries[j]
                    if domain_a == domain_b:
                        continue
                    if name_a < name_b:
                        pair = (name_a, name_b)
                    else:
                        pair = (name_b, name_a)
                    pair_roles[pair].add(role)

        analogies: List[Dict[str, Any]] = []
        for (name_a, name_b), shared_roles in pair_roles.items():
            concept_a = self.concepts.get(name_a)
            concept_b = self.concepts.get(name_b)
            if not concept_a or not concept_b:
                continue
            analogies.append(
                {
                    "source": concept_a.name,
                    "source_domain": concept_a.domain,
                    "target": concept_b.name,
                    "target_domain": concept_b.domain,
                    "shared_roles": sorted(shared_roles),
                    "score": round(min(0.99, 0.2 + 0.15 * len(shared_roles)), 3),
                }
            )

        analogies.sort(key=lambda x: (-x["score"], x["source"], x["target"]))
        return analogies

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concepts": [c.to_dict() for c in sorted(self.concepts.values(), key=lambda c: c.name)],
            "stats": self.stats(),
        }