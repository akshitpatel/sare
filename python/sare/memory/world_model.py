"""
WorldModel — Human Imagination World Model for SARE-HX

A rich mental model that goes beyond flat fact storage.
Models how humans understand the world through:
  - Schemas (mental frames / templates)
  - Causal links (why things happen)
  - Mental simulation (trace consequences)
  - Counterfactual reasoning (what-if)
  - Contradiction detection (consistency)
  - Imagination (plausible extensions)
  - Analogy generation (structural parallels)

Persists to: data/memory/world_model_v2.json
Backwards-compatible with: data/memory/world_model.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def llm_available_check() -> bool:
    """Lazy check — avoids circular import at module level."""
    try:
        from sare.interface.llm_bridge import llm_available
        return llm_available()
    except Exception:
        return False

log = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Schema:
    """
    A mental frame / template for understanding a class of situations.
    Enhanced with v3 self-learning fields (structural_signature, invariants, decay).
    """
    name: str
    slots: Dict[str, str]                    # slot_name -> expected_type/concept (v2)
    constraints: List[str]                   # symbolic constraint strings
    examples: List[Tuple[str, float]]        # [(fingerprint, confidence)]
    activation: float = 0.0
    domain: str = "general"
    created_at: float = field(default_factory=time.time)
    # v3 additions (populated by SchemaInduction or LLM)
    structural_signature: str = ""           # md5 fingerprint of solve pattern
    slot_types: Dict[str, str] = field(default_factory=dict)  # role -> type
    invariants: List[str] = field(default_factory=list)       # discovered invariants
    exemplars: List[str] = field(default_factory=list)        # expression IDs
    confidence: float = 0.5
    use_count: int = 0

    def activate(self, amount: float = 0.2) -> None:
        self.activation = min(1.0, self.activation + amount)
        self.use_count += 1

    def decay(self, amount: float = 0.02) -> None:
        self.activation = max(0.0, self.activation - amount)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "slots": self.slots,
            "constraints": self.constraints,
            "examples": self.examples,
            "activation": round(self.activation, 4),
            "domain": self.domain,
            "created_at": self.created_at,
            "structural_signature": self.structural_signature,
            "slot_types": self.slot_types,
            "invariants": self.invariants,
            "exemplars": self.exemplars[-20:],
            "confidence": round(self.confidence, 4),
            "use_count": self.use_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        return cls(
            name=d["name"],
            slots=d.get("slots", {}),
            constraints=d.get("constraints", []),
            examples=[tuple(e) for e in d.get("examples", [])],
            activation=d.get("activation", 0.0),
            domain=d.get("domain", "general"),
            created_at=d.get("created_at", time.time()),
            structural_signature=d.get("structural_signature", ""),
            slot_types=d.get("slot_types", {}),
            invariants=d.get("invariants", []),
            exemplars=d.get("exemplars", []),
            confidence=d.get("confidence", 0.5),
            use_count=d.get("use_count", 0),
        )


@dataclass
class CausalLink:
    """
    A directional causal relationship: cause → effect, mediated by mechanism.
    Enhanced with staleness tracking and Bayesian update (merged from v3).
    """
    cause: str
    effect: str
    mechanism: str
    confidence: float = 0.9
    domain: str = "general"
    evidence_count: int = 1
    created_at: float = field(default_factory=time.time)
    last_observed: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        return f"{self.cause}→{self.effect}"

    @property
    def staleness(self) -> float:
        """0 = just seen, 1 = very stale (24+ hrs unobserved)."""
        hours = (time.time() - self.last_observed) / 3600
        return min(1.0, hours / 24.0)

    def observe(self, success: bool = True) -> None:
        """Bayesian-ish confidence update from a new observation."""
        self.evidence_count += 1
        self.last_observed = time.time()
        alpha = 1.0 / (self.evidence_count + 1)
        target = 0.95 if success else 0.2
        self.confidence = (1 - alpha) * self.confidence + alpha * target
        self.confidence = max(0.01, min(0.999, self.confidence))

    def to_dict(self) -> dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "mechanism": self.mechanism,
            "confidence": round(self.confidence, 4),
            "domain": self.domain,
            "evidence_count": self.evidence_count,
            "created_at": self.created_at,
            "last_observed": self.last_observed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CausalLink":
        return cls(
            cause=d["cause"],
            effect=d["effect"],
            mechanism=d["mechanism"],
            confidence=d.get("confidence", 0.9),
            domain=d.get("domain", "general"),
            evidence_count=d.get("evidence_count", 1),
            created_at=d.get("created_at", time.time()),
            last_observed=d.get("last_observed", d.get("created_at", time.time())),
        )


@dataclass
class Fact:
    """A simple world fact (backwards-compatible with world_model.json)."""
    domain: str
    fact: str
    confidence: float = 0.9
    source: str = "internal"
    timestamp: float = field(default_factory=time.time)
    contradicts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "fact": self.fact,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "timestamp": self.timestamp,
            "contradicts": self.contradicts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(
            domain=d["domain"],
            fact=d.get("fact", ""),
            confidence=d.get("confidence", 0.9),
            source=d.get("source", "internal"),
            timestamp=d.get("timestamp", time.time()),
            contradicts=d.get("contradicts", []),
        )


@dataclass
class Prediction:
    """A prediction made before an action, used to track surprise."""
    transform_name: str          # which transform was predicted to work
    expected_delta: float        # predicted energy reduction
    graph_signature: str         # fingerprint of the graph at prediction time
    domain: str = "general"
    confidence: float = 0.5      # how confident this prediction was
    timestamp: float = field(default_factory=time.time)
    # Filled in after outcome:
    actual_delta: float = 0.0
    was_correct: bool = False     # actual_delta within 30% of expected_delta
    surprise: float = 0.0        # |expected - actual| / max(expected, 0.1)

    def to_dict(self) -> dict:
        return {
            "transform_name": self.transform_name,
            "expected_delta": round(self.expected_delta, 4),
            "actual_delta": round(self.actual_delta, 4),
            "graph_signature": self.graph_signature,
            "domain": self.domain,
            "confidence": round(self.confidence, 4),
            "was_correct": self.was_correct,
            "surprise": round(self.surprise, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class Belief:
    """
    A tracked belief about a transform/domain/pattern with Bayesian evidence
    (merged from WorldModelV3).
    Dual use:
      - Structural belief about transforms: subject="algebra", predicate="dominant_transform", value="Distribute"
      - Factual triple from FactIngester:   subject="france", predicate="capital", value="Paris"
    """
    subject: str
    predicate: str
    confidence: float
    value: str = ""              # the object/answer (for factual triples)
    key: str = ""                # optional storage key (subject::predicate)
    evidence_for: int = 0
    evidence_against: int = 0
    domain: str = "general"
    last_updated: float = field(default_factory=time.time)
    valid_until: Optional[float] = None   # None = permanent; -1 = explicitly pinned; epoch = expiry

    def update(self, supports: bool) -> None:
        if supports:
            self.evidence_for += 1
        else:
            self.evidence_against += 1
        total = self.evidence_for + self.evidence_against
        self.confidence = self.evidence_for / max(total, 1)
        self.last_updated = time.time()

    @property
    def strength(self) -> str:
        if self.confidence > 0.9: return "certain"
        if self.confidence > 0.7: return "strong"
        if self.confidence > 0.4: return "moderate"
        if self.confidence > 0.2: return "weak"
        return "uncertain"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "domain": self.domain,
            "strength": self.strength,
            "last_updated": self.last_updated,
            "valid_until": self.valid_until,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Belief":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Analogy:
    """A discovered structural parallel between domains (merged from WorldModelV3)."""
    source_domain: str
    source_concept: str
    target_domain: str
    target_concept: str
    structural_mapping: Dict[str, str]
    confidence: float = 0.5
    discovered_at: float = field(default_factory=time.time)
    verification_count: int = 0
    verified: bool = False

    @property
    def key(self) -> str:
        return f"{self.source_domain}:{self.source_concept}↔{self.target_domain}:{self.target_concept}"

    def to_dict(self) -> dict:
        return {
            "source_domain": self.source_domain,
            "source_concept": self.source_concept,
            "target_domain": self.target_domain,
            "target_concept": self.target_concept,
            "structural_mapping": self.structural_mapping,
            "confidence": round(self.confidence, 4),
            "verified": self.verified,
            "verification_count": self.verification_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Analogy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Seed data ─────────────────────────────────────────────────────────────────

_SEED_CAUSAL_LINKS: List[dict] = [
    {"cause": "x + 0", "effect": "x", "mechanism": "zero is additive identity", "domain": "arithmetic", "confidence": 0.99},
    {"cause": "x * 1", "effect": "x", "mechanism": "one is multiplicative identity", "domain": "arithmetic", "confidence": 0.99},
    {"cause": "x * 0", "effect": "0", "mechanism": "zero annihilates multiplication", "domain": "arithmetic", "confidence": 0.99},
    {"cause": "not not x", "effect": "x", "mechanism": "double negation eliminates", "domain": "logic", "confidence": 0.99},
    {"cause": "a + b", "effect": "b + a", "mechanism": "addition is commutative", "domain": "arithmetic", "confidence": 0.99},
    {"cause": "a * b", "effect": "b * a", "mechanism": "multiplication is commutative", "domain": "arithmetic", "confidence": 0.99},
    {"cause": "a * (b + c)", "effect": "a*b + a*c", "mechanism": "distributive law", "domain": "arithmetic", "confidence": 0.99},
    {"cause": "x - x", "effect": "0", "mechanism": "self-subtraction is zero", "domain": "arithmetic", "confidence": 0.99},
    {"cause": "x / x", "effect": "1", "mechanism": "self-division is one", "domain": "arithmetic", "confidence": 0.95},
    {"cause": "p and true", "effect": "p", "mechanism": "true is identity for AND", "domain": "logic", "confidence": 0.99},
    {"cause": "p and false", "effect": "false", "mechanism": "false annihilates AND", "domain": "logic", "confidence": 0.99},
    {"cause": "p or false", "effect": "p", "mechanism": "false is identity for OR", "domain": "logic", "confidence": 0.99},
    {"cause": "p or true", "effect": "true", "mechanism": "true annihilates OR", "domain": "logic", "confidence": 0.99},
    {"cause": "p and p", "effect": "p", "mechanism": "idempotent law of AND", "domain": "logic", "confidence": 0.99},
    {"cause": "p or p", "effect": "p", "mechanism": "idempotent law of OR", "domain": "logic", "confidence": 0.99},
    {"cause": "p or (not p)", "effect": "true", "mechanism": "law of excluded middle", "domain": "logic", "confidence": 0.99},
    {"cause": "p and (not p)", "effect": "false", "mechanism": "law of contradiction", "domain": "logic", "confidence": 0.99},
]

_SEED_SCHEMAS: List[dict] = [
    {
        "name": "arithmetic_identity",
        "domain": "arithmetic",
        "slots": {"op": "operator", "identity_element": "constant", "operand": "variable"},
        "constraints": ["op(operand, identity_element) = operand", "identity_element is neutral"],
        "examples": [("x+0=x", 0.99), ("x*1=x", 0.99)],
        "activation": 0.0,
    },
    {
        "name": "elimination",
        "domain": "general",
        "slots": {"pattern": "expression", "result": "expression"},
        "constraints": ["result is simpler than pattern", "result has fewer nodes"],
        "examples": [("not not x → x", 0.99), ("x - x → 0", 0.99)],
        "activation": 0.0,
    },
    {
        "name": "distribution",
        "domain": "arithmetic",
        "slots": {"outer_op": "operator", "inner_op": "operator"},
        "constraints": ["outer distributes over inner", "a*(b+c) = a*b + a*c"],
        "examples": [("*(+)", 0.99)],
        "activation": 0.0,
    },
    {
        "name": "negation_pair",
        "domain": "logic",
        "slots": {"negation_op": "operator"},
        "constraints": ["double application = identity", "neg(neg(x)) = x"],
        "examples": [("not not", 0.99)],
        "activation": 0.0,
    },
    {
        "name": "commutativity",
        "domain": "arithmetic",
        "slots": {"op": "operator", "left": "expression", "right": "expression"},
        "constraints": ["op(left, right) = op(right, left)", "order irrelevant"],
        "examples": [("a+b=b+a", 0.99), ("a*b=b*a", 0.99)],
        "activation": 0.0,
    },
    {
        "name": "annihilation",
        "domain": "general",
        "slots": {"op": "operator", "annihilator": "constant"},
        "constraints": ["op(x, annihilator) = annihilator", "result overrides input"],
        "examples": [("x*0=0", 0.99), ("p and false = false", 0.99)],
        "activation": 0.0,
    },
    {
        "name": "logic_identity",
        "domain": "logic",
        "slots": {"op": "operator", "identity_element": "constant", "operand": "variable"},
        "constraints": ["op(operand, identity_element) = operand", "identity_element is neutral for op"],
        "examples": [("p and true = p", 0.99), ("p or false = p", 0.99)],
        "activation": 0.0,
    },
]

# Imagination templates: given a seed concept, what axes of variation to explore
_IMAGINATION_AXES: Dict[str, List[dict]] = {
    "addition": [
        {"hypothesis": "what if addition had 3 operands?", "plausibility": 0.7, "type": "arity_extension"},
        {"hypothesis": "what if addition was not commutative?", "plausibility": 0.4, "type": "property_negation"},
        {"hypothesis": "what if zero was not the identity for addition?", "plausibility": 0.3, "type": "identity_challenge"},
        {"hypothesis": "what if addition was idempotent (x+x=x)?", "plausibility": 0.5, "type": "idempotence"},
        {"hypothesis": "what if addition had a different identity element?", "plausibility": 0.6, "type": "identity_variation"},
    ],
    "multiplication": [
        {"hypothesis": "what if multiplication was not distributive over addition?", "plausibility": 0.3, "type": "distribution_negation"},
        {"hypothesis": "what if one was not the multiplicative identity?", "plausibility": 0.3, "type": "identity_challenge"},
        {"hypothesis": "what if zero did not annihilate multiplication?", "plausibility": 0.3, "type": "annihilation_challenge"},
        {"hypothesis": "what if multiplication was not associative?", "plausibility": 0.4, "type": "associativity_negation"},
    ],
    "negation": [
        {"hypothesis": "what if double negation did not cancel?", "plausibility": 0.3, "type": "involution_negation"},
        {"hypothesis": "what if negation had higher arity?", "plausibility": 0.5, "type": "arity_extension"},
        {"hypothesis": "what if there were multiple distinct negation operators?", "plausibility": 0.6, "type": "operator_extension"},
    ],
    "equality": [
        {"hypothesis": "what if equality was not transitive?", "plausibility": 0.2, "type": "transitivity_negation"},
        {"hypothesis": "what if equality was directional?", "plausibility": 0.5, "type": "asymmetry"},
        {"hypothesis": "what if there were degrees of equality?", "plausibility": 0.7, "type": "graded_extension"},
    ],
    "logic": [
        {"hypothesis": "what if excluded middle did not hold (intuitionistic logic)?", "plausibility": 0.8, "type": "classical_challenge"},
        {"hypothesis": "what if there were more truth values?", "plausibility": 0.9, "type": "multivalued_extension"},
        {"hypothesis": "what if contradictions were allowed (paraconsistent logic)?", "plausibility": 0.7, "type": "paraconsistency"},
    ],
    "arithmetic": [
        {"hypothesis": "what if arithmetic was modular?", "plausibility": 0.9, "type": "modular_extension"},
        {"hypothesis": "what if subtraction was not defined?", "plausibility": 0.5, "type": "operation_removal"},
        {"hypothesis": "what if division always rounded?", "plausibility": 0.8, "type": "rounding_variant"},
    ],
    "causality": [
        {"hypothesis": "what if causes could be their own effects?", "plausibility": 0.6, "type": "circular_causation"},
        {"hypothesis": "what if effects preceded causes?", "plausibility": 0.4, "type": "reverse_causation"},
        {"hypothesis": "what if all causes had equal weight?", "plausibility": 0.5, "type": "uniform_causation"},
    ],
    "simplification": [
        {"hypothesis": "what if simplification always increased complexity first?", "plausibility": 0.4, "type": "complexity_inversion"},
        {"hypothesis": "what if multiple simplification paths always converged?", "plausibility": 0.7, "type": "confluence"},
    ],
}

# Structural analogies between domains
_STRUCTURAL_ANALOGIES: List[dict] = [
    {
        "source_rule": "additive_identity",
        "source_domain": "arithmetic",
        "target_domain": "logic",
        "analogy": "and_true",
        "explanation": "0 is to + as true is to AND (both are identity elements)",
        "confidence": 0.9,
    },
    {
        "source_rule": "multiplicative_zero",
        "source_domain": "arithmetic",
        "target_domain": "logic",
        "analogy": "and_false",
        "explanation": "0 is to * as false is to AND (both annihilate their operation)",
        "confidence": 0.9,
    },
    {
        "source_rule": "double_negation",
        "source_domain": "arithmetic",
        "target_domain": "logic",
        "analogy": "double_negation_logic",
        "explanation": "neg(neg(x))=x in arithmetic maps to not(not(p))=p in logic (involution)",
        "confidence": 0.95,
    },
    {
        "source_rule": "multiplicative_identity",
        "source_domain": "arithmetic",
        "target_domain": "logic",
        "analogy": "or_false",
        "explanation": "1 is to * as false is to OR (both are identity elements)",
        "confidence": 0.85,
    },
    {
        "source_rule": "subtractive_self",
        "source_domain": "arithmetic",
        "target_domain": "logic",
        "analogy": "contradiction",
        "explanation": "x-x=0 maps to p AND NOT(p)=false (self-cancellation)",
        "confidence": 0.8,
    },
    {
        "source_rule": "additive_identity",
        "source_domain": "arithmetic",
        "target_domain": "algebra",
        "analogy": "neutral_element",
        "explanation": "Additive identity is the algebraic neutral element concept",
        "confidence": 0.95,
    },
    {
        "source_rule": "distribution",
        "source_domain": "arithmetic",
        "target_domain": "algebra",
        "analogy": "ring_distributivity",
        "explanation": "Distributive law of * over + is the defining property of a ring",
        "confidence": 0.9,
    },
]


# ── V3 Engine classes (merged from WorldModelV3) ──────────────────────────────

class CausalDiscovery:
    """Discovers causal links from solve episodes (from WorldModelV3)."""

    @staticmethod
    def induce_from_solve(expression: str, transforms: List[str],
                          delta: float, domain: str,
                          existing_links: Dict[str, CausalLink]) -> List[CausalLink]:
        new_links = []
        if not transforms or delta <= 0.01:
            return new_links
        for transform_name in transforms:
            cause  = f"pattern_with_{transform_name}_applicable"
            effect = f"energy_reduced_by_{transform_name}"
            key    = f"{cause}→{effect}"
            if key in existing_links:
                existing_links[key].observe(success=True)
            else:
                link = CausalLink(cause=cause, effect=effect, mechanism=transform_name,
                                  domain=domain, confidence=0.6, evidence_count=1)
                existing_links[key] = link
                new_links.append(link)
        # Sequential dependency links
        for i in range(len(transforms) - 1):
            t1, t2 = transforms[i], transforms[i + 1]
            key = f"{t1}_completed→{t2}_applicable"
            if key in existing_links:
                existing_links[key].observe(success=True)
            else:
                link = CausalLink(cause=f"{t1}_completed", effect=f"{t2}_applicable",
                                  mechanism="sequential_dependency", domain=domain,
                                  confidence=0.4, evidence_count=1)
                existing_links[key] = link
                new_links.append(link)
        return new_links

    @staticmethod
    def weaken_from_failure(transforms: List[str], domain: str,
                            existing_links: Dict[str, CausalLink]) -> None:
        for transform_name in transforms:
            cause  = f"pattern_with_{transform_name}_applicable"
            effect = f"energy_reduced_by_{transform_name}"
            key    = f"{cause}→{effect}"
            if key in existing_links:
                existing_links[key].observe(success=False)


class SchemaInduction:
    """Discovers schemas from recurring solve patterns (from WorldModelV3)."""

    @staticmethod
    def compute_structural_signature(transforms: List[str], domain: str) -> str:
        content = f"{domain}:{'→'.join(sorted(set(transforms)))}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    @staticmethod
    def induce_schemas(solve_history: List[dict],
                       existing_schemas: Dict[str, Schema],
                       min_exemplars: int = 3) -> List[Schema]:
        clusters: Dict[str, List[dict]] = defaultdict(list)
        for episode in solve_history:
            if not episode.get("success"):
                continue
            transforms = episode.get("transforms", [])
            domain = episode.get("domain", "general")
            if not transforms:
                continue
            sig = SchemaInduction.compute_structural_signature(transforms, domain)
            clusters[sig].append(episode)

        new_schemas = []
        for sig, episodes in clusters.items():
            if len(episodes) < min_exemplars:
                continue
            if sig in existing_schemas:
                existing_schemas[sig].confidence = min(0.99, existing_schemas[sig].confidence + 0.05)
                for ep in episodes[-5:]:
                    pid = ep.get("expression", ep.get("problem_id", ""))
                    if pid and pid not in existing_schemas[sig].exemplars:
                        existing_schemas[sig].exemplars.append(pid)
                continue

            all_transforms: set = set()
            domains: set = set()
            for ep in episodes:
                all_transforms.update(ep.get("transforms", []))
                domains.add(ep.get("domain", "general"))

            slot_types = {}
            for t in all_transforms:
                if "elim" in t or "fold" in t:    slot_types[t] = "simplifier"
                elif "solve" in t:                slot_types[t] = "solver"
                elif "factor" in t:               slot_types[t] = "restructurer"
                elif "cancel" in t:               slot_types[t] = "canceller"
                else:                             slot_types[t] = "transformer"

            invariants = []
            deltas = [ep.get("delta", 0) for ep in episodes if ep.get("delta", 0) > 0]
            if deltas:
                invariants.append(f"avg_energy_reduction={sum(deltas)/len(deltas):.2f}")

            primary_domain = max(domains, key=lambda d: sum(1 for ep in episodes if ep.get("domain") == d))
            primary_t = max(all_transforms, key=lambda t: sum(
                1 for ep in episodes if t in ep.get("transforms", [])))
            name = f"{primary_domain}_{primary_t}_pattern"

            schema = Schema(
                name=name,
                slots=slot_types,
                constraints=[f"avg_delta={invariants[0] if invariants else 'unknown'}"],
                examples=[],
                structural_signature=sig,
                slot_types=slot_types,
                invariants=invariants,
                exemplars=[ep.get("expression", "")[:50] for ep in episodes[-20:]],
                domain=primary_domain,
                confidence=0.5 + 0.05 * min(len(episodes), 10),
            )
            existing_schemas[sig] = schema
            new_schemas.append(schema)
        return new_schemas


class ContradictionDetector:
    """Checks whether a new causal link contradicts existing knowledge."""

    @staticmethod
    def check(new_link: CausalLink,
              existing_links: Dict[str, CausalLink],
              beliefs: Dict[str, "Belief"]) -> dict:
        conflicts = []
        supports  = []
        for key, link in existing_links.items():
            if link.cause == new_link.cause and link.effect != new_link.effect:
                if link.confidence > 0.7:
                    conflicts.append(
                        f"Existing {key} (conf={link.confidence:.2f}) claims "
                        f"{link.cause}→{link.effect}, not →{new_link.effect}"
                    )
            elif link.mechanism == new_link.mechanism and link.domain == new_link.domain:
                supports.append(f"Consistent with {key}")
        for bkey, belief in beliefs.items():
            if new_link.mechanism in belief.subject:
                if belief.confidence > 0.7 and "fails" in belief.predicate:
                    conflicts.append(
                        f"Belief '{bkey}' ({belief.confidence:.2f}) "
                        f"says {belief.subject} {belief.predicate}"
                    )
        return {
            "consistent": len(conflicts) == 0,
            "conflicts": conflicts,
            "supports": supports,
            "severity": "high" if len(conflicts) > 2 else "low" if conflicts else "none",
        }


# ── Main WorldModel class ──────────────────────────────────────────────────────

class WorldModel:
    """
    Human Imagination World Model for SARE-HX.

    Represents the system's internal model of how the world works — not just
    a flat fact store, but a structured web of schemas, causal links, and
    the ability to reason counterfactually and imaginatively.
    """

    DEFAULT_V2_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "world_model_v2.json"
    LEGACY_PATH     = Path(__file__).resolve().parents[3] / "data" / "memory" / "world_model.json"

    _PRUNE_EVERY = 500       # prune after this many adds
    _MAX_LINKS = 50_000      # hard cap (raised from 8K — system has 130K observations)

    def __init__(self, persist_path: Optional[Path] = None):
        self._path = Path(persist_path or self.DEFAULT_V2_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Core stores
        self._facts: Dict[str, List[Fact]] = {}            # domain → [Fact]
        self._causal_links: Dict[str, CausalLink] = {}     # key → CausalLink
        self._schemas: Dict[str, Schema] = {}              # name → Schema

        # Prediction tracking
        self._predictions: List[Prediction] = []        # recent predictions (capped at 500)
        self._belief_accuracy: Dict[str, list] = {}     # transform_name → [was_correct bools]
        self._surprise_history: List[float] = []        # last 100 surprise values
        self._last_llm_hypothesis_time: float = 0.0    # cooldown for LLM hypothesis generation
        self._llm_hypothesis_cooldown: float = 60.0    # seconds between LLM hypothesis calls

        # Auto-learning counters
        self._solve_counts: Dict[str, int] = {}         # domain → solve count (for distillation trigger)
        self._distill_interval = 50                     # distill knowledge every N solves per domain
        self._last_distill: Dict[str, float] = {}       # domain → last distill timestamp
        # Activity log for UI display
        self._activity_log: List[dict] = []             # last 50 learning events

        # Causal index for fast predict_transform() lookups
        self._causal_idx: Dict[str, Dict[str, List[CausalLink]]] = {}  # domain → mechanism_token → links
        self._causal_idx_dirty: bool = True

        # Pruning counter
        self._link_add_count: int = 0

        # ── V3 additions: richer belief + analogy tracking ────────────────────
        self._beliefs: Dict[str, Belief] = {}          # bkey → Belief
        self._analogies: Dict[str, Analogy] = {}       # analogy.key → Analogy
        self._solve_history: List[dict] = []            # for SchemaInduction
        self._v3_stats: dict = {
            "total_observations": 0,
            "links_discovered":   0,
            "schemas_induced":    0,
            "analogies_found":    0,
        }

        # Load from v2 file (or bootstrap from v1 + seeds)
        self.load()
        self._seed_if_empty()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Fact API (backwards-compatible)
    # ─────────────────────────────────────────────────────────────────────────

    def add_fact(self, domain: str, fact: str, confidence: float = 0.9,
                 source: str = "api") -> None:
        """Add a simple fact (compatible with existing WorldModel API)."""
        if not fact or not fact.strip():
            return
        f = Fact(domain=domain, fact=fact.strip(), confidence=confidence, source=source)
        self._facts.setdefault(domain, [])
        # Deduplicate by fact text
        existing = [x.fact for x in self._facts[domain]]
        if fact.strip() not in existing:
            self._facts[domain].append(f)

    def get_facts(self, domain: str) -> List[dict]:
        """Return all facts for a domain as dicts."""
        return [f.to_dict() for f in self._facts.get(domain, [])]

    def get_all_facts(self) -> Dict[str, List[dict]]:
        return {d: [f.to_dict() for f in facts] for d, facts in self._facts.items()}

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Causal Link API
    # ─────────────────────────────────────────────────────────────────────────

    def add_causal_link(self, cause: str, effect: str, mechanism: str,
                        domain: str, confidence: float = 0.85) -> CausalLink:
        """Add or reinforce a causal link."""
        link = CausalLink(
            cause=cause.strip(),
            effect=effect.strip(),
            mechanism=mechanism.strip(),
            confidence=confidence,
            domain=domain,
        )
        key = link.key
        if key in self._causal_links:
            # Reinforce existing link: bump evidence and update confidence
            existing = self._causal_links[key]
            existing.evidence_count += 1
            # Bayesian-style update: weight toward new observation
            alpha = 0.1
            existing.confidence = (1 - alpha) * existing.confidence + alpha * confidence
        else:
            self._causal_links[key] = link
        self._causal_idx_dirty = True
        self._link_add_count += 1
        if self._link_add_count % self._PRUNE_EVERY == 0:
            self._prune_weak_links()
        return self._causal_links[key]

    def get_causal_links(self, domain: Optional[str] = None) -> List[dict]:
        links = self._causal_links.values()
        if domain:
            links = [l for l in links if l.domain == domain]
        return [l.to_dict() for l in links]

    def find_causal_chain(self, start: str, max_depth: int = 4) -> List[List[str]]:
        """
        BFS to find all causal chains starting from `start`.
        Returns list of chains (each chain is [cause, intermediate..., final_effect]).
        """
        chains = []
        # Build adjacency: cause → [CausalLink]
        adj: Dict[str, List[CausalLink]] = {}
        for link in self._causal_links.values():
            adj.setdefault(link.cause, []).append(link)

        frontier = [([start], 1.0)]
        while frontier:
            chain, conf = frontier.pop(0)
            current = chain[-1]
            if len(chain) > max_depth:
                continue
            for link in adj.get(current, []):
                new_chain = chain + [link.effect]
                chains.append(new_chain)
                if link.effect not in chain:  # avoid cycles
                    frontier.append((new_chain, conf * link.confidence))

        return chains

    # ─────────────────────────────────────────────────────────────────────────
    # 2b. Causal index helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_causal_idx(self) -> None:
        """Build domain → mechanism_token → [CausalLink] index for O(1) lookups."""
        self._causal_idx = {}
        for link in self._causal_links.values():
            domain_bucket = self._causal_idx.setdefault(link.domain, {})
            mech_bucket = domain_bucket.setdefault(link.mechanism, [])
            mech_bucket.append(link)
        self._causal_idx_dirty = False

    def _get_causal_boost(self, tname: str, domain: str) -> float:
        """Return causal confidence boost for a transform, using the index."""
        if self._causal_idx_dirty:
            self._build_causal_idx()
        boost = 0.0
        for d in (domain, "general"):
            for link in self._causal_idx.get(d, {}).get(tname, []):
                boost += link.confidence * 0.2
        return min(boost, 0.5)   # cap

    def _prune_weak_links(self) -> int:
        """Remove links with low confidence AND few observations. Returns pruned count."""
        before = len(self._causal_links)
        # Keep: confidence >= 0.3 OR evidence_count >= 5
        self._causal_links = {
            k: v for k, v in self._causal_links.items()
            if v.confidence >= 0.3 or v.evidence_count >= 5
        }
        # If still too many, keep the top _MAX_LINKS by score
        if len(self._causal_links) > self._MAX_LINKS:
            ranked = sorted(self._causal_links.items(),
                            key=lambda x: x[1].confidence * math.log(x[1].evidence_count + 1),
                            reverse=True)
            self._causal_links = dict(ranked[:self._MAX_LINKS])
        self._causal_idx_dirty = True
        pruned = before - len(self._causal_links)
        if pruned:
            log.info("WorldModel: pruned %d weak causal links (%d remain)", pruned, len(self._causal_links))
        return pruned

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Schema API
    # ─────────────────────────────────────────────────────────────────────────

    def add_schema(self, schema: Schema) -> None:
        """Add or replace a schema."""
        self._schemas[schema.name] = schema

    def get_schema(self, name: str) -> Optional[Schema]:
        return self._schemas.get(name)

    def get_schemas(self, domain: Optional[str] = None) -> List[dict]:
        schemas = self._schemas.values()
        if domain:
            schemas = [s for s in schemas if s.domain == domain]
        return [s.to_dict() for s in schemas]

    def _activate_schema(self, name: str, delta: float = 0.2) -> None:
        """Increase activation of a schema, decay others."""
        if name in self._schemas:
            self._schemas[name].activation = min(1.0, self._schemas[name].activation + delta)
        # Slight decay on all others
        for sname, s in self._schemas.items():
            if sname != name:
                s.activation = max(0.0, s.activation - 0.02)

    def _find_matching_schemas(self, expression: str) -> List[Schema]:
        """Find schemas whose slots/constraints mention keywords in the expression."""
        expr_lower = expression.lower()
        matches = []
        for schema in self._schemas.values():
            score = 0.0
            # Check slot types and names
            for slot_name, slot_type in schema.slots.items():
                if slot_name.lower() in expr_lower or slot_type.lower() in expr_lower:
                    score += 0.3
            # Check constraints
            for constraint in schema.constraints:
                for token in constraint.lower().split():
                    if len(token) > 3 and token in expr_lower:
                        score += 0.2
            # Check examples
            for (example_fp, _conf) in schema.examples:
                for token in example_fp.lower().split():
                    if len(token) > 2 and token in expr_lower:
                        score += 0.1
            if score > 0.2:
                matches.append((schema, score))
        matches.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in matches]

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Mental Simulation Engine
    # ─────────────────────────────────────────────────────────────────────────

    def simulate(self, scenario: str, steps: int = 3) -> List[str]:
        """
        Mentally simulate a scenario using causal links.

        Traces what would happen if a given scenario were true by following
        causal chains and identifying which rules/schemas would be affected.

        Args:
            scenario: A natural-language or expression-style scenario description
            steps: How many causal hops to trace

        Returns:
            List of predicted consequences (strings)
        """
        scenario_lower = scenario.lower()
        consequences = []
        activated_schemas = self._find_matching_schemas(scenario)

        # Find directly relevant causal links
        direct_links = []
        for link in self._causal_links.values():
            cause_tokens = set(link.cause.lower().split())
            scenario_tokens = set(scenario_lower.split())
            overlap = cause_tokens & scenario_tokens
            if overlap or any(t in scenario_lower for t in cause_tokens if len(t) > 2):
                direct_links.append(link)

        # For each direct link, trace downstream consequences
        seen_effects = set()
        for link in direct_links[:6]:  # cap to avoid explosion
            if link.effect not in seen_effects:
                seen_effects.add(link.effect)
                consequences.append(
                    f"[direct] {link.cause} → {link.effect} (via: {link.mechanism})"
                )
            # Follow chain
            if steps > 1:
                chains = self.find_causal_chain(link.effect, max_depth=steps - 1)
                for chain in chains[:3]:
                    if len(chain) > 1:
                        chain_str = " → ".join(chain)
                        if chain_str not in seen_effects:
                            seen_effects.add(chain_str)
                            consequences.append(f"[chain] {chain_str}")

        # Schema-level consequences
        for schema in activated_schemas[:3]:
            self._activate_schema(schema.name, delta=0.15)
            consequences.append(
                f"[schema] Activates '{schema.name}' frame: {schema.constraints[0] if schema.constraints else ''}"
            )

        # If no direct links found, generate generic reasoning
        if not consequences:
            consequences.append(
                f"[inference] No direct causal links found for scenario. "
                f"Related schemas: {[s.name for s in activated_schemas[:2]] or ['none']}"
            )
            # Try to find indirect links by keyword
            keywords = [w for w in scenario_lower.split() if len(w) > 3]
            for link in self._causal_links.values():
                for kw in keywords:
                    if kw in link.mechanism.lower():
                        consequences.append(
                            f"[mechanism] Related: {link.cause} → {link.effect} ({link.mechanism})"
                        )
                        break
                if len(consequences) > 6:
                    break

        return consequences[:10]  # cap output

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Counterfactual Reasoning
    # ─────────────────────────────────────────────────────────────────────────

    def counterfactual(self, rule_name: str, negated: bool = True) -> dict:
        """
        Reason about what would happen if a rule did/did not exist.

        Finds all rules/links that depend on or were derived via the given rule,
        then estimates which domains would be impacted.

        Args:
            rule_name: Name of the rule to reason about
            negated: If True, reason about rule NOT existing; if False, rule existing

        Returns:
            Impact report dict with keys: rule, negated, direct_impact,
            chain_impact, affected_domains, severity
        """
        rule_lower = rule_name.lower().replace("_", " ")
        action_phrase = "did not exist" if negated else "existed"

        # Find causal links that mention this rule in mechanism
        directly_dependent: List[CausalLink] = []
        for link in self._causal_links.values():
            mech_lower = link.mechanism.lower()
            name_lower = link.mechanism.lower()
            if (rule_lower in mech_lower or rule_name.lower() in mech_lower
                    or rule_lower in link.cause.lower()
                    or rule_lower in link.effect.lower()):
                directly_dependent.append(link)

        # Find schemas that reference this rule
        dependent_schemas: List[Schema] = []
        for schema in self._schemas.values():
            schema_text = (
                " ".join(schema.constraints)
                + " ".join(f"{k} {v}" for k, v in schema.slots.items())
                + schema.name
            ).lower()
            if rule_lower in schema_text or rule_name.lower() in schema_text:
                dependent_schemas.append(schema)

        # Chain impact: what other links become unprovable
        chain_impacted: List[str] = []
        for link in directly_dependent:
            # Find links whose cause matches this link's effect
            for other_link in self._causal_links.values():
                if other_link.cause == link.effect and other_link.key != link.key:
                    chain_impacted.append(
                        f"{other_link.cause} → {other_link.effect}"
                    )

        affected_domains = list({l.domain for l in directly_dependent})

        # Estimate severity
        severity_score = (
            len(directly_dependent) * 0.3
            + len(chain_impacted) * 0.2
            + len(dependent_schemas) * 0.2
        )
        if severity_score < 0.5:
            severity = "low"
        elif severity_score < 1.5:
            severity = "moderate"
        else:
            severity = "high"

        return {
            "rule": rule_name,
            "negated": negated,
            "scenario": f"If '{rule_name}' {action_phrase}...",
            "direct_impact": [
                {
                    "link": l.key,
                    "cause": l.cause,
                    "effect": l.effect,
                    "mechanism": l.mechanism,
                    "domain": l.domain,
                }
                for l in directly_dependent
            ],
            "chain_impact": chain_impacted[:8],
            "dependent_schemas": [s.name for s in dependent_schemas],
            "affected_domains": affected_domains,
            "severity": severity,
            "severity_score": round(severity_score, 2),
            "summary": (
                f"Removing '{rule_name}' would directly affect {len(directly_dependent)} causal "
                f"links and {len(dependent_schemas)} schemas across domains: "
                f"{', '.join(affected_domains) or 'none'}"
            ) if negated else (
                f"Adding '{rule_name}' would enable {len(directly_dependent)} causal "
                f"links and activate {len(dependent_schemas)} schemas"
            ),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Contradiction Detection
    # ─────────────────────────────────────────────────────────────────────────

    def check_rule_consistency(self, rule_name: str, pattern: str,
                                domain: str) -> dict:
        """
        Check whether a new rule is consistent with known causal links and schemas.

        Called before rule promotion to detect contradictions.

        Returns:
            {
                "consistent": bool,
                "conflicts": [str, ...],    # descriptions of contradictions
                "support":   [str, ...],    # supporting evidence
                "warnings":  [str, ...],    # non-fatal concerns
            }
        """
        conflicts: List[str] = []
        support: List[str] = []
        warnings: List[str] = []

        rule_lower = rule_name.lower().replace("_", " ")
        pattern_lower = pattern.lower()

        # Check against causal links in the same domain
        domain_links = [l for l in self._causal_links.values() if l.domain == domain]

        for link in domain_links:
            cause_lower = link.cause.lower()
            effect_lower = link.effect.lower()

            # Direct contradiction: rule claims X → Y but existing link claims X → not-Y
            # We detect this heuristically by checking if the pattern appears as a *cause*
            # and the rule_name appears in a conflicting mechanism
            if pattern_lower and pattern_lower in cause_lower:
                # This pattern already has a known effect
                support.append(
                    f"Pattern '{pattern}' is consistent with known link: "
                    f"'{link.cause} → {link.effect}' ({link.mechanism})"
                )

            # Check if this rule's name conflicts with an existing mechanism
            # that claims the opposite (e.g., "identity doesn't hold" vs "is identity")
            negation_indicators = ["not ", "doesn't", "cannot", "false", "invalid"]
            for neg in negation_indicators:
                if neg in link.mechanism.lower() and rule_lower in link.mechanism.lower():
                    conflicts.append(
                        f"Existing link '{link.key}' mechanism '{link.mechanism}' "
                        f"conflicts with rule '{rule_name}'"
                    )

        # Check schema constraints for violations
        matching_schemas = self._find_matching_schemas(pattern)
        for schema in matching_schemas:
            # A schema constraint like "result is simpler than pattern" supports elimination rules
            if "simplif" in " ".join(schema.constraints).lower():
                support.append(
                    f"Schema '{schema.name}' supports simplification rules"
                )
            # Check if any constraint would be violated
            for constraint in schema.constraints:
                # Very basic: if constraint says X must hold and rule appears to violate it
                if "must" in constraint.lower() or "always" in constraint.lower():
                    warnings.append(
                        f"Schema '{schema.name}' has strong constraint: '{constraint}' "
                        f"— verify new rule does not violate it"
                    )

        # Check for duplicate rules (same pattern already promoted)
        for link in self._causal_links.values():
            if link.mechanism.lower() == rule_lower or link.mechanism.lower() == rule_name.lower():
                warnings.append(
                    f"Rule '{rule_name}' may duplicate existing causal link "
                    f"'{link.cause} → {link.effect}'"
                )

        # Find supporting facts in the same domain
        for fact in self._facts.get(domain, []):
            if rule_lower in fact.fact.lower():
                support.append(f"Supporting fact: '{fact.fact}' (confidence={fact.confidence})")

        consistent = len(conflicts) == 0

        return {
            "consistent": consistent,
            "conflicts": conflicts,
            "support": support,
            "warnings": warnings,
            "rule": rule_name,
            "pattern": pattern,
            "domain": domain,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Imagination / Hypothesis Generation
    # ─────────────────────────────────────────────────────────────────────────

    def imagine(self, seed_concept: str, depth: int = 2) -> List[dict]:
        """
        Generate plausible hypothetical extensions from a seed concept.

        Uses pre-wired imagination templates and causal-link-based reasoning
        to produce creative but grounded hypotheses.

        Args:
            seed_concept: Starting concept ("addition", "logic", "negation", etc.)
            depth: How many levels of follow-on imagination to pursue

        Returns:
            List of hypothesis dicts with: hypothesis, plausibility, type, depth,
            basis, implications
        """
        seed_lower = seed_concept.lower().strip()
        results: List[dict] = []
        seen_hyps: set = set()

        # --- Level 0: direct templates ---
        templates = _IMAGINATION_AXES.get(seed_lower, [])

        # Also search for partial matches
        if not templates:
            for key, tmpl_list in _IMAGINATION_AXES.items():
                if key in seed_lower or seed_lower in key:
                    templates.extend(tmpl_list)

        for tmpl in templates:
            hyp = tmpl["hypothesis"]
            if hyp in seen_hyps:
                continue
            seen_hyps.add(hyp)

            # Find supporting or contradicting causal links for this hypothesis
            basis = self._find_basis_for_hypothesis(hyp, seed_lower)
            implications = self._derive_implications(hyp, depth=1)

            results.append({
                "hypothesis": hyp,
                "plausibility": tmpl["plausibility"],
                "type": tmpl["type"],
                "depth": 0,
                "seed": seed_concept,
                "basis": basis,
                "implications": implications,
            })

        # --- Level 1+: causal-link-based imagination ---
        if depth >= 1:
            causal_hyps = self._imagine_from_causal_links(seed_lower)
            for hyp_dict in causal_hyps:
                if hyp_dict["hypothesis"] not in seen_hyps:
                    seen_hyps.add(hyp_dict["hypothesis"])
                    hyp_dict["depth"] = 1
                    hyp_dict["seed"] = seed_concept
                    results.append(hyp_dict)

        # --- Level 2: schema-based imagination ---
        if depth >= 2:
            schema_hyps = self._imagine_from_schemas(seed_lower)
            for hyp_dict in schema_hyps:
                if hyp_dict["hypothesis"] not in seen_hyps:
                    seen_hyps.add(hyp_dict["hypothesis"])
                    hyp_dict["depth"] = 2
                    hyp_dict["seed"] = seed_concept
                    results.append(hyp_dict)

        # Sort by plausibility descending
        results.sort(key=lambda x: x.get("plausibility", 0.0), reverse=True)
        return results[:15]

    def _find_basis_for_hypothesis(self, hypothesis: str, seed: str) -> List[str]:
        """Find causal links and facts that ground a hypothesis."""
        basis = []
        hyp_lower = hypothesis.lower()
        seed_lower = seed.lower()
        for link in self._causal_links.values():
            if (seed_lower in link.cause.lower()
                    or seed_lower in link.mechanism.lower()
                    or seed_lower in link.effect.lower()):
                basis.append(f"Known: {link.cause} → {link.effect} ({link.mechanism})")
            if len(basis) >= 3:
                break
        return basis

    def _derive_implications(self, hypothesis: str, depth: int = 1) -> List[str]:
        """Derive what a hypothesis implies for the existing knowledge base."""
        implications = []
        hyp_lower = hypothesis.lower()
        # Look for causal links whose mechanism is challenged by this hypothesis
        for link in self._causal_links.values():
            mech_lower = link.mechanism.lower()
            # If the hypothesis challenges the mechanism
            for keyword in ["not", "didn't", "without", "no longer"]:
                if keyword in hyp_lower:
                    # Check if link mechanism is relevant
                    for token in link.cause.lower().split():
                        if len(token) > 3 and token in hyp_lower:
                            implications.append(
                                f"Would invalidate: {link.cause} → {link.effect}"
                            )
                            break
            if len(implications) >= 3:
                break
        return implications

    def _imagine_from_causal_links(self, seed: str) -> List[dict]:
        """Generate hypotheses by inverting or extending causal links."""
        hypotheses = []
        for link in self._causal_links.values():
            seed_match = (
                seed in link.cause.lower()
                or seed in link.mechanism.lower()
                or seed in link.domain
            )
            if not seed_match:
                continue

            # Inversion hypothesis: what if the effect was different?
            hypotheses.append({
                "hypothesis": f"what if '{link.cause}' did not simplify to '{link.effect}'?",
                "plausibility": max(0.2, 0.7 - link.confidence),
                "type": "causal_inversion",
                "basis": [f"Inverts known: {link.mechanism}"],
                "implications": [f"Would break {link.domain} simplification"],
            })

            # Extension hypothesis: what if the mechanism generalised?
            if link.evidence_count >= 2:
                hypotheses.append({
                    "hypothesis": f"what if '{link.mechanism}' applied to all domains?",
                    "plausibility": 0.5,
                    "type": "domain_generalisation",
                    "basis": [f"Based on: {link.cause} → {link.effect}"],
                    "implications": [f"Would extend {link.domain} rules globally"],
                })

            if len(hypotheses) >= 6:
                break

        return hypotheses

    def _imagine_from_schemas(self, seed: str) -> List[dict]:
        """Generate hypotheses by varying schema constraints."""
        hypotheses = []
        relevant_schemas = [
            s for s in self._schemas.values()
            if seed in s.name.lower() or seed in s.domain.lower()
            or any(seed in slot for slot in s.slots)
        ]
        for schema in relevant_schemas[:3]:
            for constraint in schema.constraints[:2]:
                hypotheses.append({
                    "hypothesis": f"what if the schema '{schema.name}' constraint '{constraint}' were relaxed?",
                    "plausibility": 0.45,
                    "type": "schema_relaxation",
                    "basis": [f"Schema: {schema.name} in {schema.domain}"],
                    "implications": [f"Would weaken {schema.name} applicability"],
                })
        return hypotheses

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Analogy Generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_analogy(self, source_rule: str,
                         target_domain: str) -> Optional[dict]:
        """
        Find the structural analogue of a rule in a different domain.

        Uses pre-wired structural analogies and schema matching to propose
        the equivalent concept in the target domain.

        Args:
            source_rule: Name of the rule to find an analogy for
            target_domain: Domain to find the analogy in

        Returns:
            Analogy proposal dict, or None if no analogy found
        """
        source_lower = source_rule.lower()
        target_lower = target_domain.lower()

        # Direct lookup in pre-wired analogies
        for analogy in _STRUCTURAL_ANALOGIES:
            if (analogy["source_rule"].lower() == source_lower
                    and analogy["target_domain"].lower() == target_lower):
                return {
                    "source_rule": source_rule,
                    "target_domain": target_domain,
                    "analogy": analogy["analogy"],
                    "explanation": analogy["explanation"],
                    "confidence": analogy["confidence"],
                    "method": "pre_wired",
                }

        # Reverse lookup: maybe source is the target_domain rule
        for analogy in _STRUCTURAL_ANALOGIES:
            if (analogy["analogy"].lower() == source_lower
                    and analogy["source_domain"].lower() == target_lower):
                return {
                    "source_rule": source_rule,
                    "target_domain": analogy["source_domain"],
                    "analogy": analogy["source_rule"],
                    "explanation": f"Reverse analogy: {analogy['explanation']}",
                    "confidence": analogy["confidence"] * 0.9,
                    "method": "reverse_lookup",
                }

        # Schema-based analogy: find schemas that match source_rule structural role
        # then look for schemas in target_domain with same structural role
        source_schemas = self._find_matching_schemas(source_lower)
        if not source_schemas:
            return None

        # Find schemas in target domain
        target_schemas = [
            s for s in self._schemas.values() if s.domain == target_lower
        ]

        best_match = None
        best_score = 0.0
        for src_schema in source_schemas[:3]:
            for tgt_schema in target_schemas:
                # Score structural similarity of schemas
                score = self._schema_structural_similarity(src_schema, tgt_schema)
                if score > best_score and score > 0.3:
                    best_score = score
                    best_match = (src_schema, tgt_schema)

        if best_match:
            src_s, tgt_s = best_match
            return {
                "source_rule": source_rule,
                "target_domain": target_domain,
                "analogy": f"{tgt_s.name}_equivalent",
                "explanation": (
                    f"'{source_rule}' matches schema '{src_s.name}' which has "
                    f"structural parallel in '{tgt_s.name}' within {target_domain}"
                ),
                "confidence": round(best_score * 0.8, 3),
                "method": "schema_matching",
                "source_schema": src_s.name,
                "target_schema": tgt_s.name,
            }

        return None

    def _schema_structural_similarity(self, s1: Schema, s2: Schema) -> float:
        """Compute structural similarity between two schemas [0, 1]."""
        score = 0.0

        # Slot overlap (by type, not name)
        s1_types = set(s1.slots.values())
        s2_types = set(s2.slots.values())
        if s1_types and s2_types:
            overlap = len(s1_types & s2_types) / max(len(s1_types), len(s2_types))
            score += overlap * 0.4

        # Constraint keyword overlap
        s1_words = set(
            w for c in s1.constraints for w in c.lower().split() if len(w) > 3
        )
        s2_words = set(
            w for c in s2.constraints for w in c.lower().split() if len(w) > 3
        )
        if s1_words and s2_words:
            overlap = len(s1_words & s2_words) / max(len(s1_words), len(s2_words))
            score += overlap * 0.4

        # Same slot count?
        if len(s1.slots) == len(s2.slots):
            score += 0.2

        return round(score, 3)

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Domain Context
    # ─────────────────────────────────────────────────────────────────────────

    def get_domain_context(self, domain: str) -> dict:
        """
        Return all relevant knowledge for a domain in one structured dict.
        Now includes v3 beliefs and analogies.
        """
        facts = self.get_facts(domain)
        causal_links = self.get_causal_links(domain)
        schemas = self.get_schemas(domain)
        general_facts = self.get_facts("general") if domain != "general" else []
        general_links = self.get_causal_links("general") if domain != "general" else []
        beliefs = self.get_beliefs(domain)
        analogies = self.get_analogies(domain)
        return {
            "domain": domain,
            "facts": facts + general_facts,
            "causal_links": causal_links + general_links,
            "schemas": schemas,
            "beliefs": beliefs,
            "analogies": analogies,
            "fact_count": len(facts),
            "causal_link_count": len(causal_links),
            "schema_count": len(schemas),
            "belief_count": len(beliefs),
            "analogy_count": len(analogies),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 10. Summary
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a top-level summary of the world model state (v2+v3 unified)."""
        domains = (set(self._facts.keys())
                   | {l.domain for l in self._causal_links.values()}
                   | {s.domain for s in self._schemas.values()})
        return {
            "fact_count": sum(len(v) for v in self._facts.values()),
            "causal_link_count": len(self._causal_links),
            "schema_count": len(self._schemas),
            "belief_count": len(self._beliefs),
            "analogy_count": len(self._analogies) + len(_STRUCTURAL_ANALOGIES),
            "discovered_analogies": len(self._analogies),
            "solve_history_size": len(self._solve_history),
            "domains": sorted(domains),
            "v3_stats": self._v3_stats,
            "top_beliefs": sorted(
                [b.to_dict() for b in self._beliefs.values()],
                key=lambda x: x["confidence"], reverse=True
            )[:10],
            "top_causal_links": [
                {"key": l.key, "mechanism": l.mechanism, "domain": l.domain,
                 "confidence": round(l.confidence, 3)}
                for l in sorted(self._causal_links.values(),
                                key=lambda x: x.confidence, reverse=True)[:5]
            ],
            "imagination_seed_count": len(_IMAGINATION_AXES),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 11. Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save unified world model to JSON (atomic write via temp-file + rename)."""
        try:
            data = {
                "version": 4,       # v4 = v2 + v3 merged
                "saved_at": time.time(),
                "facts": {d: [f.to_dict() for f in facts]
                          for d, facts in self._facts.items()},
                "causal_links": {k: l.to_dict() for k, l in self._causal_links.items()},
                "schemas": {n: s.to_dict() for n, s in self._schemas.items()},
                "predictions": [p.to_dict() for p in self._predictions[-200:]],
                "belief_accuracy": {k: v[-50:] for k, v in self._belief_accuracy.items()},
                "surprise_history": self._surprise_history[-100:],
                # V3 additions
                "beliefs": {k: v.to_dict() for k, v in self._beliefs.items()},
                "analogies": {k: v.to_dict() for k, v in self._analogies.items()},
                "solve_history": self._solve_history[-500:],
                "v3_stats": self._v3_stats,
            }
            import os
            tmp = self._path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fp:
                json.dump(data, fp, indent=2)
            os.replace(tmp, self._path)
            log.debug("WorldModel saved: %d facts, %d links, %d schemas, %d beliefs",
                      sum(len(v) for v in self._facts.values()),
                      len(self._causal_links), len(self._schemas), len(self._beliefs))
        except OSError as e:
            log.warning("WorldModel save error: %s", e)

    def load(self) -> None:
        """
        Load unified world model. Handles v2, v3, and v4 (combined) formats.
        Also migrates v3 data if world_model_v3.json exists.
        """
        V3_PATH = self._path.parent / "world_model_v3.json"

        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as fp:
                    data = json.load(fp)

                for domain, fact_list in data.get("facts", {}).items():
                    self._facts[domain] = [Fact.from_dict(f) for f in fact_list]
                for key, link_dict in data.get("causal_links", {}).items():
                    link = CausalLink.from_dict(link_dict)
                    self._causal_links[link.key] = link
                for name, schema_dict in data.get("schemas", {}).items():
                    self._schemas[name] = Schema.from_dict(schema_dict)
                self._predictions = [
                    Prediction(**{k: v for k, v in p.items()
                                  if k in Prediction.__dataclass_fields__})
                    for p in data.get("predictions", [])
                ]
                self._belief_accuracy = data.get("belief_accuracy", {})
                self._surprise_history = data.get("surprise_history", [])
                # V3/V4 fields
                for k, v in data.get("beliefs", {}).items():
                    try:
                        self._beliefs[k] = Belief.from_dict(v)
                    except Exception as e:
                        log.warning("[world_model] Skipped corrupt belief entry '%s': %s", k, e)
                for k, v in data.get("analogies", {}).items():
                    try:
                        self._analogies[k] = Analogy.from_dict(v)
                    except Exception as e:
                        log.warning("[world_model] Skipped corrupt analogy entry '%s': %s", k, e)
                self._solve_history = data.get("solve_history", [])
                self._v3_stats.update(data.get("v3_stats", {}))

                log.info(
                    "WorldModel loaded (v%s): %d facts, %d links, %d schemas, "
                    "%d beliefs, %d analogies",
                    data.get("version", "?"),
                    sum(len(v) for v in self._facts.values()),
                    len(self._causal_links), len(self._schemas),
                    len(self._beliefs), len(self._analogies),
                )
                if len(self._causal_links) > self._MAX_LINKS:
                    self._prune_weak_links()

                # One-time migration: absorb v3 data if present
                if V3_PATH.exists():
                    self._migrate_from_v3(V3_PATH)
                return
            except Exception as e:
                log.warning("WorldModel load error (will try legacy): %s", e)

        # Import legacy facts from world_model.json
        if self.LEGACY_PATH.exists():
            try:
                with open(self.LEGACY_PATH, encoding="utf-8") as fp:
                    legacy = json.load(fp)
                imported = 0
                for domain, fact_list in legacy.items():
                    if isinstance(fact_list, list):
                        for item in fact_list:
                            if isinstance(item, dict) and item.get("fact"):
                                self.add_fact(
                                    domain=item.get("domain", domain),
                                    fact=item["fact"],
                                    confidence=item.get("confidence", 0.9),
                                    source=item.get("source", "legacy"),
                                )
                                imported += 1
                log.info("WorldModel: imported %d legacy facts from world_model.json", imported)
            except Exception as e:
                log.warning("WorldModel legacy import error: %s", e)

    def _migrate_from_v3(self, v3_path: Path) -> None:
        """One-time: absorb beliefs, analogies, and solve_history from v3 JSON."""
        try:
            with open(v3_path, encoding="utf-8") as fp:
                v3 = json.load(fp)
            absorbed = 0
            for k, v in v3.get("beliefs", {}).items():
                if k not in self._beliefs:
                    try:
                        self._beliefs[k] = Belief.from_dict(v)
                        absorbed += 1
                    except Exception as e:
                        log.warning("[world_model] Skipped corrupt v3 belief '%s': %s", k, e)
            for k, v in v3.get("analogies", {}).items():
                if k not in self._analogies:
                    try:
                        self._analogies[k] = Analogy.from_dict(v)
                        absorbed += 1
                    except Exception as e:
                        log.warning("[world_model] Skipped corrupt v3 analogy '%s': %s", k, e)
            for ep in v3.get("solve_history", []):
                if ep not in self._solve_history:
                    self._solve_history.append(ep)
            if len(self._solve_history) > 2000:
                self._solve_history = self._solve_history[-2000:]
            log.info("WorldModel: migrated %d items from v3 JSON", absorbed)
        except Exception as e:
            log.debug("WorldModel v3 migration skipped: %s", e)

    # ─────────────────────────────────────────────────────────────────────────
    # 12. Prediction & Belief-Update Engine
    # ─────────────────────────────────────────────────────────────────────────

    def predict_transform(self, graph, transforms: list, domain: str = "general") -> "Prediction":
        """
        Before solving, predict which transform will work best.
        Uses causal links + past belief accuracy to estimate expected delta.
        Returns a Prediction object (to be fulfilled via record_outcome()).
        """
        sig = self._graph_signature(graph)

        # Build scores for each transform
        scores = {}
        for t in transforms:
            tname = t.name() if hasattr(t, 'name') else str(t)
            # Base score from belief accuracy history
            acc_list = self._belief_accuracy.get(tname, [])
            base_acc = sum(acc_list[-20:]) / len(acc_list[-20:]) if acc_list else 0.4

            # Boost from causal links matching this transform's domain (indexed)
            causal_boost = self._get_causal_boost(tname, domain)

            # Check if this transform matches any active schema
            schema_boost = 0.0
            for schema in self._schemas.values():
                if schema.domain == domain and schema.activation > 0.3:
                    schema_boost += 0.1

            scores[tname] = min(1.0, base_acc + causal_boost + schema_boost)

        if not scores:
            best_name = "unknown"
            best_score = 0.3
        else:
            best_name = max(scores, key=scores.get)
            best_score = scores[best_name]

        # Expected delta: use historical average for this transform if available
        expected = self._expected_delta_for(best_name, domain)

        pred = Prediction(
            transform_name=best_name,
            expected_delta=expected,
            graph_signature=sig,
            domain=domain,
            confidence=best_score,
        )
        return pred

    def record_outcome(self, prediction: "Prediction", actual_transforms: list,
                       actual_delta: float, domain: str = "general") -> float:
        """
        After solving, compare prediction to actual outcome.
        Updates belief accuracy, causal link confidence, surprise history.
        Returns the surprise value (0=perfect prediction, 1+=big surprise).
        """
        prediction.actual_delta = actual_delta
        expected = prediction.expected_delta
        surprise = abs(expected - actual_delta) / max(abs(expected), 0.1)
        surprise = min(surprise, 5.0)  # cap at 5x surprise
        prediction.surprise = round(surprise, 4)

        # Was the prediction directionally correct AND within 100% of magnitude?
        correct_direction = (expected > 0.1) == (actual_delta > 0.1)
        correct_magnitude = abs(expected - actual_delta) < max(abs(expected), 1.0) * 2
        prediction.was_correct = correct_direction and correct_magnitude

        # Update belief accuracy for the predicted transform
        pname = prediction.transform_name
        self._belief_accuracy.setdefault(pname, [])
        self._belief_accuracy[pname].append(1.0 if prediction.was_correct else 0.0)
        if len(self._belief_accuracy[pname]) > 100:
            self._belief_accuracy[pname] = self._belief_accuracy[pname][-100:]

        # Update accuracy for ACTUAL transforms used
        for tname in actual_transforms:
            self._belief_accuracy.setdefault(tname, [])
            self._belief_accuracy[tname].append(1.0 if actual_delta > 0.1 else 0.0)
            if len(self._belief_accuracy[tname]) > 100:
                self._belief_accuracy[tname] = self._belief_accuracy[tname][-100:]

        # Update causal link confidence via Bayesian update
        for link in self._causal_links.values():
            if link.domain == domain or link.domain == "general":
                for tname in actual_transforms:
                    if tname in link.mechanism:
                        if actual_delta > 0.1:
                            # Positive evidence: strengthen link
                            link.confidence = min(0.999, link.confidence + 0.01 * (1 - link.confidence))
                        else:
                            # Negative evidence: weaken link slightly
                            link.confidence = max(0.1, link.confidence - 0.005)
                        link.evidence_count += 1

        # Track surprise history
        self._surprise_history.append(surprise)
        if len(self._surprise_history) > 200:
            self._surprise_history = self._surprise_history[-200:]

        # Store prediction
        prediction.domain = domain
        self._predictions.append(prediction)
        if len(self._predictions) > 500:
            self._predictions = self._predictions[-500:]

        # High surprise → ask LLM to generate a hypothesis (async, best-effort)
        if surprise > 2.0:
            now = time.time()
            if now - self._last_llm_hypothesis_time >= self._llm_hypothesis_cooldown:
                self._async_generate_hypothesis(domain, prediction.transform_name,
                                                expected, actual_delta)
                self._last_llm_hypothesis_time = now

        return surprise

    def update_from_prediction_error(self, error) -> None:
        """
        Phase B integration: Called by PredictiveEngine when surprise > 1.5.

        error: PredictionError (from cognition/predictive_engine.py)
          .predicted_transform, .actual_transform, .predicted_delta, .actual_delta
          .surprise, .domain

        Decays confidence of wrong predictions, boosts correct ones.
        Triggers LLM hypothesis generation when surprise is very high.
        """
        predicted_t = getattr(error, "predicted_transform", "")
        actual_t    = getattr(error, "actual_transform", "")
        predicted_d = float(getattr(error, "predicted_delta", 0.0))
        actual_d    = float(getattr(error, "actual_delta", 0.0))
        surprise    = float(getattr(error, "surprise", 0.0))
        domain      = str(getattr(error, "domain", "general"))

        # Update belief accuracy: predicted transform confidence
        if predicted_t:
            was_correct = (predicted_t == actual_t)
            self._belief_accuracy.setdefault(predicted_t, [])
            self._belief_accuracy[predicted_t].append(1.0 if was_correct else 0.0)
            if len(self._belief_accuracy[predicted_t]) > 100:
                self._belief_accuracy[predicted_t] = self._belief_accuracy[predicted_t][-100:]

        # Update causal link confidences
        for link in self._causal_links.values():
            if link.domain not in (domain, "general"):
                continue
            if actual_t and actual_t in link.mechanism:
                if actual_d > 0.1:
                    link.confidence = min(0.999, link.confidence + 0.008 * (1 - link.confidence))
                else:
                    link.confidence = max(0.1, link.confidence - 0.004)

        # Track surprise
        self._surprise_history.append(surprise)
        if len(self._surprise_history) > 200:
            self._surprise_history = self._surprise_history[-200:]

        # Very high surprise → generate LLM hypothesis
        if surprise > 2.0 and predicted_t:
            now = time.time()
            if now - self._last_llm_hypothesis_time >= self._llm_hypothesis_cooldown:
                self._async_generate_hypothesis(domain, predicted_t, predicted_d, actual_d)
                self._last_llm_hypothesis_time = now

    def _async_generate_hypothesis(self, domain: str, transform_name: str,
                                   expected: float, actual: float) -> None:
        """Fire-and-forget: ask Qwen3.5 why prediction was wrong, store hypothesis."""
        import threading

        def _run():
            try:
                from sare.interface.llm_bridge import _call_llm
                import json as _json
                prompt = (
                    f"In {domain} math, the transform '{transform_name}' was expected to reduce "
                    f"expression complexity by {expected:.1f} but actually reduced it by {actual:.1f}. "
                    "Write ONE hypothesis explaining why as a JSON object with exactly these keys: "
                    '{"hypothesis": "...", "pattern": "...", "confidence": 0.0}. '
                    "Return ONLY valid JSON, no markdown, no explanation."
                )
                raw = _call_llm(prompt)
                raw = raw.strip().strip("`")
                if raw.startswith("json"):
                    raw = raw[4:].strip()
                hyp = _json.loads(raw)
                hyp["domain"] = domain
                hyp["transform"] = transform_name
                hyp["expected"] = expected
                hyp["actual"] = actual
                hyp["surprise"] = abs(expected - actual) / max(abs(expected), 0.1)
                hyp["evidence"] = 1
                hyp["timestamp"] = time.time()
                if not hasattr(self, "_hypotheses"):
                    self._hypotheses = []
                # Merge with existing hypothesis for same pattern or append new
                merged = False
                for existing in self._hypotheses:
                    if (existing.get("pattern") == hyp.get("pattern")
                            and existing.get("domain") == domain):
                        existing["evidence"] = existing.get("evidence", 1) + 1
                        existing["confidence"] = min(
                            0.95, existing.get("confidence", 0.3) + 0.05
                        )
                        merged = True
                        break
                if not merged:
                    self._hypotheses.append(hyp)
                    if len(self._hypotheses) > 100:
                        self._hypotheses = self._hypotheses[-100:]
                log.info(
                    "[WorldModel] LLM hypothesis for surprise %.1f in %s: %s",
                    hyp["surprise"], domain, hyp.get("hypothesis", "")[:80],
                )
                # Persist hypotheses alongside world model
                _hyp_path = Path(__file__).resolve().parents[3] / "data" / "memory" / "world_hypotheses.json"
                try:
                    import os as _os
                    _tmp = _hyp_path.parent / f"{_hyp_path.stem}.{os.getpid()}.tmp"
                    _tmp.write_text(_json.dumps(self._hypotheses, indent=2))
                    _os.replace(_tmp, _hyp_path)
                except Exception:
                    pass
            except Exception as exc:
                log.debug("[WorldModel] LLM hypothesis generation failed: %s", exc)

        threading.Thread(target=_run, daemon=True, name="WorldModelHypothesis").start()

    def get_high_surprise_domains(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Return domains where prediction accuracy is lowest (most to learn).
        These are the most valuable curriculum targets.
        """
        domain_surprise: Dict[str, List[float]] = {}
        for p in self._predictions[-200:]:
            domain_surprise.setdefault(p.domain, []).append(p.surprise)

        ranked = []
        for domain, surprises in domain_surprise.items():
            avg = sum(surprises) / len(surprises)
            ranked.append((domain, round(avg, 3)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def _expected_delta_for(self, transform_name: str, domain: str) -> float:
        """Estimate expected energy delta based on past outcomes for this transform."""
        # Use recent predictions for this transform where it was actually used
        relevant = [
            p.actual_delta for p in self._predictions[-100:]
            if transform_name in (p.transform_name,)
            and p.actual_delta > 0
            and p.domain == domain
        ]
        if relevant:
            return round(sum(relevant) / len(relevant), 3)
        # Fallback: use overall average if we have enough data
        all_deltas = [
            p.actual_delta for p in self._predictions[-200:]
            if p.actual_delta > 0
        ]
        if len(all_deltas) >= 10:
            return round(sum(all_deltas) / len(all_deltas), 3)
        # Bootstrap defaults (calibrated to actual observed deltas)
        defaults = {
            "add_zero_elim": 3.1, "mul_one_elim": 3.1,
            "const_fold": 3.0, "double_neg": 4.0,
            "mul_zero_elim": 3.1, "distributive_expand": 5.0,
            "combine_like_terms": 5.0, "linear_eq_solve": 6.0,
            "additive_cancellation": 4.0, "self_subtraction": 3.5,
        }
        return defaults.get(transform_name, 4.0)

    def _graph_signature(self, graph) -> str:
        """Compact fingerprint of a graph for prediction lookup."""
        try:
            ops = sorted([n.label for n in graph.nodes if n.type == "operator"])
            consts = [n.label for n in graph.nodes if n.type == "constant"]
            return f"ops={'|'.join(ops)}_nc={graph.node_count}_cc={len(consts)}"
        except Exception:
            return "unknown"

    def prediction_stats(self) -> dict:
        """Summary of prediction accuracy and surprise levels."""
        if not self._predictions:
            return {"total": 0, "accuracy": 0.0, "avg_surprise": 0.0}

        recent = self._predictions[-100:]
        accuracy = sum(1 for p in recent if p.was_correct) / len(recent)
        avg_surprise = sum(p.surprise for p in recent) / len(recent)

        # Per-transform accuracy
        transform_acc = {}
        for tname, acc_list in self._belief_accuracy.items():
            if acc_list:
                transform_acc[tname] = round(sum(acc_list[-20:]) / len(acc_list[-20:]), 3)

        return {
            "total": len(self._predictions),
            "accuracy": round(accuracy, 3),
            "avg_surprise": round(avg_surprise, 3),
            "current_surprise": round(self._surprise_history[-1], 3) if self._surprise_history else 0.0,
            "high_surprise_domains": self.get_high_surprise_domains(),
            "transform_accuracy": transform_acc,
        }

    # ── W2: LLM Schema Learning ────────────────────────────────────────────────

    def learn_schema_from_llm(self, domain: str) -> Optional[dict]:
        """
        After 10+ successes in a domain, ask Qwen3.5 to synthesize a domain schema.
        Returns the schema dict if successful, None otherwise.
        """
        # Collect successful predictions for this domain
        successes = [
            p for p in self._predictions[-200:]
            if p.domain == domain and p.actual_delta > 0.5
        ]
        if len(successes) < 10:
            return None  # Not enough data yet

        # Build summary for LLM
        examples = []
        for p in successes[:15]:
            examples.append({
                "transform": p.transform_name,
                "delta": round(p.actual_delta, 2),
            })

        try:
            from sare.interface.llm_bridge import _call_llm
            import json as _j
            prompt = (
                f"You are analyzing patterns in {domain} math simplification.\n"
                f"These transforms worked well (energy reduction > 0.5):\n"
                + "\n".join(f"  {e['transform']}: delta={e['delta']}" for e in examples)
                + "\n\nSynthesize a schema JSON with keys:\n"
                '{"name": "...", "rules": ["rule1", "rule2", ...], '
                '"patterns": ["pattern1", ...], "best_transforms": ["t1", "t2", ...]}\n'
                "Return ONLY valid JSON, no markdown."
            )
            raw = _call_llm(prompt)
            raw = raw.strip().strip("`").lstrip("json").strip()
            schema_data = _j.loads(raw)
            schema_data["domain"] = domain
            schema_data["evidence"] = len(successes)
            schema_data["learned_at"] = time.time()

            # Store as a Schema object
            schema = Schema(
                name=schema_data.get("name", f"llm_schema_{domain}"),
                slots={p: "transform" for p in schema_data.get("best_transforms", [])},
                constraints=schema_data.get("rules", []),
                examples=[(e["transform"], e["delta"]) for e in examples[:5]],
                domain=domain,
            )
            self._schemas[schema.name] = schema
            log.info("[WorldModel] LLM schema learned for %s: %s", domain, schema.name)
            return schema_data
        except Exception as exc:
            log.debug("[WorldModel] Schema learning failed for %s: %s", domain, exc)
            return None

    # ── W3: Contradiction Detection via LLM ───────────────────────────────────

    def check_rule_consistency_with_llm(self, rule_name_a: str, rule_name_b: str) -> dict:
        """
        Ask Qwen3.5 whether two promoted rules are consistent.
        Returns {"consistent": bool, "reason": str, "action": str}.
        """
        try:
            from sare.interface.llm_bridge import _call_llm
            import json as _j
            prompt = (
                f"Are these two math simplification rules consistent with each other?\n"
                f"Rule A: '{rule_name_a}'\n"
                f"Rule B: '{rule_name_b}'\n"
                "Answer as JSON: "
                '{"consistent": true/false, "reason": "...", "action": "keep_both|prefer_a|prefer_b|investigate"}\n'
                "Return ONLY valid JSON."
            )
            raw = _call_llm(prompt)
            raw = raw.strip().strip("`").lstrip("json").strip()
            result = _j.loads(raw)
            result.setdefault("consistent", True)
            result.setdefault("reason", "LLM assessment")
            result.setdefault("action", "keep_both")
            log.info(
                "[WorldModel] Consistency check %s vs %s: consistent=%s",
                rule_name_a, rule_name_b, result["consistent"],
            )
            return result
        except Exception as exc:
            log.debug("[WorldModel] Consistency check failed: %s", exc)
            return {"consistent": True, "reason": f"check_failed:{exc}", "action": "keep_both"}

    def check_all_rules_consistency(self, rule_names: List[str]) -> List[dict]:
        """
        Run pairwise LLM consistency checks on recently promoted rules.
        Only checks rules that have been promoted recently (last 10).
        Returns list of conflict reports.
        """
        conflicts = []
        recent = rule_names[-10:] if len(rule_names) > 10 else rule_names
        # Only check adjacent pairs to limit LLM calls (O(n) not O(n^2))
        for i in range(len(recent) - 1):
            result = self.check_rule_consistency_with_llm(recent[i], recent[i + 1])
            if not result.get("consistent", True):
                conflicts.append({
                    "rule_a": recent[i],
                    "rule_b": recent[i + 1],
                    **result,
                })
        return conflicts

    # ─────────────────────────────────────────────────────────────────────────
    # 13. Solve & Rule Observation Hooks
    # ─────────────────────────────────────────────────────────────────────────

    def observe_solve(self, expression: str, transforms_used: List[str],
                      energy_delta: float, domain: str, solved: bool = True) -> None:
        """
        Called after each solve attempt. Induces/reinforces causal links,
        updates belief accuracy, Bayesian beliefs, and solve history for
        schema induction (v2 + v3 unified learning hook).
        """
        self._v3_stats["total_observations"] += 1

        if solved and energy_delta > 0.05 and transforms_used:
            # V2: per-transform causal links
            for tname in transforms_used:
                cause = f"pattern_with_{tname}_applicable"
                effect = f"energy_reduced_by_{tname}"
                self.add_causal_link(cause, effect, tname, domain,
                                     confidence=min(0.9, 0.5 + energy_delta * 0.1))
                self._belief_accuracy.setdefault(tname, [])
                self._belief_accuracy[tname].append(1.0)
                if len(self._belief_accuracy[tname]) > 100:
                    self._belief_accuracy[tname] = self._belief_accuracy[tname][-100:]

            # V2: sequential dependency links
            for i in range(len(transforms_used) - 1):
                t1, t2 = transforms_used[i], transforms_used[i + 1]
                self.add_causal_link(f"{t1}_completed", f"{t2}_applicable",
                                     "sequential_dependency", domain, confidence=0.4)

            # V2: expression-level fact
            if expression and len(expression) < 100:
                self.add_fact(domain, f"{expression} solvable via {','.join(transforms_used[:3])}",
                              confidence=0.7, source="solve_observation")

            # V3: CausalDiscovery for richer links
            new_links = CausalDiscovery.induce_from_solve(
                expression, transforms_used, energy_delta, domain, self._causal_links
            )
            self._v3_stats["links_discovered"] += len(new_links)

            # V3: Bayesian belief update (per-transform)
            for tname in transforms_used:
                bkey = f"transform:{tname}:effective_in:{domain}"
                if bkey not in self._beliefs:
                    self._beliefs[bkey] = Belief(
                        subject=tname, predicate=f"effective in {domain}",
                        confidence=0.5, domain=domain
                    )
                self._beliefs[bkey].update(supports=True)

            # V3: domain solvable belief
            dkey = f"domain:{domain}:solvable"
            if dkey not in self._beliefs:
                self._beliefs[dkey] = Belief(subject=domain, predicate="solvable",
                                             confidence=0.5, domain=domain)
            self._beliefs[dkey].update(supports=True)

        else:
            # Weaken on failure (v2 + v3)
            for tname in transforms_used:
                self._belief_accuracy.setdefault(tname, [])
                self._belief_accuracy[tname].append(0.0)
                if len(self._belief_accuracy[tname]) > 100:
                    self._belief_accuracy[tname] = self._belief_accuracy[tname][-100:]
            CausalDiscovery.weaken_from_failure(transforms_used, domain, self._causal_links)
            for tname in transforms_used:
                bkey = f"transform:{tname}:effective_in:{domain}"
                if bkey in self._beliefs:
                    self._beliefs[bkey].update(supports=False)

        # V3: Record in solve history for periodic schema induction
        self._solve_history.append({
            "expression": expression, "transforms": transforms_used,
            "delta": energy_delta, "domain": domain, "success": solved,
            "timestamp": time.time(),
        })
        if len(self._solve_history) > 2000:
            self._solve_history = self._solve_history[-2000:]

        # V3: Periodically induce schemas (every 10 observations)
        if self._v3_stats["total_observations"] % 10 == 0:
            new_schemas = SchemaInduction.induce_schemas(
                self._solve_history, self._schemas, min_exemplars=3
            )
            self._v3_stats["schemas_induced"] += len(new_schemas)
            if new_schemas:
                log.debug("WorldModel: induced %d new schemas", len(new_schemas))

        # V3: Decay schema activations
        for schema in self._schemas.values():
            schema.decay(0.01)

        self._causal_idx_dirty = True

        # Reactive wiring: publish "surprise_high" when recent average surprise > 2.5
        try:
            if self._surprise_history:
                _recent = self._surprise_history[-10:]
                _avg_surprise = sum(_recent) / len(_recent)
                if _avg_surprise > 2.5:
                    from sare.core.event_bus import get_event_bus
                    get_event_bus().publish("surprise_high", {
                        "domain": domain,
                        "avg_surprise": round(_avg_surprise, 3),
                        "expression": expression,
                    })
        except Exception:
            pass

    def observe_rule_promotion(self, rule_name: str, domain: str, pattern: str = "",
                               confidence: float = 0.9) -> None:
        """Called when a rule is promoted. Records it as a domain fact and reinforces its causal link."""
        self.add_fact(domain, f"Rule '{rule_name}' is valid in {domain}: {pattern}",
                      confidence=confidence, source="rule_promotion")
        # Strengthen the causal link for this rule if it exists
        cause = f"pattern_with_{rule_name}_applicable"
        effect = f"energy_reduced_by_{rule_name}"
        key = f"{cause}→{effect}"
        if key in self._causal_links:
            self._causal_links[key].evidence_count += 10  # strong boost
            self._causal_links[key].confidence = min(0.99, self._causal_links[key].confidence + 0.1)
        else:
            self.add_causal_link(cause, effect, rule_name, domain, confidence=confidence)
        self._causal_idx_dirty = True

    # ─────────────────────────────────────────────────────────────────────────
    # 14. V3-style Predict / Analogy / Consistency
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, expression: str, domain: str,
                available_transforms: List[str]) -> dict:
        """
        V3-style prediction: returns ranked transform list with Bayesian belief scores.
        Uses beliefs (v3) + belief_accuracy (v2) + causal link boosts.
        """
        scores: Dict[str, float] = {}
        for t in available_transforms:
            bkey = f"transform:{t}:effective_in:{domain}"
            if bkey in self._beliefs:
                scores[t] = self._beliefs[bkey].confidence
            else:
                acc = self._belief_accuracy.get(t, [])
                scores[t] = sum(acc[-20:]) / len(acc[-20:]) if acc else 0.3
            scores[t] = min(1.0, scores[t] + self._get_causal_boost(t, domain))

        ranked = sorted(scores, key=scores.get, reverse=True)[:3]
        # Historical expected delta
        relevant = [ep["delta"] for ep in self._solve_history[-200:]
                    if ep.get("domain") == domain and ep.get("success")]
        expected_delta = sum(relevant) / len(relevant) if relevant else 2.0
        return {
            "predicted_transforms": ranked,
            "scores": {t: round(scores.get(t, 0.3), 3) for t in ranked},
            "expected_delta": round(expected_delta, 3),
            "domain": domain,
        }

    def discover_analogies(self) -> List[Analogy]:
        """
        Discover structural parallels between domains by comparing
        transform patterns and their roles in solve history.
        """
        domain_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for ep in self._solve_history:
            if not ep.get("success"):
                continue
            domain = ep.get("domain", "general")
            for t in ep.get("transforms", []):
                role = self._classify_transform_role(t)
                domain_patterns[domain][role] += 1

        domains = list(domain_patterns.keys())
        new_analogies: List[Analogy] = []

        for i, d1 in enumerate(domains):
            for d2 in domains[i + 1:]:
                roles1 = set(domain_patterns[d1].keys())
                roles2 = set(domain_patterns[d2].keys())
                shared = roles1 & roles2
                if len(shared) < 2:
                    continue

                mapping: Dict[str, str] = {}
                from collections import Counter
                for role in shared:
                    t1_list = [t for ep in self._solve_history
                               if ep.get("domain") == d1 and ep.get("success")
                               for t in ep.get("transforms", [])
                               if self._classify_transform_role(t) == role]
                    t2_list = [t for ep in self._solve_history
                               if ep.get("domain") == d2 and ep.get("success")
                               for t in ep.get("transforms", [])
                               if self._classify_transform_role(t) == role]
                    if t1_list and t2_list:
                        mapping[Counter(t1_list).most_common(1)[0][0]] = Counter(t2_list).most_common(1)[0][0]
                if not mapping:
                    continue

                overlap = len(shared) / max(len(roles1 | roles2), 1)
                confidence = min(0.9, overlap * 0.7 + 0.1 * min(len(mapping), 5))
                for src_c, tgt_c in mapping.items():
                    analogy = Analogy(
                        source_domain=d1, source_concept=src_c,
                        target_domain=d2, target_concept=tgt_c,
                        structural_mapping=mapping, confidence=confidence,
                    )
                    if analogy.key not in self._analogies:
                        self._analogies[analogy.key] = analogy
                        new_analogies.append(analogy)

        self._v3_stats["analogies_found"] += len(new_analogies)
        if new_analogies:
            log.info("WorldModel: discovered %d new analogies", len(new_analogies))
        return new_analogies

    def check_consistency(self, cause: str, effect: str,
                          mechanism: str, domain: str) -> dict:
        """
        V3-style: check if a proposed causal link is consistent with existing
        knowledge and Bayesian beliefs.
        """
        new_link = CausalLink(cause=cause, effect=effect, mechanism=mechanism, domain=domain)
        return ContradictionDetector.check(new_link, self._causal_links, self._beliefs)

    def update_belief(self, subject: str, predicate: str, value: str,
                      confidence: float = 0.75, domain: str = "general") -> None:
        """Store a structured (subject, predicate, value) triple as a queryable belief.

        This is the primary write path for FactIngester — creates beliefs that
        get_belief() can retrieve directly without scanning prose facts.
        """
        key = f"{subject.lower()}::{predicate.lower()}"
        existing = self._beliefs.get(key)
        if existing is not None:
            # Update confidence toward new value (Bayesian-ish blend)
            existing.confidence = min(1.0, existing.confidence * 0.7 + confidence * 0.3)
            existing.value = value  # use latest answer
        else:
            self._beliefs[key] = Belief(
                key=key,
                value=value,
                confidence=confidence,
                domain=domain,
                subject=subject.lower(),
                predicate=predicate.lower(),
            )

    def get_belief(self, subject: str, predicate: str) -> Optional["Belief"]:
        """Direct structured lookup: (subject, predicate) → Belief or None."""
        key = f"{subject.lower()}::{predicate.lower()}"
        return self._beliefs.get(key)

    def search_beliefs(self, query: str, domain: Optional[str] = None,
                       top_k: int = 5) -> List[dict]:
        """Keyword search over structured beliefs. Returns top_k by confidence."""
        q = query.lower()
        matches = []
        for b in self._beliefs.values():
            if domain and b.domain != domain:
                continue
            score = 0.0
            if b.subject and b.subject in q:
                score += 0.6
            if b.predicate and b.predicate in q:
                score += 0.3
            if b.value and any(w in q for w in b.value.lower().split()[:4]):
                score += 0.1
            if score > 0:
                matches.append((score * b.confidence, b))
        matches.sort(key=lambda x: x[0], reverse=True)
        return [b.to_dict() for _, b in matches[:top_k]]

    def get_beliefs(self, domain: Optional[str] = None) -> List[dict]:
        """Return tracked Bayesian beliefs, optionally filtered by domain."""
        beliefs = self._beliefs.values()
        if domain:
            beliefs = [b for b in beliefs if b.domain == domain]
        return [b.to_dict() for b in sorted(beliefs, key=lambda b: b.confidence, reverse=True)]

    def expire_stale_beliefs(self, max_age_seconds: float = 86400,
                             decay_factor: float = 0.85) -> int:
        """Decay confidence of beliefs older than max_age_seconds.
        Beliefs with valid_until == -1 are pinned and never decayed.
        Returns count of beliefs decayed."""
        now = time.time()
        decayed = 0
        for b in self._beliefs.values():
            if b.valid_until == -1:
                continue  # explicitly pinned
            age = now - (b.last_updated or now)
            if age > max_age_seconds and b.confidence > 0.05:
                b.confidence = round(b.confidence * decay_factor, 4)
                decayed += 1
        return decayed

    def get_analogies(self, domain: Optional[str] = None) -> List[dict]:
        """Return discovered analogies, optionally filtered by domain."""
        analogies = self._analogies.values()
        if domain:
            analogies = [a for a in analogies
                         if a.source_domain == domain or a.target_domain == domain]
        return [a.to_dict() for a in analogies]

    def _classify_transform_role(self, transform_name: str) -> str:
        """Classify a transform name into a structural role (for analogy discovery)."""
        t = transform_name.lower()
        if "zero" in t and ("add" in t or "elim" in t): return "additive_identity"
        if "one"  in t and ("mul" in t or "elim" in t): return "multiplicative_identity"
        if "zero" in t and "mul" in t:                  return "annihilation"
        if "neg" in t or "double" in t:                 return "involution"
        if "fold" in t or "const" in t:                 return "evaluation"
        if "subtract" in t or "self" in t:              return "self_inverse"
        if "solve" in t or "equation" in t:             return "equation_solving"
        if "factor" in t:                               return "factoring"
        if "distribut" in t or "expand" in t:           return "distribution"
        if "cancel" in t:                               return "cancellation"
        if "combin" in t or "like" in t:                return "combination"
        if "power" in t or "exp" in t:                  return "exponentiation"
        if "commut" in t:                               return "commutativity"
        return "general"

    def _seed_if_empty(self) -> None:
        """Add seed causal links and schemas if not already present."""
        for seed in _SEED_CAUSAL_LINKS:
            link = CausalLink(
                cause=seed["cause"],
                effect=seed["effect"],
                mechanism=seed["mechanism"],
                confidence=seed["confidence"],
                domain=seed["domain"],
            )
            if link.key not in self._causal_links:
                self._causal_links[link.key] = link

        for s_dict in _SEED_SCHEMAS:
            name = s_dict["name"]
            if name not in self._schemas:
                self._schemas[name] = Schema.from_dict({**s_dict, "created_at": time.time()})

        log.debug(
            "WorldModel seeded: %d causal links, %d schemas",
            len(self._causal_links),
            len(self._schemas),
        )

    # ── Auto-learning: proof enrichment & distillation ─────────────────────────

    def enrich_from_proof(self, proof_steps: List[str], domain: str,
                          solved_expr: str, energy_delta: float) -> None:
        """
        Synchronous (no LLM). After a successful solve, record which transforms
        were used and strengthen their causal links for this domain.
        Increments solve count and triggers async LLM learning at thresholds.
        """
        import threading as _t
        # Strengthen domain causal link: each proof step → domain success
        for step in proof_steps:
            if not isinstance(step, str):
                continue
            # Add/reinforce: applying transform X in domain Y simplifies expressions
            cause = f"{step}({solved_expr[:40]})"
            effect = "simplified"
            self.add_causal_link(cause, effect, f"{step} reduces energy", domain,
                                 min(0.98, 0.7 + 0.05 * energy_delta))

        # Increment domain solve count
        self._solve_counts[domain] = self._solve_counts.get(domain, 0) + 1
        count = self._solve_counts[domain]

        # Every _distill_interval solves: trigger LLM distillation in background
        cooldown = 300  # 5-minute cooldown between distillations per domain
        last = self._last_distill.get(domain, 0.0)
        if count % self._distill_interval == 0 and (time.time() - last) > cooldown:
            self._last_distill[domain] = time.time()
            links = self.get_causal_links(domain)
            facts = self.get_facts(domain)
            _t.Thread(
                target=self._async_distill, args=(domain, links, facts), daemon=True
            ).start()

        # Every 10 solves: trigger LLM proof learning for deeper rule extraction
        if count % 10 == 0 and proof_steps and llm_available_check():
            _t.Thread(
                target=self._async_learn_from_proof,
                args=(solved_expr, domain, proof_steps, energy_delta),
                daemon=True,
            ).start()

    def _async_learn_from_proof(self, expr: str, domain: str,
                                 proof_steps: List[str], energy_delta: float) -> None:
        """Background: ask LLM to extract abstract knowledge from a proof."""
        try:
            from sare.interface.llm_bridge import learn_from_proof
            result = learn_from_proof(expr, domain, proof_steps, energy_delta)
            fa = result.get("facts_added", 0)
            la = result.get("links_added", 0)
            variants = result.get("variants", [])
            if fa or la or variants:
                log.info("[WorldModel] Proof learning: +%d facts, +%d links, %d variants in %s",
                         fa, la, len(variants), domain)
                self.log_activity("proof_learn", domain,
                    f"Extracted {fa} facts, {la} causal links from proof of '{expr[:40]}'",
                    facts_added=fa, links_added=la)
            # Store variants as seeds for curriculum
            if variants:
                self._pending_variants = getattr(self, "_pending_variants", [])
                self._pending_variants.extend([(v, domain) for v in variants])
                self._pending_variants = self._pending_variants[-50:]  # cap
        except Exception as e:
            log.debug("[WorldModel] _async_learn_from_proof error: %s", e)

    def _async_distill(self, domain: str, links: List[dict], facts: List[dict]) -> None:
        """Background: compress accumulated knowledge into higher-level rules."""
        try:
            from sare.interface.llm_bridge import distill_domain_knowledge
            result = distill_domain_knowledge(domain, links, facts)
            fa = result.get("facts_added", 0)
            la = result.get("links_added", 0)
            rules = result.get("distilled_rules", [])
            if fa or la or rules:
                log.info("[WorldModel] Distilled %s: +%d facts, +%d links, %d rules",
                         domain, fa, la, len(rules))
                rule_names = ", ".join(r.get("name","?") for r in rules[:3])
                self.log_activity("distill", domain,
                    f"Distilled {len(links)} links → {len(rules)} principles ({rule_names})",
                    facts_added=fa, links_added=la)
        except Exception as e:
            log.debug("[WorldModel] _async_distill error: %s", e)

    def pop_pending_variants(self) -> List[tuple]:
        """Return and clear LLM-generated problem variants (expr, domain) pairs."""
        variants = getattr(self, "_pending_variants", [])
        self._pending_variants = []
        return variants

    def log_activity(self, event_type: str, domain: str, message: str,
                     facts_added: int = 0, links_added: int = 0) -> None:
        """Record an auto-learning event for UI display."""
        self._activity_log.append({
            "ts": time.time(),
            "type": event_type,   # proof_learn | failure_analysis | distill | reflection | schema
            "domain": domain,
            "message": message,
            "facts_added": facts_added,
            "links_added": links_added,
        })
        self._activity_log = self._activity_log[-50:]   # keep last 50

    def get_activity_log(self) -> List[dict]:
        """Return recent auto-learning events, newest first."""
        return list(reversed(self._activity_log))


# ── Module-level integration points ───────────────────────────────────────────

_SINGLETON: Optional[WorldModel] = None


def get_world_model() -> WorldModel:
    """Singleton accessor — creates and loads on first call."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = WorldModel()
    return _SINGLETON


def on_rule_promoted(rule_name: str, domain: str, pattern: str) -> dict:
    """
    Called after a rule is promoted by CausalInduction.

    Checks the new rule for consistency with existing world knowledge,
    and if consistent, adds a causal link recording the promotion.

    Returns:
        Consistency check result dict.
    """
    wm = get_world_model()
    result = wm.check_rule_consistency(rule_name, pattern, domain)
    if result["consistent"]:
        wm.add_causal_link(
            cause=pattern,
            effect="simplified",
            mechanism=rule_name,
            domain=domain,
            confidence=0.85,
        )
        wm.save()
    return result
