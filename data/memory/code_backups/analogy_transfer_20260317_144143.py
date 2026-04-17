"""
AnalogyTransfer — TODO-I: Cross-Domain Structural Generalisation
================================================================
Implements structural analogy transfer for SARE-HX.

The core insight: a rule that reduces energy in arithmetic ("a + 0 = a")
has a structurally identical counterpart in logic ("p ∧ ⊤ = p") and
algebra ("a + identity = a"). If SARE learns the arithmetic rule, it
should automatically *transfer* that rule to other domains.

This is what human mathematicians do: they recognise that the
"multiplicative identity" law in integers generalises to matrices,
polynomials, functions, and beyond — without relearning from scratch.

Algorithm:
    1. Build a structural signature for each ConceptRegistry rule
       (node-type pattern, arity, operator type)
    2. Cluster rules by structural similarity across domains
    3. For each cluster, generate *transfer rules* for missing domains
    4. Register transfer rules with lower confidence (0.6–0.8) so they
       are tried but can be rejected if they don't reduce energy

Usage::

    transfer = AnalogyTransfer(concept_registry)
    suggestions = transfer.transfer_rules(source_domain="arithmetic",
                                          target_domain="algebra")
    transfer.apply_suggestions(suggestions)  # registers in ConceptRegistry
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


# ── Rule structural signature ─────────────────────────────────────────────────

@dataclass
class StructuralSignature:
    """Abstract shape of a rule, domain-independent."""
    arity:           int          # number of operands (1=unary, 2=binary, 3=ternary)
    operator_type:   str          # "identity","annihilator","involution","absorption","distribution"
    operand_pattern: str          # e.g. "X op CONST" or "op(op(X))"
    operator_symbol: str          # symbolic category: "add","mul","and","or","not","xor"
    domain_family:   str          # "arithmetic","logic","algebra","sets"


@dataclass
class TransferRule:
    """A rule derived by analogical transfer from source → target domain."""
    name:            str
    source_rule:     str          # original rule name
    source_domain:   str
    target_domain:   str
    pattern_description: str
    replacement_description: str
    confidence:      float        # always lower than source (0.5–0.8)
    observations:    int          # start low (transfer hypothesis)
    reasoning:       str          # explanation of why this transfer was made
    notes:           str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":                  self.name,
            "source_rule":           self.source_rule,
            "source_domain":         self.source_domain,
            "target_domain":         self.target_domain,
            "pattern_description":   self.pattern_description,
            "replacement_description": self.replacement_description,
            "confidence":            round(self.confidence, 3),
            "observations":          self.observations,
            "reasoning":             self.reasoning,
            "notes":                 self.notes,
        }


# ── Structural analogy library ────────────────────────────────────────────────

# Maps (structural_type, source_domain) → target_domain_templates
_ANALOGY_MAP: Dict[Tuple[str, str], List[Dict]] = {
    # Additive identity: a + 0 = a → a ∧ ⊤ = a (logic), a + 0 = a (algebra)
    ("identity_add", "arithmetic"): [
        {"target": "logic",   "pattern": "p AND TRUE",  "replacement": "p",
         "conf": 0.90, "note": "p ∧ ⊤ = p (analogous to a+0=a)"},
        {"target": "algebra", "pattern": "x + zero_element", "replacement": "x",
         "conf": 0.80, "note": "Additive identity generalises to any ring"},
        {"target": "sets",    "pattern": "A ∪ ∅",       "replacement": "A",
         "conf": 0.90, "note": "A ∪ ∅ = A (set-theoretic additive identity)"},
    ],
    # Multiplicative identity: a * 1 = a
    ("identity_mul", "arithmetic"): [
        {"target": "logic",   "pattern": "p OR FALSE",  "replacement": "p",
         "conf": 0.85, "note": "p ∨ ⊥ = p (analogous to a*1=a in some interpretations)"},
        {"target": "sets",    "pattern": "A ∩ U",       "replacement": "A",
         "conf": 0.90, "note": "A ∩ U = A (universal set as multiplicative identity)"},
    ],
    # Annihilator: a * 0 = 0
    ("annihilator_mul", "arithmetic"): [
        {"target": "logic",   "pattern": "p AND FALSE", "replacement": "FALSE",
         "conf": 0.92, "note": "p ∧ ⊥ = ⊥ (logical annihilator)"},
        {"target": "sets",    "pattern": "A ∩ ∅",       "replacement": "∅",
         "conf": 0.92, "note": "A ∩ ∅ = ∅ (set annihilator)"},
    ],
    # Idempotency: a + a = a (logic), a * a = a (sets)
    ("idempotent", "logic"): [
        {"target": "sets",    "pattern": "A ∪ A",       "replacement": "A",
         "conf": 0.92, "note": "A ∪ A = A (idempotency), analogous to p∨p=p"},
        {"target": "sets",    "pattern": "A ∩ A",       "replacement": "A",
         "conf": 0.92, "note": "A ∩ A = A (idempotency), analogous to p∧p=p"},
    ],
    # Involution: ¬¬p = p → --x = x
    ("involution", "logic"): [
        {"target": "arithmetic", "pattern": "neg(neg(x))", "replacement": "x",
         "conf": 0.85, "note": "Double negation in arithmetic: --x = x"},
        {"target": "algebra",    "pattern": "(-(-x))",      "replacement": "x",
         "conf": 0.85, "note": "Double negation: -(-x) = x"},
    ],
    # De Morgan: arithmetic has no direct analog but code does
    ("de_morgan", "logic"): [
        {"target": "code",    "pattern": "NOT(a AND b)",  "replacement": "NOT(a) OR NOT(b)",
         "conf": 0.88, "note": "De Morgan in code: !(a && b) == !a || !b"},
    ],
    # Subtractive self: a - a = 0 → analogous to p XOR p = FALSE
    ("self_inverse", "arithmetic"): [
        {"target": "logic",   "pattern": "p XOR p",      "replacement": "FALSE",
         "conf": 0.88, "note": "p⊕p = ⊥ analogous to a-a=0"},
        {"target": "sets",    "pattern": "A - A",         "replacement": "∅",
         "conf": 0.85, "note": "A-A = ∅ (set-theoretic self-subtraction)"},
    ],
    # Distributive: a*(b+c) = ab+ac → arithmetic to algebra
    ("distributive", "arithmetic"): [
        {"target": "algebra", "pattern": "a * (b + c)",  "replacement": "(a*b) + (a*c)",
         "conf": 0.85, "note": "Distributive law carries to algebra"},
        {"target": "logic",   "pattern": "p AND (q OR r)", "replacement": "(p AND q) OR (p AND r)",
         "conf": 0.90, "note": "Logical distributivity: p∧(q∨r) = (p∧q)∨(p∧r)"},
        {"target": "sets",    "pattern": "A ∩ (B ∪ C)",  "replacement": "(A∩B) ∪ (A∩C)",
         "conf": 0.90, "note": "Set distributivity: A∩(B∪C) = (A∩B)∪(A∩C)"},
    ],
    # Complement: A ∪ Aᶜ = U → p ∨ ¬p = ⊤
    ("complement_max", "sets"): [
        {"target": "logic",   "pattern": "p OR NOT(p)",  "replacement": "TRUE",
         "conf": 0.95, "note": "Excluded middle analogous to A ∪ Aᶜ = U"},
    ],
    ("complement_min", "sets"): [
        {"target": "logic",   "pattern": "p AND NOT(p)", "replacement": "FALSE",
         "conf": 0.95, "note": "Contradiction analogous to A ∩ Aᶜ = ∅"},
    ],
}

# Structural type classifier: rule name → structural type
_RULE_TYPE_PATTERNS = [
    (r"additive_identity",    "identity_add"),
    (r"multiplicative_identity", "identity_mul"),
    (r"multiplicative_zero",  "annihilator_mul"),
    (r"and_false",            "annihilator_mul"),
    (r"or_true",              "annihilator_add"),
    (r"idempotent",           "idempotent"),
    (r"double_negation",      "involution"),
    (r"de_morgan",            "de_morgan"),
    (r"subtractive_self",     "self_inverse"),
    (r"xor_self",             "self_inverse"),
    (r"distributive",         "distributive"),
    (r"complement_union",     "complement_max"),
    (r"complement_intersect", "complement_min"),
    (r"and_true",             "identity_mul"),
    (r"or_false",             "identity_add"),
]


# ── AnalogyTransfer ───────────────────────────────────────────────────────────

class AnalogyTransfer:
    """
    Cross-domain analogy transfer engine.

    Given knowledge of rules in one domain, auto-generates candidate rules
    in structurally analogous domains. Confidence is always lower than the
    source (SARE must still validate them via CausalInduction).
    """

    def __init__(self, concept_registry=None, confidence_decay: float = 0.15):
        """
        Parameters
        ----------
        concept_registry : live C++ ConceptRegistry (optional, for reading/writing rules).
        confidence_decay : how much to reduce confidence for transferred rules.
        """
        self.registry = concept_registry
        self.decay    = confidence_decay
        self._applied: Set[str] = set()  # track already-applied transfers

    # ── Main API ──────────────────────────────────────────────────────────

    def transfer_from_domain(
        self,
        source_domain: str,
        target_domain: Optional[str] = None,
        top_k: int = 20,
    ) -> List[TransferRule]:
        """
        Find all analogical transfers available from source_domain.
        If target_domain is None, generate for ALL domains.

        Returns list of TransferRule objects to apply.
        """
        transfers: List[TransferRule] = []

        for (struct_type, src_dom), templates in _ANALOGY_MAP.items():
            if src_dom != source_domain:
                continue
            for tmpl in templates:
                if target_domain and tmpl["target"] != target_domain:
                    continue
                rule_name = f"transfer_{struct_type}_{source_domain}_to_{tmpl['target']}"
                if rule_name in self._applied:
                    continue
                tr = TransferRule(
                    name=rule_name,
                    source_rule=struct_type,
                    source_domain=source_domain,
                    target_domain=tmpl["target"],
                    pattern_description=tmpl["pattern"],
                    replacement_description=tmpl["replacement"],
                    confidence=max(0.4, tmpl["conf"] - self.decay),
                    observations=100,  # start with modest evidence
                    reasoning=(
                        f"Analogical transfer from '{source_domain}' to '{tmpl['target']}'. "
                        f"Structural type: {struct_type}. "
                        f"Source pattern: {tmpl['pattern']}. {tmpl['note']}"
                    ),
                    notes=tmpl["note"],
                )
                transfers.append(tr)

        return transfers[:top_k]

    def transfer_all_domains(self) -> List[TransferRule]:
        """Generate the complete cross-domain transfer set."""
        all_transfers: List[TransferRule] = []
        seen_srcs = set(src for _, src in _ANALOGY_MAP.keys())
        for src_domain in seen_srcs:
            all_transfers.extend(self.transfer_from_domain(src_domain))
        return all_transfers

    def apply_to_registry(
        self, transfers: List[TransferRule]
    ) -> Tuple[int, int]:
        """
        Register transfer rules into ConceptRegistry.

        Returns (applied_count, skipped_count).
        """
        applied = 0
        skipped = 0
        for tr in transfers:
            if tr.name in self._applied:
                skipped += 1
                continue
            if self.registry:
                try:
                    self.registry.add_rule(
                        name=tr.name,
                        domain=tr.target_domain,
                        pattern_description=tr.pattern_description,
                        replacement_description=tr.replacement_description,
                        confidence=tr.confidence,
                        observations=tr.observations,
                    )
                    applied += 1
                    self._applied.add(tr.name)
                    log.info("[analogy] Applied transfer: %s → %s (%.0f%%)",
                             tr.source_domain, tr.target_domain, tr.confidence * 100)
                except Exception as e:
                    log.debug("[analogy] Registry write failed for %s: %s", tr.name, e)
                    skipped += 1
            else:
                # No registry — just track it
                self._applied.add(tr.name)
                applied += 1
        return applied, skipped

    def classify_rule_type(self, rule_name: str) -> str:
        """Classify a rule name into its structural type."""
        import re
        for pattern, stype in _RULE_TYPE_PATTERNS:
            if re.search(pattern, rule_name):
                return stype
        return "unknown"

    def summary(self) -> Dict[str, Any]:
        """Return current transfer state for API/UI."""
        total_possible = sum(len(v) for v in _ANALOGY_MAP.values())
        return {
            "applied_transfers": len(self._applied),
            "total_possible":    total_possible,
            "coverage_pct":      round(len(self._applied) / max(total_possible, 1) * 100, 1),
            "source_domains":    list(set(src for _, src in _ANALOGY_MAP.keys())),
            "analogy_map_size":  len(_ANALOGY_MAP),
        }
