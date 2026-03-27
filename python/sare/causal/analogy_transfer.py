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
import re
from collections import defaultdict
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
         "conf": 0.95, "note": "A ∪ A = A (set union idempotency)"},
    ],
    # Commutativity: a + b = b + a
    ("commutative", "arithmetic"): [
        {"target": "logic",   "pattern": "p AND q",     "replacement": "q AND p",
         "conf": 0.95, "note": "Conjunction is commutative"},
        {"target": "sets",    "pattern": "A ∪ B",       "replacement": "B ∪ A",
         "conf": 0.95, "note": "Set union is commutative"},
    ],
    # Associativity: (a + b) + c = a + (b + c)
    ("associative", "arithmetic"): [
        {"target": "logic",   "pattern": "(p AND q) AND r", "replacement": "p AND (q AND r)",
         "conf": 0.95, "note": "Conjunction is associative"},
    ],
    # De Morgan's laws (logic ↔ sets)
    ("de_morgan", "logic"): [
        {"target": "sets",    "pattern": "NOT (A ∪ B)", "replacement": "(NOT A) ∩ (NOT B)",
         "conf": 0.90, "note": "De Morgan's law for sets"},
    ],
    # Double negation: ¬¬p = p
    ("double_negation", "logic"): [
        {"target": "arithmetic", "pattern": "-(-x)",    "replacement": "x",
         "conf": 0.85, "note": "Double negation in additive inverse"},
    ],
}

# Patterns for classifying rule names into structural types
_RULE_TYPE_PATTERNS = [
    (r"identity.*add|add.*identity|additive.*identity", "identity_add"),
    (r"identity.*mul|mul.*identity|multiplicative.*identity", "identity_mul"),
    (r"annihilator.*mul|mul.*zero|zero.*mul", "annihilator_mul"),
    (r"idempotent", "idempotent"),
    (r"commutative|commute", "commutative"),
    (r"associative|associate", "associative"),
    (r"de.?morgan", "de_morgan"),
    (r"double.*negat|negation.*twice", "double_negation"),
    (r"distribut", "distribution"),
    (r"absorpt", "absorption"),
    (r"involution", "involution"),
]


# ── AnalogyTransfer engine ────────────────────────────────────────────────────

class AnalogyTransfer:
    """
    Cross-domain structural analogy transfer engine.

    Analyses rules in the ConceptRegistry, identifies structural
    similarities, and proposes transfer rules for missing domains.
    """

    def __init__(self, concept_registry: Any = None, decay: float = 0.1):
        """
        Args:
            concept_registry: ConceptRegistry instance (optional).
            decay: Confidence decay for transferred rules (0.0–0.3).
        """
        self.registry = concept_registry
        self.decay = decay
        self._applied: Set[str] = set()  # track already-applied transfers
        # outcome tracking: rule_name -> {"attempts": int, "successes": int, "total_delta": float}
        self._outcomes: Dict[str, Dict[str, Any]] = {}
        self._outcome_count: int = 0

    # ── Core transfer logic ────────────────────────────────────────────────

    def transfer_from_domain(
        self,
        source_domain: str,
        target_domain: Optional[str] = None,
        top_k: int = 10,
    ) -> List[TransferRule]:
        """
        Generate transfer rules from source_domain to target_domain(s).

        Combines static analogy map with dynamic discovery from registry.
        """
        # Static transfers from hardcoded map
        static_transfers = self._static_transfers(source_domain, target_domain)
        
        # Dynamic discovery from registry (if available)
        dynamic_transfers = self._dynamic_discovery(source_domain, target_domain)
        
        # Combine and deduplicate
        all_transfers = static_transfers + dynamic_transfers
        seen_names = set()
        unique_transfers = []
        for tr in all_transfers:
            if tr.name not in seen_names:
                seen_names.add(tr.name)
                unique_transfers.append(tr)
        
        return unique_transfers[:top_k]

    def _static_transfers(
        self,
        source_domain: str,
        target_domain: Optional[str] = None,
    ) -> List[TransferRule]:
        """Generate transfers from the hardcoded _ANALOGY_MAP."""
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

        return transfers

    def _dynamic_discovery(
        self,
        source_domain: str,
        target_domain: Optional[str] = None,
    ) -> List[TransferRule]:
        """
        Dynamically discover new analogies by analyzing ConceptRegistry.
        
        Looks for rules in source_domain that have structural counterparts
        in other domains not covered by the static map.
        """
        if not self.registry:
            return []
        
        transfers: List[TransferRule] = []
        
        try:
            # Get all rules from registry
            all_rules = self._get_registry_rules()
            if not all_rules:
                return []
            
            # Group rules by domain
            rules_by_domain: Dict[str, List[Dict]] = defaultdict(list)
            for rule in all_rules:
                domain = rule.get("domain", "")
                if domain:
                    rules_by_domain[domain].append(rule)
            
            # Get source domain rules
            source_rules = rules_by_domain.get(source_domain, [])
            if not source_rules:
                return []
            
            # Get all available domains
            available_domains = set(rules_by_domain.keys())
            available_domains.discard(source_domain)
            
            # Filter target domain if specified
            if target_domain:
                available_domains = {target_domain} if target_domain in available_domains else set()
            
            # For each source rule, try to find structural matches in other domains
            for src_rule in source_rules:
                src_name = src_rule.get("name", "")
                if not src_name:
                    continue
                
                # Classify the source rule's structural type
                struct_type = self.classify_rule_type(src_name)
                if struct_type == "unknown":
                    continue
                
                # Check if this structural type already has static transfers
                static_key = (struct_type, source_domain)
                if static_key in _ANALOGY_MAP:
                    # Static map already covers this type, skip dynamic discovery
                    continue
                
                # Get source rule's pattern and replacement
                src_pattern = src_rule.get("pattern_description", "")
                src_replacement = src_rule.get("replacement_description", "")
                if not src_pattern or not src_replacement:
                    continue
                
                # Source rule must have sufficient confidence and observations
                src_conf = src_rule.get("confidence", 0.0)
                src_obs = src_rule.get("observations", 0)
                if src_conf < 0.7 or src_obs < 10:
                    continue
                
                # Look for rules in target domains with similar structure
                for tgt_domain in available_domains:
                    tgt_rules = rules_by_domain.get(tgt_domain, [])
                    
                    # Check if target domain already has a rule of this structural type
                    has_matching_type = False
                    for tgt_rule in tgt_rules:
                        tgt_name = tgt_rule.get("name", "")
                        if self.classify_rule_type(tgt_name) == struct_type:
                            has_matching_type = True
                            break
                    
                    if has_matching_type:
                        # Target domain already has this structural pattern
                        continue
                    
                    # Check for superficial structural similarity
                    if not self._has_structural_similarity(src_pattern, src_replacement, tgt_rules):
                        continue
                    
                    # Create dynamic transfer rule
                    rule_name = f"dynamic_transfer_{struct_type}_{source_domain}_to_{tgt_domain}"
                    if rule_name in self._applied:
                        continue
                    
                    # Lower confidence for dynamically discovered rules
                    dynamic_conf = max(0.4, src_conf * 0.7 - self.decay)
                    
                    tr = TransferRule(
                        name=rule_name,
                        source_rule=src_name,
                        source_domain=source_domain,
                        target_domain=tgt_domain,
                        pattern_description=src_pattern,
                        replacement_description=src_replacement,
                        confidence=dynamic_conf,
                        observations=5,  # very low initial evidence
                        reasoning=(
                            f"Dynamic analogy discovery: '{src_name}' in {source_domain} "
                            f"has structural similarity to potential rules in {tgt_domain}. "
                            f"Structural type: {struct_type}. "
                            f"Source confidence: {src_conf:.2f}, observations: {src_obs}."
                        ),
                        notes=f"Dynamically discovered analogy (confidence decayed from {src_conf:.2f})",
                    )
                    transfers.append(tr)
                    
                    # Limit dynamic discoveries to avoid noise
                    if len(transfers) >= 3:
                        return transfers
        
        except Exception as e:
            log.debug("[analogy] Dynamic discovery failed: %s", e)
        
        return transfers

    def _get_registry_rules(self) -> List[Dict]:
        """Safely extract rules from the ConceptRegistry."""
        if not self.registry:
            return []
        
        rules = []
        try:
            # Try different registry interfaces
            if hasattr(self.registry, 'get_all_rules'):
                rules = self.registry.get_all_rules()
            elif hasattr(self.registry, 'rules'):
                rules = self.registry.rules
            elif hasattr(self.registry, '_rules'):
                rules = self.registry._rules
            elif isinstance(self.registry, dict) and 'rules' in self.registry:
                rules = self.registry['rules']
            
            # Ensure we have a list of dicts
            if isinstance(rules, dict):
                rules = list(rules.values())
            elif not isinstance(rules, list):
                rules = []
                
        except Exception as e:
            log.debug("[analogy] Could not extract rules from registry: %s", e)
        
        return rules

    def _has_structural_similarity(
        self,
        src_pattern: str,
        src_replacement: str,
        tgt_rules: List[Dict],
    ) -> bool:
        """
        Check if target domain has rules with similar structural patterns.
        
        Uses simple pattern matching to avoid false positives.
        """
        if not tgt_rules:
            return False
        
        # Normalize patterns for comparison
        src_pattern_norm = self._normalize_pattern(src_pattern)
        src_repl_norm = self._normalize_pattern(src_replacement)
        
        # Look for rules with similar structure in target domain
        for tgt_rule in tgt_rules:
            tgt_pattern = tgt_rule.get("pattern_description", "")
            tgt_replacement = tgt_rule.get("replacement_description", "")
            
            if not tgt_pattern or not tgt_replacement:
                continue
            
            tgt_pattern_norm = self._normalize_pattern(tgt_pattern)
            tgt_repl_norm = self._normalize_pattern(tgt_replacement)
            
            # Check for structural similarity (same operator pattern)
            if (self._extract_operator(src_pattern_norm) == 
                self._extract_operator(tgt_pattern_norm)):
                # Similar operator structure found
                return True
        
        return False

    def _normalize_pattern(self, pattern: str) -> str:
        """Normalize a pattern string for comparison."""
        # Remove domain-specific variable names
        pattern = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', 'VAR', pattern)
        # Normalize whitespace
        pattern = re.sub(r'\s+', ' ', pattern).strip()
        return pattern.lower()

    def _extract_operator(self, pattern: str) -> str:
        """Extract the main operator from a normalized pattern."""
        operators = ['and', 'or', 'not', 'xor', '+', '*', '∪', '∩', '∧', '∨', '¬']
        for op in operators:
            if op in pattern:
                return op
        return ""

    def transfer_all_domains(self) -> List[TransferRule]:
        """Generate the complete cross-domain transfer set."""
        all_transfers: List[TransferRule] = []
        seen_srcs = set(src for _, src in _ANALOGY_MAP.keys())

        # Add transfers from static map
        for src_domain in seen_srcs:
            all_transfers.extend(self._static_transfers(src_domain))

        # Add dynamic discoveries for all domains
        if self.registry:
            try:
                all_rules = self._get_registry_rules()
                domains_with_rules = set(rule.get("domain", "") for rule in all_rules)
                domains_with_rules.discard("")

                for src_domain in domains_with_rules:
                    all_transfers.extend(self._dynamic_discovery(src_domain))
            except Exception as e:
                log.debug("[analogy] Dynamic discovery in transfer_all_domains failed: %s", e)

        # Signature-based transfer from WorldModel composite-derived facts (no LLM needed)
        all_transfers.extend(self._signature_transfer_from_kb())

        return all_transfers

    def _signature_transfer_from_kb(self) -> List[TransferRule]:
        """Derive transfer rules by matching structural signatures of KB facts across domains.

        For each (subject, predicate, value) triple in WorldModel, compute a structural
        signature (predicate category × value type). When the same signature appears in
        multiple domains, propose cross-domain transfer rules — no LLM required.
        """
        transfers: List[TransferRule] = []
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()

            # Signature: (predicate_category, value_type) → [(domain, subject, predicate, value)]
            _PRED_CATS = {
                "is_a": "isa", "isa": "isa", "type_of": "isa",
                "has_property": "property", "color": "property", "habitat": "property",
                "breathes": "biological", "has_fur": "biological", "is_mammal": "biological",
                "is_conductor": "physical", "has_mass": "physical", "has_velocity": "physical",
                "causes_damage": "causal", "can_combust": "causal",
                "is_atom": "chemical", "ph_below_7": "chemical",
                "is_integer": "math", "is_prime": "math",
                "capital": "geo", "country": "geo",
            }
            def _val_type(v: str) -> str:
                if v in ("yes", "no", "true", "false"): return "bool"
                if v.replace(".", "").replace("-", "").isdigit(): return "numeric"
                return "entity"

            sig_index: dict = {}  # (pred_cat, val_type) → [(domain, subj, pred, val)]
            for b in wm.get_beliefs():
                subj = str(b.get("subject", "") or "").lower().strip()
                pred = str(b.get("predicate", "") or "").lower().strip()
                val  = str(b.get("value",    "") or "").lower().strip()
                dom  = str(b.get("domain",   "") or "general").lower()
                conf = float(b.get("confidence", 0.5) or 0.5)
                if not (subj and pred and val) or conf < 0.6:
                    continue
                cat = _PRED_CATS.get(pred, None)
                if not cat:
                    continue
                sig = (cat, _val_type(val))
                sig_index.setdefault(sig, []).append((dom, subj, pred, val))

            # For signatures appearing in ≥2 distinct domains, generate transfer proposal
            _DOMAIN_PAIRS = [("factual", "science"), ("science", "reasoning"),
                             ("factual", "reasoning"), ("science", "analogy")]
            for sig, entries in sig_index.items():
                cat, vtype = sig
                domains_seen = {e[0] for e in entries}
                for src_dom, tgt_dom in _DOMAIN_PAIRS:
                    if src_dom not in domains_seen or tgt_dom in domains_seen:
                        continue
                    # src has this signature but tgt doesn't — propose transfer
                    example = next(e for e in entries if e[0] == src_dom)
                    rule_name = f"kb_sig_transfer_{cat}_{src_dom}_to_{tgt_dom}"
                    if rule_name in self._applied:
                        continue
                    tr = TransferRule(
                        name=rule_name,
                        source_rule=example[2],
                        source_domain=src_dom,
                        target_domain=tgt_dom,
                        pattern_description=f"{example[2]}=({vtype})",
                        replacement_description=f"apply {cat} knowledge in {tgt_dom}",
                        confidence=0.55,
                        observations=len(entries),
                        reasoning=(
                            f"Signature-based transfer: '{cat}' predicate class with {vtype} values "
                            f"found in {src_dom} domain (e.g. {example[1]} {example[2]}={example[3]}). "
                            f"No equivalent found in {tgt_dom} — proposing structural analogy."
                        ),
                        notes="KB signature transfer — no LLM required",
                    )
                    transfers.append(tr)
                    if len(transfers) >= 5:
                        return transfers
        except Exception as e:
            log.debug("[analogy] Signature transfer failed: %s", e)
        return transfers

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
        for pattern, stype in _RULE_TYPE_PATTERNS:
            if re.search(pattern, rule_name, re.IGNORECASE):
                return stype
        return "unknown"

    def record_transfer_outcome(self, rule_name: str, domain: str, success: bool, delta: float) -> None:
        """Track whether a transferred rule actually helped solve a problem.

        Args:
            rule_name: Name of the transfer rule used.
            domain:    Domain in which the rule was applied.
            success:   Whether the rule contributed to solving the problem.
            delta:     Energy reduction achieved (positive = helpful).
        """
        key = rule_name
        if key not in self._outcomes:
            self._outcomes[key] = {"attempts": 0, "successes": 0, "total_delta": 0.0, "domain": domain}
        entry = self._outcomes[key]
        entry["attempts"] += 1
        entry["total_delta"] += delta
        if success:
            entry["successes"] += 1

        self._outcome_count += 1
        if self._outcome_count % 100 == 0:
            self.prune_weak_transfers()

    def get_effective_transfers(self) -> List[dict]:
        """Return transfers with success_rate >= 0.5 and at least 3 attempts."""
        effective = []
        for rule_name, entry in self._outcomes.items():
            attempts = entry["attempts"]
            if attempts < 3:
                continue
            success_rate = entry["successes"] / attempts
            if success_rate >= 0.5:
                effective.append({
                    "rule_name":    rule_name,
                    "domain":       entry["domain"],
                    "attempts":     attempts,
                    "successes":    entry["successes"],
                    "success_rate": round(success_rate, 3),
                    "avg_delta":    round(entry["total_delta"] / attempts, 4),
                })
        return effective

    def prune_weak_transfers(self) -> int:
        """Remove transfers with success_rate < 0.3 after 5+ attempts.

        Returns:
            Number of rules pruned.
        """
        to_prune = []
        for rule_name, entry in self._outcomes.items():
            if entry["attempts"] >= 5:
                success_rate = entry["successes"] / entry["attempts"]
                if success_rate < 0.3:
                    to_prune.append(rule_name)

        for rule_name in to_prune:
            del self._outcomes[rule_name]
            self._applied.discard(rule_name)
            log.info("[analogy] Pruned weak transfer rule: %s", rule_name)

        if to_prune:
            log.info("[analogy] Pruned %d weak transfer rule(s)", len(to_prune))
        return len(to_prune)

    def summary(self) -> Dict[str, Any]:
        """Return current transfer state for API/UI."""
        total_possible = sum(len(v) for v in _ANALOGY_MAP.values())
        return {
            "applied_transfers": len(self._applied),
            "total_possible":    total_possible,
            "coverage_pct":      round(len(self._applied) / max(total_possible, 1) * 100, 1),
            "source_domains":    list(set(src for _, src in _ANALOGY_MAP.keys())),
            "analogy_map_size":  len(_ANALOGY_MAP),
            "dynamic_discovery_enabled": self.registry is not None,
            "tracked_outcomes":  len(self._outcomes),
            "effective_transfers": len(self.get_effective_transfers()),
        }