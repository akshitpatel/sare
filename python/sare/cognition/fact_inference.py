"""
FactInference — One-hop chaining engine that derives new facts from stored facts.

Chain types:
  - Transitivity: (A is_a B) + (B has_property V) → (A has_property V)
  - Symmetry: (A related_to B) → (B related_to A) for commutative predicates
  - Composition: unit conversions and proportional chains

Derived facts are stored back with confidence × 0.9 (decay per hop).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

_SYMMETRIC_PREDICATES = {"related_to", "similar_to", "equivalent_to", "associated_with"}
_TRANSITIVE_PREDICATES = {"is_a", "isa", "type_of", "part_of", "subset_of"}


# ── Multi-premise composite rules ─────────────────────────────────────────────

@dataclass
class CompositeRule:
    """A rule with N premises: IF all (predicate, value) pairs hold for subject X
    THEN derive (X, conclusion_predicate, conclusion_value).
    Empty value = any value (just requires the predicate to exist).
    """
    premises: List[Tuple[str, str]]   # [(predicate, required_value_or_empty), ...]
    conclusion_predicate: str
    conclusion_value: str
    base_confidence: float = 0.80


_COMPOSITE_RULES: List[CompositeRule] = [
    # Biology — multi-trait classification
    CompositeRule([("breathes", ""), ("has_fur", "")],            "is_mammal",         "yes",  0.85),
    CompositeRule([("breathes", ""), ("has_feathers", "")],       "is_bird",           "yes",  0.85),
    CompositeRule([("breathes", ""), ("cold_blooded", "yes")],    "is_reptile_or_fish","yes",  0.75),
    CompositeRule([("is_mammal", "yes"), ("lives_in_ocean","yes")],"is_aquatic_mammal","yes",  0.85),
    CompositeRule([("is_bird", "yes"), ("cannot_fly", "yes")],    "is_flightless_bird","yes",  0.90),
    # Physics — derived quantities
    CompositeRule([("has_mass", ""), ("has_velocity", "")],       "has_kinetic_energy","yes",  0.90),
    CompositeRule([("is_conductor", "yes"), ("voltage_applied","yes")], "carries_current","yes",0.85),
    CompositeRule([("has_high_temperature","yes"),("contains_oxygen","yes")],"can_combust","yes",0.80),
    # Chemistry
    CompositeRule([("proton_count", ""), ("electron_count", "")], "is_atom",           "yes",  0.85),
    CompositeRule([("ph_below_7","yes"), ("is_liquid","yes")],    "turns_litmus_red",  "yes",  0.90),
    # Logic / Math
    CompositeRule([("is_integer","yes"), ("greater_than_zero","yes")], "is_natural_number","yes",0.90),
    CompositeRule([("is_prime","yes"), ("is_even","yes")],        "equals_two",        "yes",  0.95),
    # Geography
    CompositeRule([("capital", ""), ("country", "")],             "is_capital_city",   "yes",  0.90),
    # Causality
    CompositeRule([("causes_damage","yes"),("is_natural_disaster","yes")],"requires_evacuation","yes",0.80),
]


class FactInference:
    """Derives new facts via one-hop chaining over WorldModel and CommonSenseBase."""

    def infer_from_domain(self, domain: str, max_new: int = 10) -> List[Tuple[str, str, str]]:
        """
        Derive new (subject, predicate, derived_value) triples for the given domain.
        Stores derived facts back into WorldModel.
        Returns list of newly derived triples.
        """
        derived: List[Tuple[str, str, str]] = []

        derived.extend(self._transitivity_chain(domain, max_new))
        if len(derived) < max_new:
            derived.extend(self._symmetry_chain(max_new - len(derived)))
        if len(derived) < max_new:
            derived.extend(self._commonsense_chain(domain, max_new - len(derived)))
        if len(derived) < max_new:
            derived.extend(self._apply_composite_rules(domain, max_new - len(derived)))

        # Store derived facts back (single-hop chains — composite rules store inline)
        for subject, predicate, value in derived[:max_new]:
            self._store(domain, subject, predicate, value)

        return derived[:max_new]

    # ── Chain implementations ──────────────────────────────────────────────────

    def _transitivity_chain(self, domain: str, max_new: int) -> List[Tuple[str, str, str]]:
        """
        Transitivity: if (A, is_a, B) and (B, property, V) then (A, property, V).
        Operates over WorldModel facts for the given domain.
        """
        results: List[Tuple[str, str, str]] = []
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            facts = wm.get_facts(domain) + wm.get_facts("general")

            # Build a subject→predicate→value index from fact strings
            index: dict = {}  # subject → {predicate: [values]}
            for fd in facts:
                fact_text = fd.get("fact", "")
                conf      = fd.get("confidence", 0.5)
                # Parse "subject predicate: value" format
                if ": " in fact_text:
                    parts = fact_text.split(": ", 1)
                    sp = parts[0].strip().rsplit(" ", 1)
                    if len(sp) == 2:
                        subj, pred = sp
                        val  = parts[1].strip()
                        index.setdefault(subj, {}).setdefault(pred, []).append((val, conf))

            # Find is_a chains
            for subj_a, preds_a in index.items():
                for pred_a, vals_a in preds_a.items():
                    if pred_a.lower() not in _TRANSITIVE_PREDICATES:
                        continue
                    for val_a, conf_a in vals_a:
                        subj_b = val_a.lower()
                        if subj_b not in index:
                            continue
                        for pred_b, vals_b in index[subj_b].items():
                            if pred_b.lower() in _TRANSITIVE_PREDICATES:
                                continue
                            for val_b, conf_b in vals_b:
                                triple = (subj_a, pred_b, val_b)
                                # Only add if not already known
                                existing = index.get(subj_a, {}).get(pred_b, [])
                                if not any(v == val_b for v, _ in existing):
                                    results.append(triple)
                                    if len(results) >= max_new:
                                        return results
        except Exception as e:
            log.debug("[FactInference] Transitivity chain error: %s", e)
        return results

    def _symmetry_chain(self, max_new: int) -> List[Tuple[str, str, str]]:
        """
        Symmetry: (A, related_to, B) → (B, related_to, A) for symmetric predicates.
        Operates over CommonSenseBase forward/backward index.
        """
        results: List[Tuple[str, str, str]] = []
        try:
            from sare.knowledge.commonsense import CommonSenseBase
            cs = CommonSenseBase()
            cs.load()

            for subj, edges in list(cs._forward.items())[:50]:
                for rel, obj in edges:
                    if rel.lower() not in _SYMMETRIC_PREDICATES:
                        continue
                    # Check if reverse already exists
                    rev = cs._forward.get(obj, [])
                    if not any(r == rel and o == subj for r, o in rev):
                        results.append((obj, rel, subj))
                    if len(results) >= max_new:
                        return results
        except Exception as e:
            log.debug("[FactInference] Symmetry chain error: %s", e)
        return results

    def _commonsense_chain(self, domain: str, max_new: int) -> List[Tuple[str, str, str]]:
        """
        Composition: derive domain-relevant facts from commonsense triples.
        E.g., if (fire HasProperty hot) and domain='science', add fact about fire.
        """
        results: List[Tuple[str, str, str]] = []
        try:
            from sare.knowledge.commonsense import CommonSenseBase
            cs = CommonSenseBase()
            cs.load()

            # Domain keyword mapping
            domain_concepts = {
                "science":   ["atom", "cell", "energy", "force", "acid", "base", "electron"],
                "factual":   ["human", "animal", "plant", "dog", "fire", "water"],
                "reasoning": ["logic", "algorithm", "number", "equality"],
                "analogy":   ["school", "hospital", "law", "money", "friendship"],
            }.get(domain, [])

            for concept in domain_concepts:
                facts = cs.query(concept, depth=1)
                for item in facts[:3]:
                    subj = item.get("subject", "")
                    rel  = item.get("relation", "")
                    obj  = item.get("object",  "")
                    if subj and rel and obj:
                        results.append((subj, rel, obj))
                    if len(results) >= max_new:
                        return results
        except Exception as e:
            log.debug("[FactInference] Commonsense chain error: %s", e)
        return results

    # ── Multi-premise composite rule application ──────────────────────────────

    def _apply_composite_rules(self, domain: str, max_new: int = 10) -> List[Tuple[str, str, str]]:
        """Forward chain using _COMPOSITE_RULES: derive new facts when ALL premises hold."""
        derived: List[Tuple[str, str, str]] = []
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()

            # Build subject → {predicate: (value, confidence)} index from all beliefs
            subject_facts: dict = {}
            for b in wm.get_beliefs():
                subj = str(b.get("subject", "") or "").lower().strip()
                pred = str(b.get("predicate", "") or "").lower().strip()
                val  = str(b.get("value", "") or "").lower().strip()
                conf = float(b.get("confidence", 0.5) or 0.5)
                if subj and pred and val:
                    subject_facts.setdefault(subj, {})[pred] = (val, conf)

            for rule in _COMPOSITE_RULES:
                for subject, facts in subject_facts.items():
                    # Skip if conclusion already known
                    if wm.get_belief(subject, rule.conclusion_predicate) is not None:
                        continue
                    # Check all premises
                    all_match = True
                    min_conf = 1.0
                    for req_pred, req_val in rule.premises:
                        if req_pred not in facts:
                            all_match = False
                            break
                        actual_val, conf = facts[req_pred]
                        if req_val and actual_val != req_val.lower():
                            all_match = False
                            break
                        min_conf = min(min_conf, conf)
                    if all_match:
                        derived_conf = round(min_conf * rule.base_confidence, 3)
                        wm.update_belief(subject, rule.conclusion_predicate,
                                         rule.conclusion_value, confidence=derived_conf,
                                         domain=domain)
                        derived.append((subject, rule.conclusion_predicate, rule.conclusion_value))
                        # Store causal justification: why was this derived?
                        premise_trace = "; ".join(
                            f"{p}={v if v else '?'}" for p, v in rule.premises
                        )
                        justification = f"{subject} {rule.conclusion_predicate}: {rule.conclusion_value} because ({premise_trace})"
                        wm.add_fact(domain=domain, fact=justification, confidence=derived_conf * 0.9)
                        log.debug("[FactInference] Composite: %s → (%s, %s) [because: %s]",
                                  subject, rule.conclusion_predicate, rule.conclusion_value, premise_trace)
                        if len(derived) >= max_new:
                            return derived
        except Exception as e:
            log.debug("[FactInference] Composite rules error: %s", e)
        return derived

    # ── N-hop backward chaining ───────────────────────────────────────────────

    def chain_to_goal(
        self,
        goal_subject: str,
        goal_predicate: str,
        domain: str,
        max_depth: int = 3,
    ) -> Optional[str]:
        """BFS backward chain: traverse is_a/type_of links to find (goal_subject, goal_predicate).

        Returns the derived value if found, else None.
        Stores the derived fact in WorldModel with confidence decay.
        """
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()

            # Direct hit first
            belief = wm.get_belief(goal_subject, goal_predicate)
            if belief is not None and belief.value:
                return belief.value

            # BFS over is_a / type_of chain
            queue: deque = deque([(goal_subject.lower(), 0, 1.0)])
            visited = {goal_subject.lower()}

            while queue:
                subj, depth, conf_so_far = queue.popleft()
                if depth >= max_depth:
                    continue
                for tp in _TRANSITIVE_PREDICATES:
                    b = wm.get_belief(subj, tp)
                    if b is None or not b.value:
                        continue
                    parent = b.value.lower().strip()
                    if not parent or parent in visited:
                        continue
                    visited.add(parent)
                    # Check if parent has the goal predicate
                    pb = wm.get_belief(parent, goal_predicate)
                    if pb is not None and pb.value:
                        # conf_so_far already includes b.confidence from the previous queue.append,
                        # so don't multiply b.confidence again — that was the double-counting bug.
                        derived_conf = round(conf_so_far * pb.confidence * (0.9 ** (depth + 1)), 3)
                        wm.update_belief(
                            goal_subject, goal_predicate, pb.value,
                            confidence=derived_conf, domain=domain,
                        )
                        log.debug("[FactInference] chain_to_goal: %s -[%s]-> %s -[%s]-> %s",
                                  goal_subject, tp, parent, goal_predicate, pb.value)
                        return pb.value
                    queue.append((parent, depth + 1, conf_so_far * b.confidence))
        except Exception as e:
            log.debug("[FactInference] chain_to_goal error: %s", e)
        return None

    # ── Storage ────────────────────────────────────────────────────────────────

    def _store(self, domain: str, subject: str, predicate: str, value: str,
               base_confidence: float = 0.7) -> None:
        """Store a derived fact with 0.9× confidence decay.
        If the belief already exists with the same value, Bayesian-blend confidences
        (independent evidence: new_conf = 1 - (1-old)*(1-new)) so repeated inferences
        compound correctly rather than overwriting."""
        derived_conf = round(base_confidence * 0.9, 3)
        fact_str = f"{subject} {predicate}: {value} [inferred]"
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            existing = wm.get_belief(subject, predicate)
            if existing is not None and str(existing.value).lower() == str(value).lower():
                old_conf = float(existing.confidence or 0.0)
                blended  = round(1.0 - (1.0 - old_conf) * (1.0 - derived_conf), 4)
                wm.update_belief(subject, predicate, value,
                                 confidence=blended, domain=domain)
            else:
                wm.add_fact(domain=domain, fact=fact_str, confidence=derived_conf)
        except Exception as e:
            log.debug("[FactInference] Store failed: %s", e)


    def infer_iterative(self, domain: str, max_rounds: int = 3,
                        max_new_per_round: int = 20) -> int:
        """Run forward chaining until convergence or max_rounds.
        Returns total count of new facts derived across all rounds."""
        total = 0
        for _ in range(max_rounds):
            new = self.infer_from_domain(domain, max_new=max_new_per_round)
            if not new:
                break   # converged — no new facts this round
            total += len(new)
        return total


_FACT_INFERENCE_SINGLETON: Optional[FactInference] = None


def get_fact_inference() -> FactInference:
    global _FACT_INFERENCE_SINGLETON
    if _FACT_INFERENCE_SINGLETON is None:
        _FACT_INFERENCE_SINGLETON = FactInference()
    return _FACT_INFERENCE_SINGLETON
