"""
ConceptRule Transform — TODO-06 Implementation

Wraps an AbstractRule (from ConceptRegistry) in the Transform interface
so learned rules can participate in BeamSearch / MCTS just like primitives.

This closes the final "dead wire": rules learned via Reflection+CausalInduction
now actually influence search behavior instead of just sitting in the registry.

Pattern matching strategy:
  - Match by node TYPE sequence (structural signature)
  - Uses the rule's semantic metadata to infer which operator/constant types
    the pattern describes
  - Apply: performs the structural simplification encoded in the rule's name/domain

Confidence weighting:
  - Energy delta offered = base_delta * rule.confidence
  - Low-confidence rules (0.65-0.79) offer smaller deltas → explored less greedily
  - High-confidence rules (0.9+) behave like primitives
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter

log = logging.getLogger(__name__)

# Import from engine (same package)
from sare.engine import Transform, Graph, Node   # type: ignore


# ── Rule-name → structural matching logic ─────────────────────

_ARITHMETIC_PATTERNS: Dict[str, dict] = {
    # name → {operator, constant_label, energy_delta}
    "additive_identity": {
        "operators": ["+", "add"],
        "zero_label": "0",
        "pattern": "op_with_zero_child",
        "delta": -3.0,
    },
    "additive_identity_commuted": {
        "operators": ["+", "add"],
        "zero_label": "0",
        "pattern": "op_with_zero_child",
        "delta": -3.0,
    },
    "multiplicative_identity": {
        "operators": ["*", "mul"],
        "zero_label": "1",
        "pattern": "op_with_one_child",
        "delta": -3.0,
    },
    "multiplicative_zero": {
        "operators": ["*", "mul"],
        "zero_label": "0",
        "pattern": "op_makes_zero",
        "delta": -4.0,
    },
    "double_negation": {
        "operators": ["neg", "-"],
        "pattern": "double_unary",
        "delta": -4.0,
    },
    "subtractive_self": {
        "operators": ["-", "sub"],
        "pattern": "self_minus_self",
        "delta": -4.0,
    },
    "equation_subtract": {
        "operators": ["=", "==", "eq"],
        "pattern": "equation_shift_term",
        "delta": -2.0,  # Penalty for increasing size, but offset by isolating var
    },
    "equation_divide": {
        "operators": ["=", "==", "eq"],
        "pattern": "equation_divide_coeff",
        "delta": -1.0,  # Isolating the variable, usually leads to simplifications
    },
}

_LOGIC_PATTERNS: Dict[str, dict] = {
    "double_negation_logic": {
        "operators": ["not", "!", "¬"],
        "pattern": "double_unary",
        "delta": -4.0,
    },
    "and_true": {
        "operators": ["and", "&&", "∧"],
        "constant_label": "true",
        "pattern": "op_with_true_child",
        "delta": -3.0,
    },
    "and_false": {
        "operators": ["and", "&&", "∧"],
        "constant_label": "false",
        "pattern": "op_makes_false",
        "delta": -4.0,
    },
    "or_true": {
        "operators": ["or", "||", "∨"],
        "constant_label": "true",
        "pattern": "op_makes_true",
        "delta": -4.0,
    },
    "or_false": {
        "operators": ["or", "||", "∨"],
        "constant_label": "false",
        "pattern": "op_with_false_child",
        "delta": -3.0,
    },
    "idempotent_and": {
        "operators": ["and", "&&", "∧"],
        "pattern": "self_op_self",
        "delta": -3.0,
    },
    "idempotent_or": {
        "operators": ["or", "||", "∨"],
        "pattern": "self_op_self",
        "delta": -3.0,
    },
    "exclusion_middle": {
        "operators": ["or", "||", "∨"],
        "pattern": "p_or_not_p",
        "delta": -5.0,
    },
    "contradiction": {
        "operators": ["and", "&&", "∧"],
        "pattern": "p_and_not_p",
        "delta": -5.0,
    },
}


class ConceptRule(Transform):
    """
    A dynamically-learned transform derived from an AbstractRule.

    Wraps a rule from ConceptRegistry and exposes it as a beamSearch
    transform. Works for both seed rules and online-learned rules.
    """

    def __init__(self, rule):
        """
        rule: either a C++ AbstractRule (sare_bindings.AbstractRule)
              or a Python dict from knowledge_seeds.json
        """
        self._rule = rule
        self._rule_name: str = _get_attr(rule, "name", "unknown_rule")
        self._domain: str   = _get_attr(rule, "domain", "general")
        self._confidence: float = float(_get_attr(rule, "confidence", 0.5))
        self._pattern_info = self._resolve_pattern()

    # ── Transform interface ────────────────────────────────────

    def name(self) -> str:
        return f"concept_{self._rule_name}"

    def match(self, graph: Graph) -> List[dict]:
        if not self._pattern_info:
            return []
        try:
            return self._match_pattern(graph, self._pattern_info)
        except Exception as e:
            log.debug("ConceptRule.match error (%s): %s", self._rule_name, e)
            return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        try:
            return self._apply_pattern(graph, context, self._pattern_info)
        except Exception as e:
            log.debug("ConceptRule.apply error (%s): %s", self._rule_name, e)
            return graph, 0.0

    def estimate_delta(self, graph: Graph, context: dict) -> float:
        base = self._pattern_info.get("delta", -2.0) if self._pattern_info else -2.0
        return base * self._confidence

    # ── Internal: pattern resolution ──────────────────────────

    def _resolve_pattern(self) -> Optional[dict]:
        """Look up the structural pattern for this rule name."""
        name = self._rule_name

        # --- Analogy Transfer Rule Name Resolution ---
        if name.startswith("transfer_"):
            if "identity_add" in name and "logic" in name: name = "and_true"
            elif "identity_mul" in name and "logic" in name: name = "or_false"
            elif "annihilator_mul" in name and "logic" in name: name = "and_false"
            elif "involution" in name and "arithmetic" in name: name = "double_negation"
            elif "involution" in name and "algebra" in name: name = "double_negation"
            elif "self_inverse" in name and "logic" in name: name = "xor_self"
            elif "complement_max" in name and "logic" in name: name = "exclusion_middle"
            elif "complement_min" in name and "logic" in name: name = "contradiction"
            else:
                # If analog target isn't explicitly mapped here, fall back to exact match attempts
                name = name.split("_to_")[-1]

        info = _ARITHMETIC_PATTERNS.get(name)
        if not info:
            info = _LOGIC_PATTERNS.get(name)
        return info

    def _match_pattern(self, graph: Graph, info: dict) -> List[dict]:
        pattern = info.get("pattern", "")
        operators = info.get("operators", [])

        matches = []
        for n in graph.nodes:
            if n.type not in ("operator",):
                continue
            if n.label not in operators:
                continue

            children_edges = graph.outgoing(n.id)
            children = [graph.get_node(e.target) for e in children_edges]
            children = [c for c in children if c]

            ctx = self._try_pattern(n, children, children_edges, pattern, info, graph)
            if ctx:
                matches.append(ctx)

        return matches

    def _try_pattern(self, op_node: Node, children: List[Node],
                     children_edges, pattern: str, info: dict, graph: Graph) -> Optional[dict]:
        """Try to match one specific pattern against an operator node."""

        const_label = info.get("zero_label") or info.get("constant_label", "")

        if pattern in ("op_with_zero_child", "op_with_one_child", "op_with_true_child",
                       "op_with_false_child"):
            # x OP const → x  (eliminate the constant child)
            for i, c in enumerate(children):
                if c.type == "constant" and c.label == const_label:
                    # Find the other child
                    other_idx = 1 - i if len(children) == 2 else None
                    if other_idx is not None:
                        return {
                            "op": op_node.id,
                            "const": c.id,
                            "keep": children[other_idx].id,
                            "_action": "keep_other",
                        }
            return None

        elif pattern in ("op_makes_zero", "op_makes_false"):
            # x OP 0 → 0
            for c in children:
                if c.type == "constant" and c.label == const_label:
                    return {
                        "op": op_node.id,
                        "children": [c.id for c in children],
                        "_action": "replace_with_const",
                        "_const_label": const_label,
                        "_const_type": "constant",
                    }
            return None

        elif pattern == "op_makes_true":
            # p OR true → true
            for c in children:
                if c.type == "constant" and c.label == const_label:
                    return {
                        "op": op_node.id,
                        "children": [c.id for c in children],
                        "_action": "replace_with_const",
                        "_const_label": const_label,
                        "_const_type": "constant",
                    }
            return None

        elif pattern == "double_unary":
            # neg(neg(x)) → x
            if len(children) == 1:
                inner = children[0]
                if (inner.type == "operator" and inner.label == op_node.label
                        and len(graph.outgoing(inner.id)) == 1):
                    inner_child_edge = graph.outgoing(inner.id)[0]
                    return {
                        "op": op_node.id,
                        "inner_op": inner.id,
                        "keep": inner_child_edge.target,
                        "_action": "double_unary_elim",
                    }
            return None

        elif pattern == "self_minus_self":
            # x - x → 0
            if len(children) == 2 and children[0].label == children[1].label:
                return {
                    "op": op_node.id,
                    "left": children[0].id,
                    "right": children[1].id,
                    "_action": "replace_with_const",
                    "_const_label": "0",
                    "_const_type": "constant",
                }
            return None

        elif pattern == "self_op_self":
            # p AND p → p   or   p OR p → p
            if len(children) == 2 and children[0].label == children[1].label:
                return {
                    "op": op_node.id,
                    "const": children[1].id,
                    "keep": children[0].id,
                    "_action": "keep_other",
                }
            return None

        elif pattern == "p_or_not_p":
            # p OR NOT(p) → true
            if len(children) == 2:
                for i, c in enumerate(children):
                    other = children[1 - i]
                    if (c.type == "operator" and c.label in ("not", "!", "¬")
                            and len(graph.outgoing(c.id)) == 1):
                        inner_edge = graph.outgoing(c.id)[0]
                        inner = graph.get_node(inner_edge.target)
                        if inner and inner.label == other.label:
                            return {
                                "op": op_node.id,
                                "children": [c.id for c in children],
                                "_action": "replace_with_const",
                                "_const_label": "true",
                                "_const_type": "constant",
                            }
            return None

        elif pattern == "p_and_not_p":
            # p AND NOT(p) → false
            if len(children) == 2:
                for i, c in enumerate(children):
                    other = children[1 - i]
                    if (c.type == "operator" and c.label in ("not", "!", "¬")
                            and len(graph.outgoing(c.id)) == 1):
                        inner_edge = graph.outgoing(c.id)[0]
                        inner = graph.get_node(inner_edge.target)
                        if inner and inner.label == other.label:
                            return {
                                "op": op_node.id,
                                "children": [c.id for c in children],
                                "_action": "replace_with_const",
                                "_const_label": "false",
                                "_const_type": "constant",
                            }
            return None

        elif pattern == "equation_shift_term":
            # a + b = c → a = c - b or b = c - a
            if len(children) == 2:
                # We want to identify if one child is an addition and the other is a term
                for i, side in enumerate(children):
                    other_side = children[1 - i]
                    # if LHS is an addition: X + Y = RHS
                    if side.type == "operator" and side.label in ("+", "add"):
                        sub_children = [graph.get_node(e.target) for e in graph.outgoing(side.id)]
                        sub_children = [c for c in sub_children if c]
                        if len(sub_children) == 2:
                            # Propose subtracting sub_children[0]
                            return {
                                "eq_op": op_node.id,
                                "add_op": side.id,
                                "term_to_shift": sub_children[0].id,
                                "term_to_keep": sub_children[1].id,
                                "other_side": other_side.id,
                                "_action": "shift_addition_term",
                            }
            return None

        elif pattern == "equation_divide_coeff":
            # a * b = c → a = c / b
            if len(children) == 2:
                for i, side in enumerate(children):
                    other_side = children[1 - i]
                    # if LHS is multiplication: X * Y = RHS
                    if side.type == "operator" and side.label in ("*", "mul"):
                        sub_children = [graph.get_node(e.target) for e in graph.outgoing(side.id)]
                        sub_children = [c for c in sub_children if c]
                        if len(sub_children) == 2:
                            # If one is a constant, prefer dividing by the constant
                            shift_idx = 0
                            if sub_children[1].type == "constant":
                                shift_idx = 1
                            elif sub_children[0].type == "constant":
                                shift_idx = 0
                            else:
                                shift_idx = 0 # default
                            
                            return {
                                "eq_op": op_node.id,
                                "mul_op": side.id,
                                "term_to_shift": sub_children[shift_idx].id,
                                "term_to_keep": sub_children[1 - shift_idx].id,
                                "other_side": other_side.id,
                                "_action": "divide_multiplication_coeff",
                            }
            return None

        return None

    def _apply_pattern(self, graph: Graph, ctx: dict,
                       info: Optional[dict]) -> Tuple[Graph, float]:
        g = graph.clone()
        action = ctx.get("_action", "")
        delta = info.get("delta", -2.0) if info else -2.0

        if action == "keep_other":
            # Replace op_node with "keep" child — rewire parent edges
            op_id   = ctx["op"]
            keep_id = ctx["keep"]
            drop_id = ctx.get("const")
            for e in g.edges:
                if e.target == op_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, keep_id, e.relationship_type)
            g.remove_node(op_id)
            if drop_id:
                g.remove_node(drop_id)

        elif action == "replace_with_const":
            op_id      = ctx["op"]
            const_label = ctx.get("_const_label", "0")
            const_type  = ctx.get("_const_type", "constant")
            # Mutate the op node into a constant
            op_node = g.get_node(op_id)
            if op_node:
                op_node.type  = const_type
                op_node.label = const_label
                # Remove all children
                for e in list(g.outgoing(op_id)):
                    target = e.target
                    g.remove_edge(e.id)
                    if not g.incoming(target):
                        g.remove_node(target)

        elif action == "double_unary_elim":
            op_id      = ctx["op"]
            inner_id   = ctx["inner_op"]
            keep_id    = ctx["keep"]
            for e in g.edges:
                if e.target == op_id:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, keep_id, e.relationship_type)
            g.remove_node(op_id)
            g.remove_node(inner_id)

        elif action == "shift_addition_term":
            # X + Y = RHS  →  Y = RHS - X
            eq_op   = ctx["eq_op"]
            add_op  = ctx["add_op"]
            t_shift = ctx["term_to_shift"]
            t_keep  = ctx["term_to_keep"]
            rhs     = ctx["other_side"]

            # 1. Rewire `t_keep` to directly connect to `eq_op` instead of `add_op`
            for e in list(g.edges):
                if e.target == add_op and e.source == eq_op:
                    g.remove_edge(e.id)
                    g.add_edge(eq_op, t_keep, e.relationship_type)

            # 2. Create new Subtraction operator under `eq_op` replacing `rhs`
            sub_id = g.add_node("operator", "-")
            
            # Detach RHS from eq_op and attach to sub
            for e in list(g.edges):
                if e.target == rhs and e.source == eq_op:
                    g.remove_edge(e.id)
                    g.add_edge(eq_op, sub_id, e.relationship_type)
            
            # Connect Subtraction to RHS and shifted term
            g.add_edge(sub_id, rhs, "left")
            
            # Detach shifted term from add_op and attach to sub
            for e in list(g.edges):
                if e.target == t_shift and e.source == add_op:
                    g.remove_edge(e.id)
                    g.add_edge(sub_id, t_shift, "right")

            # Remove the old addition op
            g.remove_node(add_op)

        elif action == "divide_multiplication_coeff":
            # X * Y = RHS  →  Y = RHS / X
            eq_op   = ctx["eq_op"]
            mul_op  = ctx["mul_op"]
            t_shift = ctx["term_to_shift"]
            t_keep  = ctx["term_to_keep"]
            rhs     = ctx["other_side"]

            # 1. Rewire `t_keep` to directly connect to `eq_op` instead of `mul_op`
            for e in list(g.edges):
                if e.target == mul_op and e.source == eq_op:
                    g.remove_edge(e.id)
                    g.add_edge(eq_op, t_keep, e.relationship_type)

            # 2. Create new Division operator under `eq_op` replacing `rhs`
            div_id = g.add_node("operator", "/")
            
            # Detach RHS from eq_op and attach to div
            for e in list(g.edges):
                if e.target == rhs and e.source == eq_op:
                    g.remove_edge(e.id)
                    g.add_edge(eq_op, div_id, e.relationship_type)
            
            # Connect Division to RHS and shifted term
            g.add_edge(div_id, rhs, "numerator")
            
            # Detach shifted term from mul_op and attach to div
            for e in list(g.edges):
                if e.target == t_shift and e.source == mul_op:
                    g.remove_edge(e.id)
                    g.add_edge(div_id, t_shift, "denominator")

            # Remove the old mul op
            g.remove_node(mul_op)

        return g, delta * self._confidence


# ── Helper ─────────────────────────────────────────────────────

def _get_attr(obj, attr: str, default):
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# ── Public factory ──────────────────────────────────────────────

def concept_transforms_from_registry(concept_registry,
                                     min_confidence: float = 0.65) -> List[ConceptRule]:
    """
    Build ConceptRule transforms from all high-confidence rules in ConceptRegistry.

    Called from get_transforms() to inject learned rules into search.
    """
    if not concept_registry:
        return []

    transforms = []
    try:
        rules = concept_registry.get_rules()
        for rule in rules:
            conf = float(_get_attr(rule, "confidence", 0.0))
            if conf >= min_confidence:
                ct = ConceptRule(rule)
                if ct._pattern_info:  # Only include rules with known patterns
                    transforms.append(ct)
    except Exception as e:
        log.warning("concept_transforms_from_registry error: %s", e)

    if transforms:
        log.info("Injected %d concept transforms from registry (min_conf=%.2f)",
                 len(transforms), min_confidence)
    return transforms
