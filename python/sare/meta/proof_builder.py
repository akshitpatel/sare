"""
ProofBuilder — TODO-08: Human-readable proof output
=====================================================
Converts SARE's solve trace (list of applied transforms + graph states)
into a human-readable, step-by-step natural language proof.

Usage::

    proof = ProofBuilder().build(
        expression="(x + 0) * 1",
        transforms_applied=["additive_identity", "multiplicative_identity"],
        initial_energy=5.2,
        final_energy=1.0,
    )
    print(proof.text)
    print(proof.to_dict())
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Natural-language templates for each known transform ──────────────────────

_RULE_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    # Additive
    "additive_identity": {
        "name": "Additive Identity (x + 0 = x)",
        "desc": "Adding zero to any expression leaves it unchanged.",
        "template": "Remove the redundant '+ 0': {before} → {after}",
    },
    "additive_inverse": {
        "name": "Additive Inverse (x − x = 0)",
        "desc": "A value subtracted from itself is zero.",
        "template": "Cancel equal terms: {before} → 0",
    },
    # Multiplicative
    "multiplicative_identity": {
        "name": "Multiplicative Identity (x × 1 = x)",
        "desc": "Multiplying by one leaves the value unchanged.",
        "template": "Remove the redundant '× 1': {before} → {after}",
    },
    "multiplicative_zero": {
        "name": "Multiplicative Zero (x × 0 = 0)",
        "desc": "Any value multiplied by zero is zero.",
        "template": "The entire sub-expression evaluates to 0: {before} → 0",
    },
    "multiplicative_inverse": {
        "name": "Multiplicative Inverse (x / x = 1)",
        "desc": "A non-zero value divided by itself equals one.",
        "template": "Cancel equal numerator and denominator: {before} → 1",
    },
    # Double negation
    "double_negation": {
        "name": "Double Negation (¬¬x = x)",
        "desc": "Two negations cancel each other out.",
        "template": "Drop the double negation: {before} → {after}",
    },
    "double_negation_arith": {
        "name": "Double Negation Arithmetic (−(−x) = x)",
        "desc": "Two negative signs cancel each other.",
        "template": "Remove the double minus: {before} → {after}",
    },
    # Boolean
    "boolean_and_true": {
        "name": "Boolean AND True (x ∧ True = x)",
        "desc": "AND-ing with True is redundant.",
        "template": "Simplify: {before} → {after}",
    },
    "boolean_and_false": {
        "name": "Boolean AND False (x ∧ False = False)",
        "desc": "AND-ing with False always gives False.",
        "template": "Short-circuit: {before} → False",
    },
    "boolean_or_false": {
        "name": "Boolean OR False (x ∨ False = x)",
        "desc": "OR-ing with False is redundant.",
        "template": "Simplify: {before} → {after}",
    },
    "boolean_or_true": {
        "name": "Boolean OR True (x ∨ True = True)",
        "desc": "OR-ing with True always gives True.",
        "template": "Short-circuit: {before} → True",
    },
    "idempotent_and": {
        "name": "Idempotent AND (x ∧ x = x)",
        "desc": "A value AND-ed with itself is itself.",
        "template": "Merge duplicate: {before} → {after}",
    },
    "idempotent_or": {
        "name": "Idempotent OR (x ∨ x = x)",
        "desc": "A value OR-ed with itself is itself.",
        "template": "Merge duplicate: {before} → {after}",
    },
    # Constant folding
    "constant_folding": {
        "name": "Constant Folding",
        "desc": "Evaluate a numeric computation at compile time.",
        "template": "Compute the constant: {before} → {after}",
    },
    "add_one_elimination": {
        "name": "Increment Simplification (+1 rule)",
        "desc": "Simplify an increment expression.",
        "template": "Simplify: {before} → {after}",
    },
    # Exponent
    "zero_exponent": {
        "name": "Zero Exponent (x⁰ = 1)",
        "desc": "Any non-zero value raised to the power 0 is 1.",
        "template": "Apply zero-exponent rule: {before} → 1",
    },
    "one_exponent": {
        "name": "Unit Exponent (x¹ = x)",
        "desc": "Raising a value to the first power is redundant.",
        "template": "Drop the exponent: {before} → {after}",
    },
    # Macros / learned concepts
    "concept_": {
        "name": "Learned Rule (ConceptRegistry)",
        "desc": "A rule previously learned from experience.",
        "template": "Apply learned pattern: {before} → {after}",
    },
}

_GENERIC = {
    "name":     "Algebraic Simplification",
    "desc":     "Simplify the expression using a known identity.",
    "template": "Apply rule: {before} → {after}",
}


def _lookup(transform_name: str) -> Dict[str, str]:
    if transform_name in _RULE_EXPLANATIONS:
        return _RULE_EXPLANATIONS[transform_name]
    # prefix match (e.g. concept_additive_identity)
    for prefix, info in _RULE_EXPLANATIONS.items():
        if transform_name.startswith(prefix):
            return dict(info, name=f"Learned Rule '{transform_name}'")
    return dict(_GENERIC, name=transform_name.replace("_", " ").title())


# ── Proof step dataclass ──────────────────────────────────────────────────────

@dataclass
class ProofStep:
    number:      int
    transform:   str
    rule_name:   str
    explanation: str
    detail:      str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step":        self.number,
            "transform":   self.transform,
            "rule_name":   self.rule_name,
            "explanation": self.explanation,
            "detail":      self.detail,
        }


# ── Proof dataclass ───────────────────────────────────────────────────────────

@dataclass
class Proof:
    expression:   str
    conclusion:   str
    steps:        List[ProofStep]
    initial_energy: float
    final_energy:   float
    reduction_pct:  float
    text:           str = field(default="")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expression":     self.expression,
            "conclusion":     self.conclusion,
            "initial_energy": round(self.initial_energy, 4),
            "final_energy":   round(self.final_energy, 4),
            "reduction_pct":  round(self.reduction_pct, 2),
            "steps":          [s.to_dict() for s in self.steps],
            "text":           self.text,
        }


# ── ProofBuilder ──────────────────────────────────────────────────────────────

class ProofBuilder:
    """
    Builds a human-readable proof from a solve trace.

    Call `build(...)` with the expression, list of applied transform names,
    and energy values. Returns a `Proof` object.
    """

    def build(
        self,
        expression: str,
        transforms_applied: List[str],
        initial_energy: float,
        final_energy: float,
        domain: str = "general",
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Proof:
        """
        Build a proof from a solve trace.

        Parameters
        ----------
        expression         : Input expression string (raw text).
        transforms_applied : Ordered list of transform names applied.
        initial_energy     : Energy before solve.
        final_energy       : Energy after solve.
        domain             : Problem domain (arithmetic, logic, …).
        extra_context      : Optional dict with 'before'/'after' graph reprints.

        Returns
        -------
        Proof object with `text` and `to_dict()`.
        """
        ctx = extra_context or {}
        steps: List[ProofStep] = []
        delta = initial_energy - final_energy
        pct   = (delta / initial_energy * 100) if initial_energy > 0 else 0.0

        for i, tname in enumerate(transforms_applied, start=1):
            info   = _lookup(tname)
            before = ctx.get(f"step_{i}_before", expression)
            after  = ctx.get(f"step_{i}_after",  "…")
            detail = info["template"].format(before=before, after=after)
            steps.append(ProofStep(
                number=i,
                transform=tname,
                rule_name=info["name"],
                explanation=info["desc"],
                detail=detail,
            ))

        conclusion = self._conclusion(expression, transforms_applied, delta, pct, domain)
        text       = self._render_text(expression, steps, delta, pct, conclusion)

        return Proof(
            expression=expression,
            conclusion=conclusion,
            steps=steps,
            initial_energy=initial_energy,
            final_energy=final_energy,
            reduction_pct=pct,
            text=text,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _conclusion(
        self,
        expr: str,
        transforms: List[str],
        delta: float,
        pct: float,
        domain: str,
    ) -> str:
        if not transforms:
            return f"The expression '{expr}' is already in its simplest form — no rules apply."
        n = len(transforms)
        names = [_lookup(t)["name"] for t in transforms]
        rule_list = ", ".join(f"'{n}'" for n in names[:3])
        if len(names) > 3:
            rule_list += f", and {len(names)-3} more rule(s)"
        return (
            f"The expression '{expr}' was simplified in {n} step(s) "
            f"using {rule_list}. "
            f"Energy was reduced by {delta:.3f} ({pct:.1f}%), "
            f"confirming the expression is now in a more canonical form."
        )

    def _render_text(
        self,
        expression: str,
        steps: List[ProofStep],
        delta: float,
        pct: float,
        conclusion: str,
    ) -> str:
        lines = [
            f"══ Proof of Simplification ══",
            f"Input: {expression}",
            f"",
        ]
        if not steps:
            lines.append("  (no transforms applied — expression is already minimal)")
        else:
            for s in steps:
                lines.append(f"  Step {s.number}. {s.rule_name}")
                lines.append(f"    ↳ {s.explanation}")
                lines.append(f"    ↳ {s.detail}")
                lines.append("")
        lines += [
            f"Energy reduction: {delta:.4f}  ({pct:.1f}%)",
            f"",
            f"Conclusion: {conclusion}",
        ]
        return "\n".join(lines)
