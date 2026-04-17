"""
Language Grounding — Links symbolic rules to natural language understanding.

Each rule the system knows gets grounded with:
  1. A plain English explanation ("adding zero to anything gives you the same thing")
  2. Concrete examples ("5 + 0 = 5", "apple + nothing = apple")
  3. A physical analogy ("like adding an empty box to a pile — nothing changes")
  4. WHY it works (the causal mechanism)

This is NOT an LLM wrapper. It builds grounded understanding from
the system's own experience — which rules it has used, how often,
and what structural role they play.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "memory"


@dataclass
class GroundedConcept:
    """A concept linked to natural language understanding."""
    name: str
    formal_rule: str           # "x + 0 = x"
    explanation: str           # plain English
    examples: List[str]        # concrete instances
    analogy: str               # physical world analogy
    why: str                   # causal mechanism
    domain: str
    structural_role: str       # identity, annihilation, involution, etc.
    confidence: float = 0.5
    use_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "formal_rule": self.formal_rule,
            "explanation": self.explanation, "examples": self.examples,
            "analogy": self.analogy, "why": self.why,
            "domain": self.domain, "role": self.structural_role,
            "confidence": round(self.confidence, 3), "use_count": self.use_count,
        }


# Built-in grounding templates per structural role
_ROLE_GROUNDINGS = {
    "identity": {
        "explanation_template": "Applying {op} with {element} to anything gives you the same thing back.",
        "analogy_template": "Like adding an empty box to a pile — nothing changes.",
        "why_template": "{element} is the 'do nothing' element for {op}.",
    },
    "annihilation": {
        "explanation_template": "Applying {op} with {element} to anything always gives {element}.",
        "analogy_template": "Like multiplying by zero — everything collapses to nothing.",
        "why_template": "{element} is so dominant that it overrides whatever the other input is.",
    },
    "involution": {
        "explanation_template": "Applying {op} twice cancels out — you get back what you started with.",
        "analogy_template": "Like flipping a light switch twice — you end up where you started.",
        "why_template": "{op} is its own inverse: doing it undoes itself.",
    },
    "self_inverse": {
        "explanation_template": "Applying {op} to the same thing twice gives the neutral element.",
        "analogy_template": "Like subtracting your age from your age — you get zero.",
        "why_template": "Any quantity {op}'d with itself cancels out.",
    },
    "evaluation": {
        "explanation_template": "When both inputs are known numbers, just compute the result.",
        "analogy_template": "Like using a calculator — if you know both numbers, just do the math.",
        "why_template": "Concrete values can always be computed directly.",
    },
    "combination": {
        "explanation_template": "When the same variable appears multiple times, combine the coefficients.",
        "analogy_template": "Like counting: 3 apples + 2 apples = 5 apples.",
        "why_template": "Same-type terms can be merged by adding their multipliers.",
    },
    "equation_solving": {
        "explanation_template": "Isolate the unknown by applying inverse operations to both sides.",
        "analogy_template": "Like a balance scale — whatever you do to one side, do to the other.",
        "why_template": "Inverse operations undo each other, leaving the variable alone.",
    },
    "distribution": {
        "explanation_template": "Multiply each term inside the parentheses separately.",
        "analogy_template": "Like giving 3 cookies to each of 2+1 kids — give 3 to each group.",
        "why_template": "Multiplication distributes over addition by definition.",
    },
    "commutativity": {
        "explanation_template": "The order of the inputs doesn't matter for this operation.",
        "analogy_template": "Like mixing paint — red+blue is the same as blue+red.",
        "why_template": "This operation is symmetric in its arguments.",
    },
    "cancellation": {
        "explanation_template": "Adding then subtracting the same amount cancels out.",
        "analogy_template": "Like walking 5 steps forward then 5 steps back — you're where you started.",
        "why_template": "Addition and subtraction are inverse operations.",
    },
}


class LanguageGrounding:
    """
    Grounds symbolic rules in natural language.
    Builds understanding from experience, not from hardcoded explanations.
    """

    def __init__(self):
        self._concepts: Dict[str, GroundedConcept] = {}
        self._load()

    def ground(self, rule_name: str, formal_rule: str, domain: str = "general",
               structural_role: str = "identity", operator: str = "",
               element: str = "") -> GroundedConcept:
        """Alias for ground_rule with shorter signature."""
        return self.ground_rule(rule_name, formal_rule, domain, structural_role, operator, element)

    def ground_rule(self, rule_name: str, formal_rule: str, domain: str,
                    structural_role: str, operator: str = "",
                    element: str = "") -> GroundedConcept:
        """Ground a rule with natural language explanation."""
        if rule_name in self._concepts:
            self._concepts[rule_name].use_count += 1
            return self._concepts[rule_name]

        # Get template for this role
        templates = _ROLE_GROUNDINGS.get(structural_role, {})
        explanation = templates.get("explanation_template", f"{rule_name} simplifies expressions.").format(
            op=operator or "this operation", element=element or "the special element"
        )
        analogy = templates.get("analogy_template", "").format(
            op=operator or "this operation", element=element or "the special element"
        )
        why = templates.get("why_template", "").format(
            op=operator or "this operation", element=element or "the special element"
        )

        # Generate concrete examples
        examples = self._generate_examples(operator, element, structural_role)

        concept = GroundedConcept(
            name=rule_name, formal_rule=formal_rule,
            explanation=explanation, examples=examples,
            analogy=analogy, why=why,
            domain=domain, structural_role=structural_role,
        )
        self._concepts[rule_name] = concept
        return concept

    def ground_from_transform(self, transform_name: str, domain: str = "general") -> Optional[GroundedConcept]:
        """Auto-ground a transform by inferring its structural role."""
        t = transform_name.lower()

        # Infer role and parameters from name
        if "zero" in t and ("add" in t or "elim" in t):
            return self.ground_rule(transform_name, "x + 0 = x", domain, "identity", "+", "0")
        if "one" in t and ("mul" in t or "elim" in t):
            return self.ground_rule(transform_name, "x * 1 = x", domain, "identity", "*", "1")
        if "zero" in t and "mul" in t:
            return self.ground_rule(transform_name, "x * 0 = 0", domain, "annihilation", "*", "0")
        if "neg" in t or "double" in t or "not" in t:
            return self.ground_rule(transform_name, "neg(neg(x)) = x", domain, "involution", "neg", "")
        if "subtract" in t or "self" in t:
            return self.ground_rule(transform_name, "x - x = 0", domain, "self_inverse", "-", "")
        if "fold" in t or "const" in t:
            return self.ground_rule(transform_name, "c1 op c2 = c3", domain, "evaluation", "op", "")
        if "combin" in t or "like" in t:
            return self.ground_rule(transform_name, "ax + bx = (a+b)x", domain, "combination", "+", "")
        if "solve" in t or "equation" in t:
            return self.ground_rule(transform_name, "x + c = d → x = d-c", domain, "equation_solving", "=", "")
        if "distribut" in t:
            return self.ground_rule(transform_name, "a*(b+c) = ab+ac", domain, "distribution", "*", "")
        if "commut" in t:
            return self.ground_rule(transform_name, "a+b = b+a", domain, "commutativity", "+", "")
        if "cancel" in t:
            return self.ground_rule(transform_name, "(x+c)-c = x", domain, "cancellation", "+/-", "")
        if "power" in t and "zero" in t:
            return self.ground_rule(transform_name, "x^0 = 1", domain, "identity", "^", "0")
        if "div" in t and ("self" in t or "one" in t):
            return self.ground_rule(transform_name, "x/1 = x", domain, "identity", "/", "1")
        if "bool" in t and "and" in t and "true" in t:
            return self.ground_rule(transform_name, "x and true = x", domain or "logic", "identity", "and", "true")
        if "bool" in t and "and" in t and "false" in t:
            return self.ground_rule(transform_name, "x and false = false", domain or "logic", "annihilation", "and", "false")
        if "bool" in t and "or" in t and "false" in t:
            return self.ground_rule(transform_name, "x or false = x", domain or "logic", "identity", "or", "false")
        if "bool" in t and "or" in t and "true" in t:
            return self.ground_rule(transform_name, "x or true = true", domain or "logic", "annihilation", "or", "true")
        if "bool" in t and "idempotent" in t:
            return self.ground_rule(transform_name, "x and x = x", domain or "logic", "commutativity", "and", "")
        if "trig" in t or "sin" in t:
            return self.ground_rule(transform_name, "sin(0) = 0", domain or "trigonometry", "identity", "sin", "0")
        if "cos" in t:
            return self.ground_rule(transform_name, "cos(0) = 1", domain or "trigonometry", "identity", "cos", "0")
        if "log" in t and "one" in t:
            return self.ground_rule(transform_name, "log(1) = 0", domain or "trigonometry", "identity", "log", "1")
        if "sqrt" in t:
            return self.ground_rule(transform_name, "sqrt(x^2) = x", domain or "trigonometry", "involution", "sqrt", "")
        if "deriv" in t and "const" in t:
            return self.ground_rule(transform_name, "d/dx(c) = 0", domain or "calculus", "evaluation", "d/dx", "")
        if "deriv" in t and "linear" in t:
            return self.ground_rule(transform_name, "d/dx(x) = 1", domain or "calculus", "identity", "d/dx", "")
        if "deriv" in t and "power" in t:
            return self.ground_rule(transform_name, "d/dx(x^n) = n*x^(n-1)", domain or "calculus", "evaluation", "d/dx", "")

        return None

    def explain(self, transform_name: str, expression: str = "",
                domain: str = "general") -> str:
        """Get a natural language explanation for a transform application."""
        concept = self._concepts.get(transform_name)
        if not concept:
            concept = self.ground_from_transform(transform_name, domain)
        if not concept:
            return f"Applied {transform_name} to simplify the expression."

        parts = [concept.explanation]
        if expression:
            parts.append(f"In this case: {expression}")
        if concept.analogy:
            parts.append(f"Think of it like: {concept.analogy}")
        return " ".join(parts)

    def explain_solve(self, transforms: List[str], expression: str = "",
                      domain: str = "general") -> str:
        """Explain a full solve trace in natural language."""
        if not transforms:
            return "No simplification was needed."

        steps = []
        for i, t in enumerate(transforms):
            explanation = self.explain(t, domain=domain)
            steps.append(f"Step {i+1}: {explanation}")

        intro = f"To simplify '{expression}':" if expression else "Here's what happened:"
        return f"{intro}\n" + "\n".join(steps)

    def _generate_examples(self, operator: str, element: str,
                           role: str) -> List[str]:
        """Generate concrete examples for a rule."""
        examples = []
        if role == "identity" and operator and element:
            for val in ["5", "100", "x", "y"]:
                examples.append(f"{val} {operator} {element} = {val}")
        elif role == "annihilation" and operator and element:
            for val in ["5", "100", "x"]:
                examples.append(f"{val} {operator} {element} = {element}")
        elif role == "involution" and operator:
            for val in ["x", "5", "true"]:
                examples.append(f"{operator}({operator}({val})) = {val}")
        elif role == "self_inverse" and operator:
            for val in ["5", "x", "100"]:
                examples.append(f"{val} {operator} {val} = 0")
        elif role == "evaluation":
            examples = ["3 + 4 = 7", "5 * 6 = 30", "10 - 3 = 7"]
        return examples[:5]

    def get_all_concepts(self) -> List[dict]:
        return [c.to_dict() for c in self._concepts.values()]

    def stats(self) -> dict:
        roles = {}
        for c in self._concepts.values():
            roles[c.structural_role] = roles.get(c.structural_role, 0) + 1
        return {
            "total_concepts": len(self._concepts),
            "roles_grounded": roles,
            "most_used": sorted(
                [c.to_dict() for c in self._concepts.values()],
                key=lambda x: x["use_count"], reverse=True
            )[:5],
        }

    def save(self):
        path = DATA_DIR / "language_grounding.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {c.name: c.to_dict() for c in self._concepts.values()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        path = DATA_DIR / "language_grounding.json"
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                for name, d in data.items():
                    self._concepts[name] = GroundedConcept(
                        name=d["name"], formal_rule=d["formal_rule"],
                        explanation=d["explanation"], examples=d["examples"],
                        analogy=d["analogy"], why=d["why"],
                        domain=d["domain"], structural_role=d["role"],
                        confidence=d.get("confidence", 0.5),
                        use_count=d.get("use_count", 0),
                    )
            except Exception:
                pass

    def parse_instruction(self, instruction: str) -> Optional[dict]:
        """
        Parse a natural language instruction and map it to a transform.
        Returns: {"transform": str, "confidence": float} or None
        """
        instruction = instruction.lower().strip()

        # Map natural language patterns to transforms
        patterns = {
            # Identity patterns
            r"add\s+zero|plus\s+zero|add\s+nothing": ["add_zero_elim", "concept_additive_identity"],
            r"multiply\s+by\s+one|times\s+one": ["mul_one_elim", "concept_multiplicative_identity"],
            r"multiply\s+by\s+zero|times\s+zero": ["mul_zero_elim", "concept_multiplicative_zero"],
            r"power\s+of\s+zero|to\s+the\s+zero": ["power_zero_elim"],
            r"power\s+of\s+one|to\s+the\s+one": ["power_one_elim"],

            # Negation patterns
            r"double\s+negation|neg\s+neg|not\s+not": ["double_neg", "concept_double_negation"],
            r"cancel\s+negative|remove\s+negative": ["double_neg"],

            # Combination patterns
            r"combine\s+like\s+terms|add\s+similar\s+terms": ["combine_like_terms"],
            r"group\s+similar\s+terms": ["combine_like_terms"],

            # Distribution patterns
            r"distribute|expand\s+the\s+parentheses|multiply\s+through": ["distributive_expand"],
            r"expand\s+expression|multiply\s+out": ["distributive_expand"],

            # Factoring patterns
            r"factor\s+out|factor\s+common|factor\s+expression": ["factor_common"],
            r"pull\s+out\s+common|extract\s+common": ["factor_common"],

            # Cancellation patterns
            r"cancel\s+out|additive\s+cancel|subtract\s+to\s+cancel": ["additive_cancellation"],
            r"subtract\s+same|remove\s+same": ["additive_cancellation"],

            # Evaluation patterns
            r"compute|calculate|evaluate|simplify": ["const_fold"],
            r"do\s+the\s+math|calculate\s+the\s+result": ["const_fold"],
        }

        import re
        for pattern, transforms in patterns.items():
            if re.search(pattern, instruction):
                # Return the most confident match (first in list)
                return {"transform": transforms[0], "confidence": 0.8}

        # Try to match against existing explanations
        for name, concept in self._concepts.items():
            if concept.explanation.lower() in instruction or instruction in concept.explanation.lower():
                # Extract transform name from concept name
                transform_name = name.replace("_grounded", "")
                return {"transform": transform_name, "confidence": concept.confidence}

        return None

    def learn_from_instruction(self, instruction: str, expression: str = "",
                                success: bool = True) -> Optional[str]:
        """
        Learn from a natural language instruction.
        If the instruction maps to a transform, apply it and return the result.
        """
        result = self.parse_instruction(instruction)
        if result:
            transform_name = result["transform"]
            log.info(f"Language grounding: '{instruction}' -> {transform_name} (conf={result['confidence']:.2f})")
            return transform_name
        return None

    def suggest_transforms_for_goal(self, goal: str) -> List[dict]:
        """
        Given a natural language goal, suggest transforms that might help.
        Returns: [{"transform": str, "reason": str, "confidence": float}]
        """
        goal = goal.lower().strip()
        suggestions = []

        # Goal patterns
        goal_patterns = {
            r"simplify|reduce": {
                "transforms": ["const_fold", "combine_like_terms", "add_zero_elim", "mul_one_elim"],
                "reason": "Simplification reduces complexity"
            },
            r"solve.*equation|find.*x|isolate.*variable": {
                "transforms": ["linear_equation_solve", "additive_cancellation", "multiply_equation_solve"],
                "reason": "Equation solving isolates the unknown"
            },
            r"expand|distribute|multiply\s+out": {
                "transforms": ["distributive_expand"],
                "reason": "Expansion removes parentheses"
            },
            r"factor|factor\s+out": {
                "transforms": ["factor_common"],
                "reason": "Factoring reveals common structure"
            },
            r"prove|show|demonstrate": {
                "transforms": ["double_neg", "additive_cancellation"],
                "reason": "Proof often uses identity and cancellation"
            },
        }

        import re
        for pattern, info in goal_patterns.items():
            if re.search(pattern, goal):
                for transform in info["transforms"]:
                    suggestions.append({
                        "transform": transform,
                        "reason": info["reason"],
                        "confidence": 0.7
                    })

        return suggestions
