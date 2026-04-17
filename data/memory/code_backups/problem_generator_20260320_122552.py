"""
Open-Ended Problem Generator — Creates structurally diverse problems.

Instead of a fixed curriculum, this generates novel combinations from
known operators/patterns. Problems are:
  - Structurally diverse (not just variations of the same template)
  - Difficulty-calibrated (based on SelfModel competence)
  - Domain-spanning (mix arithmetic, logic, algebra, etc.)
  - Novel (combinations the system hasn't seen before)
"""

from __future__ import annotations

import random
import hashlib
from typing import List, Optional, Set


# Operator pools per domain
_OPERATORS = {
    "arithmetic": ["+", "-", "*", "/", "^"],
    "logic": ["and", "or", "not"],
    "algebra": ["=", "+", "-", "*"],
    "trigonometry": ["sin", "cos", "tan", "log", "sqrt"],
    "calculus": ["derivative", "diff", "integral"],
    "probability": ["P", "prob", "E", "Var"],
    "geometry": ["angle_sum", "sqrt"],
    "propositional": ["implies", "and", "or", "not"],
    "advanced_calculus": ["derivative", "diff", "integral", "chain"],
    "probability_statistics": ["P", "prob", "E", "Var"],
    "matrix_operations": ["+", "*", "det"],
}

_IDENTITY_ELEMENTS = {
    "+": "0", "-": "0", "*": "1", "/": "1", "^": "1",
    "and": "true", "or": "false",
}

_ABSORBING_ELEMENTS = {
    "*": "0", "and": "false", "or": "true",
}

_VARIABLES = ["x", "y", "a", "b", "z", "p", "q", "m", "n"]
_CONSTANTS = ["0", "1", "2", "3", "4", "5", "7", "10"]

# Novel domain operators for open-ended discovery
_NOVEL_DOMAINS: dict = {
    "set_theory":        ["union", "intersection", "complement", "subset"],
    "number_theory":     ["mod", "gcd", "lcm", "prime"],
    "abstract_algebra":  ["compose", "inverse", "identity", "closure"],
    "information_theory":["entropy", "mutual_info", "kl_div"],
    "graph_theory":      ["path", "cycle", "degree", "connected"],
}

# Cross-domain analogy templates: same structure, different domain symbols
_CROSS_DOMAIN_ANALOGIES: list = [
    # arithmetic identity → logic identity → set identity
    {"exprs": ["x + 0", "p and true", "x union empty"], "role": "identity", "difficulty": 0.2},
    # annihilation
    {"exprs": ["x * 0", "p and false", "x intersection empty"], "role": "annihilation", "difficulty": 0.2},
    # involution / double-negation
    {"exprs": ["not not p", "neg neg x", "complement(complement(A))"], "role": "involution", "difficulty": 0.3},
    # commutativity
    {"exprs": ["x + y", "p or q", "x union y"], "role": "commutativity", "difficulty": 0.3},
    # distributivity
    {"exprs": ["x * (y + z)", "p and (q or r)", "x intersection (y union z)"], "role": "distributivity", "difficulty": 0.5},
]


class ProblemGenerator:
    """Generates structurally diverse problems from known patterns."""

    def __init__(self):
        self._generated: Set[str] = set()
        self._templates = self._build_templates()

    def _build_templates(self) -> List[dict]:
        """Build problem templates from structural roles."""
        templates = []

        # Identity: op(x, identity_element)
        for op, elem in _IDENTITY_ELEMENTS.items():
            for var in _VARIABLES[:4]:
                templates.append({"expr": f"{var} {op} {elem}", "domain": self._op_domain(op), "difficulty": 0.1, "role": "identity"})
                templates.append({"expr": f"{elem} {op} {var}", "domain": self._op_domain(op), "difficulty": 0.15, "role": "identity_commuted"})

        # Annihilation: op(x, absorbing_element)
        for op, elem in _ABSORBING_ELEMENTS.items():
            for var in _VARIABLES[:3]:
                templates.append({"expr": f"{var} {op} {elem}", "domain": self._op_domain(op), "difficulty": 0.2, "role": "annihilation"})

        # Self-inverse: op(x, x)
        for op in ["-", "/"]:
            for var in _VARIABLES[:3]:
                templates.append({"expr": f"{var} {op} {var}", "domain": "arithmetic", "difficulty": 0.2, "role": "self_inverse"})

        # Double negation
        for var in _VARIABLES[:3]:
            templates.append({"expr": f"neg neg {var}", "domain": "arithmetic", "difficulty": 0.25, "role": "involution"})
            templates.append({"expr": f"not not {var}", "domain": "logic", "difficulty": 0.25, "role": "involution"})

        # Constant folding
        for a in _CONSTANTS[:5]:
            for b in _CONSTANTS[:5]:
                if a != b:
                    for op in ["+", "-", "*"]:
                        templates.append({"expr": f"{a} {op} {b}", "domain": "arithmetic", "difficulty": 0.1, "role": "evaluation"})

        # Power rules
        for var in _VARIABLES[:3]:
            templates.append({"expr": f"{var} ^ 0", "domain": "arithmetic", "difficulty": 0.2, "role": "power"})
            templates.append({"expr": f"{var} ^ 1", "domain": "arithmetic", "difficulty": 0.2, "role": "power"})

        # Trigonometry: trig identities at zero
        for fn in ["sin", "tan"]:
            templates.append({"expr": f"{fn}(0)", "domain": "trigonometry", "difficulty": 0.3, "role": "trig_zero"})
        templates.append({"expr": "cos(0)", "domain": "trigonometry", "difficulty": 0.3, "role": "trig_one"})
        templates.append({"expr": "log(1)", "domain": "trigonometry", "difficulty": 0.3, "role": "log_one"})
        for var in _VARIABLES[:3]:
            templates.append({"expr": f"sqrt({var}^2)", "domain": "trigonometry", "difficulty": 0.4, "role": "sqrt_square"})
        templates.append({"expr": "sin(0) + cos(0)", "domain": "trigonometry", "difficulty": 0.5, "role": "compound_trig"})
        templates.append({"expr": "sin(0) * x", "domain": "trigonometry", "difficulty": 0.5, "role": "compound_trig"})

        # Calculus: derivative rules
        for c in ["2", "5", "10"]:
            templates.append({"expr": f"derivative({c})", "domain": "calculus", "difficulty": 0.5, "role": "deriv_const"})
        for var in _VARIABLES[:2]:
            templates.append({"expr": f"derivative({var})", "domain": "calculus", "difficulty": 0.5, "role": "deriv_linear"})
            templates.append({"expr": f"derivative({var}^2)", "domain": "calculus", "difficulty": 0.6, "role": "deriv_power"})
            templates.append({"expr": f"derivative({var}^3)", "domain": "calculus", "difficulty": 0.7, "role": "deriv_power"})

        # Probability
        templates.append({"expr": "p(empty)", "domain": "probability", "difficulty": 0.3, "role": "prob_empty"})
        templates.append({"expr": "p(Omega)", "domain": "probability", "difficulty": 0.3, "role": "prob_universal"})
        for var in _VARIABLES[:3]:
            templates.append({"expr": f"p({var}) + p(not {var})",
                               "domain": "probability", "difficulty": 0.5, "role": "prob_complement"})

        # Geometry
        templates.append({"expr": "angle_sum(triangle)", "domain": "geometry", "difficulty": 0.3, "role": "angle_sum"})
        for a, b in [("3", "4"), ("5", "12"), ("6", "8")]:
            templates.append({"expr": f"{a}^2 + {b}^2",
                               "domain": "geometry", "difficulty": 0.4, "role": "pythagorean"})

        # Propositional logic: implication and De Morgan
        for p, q in [("p", "q"), ("x", "y"), ("a", "b")]:
            templates.append({"expr": f"{p} implies {q}",
                               "domain": "propositional", "difficulty": 0.5, "role": "implication"})
        for var in _VARIABLES[:2]:
            for var2 in _VARIABLES[1:3]:
                templates.append({"expr": f"not ({var} and {var2})",
                                   "domain": "propositional", "difficulty": 0.6, "role": "demorgan"})
                templates.append({"expr": f"not ({var} or {var2})",
                                   "domain": "propositional", "difficulty": 0.6, "role": "demorgan"})

        # Linear equations
        for var in _VARIABLES[:2]:
            for c in ["2", "3", "5", "7"]:
                for d in ["4", "6", "10", "12"]:
                    templates.append({"expr": f"{var} + {c} = {d}", "domain": "algebra", "difficulty": 0.4, "role": "equation"})
                    templates.append({"expr": f"{c} * {var} = {d}", "domain": "algebra", "difficulty": 0.5, "role": "equation"})

        # Combining like terms
        for var in _VARIABLES[:2]:
            for a in ["2", "3", "4"]:
                for b in ["2", "3", "5"]:
                    templates.append({"expr": f"{a} * {var} + {b} * {var}", "domain": "arithmetic", "difficulty": 0.4, "role": "combination"})

        # Advanced Calculus: chain rule and trig derivatives
        for var in _VARIABLES[:2]:
            templates.append({"expr": f"derivative(sin({var}^2))", "domain": "advanced_calculus", "difficulty": 0.7, "role": "chain_rule"})
            templates.append({"expr": f"derivative(exp({var}^2))", "domain": "advanced_calculus", "difficulty": 0.7, "role": "chain_rule_exp"})
            templates.append({"expr": f"derivative(exp({var}))", "domain": "advanced_calculus", "difficulty": 0.5, "role": "deriv_exp"})
            templates.append({"expr": f"derivative(ln({var}))", "domain": "advanced_calculus", "difficulty": 0.5, "role": "deriv_ln"})

        # Integration: power rule and sum rule
        for c in ["2", "3", "5"]:
            templates.append({"expr": f"integral({c})", "domain": "integration", "difficulty": 0.4, "role": "integ_constant"})
        for var in _VARIABLES[:2]:
            templates.append({"expr": f"integral({var})", "domain": "integration", "difficulty": 0.4, "role": "integ_linear"})
            templates.append({"expr": f"integral({var}^2)", "domain": "integration", "difficulty": 0.5, "role": "integ_power"})
            templates.append({"expr": f"integral({var}^3)", "domain": "integration", "difficulty": 0.5, "role": "integ_power"})
            templates.append({"expr": f"integral({var} + {var}^2)", "domain": "integration", "difficulty": 0.6, "role": "integ_sum"})
            templates.append({"expr": f"integral({var}^2 + {var}^3)", "domain": "integration", "difficulty": 0.7, "role": "integ_sum"})

        # Probability & Statistics: complement and universal only (solvable)
        templates.append({"expr": "P(A) + P(not A)", "domain": "probability_statistics", "difficulty": 0.5, "role": "prob_complement"})

        # Matrix Operations: distributive forms (solvable)
        templates.append({"expr": "2 * (a + b)", "domain": "matrix_operations", "difficulty": 0.4, "role": "scalar_mul"})
        templates.append({"expr": "3 * (a + b)", "domain": "matrix_operations", "difficulty": 0.4, "role": "scalar_mul"})

        return templates

    def _op_domain(self, op: str) -> str:
        if op in ("and", "or", "not"): return "logic"
        if op == "=": return "algebra"
        return "arithmetic"

    def generate_batch(self, n: int = 10, max_difficulty: float = 1.0,
                       domains: List[str] = None) -> List[dict]:
        """Generate n novel problems (50% template, 20% compositional, 30% cross-domain)."""
        candidates = [t for t in self._templates if t["difficulty"] <= max_difficulty]
        if domains:
            candidates = [t for t in candidates if t["domain"] in domains]
        if not candidates:
            candidates = self._templates

        selected = []
        attempts = 0
        while len(selected) < n and attempts < n * 8:
            attempts += 1
            roll = random.random()

            if roll < 0.50 or len(selected) == 0:
                # 50%: standard template
                t = random.choice(candidates)
                expr = t["expr"]
            elif roll < 0.70:
                # 20%: same-domain compositional
                pool = [t for t in candidates if t["difficulty"] < 0.4]
                if len(pool) >= 2:
                    t1, t2 = random.sample(pool, 2)
                    expr = f"({t1['expr']}) {random.choice(['+', '*'])} ({t2['expr']})"
                else:
                    expr = random.choice(candidates)["expr"]
            else:
                # 30%: cross-domain analogy (open-ended exploration)
                cross = self.generate_cross_domain(1)
                if cross:
                    selected.extend(cross[:1])
                    continue
                expr = random.choice(candidates)["expr"]

            sig = hashlib.md5(expr.encode()).hexdigest()[:8]
            if sig in self._generated:
                continue
            self._generated.add(sig)
            selected.append({
                "expression": expr,
                "domain": self._detect_domain(expr),
                "difficulty": self._estimate_difficulty(expr),
                "novel": True,
            })
        return selected

    def generate_cross_domain(self, n: int = 3) -> List[dict]:
        """Generate problems that span multiple domains via structural analogy."""
        selected = []
        shuffled = _CROSS_DOMAIN_ANALOGIES.copy()
        random.shuffle(shuffled)
        for analogy in shuffled:
            if len(selected) >= n:
                break
            expr = random.choice(analogy["exprs"])
            sig = hashlib.md5(expr.encode()).hexdigest()[:8]
            if sig in self._generated:
                continue
            self._generated.add(sig)
            selected.append({
                "expression": expr,
                "domain": self._detect_domain(expr),
                "difficulty": analogy["difficulty"],
                "novel": True,
                "role": analogy["role"],
                "cross_domain": True,
            })
        return selected

    def generate_novel(self, domain: str, n: int = 3) -> List[dict]:
        """Generate problems in a truly novel domain using operator templates."""
        ops = _NOVEL_DOMAINS.get(domain, [])
        if not ops:
            return self.generate_for_domain(domain, n)
        selected = []
        attempts = 0
        vars_ = _VARIABLES[:4]
        while len(selected) < n and attempts < n * 6:
            attempts += 1
            op = random.choice(ops)
            a, b = random.sample(vars_, 2)
            difficulty = 0.4 + random.random() * 0.3
            exprs = [
                f"{op}({a}, {b})",
                f"{a} {op} {b}",
                f"{op}({a})",
            ]
            expr = random.choice(exprs)
            sig = hashlib.md5(expr.encode()).hexdigest()[:8]
            if sig in self._generated:
                continue
            self._generated.add(sig)
            selected.append({
                "expression": expr,
                "domain": domain,
                "difficulty": difficulty,
                "novel": True,
            })
        return selected

    # Map curriculum domain names → template domain names
    _DOMAIN_ALIAS = {
        "logic_basics": "logic",
        "set_theory": "logic",
        "negation": "logic",
        "propositional_logic": "propositional",
        "constant_arithmetic": "arithmetic",
        "identity_basics": "arithmetic",
        "annihilation": "arithmetic",
        "combining_terms": "algebra",
        "linear_equations": "algebra",
        "distribution": "algebra",
        "factoring": "algebra",
        "cancellation_patterns": "algebra",
        "complex_simplification": "algebra",
        "power_rules": "algebra",
        "advanced_calculus": "calculus",
        "probability_statistics": "probability",
        "matrix_operations": "arithmetic",
    }

    def generate_for_domain(self, domain: str, n: int = 5,
                            difficulty_range: tuple = (0.0, 1.0)) -> List[dict]:
        """Generate problems specifically for a domain."""
        mapped = self._DOMAIN_ALIAS.get(domain, domain)
        candidates = [t for t in self._templates
                      if t["domain"] in (domain, mapped)
                      and difficulty_range[0] <= t["difficulty"] <= difficulty_range[1]]
        if not candidates:
            return []

        selected = []
        for _ in range(min(n, len(candidates))):
            t = random.choice(candidates)
            selected.append({
                "expression": t["expr"],
                "domain": domain,
                "difficulty": t["difficulty"],
                "role": t.get("role", ""),
            })
        return selected

    def _detect_domain(self, expr: str) -> str:
        e = expr.lower()
        if any(fn in e for fn in ("derivative", "diff", "integral")): return "calculus"
        if any(fn in e for fn in ("sin", "cos", "tan", "log", "sqrt")): return "trigonometry"
        if any(fn in e for fn in ("p(", "prob(", "p(empty", "omega")): return "probability"
        if any(fn in e for fn in ("angle_sum", "triangle")): return "geometry"
        if any(fn in e for fn in ("implies", "→", "==>")): return "propositional"
        if "and" in e or "or" in e or "not" in e: return "logic"
        if "=" in e: return "algebra"
        return "arithmetic"

    def _estimate_difficulty(self, expr: str) -> float:
        ops = sum(1 for c in expr if c in "+-*/^=")
        parens = expr.count("(")
        return min(1.0, ops * 0.12 + parens * 0.2 + len(expr) * 0.005)
