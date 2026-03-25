"""
ProblemFactory — Algorithmic symbolic math problem generator for SARE-HX.

Generates unlimited structured math problems deterministically (no LLM needed).
Problems cover algebra, arithmetic, simplification, logic, calculus, and
trigonometry.  Each generated problem is a dict with keys:
    expression  : str   — the input expression / equation
    domain      : str   — e.g. "algebra", "arithmetic"
    expected    : str   — expected simplified answer
    difficulty  : float — 0.0 (easy) to 1.0 (hard)
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Dict, List, Optional

# ── Module-level singleton ─────────────────────────────────────────────────────

_FACTORY_SINGLETON: "ProblemFactory | None" = None


def get_problem_factory() -> "ProblemFactory":
    """Return the module-level singleton ProblemFactory."""
    global _FACTORY_SINGLETON
    if _FACTORY_SINGLETON is None:
        _FACTORY_SINGLETON = ProblemFactory()
    return _FACTORY_SINGLETON


# ── Helpers ────────────────────────────────────────────────────────────────────

def _nonzero(rng: random.Random, lo: int, hi: int) -> int:
    """Return a random integer in [lo, hi] excluding 0."""
    v = 0
    while v == 0:
        v = rng.randint(lo, hi)
    return v


def _fmt_coeff(c: int, var: str = "x", leading: bool = True) -> str:
    """Format  c*var  as a compact string (e.g. 3*x, -x, x)."""
    if c == 1:
        return var if leading else f"+ {var}"
    if c == -1:
        return f"-{var}" if leading else f"- {var}"
    if c >= 0:
        return f"{c}*{var}" if leading else f"+ {c}*{var}"
    return f"{c}*{var}" if leading else f"- {abs(c)}*{var}"


def _fmt_const(c: int, leading: bool = True) -> str:
    if leading:
        return str(c)
    return f"+ {c}" if c >= 0 else f"- {abs(c)}"


def _difficulty(abs_max_coeff: int, has_fractions: bool = False,
                degree: int = 1) -> float:
    """Heuristic difficulty score 0..1."""
    base = min(abs_max_coeff / 20.0, 0.6)
    base += 0.15 * (degree - 1)
    if has_fractions:
        base += 0.15
    return round(min(base, 1.0), 2)


# ── ProblemFactory ─────────────────────────────────────────────────────────────

class ProblemFactory:
    """Generates structured math problems algorithmically."""

    # ── Linear equations: ax + b = c ─────────────────────────────────────────

    def generate_linear(self, n: int = 50) -> List[dict]:
        """
        Generate ax + b = c problems including negatives and fractions.
        Answer: x = (c - b) / a.
        """
        rng = random.Random(1001)
        problems: List[dict] = []

        for i in range(n):
            # Increase difficulty as i grows
            hard = i >= n // 2
            lo, hi = (-10, 10) if not hard else (-20, 20)
            a = _nonzero(rng, lo, hi)
            b = rng.randint(lo, hi)
            # Choose c so that the answer is "nice" (integer) for the first half,
            # fractional for the harder half.
            if not hard:
                x_val = rng.randint(-10, 10)
                c = a * x_val + b
                x_str = str(x_val)
                has_frac = False
            else:
                # c is arbitrary; x = (c-b)/a may be a fraction
                c = rng.randint(lo, hi)
                frac = Fraction(c - b, a)
                x_str = str(frac) if frac.denominator != 1 else str(int(frac))
                has_frac = frac.denominator != 1

            a_str = _fmt_coeff(a, "x", leading=True)
            b_str = _fmt_const(b, leading=False)
            expr = f"{a_str} {b_str} = {c}"

            problems.append({
                "expression": expr,
                "domain": "algebra",
                "expected": f"x = {x_str}",
                "difficulty": _difficulty(max(abs(a), abs(b), abs(c)), has_frac, 1),
            })

        return problems

    # ── Quadratic equations: ax² + bx + c = 0 ────────────────────────────────

    def generate_quadratic(self, n: int = 30) -> List[dict]:
        """
        Generate ax² + bx + c = 0 problems in factored form (integer roots).
        Answer: x = r1 or x = r2.
        """
        rng = random.Random(2002)
        problems: List[dict] = []

        for i in range(n):
            # Generate integer roots then expand
            r1 = rng.randint(-8, 8)
            r2 = rng.randint(-8, 8)
            # a*(x - r1)*(x - r2) expanded:
            a = _nonzero(rng, -3, 3)
            b = -a * (r1 + r2)
            c = a * r1 * r2

            # Expression string
            a_str = "" if a == 1 else (f"-" if a == -1 else f"{a}*")
            b_str = _fmt_coeff(b, "x", leading=False)
            c_str = _fmt_const(c, leading=False)
            expr = f"{a_str}x^2 {b_str} {c_str} = 0"

            # Answer
            roots = sorted(set([r1, r2]))
            ans = " or ".join(f"x = {r}" for r in roots)

            diff = _difficulty(max(abs(a), abs(b), abs(c)), False, 2)
            problems.append({
                "expression": expr,
                "domain": "algebra",
                "expected": ans,
                "difficulty": diff,
            })

        return problems

    # ── Arithmetic ────────────────────────────────────────────────────────────

    def generate_arithmetic(self, n: int = 50) -> List[dict]:
        """
        Mixed arithmetic: multiplication, division, modulo, powers, roots.
        """
        rng = random.Random(3003)
        problems: List[dict] = []

        ops = ["mul", "div", "mod", "pow", "sqrt", "add_frac", "sub_frac"]

        for i in range(n):
            op = ops[i % len(ops)]
            try:
                if op == "mul":
                    a = rng.randint(-15, 15)
                    b = rng.randint(-15, 15)
                    expr = f"{a} * {b}"
                    ans = str(a * b)
                    diff = _difficulty(max(abs(a), abs(b)))

                elif op == "div":
                    b = _nonzero(rng, -12, 12)
                    q = rng.randint(-12, 12)
                    a = b * q
                    expr = f"{a} / {b}"
                    ans = str(q)
                    diff = _difficulty(max(abs(a), abs(b)))

                elif op == "mod":
                    b = _nonzero(rng, 2, 15)
                    a = rng.randint(0, 50)
                    expr = f"{a} % {b}"
                    ans = str(a % b)
                    diff = 0.3

                elif op == "pow":
                    base = rng.randint(-5, 5)
                    exp = rng.randint(2, 4)
                    expr = f"{base}^{exp}"
                    ans = str(base ** exp)
                    diff = _difficulty(abs(base), degree=exp)

                elif op == "sqrt":
                    sq = rng.randint(1, 12) ** 2
                    expr = f"sqrt({sq})"
                    ans = str(int(math.isqrt(sq)))
                    diff = 0.2

                elif op == "add_frac":
                    # a/b + c/d with small integers
                    b2 = _nonzero(rng, 1, 8)
                    d = _nonzero(rng, 1, 8)
                    a2 = rng.randint(1, 7)
                    c = rng.randint(1, 7)
                    frac = Fraction(a2, b2) + Fraction(c, d)
                    expr = f"{a2}/{b2} + {c}/{d}"
                    ans = str(frac) if frac.denominator != 1 else str(int(frac))
                    diff = 0.35

                else:  # sub_frac
                    b2 = _nonzero(rng, 1, 8)
                    d = _nonzero(rng, 1, 8)
                    a2 = rng.randint(1, 7)
                    c = rng.randint(1, 7)
                    frac = Fraction(a2, b2) - Fraction(c, d)
                    expr = f"{a2}/{b2} - {c}/{d}"
                    ans = str(frac) if frac.denominator != 1 else str(int(frac))
                    diff = 0.35

                problems.append({
                    "expression": expr,
                    "domain": "arithmetic",
                    "expected": ans,
                    "difficulty": diff,
                })
            except Exception:
                # Skip any degenerate case
                continue

        return problems

    # ── Simplification ────────────────────────────────────────────────────────

    def generate_simplification(self, n: int = 50) -> List[dict]:
        """
        Simplification problems: trig identities, cancel like terms,
        distribution, double negation in algebra.
        """
        rng = random.Random(4004)
        problems: List[dict] = []

        templates = [
            # (expression_template, expected, domain, difficulty)
            # Trig identities
            ("sin(x)^2 + cos(x)^2", "1", "trigonometry", 0.1),
            ("1 - sin(x)^2", "cos(x)^2", "trigonometry", 0.2),
            ("1 - cos(x)^2", "sin(x)^2", "trigonometry", 0.2),
            ("sin(2*x)", "2*sin(x)*cos(x)", "trigonometry", 0.3),
            ("cos(2*x)", "cos(x)^2 - sin(x)^2", "trigonometry", 0.3),
            ("tan(x)", "sin(x)/cos(x)", "trigonometry", 0.2),
            ("sin(x + y)", "sin(x)*cos(y) + cos(x)*sin(y)", "trigonometry", 0.4),
            ("cos(x + y)", "cos(x)*cos(y) - sin(x)*sin(y)", "trigonometry", 0.4),
            # Algebra: distribution
            ("a*(b + c)", "a*b + a*c", "algebra", 0.1),
            ("(x + 1)*(x - 1)", "x^2 - 1", "algebra", 0.2),
            ("(x + 2)^2", "x^2 + 4*x + 4", "algebra", 0.3),
            ("(x - 3)^2", "x^2 - 6*x + 9", "algebra", 0.3),
            ("(x + y)*(x - y)", "x^2 - y^2", "algebra", 0.2),
            ("(2*x + 1)^2", "4*x^2 + 4*x + 1", "algebra", 0.4),
            ("(3*x - 2)*(3*x + 2)", "9*x^2 - 4", "algebra", 0.35),
            # Like-term cancellation
            ("x + x", "2*x", "algebra", 0.1),
            ("3*x + 2*x", "5*x", "algebra", 0.1),
            ("5*x - 3*x", "2*x", "algebra", 0.1),
            ("x*y + x*y", "2*x*y", "algebra", 0.15),
            ("a^2 + a^2", "2*a^2", "algebra", 0.15),
            ("x/x", "1", "algebra", 0.1),
            ("x^2/x", "x", "algebra", 0.2),
            ("(a*b)/b", "a", "algebra", 0.2),
            # Exponential / log simplifications
            ("e^(ln(x))", "x", "algebra", 0.3),
            ("ln(e^x)", "x", "algebra", 0.3),
            ("log(a) + log(b)", "log(a*b)", "algebra", 0.3),
            ("log(a) - log(b)", "log(a/b)", "algebra", 0.3),
            ("n*log(a)", "log(a^n)", "algebra", 0.35),
            # Double negation (algebraic)
            ("--x", "x", "algebra", 0.1),
            ("-(-(a + b))", "a + b", "algebra", 0.2),
        ]

        # Cycle through templates, adding numeric variations
        for i in range(n):
            t = templates[i % len(templates)]
            expr, expected, domain, diff = t
            # Add slight variation for later cycles
            if i >= len(templates):
                # Prefix the expression with a trivially addable constant that cancels
                c = rng.randint(1, 5)
                expr = f"({expr} + {c}) - {c}"
                # expected stays the same since +c -c = 0
            problems.append({
                "expression": expr,
                "domain": domain,
                "expected": expected,
                "difficulty": round(min(diff + 0.05 * (i // len(templates)), 1.0), 2),
            })

        return problems

    # ── Logic ────────────────────────────────────────────────────────────────

    def generate_logic(self, n: int = 30) -> List[dict]:
        """
        Propositional logic simplifications: double negation, and/or
        with true/false, De Morgan.
        """
        rng = random.Random(5005)
        problems: List[dict] = []

        templates = [
            # Double negation
            ("not(not(p))", "p", 0.1),
            ("not(not(q))", "q", 0.1),
            ("not(not(p and q))", "p and q", 0.2),
            ("not(not(p or q))", "p or q", 0.2),
            # And with true/false
            ("p and true", "p", 0.1),
            ("p and false", "false", 0.1),
            ("true and p", "p", 0.1),
            ("false and p", "false", 0.1),
            # Or with true/false
            ("p or true", "true", 0.1),
            ("p or false", "p", 0.1),
            ("true or p", "true", 0.1),
            ("false or p", "p", 0.1),
            # Idempotent
            ("p and p", "p", 0.15),
            ("p or p", "p", 0.15),
            # Tautology / contradiction
            ("p or not(p)", "true", 0.2),
            ("p and not(p)", "false", 0.2),
            # De Morgan
            ("not(p and q)", "not(p) or not(q)", 0.3),
            ("not(p or q)", "not(p) and not(q)", 0.3),
            ("not(p) or not(q)", "not(p and q)", 0.35),
            ("not(p) and not(q)", "not(p or q)", 0.35),
            # Implication
            ("p implies q", "not(p) or q", 0.4),
            ("not(p implies q)", "p and not(q)", 0.45),
            # Contrapositive
            ("not(q) implies not(p)", "p implies q", 0.4),
            # Absorption
            ("p and (p or q)", "p", 0.3),
            ("p or (p and q)", "p", 0.3),
            # Distributive
            ("p and (q or r)", "(p and q) or (p and r)", 0.4),
            ("p or (q and r)", "(p or q) and (p or r)", 0.4),
            # Commutative
            ("p and q", "q and p", 0.1),
            ("p or q", "q or p", 0.1),
            # Double application
            ("not(not(p)) and q", "p and q", 0.25),
        ]

        for i in range(n):
            t = templates[i % len(templates)]
            expr, expected, diff = t
            # For later cycles, nest more deeply
            if i >= len(templates):
                expr = f"not(not({expr}))"
                expected = f"{expected}"
                diff = min(diff + 0.1, 1.0)
            problems.append({
                "expression": expr,
                "domain": "logic",
                "expected": expected,
                "difficulty": round(diff, 2),
            })

        return problems

    # ── Calculus ─────────────────────────────────────────────────────────────

    def generate_calculus(self, n: int = 20) -> List[dict]:
        """
        Derivative and integral problems: power rule, standard functions,
        chain / product rule patterns.
        """
        rng = random.Random(6006)
        problems: List[dict] = []

        # Build from templates with varying exponents / coefficients
        for i in range(n):
            kind = i % 4

            if kind == 0:
                # Derivative of a*x^n
                a = _nonzero(rng, -8, 8)
                exp = rng.randint(1, 5)
                new_a = a * exp
                new_exp = exp - 1
                expr = f"derivative({a}*x^{exp})"
                if new_exp == 0:
                    ans = str(new_a)
                elif new_exp == 1:
                    ans = f"{new_a}*x" if new_a != 1 else "x"
                else:
                    ans = f"{new_a}*x^{new_exp}"
                diff = _difficulty(max(abs(a), exp), degree=exp)

            elif kind == 1:
                # Integral of a*x^n
                a = _nonzero(rng, -6, 6)
                exp = rng.randint(0, 4)
                new_exp = exp + 1
                # Simplify coefficient: a / new_exp as fraction
                frac = Fraction(a, new_exp)
                c_str = str(frac) if frac.denominator != 1 else str(int(frac))
                expr = f"integral({a}*x^{exp})"
                ans = f"{c_str}*x^{new_exp} + C"
                diff = _difficulty(max(abs(a), new_exp), has_fractions=(frac.denominator != 1))

            elif kind == 2:
                # Derivative of standard functions
                fn, d_fn, d_diff = rng.choice([
                    ("sin(x)", "cos(x)", 0.2),
                    ("cos(x)", "-sin(x)", 0.2),
                    ("e^x", "e^x", 0.2),
                    ("ln(x)", "1/x", 0.25),
                    ("tan(x)", "sec(x)^2", 0.3),
                ])
                expr = f"derivative({fn})"
                ans = d_fn
                diff = d_diff

            else:
                # Derivative of a*f(x) + b*g(x) (linearity)
                a = _nonzero(rng, -5, 5)
                b = _nonzero(rng, -5, 5)
                f1, df1 = rng.choice([("sin(x)", "cos(x)"), ("e^x", "e^x")])
                f2, df2 = rng.choice([("cos(x)", "-sin(x)"), ("ln(x)", "1/x")])
                a_str = str(a) if a != 1 else ""
                b_str = str(b) if b != 1 else ""
                expr = f"derivative({a}*{f1} + {b}*{f2})"
                t1 = f"{a}*{df1}" if a != 1 else df1
                t2 = f"{b}*{df2}" if b != 1 else df2
                ans = f"{t1} + {t2}"
                diff = 0.5

            problems.append({
                "expression": expr,
                "domain": "calculus",
                "expected": ans,
                "difficulty": round(diff, 2),
            })

        return problems

    # ── Chemistry ─────────────────────────────────────────────────────────────

    def generate_chemistry(self, n: int = 20) -> List[dict]:
        """Generate chemistry domain problems."""
        templates = [
            {"expression": "P * V = n * R * T", "domain": "chemistry", "difficulty": 0.4},
            {"expression": "1 * H2 + O2", "domain": "chemistry", "difficulty": 0.2},
            {"expression": "2 * H2 + 1 * O2", "domain": "chemistry", "difficulty": 0.2},
            {"expression": "n * N_A", "domain": "chemistry", "difficulty": 0.3},
            {"expression": "mass = moles * molar_mass", "domain": "chemistry", "difficulty": 0.4},
            {"expression": "pH = 0 - log_H", "domain": "chemistry", "difficulty": 0.5},
            {"expression": "rate = k * A * B", "domain": "chemistry", "difficulty": 0.5},
            {"expression": "delta_G = delta_H - T * delta_S", "domain": "chemistry", "difficulty": 0.6},
            {"expression": "C1 * V1 = C2 * V2", "domain": "chemistry", "difficulty": 0.4},
            {"expression": "yield = actual / theoretical * 100", "domain": "chemistry", "difficulty": 0.5},
            {"expression": "1 * NaCl + H2O", "domain": "chemistry", "difficulty": 0.2},
            {"expression": "2 * H2O", "domain": "chemistry", "difficulty": 0.1},
            {"expression": "E = h * frequency", "domain": "chemistry", "difficulty": 0.4},
            {"expression": "lambda = h / p", "domain": "chemistry", "difficulty": 0.5},
            {"expression": "n1 * V1 = n2 * V2", "domain": "chemistry", "difficulty": 0.4},
            {"expression": "K_eq = products / reactants", "domain": "chemistry", "difficulty": 0.6},
            {"expression": "entropy = k * ln_W", "domain": "chemistry", "difficulty": 0.7},
            {"expression": "activation_E = R * T * ln_k", "domain": "chemistry", "difficulty": 0.7},
            {"expression": "1 * Fe + 1 * S = FeS", "domain": "chemistry", "difficulty": 0.3},
            {"expression": "molarity = moles / volume", "domain": "chemistry", "difficulty": 0.3},
        ]
        result = templates[:n] if n <= len(templates) else templates + random.choices(templates, k=n - len(templates))
        return result[:n]

    # ── Batch ────────────────────────────────────────────────────────────────

    def generate_batch(self, total: int = 200, seed: Optional[int] = None) -> List[dict]:
        """
        Generate a mixed batch of all problem types.

        Each dict: {expression, domain, expected, difficulty}.
        Problems are shuffled deterministically when ``seed`` is given.
        """
        rng = random.Random(seed if seed is not None else 0)

        # Proportional counts (sum to ~200 by default)
        linear_n      = max(1, int(total * 0.25))  # 25% linear
        quad_n        = max(1, int(total * 0.15))  # 15% quadratic
        arith_n       = max(1, int(total * 0.25))  # 25% arithmetic
        simp_n        = max(1, int(total * 0.15))  # 15% simplification
        logic_n       = max(1, int(total * 0.10))  # 10% logic
        calc_n        = max(1, int(total * 0.10))  # 10% calculus

        pool: List[dict] = []
        try:
            pool += self.generate_linear(linear_n)
        except Exception:
            pass
        try:
            pool += self.generate_quadratic(quad_n)
        except Exception:
            pass
        try:
            pool += self.generate_arithmetic(arith_n)
        except Exception:
            pass
        try:
            pool += self.generate_simplification(simp_n)
        except Exception:
            pass
        try:
            pool += self.generate_logic(logic_n)
        except Exception:
            pass
        try:
            pool += self.generate_calculus(calc_n)
        except Exception:
            pass
        try:
            chem_n = max(1, int(total * 0.05))  # 5% chemistry
            pool += self.generate_chemistry(chem_n)
        except Exception:
            pass

        rng.shuffle(pool)
        return pool[:total]

    def generate_hard(self, domain: str, n: int = 2, complexity: int = 3) -> List[dict]:
        """
        Generate harder problems for a mastered domain via expression mutation.

        complexity=1: single nesting (x+0)*(y*1)
        complexity=2: double redundancy ((x+0)+0)*(1*(y*1))
        complexity=3: 5+ ops, cross-domain mix or multi-step chains
        """
        rng = random.Random(hash(domain) ^ complexity)
        vars_ = ["x", "y", "a", "b", "m", "n"]
        results: List[dict] = []

        # Templates by complexity level
        c1 = [
            "({v1}+0)*({v2}*1)", "{v1}*1+{v2}*1", "({v1}-{v1})+{v2}",
            "{v1}+0+{v2}*1", "({v1}+{v2})*1",
        ]
        c2 = [
            "(({v1}+0)+0)*(1*({v2}*1))", "{v1}*1*1+{v2}+0",
            "not not ({v1}+0)", "(({v1}+{v2})*1+0)*1",
        ]
        c3_algebra = [
            "{v1}^2 + 2*{v1}*{v2} + {v2}^2",   # (v1+v2)^2
            "{v1}^2 - {v2}^2",                    # difference of squares
            "({v1}+{v2})*({v1}-{v2})",            # expand to v1^2-v2^2
            "{v1}^2 + 0*{v1} + 0",               # trivial quadratic
        ]
        c3_calculus = [
            "d/dx({v1}^2 + {v2}^2)", "d/dx({v1}*{v2})", "integral({v1}^2 + 1)",
        ]
        c3_logic = [
            "not not not not {v1}", "({v1} and True) or False",
            "not ({v1} and False)", "({v1} or False) and True",
        ]

        if complexity == 1:
            templates = c1
        elif complexity == 2:
            templates = c2
        else:
            if domain in ("calculus",):
                templates = c3_calculus
            elif domain in ("logic",):
                templates = c3_logic
            else:
                templates = c3_algebra

        for _ in range(n):
            tmpl = rng.choice(templates)
            v1, v2 = rng.choice(vars_), rng.choice(vars_)
            try:
                expr = tmpl.format(v1=v1, v2=v2)
                results.append({"expression": expr, "domain": domain, "difficulty": complexity + 5})
            except Exception:
                pass

        return results
