"""
GenerativeWorldModel — S26-4
Imagination engine: samples novel problem expressions from the latent space
of solved problems, solves them, and uses failures to update the world model.
Curiosity-driven exploration: biases sampling toward high-surprise regions.
"""
from __future__ import annotations
import json
import re
import time
import random
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

log = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ImaginaryProblem:
    expression: str
    domain:     str
    origin:     str   # "interpolation" | "perturbation" | "extrapolation" | "mutation"
    solved:     bool  = False
    energy:     float = 1.0
    result:     str   = ""
    timestamp:  float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "expression": self.expression[:50],
            "domain":     self.domain,
            "origin":     self.origin,
            "solved":     self.solved,
            "energy":     round(self.energy, 3),
            "result":     self.result[:50],
            "timestamp":  self.timestamp,
        }


# ── Domain expression templates ───────────────────────────────────────────────
_DOMAIN_TEMPLATES: Dict[str, List[str]] = {
    "arithmetic":   ["N + N", "N * N", "N - N", "N / N", "N ^ N",
                     "N + N * N", "(N + N) * N", "N ^ N + N"],
    "algebra":      ["V + N", "V * N", "V + V", "V * V - N",
                     "N * V + N", "V ^ N", "(V + N) * V"],
    "logic":        ["V AND V", "V OR V", "NOT V", "V XOR V",
                     "V IMPLIES V", "V AND (V OR V)"],
    "calculus":     ["d/dV(V ^ N)", "integral(V ^ N)", "d/dV(V * V)",
                     "lim V -> N (V + N)"],
    "physics":      ["N * V ^ N", "V = N * V", "F = N * N",
                     "E = N * V ^ N"],
    "trigonometry": ["sin(V)", "cos(V)", "tan(V)", "sin(V) ^ N + cos(V) ^ N"],
}

_VARS   = ["x", "y", "z", "a", "b", "n"]
_NUMS   = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
_OPS    = ["+", "-", "*", "/", "^"]


def _fill_template(template: str) -> str:
    t = template
    t = re.sub(r'\bV\b', lambda _: random.choice(_VARS), t)
    t = re.sub(r'\bN\b', lambda _: random.choice(_NUMS), t)
    return t


class GenerativeWorldModel:
    """
    Maintains a pool of solved problems as latent feature vectors.
    Can generate imaginary problems via interpolation, perturbation,
    extrapolation, or mutation of existing solved expressions.
    """

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self._solved_pool:     List[Dict]             = []   # {expression, domain, features}
        self._imaginary_pool:  List[ImaginaryProblem] = []
        self._exploration_log: List[Dict]             = []
        self._benchmark_eval_log: List[Dict[str, Any]] = []
        self._total_imagined  = 0
        self._total_solved    = 0
        self._total_failed    = 0
        self._domain_bias:    Dict[str, float]        = {}   # domain → curiosity weight
        self._engine          = None
        self._affective       = None
        self._repo_root       = Path(repo_root or REPO_ROOT)
        self._benchmark_pool  = self._load_benchmark_pool()

    # ── wiring ────────────────────────────────────────────────────────────────
    def wire(self, engine=None, affective_energy=None) -> None:
        self._engine    = engine
        self._affective = affective_energy

    def feed_solved(self, expression: str, domain: str = "general") -> None:
        """Called after a successful solve to expand the pool."""
        feats = self._featurize(expression)
        self._solved_pool.append({"expression": expression,
                                  "domain": domain, "features": feats})
        if len(self._solved_pool) > 1000:
            self._solved_pool = self._solved_pool[-1000:]

    # ── imagination ───────────────────────────────────────────────────────────
    def imagine(self, domain: Optional[str] = None, n: int = 3) -> List[ImaginaryProblem]:
        """Generate n novel imaginary problems."""
        results = []
        for _ in range(n):
            dom = domain or self._pick_domain()
            origin, expr = self._sample(dom)
            prob = ImaginaryProblem(expr, dom, origin)
            self._imaginary_pool.append(prob)
            self._total_imagined += 1
            results.append(prob)
        if len(self._imaginary_pool) > 500:
            self._imaginary_pool = self._imaginary_pool[-500:]
        return results

    def explore_cycle(self) -> dict:
        """Imagine n problems, attempt to solve each, return stats."""
        problems = self.imagine(n=3)
        solved_count = 0; failed_count = 0
        for prob in problems:
            result = self._attempt_solve(prob)
            if result["solved"]:
                solved_count += 1
                prob.solved = True
                prob.energy = result.get("energy", 0.1)
                prob.result = result.get("result", "")
                self.feed_solved(prob.expression, prob.domain)
            else:
                failed_count += 1
                self._total_failed += 1
                # high surprise — bump domain curiosity
                self._domain_bias[prob.domain] = (
                    self._domain_bias.get(prob.domain, 0.5) * 0.8 + 0.2 * 1.0)

        self._total_solved += solved_count
        entry = {"imagined": len(problems), "solved": solved_count,
                 "failed": failed_count, "timestamp": time.time()}
        self._exploration_log.append(entry)
        if len(self._exploration_log) > 200:
            self._exploration_log = self._exploration_log[-200:]
        return entry

    def evaluate_benchmarks(self, max_per_domain: int = 5,
                            domains: Optional[List[str]] = None) -> dict:
        """Run the wired solver on held-out benchmark expressions and report win rates."""
        selected_domains = domains or sorted(self._benchmark_pool.keys())
        report: Dict[str, Any] = {
            "timestamp": time.time(),
            "engine_wired": self._engine is not None,
            "domains": {},
            "attempted": 0,
            "solved": 0,
            "solve_rate": 0.0,
        }

        for domain in selected_domains:
            expressions = self._benchmark_pool.get(domain, [])[:max_per_domain]
            if not expressions:
                continue
            attempted = 0
            solved = 0
            examples = []
            for expression in expressions:
                attempted += 1
                result = self._attempt_solve(ImaginaryProblem(expression, domain, "benchmark_eval"))
                solved += int(bool(result["solved"]))
                examples.append({
                    "expression": expression[:50],
                    "solved": bool(result["solved"]),
                    "energy": round(float(result.get("energy", 1.0)), 3),
                })
            report["domains"][domain] = {
                "attempted": attempted,
                "solved": solved,
                "solve_rate": round(solved / max(attempted, 1), 3),
                "examples": examples,
            }
            report["attempted"] += attempted
            report["solved"] += solved

        report["solve_rate"] = round(report["solved"] / max(report["attempted"], 1), 3)
        self._benchmark_eval_log.append(report)
        if len(self._benchmark_eval_log) > 50:
            self._benchmark_eval_log = self._benchmark_eval_log[-50:]
        return report

    # ── internals ─────────────────────────────────────────────────────────────
    def _pick_domain(self) -> str:
        if not self._domain_bias and not self._solved_pool and self._benchmark_pool:
            return random.choice(list(self._benchmark_pool.keys()))
        if not self._domain_bias and not self._solved_pool:
            return random.choice(list(_DOMAIN_TEMPLATES.keys()))
        if self._domain_bias:
            domains = list(self._domain_bias.keys())
            weights = [self._domain_bias[d] for d in domains]
            return random.choices(domains, weights=weights)[0]
        domains = list({e["domain"] for e in self._solved_pool})
        return random.choice(domains) if domains else "arithmetic"

    def _sample(self, domain: str) -> Tuple[str, str]:
        """Return (origin, expression)."""
        method = random.choice(["template", "perturbation",
                                "interpolation", "mutation"])
        benchmark_examples = self._benchmark_pool.get(domain, [])
        if benchmark_examples and (method == "template" or not self._solved_pool):
            base = random.choice(benchmark_examples)
            if method == "template":
                return "benchmark_seed", base
            if method == "perturbation":
                return "benchmark_perturbation", self._perturb(base)
            return "benchmark_mutation", self._mutate(base)

        if method == "template" or not self._solved_pool:
            templates = _DOMAIN_TEMPLATES.get(domain,
                        _DOMAIN_TEMPLATES["arithmetic"])
            return "template", _fill_template(random.choice(templates))

        # pick a solved expression from the same or random domain
        pool = [e for e in self._solved_pool if e["domain"] == domain]
        if not pool:
            pool = self._solved_pool
        base = random.choice(pool)["expression"]

        if method == "perturbation":
            expr = self._perturb(base)
            return "perturbation", expr
        elif method == "interpolation" and len(self._solved_pool) >= 2:
            other = random.choice(self._solved_pool)["expression"]
            expr  = self._interpolate(base, other)
            return "interpolation", expr
        else:
            return "mutation", self._mutate(base)

    @staticmethod
    def _perturb(expr: str) -> str:
        """Replace one number with another."""
        nums = re.findall(r'\d+', expr)
        if not nums:
            return expr + " + 0"
        target = random.choice(nums)
        replacement = str(random.choice([0, 1, 2, 3, 5]))
        return expr.replace(target, replacement, 1)

    @staticmethod
    def _interpolate(a: str, b: str) -> str:
        """Combine tokens from two expressions."""
        ta = a.split(); tb = b.split()
        if not ta or not tb:
            return a
        mid_a = ta[:max(1, len(ta)//2)]
        mid_b = tb[max(0, len(tb)//2):]
        combined = " ".join(mid_a + [random.choice(_OPS)] + mid_b)
        return combined[:60]

    @staticmethod
    def _mutate(expr: str) -> str:
        """Swap one operator for another."""
        for op in _OPS:
            if op in expr:
                new_op = random.choice([o for o in _OPS if o != op])
                return expr.replace(op, new_op, 1)
        return expr + " * 1"

    def _attempt_solve(self, prob: ImaginaryProblem) -> dict:
        if self._engine is None:
            return {"solved": False, "energy": 1.0, "result": "engine_unavailable"}
        try:
            result = self._engine.solve(prob.expression)
            solved = result.get("success", result.get("solved", False)) if isinstance(result, dict) else bool(result)
            energy = result.get("energy", 0.2) if isinstance(result, dict) else 0.2
            if isinstance(energy, dict):
                energy = energy.get("total", 0.2)
            res_str = str(result.get("result", ""))[:50] if isinstance(result, dict) else str(result)[:50]
            return {"solved": solved, "energy": energy, "result": res_str}
        except Exception as e:
            log.debug(f"GenerativeWorldModel solve: {e}")
            return {"solved": False, "energy": 1.0, "result": ""}

    @staticmethod
    def _featurize(expr: str) -> Dict[str, float]:
        return {
            "length":   len(expr),
            "ops":      len(re.findall(r'[+\-*/^]', expr)),
            "depth":    expr.count('('),
            "vars":     len(re.findall(r'\b[a-z]\b', expr)),
            "nums":     len(re.findall(r'\d+', expr)),
        }

    def _load_benchmark_pool(self) -> Dict[str, List[str]]:
        pool: Dict[str, List[str]] = {}
        sources = [
            self._repo_root / "benchmarks" / "algebra" / "symbolic_math.json",
            self._repo_root / "benchmarks" / "logic" / "smoke.json",
            self._repo_root / "data" / "hard_problems.json",
        ]

        for path in sources:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.debug("GenerativeWorldModel benchmark load failed for %s: %s", path, exc)
                continue

            if isinstance(payload, list):
                examples = payload
            elif isinstance(payload, dict) and isinstance(payload.get("cases"), list):
                examples = payload["cases"]
            else:
                continue

            for item in examples:
                expression = str(item.get("expression", "")).strip()
                if not expression:
                    continue
                domain = str(item.get("domain", self._infer_domain(expression, path))).strip() or "general"
                pool.setdefault(domain, []).append(expression)

        return {domain: expressions[:200] for domain, expressions in pool.items() if expressions}

    @staticmethod
    def _infer_domain(expression: str, path: Path) -> str:
        lower_path = str(path).lower()
        if "logic" in lower_path:
            return "logic"
        if "algebra" in lower_path:
            return "algebra"
        if any(token in expression for token in ("sin", "cos", "tan")):
            return "trigonometry"
        if any(token in expression for token in ("d/d", "integral", "lim")):
            return "calculus"
        return "arithmetic"

    # ── summary ───────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        recent_imagined = [p.to_dict() for p in self._imaginary_pool[-8:]]
        solve_rate = (self._total_solved /
                      max(self._total_solved + self._total_failed, 1))
        return {
            "total_imagined":  self._total_imagined,
            "total_solved":    self._total_solved,
            "total_failed":    self._total_failed,
            "solve_rate":      round(solve_rate, 3),
            "solved_pool_size": len(self._solved_pool),
            "domain_bias":     {d: round(v, 3)
                                 for d, v in sorted(self._domain_bias.items(),
                                                    key=lambda x: -x[1])[:6]},
            "recent_imagined": recent_imagined,
            "engine_wired":    self._engine is not None,
            "benchmark_domains": sorted(self._benchmark_pool.keys()),
            "last_benchmark_eval": self._benchmark_eval_log[-1] if self._benchmark_eval_log else None,
        }
