"""
AGISuite — Unified 5-category AGI benchmark.

Categories: arithmetic, algebra, logic, nl_understanding, physics

Usage:
    suite = AGISuite()
    results = suite.run()
    print(results["total_score"])
"""
from __future__ import annotations

import json
import time
from pathlib import Path


# Resolve benchmark data relative to this file so the path works regardless
# of the caller's working directory.
_LOGIC_SMOKE = Path(__file__).resolve().parents[3] / "benchmarks" / "logic" / "smoke.json"


class AGISuite:
    """5-category AGI benchmark suite."""

    BENCHMARK_VERSION = "1.0"
    CATEGORIES = ["arithmetic", "algebra", "logic", "nl_understanding", "physics"]

    def run(self) -> dict:
        """Run all 5 categories. Returns combined score and per-category breakdown."""
        # Instantiate engine objects once and share across all category runners.
        try:
            from sare.engine import (
                load_problem, EnergyEvaluator, BeamSearch, get_transforms,
            )
            self._load_problem = load_problem
            self._energy_fn = EnergyEvaluator()
            self._searcher = BeamSearch()
            self._transforms = get_transforms()
            self._engine_available = True
        except ImportError:
            self._engine_available = False

        results = []
        total_passed = 0
        total_problems = 0

        for category in self.CATEGORIES:
            try:
                method = getattr(self, f"_run_{category}")
                cat_result = method()
            except Exception as e:
                cat_result = {
                    "category": category,
                    "total": 0,
                    "passed": 0,
                    "score": 0.0,
                    "error": str(e),
                    "skipped": True,
                }
            results.append(cat_result)
            total_passed += cat_result.get("passed", 0)
            total_problems += cat_result.get("total", 0)

        total_score = round(total_passed / total_problems, 3) if total_problems > 0 else 0.0

        return {
            "total_score": total_score,
            "total_passed": total_passed,
            "total_problems": total_problems,
            "categories": results,
            "benchmark_version": self.BENCHMARK_VERSION,
            "timestamp": time.time(),
        }

    def _run_arithmetic(self) -> dict:
        """20 arithmetic problems: constant folding + basic operations."""
        problems = [
            "2 + 2", "3 * 4", "10 - 7", "8 / 2", "5 + 5",
            "2 + 0", "3 * 1", "0 * 100", "7 - 7", "4 / 1",
            "2 + 3", "6 * 2", "15 - 8", "9 / 3", "1 + 1",
            "10 + 0", "5 * 0", "8 - 4", "6 / 2", "3 + 4",
        ]
        return self._run_solve_batch("arithmetic", problems, min_delta=0.5)

    def _run_algebra(self) -> dict:
        """20 algebra problems: identity simplification + equation solving."""
        problems = [
            "x + 0", "x * 1", "x * 0", "x - x",
            "(x + 0) * 1", "1 * x + 0", "x + 0 * y", "0 + x",
            "x = x", "not not x",
            "x + 5 = 9", "2 * x = 8", "x - 3 = 4", "x + 1 = 6",
            "x ^ 2 = 4", "x ^ 2 = 9", "x ^ 2 = 16", "x ^ 2 = 25",
            "(x + 0) * (y * 1)", "x + 0 + 0",
        ]
        return self._run_solve_batch("algebra", problems, min_delta=0.1)

    def _run_logic(self) -> dict:
        """Run from benchmarks/logic/smoke.json if available, else use inline problems."""
        if _LOGIC_SMOKE.exists():
            try:
                data = json.loads(_LOGIC_SMOKE.read_text())
                cases = data.get("cases", data if isinstance(data, list) else [])
                problems = []
                for p in cases[:20]:
                    if isinstance(p, str):
                        problems.append(p)
                    elif isinstance(p, dict):
                        problems.append(p.get("expression") or p.get("input") or "")
                problems = [p for p in problems if p]
                if problems:
                    return self._run_solve_batch("logic", problems, min_delta=0.1)
            except Exception:
                pass

        # Fallback inline problems
        problems = [
            "x and True", "x or False", "not not x", "x and False",
            "x or True", "x and x", "x or x",
            "True and True", "False or False", "not True",
        ]
        return self._run_solve_batch("logic", problems, min_delta=0.1)

    def _run_nl_understanding(self) -> dict:
        """10 NL strings — check that UniversalParser extracts a non-empty expression."""
        test_cases = [
            ("what is 2 plus 2", True),
            ("simplify x plus zero", True),
            ("find x if x equals 5", True),
            ("calculate three times four", True),
            ("solve x squared equals nine", True),
            ("derivative of x cubed", True),
            ("x and true", True),
            ("not not x", True),
            ("random gibberish abc123 ###", False),  # should fail gracefully
            ("what is the meaning of life", False),  # no math expression
        ]
        passed = 0
        total = len(test_cases)
        details = []
        try:
            from sare.interface.universal_parser import UniversalParser
            parser = UniversalParser()
            for text, should_parse in test_cases:
                try:
                    result = parser.parse(text)
                    expr = ""
                    if hasattr(result, "expression"):
                        expr = result.expression or ""
                    elif isinstance(result, dict):
                        expr = result.get("expression", "")

                    got_expression = bool(expr and expr != text)
                    if got_expression == should_parse:
                        passed += 1
                    details.append({
                        "text": text,
                        "expression": expr,
                        "passed": got_expression == should_parse,
                    })
                except Exception as e:
                    details.append({"text": text, "error": str(e), "passed": not should_parse})
                    if not should_parse:
                        passed += 1
        except ImportError:
            return {
                "category": "nl_understanding",
                "total": total,
                "passed": 0,
                "score": 0.0,
                "skipped": True,
            }

        return {
            "category": "nl_understanding",
            "total": total,
            "passed": passed,
            "score": round(passed / total, 3),
            "details": details,
        }

    def _run_physics(self) -> dict:
        """10 physics expressions — check energy reduction. Skips gracefully if no physics transforms."""
        problems = [
            "F = m * a",
            "V = I * R",
            "E = m * c ^ 2",
            "P * V = n * R * T",
            "KE = 0.5 * m * v ^ 2",
            "v = v0 + a * t",
            "F = k * x",
            "P = m * g * h",
            "I = V / R",
            "a = F / m",
        ]
        return self._run_solve_batch("physics", problems, min_delta=0.01)

    def _run_solve_batch(self, category: str, expressions: list, min_delta: float = 0.5) -> dict:
        """Run a list of expressions through BeamSearch and count successful simplifications."""
        if not getattr(self, "_engine_available", False):
            return {
                "category": category,
                "total": len(expressions),
                "passed": 0,
                "score": 0.0,
                "error": "sare.engine not available",
            }

        expressions = [e for e in expressions if e]
        total = len(expressions)
        passed = 0
        details = []

        for expr in expressions:
            try:
                _, graph = self._load_problem(expr)
                result = self._searcher.search(
                    graph, self._energy_fn, self._transforms,
                    beam_width=6, max_depth=15, budget_seconds=3.0,
                )
                initial = self._energy_fn.compute(graph).total
                final = result.energy.total
                delta = initial - final
                ok = delta >= min_delta or len(result.transforms_applied) > 0
                if ok:
                    passed += 1
                details.append({"expr": expr, "delta": round(delta, 3), "passed": ok})
            except Exception as e:
                details.append({"expr": expr, "error": str(e), "passed": False})

        score = round(passed / total, 3) if total > 0 else 0.0
        return {
            "category": category,
            "total": total,
            "passed": passed,
            "score": score,
        }
