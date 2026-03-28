"""
AGISuite — Unified 10-category AGI benchmark.

Categories:
    arithmetic, algebra, logic, nl_understanding, physics,
    chemistry, code, symbolic_math, arc, synthesis

Usage:
    suite = AGISuite()
    results = suite.run()
    print(results["total_score"])
"""
from __future__ import annotations

import json
import os
import threading as _thr
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOGIC_SMOKE      = _REPO_ROOT / "benchmarks" / "logic" / "smoke.json"
_SYMBOLIC_MATH    = _REPO_ROOT / "benchmarks" / "algebra" / "symbolic_math.json"
_CODING_SIMPLIFY  = _REPO_ROOT / "benchmarks" / "coding" / "simplify.json"
_BENCHMARK_HIST   = _REPO_ROOT / "data" / "memory" / "benchmark_history.json"


class AGISuite:
    """10-category unified AGI benchmark suite."""

    BENCHMARK_VERSION = "2.0"
    CATEGORIES = [
        "arithmetic", "algebra", "logic", "nl_understanding",
        "physics", "chemistry", "code", "symbolic_math",
        "arc", "synthesis",
    ]

    def run(self) -> dict:
        """Run all 10 categories. Returns combined score and per-category breakdown."""
        try:
            from sare.engine import load_problem, EnergyEvaluator, BeamSearch, get_transforms
            self._load_problem = load_problem
            self._energy_fn    = EnergyEvaluator()
            self._searcher     = BeamSearch()
            self._transforms   = get_transforms()
            self._engine_ok    = True
        except ImportError:
            self._engine_ok = False

        results        = []
        total_passed   = 0
        total_problems = 0

        for category in self.CATEGORIES:
            try:
                cat_result = getattr(self, f"_run_{category}")()
            except Exception as e:
                cat_result = {
                    "category": category,
                    "total": 0, "passed": 0, "score": 0.0,
                    "error": str(e), "skipped": True,
                }
            results.append(cat_result)
            total_passed   += cat_result.get("passed", 0)
            total_problems += cat_result.get("total", 0)

        total_score = round(total_passed / total_problems, 3) if total_problems > 0 else 0.0

        result = {
            "total_score":    total_score,
            "total_passed":   total_passed,
            "total_problems": total_problems,
            "categories":     results,
            "benchmark_version": self.BENCHMARK_VERSION,
            "timestamp": time.time(),
        }
        self._save_history(result)
        return result

    # ── Category runners ────────────────────────────────────────────────────

    def _run_arithmetic(self) -> dict:
        problems = [
            "2 + 2", "3 * 4", "10 - 7", "8 / 2", "5 + 5",
            "2 + 0", "3 * 1", "0 * 100", "7 - 7", "4 / 1",
            "2 + 3", "6 * 2", "15 - 8", "9 / 3", "1 + 1",
            "10 + 0", "5 * 0", "8 - 4", "6 / 2", "3 + 4",
        ]
        return self._run_solve_batch("arithmetic", problems, min_delta=0.5)

    def _run_algebra(self) -> dict:
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
        if _LOGIC_SMOKE.exists():
            try:
                data   = json.loads(_LOGIC_SMOKE.read_text())
                cases  = data.get("cases", data if isinstance(data, list) else [])
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
        # Fallback
        problems = [
            "x and True", "x or False", "not not x", "x and False",
            "x or True", "x and x", "x or x",
            "True and True", "False or False", "not True",
        ]
        return self._run_solve_batch("logic", problems, min_delta=0.1)

    def _run_nl_understanding(self) -> dict:
        test_cases = [
            ("what is 2 plus 2", True),
            ("simplify x plus zero", True),
            ("find x if x equals 5", True),
            ("calculate three times four", True),
            ("solve x squared equals nine", True),
            ("derivative of x cubed", True),
            ("x and true", True),
            ("not not x", True),
            ("random gibberish abc123 ###", False),
            ("what is the meaning of life", False),
        ]
        passed = 0
        total  = len(test_cases)
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
                    ok = (got_expression == should_parse)
                    if ok:
                        passed += 1
                    details.append({"text": text, "expression": expr, "passed": ok})
                except Exception as e:
                    details.append({"text": text, "error": str(e), "passed": not should_parse})
                    if not should_parse:
                        passed += 1
        except ImportError:
            return {"category": "nl_understanding", "total": total, "passed": 0,
                    "score": 0.0, "skipped": True}

        return {"category": "nl_understanding", "total": total, "passed": passed,
                "score": round(passed / total, 3), "details": details}

    def _run_physics(self) -> dict:
        problems = [
            "F = m * a", "V = I * R", "E = m * c ^ 2",
            "P * V = n * R * T", "KE = 0.5 * m * v ^ 2",
            "v = v0 + a * t", "F = k * x", "P = m * g * h",
            "I = V / R", "a = F / m",
        ]
        return self._run_solve_batch("physics", problems, min_delta=0.01)

    def _run_chemistry(self) -> dict:
        """10 chemistry expressions using stoichiometry / ideal gas / mass conservation."""
        problems = [
            "2 * H2 + O2",
            "P * V = n * R * T",
            "n * Avogadro = N",
            "mass_reactants = mass_products",
            "CH4 + 2 * O2",
            "N2 + 3 * H2",
            "C6H12O6 + 6 * O2",
            "2 * Na + Cl2",
            "CaCO3 = CaO + CO2",
            "H2 + Cl2 = 2 * HCl",
        ]
        return self._run_solve_batch("chemistry", problems, min_delta=0.01)

    def _run_code(self) -> dict:
        """Code simplification problems — from file or inline fallback."""
        problems = []
        if _CODING_SIMPLIFY.exists():
            try:
                data  = json.loads(_CODING_SIMPLIFY.read_text())
                cases = data.get("cases", [])
                problems = [c["expression"] for c in cases if c.get("expression")]
            except Exception:
                pass
        if not problems:
            problems = [
                "3 + 5", "4 * 6", "x * 0", "x + 0", "x * 1",
                "0 + y", "y * 1", "z * 0", "1 * a", "b + 0",
            ]
        return self._run_solve_batch("code", problems, min_delta=0.1)

    def _run_symbolic_math(self) -> dict:
        """Up to 40 problems from benchmarks/algebra/symbolic_math.json."""
        if not _SYMBOLIC_MATH.exists():
            return {"category": "symbolic_math", "total": 0, "passed": 0,
                    "score": 0.0, "skipped": True, "error": "symbolic_math.json not found"}
        try:
            entries = json.loads(_SYMBOLIC_MATH.read_text())
            if isinstance(entries, dict):
                entries = entries.get("cases", [])
            # Sample evenly across difficulty levels
            sorted_entries = sorted(entries, key=lambda e: e.get("difficulty", 1))
            step = max(1, len(sorted_entries) // 40)
            sampled = sorted_entries[::step][:40]
            problems = [e["expression"] for e in sampled if e.get("expression")]
            return self._run_solve_batch("symbolic_math", problems, min_delta=0.1)
        except Exception as e:
            return {"category": "symbolic_math", "total": 0, "passed": 0,
                    "score": 0.0, "error": str(e)}

    def _run_arc(self) -> dict:
        """ARC-Lite abstract reasoning via ARCRunner."""
        try:
            from sare.benchmarks.arc_runner import ARCRunner
            runner = ARCRunner()
            r = runner.run()
            return {
                "category": "arc",
                "total":  r.get("total", 0),
                "passed": r.get("passed", r.get("total", 0) * r.get("accuracy", 0)),
                "score":  round(r.get("accuracy", 0.0), 3),
                "by_type": r.get("by_type", {}),
            }
        except Exception as e:
            return {"category": "arc", "total": 0, "passed": 0,
                    "score": 0.0, "error": str(e), "skipped": True}

    def _run_synthesis(self) -> dict:
        """Check LLM Transform Synthesizer health: promoted count, inventiveness, readiness."""
        total = 5  # 5 health checks
        passed = 0
        details = []

        # Check 1: synthesizer module loads
        try:
            from sare.learning.llm_transform_synthesizer import get_llm_synthesizer
            synth = get_llm_synthesizer()
            passed += 1
            details.append({"check": "module_loads", "passed": True})
        except Exception as e:
            details.append({"check": "module_loads", "passed": False, "error": str(e)})
            synth = None

        # Check 2: synthesized transforms file exists
        synth_dir = _REPO_ROOT / "data" / "memory" / "synthesized_modules"
        has_synths = synth_dir.exists() and len(list(synth_dir.glob("*.py"))) > 0
        if has_synths:
            passed += 1
        details.append({"check": "synthesized_files_exist", "passed": has_synths,
                        "count": len(list(synth_dir.glob("*.py"))) if synth_dir.exists() else 0})

        # Check 3: synth_attempts.json has entries
        attempts_path = _REPO_ROOT / "data" / "memory" / "synth_attempts.json"
        has_attempts = False
        attempt_count = 0
        if attempts_path.exists():
            try:
                attempts = json.loads(attempts_path.read_text())
                attempt_count = len(attempts) if isinstance(attempts, list) else 0
                has_attempts = attempt_count > 0
            except Exception:
                pass
        if has_attempts:
            passed += 1
        details.append({"check": "attempts_recorded", "passed": has_attempts,
                        "count": attempt_count})

        # Check 4: at least one promoted synthesis
        promoted_count = 0
        if attempts_path.exists():
            try:
                attempts = json.loads(attempts_path.read_text())
                if isinstance(attempts, list):
                    promoted_count = sum(1 for a in attempts if a.get("status") == "promoted")
            except Exception:
                pass
        if promoted_count > 0:
            passed += 1
        details.append({"check": "has_promoted_transform", "passed": promoted_count > 0,
                        "count": promoted_count})

        # Check 5: inventiveness score available on at least one attempt
        has_inventiveness = False
        if attempts_path.exists():
            try:
                attempts = json.loads(attempts_path.read_text())
                if isinstance(attempts, list):
                    has_inventiveness = any("inventiveness" in a for a in attempts)
            except Exception:
                pass
        if has_inventiveness:
            passed += 1
        details.append({"check": "inventiveness_scored", "passed": has_inventiveness})

        return {
            "category": "synthesis",
            "total": total,
            "passed": passed,
            "score": round(passed / total, 3),
            "details": details,
        }

    # ── Engine helpers ───────────────────────────────────────────────────────

    def _run_solve_batch(self, category: str, expressions: list,
                         min_delta: float = 0.5) -> dict:
        if not getattr(self, "_engine_ok", False):
            return {"category": category, "total": len(expressions),
                    "passed": 0, "score": 0.0, "error": "sare.engine not available"}

        expressions = [e for e in expressions if e]
        total   = len(expressions)
        passed  = 0
        details = []

        for expr in expressions:
            try:
                _, graph = self._load_problem(expr)
                result   = self._searcher.search(
                    graph, self._energy_fn, self._transforms,
                    beam_width=6, max_depth=15, budget_seconds=3.0,
                )
                initial = self._energy_fn.compute(graph).total
                final   = result.energy.total
                delta   = initial - final
                ok = delta >= min_delta or len(result.transforms_applied) > 0
                if ok:
                    passed += 1
                details.append({"expr": expr, "delta": round(delta, 3), "passed": ok})
            except Exception as e:
                details.append({"expr": expr, "error": str(e), "passed": False})

        score = round(passed / total, 3) if total > 0 else 0.0
        return {"category": category, "total": total, "passed": passed, "score": score}

    # ── History persistence ──────────────────────────────────────────────────

    def _save_history(self, result: dict) -> None:
        """Append this run to benchmark_history.json for trend analysis."""
        try:
            import datetime
            entry = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "version": result["benchmark_version"],
                "total_score": result["total_score"],
                "total_passed": result["total_passed"],
                "total_problems": result["total_problems"],
                "by_category": {
                    c["category"]: {
                        "passed": c.get("passed", 0),
                        "total":  c.get("total", 0),
                        "score":  c.get("score", 0.0),
                    }
                    for c in result["categories"]
                },
            }
            history = []
            if _BENCHMARK_HIST.exists():
                try:
                    history = json.loads(_BENCHMARK_HIST.read_text())
                    if not isinstance(history, list):
                        history = []
                except Exception:
                    history = []
            history.append(entry)
            # Keep last 500 entries
            history = history[-500:]
            tmp = _BENCHMARK_HIST.parent / f"{_BENCHMARK_HIST.stem}.{os.getpid()}.{_thr.get_ident()}.tmp"
            tmp.write_text(json.dumps(history, indent=2))
            os.replace(tmp, _BENCHMARK_HIST)
        except Exception:
            pass
