"""
AlgorithmInventor — SARE invents new search and reasoning algorithms
====================================================================
When beam search consistently fails on a problem class, the system
doesn't just ask for new transforms — it invents a new SEARCH ALGORITHM.

Human analogy: when standard approaches fail, mathematicians invent
new proof techniques (induction, contradiction, diagonalization).

Invented algorithms implement the Searcher interface:
    class MyAlgorithm:
        def solve(self, graph, transforms, energy_fn,
                  beam_width, budget_seconds) -> SearchResult

Current beam search limitations (targets for invention):
  - Stuck in local minima → needs: simulated annealing / MCTS
  - Too many transforms → needs: pattern-matching shortcut
  - Cyclic rewrites → needs: visited-set with fingerprinting
  - Long chains needed → needs: bidirectional search

Validation: invented algorithm must outperform BeamSearch on at least
one problem class from recent failure history.

Usage::
    ai = get_algorithm_inventor()
    result = ai.invent(
        failure_patterns=["cyclic_rewrite", "local_minimum"],
        domain="algebra",
    )
    print(result["algorithm_name"])  # e.g. "SimulatedAnnealingSearch"
    print(result["promoted"])
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_ROOT   = Path(__file__).resolve().parents[4]
_MEMORY = _ROOT / "data" / "memory"
_SYNTH  = _MEMORY / "synthesized_modules"
_ALG_REGISTRY = _MEMORY / "invented_algorithms.json"


@dataclass
class InventedAlgorithm:
    name:           str
    description:    str
    targets_pattern: str      # what failure mode it fixes
    domain:         str
    code_path:      str
    promoted:       bool      = False
    validation_score: float   = 0.0
    times_selected: int       = 0
    avg_improvement: float    = 0.0
    created_at:     float     = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class AlgorithmInventor:
    """
    Invents new search algorithms when existing ones consistently fail.
    Implements the meta-learning loop: learn HOW to search, not just WHAT to search.
    """

    COOLDOWN_SECONDS = 1800    # 30 minutes between inventions
    IMPROVEMENT_THRESHOLD = 0.1  # must beat BeamSearch by 10%+

    def __init__(self):
        self._algorithms:  List[InventedAlgorithm] = []
        self._cooldowns:   Dict[str, float] = {}
        self._load()

    # ── Main API ─────────────────────────────────────────────────────────────

    def invent(
        self,
        failure_patterns: List[str],
        domain:           str,
        failed_exprs:     Optional[List[str]] = None,
        force:            bool = False,
    ) -> dict:
        """
        Invent a new search algorithm targeting the given failure patterns.
        """
        from sare.interface.llm_bridge import _call_llm, llm_available
        if not llm_available():
            return {"promoted": False, "message": "LLM unavailable"}

        key = f"{domain}:{','.join(sorted(failure_patterns))}"
        if not force:
            last = self._cooldowns.get(key, 0)
            if time.time() - last < self.COOLDOWN_SECONDS:
                return {"promoted": False, "message": "cooldown"}
        self._cooldowns[key] = time.time()

        patterns_str = ", ".join(failure_patterns) if failure_patterns else "general stuck"
        exprs_str = "\n".join(f"  - {e}" for e in (failed_exprs or [])[:6])
        log.info("[AlgorithmInventor] Inventing for domain=%s patterns=%s",
                 domain, patterns_str)

        # Step 1: Design the algorithm
        design = self._design_algorithm(patterns_str, domain, exprs_str, _call_llm)
        if not design:
            return {"promoted": False, "message": "Design step failed"}

        alg_name    = self._extract_name(design)
        description = self._extract_description(design)
        log.info("[AlgorithmInventor] Designing: %s", alg_name)

        # Step 2: Implement it
        code = self._implement_algorithm(alg_name, design, domain, _call_llm)
        if not code:
            return {"promoted": False, "message": "Implementation step failed",
                    "algorithm_name": alg_name}

        from sare.neuro.symbol_creator import _strip_fences, _is_safe
        code = _strip_fences(code)
        safe, reason = _is_safe(code)
        if not safe:
            return {"promoted": False, "message": f"Safety: {reason}",
                    "algorithm_name": alg_name}

        # Step 3: Import test
        ok, err = self._test_import(alg_name, code)
        if not ok:
            return {"promoted": False, "message": f"Import fail: {err}",
                    "algorithm_name": alg_name}

        # Step 4: Benchmark against BeamSearch
        improvement, details = self._benchmark(alg_name, code, domain, failed_exprs or [])

        # Step 5: Save
        path = _SYNTH / f"{alg_name}.py"
        path.write_text(code, encoding="utf-8")

        alg = InventedAlgorithm(
            name=alg_name,
            description=description,
            targets_pattern=patterns_str,
            domain=domain,
            code_path=str(path),
            promoted=(improvement >= self.IMPROVEMENT_THRESHOLD),
            validation_score=improvement,
        )
        self._algorithms.append(alg)
        self._save()

        if alg.promoted:
            log.info("[AlgorithmInventor] ✓ Algorithm '%s' promoted (+%.1f%% vs BeamSearch)",
                     alg_name, improvement * 100)
            # Dopamine burst
            try:
                from sare.neuro.dopamine import get_dopamine_system
                get_dopamine_system().receive_reward(
                    "algorithm_invented", domain=domain, delta=improvement * 20
                )
            except Exception:
                pass

        return {
            "promoted":        alg.promoted,
            "algorithm_name":  alg_name,
            "description":     description,
            "improvement":     improvement,
            "benchmark":       details,
            "message":         "promoted" if alg.promoted else "saved_not_promoted",
        }

    def load_best_algorithm(self, domain: str = "general"):
        """
        Load and return the best promoted algorithm for this domain.
        Returns None if no suitable algorithm found.
        """
        candidates = [
            a for a in self._algorithms
            if a.promoted and (a.domain == domain or a.domain == "general")
        ]
        if not candidates:
            return None
        # Pick by avg_improvement or validation_score
        best = max(candidates, key=lambda a: a.validation_score + a.avg_improvement * 0.3)
        try:
            tname = f"_sare_alg_{best.name}"
            if tname in sys.modules:
                mod = sys.modules[tname]
            else:
                spec = importlib.util.spec_from_file_location(tname, best.code_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.modules[tname] = mod
            cls = getattr(mod, best.name, None)
            if cls:
                best.times_selected += 1
                return cls()
        except Exception as e:
            log.debug("[AlgorithmInventor] Load error %s: %s", best.name, e)
        return None

    def record_performance(self, alg_name: str, improvement: float):
        """Record that an algorithm achieved `improvement` over baseline."""
        for a in self._algorithms:
            if a.name == alg_name:
                n = a.times_selected or 1
                a.avg_improvement = (a.avg_improvement * (n-1) + improvement) / n
        self._save()

    def get_status(self) -> dict:
        return {
            "total_invented": len(self._algorithms),
            "promoted":       sum(1 for a in self._algorithms if a.promoted),
            "recent":         [a.to_dict() for a in self._algorithms[-5:]],
        }

    # ── LLM prompts ──────────────────────────────────────────────────────────

    def _design_algorithm(self, patterns, domain, exprs_str, llm) -> str:
        prompt = (
            "You are designing a search algorithm for an AGI symbolic reasoning system.\n\n"
            f"DOMAIN: {domain}\n"
            f"FAILURE PATTERNS: {patterns}\n"
            f"FAILED EXPRESSIONS:\n{exprs_str if exprs_str else '  (none provided)'}\n\n"
            "The current algorithm (BeamSearch) fails because of these patterns.\n"
            "Design a NEW search algorithm specifically addressing these failures.\n\n"
            "Known search strategies you can draw from:\n"
            "  - Simulated Annealing: allow worse states with decreasing probability\n"
            "  - Monte Carlo Tree Search: UCB1-guided expansion\n"
            "  - Bidirectional Search: search from goal backward\n"
            "  - Pattern Matching Shortcut: fingerprint → known solution\n"
            "  - Iterative Deepening: progressively deeper beam\n"
            "  - Diversity Beam: maintain diverse candidates (no near-duplicates)\n"
            "  - Adaptive Beam: widen beam when stuck, narrow when on track\n"
            "  - Hybrid: combine two of the above\n\n"
            "Format:\n"
            "ALGORITHM_NAME: <PascalCase, no spaces>\n"
            "DESCRIPTION: <one sentence>\n"
            "ADDRESSES: <which failure pattern this fixes and why>\n"
            "APPROACH: <2-3 sentences on how the algorithm works>\n"
            "INTERFACE: <describe the solve(graph, transforms, energy_fn, beam_width, budget_seconds) method>"
        )
        return llm(prompt, use_synthesis_model=True)

    def _implement_algorithm(self, name, design, domain, llm) -> str:
        prompt = (
            f"Implement the following search algorithm for SARE-HX as Python code:\n\n"
            f"DESIGN:\n{design}\n\n"
            f"Your class must implement this EXACT interface:\n\n"
            f"```python\n"
            f"class {name}:\n"
            f"    def solve(self, graph, transforms: list,\n"
            f"              energy_fn, beam_width: int = 8,\n"
            f"              budget_seconds: float = 5.0):\n"
            f"        '''\n"
            f"        Returns an object with:\n"
            f"          .graph           — best graph found\n"
            f"          .energy          — final energy\n"
            f"          .solved          — bool (energy reduced)\n"
            f"          .transforms_applied — List[str] of transform names used\n"
            f"        Returns None if no improvement found.\n"
            f"        '''\n"
            f"        ...\n"
            f"```\n\n"
            "REQUIREMENTS:\n"
            "1. from sare.engine import EnergyEvaluator for energy computation\n"
            "2. Use `transform.apply(graph)` and `transform.name()` for each transform\n"
            "3. Use `energy_fn.compute(graph).total` for energy\n"
            "4. Stay within `budget_seconds` using `time.time()`\n"
            "5. No eval/exec/subprocess/socket imports\n"
            "6. Return a simple namespace/dataclass — NOT the SARE SearchResult class\n\n"
            "Return ONLY the Python code, no markdown:"
        )
        return llm(prompt, use_synthesis_model=True)

    # ── Benchmarking ─────────────────────────────────────────────────────────

    def _benchmark(
        self,
        name: str,
        code: str,
        domain: str,
        failed_exprs: List[str],
    ) -> tuple:
        """
        Compare invented algorithm vs BeamSearch on test problems.
        Returns (improvement_ratio, details_dict).
        """
        try:
            # Load invented algorithm
            test_id = f"_sare_alg_bench_{name}_{int(time.time())}"
            path = _SYNTH / f"_bench_{name}.py"
            path.write_text(code, encoding="utf-8")
            spec = importlib.util.spec_from_file_location(test_id, str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls = getattr(mod, name, None)
            path.unlink(missing_ok=True)
            sys.modules.pop(test_id, None)

            if not cls:
                return 0.0, {"error": "class not found"}

            alg_instance = cls()

            from sare.engine import BeamSearch, EnergyEvaluator, get_transforms, load_problem
            baseline = BeamSearch(beam_width=6, budget_seconds=3.0)
            ev       = EnergyEvaluator()
            tfs      = get_transforms(include_macros=True)

            # Test problems — use failed exprs + standard ones
            test_exprs = list(dict.fromkeys(
                (failed_exprs or [])[:4] +
                ["x + 0", "x * 1", "not not x", "x * 0", "x - x"]
            ))[:8]

            baseline_solves = 0
            alg_solves      = 0

            for expr in test_exprs:
                try:
                    _, g = load_problem(expr)
                    if g is None:
                        continue

                    e0 = ev.compute(g).total

                    # BeamSearch baseline
                    try:
                        r_base = baseline.solve(g, tfs)
                        if r_base and ev.compute(r_base.graph).total < e0:
                            baseline_solves += 1
                    except Exception:
                        pass

                    # Invented algorithm
                    try:
                        r_new = alg_instance.solve(g, tfs, ev, 6, 2.0)
                        if r_new:
                            e_new = (
                                ev.compute(r_new.graph).total
                                if hasattr(r_new, "graph") and r_new.graph is not None
                                else e0
                            )
                            if e_new < e0:
                                alg_solves += 1
                    except Exception:
                        pass
                except Exception:
                    pass

            n = max(len(test_exprs), 1)
            improvement = (alg_solves - baseline_solves) / n
            details = {
                "problems_tested": n,
                "baseline_solves": baseline_solves,
                "alg_solves":      alg_solves,
                "improvement":     round(improvement, 3),
            }
            return improvement, details

        except Exception as e:
            log.debug("[AlgorithmInventor] Benchmark error: %s", e)
            return 0.0, {"error": str(e)}

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _extract_name(self, text: str) -> str:
        import re
        m = re.search(r'ALGORITHM_NAME\s*:\s*([A-Za-z][A-Za-z0-9_]*)', text)
        if m:
            return m.group(1)
        m2 = re.search(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+(?:Search|Solver|Strategy)?)\b', text)
        return m2.group(1) if m2 else "AdaptiveSearch"

    def _extract_description(self, text: str) -> str:
        import re
        m = re.search(r'DESCRIPTION\s*:\s*(.+)', text)
        return m.group(1).strip() if m else ""

    def _test_import(self, name: str, code: str) -> tuple:
        test_id = f"_sare_alg_import_{name}_{int(time.time())}"
        path = _SYNTH / f"_imptest_{name}.py"
        try:
            path.write_text(code, encoding="utf-8")
            spec = importlib.util.spec_from_file_location(test_id, str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return True, ""
        except Exception as e:
            return False, str(e)
        finally:
            path.unlink(missing_ok=True)
            sys.modules.pop(test_id, None)

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self):
        try:
            data = [a.to_dict() for a in self._algorithms[-100:]]
            _ALG_REGISTRY.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as e:
            log.debug("[AlgorithmInventor] Save error: %s", e)

    def _load(self):
        if not _ALG_REGISTRY.exists():
            return
        try:
            data = json.loads(_ALG_REGISTRY.read_text())
            for d in data:
                a = InventedAlgorithm(**{k: v for k, v in d.items()
                                         if k in InventedAlgorithm.__dataclass_fields__})
                self._algorithms.append(a)
        except Exception as e:
            log.debug("[AlgorithmInventor] Load error: %s", e)


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: Optional[AlgorithmInventor] = None

def get_algorithm_inventor() -> AlgorithmInventor:
    global _instance
    if _instance is None:
        _instance = AlgorithmInventor()
    return _instance
