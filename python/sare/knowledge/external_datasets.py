"""
ExternalDatasetIngester — Loads curated + downloaded problem sets into SARE-HX.

Datasets:
  curated_math   — 50 handcrafted algebra/logic/calculus/physics problems
  curated_logic  — 10 logic puzzle chains
  gsm8k          — Grade-school math (downloaded from GitHub on first run)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "external_datasets"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

_CURATED_MATH: List[tuple] = [
    # (expression, domain, description)
    ("a + 0",               "algebra",    "additive identity"),
    ("a * 1",               "algebra",    "multiplicative identity"),
    ("a * 0",               "algebra",    "zero product"),
    ("a - a",               "algebra",    "self subtraction"),
    ("a / a",               "algebra",    "self division"),
    ("a ^ 1",               "algebra",    "power one"),
    ("a ^ 0",               "algebra",    "power zero"),
    ("2*x + 3*x",           "algebra",    "combine like terms"),
    ("1*x + 0",             "algebra",    "mul identity + add zero"),
    ("1*x + 0 + 1*y + 0",   "algebra",    "multi identity chain"),
    ("x*x - 1*x*1",         "algebra",    "mul identity chain"),
    ("2*(3*x + 0) + 1*y - y","algebra",   "multi-step simplification"),
    ("x^2 + 2*x + 1",       "algebra",    "perfect square trinomial"),
    ("(a+b)*(a-b)",         "algebra",    "difference of squares"),
    ("a*b + a*c",           "algebra",    "factor distributive"),
    ("3*x + 0 + 2*x + 0",   "algebra",    "combine + zero elim"),
    ("x^0 + b*1 + c*0",     "algebra",    "three identity chain"),
    ("x/1 + y*1 - z*0",     "algebra",    "div+mul identity+zero"),
    ("NOT NOT A",           "logic",      "double negation"),
    ("NOT NOT NOT NOT A",   "logic",      "quad double negation"),
    ("A AND TRUE",          "logic",      "and identity"),
    ("A OR FALSE",          "logic",      "or identity"),
    ("A AND FALSE",         "logic",      "and annihilation"),
    ("A OR TRUE",           "logic",      "or annihilation"),
    ("A AND A",             "logic",      "and idempotent"),
    ("A AND TRUE AND TRUE", "logic",      "chained and identity"),
    ("A OR FALSE OR FALSE",  "logic",     "chained or identity"),
    ("NOT (NOT (A AND TRUE))","logic",    "nested double neg + identity"),
    ("A AND (TRUE AND TRUE)","logic",     "nested and identity"),
    ("d/dx(x^2 + x + 1)",   "calculus",   "sum rule + power rule"),
    ("d/dx(3*x^3)",         "calculus",   "constant factor + power rule"),
    ("d/dx(x^1 + x^0)",     "calculus",   "power rule + zero"),
    ("d/dx(sin(x) + cos(x))","calculus",  "sin+cos derivative"),
    ("d/dx(x^3 + 0)",       "calculus",   "power rule + zero elim"),
    ("d/dx(1*x^2)",         "calculus",   "identity + power rule"),
    ("integral(x^2 + x)",   "calculus",   "sum rule integration"),
    ("F = 1*m*a",           "physics",    "newton + identity"),
    ("V = 1*I*R",           "physics",    "ohms law + identity"),
    ("KE = 0.5*1*v^2",      "physics",    "kinetic energy + identity"),
    ("F = m*0",             "physics",    "zero acceleration"),
    ("PE = m*g*0",          "physics",    "potential energy zero height"),
    ("v = v0 + 0*t",        "physics",    "kinematics zero accel"),
    ("1*F = 1*m*1*a",       "physics",    "newton with identities"),
    ("sin^2(x) + cos^2(x)", "trig",       "pythagorean identity"),
    ("1 + tan^2(x)",        "trig",       "trig identity"),
    ("A AND A AND A",       "logic",      "idempotent chain"),
    ("(A AND TRUE) OR FALSE","logic",     "identity chain"),
    ("A OR (A AND FALSE)",  "logic",      "absorption variant"),
    ("NOT NOT (A AND TRUE)","logic",      "double neg + identity"),
    ("2*x + 4 = 10",        "equations",  "linear equation"),
    ("x + x + x",           "algebra",    "triple combine"),
]

_CURATED_LOGIC_PUZZLES: List[tuple] = [
    ("NOT NOT A AND TRUE",  "logic", "double neg + identity"),
    ("(A OR FALSE) AND (B OR FALSE)", "logic", "double or identity"),
    ("NOT (A AND FALSE)",   "logic", "not annihilator"),
    ("NOT (A OR TRUE)",     "logic", "not true"),
    ("(A AND A) OR FALSE",  "logic", "idempotent + or identity"),
    ("NOT NOT NOT A",       "logic", "triple negation"),
    ("A AND TRUE AND B AND TRUE", "logic", "multi and identity"),
    ("(NOT NOT A) AND (NOT NOT B)", "logic", "double neg both sides"),
    ("A OR (FALSE OR FALSE)", "logic", "nested or identity"),
    ("(A AND TRUE) AND (B AND TRUE)", "logic", "nested and identity x2"),
]


class ExternalDatasetIngester:
    def __init__(self):
        self._problems: List[dict] = []
        self._loaded:   Dict[str, int] = {}

    def load_curated(self) -> int:
        count = 0
        for expr, domain, desc in _CURATED_MATH:
            self._problems.append({"expression": expr, "domain": domain,
                                   "description": desc, "source": "curated_math",
                                   "difficulty": self._est_difficulty(expr)})
            count += 1
        for expr, domain, desc in _CURATED_LOGIC_PUZZLES:
            self._problems.append({"expression": expr, "domain": domain,
                                   "description": desc, "source": "curated_logic",
                                   "difficulty": 0.55})
            count += 1
        self._loaded["curated"] = count
        log.info("ExternalDatasets: %d curated problems loaded", count)
        return count

    def load_gsm8k_sample(self, max_problems: int = 100) -> int:
        cache = _CACHE_DIR / "gsm8k_test.jsonl"
        if not cache.exists():
            try:
                log.info("Downloading GSM8K sample...")
                req = urllib.request.Request(_GSM8K_URL,
                                             headers={"User-Agent": "SARE-HX/1.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    cache.write_bytes(r.read())
                log.info("GSM8K downloaded (%d bytes)", cache.stat().st_size)
            except Exception as e:
                log.warning("GSM8K download failed: %s", e)
                return 0

        count = 0
        try:
            for line in cache.read_text(encoding="utf-8").splitlines()[:max_problems]:
                try:
                    item = json.loads(line)
                    q = item.get("question", "")
                    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", q)
                    if len(nums) >= 2:
                        expr = " + ".join(nums[:3])
                        self._problems.append({"expression": expr, "domain": "arithmetic",
                                               "description": q[:80], "source": "gsm8k",
                                               "difficulty": 0.4})
                        count += 1
                except Exception:
                    continue
        except Exception as e:
            log.warning("GSM8K parse error: %s", e)

        self._loaded["gsm8k"] = count
        log.info("ExternalDatasets: %d GSM8K problems loaded", count)
        return count

    def inject_into_curriculum(self, curriculum_gen) -> int:
        injected = 0
        for prob in self._problems:
            try:
                from sare.engine import load_problem
                _, g = load_problem(prob["expression"])
                curriculum_gen.add_seed(g)
                injected += 1
            except Exception:
                pass
        log.info("ExternalDatasets: injected %d/%d problems", injected, len(self._problems))
        return injected

    def summary(self) -> dict:
        domains: Dict[str, int] = {}
        for p in self._problems:
            domains[p["domain"]] = domains.get(p["domain"], 0) + 1
        return {"total": len(self._problems), "by_source": self._loaded, "by_domain": domains}

    @staticmethod
    def _est_difficulty(expr: str) -> float:
        score = expr.count("(") + expr.count("^") + len(expr.split()) // 4
        return min(0.9, max(0.2, score * 0.1))


_instance: Optional[ExternalDatasetIngester] = None

def get_dataset_ingester() -> ExternalDatasetIngester:
    global _instance
    if _instance is None:
        _instance = ExternalDatasetIngester()
    return _instance
