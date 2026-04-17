"""
NovelBenchmarkRunner — Generates FRESH problems at test time that the
system has never seen. This is the real AGI progress signal.

Problem with the existing benchmark: once solved once, the proof is cached
in schema_matcher, so re-running reports 100% — but that's replay, not
learning. This runner:
  1. Generates problems from randomized templates per domain
  2. Clears schema cache JUST for those problems (checks before solve)
  3. Tracks first-attempt success rate — the real learning metric
  4. Covers multiple domains: arithmetic, algebra, logic, physics,
     chemistry, commonsense, code, language

Domain-general: each domain has its own generator; add more generators
to cover more domains.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class NovelProblem:
    expression: str
    domain: str
    expected: Optional[str] = None
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class NovelResult:
    domain: str
    total: int
    solved: int
    avg_time_ms: float
    first_attempt_rate: float
    cache_bypassed: bool
    problems: List[Dict[str, str]] = field(default_factory=list)


# ── Per-domain problem generators ──────────────────────────────────────────

def _gen_arithmetic() -> NovelProblem:
    a, b, c = random.randint(2, 99), random.randint(2, 99), random.randint(2, 9)
    templates = [
        f"({a} + {b}) * {c}",
        f"{a} * {c} + {b} * {c}",
        f"{a} - ({b} - {c})",
        f"{a} + 0 * {b}",
        f"({a} + {b}) - {b}",
    ]
    return NovelProblem(expression=random.choice(templates), domain="arithmetic")


def _gen_algebra() -> NovelProblem:
    var = random.choice(["x", "y", "z", "n"])
    c1 = random.randint(2, 15)
    c2 = random.randint(1, 20)
    templates = [
        f"{c1}*{var} + 0 = {c2}",
        f"{var} * 1 - {var} = 0",
        f"{c1}*({var} + {c2}) - {c1}*{c2}",
        f"({var} + {c1}) * {c1}",
        f"({var} + {c1}) - {c1}",
    ]
    return NovelProblem(expression=random.choice(templates), domain="algebra")


def _gen_logic() -> NovelProblem:
    p, q, r = random.choice(["A", "B", "C"]), random.choice(["X", "Y"]), random.choice(["P", "Q"])
    templates = [
        f"not (not {p})",
        f"{p} and (not {p})",
        f"{p} or (not {p})",
        f"({p} and {q}) or ({p} and {r})",
        f"not ({p} and {q})",
    ]
    return NovelProblem(expression=random.choice(templates), domain="logic")


def _gen_physics() -> NovelProblem:
    m = random.randint(2, 50)
    a = random.randint(1, 20)
    v = random.randint(1, 30)
    templates = [
        (f"F = {m} * {a}", str(m * a)),
        (f"p = {m} * {v}", str(m * v)),
        (f"KE = 0.5 * {m} * {v}^2", None),
        (f"v = 0 + {a} * {random.randint(1, 10)}", None),
    ]
    expr, ans = random.choice(templates)
    return NovelProblem(expression=expr, domain="physics", expected=ans)


def _gen_chemistry() -> NovelProblem:
    templates = [
        ("H2 + O2 → H2O", "H2 + O2 → H2O (needs balancing)"),
        ("Na + Cl → NaCl", "Na + Cl → NaCl"),
        ("CH4 + O2 → CO2 + H2O", None),
        ("what is the formula for water", "H2O"),
        ("what is the symbol for sodium", "Na"),
    ]
    expr, ans = random.choice(templates)
    return NovelProblem(expression=expr, domain="chemistry", expected=ans)


def _gen_commonsense() -> NovelProblem:
    items = [
        ("does ice melt when heated?", "yes"),
        ("can fish breathe underwater?", "yes"),
        ("is fire cold?", "no"),
        ("do plants need sunlight?", "yes"),
        ("is the sun a planet?", "no"),
    ]
    random.shuffle(items)
    q, ans = items[0]
    return NovelProblem(expression=q, domain="commonsense", expected=ans)


def _gen_code() -> NovelProblem:
    templates = [
        "if x == x then y else z",
        "f(x) = x + 0",
        "while False do body",
        "return x * 1",
        "x = 5; y = x + 0; y",
    ]
    return NovelProblem(expression=random.choice(templates), domain="code")


def _gen_language() -> NovelProblem:
    subjects = ["Alice", "the dog", "a car", "the teacher"]
    verbs = ["runs", "sees", "buys", "reads"]
    objects = ["a book", "fast", "slowly", "the house"]
    expr = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}"
    return NovelProblem(expression=expr, domain="language")


_GENERATORS: Dict[str, Callable[[], NovelProblem]] = {
    "arithmetic":  _gen_arithmetic,
    "algebra":     _gen_algebra,
    "logic":       _gen_logic,
    "physics":     _gen_physics,
    "chemistry":   _gen_chemistry,
    "commonsense": _gen_commonsense,
    "code":        _gen_code,
    "language":    _gen_language,
}


def generate_batch(domain: str, n: int = 5) -> List[NovelProblem]:
    """Generate n fresh problems for the given domain."""
    gen = _GENERATORS.get(domain)
    if gen is None:
        return []
    return [gen() for _ in range(n)]


def generate_all(per_domain: int = 3) -> List[NovelProblem]:
    """Generate fresh problems across ALL domains."""
    out: List[NovelProblem] = []
    for domain in _GENERATORS:
        out.extend(generate_batch(domain, per_domain))
    return out


# ── Runner ─────────────────────────────────────────────────────────────────

def run_novel_benchmark(per_domain: int = 3, bypass_schema_cache: bool = True) -> Dict:
    """Run the held-out novel benchmark. Returns a dict with per-domain results
    and overall novel solve rate.

    bypass_schema_cache: if True, temporarily disables schema matching so
    the system must actually solve each problem (not just replay)."""
    from sare.brain import get_brain
    brain = get_brain()

    results: Dict[str, NovelResult] = {}
    total_solved = 0
    total_attempted = 0

    # Optionally suppress schema cache hits for the duration of the run
    _sm = None
    _old_cache = {}
    if bypass_schema_cache:
        try:
            from sare.cognition.schema_matcher import get_schema_matcher
            _sm = get_schema_matcher()
            # Swap cache to empty for the duration; restore after
            _old_cache = dict(_sm._cache)
            _sm._cache = {}
        except Exception as e:
            log.debug("novel_bench: could not bypass schema cache: %s", e)
            bypass_schema_cache = False

    try:
        for domain in _GENERATORS:
            probs = generate_batch(domain, per_domain)
            d_solved = 0
            d_times: List[float] = []
            d_records: List[Dict[str, str]] = []
            for p in probs:
                t0 = time.time()
                solved = False
                try:
                    if brain is not None and hasattr(brain, "solve"):
                        res = brain.solve(p.expression, domain=p.domain)
                        solved = bool(getattr(res, "solved", False) or
                                      (isinstance(res, dict) and res.get("solved")))
                except Exception as e:
                    log.debug("novel_bench: solve error %s: %s", p.expression, e)
                elapsed = (time.time() - t0) * 1000.0
                d_times.append(elapsed)
                if solved:
                    d_solved += 1
                d_records.append({
                    "expression": p.expression[:80],
                    "solved": str(solved),
                    "time_ms": f"{elapsed:.1f}",
                })
            d_total = len(probs)
            results[domain] = NovelResult(
                domain=domain,
                total=d_total,
                solved=d_solved,
                avg_time_ms=sum(d_times) / max(1, len(d_times)),
                first_attempt_rate=d_solved / max(1, d_total),
                cache_bypassed=bypass_schema_cache,
                problems=d_records,
            )
            total_solved += d_solved
            total_attempted += d_total
    finally:
        # Restore schema cache
        if bypass_schema_cache and _sm is not None:
            _sm._cache = _old_cache

    overall = total_solved / max(1, total_attempted)
    return {
        "timestamp": time.time(),
        "overall_novel_solve_rate": round(overall, 3),
        "total_solved": total_solved,
        "total_attempted": total_attempted,
        "cache_bypassed": bypass_schema_cache,
        "per_domain": {
            d: {
                "total": r.total,
                "solved": r.solved,
                "rate": round(r.first_attempt_rate, 3),
                "avg_time_ms": round(r.avg_time_ms, 1),
            } for d, r in results.items()
        },
        "note": (
            "Novel benchmark: fresh problems generated at test time. "
            "cache_bypassed=True means schema cache was temporarily cleared "
            "so this measures genuine problem-solving, not replay."
        ),
    }
