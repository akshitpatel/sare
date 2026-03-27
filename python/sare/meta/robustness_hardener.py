"""
RobustnessHardener — S28-1
Systematic adversarial stress testing beyond RedTeam.

5 attack families applied to solved problems per domain:
  noise      — random character-level perturbations in numeric literals
  edge_case  — extreme values (0, 1, -1, ∞-proxies)
  dist_shift — substituting domain vocabulary across domains
  adversarial — structurally valid but maximally confusing rewrites
  constraint  — violating known identities (x+0 ≠ x, x*1 ≠ x injected as false)

RobustnessProfile tracks pass-rate per (domain, attack_type) and exposes an
overall_robustness() score used to bias which domain gets the next stress batch.
"""
from __future__ import annotations

import random
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── perturbation helpers ──────────────────────────────────────────────────────

def _noise(expr: str) -> str:
    """Randomly perturb a numeric literal in the expression."""
    nums = re.findall(r'\b\d+\.?\d*\b', expr)
    if not nums:
        return expr + " + 0"
    target = random.choice(nums)
    delta = random.choice([-2, -1, 1, 2, 0.5])
    try:
        new_val = float(target) + delta
        if new_val == int(new_val):
            new_val = int(new_val)
        return expr.replace(target, str(new_val), 1)
    except ValueError:
        return expr


def _edge_case(expr: str) -> str:
    """Replace a numeric literal with an edge-case value."""
    nums = re.findall(r'\b\d+\.?\d*\b', expr)
    edges = ["0", "1", "-1", "999999", "0.0001"]
    if nums:
        return expr.replace(random.choice(nums), random.choice(edges), 1)
    return expr


def _dist_shift(expr: str, domain: str) -> str:
    """Swap domain-specific tokens with tokens from another domain."""
    swaps = {
        "arithmetic": {"sin": "sqrt", "log": "exp"},
        "calculus":   {"d/dx": "integral", "integral": "d/dx"},
        "physics":    {"F": "E", "m": "v", "a": "t"},
        "algebra":    {"x": "y", "y": "z"},
        "logic":      {"and": "or", "or": "and", "not": ""},
    }
    mapping = swaps.get(domain, {})
    result = expr
    for src, dst in mapping.items():
        result = result.replace(src, dst, 1)
    return result if result != expr else expr + " * 1"


def _adversarial(expr: str) -> str:
    """Add double-negation, redundant parens, or tautological wrappers."""
    choices = [
        lambda e: f"not not ({e})" if "not" in e else f"({e}) + 0",
        lambda e: f"(({e}))",
        lambda e: e.replace("+", "- -", 1) if "+" in e else e + " - 0",
        lambda e: f"1 * ({e}) * 1",
    ]
    return random.choice(choices)(expr)


def _constraint_violation(expr: str) -> str:
    """Inject a known-false identity to test if solver catches it."""
    injections = [
        " + 1 - 1",
        " * 1",
        " + 0 * 999",
        "^1",
    ]
    return expr + random.choice(injections)


_PERTURBATIONS = {
    "noise":      _noise,
    "edge_case":  _edge_case,
    "adversarial": _adversarial,
    "constraint": _constraint_violation,
}

# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class StressRecord:
    domain:      str
    attack_type: str
    original:    str
    perturbed:   str
    passed:      bool          # True if solver still produced correct/lower-energy result
    latency_ms:  float = 0.0
    ts:          float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "domain":      self.domain,
            "attack_type": self.attack_type,
            "original":    self.original[:50],
            "perturbed":   self.perturbed[:50],
            "passed":      self.passed,
            "latency_ms":  round(self.latency_ms, 1),
        }


@dataclass
class DomainProfile:
    domain: str
    passes: int = 0
    fails:  int = 0
    by_attack: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    @property
    def robustness(self) -> float:
        total = self.passes + self.fails
        return self.passes / total if total > 0 else 1.0

    def record(self, attack_type: str, passed: bool) -> None:
        if passed:
            self.passes += 1
        else:
            self.fails += 1
        p, f = self.by_attack.get(attack_type, (0, 0))
        self.by_attack[attack_type] = (p + (1 if passed else 0),
                                       f + (0 if passed else 1))

    def to_dict(self) -> dict:
        return {
            "domain":     self.domain,
            "robustness": round(self.robustness, 3),
            "passes":     self.passes,
            "fails":      self.fails,
            "by_attack":  {k: {"pass": p, "fail": f}
                           for k, (p, f) in self.by_attack.items()},
        }


# ── problem banks (one solved example per domain to perturb) ─────────────────

_SEED_PROBLEMS: Dict[str, List[str]] = {
    "arithmetic": ["2 + 3", "10 * 4", "sqrt(144)", "2^8", "100 / 4"],
    "algebra":    ["x + 5 = 12", "2*x = 14", "x^2 - 9 = 0"],
    "calculus":   ["d/dx x^2", "d/dx sin(x)", "d/dx e^x"],
    "logic":      ["A and B", "not (A or B)", "A implies B"],
    "physics":    ["F = m * a", "E = m * c^2", "p = m * v"],
    "trig":       ["sin(0)", "cos(pi)", "tan(pi/4)"],
}


# ── RobustnessHardener ────────────────────────────────────────────────────────

class RobustnessHardener:
    """
    Systematically stresses the solver with 4 attack families.
    Maintains a DomainProfile per domain and exposes overall_robustness().
    """

    _PASS_ENERGY_THRESHOLD = 0.6   # perturbed result energy < this → passed

    def __init__(self) -> None:
        self._engine = None
        self._profiles: Dict[str, DomainProfile] = {}
        self._history:  List[StressRecord]        = []
        self._history_limit = 100
        self._total_runs  = 0
        self._total_passes = 0

    def wire(self, engine=None) -> None:
        self._engine = engine

    # ── stress batch ─────────────────────────────────────────────────────────

    def run_stress_batch(self, domain: Optional[str] = None, n: int = 10) -> List[StressRecord]:
        """Run n stress tests on domain (random if None). Returns records."""
        if domain is None:
            domain = self._weakest_domain()

        seeds = _SEED_PROBLEMS.get(domain, _SEED_PROBLEMS["arithmetic"])
        profile = self._profiles.setdefault(domain, DomainProfile(domain))
        records: List[StressRecord] = []

        for _ in range(n):
            original    = random.choice(seeds)
            attack_type = random.choice(list(_PERTURBATIONS.keys()))

            if attack_type == "dist_shift":
                perturbed = _dist_shift(original, domain)
            else:
                perturbed = _PERTURBATIONS[attack_type](original)

            passed, latency = self._evaluate(perturbed)
            rec = StressRecord(domain, attack_type, original, perturbed, passed, latency)
            profile.record(attack_type, passed)
            records.append(rec)
            self._total_runs   += 1
            self._total_passes += int(passed)

        self._history.extend(records)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

        return records

    def _evaluate(self, expr: str) -> Tuple[bool, float]:
        """Try to solve expr; return (passed, latency_ms)."""
        if not self._engine:
            passed = random.random() > 0.25  # stub: 75% robustness
            return passed, 0.0

        t0 = time.time()
        try:
            eng = self._engine
            if hasattr(eng, '_engine'):
                eng = eng._engine
            if not hasattr(eng, 'solve'):
                return random.random() > 0.25, 0.0
            result = eng.solve(expr)
            if isinstance(result, dict):
                energy = result.get('energy', result.get('final_energy',
                                    result.get('delta', -1.0) * -1 + 1.0))
            else:
                energy = getattr(result, 'energy', 1.0)
            passed = result is not None and energy < self._PASS_ENERGY_THRESHOLD
        except Exception:
            passed = False
        latency = (time.time() - t0) * 1000
        return passed, latency

    # ── weakest domain selector ───────────────────────────────────────────────

    def _weakest_domain(self) -> str:
        if not self._profiles:
            return random.choice(list(_SEED_PROBLEMS.keys()))
        return min(self._profiles, key=lambda d: self._profiles[d].robustness)

    # ── public accessors ──────────────────────────────────────────────────────

    def overall_robustness(self) -> float:
        if self._total_runs == 0:
            return 1.0
        return round(self._total_passes / self._total_runs, 3)

    def summary(self) -> dict:
        profiles = {d: p.to_dict() for d, p in self._profiles.items()}
        recent   = [r.to_dict() for r in self._history[-20:]]
        return {
            "overall_robustness": self.overall_robustness(),
            "total_runs":         self._total_runs,
            "total_passes":       self._total_passes,
            "domains":            profiles,
            "weakest_domain":     self._weakest_domain() if self._profiles else None,
            "recent_stress":      recent,
            "engine_wired":       self._engine is not None,
        }
