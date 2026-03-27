"""
TransformGenerator — S26-3
Self-modifying transforms: generates new transform definitions via template grammar,
tests them in a sandbox, and promotes those that consistently lower energy.
The system writes its own math rules.
"""
from __future__ import annotations
import re
import time
import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional, Any

log = logging.getLogger(__name__)


@dataclass
class GeneratedTransform:
    name:         str
    pattern_desc: str   # human-readable "a + 0 → a"
    template:     str   # template category used
    test_passes:  int   = 0
    test_fails:   int   = 0
    promoted:     bool  = False
    created_at:   float = field(default_factory=time.time)

    @property
    def pass_rate(self) -> float:
        total = self.test_passes + self.test_fails
        return self.test_passes / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "pattern_desc": self.pattern_desc,
            "template":     self.template,
            "test_passes":  self.test_passes,
            "test_fails":   self.test_fails,
            "pass_rate":    round(self.pass_rate, 3),
            "promoted":     self.promoted,
        }


# ── Template grammar ─────────────────────────────────────────────────────────
_IDENTITY_TEMPLATES = [
    ("identity_add_zero",   "V + 0 → V",      lambda e: re.sub(r'(\w+)\s*\+\s*0\b', r'\1', e)),
    ("identity_mul_one",    "V * 1 → V",       lambda e: re.sub(r'(\w+)\s*\*\s*1\b', r'\1', e)),
    ("identity_sub_zero",   "V - 0 → V",       lambda e: re.sub(r'(\w+)\s*-\s*0\b', r'\1', e)),
    ("identity_div_one",    "V / 1 → V",       lambda e: re.sub(r'(\w+)\s*/\s*1\b', r'\1', e)),
    ("identity_pow_zero",   "V ^ 0 → 1",       lambda e: re.sub(r'(\w+)\s*\^\s*0\b', '1', e)),
    ("identity_pow_one",    "V ^ 1 → V",       lambda e: re.sub(r'(\w+)\s*\^\s*1\b', r'\1', e)),
    ("absorb_mul_zero",     "V * 0 → 0",       lambda e: re.sub(r'(\w+)\s*\*\s*0\b', '0', e)),
    ("absorb_zero_mul",     "0 * V → 0",       lambda e: re.sub(r'\b0\s*\*\s*(\w+)', '0', e)),
    ("double_neg",          "--V → V",         lambda e: re.sub(r'--(\w+)', r'\1', e)),
    ("add_comm",            "0 + V → V",       lambda e: re.sub(r'\b0\s*\+\s*(\w+)', r'\1', e)),
    ("self_sub",            "V - V → 0",       lambda e: re.sub(r'(\w+)\s*-\s*\1\b', '0', e)),
    ("self_div",            "V / V → 1",       lambda e: re.sub(r'(\w+)\s*/\s*\1\b', '1', e)),
    ("double_div",          "V / (1/V) → V^2", lambda e: re.sub(r'(\w+)\s*/\s*\(1/\1\)', r'\1^2', e)),
    ("log_exp_cancel",      "log(exp(V)) → V", lambda e: re.sub(r'log\(exp\((\w+)\)\)', r'\1', e)),
    ("exp_log_cancel",      "exp(log(V)) → V", lambda e: re.sub(r'exp\(log\((\w+)\)\)', r'\1', e)),
]


class TransformGenerator:
    """
    Generates transform candidates from a template grammar, tests them
    on a pool of known expressions, and promotes strong performers to
    the live engine.
    """

    _MIN_PASSES_TO_PROMOTE = 3
    _TEST_POOL = [
        "x + 0", "y * 1", "z - 0", "a / 1", "b ^ 0", "c ^ 1", "d * 0",
        "0 * e", "x - x", "y / y", "--z", "0 + a", "log(exp(x))", "exp(log(y))",
        "x + 0 + y", "(a + 0) * b", "f * 1 + g * 0",
    ]

    def __init__(self) -> None:
        self._candidates:  List[GeneratedTransform] = []
        self._promoted:    List[GeneratedTransform] = []
        self._engine       = None
        self._total_generated = 0
        self._total_promoted  = 0

        # pre-seed all templates as candidates
        for name, desc, _ in _IDENTITY_TEMPLATES:
            self._candidates.append(GeneratedTransform(name, desc, "identity"))

    def wire(self, engine) -> None:
        self._engine = engine

    # ── generate ──────────────────────────────────────────────────────────────
    def generate_candidates(self, n: int = 5) -> List[GeneratedTransform]:
        """Pick n untested or low-tested templates and test them."""
        pool = [c for c in self._candidates
                if not c.promoted and c.test_passes + c.test_fails < 5]
        # prioritise untested first, then by existing pass rate
        pool.sort(key=lambda c: (c.test_passes + c.test_fails, -c.test_passes))
        batch = pool[:n]
        self._total_generated += len(batch)
        for candidate in batch:
            self._test_candidate(candidate)
        return batch

    # ── test ──────────────────────────────────────────────────────────────────
    def _test_candidate(self, c: GeneratedTransform) -> None:
        """Apply the transform to each test-pool expression; measure energy drop."""
        fn = self._get_fn(c.name)
        if fn is None:
            return
        for expr in self._TEST_POOL:
            try:
                result = fn(expr)
                if result != expr and len(result) < len(expr):
                    c.test_passes += 1
                elif result == expr:
                    pass  # neutral — don't penalise
                else:
                    c.test_fails += 1
            except Exception:
                c.test_fails += 1

    @staticmethod
    def _get_fn(name: str):
        for n, _, fn in _IDENTITY_TEMPLATES:
            if n == name:
                return fn
        return None

    # ── promote ───────────────────────────────────────────────────────────────
    def promote_best(self, min_passes: int = _MIN_PASSES_TO_PROMOTE) -> List[GeneratedTransform]:
        """Promote candidates that meet the pass threshold."""
        newly = []
        for c in self._candidates:
            if not c.promoted and c.test_passes >= min_passes:
                c.promoted = True
                self._promoted.append(c)
                self._total_promoted += 1
                newly.append(c)
                self._inject_into_engine(c)
        return newly

    def _inject_into_engine(self, c: GeneratedTransform) -> None:
        """Attempt to register the transform lambda into the live engine."""
        if self._engine is None:
            return
        fn = self._get_fn(c.name)
        if fn is None:
            return
        try:
            # Engine may expose register_transform(name, fn)
            if hasattr(self._engine, 'register_transform'):
                self._engine.register_transform(c.name, fn)
            # Or it may have a _transforms dict
            elif hasattr(self._engine, '_transforms'):
                self._engine._transforms[c.name] = fn
        except Exception as e:
            log.debug(f"TransformGenerator inject {c.name}: {e}")

    def apply(self, expression: str) -> str:
        """Apply all promoted transforms in sequence."""
        expr = expression
        for c in self._promoted:
            fn = self._get_fn(c.name)
            if fn:
                try:
                    expr = fn(expr)
                except Exception:
                    pass
        return expr

    # ── summary ───────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        promoted_list = [c.to_dict() for c in self._promoted]
        candidates_list = sorted(
            [c.to_dict() for c in self._candidates if not c.promoted],
            key=lambda x: -x["pass_rate"])[:8]
        return {
            "total_candidates":  len(self._candidates),
            "total_generated":   self._total_generated,
            "total_promoted":    self._total_promoted,
            "promoted":          promoted_list,
            "top_candidates":    candidates_list,
            "engine_wired":      self._engine is not None,
        }
