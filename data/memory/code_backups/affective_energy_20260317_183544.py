"""
AffectiveEnergy — S26-2
Multi-component energy function: E_syntax + E_surprise + E_novelty + E_beauty.
Gives the system intrinsic curiosity: it actively seeks surprising, beautiful problems.
"""
from __future__ import annotations
import math
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class AffectiveScore:
    expression: str
    e_syntax:   float   # complexity (lower = simpler)
    e_surprise: float   # unexpectedness (higher = more interesting)
    e_novelty:  float   # distance from known (higher = more novel)
    e_beauty:   float   # structural elegance (higher = more beautiful)
    total:      float
    interest_flag: bool  # True if worth prioritising

    def to_dict(self) -> dict:
        return {
            "expression":    self.expression[:40],
            "e_syntax":      round(self.e_syntax, 3),
            "e_surprise":    round(self.e_surprise, 3),
            "e_novelty":     round(self.e_novelty, 3),
            "e_beauty":      round(self.e_beauty, 3),
            "total":         round(self.total, 3),
            "interest_flag": self.interest_flag,
        }


@dataclass
class CuriosityEvent:
    expression: str
    domain:     str
    score:      float
    timestamp:  float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"expression": self.expression[:40], "domain": self.domain,
                "score": round(self.score, 3), "timestamp": self.timestamp}


class AffectiveEnergy:
    """
    Maintains surprise history per expression-type and concept centroids
    per domain so it can score any new expression across four affective axes.
    """

    _INTEREST_THRESHOLD = 0.55
    _SURPRISE_DECAY     = 0.85   # EMA factor

    def __init__(self) -> None:
        self._surprise_history: Dict[str, float]       = {}   # expr_key → running surprise
        self._concept_centroids: Dict[str, set]        = {}   # domain → token set
        self._curiosity_events:  List[CuriosityEvent]  = []
        self._total_computed     = 0
        self._total_interesting  = 0
        self._domain_curiosity:  Dict[str, float]      = {}   # domain → avg interest

    # ── wiring ────────────────────────────────────────────────────────────────
    def calibrate_from_concepts(self, concept_graph) -> None:
        """Seed concept centroids from the concept graph."""
        try:
            for name, c in list(getattr(concept_graph, '_concepts', {}).items()):
                domain = getattr(c, 'domain', 'general')
                tokens = set(re.findall(r'\w+', name.lower()))
                for rule in getattr(c, 'symbolic_rules', [])[:5]:
                    tokens |= set(re.findall(r'\w+', str(rule).lower()))
                if domain not in self._concept_centroids:
                    self._concept_centroids[domain] = set()
                self._concept_centroids[domain] |= tokens
        except Exception as e:
            log.debug(f"AffectiveEnergy calibrate: {e}")

    # ── main API ──────────────────────────────────────────────────────────────
    def compute(self, expression: str, domain: str = "general",
                prediction_error: float = 0.0) -> AffectiveScore:
        """Compute affective energy for an expression."""
        self._total_computed += 1
        tokens = re.findall(r'\w+', expression.lower())

        e_syntax   = self._syntax_energy(expression)
        e_surprise = self._surprise_energy(expression, prediction_error)
        e_novelty  = self._novelty_energy(tokens, domain)
        e_beauty   = self._beauty_energy(expression, tokens)

        # weighted sum — surprise & beauty drive interest
        total = (0.3 * e_syntax + 0.3 * e_surprise +
                 0.2 * e_novelty + 0.2 * e_beauty)
        flag  = total >= self._INTEREST_THRESHOLD

        if flag:
            self._total_interesting += 1
            ev = CuriosityEvent(expression, domain, total)
            self._curiosity_events.append(ev)
            if len(self._curiosity_events) > 500:
                self._curiosity_events = self._curiosity_events[-500:]
            prev = self._domain_curiosity.get(domain, 0.0)
            self._domain_curiosity[domain] = 0.8 * prev + 0.2 * total

        return AffectiveScore(expression, e_syntax, e_surprise,
                              e_novelty, e_beauty, total, flag)

    def register_solve(self, expression: str, prediction_error: float) -> None:
        """Update surprise EMA after a solve."""
        key = self._expr_key(expression)
        prev = self._surprise_history.get(key, prediction_error)
        self._surprise_history[key] = (self._SURPRISE_DECAY * prev +
                                       (1 - self._SURPRISE_DECAY) * prediction_error)

    def get_curiosity_bias(self, top_n: int = 5) -> List[dict]:
        """Return top domains to explore, sorted by average interest."""
        ranked = sorted(self._domain_curiosity.items(),
                        key=lambda x: x[1], reverse=True)
        return [{"domain": d, "avg_interest": round(v, 3)}
                for d, v in ranked[:top_n]]

    # ── component scorers ─────────────────────────────────────────────────────
    def _syntax_energy(self, expr: str) -> float:
        """Normalised token complexity [0,1]."""
        n = len(expr)
        ops = len(re.findall(r'[+\-*/^=]', expr))
        depth = expr.count('(')
        raw = (n / 80.0) + (ops / 10.0) + (depth / 5.0)
        return min(raw / 3.0, 1.0)

    def _surprise_energy(self, expr: str, prediction_error: float) -> float:
        """How much this expression type has surprised us historically."""
        key   = self._expr_key(expr)
        hist  = self._surprise_history.get(key, prediction_error)
        return min(hist * 1.5, 1.0)

    def _novelty_energy(self, tokens: List[str], domain: str) -> float:
        """Jaccard distance from domain centroid."""
        centroid = self._concept_centroids.get(domain,
                   self._concept_centroids.get('general', set()))
        if not centroid or not tokens:
            return 0.5
        tok_set = set(tokens)
        inter   = len(tok_set & centroid)
        union   = len(tok_set | centroid)
        jaccard = inter / union if union else 0
        return 1.0 - jaccard   # high novelty = far from centroid

    def _beauty_energy(self, expr: str, tokens: List[str]) -> float:
        """Structural elegance: symmetry + brevity + balanced parens."""
        score = 0.0
        # Balanced parentheses bonus
        depth = 0; balanced = True
        for c in expr:
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            if depth < 0: balanced = False; break
        if balanced and depth == 0:
            score += 0.3
        # Brevity bonus (short expressions that are also correct)
        if len(tokens) <= 5:
            score += 0.3
        # Repeated-structure bonus (e.g. "a + a", "x * x")
        tok_freq = {}
        for t in tokens:
            tok_freq[t] = tok_freq.get(t, 0) + 1
        repeats = sum(1 for v in tok_freq.values() if v > 1)
        score += min(0.4, repeats * 0.15)
        return min(score, 1.0)

    @staticmethod
    def _expr_key(expr: str) -> str:
        """Abstract key: replace numbers with N, vars with V."""
        k = re.sub(r'\d+(\.\d+)?', 'N', expr.lower())
        k = re.sub(r'\b[a-z]\b', 'V', k)
        return k[:40]

    # ── summary ───────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        recent = [e.to_dict() for e in self._curiosity_events[-8:]]
        return {
            "total_computed":    self._total_computed,
            "total_interesting": self._total_interesting,
            "interest_rate":     round(self._total_interesting /
                                       max(self._total_computed, 1), 3),
            "curiosity_bias":    self.get_curiosity_bias(),
            "domains_calibrated": list(self._concept_centroids.keys()),
            "recent_interesting": recent,
        }
