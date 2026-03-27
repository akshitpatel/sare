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
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Dict, List

log = logging.getLogger(__name__)


@dataclass
class AffectiveScore:
    expression: str
    e_syntax:   float
    e_surprise: float
    e_novelty:  float
    e_beauty:   float
    total:      float
    interest_flag: bool

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
        return {
            "expression": self.expression[:40],
            "domain": self.domain,
            "score": round(self.score, 3),
            "timestamp": self.timestamp,
        }


class AffectiveEnergy:
    """
    Maintains surprise history per expression-type and concept centroids
    per domain so it can score any new expression across four affective axes.
    """

    _INTEREST_THRESHOLD = 0.55
    _SURPRISE_DECAY     = 0.85

    def __init__(self) -> None:
        self._surprise_history: Dict[str, float] = {}
        self._concept_centroids: Dict[str, set] = {}
        self._curiosity_events: List[CuriosityEvent] = []
        self._total_computed = 0
        self._total_interesting = 0
        self._domain_curiosity: Dict[str, float] = {}

    def calibrate_from_concepts(self, concept_graph) -> None:
        """Seed concept centroids from the concept graph."""
        try:
            for name, c in list(getattr(concept_graph, "_concepts", {}).items()):
                domain = getattr(c, "domain", "general")
                tokens = set(re.findall(r"\w+", name.lower()))
                for rule in getattr(c, "symbolic_rules", [])[:5]:
                    tokens |= set(re.findall(r"\w+", str(rule).lower()))
                if domain not in self._concept_centroids:
                    self._concept_centroids[domain] = set()
                self._concept_centroids[domain] |= tokens
        except Exception as e:
            log.debug(f"AffectiveEnergy calibrate: {e}")

    def compute(self, expression: str, domain: str = "general",
                prediction_error: float = 0.0) -> AffectiveScore:
        """Compute affective energy for an expression."""
        self._total_computed += 1
        tokens = re.findall(r"\w+", expression.lower())

        e_syntax = self._syntax_energy(expression)
        e_surprise = self._surprise_energy(expression, prediction_error)
        e_novelty = self._novelty_energy(tokens, domain)
        e_beauty = self._beauty_energy(expression, tokens)

        total = (0.3 * e_syntax + 0.3 * e_surprise +
                 0.2 * e_novelty + 0.2 * e_beauty)
        flag = total >= self._INTEREST_THRESHOLD

        if flag:
            self._total_interesting += 1
            ev = CuriosityEvent(expression, domain, total)
            self._curiosity_events.append(ev)
            if len(self._curiosity_events) > 500:
                self._curiosity_events = self._curiosity_events[-500:]
            prev = self._domain_curiosity.get(domain, 0.0)
            self._domain_curiosity[domain] = 0.8 * prev + 0.2 * total

        return AffectiveScore(
            expression, e_syntax, e_surprise, e_novelty, e_beauty, total, flag
        )

    def register_solve(self, expression: str, prediction_error: float) -> None:
        """Update surprise EMA after a solve."""
        key = self._expr_key(expression)
        prev = self._surprise_history.get(key, prediction_error)
        self._surprise_history[key] = (
            self._SURPRISE_DECAY * prev +
            (1 - self._SURPRISE_DECAY) * prediction_error
        )

    def get_curiosity_bias(self, top_n: int = 5) -> List[dict]:
        """Return top domains to explore, sorted by average interest."""
        ranked = sorted(
            self._domain_curiosity.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [{"domain": d, "avg_interest": round(v, 3)} for d, v in ranked[:top_n]]

    def _syntax_energy(self, expr: str) -> float:
        """Normalised token complexity [0,1]."""
        n = len(expr)
        ops = len(re.findall(r"[+\-*/^=]", expr))
        depth = expr.count("(")
        raw = (n / 80.0) + (ops / 10.0) + (depth / 5.0)
        return min(raw / 3.0, 1.0)

    def _surprise_energy(self, expr: str, prediction_error: float) -> float:
        """How much this expression type has surprised us historically."""
        key = self._expr_key(expr)
        hist = self._surprise_history.get(key, prediction_error)
        return min(hist * 1.5, 1.0)

    def _novelty_energy(self, tokens: List[str], domain: str) -> float:
        """Jaccard distance from domain centroid."""
        centroid = self._concept_centroids.get(domain)
        token_set = set(tokens)
        if not token_set:
            return 0.0
        if not centroid:
            return 1.0
        union = token_set | centroid
        if not union:
            return 0.0
        inter = token_set & centroid
        return 1.0 - (len(inter) / float(len(union)))

    def _beauty_energy(self, expr: str, tokens: List[str]) -> float:
        """
        Structural elegance heuristic:
        rewards symmetry, compact repetition, and balanced delimiters.
        """
        if not expr:
            return 0.0

        score = 0.0

        if expr.count("(") == expr.count(")"):
            score += 0.2

        if "=" in expr:
            left, right = expr.split("=", 1)
            ll = len(left.strip())
            rl = len(right.strip())
            if max(ll, rl) > 0:
                symmetry = 1.0 - abs(ll - rl) / max(ll, rl)
                score += 0.3 * max(0.0, symmetry)

        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        repetition_compactness = 1.0 - min(unique_ratio, 1.0)
        score += 0.2 * repetition_compactness

        if re.search(r"(\b\w+\b).*\1", expr.lower()):
            score += 0.1

        if re.search(r"\b(\w+)\s+\1\b", expr.lower()):
            score += 0.1

        if any(op in expr for op in ["+", "-", "*", "/", "^", "="]):
            score += 0.1

        return min(max(score, 0.0), 1.0)

    @staticmethod
    @lru_cache(maxsize=2048)
    def _expr_key(expression: str) -> str:
        expr = expression.lower().strip()
        expr = re.sub(r"\d+", "#", expr)
        expr = re.sub(r"\s+", " ", expr)
        expr = re.sub(r"[^\w\s+\-*/^=()]", "", expr)
        return expr

    def stats(self) -> dict:
        interesting_rate = (
            self._total_interesting / self._total_computed
            if self._total_computed else 0.0
        )
        return {
            "total_computed": self._total_computed,
            "total_interesting": self._total_interesting,
            "interesting_rate": round(interesting_rate, 3),
            "tracked_domains": len(self._domain_curiosity),
            "surprise_keys": len(self._surprise_history),
            "curiosity_events": len(self._curiosity_events),
        }