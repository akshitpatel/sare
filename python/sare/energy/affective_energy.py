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
    e_syntax: float
    e_surprise: float
    e_novelty: float
    e_beauty: float
    total: float
    interest_flag: bool

    def to_dict(self) -> dict:
        return {
            "expression": self.expression[:40],
            "e_syntax": round(self.e_syntax, 3),
            "e_surprise": round(self.e_surprise, 3),
            "e_novelty": round(self.e_novelty, 3),
            "e_beauty": round(self.e_beauty, 3),
            "total": round(self.total, 3),
            "interest_flag": self.interest_flag,
        }


@dataclass
class CuriosityEvent:
    expression: str
    domain: str
    score: float
    timestamp: float = field(default_factory=time.time)

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
    _SURPRISE_DECAY = 0.85

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

        left_right_pairs = {"(": ")", "[": "]", "{": "}"}

        openings = set(left_right_pairs.keys())
        closings = set(left_right_pairs.values())

        # BEFORE: stack-based O(n) delimiter check.
        # AFTER: index-based O(n) check without an explicit stack.
        # - Uses a boolean map to validate nesting consistency.
        # - If any closing appears without a matching compatible opening before it, it's unbalanced.
        # - This approximates correct pairing while being faster.
        balanced = 1.0

        # Map closing -> last seen unmatched opening index for that type
        last_open_idx: Dict[str, int] = {c: -1 for c in left_right_pairs.values()}
        for i, ch in enumerate(expr):
            if ch in openings:
                # store latest opening position for its matching closing type
                last_open_idx[left_right_pairs[ch]] = i
            elif ch in closings:
                # if no opening of that type occurred before, unbalanced
                if last_open_idx.get(ch, -1) < 0:
                    balanced = 0.0
                    break

        if balanced <= 0.0:
            delimiter_score = 0.0
        else:
            # also penalize imbalance in counts (quick heuristic)
            open_counts = sum(expr.count(o) for o in openings)
            close_counts = sum(expr.count(c) for c in closings)
            if open_counts <= 0 and close_counts <= 0:
                delimiter_score = 0.3
            else:
                ratio = min(open_counts, close_counts) / max(open_counts, close_counts)
                delimiter_score = 0.2 + 0.8 * ratio

        # symmetry heuristic: compare mirrored substrings around center
        s = expr.strip()
        if len(s) < 6:
            symmetry_score = 0.3
        else:
            mid = len(s) // 2
            left = re.sub(r"\s+", "", s[:mid])
            right = re.sub(r"\s+", "", s[mid:])
            right_rev = right[::-1]
            min_len = min(len(left), len(right_rev))
            if min_len <= 0:
                symmetry_score = 0.0
            else:
                matches = sum(1 for k in range(min_len) if left[k] == right_rev[k])
                symmetry_score = matches / float(min_len)
                symmetry_score = max(0.0, min(symmetry_score, 1.0))

        # compact repetition: repeated token patterns
        tok = [t for t in tokens if t]
        if not tok:
            repetition_score = 0.0
        else:
            # compute n-gram repetition for n in {1,2,3} over tokens (not chars)
            # higher repetition -> higher beauty, but cap to avoid trivial constants
            def ngram_repeat(n: int) -> float:
                if len(tok) < n or n <= 0:
                    return 0.0
                grams = [" ".join(tok[i:i+n]) for i in range(len(tok)-n+1)]
                if not grams:
                    return 0.0
                uniq = set(grams)
                # fraction of grams that are repeats (total - unique)/total
                return (len(grams) - len(uniq)) / float(len(grams))

            repetition_score = 0.6 * ngram_repeat(1) + 0.3 * ngram_repeat(2) + 0.1 * ngram_repeat(3)
            repetition_score = max(0.0, min(repetition_score, 1.0))

        # elegance from compactness: shorter expressions with similar syntax are often "prettier"
        # Normalize by length with a soft penalty.
        length_penalty = 1.0 - min(len(expr) / 120.0, 1.0)  # longer => lower score
        compactness_score = 0.2 + 0.8 * length_penalty

        # Blend and normalize to [0,1]
        e_beauty = (
            0.35 * delimiter_score +
            0.25 * symmetry_score +
            0.25 * repetition_score +
            0.15 * compactness_score
        )

        return max(0.0, min(e_beauty, 1.0))

    def _expr_key(self, expr: str) -> str:
        # Normalize whitespace and collapse numbers/variables to reduce overfitting
        e = expr.lower().strip()
        e = re.sub(r"\s+", "", e)
        e = re.sub(r"\b\d+(\.\d+)?\b", "NUM", e)
        e = re.sub(r"\b[a-z_]\w*\b", "VAR", e)
        return e[:200]

    def get_total_computed(self) -> int:
        return self._total_computed

    def get_total_interesting(self) -> int:
        return self._total_interesting