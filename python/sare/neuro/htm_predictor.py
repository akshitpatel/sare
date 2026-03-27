"""
HTM-Inspired Transform Sequence Predictor

A lightweight Hierarchical Temporal Memory (HTM)-inspired module that learns
transform co-occurrence patterns from solve sequences and predicts which
transform is likely to succeed next.

Unlike everything else in SARE which is reactive (record → update stats),
this is PROACTIVE: it builds a forward model of transform sequences,
enabling the system to predict useful transforms before trying them.

Key difference from WorldModel.predict_transform():
  WorldModel: "what transform works for domain X?" (domain-level)
  HTM:        "given I just applied transforms [A, B], what comes next?" (sequence-level)

No LLM, no external data. Pure n-gram + sparse coincidence counting.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_HTM_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "htm_predictor.json"

# Context window for n-gram sequences
_CONTEXT_LEN = 3   # "given last 3 transforms, predict next"
_MIN_EVIDENCE = 2  # minimum co-occurrences before trusting a prediction
_TOP_K = 5         # return top-K predictions


@dataclass
class TransformPrediction:
    """A ranked prediction for the next useful transform."""
    transform_name: str
    score: float          # 0.0–1.0 confidence
    source: str           # "bigram" | "trigram" | "domain_prior"
    context: List[str]    # the sequence that led to this prediction

    def to_dict(self) -> dict:
        return {
            "transform_name": self.transform_name,
            "score": round(self.score, 4),
            "source": self.source,
            "context": self.context,
        }


class HTMPredictor:
    """
    Sparse n-gram co-occurrence predictor for transform sequences.

    Stores:
      bigram_counts[domain][prev_t][next_t]   = count
      trigram_counts[domain][t1,t2][next_t]   = count
      domain_counts[domain][transform]        = total uses

    Predictions blend bigram + trigram evidence, falling back to
    domain-level frequency when context is sparse.
    """

    def __init__(self):
        # {domain: {prev_transform: {next_transform: count}}}
        self._bigrams:  Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        # {domain: {(t1,t2): {next_transform: count}}}
        self._trigrams: Dict[str, Dict[Tuple, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        # {domain: {transform: count}}
        self._domain_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Total sequences observed
        self._total_sequences = 0
        self._total_predictions = 0
        self._correct_predictions = 0
        # Recent sequences for online learning (ring buffer)
        self._recent: deque = deque(maxlen=500)
        self._update_count = 0
        self.load()

    # ── Learning ──────────────────────────────────────────────────────────────

    def observe_sequence(self, transforms: List[str], domain: str, success: bool = True) -> None:
        """
        Record a completed transform sequence. Called after every solve.

        Args:
            transforms: ordered list of transform names applied
            domain: problem domain (algebra, logic, etc.)
            success: whether the sequence solved the problem
        """
        if not transforms:
            return

        weight = 2 if success else 1  # successful sequences count double

        # Domain-level counts
        for t in transforms:
            self._domain_counts[domain][t] += weight

        # Bigrams: each consecutive pair
        for i in range(len(transforms) - 1):
            t1, t2 = transforms[i], transforms[i + 1]
            self._bigrams[domain][t1][t2] += weight

        # Trigrams: each consecutive triple
        for i in range(len(transforms) - 2):
            t1, t2, t3 = transforms[i], transforms[i + 1], transforms[i + 2]
            self._trigrams[domain][(t1, t2)][t3] += weight

        self._total_sequences += 1
        self._recent.append((domain, transforms, success))

        self._update_count += 1
        if self._update_count % 100 == 0:
            self.save()

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_next(
        self,
        recent_transforms: List[str],
        domain: str,
        available_transforms: Optional[List[str]] = None,
        top_k: int = _TOP_K,
    ) -> List[TransformPrediction]:
        """
        Given a partial sequence, predict the most useful next transform.

        Args:
            recent_transforms: transforms applied so far this solve (most recent last)
            domain: current domain
            available_transforms: restrict predictions to this set (optional)
            top_k: how many predictions to return

        Returns:
            Ranked list of TransformPrediction (best first).
        """
        candidates: Dict[str, float] = defaultdict(float)

        # Source 1: Trigram (highest priority — most specific context)
        if len(recent_transforms) >= 2:
            ctx = (recent_transforms[-2], recent_transforms[-1])
            tri_counts = self._trigrams.get(domain, {}).get(ctx, {})
            total_tri = sum(tri_counts.values())
            if total_tri >= _MIN_EVIDENCE:
                for t, c in tri_counts.items():
                    candidates[t] += (c / total_tri) * 0.6  # weight 0.6

        # Source 2: Bigram
        if recent_transforms:
            prev = recent_transforms[-1]
            bi_counts = self._bigrams.get(domain, {}).get(prev, {})
            total_bi = sum(bi_counts.values())
            if total_bi >= _MIN_EVIDENCE:
                for t, c in bi_counts.items():
                    candidates[t] += (c / total_bi) * 0.3  # weight 0.3

        # Source 3: Domain prior (unigram frequency)
        dom_counts = self._domain_counts.get(domain, {})
        total_dom = sum(dom_counts.values())
        if total_dom > 0:
            for t, c in dom_counts.items():
                candidates[t] += (c / total_dom) * 0.1   # weight 0.1

        # Filter to available transforms if provided
        if available_transforms:
            avail_set = set(available_transforms)
            candidates = {t: s for t, s in candidates.items() if t in avail_set}

        # Remove transforms already in recent sequence (avoid immediate repeats
        # unless they're the only option)
        recent_set = set(recent_transforms[-2:]) if len(recent_transforms) >= 2 else set()
        filtered = {t: s for t, s in candidates.items() if t not in recent_set}
        if not filtered:
            filtered = candidates  # fallback: allow repeats if nothing else

        # Sort and build result
        ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for t, score in ranked:
            if score < 0.01:
                break
            # Determine source label
            ctx2 = tuple(recent_transforms[-2:]) if len(recent_transforms) >= 2 else ()
            if ctx2 in self._trigrams.get(domain, {}):
                source = "trigram"
            elif recent_transforms and recent_transforms[-1] in self._bigrams.get(domain, {}):
                source = "bigram"
            else:
                source = "domain_prior"
            results.append(TransformPrediction(
                transform_name=t,
                score=min(1.0, score),
                source=source,
                context=list(recent_transforms[-_CONTEXT_LEN:]),
            ))

        return results

    def rerank_transforms(
        self,
        transforms: list,
        recent_applied: List[str],
        domain: str,
    ) -> list:
        """
        Re-rank a list of Transform objects using HTM predictions.
        Transforms with high prediction scores bubble to the front.

        Returns: reordered transforms list (same objects, different order).
        """
        if not transforms:
            return transforms

        avail = [t.name() if hasattr(t, "name") else str(t) for t in transforms]
        preds = self.predict_next(recent_applied, domain, avail, top_k=len(transforms))

        if not preds:
            return transforms

        pred_score = {p.transform_name: p.score for p in preds}

        # Stable sort: HTM score as primary, original index as tiebreak
        indexed = list(enumerate(transforms))
        indexed.sort(
            key=lambda pair: (
                -pred_score.get(
                    pair[1].name() if hasattr(pair[1], "name") else str(pair[1]), 0.0
                ),
                pair[0],
            )
        )
        return [t for _, t in indexed]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def record_outcome(self, predicted_name: str, actual_name: str) -> None:
        """Track prediction accuracy."""
        self._total_predictions += 1
        if predicted_name == actual_name:
            self._correct_predictions += 1

    def get_stats(self) -> dict:
        total_bigrams = sum(
            sum(v.values())
            for dom in self._bigrams.values()
            for v in dom.values()
        )
        total_trigrams = sum(
            sum(v.values())
            for dom in self._trigrams.values()
            for v in dom.values()
        )
        accuracy = (
            self._correct_predictions / self._total_predictions
            if self._total_predictions > 0 else 0.0
        )
        return {
            "total_sequences": self._total_sequences,
            "total_bigrams": total_bigrams,
            "total_trigrams": total_trigrams,
            "domains": list(self._domain_counts.keys()),
            "prediction_accuracy": round(accuracy, 4),
            "total_predictions": self._total_predictions,
        }

    def top_sequences(self, domain: str, n: int = 5) -> List[dict]:
        """Return the most common bigram transitions for a domain."""
        bi = self._bigrams.get(domain, {})
        pairs = []
        for t1, nexts in bi.items():
            for t2, cnt in nexts.items():
                pairs.append((t1, t2, cnt))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return [{"from": t1, "to": t2, "count": cnt} for t1, t2, cnt in pairs[:n]]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        try:
            _HTM_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "bigrams": {
                    dom: {t1: dict(nexts) for t1, nexts in bi.items()}
                    for dom, bi in self._bigrams.items()
                },
                "trigrams": {
                    dom: {f"{k[0]}|||{k[1]}": dict(nexts) for k, nexts in tri.items()}
                    for dom, tri in self._trigrams.items()
                },
                "domain_counts": {dom: dict(counts) for dom, counts in self._domain_counts.items()},
                "total_sequences": self._total_sequences,
                "total_predictions": self._total_predictions,
                "correct_predictions": self._correct_predictions,
                "saved_at": time.time(),
            }
            tmp = _HTM_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, _HTM_PATH)
        except Exception as e:
            log.debug("[HTM] save failed: %s", e)

    def load(self) -> None:
        try:
            if not _HTM_PATH.exists():
                return
            data = json.loads(_HTM_PATH.read_text())
            for dom, bi in data.get("bigrams", {}).items():
                for t1, nexts in bi.items():
                    for t2, cnt in nexts.items():
                        self._bigrams[dom][t1][t2] = cnt
            for dom, tri in data.get("trigrams", {}).items():
                for key_str, nexts in tri.items():
                    parts = key_str.split("|||")
                    if len(parts) == 2:
                        for t3, cnt in nexts.items():
                            self._trigrams[dom][(parts[0], parts[1])][t3] = cnt
            for dom, counts in data.get("domain_counts", {}).items():
                self._domain_counts[dom].update(counts)
            self._total_sequences   = data.get("total_sequences", 0)
            self._total_predictions = data.get("total_predictions", 0)
            self._correct_predictions = data.get("correct_predictions", 0)
            log.info("[HTM] Loaded: %d sequences, %d domains",
                     self._total_sequences, len(self._domain_counts))
        except Exception as e:
            log.debug("[HTM] load failed: %s", e)


# ── Singleton ─────────────────────────────────────────────────────────────────

_SINGLETON: Optional[HTMPredictor] = None


def get_htm_predictor() -> HTMPredictor:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = HTMPredictor()
    return _SINGLETON
