"""
PredictiveEngine — Friston Free Energy Principle for SARE-HX
=============================================================

The brain doesn't react — it predicts, then learns from prediction error.

Central idea (Friston free energy / active inference):
  Action selection = minimise EXPECTED FREE ENERGY (EFE):

    EFE(T) = epistemic_value(T) + pragmatic_value(T)

    epistemic  = info-gain from trying T (high if T rarely tried in domain)
    pragmatic  = expected energy reduction (from world_model.predict_transform())

Stage-dependent weighting:
  TODDLER (stage 1): weight epistemic >> pragmatic → genuine exploration
  TEENAGER (stage 4): balanced
  RESEARCHER (stage 7): weight pragmatic >> epistemic → efficient exploitation

Usage::
    pe = get_predictive_engine()
    ranked = pe.select_action(graph, transforms, world_model, stage_level)
    # ... run search ...
    error = pe.observe_outcome(prediction, actual_transform, actual_delta, world_model)
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_MEMORY = Path(__file__).resolve().parents[4] / "data" / "memory"


@dataclass
class PredictionError:
    """Outcome of a single prediction cycle."""
    predicted_transform: str
    actual_transform:    str
    predicted_delta:     float
    actual_delta:        float
    surprise:            float     # |predicted_delta - actual_delta|
    domain:              str
    timestamp:           float = field(default_factory=time.time)

    @property
    def was_correct(self) -> bool:
        return self.predicted_transform == self.actual_transform

    def to_dict(self) -> dict:
        return {
            "predicted_transform": self.predicted_transform,
            "actual_transform":    self.actual_transform,
            "predicted_delta":     round(self.predicted_delta, 4),
            "actual_delta":        round(self.actual_delta, 4),
            "surprise":            round(self.surprise, 4),
            "was_correct":         self.was_correct,
            "domain":              self.domain,
            "timestamp":           self.timestamp,
        }


class PredictiveEngine:
    """
    Implements active inference (Friston free energy) for transform selection.

    Before each solve: selects transforms by EFE = epistemic + pragmatic value.
    After each solve:  computes prediction error, updates world model confidence.

    Stats are stored per domain to allow domain-specific exploration rates.
    """

    PERSIST_PATH = _MEMORY / "predictive_engine.json"

    # Stage → (epistemic_weight, pragmatic_weight)
    # Stage 0 (INFANT): pure exploration; Stage 7 (RESEARCHER): pure exploitation
    _STAGE_WEIGHTS = {
        0: (0.90, 0.10),   # INFANT
        1: (0.80, 0.20),   # TODDLER
        2: (0.65, 0.35),   # CHILD
        3: (0.50, 0.50),   # PRETEEN
        4: (0.35, 0.65),   # TEENAGER
        5: (0.20, 0.80),   # UNDERGRADUATE
        6: (0.10, 0.90),   # GRADUATE
        7: (0.05, 0.95),   # RESEARCHER
    }

    def __init__(self):
        # Per-domain, per-transform try counts (for epistemic value)
        self._try_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Recent prediction history
        self._history: deque = deque(maxlen=500)
        # Per-domain average surprise (tracks how unpredictable each domain is)
        self._domain_surprise: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._total_predictions: int = 0
        self._correct_predictions: int = 0
        self._load()

    def select_action(
        self,
        graph,
        transforms: list,
        world_model=None,
        stage_level: int = 0,
        domain: str = "general",
    ) -> List[Tuple]:
        """
        Rank transforms by Expected Free Energy (EFE).

        Returns list of (transform, efe_score) sorted descending.
        """
        if not transforms:
            return []

        e_w, p_w = self._STAGE_WEIGHTS.get(stage_level, (0.5, 0.5))

        # Get pragmatic values from world model
        wm_predictions: Dict[str, float] = {}
        if world_model is not None and hasattr(world_model, "predict_transform"):
            try:
                pred = world_model.predict_transform(graph, [
                    (t.name() if callable(getattr(t, "name", None)) else
                     getattr(t, "name", t.__class__.__name__))
                    for t in transforms
                ])
                if pred is not None:
                    pname = getattr(pred, "transform_name", "")
                    pdelta = float(getattr(pred, "expected_delta", 0.0))
                    if pname:
                        wm_predictions[pname] = pdelta
            except Exception:
                pass

        scored: List[Tuple] = []
        domain_counts = self._try_counts.get(domain, {})
        max_count = max(domain_counts.values(), default=1)

        for t in transforms:
            name = ""
            try:
                name = t.name() if callable(getattr(t, "name", None)) else str(getattr(t, "name", t.__class__.__name__))
            except Exception:
                name = t.__class__.__name__

            # Epistemic value: inverse of how often we've tried this transform
            tries = domain_counts.get(name, 0)
            # Normalise: 0 tries → epistemic=1.0, many tries → epistemic→0
            epistemic = 1.0 / (1.0 + math.log1p(tries / max(max_count, 1)))

            # Pragmatic value: expected energy reduction from world model
            if wm_predictions:
                pragmatic = wm_predictions.get(name, 0.1)
                # Normalise to [0, 1] (clamp)
                pragmatic = max(0.0, min(1.0, pragmatic / 10.0))
            else:
                # Fall back: slight bias toward transforms tried more often
                # (exploit known-good transforms when no WM data)
                pragmatic = min(1.0, tries / max(max_count * 2, 1))

            efe = e_w * epistemic + p_w * pragmatic
            scored.append((t, efe))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def observe_outcome(
        self,
        predicted_transform: str,
        actual_transform: str,
        actual_delta: float,
        predicted_delta: float = 0.0,
        domain: str = "general",
        world_model=None,
    ) -> PredictionError:
        """
        Record the outcome of a solve attempt. Update try counts and surprise.
        Calls world_model.update_from_prediction_error() if surprise is high.
        """
        # Update try counts
        self._try_counts[domain][actual_transform] += 1

        surprise = abs(predicted_delta - actual_delta)
        error = PredictionError(
            predicted_transform=predicted_transform,
            actual_transform=actual_transform,
            predicted_delta=predicted_delta,
            actual_delta=actual_delta,
            surprise=surprise,
            domain=domain,
        )
        self._history.append(error)
        self._domain_surprise[domain].append(surprise)
        self._total_predictions += 1
        if error.was_correct:
            self._correct_predictions += 1

        # Propagate to world model when surprise is high
        if world_model is not None and surprise > 1.5:
            try:
                if hasattr(world_model, "update_from_prediction_error"):
                    world_model.update_from_prediction_error(error)
            except Exception:
                pass

        log.debug(
            "[PredictiveEngine] pred=%s actual=%s Δ_pred=%.2f Δ_actual=%.2f surprise=%.2f",
            predicted_transform, actual_transform, predicted_delta, actual_delta, surprise,
        )
        return error

    def get_epistemic_value(self, transform_name: str, domain: str) -> float:
        """How much information would we gain by trying this transform?"""
        tries = self._try_counts.get(domain, {}).get(transform_name, 0)
        return 1.0 / (1.0 + math.log1p(tries))

    def get_avg_surprise(self, domain: str) -> float:
        """Average surprise level for a domain (how unpredictable is it?)."""
        surprises = list(self._domain_surprise.get(domain, []))
        if not surprises:
            return 0.0
        return sum(surprises) / len(surprises)

    def get_status(self) -> dict:
        accuracy = (self._correct_predictions / max(self._total_predictions, 1))
        domain_surprises = {
            d: round(self.get_avg_surprise(d), 3)
            for d in self._domain_surprise
        }
        return {
            "total_predictions":   self._total_predictions,
            "accuracy":            round(accuracy, 3),
            "tracked_domains":     len(self._try_counts),
            "domain_avg_surprise": domain_surprises,
            "recent_errors":       [e.to_dict() for e in list(self._history)[-5:]],
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        try:
            _MEMORY.mkdir(parents=True, exist_ok=True)
            data = {
                "try_counts": {d: dict(v) for d, v in self._try_counts.items()},
                "total_predictions": self._total_predictions,
                "correct_predictions": self._correct_predictions,
                "saved_at": time.time(),
            }
            tmp = self.PERSIST_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.PERSIST_PATH)
        except OSError as e:
            log.debug("[PredictiveEngine] Save error: %s", e)

    def _load(self):
        if not self.PERSIST_PATH.exists():
            return
        try:
            d = json.loads(self.PERSIST_PATH.read_text())
            for dom, counts in d.get("try_counts", {}).items():
                for t, c in counts.items():
                    self._try_counts[dom][t] = int(c)
            self._total_predictions = int(d.get("total_predictions", 0))
            self._correct_predictions = int(d.get("correct_predictions", 0))
        except Exception as e:
            log.debug("[PredictiveEngine] Load error: %s", e)

    def save(self):
        self._save()


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: Optional[PredictiveEngine] = None


def get_predictive_engine() -> PredictiveEngine:
    global _instance
    if _instance is None:
        _instance = PredictiveEngine()
    return _instance
