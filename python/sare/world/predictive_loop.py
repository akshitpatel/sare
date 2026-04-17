"""
PredictiveWorldLoop — Gap 1: Continuous World Interaction

Implements the sensorimotor loop:

    state → predict → act → observe → update
      ↑___________________________________|

This is the core of how real intelligence works:

  1. PREDICT: Given current world state, predict what will happen next
             (e.g., "if I apply add_zero_elim, result will be x")
  2. ACT:     Apply a transform/action to the world
  3. OBSERVE: See what actually happened
  4. ERROR:   Compute prediction error (predicted vs actual)
  5. UPDATE:  Update the world model to reduce future error

Without this loop:
  - Concept formation stays purely symbolic (no grounding in observation)
  - Transforms are applied without understanding WHY they work
  - The brain cannot learn from being wrong

With this loop:
  - Every failure teaches something (prediction error > 0 → update model)
  - Causal understanding deepens: "rule X applies when conditions Y hold"
  - Physics intuitions improve through prediction + correction
  - Symbolic rules get grounded in simulated observation
"""

from __future__ import annotations

import time
import logging
import math
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ── State representation ───────────────────────────────────────────────────────

@dataclass
class WorldState:
    """
    A symbolic snapshot of the world at a point in time.

    Tracks:
      - expression: the current symbolic expression being worked on
      - domain: which domain it belongs to
      - energy: current complexity/energy measure
      - properties: domain-specific properties (e.g., physics: position, velocity)
      - step: which step in the current episode
    """
    expression: str
    domain: str = "general"
    energy: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    step: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "domain": self.domain,
            "energy": round(self.energy, 4),
            "properties": self.properties,
            "step": self.step,
        }


@dataclass
class Prediction:
    """
    What the system predicts will happen after an action.

    Fields:
      - predicted_result: what expression/state we expect
      - predicted_delta: how much energy we expect to reduce
      - predicted_transform: which transform we plan to apply
      - confidence: how confident we are (0-1, based on past accuracy)
    """
    predicted_result: str
    predicted_delta: float
    predicted_transform: str
    confidence: float = 0.5
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "predicted_result": self.predicted_result,
            "predicted_delta": round(self.predicted_delta, 4),
            "predicted_transform": self.predicted_transform,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
        }


@dataclass
class Observation:
    """
    What actually happened after applying the action.

    Fields:
      - actual_result: what expression/state resulted
      - actual_delta: how much energy actually changed
      - transform_used: which transform was actually applied
      - success: whether the action improved things
    """
    actual_result: str
    actual_delta: float
    transform_used: str
    success: bool
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "actual_result": self.actual_result,
            "actual_delta": round(self.actual_delta, 4),
            "transform_used": self.transform_used,
            "success": self.success,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


@dataclass
class PredictionError:
    """
    The discrepancy between prediction and observation.

    High error → model was wrong → important learning signal.
    Low error  → model was right → confirms current understanding.
    """
    expression: str
    predicted_delta: float
    actual_delta: float
    delta_error: float          # |predicted - actual|
    result_match: bool          # predicted_result == actual_result
    transform: str
    magnitude: float            # overall error magnitude (0=perfect, 1=very wrong)

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "predicted_delta": round(self.predicted_delta, 4),
            "actual_delta": round(self.actual_delta, 4),
            "delta_error": round(self.delta_error, 4),
            "result_match": self.result_match,
            "transform": self.transform,
            "magnitude": round(self.magnitude, 3),
        }


# ── Predictive World Loop ──────────────────────────────────────────────────────

class PredictiveWorldLoop:
    """
    Implements the sensorimotor learning loop for SARE-HX.

    State machine:
      IDLE → PREDICTING → ACTING → OBSERVING → UPDATING → IDLE

    Each cycle:
      1. Take a WorldState (expression + domain)
      2. Predict: use transform history to predict best next action + expected result
      3. Act: apply the predicted transform via the engine
      4. Observe: record what actually happened
      5. Update: adjust transform confidence scores based on prediction error

    Over time:
      - High-error transforms get flagged for investigation
      - Low-error transforms get promoted as reliable rules
      - Domain-specific accuracy builds up → grounded symbolic understanding
    """

    def __init__(self):
        self._current_state: Optional[WorldState] = None
        self._last_prediction: Optional[Prediction] = None
        self._last_observation: Optional[Observation] = None
        self._error_log: deque = deque(maxlen=200)
        self._transform_accuracy: Dict[str, List[float]] = {}
        self._domain_accuracy: Dict[str, List[float]] = {}

        # Open gap fix (approved): bounded surprise event buffer to prevent unbounded growth.
        # DreamConsolidator consumes recent surprise events; we only need a rolling window.
        self._surprise_events: deque = deque(maxlen=5000)

        # basic episode bookkeeping
        self._cycle_count: int = 0
        self._update_count: int = 0
        self._last_error_magnitude: float = 0.0
        self._last_surprise_event: Optional[dict] = None

    def predict(self, state: WorldState, candidate_transforms: List[str]) -> Prediction:
        """
        Predict the result of applying a chosen transform.
        In this implementation we approximate:
          - predicted_delta based on historical transform accuracy
          - predicted_result as 'unknown' placeholder (actual engine will determine)
        """
        self._current_state = state
        self._cycle_count += 1

        # Select transform with highest estimated confidence
        best_transform = candidate_transforms[0] if candidate_transforms else "unknown_transform"
        best_conf = -1.0

        for t in candidate_transforms:
            acc = self._transform_confidence(t)
            if acc > best_conf:
                best_conf = acc
                best_transform = t

        # Predicted delta: use confidence to bias towards reducing energy
        # If confident, predict more improvement; otherwise predict small change.
        predicted_delta = (0.15 + 0.75 * best_conf) * (-1.0)  # energy should decrease -> negative delta
        predicted_result = "PREDICTED_UNKNOWN"

        self._last_prediction = Prediction(
            predicted_result=predicted_result,
            predicted_delta=predicted_delta,
            predicted_transform=best_transform,
            confidence=best_conf,
            reasoning="Estimated from historical transform accuracy.",
        )
        return self._last_prediction

    def act(self, engine: Any) -> Observation:
        """
        Apply the predicted transform via the engine.

        The engine is expected to expose:
          apply_transform(transform_name, expression, domain) -> (new_expression, energy_delta)
        If it doesn't, this method will raise an AttributeError.
        """
        if self._current_state is None or self._last_prediction is None:
            raise RuntimeError("Predict must be called before act.")

        start = time.time()

        transform_name = self._last_prediction.predicted_transform
        expression = self._current_state.expression
        domain = self._current_state.domain

        # Engine contract is not part of this file's public interface; keep defensive.
        new_expr, energy_delta = engine.apply_transform(transform_name, expression, domain)

        success = energy_delta < 0  # improvement typically means energy decreased
        elapsed_ms = (time.time() - start) * 1000.0

        self._last_observation = Observation(
            actual_result=str(new_expr),
            actual_delta=float(energy_delta),
            transform_used=str(transform_name),
            success=bool(success),
            elapsed_ms=elapsed_ms,
        )
        return self._last_observation

    def update(self, world_model: Any = None, induction_callback: Optional[Any] = None) -> PredictionError:
        """
        Update internal statistics based on prediction error.
        Optionally record into a world model / induction callback.
        """
        if self._current_state is None or self._last_prediction is None or self._last_observation is None:
            raise RuntimeError("Predict and act must be called before update.")

        pred = self._last_prediction
        obs = self._last_observation

        delta_error = abs(pred.predicted_delta - obs.actual_delta)
        result_match = (pred.predicted_result == obs.actual_result)
        magnitude = self._compute_error_magnitude(delta_error, pred.confidence, result_match)

        err = PredictionError(
            expression=self._current_state.expression,
            predicted_delta=pred.predicted_delta,
            actual_delta=obs.actual_delta,
            delta_error=delta_error,
            result_match=result_match,
            transform=obs.transform_used,
            magnitude=magnitude,
        )

        self._error_log.append(err.to_dict())
        self._last_error_magnitude = magnitude
        self._update_count += 1

        # Update transform and domain accuracy buffers
        self._record_transform_error(obs.transform_used, -magnitude)  # higher is better
        self._record_domain_error(self._current_state.domain, -magnitude)

        # Approved change target: record surprise must now feed bounded buffer.
        self._record_surprise(
            {
                "timestamp": time.time(),
                "cycle": self._cycle_count,
                "domain": self._current_state.domain,
                "expression": self._current_state.expression,
                "predicted_transform": pred.predicted_transform,
                "transform_used": obs.transform_used,
                "predicted_delta": pred.predicted_delta,
                "actual_delta": obs.actual_delta,
                "delta_error": delta_error,
                "result_match": result_match,
                "confidence": pred.confidence,
                "magnitude": magnitude,
                "success": obs.success,
            }
        )

        # If a world model is provided, allow it to consume the error signal.
        if world_model is not None:
            try:
                # Prefer a stable, optional method if present.
                if hasattr(world_model, "record_outcome"):
                    world_model.record_outcome(
                        prediction=pred.predicted_transform,
                        actual_transforms=[obs.transform_used],
                        actual_delta=obs.actual_delta - pred.predicted_delta,
                        domain=self._current_state.domain,
                    )
            except Exception:
                log.exception("world_model.record_outcome failed (ignored).")

        if induction_callback is not None:
            try:
                induction_callback(err)
            except Exception:
                log.exception("induction_callback failed (ignored).")

        self._last_error_magnitude = magnitude
        return err

    def _record_surprise(self, event: Dict[str, Any]) -> None:
        """
        Record a surprise / prediction error event for DreamConsolidator and analysis.

        Approved change:
          - Ensure bounded retention (no unbounded list growth).
          - Also maintain quick access to the last event.
        """
        try:
            # Normalize magnitude to float; ensure event is serializable-ish.
            if "magnitude" in event:
                event["magnitude"] = float(event["magnitude"])
            if "delta_error" in event:
                event["delta_error"] = float(event["delta_error"])
            self._surprise_events.append(event)
            self._last_surprise_event = event
        except Exception:
            log.exception("Failed to record surprise event (ignored).")

    def _record_transform_error(self, transform: str, score: float) -> None:
        buf = self._transform_accuracy.setdefault(transform, [])
        buf.append(float(score))
        # Keep bounded; error signals are local history.
        if len(buf) > 4000:
            del buf[: len(buf) - 4000]

    def _record_domain_error(self, domain: str, score: float) -> None:
        buf = self._domain_accuracy.setdefault(domain, [])
        buf.append(float(score))
        if len(buf) > 4000:
            del buf[: len(buf) - 4000]

    def _transform_confidence(self, transform: str) -> float:
        """
        Convert recent negative error scores into confidence in [0,1].
        Higher recent average error-score -> higher confidence.
        """
        buf = self._transform_accuracy.get(transform)
        if not buf:
            return 0.5
        # buf stores negative magnitudes; higher means better.
        avg = sum(buf[-200:]) / max(1, min(len(buf), 200))
        # avg is in rough [-inf, 0]; map to [0,1]
        # if avg == 0 -> 1.0; if avg is very negative -> approaches 0.
        conf = 1.0 / (1.0 + abs(avg) * 2.5)
        return float(max(0.0, min(1.0, conf)))

    def _domain_confidence(self, domain: str) -> float:
        buf = self._domain_accuracy.get(domain)
        if not buf:
            return 0.5
        avg = sum(buf[-200:]) / max(1, min(len(buf), 200))
        conf = 1.0 / (1.0 + abs(avg) * 2.2)
        return float(max(0.0, min(1.0, conf)))

    def _compute_error_magnitude(self, delta_error: float, confidence: float, result_match: bool) -> float:
        """
        Compute a normalized surprise magnitude in [0,1].
        - Larger delta_error => higher magnitude
        - Lower confidence => higher surprise
        - Result mismatch => extra surprise
        """
        # Normalize delta_error with a soft cap
        # Typical magnitudes should remain in a useful range for learning signals.
        norm = 1.0 - math.exp(-max(0.0, delta_error) * 2.0)  # in [0,1)
        conf_factor = 1.0 - max(0.0, min(1.0, confidence)) * 0.85  # lower conf -> larger
        match_factor = 0.35 if not result_match else 0.0
        magnitude = norm * conf_factor + match_factor
        return float(max(0.0, min(1.0, magnitude)))

    def get_recent_surprise_events(self, max_events: int = 50) -> List[dict]:
        """
        Return the most recent surprise events (oldest->newest within the returned window).
        """
        if max_events <= 0:
            return []
        n = min(int(max_events), len(self._surprise_events))
        if n == 0:
            return []
        # deque doesn't support slicing; take tail via iteration
        events = list(self._surprise_events)[-n:]
        return events

    def summary(self) -> dict:
        """
        Provide a compact status snapshot for other modules (e.g., evolution monitors).
        Approved change:
          - Ensure summary uses bounded surprise buffer semantics (no unbounded list growth).
        """
        recent_n = min(200, len(self._surprise_events))
        if recent_n > 0:
            tail = list(self._surprise_events)[-recent_n:]
            avg_mag = sum(float(e.get("magnitude", 0.0)) for e in tail) / recent_n
            top_mag = max(float(e.get("magnitude", 0.0)) for e in tail)
            high_count = sum(1 for e in tail if float(e.get("magnitude", 0.0)) >= 0.7)
        else:
            avg_mag = 0.0
            top_mag = 0.0
            high_count = 0

        return {
            "cycle_count": self._cycle_count,
            "update_count": self._update_count,
            "error_log_len": len(self._error_log),
            "surprise_buffer_len": len(self._surprise_events),
            "recent_surprise_avg_magnitude": round(avg_mag, 4),
            "recent_surprise_max_magnitude": round(top_mag, 4),
            "recent_surprise_high_events": int(high_count),
            "last_error_magnitude": round(float(self._last_error_magnitude), 4),
            "last_surprise_timestamp": float(self._last_surprise_event.get("timestamp")) if self._last_surprise_event else None,
        }