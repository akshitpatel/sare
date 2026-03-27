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
        self._total_cycles: int = 0
        self._total_successes: int = 0
        self._avg_error: float = 0.0
        self._phase: str = "idle"

        # History for causal learning
        self._causal_patterns: Dict[str, int] = {}   # "transform:domain" → success count
        self._surprise_events: List[dict] = []       # high-error events

    # ── Predict ────────────────────────────────────────────────────────────────

    def predict(self, state: WorldState, available_transforms: List[str]) -> Prediction:
        """
        Predict the best transform to apply and its expected outcome.

        Strategy:
          1. Check historical accuracy of each transform in this domain
          2. Pick the transform with highest expected delta × confidence
          3. Estimate result based on past outcomes
        """
        self._current_state = state
        self._phase = "predicting"

        if not available_transforms:
            return Prediction(
                predicted_result=state.expression,
                predicted_delta=0.0,
                predicted_transform="identity",
                confidence=0.1,
                reasoning="No transforms available",
            )

        # Score each transform
        best_transform = available_transforms[0]
        best_score = -1.0
        best_delta = 0.0

        for t in available_transforms:
            key = f"{t}:{state.domain}"
            history = self._transform_accuracy.get(t, [])
            domain_hist = self._causal_patterns.get(key, 0)

            # Confidence from accuracy history
            if history:
                acc = sum(history[-10:]) / len(history[-10:])
            else:
                acc = 0.5  # uninformed prior

            # Domain bonus
            domain_bonus = min(0.2, domain_hist * 0.02)

            score = acc + domain_bonus
            if score > best_score:
                best_score = score
                best_transform = t
                # Estimate delta from past performance
                past_deltas = [e.actual_delta for e in self._error_log
                               if e.transform == t]
                best_delta = sum(past_deltas[-5:]) / len(past_deltas[-5:]) \
                    if past_deltas else 0.5

        # Predict result: for symbolic simplification, prediction is usually "simpler form"
        # We don't know the actual result without running the engine, so predict symbolically
        predicted_result = f"simplified({state.expression})"

        pred = Prediction(
            predicted_result=predicted_result,
            predicted_delta=best_delta,
            predicted_transform=best_transform,
            confidence=min(0.95, best_score),
            reasoning=f"Best transform '{best_transform}' with score {best_score:.2f} in domain '{state.domain}'",
        )
        self._last_prediction = pred
        return pred

    # ── Act ────────────────────────────────────────────────────────────────────

    def act(self, engine_fn, prediction: Prediction) -> Observation:
        """
        Apply the predicted action and return what actually happened.
        engine_fn(expression) → {result, delta, transforms_used, success}
        """
        self._phase = "acting"
        state = self._current_state
        if not state:
            return Observation("", 0.0, "none", False)

        t0 = time.time()
        try:
            result = engine_fn(state.expression)
            elapsed = (time.time() - t0) * 1000
            obs = Observation(
                actual_result=result.get("result", state.expression),
                actual_delta=float(result.get("delta", 0.0)),
                transform_used=(result.get("transforms_used") or [prediction.predicted_transform])[0],
                success=bool(result.get("success", False)),
                elapsed_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            obs = Observation(
                actual_result=state.expression,
                actual_delta=0.0,
                transform_used="error",
                success=False,
                elapsed_ms=elapsed,
            )

        self._last_observation = obs
        self._phase = "observing"
        return obs

    # ── Update ─────────────────────────────────────────────────────────────────

    def update(self, prediction: Prediction,
               observation: Observation) -> PredictionError:
        """
        Compute prediction error and update transform confidence scores.
        Returns the PredictionError (the core learning signal).
        """
        self._phase = "updating"
        state = self._current_state

        # Compute error
        delta_err = abs(prediction.predicted_delta - observation.actual_delta)
        result_match = (observation.actual_result != state.expression
                        if state else False)  # actually got a simplification
        magnitude = min(1.0, delta_err / max(abs(prediction.predicted_delta), 0.1))

        err = PredictionError(
            expression=state.expression if state else "",
            predicted_delta=prediction.predicted_delta,
            actual_delta=observation.actual_delta,
            delta_error=delta_err,
            result_match=result_match,
            transform=observation.transform_used,
            magnitude=magnitude,
        )
        self._error_log.append(err)

        # Update transform accuracy
        t = observation.transform_used
        if t not in self._transform_accuracy:
            self._transform_accuracy[t] = []
        self._transform_accuracy[t].append(1.0 if observation.success else 0.0)

        # Update causal patterns
        if state:
            key = f"{t}:{state.domain}"
            self._causal_patterns[key] = self._causal_patterns.get(key, 0) + (
                1 if observation.success else 0)
            # Domain accuracy
            if state.domain not in self._domain_accuracy:
                self._domain_accuracy[state.domain] = []
            self._domain_accuracy[state.domain].append(1.0 - magnitude)

        # Track surprise events (high prediction error = unexpected = important)
        if magnitude > 0.5:
            self._surprise_events.append({
                "expression": state.expression if state else "",
                "transform": t,
                "expected_delta": prediction.predicted_delta,
                "actual_delta": observation.actual_delta,
                "magnitude": magnitude,
                "ts": time.time(),
            })

        # Update running average error
        n = self._total_cycles + 1
        self._avg_error = ((self._avg_error * self._total_cycles) + magnitude) / n

        self._total_cycles += 1
        if observation.success:
            self._total_successes += 1

        self._phase = "idle"
        return err

    # ── Full cycle ─────────────────────────────────────────────────────────────

    def run_cycle(self, expression: str, domain: str,
                  engine_fn, available_transforms: List[str] = None) -> dict:
        """
        Run one full predict → act → observe → update cycle.
        Returns a summary dict with all four phases.
        """
        if available_transforms is None:
            available_transforms = ["add_zero_elim", "mul_one_elim",
                                    "double_neg", "identity"]

        state = WorldState(expression=expression, domain=domain,
                           energy=len(expression) * 0.1,
                           step=self._total_cycles)

        prediction = self.predict(state, available_transforms)
        observation = self.act(engine_fn, prediction)
        error = self.update(prediction, observation)

        return {
            "state": state.to_dict(),
            "prediction": prediction.to_dict(),
            "observation": observation.to_dict(),
            "error": error.to_dict(),
            "learned": error.magnitude < 0.2,  # low error = model was correct
            "surprised": error.magnitude > 0.5, # high error = model was wrong → update
        }

    def run_session(self, problems: List[dict], engine_fn) -> List[dict]:
        """
        Run the predictive loop over a batch of problems.
        problems: list of {expression, domain} dicts
        """
        results = []
        for p in problems:
            expr = p.get("expression", "x + 0")
            domain = p.get("domain", "general")
            transforms = p.get("transforms", None)
            r = self.run_cycle(expr, domain, engine_fn, transforms)
            results.append(r)
        return results

    # ── Introspection ──────────────────────────────────────────────────────────

    def best_transforms(self, n: int = 5) -> List[dict]:
        """Return the N most reliable transforms by accuracy."""
        ranked = []
        for t, hist in self._transform_accuracy.items():
            if hist:
                acc = sum(hist[-20:]) / len(hist[-20:])
                ranked.append({"transform": t, "accuracy": round(acc, 3),
                               "uses": len(hist)})
        ranked.sort(key=lambda x: x["accuracy"], reverse=True)
        return ranked[:n]

    def worst_transforms(self, n: int = 3) -> List[dict]:
        """Return transforms with poorest prediction accuracy (high surprise)."""
        ranked = []
        for t, hist in self._transform_accuracy.items():
            if len(hist) >= 3:
                acc = sum(hist[-20:]) / len(hist[-20:])
                ranked.append({"transform": t, "accuracy": round(acc, 3),
                               "uses": len(hist)})
        ranked.sort(key=lambda x: x["accuracy"])
        return ranked[:n]

    def domain_accuracy_snapshot(self) -> Dict[str, float]:
        """Return mean prediction accuracy per domain."""
        return {
            d: round(sum(hist[-20:]) / len(hist[-20:]), 3)
            for d, hist in self._domain_accuracy.items()
            if hist
        }

    def summary(self) -> dict:
        return {
            "total_cycles": self._total_cycles,
            "total_successes": self._total_successes,
            "success_rate": round(self._total_successes / max(self._total_cycles, 1), 3),
            "avg_prediction_error": round(self._avg_error, 4),
            "phase": self._phase,
            "transforms_tracked": len(self._transform_accuracy),
            "domains_tracked": len(self._domain_accuracy),
            "surprise_events": len(self._surprise_events),
            "causal_patterns": len(self._causal_patterns),
            "best_transforms": self.best_transforms(3),
            "domain_accuracy": self.domain_accuracy_snapshot(),
            "recent_errors": [e.to_dict() for e in list(self._error_log)[-5:]],
            "recent_surprises": self._surprise_events[-3:],
        }
