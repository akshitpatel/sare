"""
SensoryBridge — Physics-grounded Sensorimotor Observations

Gap: PredictiveWorldLoop currently uses the symbolic solver as its "world."
     Real sensorimotor learning requires observing an actual *physical* world.

This module bridges PredictiveWorldLoop ↔ PhysicsSimulator:
  - Intercepts predict→act cycles when the domain is physics-related
  - Routes "act" through PhysicsSimulator.simulate_*() instead of solver
  - Returns a physics-grounded Observation (actual_delta = physics energy)
  - Feeds prediction errors back as physics calibration signals

Architecture:
  PredictiveWorldLoop.run_cycle(expression, domain, engine)
       ↓  domain in physics_domains?
  SensoryBridge.act(expression, domain)
       ↓
  PhysicsSimulator.simulate_{domain}(params)
       ↓  returns PhysicsEvent
  Observation(actual_delta=event.energy_delta, transform="physics_sim")

Wiring (Brain._boot_knowledge):
    self.sensory_bridge = SensoryBridge()
    if self.physics_simulator and self.predictive_loop:
        self.sensory_bridge.wire(self.physics_simulator, self.predictive_loop)

Wiring (Brain.learn_cycle):
    # Auto-wired; SensoryBridge hooks into predictive_loop.run_cycle()
    self.sensory_bridge.tick()
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

PHYSICS_DOMAINS = {
    "physics", "mechanics", "thermodynamics",
    "electromagnetism", "optics", "quantum",
}

_PARAM_DEFAULTS: Dict[str, Dict[str, float]] = {
    "mechanics":       {"mass": 1.0, "velocity": 2.0, "height": 5.0},
    "thermodynamics":  {"temperature": 300.0, "pressure": 101325.0, "volume": 1.0},
    "electromagnetism":{"voltage": 5.0, "resistance": 10.0, "current": 0.5},
    "optics":          {"wavelength": 500.0, "angle": 30.0, "n1": 1.0, "n2": 1.5},
    "quantum":         {"n": 2, "l": 1, "hbar": 1.055e-34},
}

_METHOD_MAP: Dict[str, str] = {
    "mechanics":        "simulate_mechanics",
    "thermodynamics":   "simulate_thermodynamics",
    "electromagnetism": "simulate_electromagnetism",
    "optics":           "simulate_optics",
    "quantum":          "simulate_quantum",
    "physics":          "simulate_mechanics",    # default
}


@dataclass
class SensoryObservation:
    """
    Physics-grounded observation for PredictiveWorldLoop.update().
    Compatible with Observation dataclass in predictive_loop.py.
    """
    actual_result:    str
    actual_delta:     float
    transform_used:   str
    success:          bool
    elapsed_ms:       float
    physics_event:    Optional[Dict[str, Any]]   # raw PhysicsEvent.to_dict()
    grounded:         bool = True                 # distinguishes from solver obs

    def to_dict(self) -> dict:
        return {
            "actual_result":  self.actual_result,
            "actual_delta":   round(self.actual_delta, 4),
            "transform_used": self.transform_used,
            "success":        self.success,
            "elapsed_ms":     round(self.elapsed_ms, 2),
            "grounded":       self.grounded,
            "physics_event":  self.physics_event,
        }


@dataclass
class CalibrationRecord:
    """Tracks how well predictions matched physics observations."""
    domain:          str
    predicted_delta: float
    actual_delta:    float
    error:           float = field(init=False)
    timestamp:       float = field(default_factory=time.time)

    def __post_init__(self):
        self.error = abs(self.predicted_delta - self.actual_delta)

    def to_dict(self) -> dict:
        return {
            "domain":          self.domain,
            "predicted_delta": round(self.predicted_delta, 4),
            "actual_delta":    round(self.actual_delta, 4),
            "error":           round(self.error, 4),
        }


def _extract_params(expression: str, domain: str) -> Dict[str, float]:
    """Extract numeric parameters from expression string."""
    params = dict(_PARAM_DEFAULTS.get(domain, _PARAM_DEFAULTS["mechanics"]))
    # Look for number assignments like "mass=2.5" or "v=10"
    for match in re.finditer(r'(\w+)\s*=\s*([\d\.]+)', expression):
        key, val = match.group(1), match.group(2)
        try:
            params[key] = float(val)
        except ValueError:
            pass
    # Look for bare numbers — assign to first parameter
    bare = re.findall(r'(?<![=\w])([\d]+\.?[\d]*)', expression)
    param_keys = list(params.keys())
    for i, num in enumerate(bare[:3]):
        if i < len(param_keys):
            try:
                params[param_keys[i]] = float(num)
            except ValueError:
                pass
    return params


class SensoryBridge:
    """
    Routes PredictiveWorldLoop observations through PhysicsSimulator
    for physics-domain expressions.

    Usage::
        bridge = SensoryBridge()
        bridge.wire(physics_sim, predictive_loop)

        # Then in Brain.learn_cycle, before predictive_loop.run_cycle():
        obs = bridge.observe(expression, domain, predicted_delta)
        if obs:
            # Use physics-grounded observation instead of solver obs
    """

    def __init__(self):
        self._physics_sim       = None
        self._predictive_loop   = None
        self._calibration: List[CalibrationRecord] = []
        self._total_grounded    = 0
        self._total_fallback    = 0
        self._domain_errors: Dict[str, List[float]] = {}

    def wire(self, physics_simulator, predictive_loop) -> None:
        """Wire into PhysicsSimulator and PredictiveWorldLoop."""
        self._physics_sim     = physics_simulator
        self._predictive_loop = predictive_loop
        log.info("[SensoryBridge] wired to PhysicsSimulator + PredictiveWorldLoop")

    # ── Core observation method ───────────────────────────────────────────────

    def observe(self, expression: str, domain: str,
                predicted_delta: float = 0.5) -> Optional[SensoryObservation]:
        """
        Generate a physics-grounded observation for the given expression.
        Returns None if domain is not physics-related or simulator unavailable.
        """
        canon = self._canonicalize_domain(domain)
        if not canon or not self._physics_sim:
            return None

        method_name = _METHOD_MAP.get(canon, "simulate_mechanics")
        method = getattr(self._physics_sim, method_name, None)
        if not method:
            return None

        params = _extract_params(expression, canon)
        t0 = time.perf_counter()
        try:
            event = method(**{k: v for k, v in list(params.items())[:3]})
        except TypeError:
            # Fallback: call with no args
            try:
                event = method()
            except Exception:
                return None
        except Exception as e:
            log.debug(f"[SensoryBridge] {method_name} failed: {e}")
            return None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Extract delta from event
        if hasattr(event, "energy_delta"):
            actual_delta = event.energy_delta
        elif hasattr(event, "result") and isinstance(event.result, (int, float)):
            actual_delta = float(event.result)
        else:
            actual_delta = predicted_delta   # unchanged → zero error

        # Record calibration
        cal = CalibrationRecord(
            domain=canon,
            predicted_delta=predicted_delta,
            actual_delta=actual_delta,
        )
        self._calibration.append(cal)
        if len(self._calibration) > 200:
            self._calibration = self._calibration[-200:]

        if canon not in self._domain_errors:
            self._domain_errors[canon] = []
        self._domain_errors[canon].append(cal.error)
        if len(self._domain_errors[canon]) > 50:
            self._domain_errors[canon] = self._domain_errors[canon][-50:]

        event_dict = event.to_dict() if hasattr(event, "to_dict") else {
            "event_type": method_name, "result": str(actual_delta)
        }
        self._total_grounded += 1

        return SensoryObservation(
            actual_result  = event_dict.get("result", str(actual_delta)),
            actual_delta   = actual_delta,
            transform_used = f"physics:{canon}",
            success        = True,
            elapsed_ms     = elapsed_ms,
            physics_event  = event_dict,
        )

    def run_grounded_cycle(self, expression: str, domain: str) -> Optional[dict]:
        """
        Run a full grounded cycle via PredictiveWorldLoop using physics observation.
        Returns cycle dict or None if not applicable.
        """
        if not self._predictive_loop:
            return None

        canon = self._canonicalize_domain(domain)
        if not canon:
            self._total_fallback += 1
            return None

        def physics_engine(expr):
            obs = self.observe(expr, domain, predicted_delta=0.5)
            if obs:
                return {
                    "result":          obs.actual_result,
                    "delta":           obs.actual_delta,
                    "transforms_used": [obs.transform_used],
                    "success":         obs.success,
                }
            self._total_fallback += 1
            return None

        try:
            result = self._predictive_loop.run_cycle(
                expression, domain, physics_engine
            )
            return result
        except Exception as e:
            log.debug(f"[SensoryBridge] grounded cycle failed: {e}")
            return None

    def tick(self) -> None:
        """Called each learn_cycle — runs a grounded physics observation."""
        if not self._physics_sim or not self._predictive_loop:
            return
        # Pick a random physics expression to ground
        import random
        scenarios = [
            ("mass=2.0 velocity=3.0", "mechanics"),
            ("temperature=350.0 pressure=200000.0", "thermodynamics"),
            ("voltage=12.0 resistance=4.0", "electromagnetism"),
        ]
        expr, domain = random.choice(scenarios)
        self.run_grounded_cycle(expr, domain)

    # ── Calibration summary ───────────────────────────────────────────────────

    def avg_calibration_error(self, domain: Optional[str] = None) -> float:
        errors = self._domain_errors.get(domain, []) if domain else [
            e for lst in self._domain_errors.values() for e in lst
        ]
        return sum(errors) / max(len(errors), 1)

    def summary(self) -> dict:
        domain_cal = {}
        for dom, errors in self._domain_errors.items():
            domain_cal[dom] = {
                "observations": len(errors),
                "avg_error":    round(sum(errors) / max(len(errors), 1), 4),
            }
        recent = [c.to_dict() for c in self._calibration[-5:]]
        return {
            "total_grounded":    self._total_grounded,
            "total_fallback":    self._total_fallback,
            "domain_calibration": domain_cal,
            "avg_error_overall": round(self.avg_calibration_error(), 4),
            "recent_calibrations": recent,
            "wired":             self._physics_sim is not None,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _canonicalize_domain(domain: str) -> Optional[str]:
        d = domain.lower()
        for pd in PHYSICS_DOMAINS:
            if pd in d:
                return pd
        return None
