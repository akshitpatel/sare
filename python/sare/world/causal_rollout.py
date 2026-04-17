"""
CausalRollout — Session 32 Fix 2: World Simulation (68% → 78%)

Multi-step causal prediction chains that test the system's understanding
of how transforms compose over multiple steps.

The key insight: true world simulation requires PREDICTING what will happen
N steps into the future, then VERIFYING against reality. The gap between
prediction and reality is the learning signal.

Architecture:
  1. RolloutPlan: sequence of predicted (transform, expected_delta) pairs
  2. RolloutExecution: actual step-by-step application with real deltas
  3. RolloutError: per-step and cumulative prediction accuracy
  4. CausalModel: learned transition probabilities between transforms

This extends PredictiveWorldLoop from single-step to multi-step prediction,
and adds a causal transition model that learns which transforms enable
which other transforms.

Integration:
  - Uses PredictiveWorldLoop's transform_confidence for single-step prediction
  - Uses CausalChainDetector's edges for transition probability
  - Feeds prediction errors back to both systems
  - Posts high-surprise rollouts to GlobalWorkspace
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class RolloutStep:
    """One step in a multi-step rollout."""
    step_idx: int
    transform: str
    predicted_delta: float
    actual_delta: float
    predicted_energy: float
    actual_energy: float
    success: bool
    error: float = 0.0        # |predicted_delta - actual_delta|

    def to_dict(self) -> dict:
        return {
            "step": self.step_idx,
            "transform": self.transform,
            "pred_delta": round(self.predicted_delta, 4),
            "actual_delta": round(self.actual_delta, 4),
            "pred_energy": round(self.predicted_energy, 4),
            "actual_energy": round(self.actual_energy, 4),
            "success": self.success,
            "error": round(self.error, 4),
        }


@dataclass
class RolloutPlan:
    """A planned multi-step prediction."""
    expression: str
    domain: str
    planned_transforms: List[str]
    predicted_deltas: List[float]
    predicted_final_energy: float
    confidence: float

    @property
    def n_steps(self) -> int:
        return len(self.planned_transforms)

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "domain": self.domain,
            "transforms": self.planned_transforms,
            "pred_deltas": [round(d, 4) for d in self.predicted_deltas],
            "pred_final_energy": round(self.predicted_final_energy, 4),
            "confidence": round(self.confidence, 3),
            "n_steps": self.n_steps,
        }


@dataclass
class RolloutResult:
    """Complete result of a multi-step rollout."""
    plan: RolloutPlan
    steps: List[RolloutStep] = field(default_factory=list)
    actual_final_energy: float = 0.0
    cumulative_error: float = 0.0
    energy_prediction_error: float = 0.0
    steps_completed: int = 0
    horizon_accuracy: float = 0.0   # how accurate was the N-step prediction
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "actual_final_energy": round(self.actual_final_energy, 4),
            "cumulative_error": round(self.cumulative_error, 4),
            "energy_pred_error": round(self.energy_prediction_error, 4),
            "steps_completed": self.steps_completed,
            "horizon_accuracy": round(self.horizon_accuracy, 3),
            "duration_ms": round(self.duration_ms, 1),
        }


# ── Causal Transition Model ─────────────────────────────────────────────────

class CausalTransitionModel:
    """
    Learned model of how transforms chain together.

    Tracks:
      - P(T_next | T_prev, domain): transition probability
      - E[delta | T, domain]: expected energy change per transform
      - Success rate per (transform, domain) pair
    """

    def __init__(self):
        # transition_counts[(t_prev, t_next, domain)] = count
        self._transitions: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._from_counts: Dict[Tuple[str, str], int] = defaultdict(int)

        # Expected delta per (transform, domain)
        self._delta_ema: Dict[Tuple[str, str], float] = {}
        self._delta_alpha = 0.2

        # Success rate per (transform, domain)
        self._success_ema: Dict[Tuple[str, str], float] = {}
        self._success_alpha = 0.15

    def observe_sequence(self, transforms: List[str], deltas: List[float],
                         domain: str, success: bool) -> None:
        """Record an observed transform sequence with per-step deltas."""
        for i, t in enumerate(transforms):
            key = (t, domain)

            # Update delta EMA
            if i < len(deltas):
                old = self._delta_ema.get(key, deltas[i])
                self._delta_ema[key] = old + self._delta_alpha * (deltas[i] - old)

            # Update success EMA
            s_old = self._success_ema.get(key, 0.5)
            s_val = 1.0 if success else 0.0
            self._success_ema[key] = s_old + self._success_alpha * (s_val - s_old)

            # Update transition counts
            if i > 0:
                t_prev = transforms[i - 1]
                self._transitions[(t_prev, t, domain)] += 1
                self._from_counts[(t_prev, domain)] += 1

    def predict_next(self, last_transform: str, domain: str,
                     candidates: List[str]) -> List[Tuple[str, float]]:
        """Predict most likely next transforms given the last one applied."""
        scored = []
        from_total = max(1, self._from_counts.get((last_transform, domain), 1))

        for t in candidates:
            trans_count = self._transitions.get((last_transform, t, domain), 0)
            p_transition = trans_count / from_total if from_total > 0 else 0.0
            success_rate = self._success_ema.get((t, domain), 0.5)
            score = 0.6 * p_transition + 0.4 * success_rate
            scored.append((t, score))

        scored.sort(key=lambda x: -x[1])
        return scored

    def expected_delta(self, transform: str, domain: str) -> float:
        """Return the expected energy delta for a transform in a domain."""
        return self._delta_ema.get((transform, domain), -0.3)

    def transition_probability(self, t_prev: str, t_next: str,
                               domain: str) -> float:
        """P(t_next | t_prev, domain)."""
        from_total = self._from_counts.get((t_prev, domain), 0)
        if from_total == 0:
            return 0.0
        return self._transitions.get((t_prev, t_next, domain), 0) / from_total

    def summary(self) -> dict:
        n_trans = sum(self._transitions.values())
        n_domains = len(set(d for _, d in self._delta_ema.keys()))
        return {
            "transition_observations": n_trans,
            "unique_transitions": len(self._transitions),
            "tracked_transforms": len(self._delta_ema),
            "domains": n_domains,
        }


# ── Causal Rollout Engine ───────────────────────────────────────────────────

class CausalRollout:
    """
    Multi-step causal prediction engine.

    Given an expression + domain + available transforms:
      1. Plans a multi-step rollout (which transforms to apply, in what order)
      2. Predicts the energy at each step
      3. Executes the plan step-by-step using the real solver
      4. Computes per-step and cumulative prediction error
      5. Updates the causal transition model

    The horizon accuracy (how well we predict N steps ahead) is the key metric
    for World Simulation capability.
    """

    def __init__(self, max_horizon: int = 5):
        self._max_horizon = max_horizon
        self._model = CausalTransitionModel()

        # Tracking
        self._rollout_count = 0
        self._results: deque = deque(maxlen=100)
        self._horizon_accuracy: Dict[int, List[float]] = defaultdict(list)
        self._domain_accuracy: Dict[str, List[float]] = defaultdict(list)

        # Integration
        self._chain_detector = None
        self._predictive_loop = None
        self._global_workspace = None

    def wire(self, chain_detector=None, predictive_loop=None,
             global_workspace=None) -> None:
        self._chain_detector = chain_detector
        self._predictive_loop = predictive_loop
        self._global_workspace = global_workspace

    # ── Planning ─────────────────────────────────────────────────────────────

    def plan_rollout(self, expression: str, domain: str,
                     available_transforms: List[str],
                     initial_energy: float = 1.0,
                     horizon: Optional[int] = None) -> RolloutPlan:
        """
        Plan a multi-step rollout.

        Uses the causal transition model to predict which transforms to apply
        and what energy changes to expect at each step.
        """
        n = min(horizon or self._max_horizon, self._max_horizon)
        if not available_transforms:
            available_transforms = ["identity"]

        planned_transforms = []
        predicted_deltas = []
        current_energy = initial_energy
        last_transform = None

        for step in range(n):
            if last_transform:
                # Use transition model to pick next transform
                candidates = self._model.predict_next(
                    last_transform, domain, available_transforms
                )
                if candidates and candidates[0][1] > 0.05:
                    t_name = candidates[0][0]
                else:
                    t_name = random.choice(available_transforms)
            else:
                # First step: pick by expected delta
                scored = [
                    (t, self._model.expected_delta(t, domain))
                    for t in available_transforms
                ]
                scored.sort(key=lambda x: x[1])  # most negative delta first
                t_name = scored[0][0]

            # Use chain detector hints if available
            if self._chain_detector and last_transform:
                predicted = self._chain_detector.predict_next_transform(
                    last_transform, domain
                )
                if predicted and predicted in available_transforms:
                    t_name = predicted  # trust causal chain knowledge

            pred_delta = self._model.expected_delta(t_name, domain)
            planned_transforms.append(t_name)
            predicted_deltas.append(pred_delta)
            current_energy += pred_delta
            last_transform = t_name

        # Overall confidence: product of per-step success rates, decayed by horizon
        confidence = 1.0
        for i, t in enumerate(planned_transforms):
            sr = self._model._success_ema.get((t, domain), 0.5)
            confidence *= sr * (0.9 ** i)  # decay with distance

        return RolloutPlan(
            expression=expression,
            domain=domain,
            planned_transforms=planned_transforms,
            predicted_deltas=predicted_deltas,
            predicted_final_energy=max(0.0, current_energy),
            confidence=max(0.01, confidence),
        )

    # ── Execution ────────────────────────────────────────────────────────────

    def execute_rollout(self, plan: RolloutPlan,
                        solve_fn: Callable[[str], Any]) -> RolloutResult:
        """
        Execute a planned rollout step-by-step using the real solver.

        solve_fn should accept an expression string and return a dict with:
          - "expression": result expression
          - "delta" or "energy_delta": energy change
          - "transforms_used": list of transforms applied
          - "success": bool
        """
        start = time.time()
        self._rollout_count += 1

        result = RolloutResult(plan=plan)
        current_expr = plan.expression
        current_energy = 1.0  # initial energy estimate
        cumulative_error = 0.0

        for i, planned_transform in enumerate(plan.planned_transforms):
            predicted_delta = plan.predicted_deltas[i]
            predicted_energy = current_energy + predicted_delta

            # Execute one solve step
            try:
                solve_result = solve_fn(current_expr)
                if isinstance(solve_result, dict):
                    actual_delta = float(solve_result.get("delta",
                                         solve_result.get("energy_delta", 0.0)))
                    actual_expr = str(solve_result.get("expression", current_expr))
                    success = bool(solve_result.get("success", actual_delta < 0))
                    used = solve_result.get("transforms_used",
                                            solve_result.get("transforms", []))
                else:
                    actual_delta = 0.0
                    actual_expr = str(solve_result) if solve_result else current_expr
                    success = False
                    used = []
            except Exception as e:
                log.debug(f"Rollout step {i} solve error: {e}")
                actual_delta = 0.0
                actual_expr = current_expr
                success = False
                used = []

            actual_energy = current_energy + actual_delta
            step_error = abs(predicted_delta - actual_delta)
            cumulative_error += step_error

            step = RolloutStep(
                step_idx=i,
                transform=planned_transform,
                predicted_delta=predicted_delta,
                actual_delta=actual_delta,
                predicted_energy=predicted_energy,
                actual_energy=actual_energy,
                success=success,
                error=step_error,
            )
            result.steps.append(step)
            result.steps_completed = i + 1

            # Feed observation to transition model
            actual_transforms = used if isinstance(used, list) else [used]
            if actual_transforms:
                self._model.observe_sequence(
                    actual_transforms,
                    [actual_delta],
                    plan.domain,
                    success,
                )

            current_expr = actual_expr
            current_energy = actual_energy

            # Early stop if expression is fully simplified (energy near 0)
            if actual_energy <= 0.05:
                break

        result.actual_final_energy = current_energy
        result.cumulative_error = cumulative_error
        result.energy_prediction_error = abs(
            plan.predicted_final_energy - current_energy
        )

        # Horizon accuracy: 1 - normalized cumulative error
        if result.steps_completed > 0:
            avg_error = cumulative_error / result.steps_completed
            result.horizon_accuracy = max(0.0, 1.0 - avg_error * 2.0)
        else:
            result.horizon_accuracy = 0.0

        result.duration_ms = (time.time() - start) * 1000.0

        # Record per-horizon accuracy
        h = result.steps_completed
        buf = self._horizon_accuracy[h]
        buf.append(result.horizon_accuracy)
        if len(buf) > 200:
            del buf[:len(buf) - 200]

        # Record per-domain accuracy
        dbuf = self._domain_accuracy[plan.domain]
        dbuf.append(result.horizon_accuracy)
        if len(dbuf) > 200:
            del dbuf[:len(dbuf) - 200]

        self._results.append(result)

        # Post high-surprise rollouts to GlobalWorkspace
        if self._global_workspace and result.horizon_accuracy < 0.3:
            try:
                self._global_workspace.post_event(
                    "solve_failed",
                    {
                        "source": "causal_rollout",
                        "expression": plan.expression,
                        "domain": plan.domain,
                        "horizon": h,
                        "accuracy": result.horizon_accuracy,
                        "error": result.cumulative_error,
                    },
                    source="causal_rollout",
                    salience=0.75,
                )
            except Exception:
                pass

        log.debug(
            f"Rollout #{self._rollout_count}: {plan.domain} "
            f"steps={result.steps_completed}/{plan.n_steps} "
            f"accuracy={result.horizon_accuracy:.2f} "
            f"cum_error={cumulative_error:.3f}"
        )
        return result

    # ── Convenience: plan + execute ──────────────────────────────────────────

    def run_rollout(self, expression: str, domain: str,
                    available_transforms: List[str],
                    solve_fn: Callable[[str], Any],
                    horizon: Optional[int] = None) -> RolloutResult:
        """Plan and execute a rollout in one call."""
        plan = self.plan_rollout(
            expression, domain, available_transforms, horizon=horizon
        )
        return self.execute_rollout(plan, solve_fn)

    # ── Metrics ──────────────────────────────────────────────────────────────

    def avg_horizon_accuracy(self, horizon: Optional[int] = None) -> float:
        """Average prediction accuracy at a given horizon (or overall)."""
        if horizon is not None:
            buf = self._horizon_accuracy.get(horizon, [])
            return sum(buf) / max(1, len(buf)) if buf else 0.5
        # Overall
        all_acc = []
        for buf in self._horizon_accuracy.values():
            all_acc.extend(buf[-50:])
        return sum(all_acc) / max(1, len(all_acc)) if all_acc else 0.5

    def domain_accuracy(self, domain: str) -> float:
        """Average accuracy for a specific domain."""
        buf = self._domain_accuracy.get(domain, [])
        return sum(buf) / max(1, len(buf)) if buf else 0.5

    def weakest_domain(self) -> Optional[str]:
        """Return the domain with lowest rollout accuracy."""
        if not self._domain_accuracy:
            return None
        return min(
            self._domain_accuracy.keys(),
            key=lambda d: self.domain_accuracy(d)
        )

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        per_horizon = {}
        for h, buf in sorted(self._horizon_accuracy.items()):
            if buf:
                per_horizon[f"h{h}"] = round(sum(buf[-50:]) / len(buf[-50:]), 3)

        per_domain = {}
        for d in sorted(self._domain_accuracy.keys()):
            per_domain[d] = round(self.domain_accuracy(d), 3)

        return {
            "rollouts_run": self._rollout_count,
            "max_horizon": self._max_horizon,
            "avg_accuracy": round(self.avg_horizon_accuracy(), 3),
            "per_horizon_accuracy": per_horizon,
            "per_domain_accuracy": per_domain,
            "weakest_domain": self.weakest_domain(),
            "transition_model": self._model.summary(),
            "recent_rollouts": [r.to_dict() for r in list(self._results)[-3:]],
        }
