"""
ExperientialLearner — predict→experience→surprise→update loop for SARE-HX.

Closes the tight feedback cycle:
  1. Predict  — before each solve, predict which transform will work and expected energy delta
  2. Experience — the solve happens (externally), producing actual outcome
  3. Surprise — measure |predicted_delta - actual_delta|; high surprise = learning opportunity
  4. Update   — update WorldModel beliefs proportional to surprise
  5. Imagine  — periodically replay stored episodes with different transforms (counterfactual)

Persists to: data/memory/experiential_learner.json
No LLM calls anywhere in this file.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

log = logging.getLogger(__name__)

_PERSIST_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "experiential_learner.json"
_RING_CAPACITY = 1000   # prediction ring buffer
_SURPRISE_CAP = 200     # surprise history deque capacity
_PERSIST_EVERY = 100    # save every N experiences


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class ExperiencePrediction:
    """Holds a prediction made before a solve attempt."""
    transform_name: str
    expected_delta: float
    confidence: float
    expression: str
    domain: str


# ── Main class ────────────────────────────────────────────────────────────────

class ExperientialLearner:
    """
    Closes the predict→experience→surprise→update loop.

    Usage in ExperimentRunner or learn_daemon:
        el = get_experiential_learner()

        # Before solve:
        prediction = el.predict(expression, transforms, domain)

        # After solve:
        surprise = el.record_experience(prediction, actual_transforms_used,
                                        actual_delta, domain, solved)

        # Every N cycles:
        el.imagine_batch(n=20)  # replay stored episodes with counterfactual transforms
    """

    def __init__(self):
        # Ring buffer for predictions (list used as circular buffer, capped at _RING_CAPACITY)
        self._predictions: List[ExperiencePrediction] = []

        # Surprise history deque (recent 200 surprise values)
        self._surprise_history: Deque[float] = deque(maxlen=_SURPRISE_CAP)

        # EMA of surprise per domain: {domain: float}
        self._domain_surprise_ema: Dict[str, float] = {}

        # Running stats
        self._total_experiences: int = 0
        self._counterfactuals_added: int = 0
        self._correct_predictions: int = 0   # |predicted - actual| < 0.3

        # Save-trigger counter
        self._since_last_save: int = 0

        # Load persisted state
        self._load()

    # ── 1. Predict ────────────────────────────────────────────────────────────

    def predict(self, expression: str, transforms: list, domain: str) -> ExperiencePrediction:
        """
        Before a solve: ask WorldModel which transform will work best.

        Stores the prediction in a ring buffer (capacity 1000).
        Returns an ExperiencePrediction that should be passed to record_experience().
        """
        transform_name = "unknown"
        expected_delta = 0.0
        confidence = 0.3

        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()

            # Build a minimal graph-like object from the expression string.
            # predict_transform() uses _graph_signature() which accepts any object
            # with a reasonable repr — we pass a lightweight proxy.
            class _GraphProxy:
                def __init__(self, expr):
                    self._expr = expr
                def __repr__(self):
                    return self._expr

            proxy = _GraphProxy(expression)
            pred = wm.predict_transform(proxy, transforms, domain)
            transform_name = pred.transform_name
            expected_delta = pred.expected_delta
            confidence = pred.confidence
        except Exception as exc:
            log.debug("[ExperientialLearner] predict() error: %s", exc)

        ep = ExperiencePrediction(
            transform_name=transform_name,
            expected_delta=expected_delta,
            confidence=confidence,
            expression=expression,
            domain=domain,
        )

        # Store in ring buffer (cap at _RING_CAPACITY)
        self._predictions.append(ep)
        if len(self._predictions) > _RING_CAPACITY:
            self._predictions = self._predictions[-_RING_CAPACITY:]

        return ep

    # ── 2–4. Record, Surprise, Update ────────────────────────────────────────

    def record_experience(
        self,
        prediction: Optional[ExperiencePrediction],
        actual_transforms: List[str],
        actual_delta: float,
        domain: str,
        solved: bool,
    ) -> float:
        """
        After a solve: measure surprise and update WorldModel beliefs.

        If prediction is None (no prior predict() call), surprise is computed
        against a zero baseline (expected_delta=0.0, confidence=0.0).

        Returns the surprise value in [0, ∞).
        """
        # Graceful None handling
        if prediction is None:
            expected_delta = 0.0
            confidence = 0.0
            predicted_transform = ""
        else:
            expected_delta = prediction.expected_delta
            confidence = prediction.confidence
            predicted_transform = prediction.transform_name

        # Surprise = normalised absolute error
        surprise = abs(expected_delta - actual_delta) / (abs(expected_delta) + 0.1)

        # Track surprise
        self._surprise_history.append(surprise)
        self._total_experiences += 1

        # Update per-domain EMA
        prev_ema = self._domain_surprise_ema.get(domain, surprise)
        self._domain_surprise_ema[domain] = 0.9 * prev_ema + 0.1 * surprise

        # Correct-prediction counter
        if abs(expected_delta - actual_delta) < 0.3:
            self._correct_predictions += 1

        # WorldModel update proportional to surprise
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()

            if surprise > 0.5:
                # High surprise → strong observe_solve to update beliefs
                boosted_confidence = min(0.99, confidence * 1.5)
                # We call observe_solve; extra confidence is encoded via the
                # energy_delta (WorldModel uses it to scale add_causal_link confidence)
                effective_delta = actual_delta * (1.0 + boosted_confidence)
                expr = (prediction.expression if prediction is not None else "")
                wm.observe_solve(
                    expression=expr,
                    transforms_used=actual_transforms if actual_transforms else ["unknown"],
                    energy_delta=effective_delta,
                    domain=domain,
                    solved=solved,
                )
            elif surprise < 0.1 and predicted_transform:
                # Near-perfect prediction → reinforce belief in predicted transform
                try:
                    wm._belief_accuracy.setdefault(predicted_transform, [])
                    wm._belief_accuracy[predicted_transform].append(1.0)
                    if len(wm._belief_accuracy[predicted_transform]) > 100:
                        wm._belief_accuracy[predicted_transform] = (
                            wm._belief_accuracy[predicted_transform][-100:]
                        )
                except Exception as inner_exc:
                    log.debug("[ExperientialLearner] belief_accuracy update error: %s", inner_exc)
            else:
                # Normal observation — let WorldModel do its standard update
                expr = (prediction.expression if prediction is not None else "")
                if actual_transforms or solved:
                    wm.observe_solve(
                        expression=expr,
                        transforms_used=actual_transforms if actual_transforms else ["unknown"],
                        energy_delta=actual_delta,
                        domain=domain,
                        solved=solved,
                    )
        except Exception as exc:
            log.debug("[ExperientialLearner] record_experience WorldModel update error: %s", exc)

        # Periodic save
        self._since_last_save += 1
        if self._since_last_save >= _PERSIST_EVERY:
            self._save()
            self._since_last_save = 0

        return surprise

    # ── 5. Imagine ────────────────────────────────────────────────────────────

    def imagine_batch(self, n: int = 20) -> int:
        """
        "Imagine" counterfactual solves by replaying WorldModel solve history
        with alternative transforms.

        For each episode in recent solve history, pick a DIFFERENT transform
        and estimate what would have happened using the causal boost.
        If counterfactual delta > 0.3, add a causal link with confidence 0.3.

        Returns the number of new counterfactual beliefs added.
        """
        added = 0
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()

            history = list(wm._solve_history)
            # Take the n most recent successful solves
            recent = [ep for ep in history if ep.get("success")][-n:]
            if not recent:
                return 0

            for ep in recent:
                try:
                    domain = ep.get("domain", "general")
                    used_transforms = ep.get("transforms", [])
                    expression = ep.get("expression", "")

                    # Collect all known transforms from belief_accuracy keys
                    known_transforms = list(wm._belief_accuracy.keys())
                    if not known_transforms:
                        continue

                    # Pick a transform NOT in the used list
                    alternatives = [t for t in known_transforms if t not in used_transforms]
                    if not alternatives:
                        # Fall back: pick the least-used known transform
                        alternatives = known_transforms

                    # Choose the alternative with the highest causal boost
                    best_alt = max(
                        alternatives,
                        key=lambda t: wm._get_causal_boost(t, domain),
                    )

                    # Estimate counterfactual delta
                    cf_delta = wm._get_causal_boost(best_alt, domain)

                    if cf_delta > 0.3:
                        # Add causal link as a weak belief from imagination
                        cause = f"pattern_with_{best_alt}_applicable"
                        effect = f"energy_reduced_by_{best_alt}"
                        wm.add_causal_link(
                            cause=cause,
                            effect=effect,
                            mechanism=best_alt,
                            domain=domain,
                            confidence=0.3,
                        )
                        added += 1
                except Exception as inner_exc:
                    log.debug("[ExperientialLearner] imagine episode error: %s", inner_exc)

        except Exception as exc:
            log.debug("[ExperientialLearner] imagine_batch error: %s", exc)

        self._counterfactuals_added += added
        return added

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_weakest_domain(self) -> str:
        """
        Return the domain where the agent is most confused
        (highest surprise EMA = most to learn).

        Falls back to "arithmetic" if no domain data is available.
        """
        if not self._domain_surprise_ema:
            return "arithmetic"
        try:
            return max(self._domain_surprise_ema, key=self._domain_surprise_ema.get)
        except Exception:
            return "arithmetic"

    def get_stats(self) -> dict:
        """
        Return a summary of the experiential learning state.

        Keys:
          surprise_mean         — mean of recent surprise history
          surprise_ema          — EMA of the last surprise value across all domains
          weakest_domain        — domain with highest surprise EMA
          domain_surprise_ema   — per-domain EMA dict
          total_experiences     — total record_experience() calls
          counterfactuals_added — total counterfactual beliefs injected
          prediction_accuracy   — fraction of predictions where |pred-actual| < 0.3
        """
        hist = list(self._surprise_history)
        surprise_mean = sum(hist) / len(hist) if hist else 0.0

        # Overall EMA: mean of all domain EMAs
        ema_vals = list(self._domain_surprise_ema.values())
        surprise_ema = sum(ema_vals) / len(ema_vals) if ema_vals else 0.0

        prediction_accuracy = (
            self._correct_predictions / self._total_experiences
            if self._total_experiences > 0
            else 0.0
        )

        return {
            "surprise_mean": round(surprise_mean, 4),
            "surprise_ema": round(surprise_ema, 4),
            "weakest_domain": self.get_weakest_domain(),
            "domain_surprise_ema": {
                k: round(v, 4) for k, v in self._domain_surprise_ema.items()
            },
            "total_experiences": self._total_experiences,
            "counterfactuals_added": self._counterfactuals_added,
            "prediction_accuracy": round(prediction_accuracy, 4),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Atomically persist domain EMA and stats to disk."""
        try:
            os.makedirs(_PERSIST_PATH.parent, exist_ok=True)
            data = {
                "domain_surprise_ema": self._domain_surprise_ema,
                "total_experiences": self._total_experiences,
                "counterfactuals_added": self._counterfactuals_added,
                "correct_predictions": self._correct_predictions,
            }
            tmp = _PERSIST_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp, _PERSIST_PATH)
            log.debug("[ExperientialLearner] saved to %s", _PERSIST_PATH)
        except Exception as exc:
            log.debug("[ExperientialLearner] save error: %s", exc)

    def _load(self) -> None:
        """Load persisted state from disk if available."""
        if not _PERSIST_PATH.is_file():
            log.debug("[ExperientialLearner] no persisted state found, starting fresh.")
            return
        try:
            with open(_PERSIST_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._domain_surprise_ema = data.get("domain_surprise_ema", {})
            self._total_experiences = int(data.get("total_experiences", 0))
            self._counterfactuals_added = int(data.get("counterfactuals_added", 0))
            self._correct_predictions = int(data.get("correct_predictions", 0))
            log.debug(
                "[ExperientialLearner] loaded from %s (experiences=%d)",
                _PERSIST_PATH,
                self._total_experiences,
            )
        except Exception as exc:
            log.debug("[ExperientialLearner] load error: %s", exc)


# ── Singleton ─────────────────────────────────────────────────────────────────

_SINGLETON: Optional[ExperientialLearner] = None


def get_experiential_learner() -> ExperientialLearner:
    """Return the process-level singleton ExperientialLearner."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = ExperientialLearner()
    return _SINGLETON
