"""Epsilon-greedy algorithm selector — picks best strategy per task type."""
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "algorithm_selector.json"

_STRATEGIES = ["beam_search", "mcts", "greedy", "random_restart"]
_TASK_TYPES = ["algebra", "logic", "planning", "language", "coding", "science"]


class AlgorithmSelector:
    EPSILON = 0.1  # exploration rate

    def __init__(self):
        # win_rates[task_type][strategy] = [win1, win2, ...]
        self._win_rates: Dict[str, Dict[str, List[float]]] = {}
        self._total_selections = 0
        self.epsilon = self.EPSILON  # instance-level epsilon (MetaController may adjust)
        self._load()
        # T3-6: Meta-meta-learning controller
        self.meta_controller = MetaController(self)

    def select(self, task_type: str, strategies: Optional[List[str]] = None) -> str:
        """Epsilon-greedy: pick best strategy with prob (1-ε), random with prob ε."""
        options = strategies or _STRATEGIES
        if not options:
            return _STRATEGIES[0]

        self._total_selections += 1

        # Explore with probability epsilon (instance epsilon, adjusted by MetaController)
        if random.random() < self.epsilon:
            return random.choice(options)

        # Exploit: pick strategy with highest mean win rate for this task type
        task_data = self._win_rates.get(task_type, {})
        best, best_score = options[0], -1.0
        for s in options:
            rates = task_data.get(s, [])
            score = sum(rates) / len(rates) if rates else 0.0
            if score > best_score:
                best, best_score = s, score
        log.debug("[AlgorithmSelector] task=%s selected=%s score=%.3f", task_type, best, best_score)
        return best

    def record_outcome(self, task_type: str, strategy: str, won: bool) -> None:
        """Record a win/loss for a (task_type, strategy) pair."""
        self._win_rates.setdefault(task_type, {}).setdefault(strategy, [])
        lst = self._win_rates[task_type][strategy]
        # Compute delta: difference between this outcome and recent mean
        recent_mean = sum(lst[-10:]) / len(lst[-10:]) if lst else 0.5
        performance_delta = (1.0 if won else 0.0) - recent_mean
        lst.append(1.0 if won else 0.0)
        if len(lst) > 100:
            self._win_rates[task_type][strategy] = lst[-100:]
        self.save()
        # T3-6: Inform MetaController about this outcome
        try:
            self.meta_controller.update(performance_delta)
        except Exception:
            pass

    def save(self) -> None:
        """Persist win_rates to disk for cross-restart continuity."""
        try:
            import os
            import threading as _thr
            summary = {
                "win_rates": self._win_rates,
                "total_selections": self._total_selections,
            }
            _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _STATE_PATH.parent / f"algorithm_selector.{os.getpid()}.{_thr.get_ident()}.tmp"
            tmp.write_text(json.dumps(summary, indent=2))
            os.replace(tmp, _STATE_PATH)
        except Exception as e:
            log.warning("[AlgorithmSelector] Save failed: %s", e)

    def _load(self) -> None:
        """Restore win_rates from disk if available."""
        try:
            if _STATE_PATH.exists():
                data = json.loads(_STATE_PATH.read_text())
                self._win_rates = data.get("win_rates", {})
                self._total_selections = data.get("total_selections", 0)
        except Exception as e:
            log.debug("[AlgorithmSelector] Load failed: %s", e)

    def get_stats(self) -> dict:
        summary = {
            t: {s: round(sum(r) / len(r), 3) if r else 0.0
                for s, r in strats.items()}
            for t, strats in self._win_rates.items()
        }
        meta_stats = {}
        try:
            meta_stats = self.meta_controller.stats
        except Exception:
            pass
        return {
            "strategy_stats": summary,
            "epsilon": self.epsilon,
            "total_selections": self._total_selections,
            "status": "active" if self._win_rates else "idle",
            "meta_controller": meta_stats,
        }


class MetaController:
    """
    Controls the AlgorithmSelector's own exploration rate.

    Meta-meta-learning: learns WHEN to explore strategies vs exploit known good ones.
    If recent strategy switches led to improvements → increase epsilon (explore more)
    If recent strategy switches led to drops → decrease epsilon (exploit more)
    """

    def __init__(self, selector):
        self._selector = selector
        self._performance_history = []   # (timestamp, performance_delta)
        self._epsilon_history = []
        self._base_epsilon = getattr(selector, 'epsilon', 0.1)

    def update(self, performance_delta: float):
        """
        Update epsilon based on whether strategy switching helped.
        performance_delta > 0: recent switches improved performance → explore more
        performance_delta < 0: recent switches hurt performance → exploit more
        """
        self._performance_history.append((time.time(), performance_delta))
        # Keep last 20
        self._performance_history = self._performance_history[-20:]

        if len(self._performance_history) < 5:
            return

        recent = [d for _, d in self._performance_history[-10:]]
        avg_delta = sum(recent) / len(recent)

        # Adjust epsilon
        if avg_delta > 0.05:  # exploration is helping
            new_epsilon = min(0.3, self._base_epsilon * 1.1)
        elif avg_delta < -0.05:  # exploration is hurting
            new_epsilon = max(0.01, self._base_epsilon * 0.9)
        else:
            new_epsilon = self._base_epsilon  # no change

        if hasattr(self._selector, 'epsilon'):
            self._selector.epsilon = new_epsilon
        self._base_epsilon = new_epsilon
        self._epsilon_history.append(new_epsilon)

        log.debug("MetaController: epsilon adjusted to %.3f (avg_delta=%.3f)",
                  new_epsilon, avg_delta)

    @property
    def stats(self) -> dict:
        return {
            "current_epsilon": self._base_epsilon,
            "epsilon_history": self._epsilon_history[-10:],
            "performance_samples": len(self._performance_history),
        }


_ALGORITHM_SELECTOR_SINGLETON: Optional[AlgorithmSelector] = None


def get_algorithm_selector() -> AlgorithmSelector:
    global _ALGORITHM_SELECTOR_SINGLETON
    if _ALGORITHM_SELECTOR_SINGLETON is None:
        _ALGORITHM_SELECTOR_SINGLETON = AlgorithmSelector()
    return _ALGORITHM_SELECTOR_SINGLETON
