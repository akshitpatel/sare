"""
AutonomousTrainer — Gap 2: Continuous Autonomous Learning

Runs a background thread that feeds the Brain a continuous stream
of problems from multiple sources, 24/7, without human intervention.

Problem sources:
  1. generated_problems   — algebraic/logical expressions from ProblemGenerator
  2. physics_observations — events from PhysicsSimulator
  3. knowledge_concepts   — expressions from KnowledgeIngester concepts
  4. predictive_loop      — high-error expressions from PredictiveWorldLoop
  5. replay_failures      — past failures worth retrying

This is the critical scale upgrade:
  - Human intelligence: billions of learning events
  - SARE-HX without this: dozens per session
  - SARE-HX with this: thousands per hour, growing continuously

Key design choices:
  - Non-blocking: runs in a daemon thread, never blocks the main Brain
  - Source balancing: rotates across all 5 sources to ensure diversity
  - Curriculum-aware: harder problems as accuracy improves
  - Self-throttling: backs off if Brain is busy solving user problems
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
import concurrent.futures
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_STATS_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "autonomous_trainer_stats.json"


# ── Problem sources ────────────────────────────────────────────────────────────

@dataclass
class TrainingProblem:
    """One problem for the autonomous trainer."""
    expression: str
    domain: str
    source: str           # which source generated it
    difficulty: float     # 0.0 (easy) → 1.0 (hard)
    priority: float = 0.5 # higher = solve sooner

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "domain": self.domain,
            "source": self.source,
            "difficulty": round(self.difficulty, 2),
        }


@dataclass
class TrainingResult:
    """Result of one autonomous training step."""
    problem: TrainingProblem
    success: bool
    delta: float
    elapsed_ms: float
    source: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "expression": self.problem.expression,
            "domain": self.problem.domain,
            "source": self.source,
            "success": self.success,
            "delta": round(self.delta, 4),
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# Built-in problem seeds across domains
_SEED_PROBLEMS: List[dict] = [
    # Arithmetic (easy → hard)
    {"expr": "x + 0",              "domain": "arithmetic", "diff": 0.1},
    {"expr": "x * 1",              "domain": "arithmetic", "diff": 0.1},
    {"expr": "0 + x + 0",          "domain": "arithmetic", "diff": 0.2},
    {"expr": "x * 1 * y * 1",      "domain": "arithmetic", "diff": 0.3},
    {"expr": "(x + 0) * (y * 1)",  "domain": "arithmetic", "diff": 0.4},
    {"expr": "x + y - y",          "domain": "arithmetic", "diff": 0.5},
    {"expr": "(x * y) / y",        "domain": "arithmetic", "diff": 0.6},
    # Logic (easy → hard)
    {"expr": "A AND TRUE",         "domain": "logic",      "diff": 0.1},
    {"expr": "A OR FALSE",         "domain": "logic",      "diff": 0.1},
    {"expr": "NOT NOT A",          "domain": "logic",      "diff": 0.2},
    {"expr": "A AND TRUE OR FALSE","domain": "logic",      "diff": 0.4},
    {"expr": "NOT (A OR B)",       "domain": "logic",      "diff": 0.5},
    {"expr": "(A AND B) OR (A AND C)", "domain": "logic",  "diff": 0.6},
    # Algebra
    {"expr": "x^2 + 0",           "domain": "algebra",    "diff": 0.3},
    {"expr": "(x + 1) * 1",       "domain": "algebra",    "diff": 0.3},
    {"expr": "x^2 + 2*x + x^2",   "domain": "algebra",    "diff": 0.6},
    # Calculus
    {"expr": "d/dx(x^0)",         "domain": "calculus",   "diff": 0.4},
    {"expr": "d/dx(x^1)",         "domain": "calculus",   "diff": 0.4},
    {"expr": "d/dx(c * x^n)",     "domain": "calculus",   "diff": 0.7},
    # Thermodynamics
    {"expr": "Q = m * c * dT",    "domain": "thermodynamics", "diff": 0.5},
    {"expr": "PV = nRT",          "domain": "thermodynamics", "diff": 0.5},
]


class AutonomousTrainer:
    """
    Runs continuous background learning for SARE-HX.

    Usage:
        trainer = AutonomousTrainer()
        trainer.start(brain)     # kicks off background thread
        trainer.stop()           # graceful shutdown
        trainer.summary()        # live stats

    Sources are balanced round-robin with priority weighting.
    The trainer adapts difficulty based on recent solve rate.
    """

    def __init__(self, interval_seconds: float = 3.0,
                 batch_size: int = 20,
                 max_workers: int = 10,
                 max_history: int = 500):
        self._interval = interval_seconds
        self._batch_size = batch_size
        self._max_workers = max_workers
        self._history: deque = deque(maxlen=max_history)
        self._stats_lock = threading.Lock()
        self._source_counts: Dict[str, int] = defaultdict(int)
        self._source_successes: Dict[str, int] = defaultdict(int)
        self._domain_counts: Dict[str, int] = defaultdict(int)
        self._domain_successes: Dict[str, int] = defaultdict(int)
        self._total_problems: int = 0
        self._total_successes: int = 0
        self._current_difficulty: float = 0.3
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._batch_count: int = 0  # for periodic disk persistence
        self._brain_ref = None
        self._start_time: Optional[float] = None
        self._last_problem: Optional[TrainingProblem] = None
        self._failure_replay: deque = deque(maxlen=50)  # expressions to retry

        # Source rotation
        self._source_cycle = [
            "generated_problems",
            "seed_library",
            "knowledge_concepts",
            "failure_replay",
            "physics_expressions",
        ]
        self._source_index = 0
        # Bandit: per-source EMA solve rate for softmax selection
        self._source_ema: Dict[str, float] = {s: 0.5 for s in self._source_cycle}

    # ── Start / stop ───────────────────────────────────────────────────────────

    def start(self, brain) -> bool:
        """Start the autonomous training background thread."""
        if self._running:
            return False
        self._brain_ref = brain
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
            name="AutonomousTrainer",
        )
        self._thread.start()
        log.info(f"AutonomousTrainer started (interval={self._interval}s, workers={self._max_workers}, batch={self._batch_size})")
        return True

    def stop(self) -> bool:
        """Gracefully stop the training thread."""
        if not self._running:
            return False
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        log.info(f"AutonomousTrainer stopped after {self._total_problems} problems")
        return True

    # ── Training loop ──────────────────────────────────────────────────────────

    def _training_loop(self):
        """Main background training loop using a ThreadPool swarm."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="SareAgent") as executor:
            while self._running:
                t0 = time.time()
                try:
                    problems = self._sample_batch()
                    futures = []
                    for prob in problems:
                        if not self._running:
                            break
                        futures.append(executor.submit(self._solve_one, prob))
                    
                    # Wait for the swarm batch to finish
                    concurrent.futures.wait(futures, timeout=self._interval * 2)

                    self._adapt_difficulty()
                    self._batch_count += 1
                    if self._batch_count % 5 == 0:
                        self.save_stats()
                except Exception as e:
                    log.debug(f"AutonomousTrainer loop error: {e}")
                # Sleep remaining interval time
                elapsed = time.time() - t0
                sleep_time = max(0.1, self._interval - elapsed)
                time.sleep(sleep_time)

    def _solve_one(self, prob: TrainingProblem):
        """Solve one problem and record the result."""
        brain = self._brain_ref
        if not brain:
            return
        t0 = time.time()
        try:
            result = brain.solve(prob.expression)
            elapsed = (time.time() - t0) * 1000
            success = bool(result.get("success", False) or result.get("delta", 0) > 0.01)
            delta = float(result.get("delta", 0.0))

            tr = TrainingResult(
                problem=prob,
                success=success,
                delta=delta,
                elapsed_ms=elapsed,
                source=prob.source,
            )
            self._record(tr)

            # Feed into SelfModel if available
            if brain.self_model and hasattr(brain.self_model, 'observe'):
                try:
                    transforms = result.get("transforms_used", [])
                    brain.self_model.observe(
                        prob.domain, success, delta,
                        result.get("steps", 0), transforms,
                        confidence=0.7, strategy="beam_search",
                        elapsed_ms=elapsed,
                    )
                except Exception:
                    pass

            # Feed into PredictiveLoop if available
            if brain.predictive_loop and hasattr(brain.predictive_loop, 'run_cycle'):
                try:
                    brain.predictive_loop.run_cycle(
                        prob.expression, prob.domain,
                        lambda e: brain.solve(e),
                    )
                except Exception:
                    pass

            # Queue failures for replay
            if not success:
                # Thread-safe since deque is thread-safe for appends
                self._failure_replay.append(prob.expression)

        except Exception as e:
            log.debug(f"AutonomousTrainer solve error: {e}")

    def _record(self, tr: TrainingResult):
        """Record a training result and update stats. Thread-safe."""
        with self._stats_lock:
            self._history.append(tr)
            self._total_problems += 1
            if tr.success:
                self._total_successes += 1
            self._source_counts[tr.source] += 1
            if tr.success:
                self._source_successes[tr.source] += 1
            self._domain_counts[tr.problem.domain] += 1
            if tr.success:
                self._domain_successes[tr.problem.domain] += 1
            # Update per-source EMA solve rate for bandit selection
            if tr.source in self._source_ema:
                old = self._source_ema[tr.source]
                self._source_ema[tr.source] = 0.9 * old + 0.1 * float(tr.success)

    # ── Problem sampling ───────────────────────────────────────────────────────

    def _sample_batch(self) -> List[TrainingProblem]:
        """Sample a batch using softmax bandit over per-source EMA solve rates."""
        # Softmax bandit: sources with higher recent solve rates get sampled more
        weights = {s: self._source_ema.get(s, 0.5) + 0.1 for s in self._source_cycle}
        total = sum(weights.values())
        r = random.random() * total
        cumulative = 0.0
        source = self._source_cycle[-1]
        for s, w in weights.items():
            cumulative += w
            if r <= cumulative:
                source = s
                break
        problems = self._sample_from_source(source, self._batch_size)
        return problems

    def _sample_from_source(self, source: str,
                             n: int) -> List[TrainingProblem]:
        """Sample n problems from the named source."""
        if source == "seed_library":
            return self._from_seed_library(n)
        elif source == "generated_problems":
            return self._from_generator(n)
        elif source == "knowledge_concepts":
            return self._from_knowledge(n)
        elif source == "failure_replay":
            return self._from_failures(n)
        elif source == "physics_expressions":
            return self._from_physics(n)
        return self._from_seed_library(n)

    def _from_seed_library(self, n: int) -> List[TrainingProblem]:
        eligible = [p for p in _SEED_PROBLEMS
                    if abs(p["diff"] - self._current_difficulty) < 0.3]
        if not eligible:
            eligible = _SEED_PROBLEMS
        sampled = random.sample(eligible, min(n, len(eligible)))
        return [TrainingProblem(
            expression=p["expr"], domain=p["domain"],
            source="seed_library", difficulty=p["diff"],
        ) for p in sampled]

    def _from_generator(self, n: int) -> List[TrainingProblem]:
        brain = self._brain_ref
        problems = []
        if brain and hasattr(brain, '_pick_learning_problem'):
            for _ in range(n):
                try:
                    expr = brain._pick_learning_problem()
                    if expr:
                        selection = getattr(brain, '_last_problem_selection', {})
                        source = selection.get("source", "generated_problems")
                        domain = selection.get("domain") or self._guess_domain(expr)
                        
                        # Dynamically add to counters if new
                        if source not in self._source_counts:
                            self._source_counts[source] = 0
                            self._source_successes[source] = 0

                        problems.append(TrainingProblem(
                            expression=expr, domain=domain,
                            source=source,
                            difficulty=self._current_difficulty,
                        ))
                except Exception:
                    pass
        return problems or self._from_seed_library(n)

    def _from_knowledge(self, n: int) -> List[TrainingProblem]:
        brain = self._brain_ref
        problems = []
        if brain and brain.knowledge_ingester:
            try:
                ki = brain.knowledge_ingester
                concepts = list(ki._extracted.values())
                if concepts:
                    picked = random.sample(concepts, min(n, len(concepts)))
                    for c in picked:
                        rules = c.symbolic_rules
                        if rules:
                            problems.append(TrainingProblem(
                                expression=rules[0][:60],
                                domain=c.domain,
                                source="knowledge_concepts",
                                difficulty=0.4,
                            ))
            except Exception:
                pass
        return problems or self._from_seed_library(n)

    def _from_failures(self, n: int) -> List[TrainingProblem]:
        if not self._failure_replay:
            return self._from_seed_library(n)
        exprs = list(self._failure_replay)[-n:]
        return [TrainingProblem(
            expression=e, domain=self._guess_domain(e),
            source="failure_replay", difficulty=min(1.0, self._current_difficulty + 0.2),
        ) for e in exprs]

    def _from_physics(self, n: int) -> List[TrainingProblem]:
        brain = self._brain_ref
        problems = []
        if brain and brain.physics_simulator:
            try:
                rules = brain.physics_simulator.symbolic_rules()
                picked = random.sample(rules, min(n, len(rules)))
                for concept, rule in picked:
                    problems.append(TrainingProblem(
                        expression=rule[:60], domain="mechanics",
                        source="physics_expressions", difficulty=0.5,
                    ))
            except Exception:
                pass
        return problems or self._from_seed_library(n)

    # ── Curriculum adaptation ──────────────────────────────────────────────────

    def _adapt_difficulty(self):
        """Adjust difficulty based on recent solve rate."""
        recent = list(self._history)[-20:]
        if len(recent) < 5:
            return
        rate = sum(1 for r in recent if r.success) / len(recent)
        # If solving > 80%: increase difficulty
        if rate > 0.80:
            self._current_difficulty = min(1.0, self._current_difficulty + 0.05)
        # If solving < 40%: decrease difficulty
        elif rate < 0.40:
            self._current_difficulty = max(0.1, self._current_difficulty - 0.05)

    @staticmethod
    def _guess_domain(expr: str) -> str:
        expr = expr.lower()
        if any(k in expr for k in ["and", "or", "not", "true", "false"]):
            return "logic"
        if any(k in expr for k in ["d/dx", "integral", "lim", "sin", "cos"]):
            return "calculus"
        if any(k in expr for k in ["pv", "nrt", "heat", "entropy"]):
            return "thermodynamics"
        return "arithmetic"

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_problem(self, expression: str, domain: str = "arithmetic",
                    priority: float = 0.8):
        """Manually inject a high-priority problem into the next batch."""
        prob = TrainingProblem(expression=expression, domain=domain,
                               source="manual", difficulty=self._current_difficulty,
                               priority=priority)
        if self._brain_ref:
            threading.Thread(target=self._solve_one, args=(prob,),
                             daemon=True).start()

    def recent_rate(self, window: int = 20) -> float:
        recent = list(self._history)[-window:]
        if not recent:
            return 0.0
        return sum(1 for r in recent if r.success) / len(recent)

    def uptime_seconds(self) -> float:
        if not self._start_time:
            return 0.0
        return time.time() - self._start_time

    def summary(self) -> dict:
        return {
            "running": self._running,
            "total_problems": self._total_problems,
            "total_successes": self._total_successes,
            "recent_rate": round(self.recent_rate(), 3),
            "current_difficulty": round(self._current_difficulty, 2),
            "uptime_seconds": round(self.uptime_seconds(), 1),
            "source_stats": {
                s: {
                    "problems": self._source_counts[s],
                    "successes": self._source_successes[s],
                    "rate": round(self._source_successes[s] /
                                  max(self._source_counts[s], 1), 3),
                }
                for s in self._source_cycle
            },
            "domain_stats": {
                d: {
                    "problems": self._domain_counts[d],
                    "rate": round(self._domain_successes[d] /
                                  max(self._domain_counts[d], 1), 3),
                }
                for d in sorted(self._domain_counts)
            },
            "failure_queue": len(self._failure_replay),
            "recent_results": [r.to_dict() for r in list(self._history)[-6:]],
        }

    def save_stats(self) -> None:
        """Persist current summary to disk atomically (for web server to read)."""
        try:
            data = self.summary()
            data["saved_at"] = time.time()
            _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
            _tmp = _STATS_PATH.parent / f"autonomous_trainer_stats.{os.getpid()}.tmp"
            _tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(_tmp, _STATS_PATH)
        except Exception as e:
            log.debug("AutonomousTrainer save_stats error: %s", e)
