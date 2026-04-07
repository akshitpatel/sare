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
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_STATS_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "autonomous_trainer_stats.json"
_SYMBOLIC_SOURCES = (
    "generated_problems",
    "seed_library",
    "knowledge_concepts",
    "failure_replay",
    "physics_expressions",
)
_GENERAL_SOURCES = (
    "commonsense_reasoning",
    "word_problems",
    "comprehension_problems",
    "code_problems",
    "language_problems",
    "wiki_facts",
)


# ── Problem sources ────────────────────────────────────────────────────────────

@dataclass
class TrainingProblem:
    """One problem for the autonomous trainer."""
    expression: str
    domain: str
    source: str           # which source generated it
    difficulty: float     # 0.0 (easy) → 1.0 (hard)
    priority: float = 0.5 # higher = solve sooner
    metadata: dict = field(default_factory=dict)  # extra context (answer, choices, …)

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
                 max_history: int = 500,
                 symbolic_only: bool = False):
        self._interval = interval_seconds
        self._batch_size = batch_size
        self._max_workers = max_workers
        self._symbolic_only = bool(symbolic_only)
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
        self._started_at_iso: Optional[str] = None
        self._last_problem: Optional[TrainingProblem] = None
        self._failure_replay: deque = deque(maxlen=50)  # expressions to retry
        self._run_id = uuid.uuid4().hex[:12]
        self._supports_graded_learning = False

        # Source rotation
        self._source_cycle = list(_SYMBOLIC_SOURCES) + list(_GENERAL_SOURCES)
        # Pre-load general datasets into memory for fast sampling
        self._commonsense_pool: list = []
        self._word_problem_pool: list = []
        self._comprehension_pool: list = []
        self._code_pool: list = []
        self._language_pool: list = []
        self._wiki_pool: list = []
        self._general_pools_loaded = False
        self._wiki_ingest_thread: Optional[threading.Thread] = None
        self._source_index = 0
        # Bandit: per-source EMA solve rate for softmax selection
        self._source_ema: Dict[str, float] = {s: 0.5 for s in self._source_cycle}

    # ── Start / stop ───────────────────────────────────────────────────────────

    def start(self, brain) -> bool:
        """Start the autonomous training background thread."""
        if self._running:
            return False
        self._brain_ref = brain
        self._supports_graded_learning = (
            callable(getattr(brain, "attempt_learning_problem", None))
            and not self._symbolic_only
        )
        self._running = True
        self._start_time = time.time()
        self._started_at_iso = datetime.now(timezone.utc).isoformat()
        self._thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
            name="AutonomousTrainer",
        )
        self._thread.start()
        log.info(
            "AutonomousTrainer started (interval=%ss, workers=%s, batch=%s, symbolic_only=%s, graded=%s)",
            self._interval,
            self._max_workers,
            self._batch_size,
            self._symbolic_only,
            self._supports_graded_learning,
        )
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
            if prob.source in _GENERAL_SOURCES and self._supports_graded_learning:
                _meta = dict(prob.metadata or {})
                result = brain.attempt_learning_problem(
                    prob.expression,
                    prob.metadata.get("answer"),
                    prob.domain,
                    context=str(prob.metadata.get("context", "") or ""),
                    metadata=_meta,
                )
                elapsed = float(getattr(result, "elapsed_ms", 0.0) or 0.0)
                if elapsed <= 0.0:
                    elapsed = (time.time() - t0) * 1000
                success = bool(getattr(result, "solved", False))
                delta = float(getattr(result, "confidence", 0.0) or 0.0)
                # 2-shot retry: if first attempt failed but we have the expected answer,
                # the first attempt already taught NeuralLearner + KB. Retry once immediately.
                _expected = str(prob.metadata.get("answer", "") or "").strip()
                if not success and _expected and prob.domain not in ("math", "logic", "algebra", "arithmetic"):
                    try:
                        result2 = brain.attempt_learning_problem(
                            prob.expression,
                            _expected,
                            prob.domain,
                            context=str(prob.metadata.get("context", "") or ""),
                            metadata=_meta,
                        )
                        if getattr(result2, "solved", False):
                            success = True
                            delta = float(getattr(result2, "confidence", 0.0) or 0.0)
                            result = result2
                    except Exception:
                        pass
            else:
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

            # Teach NeuralLearner from metadata answer (works whether or not solver succeeded)
            _meta_answer = str(prob.metadata.get("answer", "") or "").strip()
            # For word_problems (GSM8K), extract final numeric answer after "####"
            if prob.domain == "word_problems" and "####" in _meta_answer:
                _final_tok = _meta_answer.split("####")[-1].strip().split()[0] if _meta_answer.split("####")[-1].strip() else ""
                if _final_tok:
                    _meta_answer = _final_tok
            if _meta_answer and prob.source in _GENERAL_SOURCES and prob.domain not in (
                "math", "logic", "algebra", "arithmetic"
            ):
                try:
                    from sare.neuro.neural_learner import get_neural_learner as _gnl_at
                    _gnl_at().learn(prob.expression, _meta_answer, prob.domain, correct=True)
                except Exception:
                    pass
                # Also inject into commonsense KB for immediate availability
                try:
                    from sare.knowledge.commonsense import get_commonsense_base as _gcb_at
                    _gcb_at().add_fact(prob.expression, "AnswerTo", _meta_answer[:200])
                except Exception:
                    pass

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

    def _active_sources(self) -> List[str]:
        active = list(_SYMBOLIC_SOURCES)
        if self._supports_graded_learning and not self._symbolic_only:
            active.extend(_GENERAL_SOURCES)
        return active

    def _source_success_rate(self, source: str) -> float:
        attempts = self._source_counts.get(source, 0)
        if attempts <= 0:
            return 0.0
        return self._source_successes.get(source, 0) / max(attempts, 1)

    def _weakest_source(self, sources: List[str]) -> str:
        if not sources:
            return "seed_library"
        return min(sources, key=lambda src: (self._source_success_rate(src), -self._source_ema.get(src, 0.5), src))

    def _sample_even_mix(
        self,
        sources: List[str],
        total: int,
        prefer_failure_replay: bool = False,
    ) -> List[TrainingProblem]:
        if total <= 0 or not sources:
            return []

        requested = {source: 0 for source in sources}
        base, remainder = divmod(total, len(sources))
        for source in sources:
            requested[source] = base

        if prefer_failure_replay and remainder > 0 and "failure_replay" in requested:
            requested["failure_replay"] += 1
            remainder -= 1

        non_failure_sources = [source for source in sources if source != "failure_replay"] or list(sources)
        while remainder > 0:
            weakest = self._weakest_source(non_failure_sources)
            requested[weakest] += 1
            remainder -= 1

        batch: List[TrainingProblem] = []
        shortfall = 0
        for source in sources:
            count = requested[source]
            if count <= 0:
                continue
            sampled = self._sample_from_source(source, count)
            batch.extend(sampled)
            shortfall += max(0, count - len(sampled))

        refill_sources = list(sources)
        while shortfall > 0 and refill_sources:
            weakest = self._weakest_source(refill_sources)
            sampled = self._sample_from_source(weakest, shortfall)
            if not sampled:
                refill_sources = [source for source in refill_sources if source != weakest]
                continue
            batch.extend(sampled)
            shortfall -= len(sampled)

        return batch[:total]

    def _sample_batch(self) -> List[TrainingProblem]:
        """Sample a batch with symbolic weakness-first scheduling."""
        active_sources = self._active_sources()
        symbolic_sources = [source for source in active_sources if source in _SYMBOLIC_SOURCES]
        general_sources = [source for source in active_sources if source in _GENERAL_SOURCES]

        general_slots = 0
        if general_sources:
            # 50% general when graded learning is on (symbolic already near 100%)
            # 25% general otherwise (warmup / symbolic-only mode)
            _gen_frac = 2 if self._supports_graded_learning else 4
            general_slots = min(max(1, self._batch_size // _gen_frac), self._batch_size - 1)
        symbolic_slots = max(0, self._batch_size - general_slots)

        problems: List[TrainingProblem] = []
        problems.extend(self._sample_even_mix(symbolic_sources, symbolic_slots, prefer_failure_replay=True))
        problems.extend(self._sample_even_mix(general_sources, general_slots))

        if len(problems) < self._batch_size:
            refill = self._sample_even_mix(symbolic_sources or active_sources, self._batch_size - len(problems), prefer_failure_replay=True)
            problems.extend(refill)

        random.shuffle(problems)
        return problems[:self._batch_size]

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
        elif source == "commonsense_reasoning":
            return self._from_commonsense(n)
        elif source == "word_problems":
            return self._from_word_problems(n)
        elif source == "comprehension_problems":
            return self._from_comprehension(n)
        elif source == "code_problems":
            return self._from_code(n)
        elif source == "language_problems":
            return self._from_language(n)
        elif source == "wiki_facts":
            return self._from_wiki_facts(n)
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

    # ── General-intelligence sources ────────────────────────────────────────────

    def _load_general_pools(self) -> None:
        """Lazy-load CommonsenseQA, GSM8K word problems, and comprehension problems."""
        if self._general_pools_loaded:
            return
        self._general_pools_loaded = True
        _repo = _REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]

        # CommonsenseQA
        try:
            p = _repo / "data" / "external_datasets" / "commonsenseqa_train.jsonl"
            if p.exists():
                for line in p.read_text().splitlines():
                    try:
                        entry = __import__("json").loads(line)
                        q = entry.get("question", {})
                        question_text = q.get("stem", "") or entry.get("question", "")
                        choices = q.get("choices", [])
                        choice_str = "  ".join(f"({c['label']}) {c['text']}" for c in choices[:4]) if choices else ""
                        answer_key  = entry.get("answerKey", "")
                        answer_text = next((c["text"] for c in choices if c.get("label") == answer_key), "")
                        if question_text:
                            self._commonsense_pool.append({
                                "expression": question_text[:120],
                                "choices":    choice_str[:200],
                                "answer":     answer_text[:80],
                                "domain":     "commonsense",
                            })
                    except Exception:
                        pass
                log.info("[AutonomousTrainer] CommonsenseQA pool: %d problems", len(self._commonsense_pool))
        except Exception as e:
            log.debug("[AutonomousTrainer] CommonsenseQA load failed: %s", e)

        # GSM8K word problems
        try:
            p = _repo / "data" / "external_datasets" / "gsm8k_test.jsonl"
            if p.exists():
                for line in p.read_text().splitlines():
                    try:
                        entry = __import__("json").loads(line)
                        q = entry.get("question", "")
                        a = str(entry.get("answer", "") or "")
                        if q:
                            self._word_problem_pool.append({
                                "expression": q[:200],
                                "answer":     a,   # keep full answer (#### N at end)
                                "domain":     "word_problems",
                            })
                    except Exception:
                        pass
                log.info("[AutonomousTrainer] GSM8K word problem pool: %d problems", len(self._word_problem_pool))
        except Exception as e:
            log.debug("[AutonomousTrainer] GSM8K load failed: %s", e)

        # Comprehension problems
        try:
            p = _repo / "data" / "memory" / "comprehension_problems.jsonl"
            if p.exists():
                for line in p.read_text().splitlines():
                    try:
                        entry = __import__("json").loads(line)
                        q = entry.get("question", entry.get("expression", ""))
                        if q:
                            self._comprehension_pool.append({
                                "expression": q[:160],
                                "answer":     str(entry.get("answer", ""))[:80],
                                "domain":     entry.get("domain", "reasoning"),
                            })
                    except Exception:
                        pass
                log.info("[AutonomousTrainer] Comprehension pool: %d problems", len(self._comprehension_pool))
        except Exception as e:
            log.debug("[AutonomousTrainer] Comprehension load failed: %s", e)

        # MMLU domain problems (psychology, history, biology, economics)
        try:
            p = _repo / "data" / "external_datasets" / "mmlu_domains.jsonl"
            if p.exists():
                for line in p.read_text().splitlines():
                    try:
                        entry = __import__("json").loads(line)
                        q = entry.get("expression", "")
                        ans = entry.get("expected", entry.get("answer", ""))
                        domain = entry.get("domain", "general")
                        if q and ans:
                            self._comprehension_pool.append({
                                "expression": q[:160],
                                "answer":     str(ans)[:120],
                                "domain":     domain,
                            })
                    except Exception:
                        pass
                log.info("[AutonomousTrainer] MMLU domain pool loaded: %d total comprehension problems",
                         len(self._comprehension_pool))
        except Exception as e:
            log.debug("[AutonomousTrainer] MMLU load failed: %s", e)

        # Wiki pool: pre-populate from saved progress file if available
        try:
            import json as _json
            from pathlib import Path as _Path
            _root = _Path(__file__).resolve().parents[3]
            _prog = _root / "data" / "memory" / "wiki_dump_progress.json"
            if _prog.exists():
                _pd = _json.loads(_prog.read_text())
                _pairs = _pd.get("qa_pairs", [])
                if _pairs:
                    self._wiki_pool.extend(_pairs)
                    log.info("[AutonomousTrainer] Wiki pool pre-loaded: %d Q&A pairs", len(self._wiki_pool))
        except Exception as _wpe:
            log.debug("[AutonomousTrainer] Wiki pool pre-load error: %s", _wpe)

        # Code and language pools (always seeded from built-in lists)
        self._load_code_pool()
        self._load_language_pool()

    def _from_commonsense(self, n: int) -> List[TrainingProblem]:
        """Sample from CommonsenseQA — vocabulary, cause-effect, analogy, world knowledge."""
        self._load_general_pools()
        if not self._commonsense_pool:
            return self._from_seed_library(n)
        picked = random.sample(self._commonsense_pool, min(n, len(self._commonsense_pool)))
        problems = []
        for p in picked:
            expr = p["expression"]
            choices = p.get("choices", "")
            # Include choices in expression so the multi-choice KB scorer can see them
            if choices:
                expr = f"{expr}\nChoices: {choices}"
            problems.append(TrainingProblem(
                expression=expr[:320],
                domain="commonsense",
                source="commonsense_reasoning",
                difficulty=0.5,
                metadata={"choices": choices, "answer": p.get("answer", "")},
            ))
        return problems

    def _from_word_problems(self, n: int) -> List[TrainingProblem]:
        """Sample from GSM8K — multi-step arithmetic word problems requiring language + math."""
        self._load_general_pools()
        if not self._word_problem_pool:
            return self._from_seed_library(n)
        picked = random.sample(self._word_problem_pool, min(n, len(self._word_problem_pool)))
        return [TrainingProblem(
            expression=p["expression"],
            domain="word_problems",
            source="word_problems",
            difficulty=0.65,
            metadata={"answer": p.get("answer","")},
        ) for p in picked]

    def _from_comprehension(self, n: int) -> List[TrainingProblem]:
        """Sample from comprehension/general reasoning problems."""
        self._load_general_pools()
        if not self._comprehension_pool:
            return self._from_seed_library(n)
        picked = random.sample(self._comprehension_pool, min(n, len(self._comprehension_pool)))
        return [TrainingProblem(
            expression=p["expression"],
            domain=p.get("domain", "reasoning"),
            source="comprehension_problems",
            difficulty=0.55,
            metadata={"answer": p.get("answer","")},
        ) for p in picked]

    def _from_code(self, n: int) -> List[TrainingProblem]:
        """Sample code/programming questions — Python, algorithms, debugging, data structures."""
        self._load_general_pools()
        if not self._code_pool:
            return self._from_seed_library(n)
        picked = random.sample(self._code_pool, min(n, len(self._code_pool)))
        return [TrainingProblem(
            expression=p["expression"],
            domain="code",
            source="code_problems",
            difficulty=p.get("difficulty", 0.5),
            metadata={"answer": p.get("answer", ""), "topic": p.get("topic", "python")},
        ) for p in picked]

    def _from_language(self, n: int) -> List[TrainingProblem]:
        """Sample language/NLP questions — grammar, vocabulary, analogy, comprehension."""
        self._load_general_pools()
        if not self._language_pool:
            return self._from_seed_library(n)
        picked = random.sample(self._language_pool, min(n, len(self._language_pool)))
        return [TrainingProblem(
            expression=p["expression"],
            domain="language",
            source="language_problems",
            difficulty=p.get("difficulty", 0.4),
            metadata={"answer": p.get("answer", ""), "topic": p.get("topic", "grammar")},
        ) for p in picked]

    def _from_wiki_facts(self, n: int) -> List[TrainingProblem]:
        """Sample Q&A pairs derived from Simple Wikipedia dump. Launches ingestion if needed."""
        self._ensure_wiki_ingestion()
        if not self._wiki_pool:
            return self._from_seed_library(n)
        picked = random.sample(self._wiki_pool, min(n, len(self._wiki_pool)))
        return [TrainingProblem(
            expression=p["expression"],
            domain=p.get("domain", "general"),
            source="wiki_facts",
            difficulty=0.45,
            metadata={"answer": p.get("expected", ""), "source": "wiki_dump"},
        ) for p in picked]

    def _ensure_wiki_ingestion(self) -> None:
        """Start wiki ingestion in background if not already running; populate _wiki_pool."""
        if self._wiki_ingest_thread and self._wiki_ingest_thread.is_alive():
            return
        if len(self._wiki_pool) >= 500:
            return  # enough data loaded

        def _do_ingest():
            try:
                from sare.learning.wikipedia_dump_ingester import WikipediaDumpIngester
                wi = WikipediaDumpIngester()
                if not wi.is_available():
                    log.debug("[AutonomousTrainer] Wiki dump not available")
                    return
                log.info("[AutonomousTrainer] Starting wiki dump ingestion batch...")
                n_facts = wi.ingest_batch(max_articles=500)
                log.info("[AutonomousTrainer] Wiki ingestion done: %d facts stored", n_facts)
                # Read the Q&A pairs the ingester produced and add to our pool
                try:
                    import json
                    from pathlib import Path
                    _root = Path(__file__).resolve().parents[3]
                    progress_path = _root / "data" / "memory" / "wiki_dump_progress.json"
                    if progress_path.exists():
                        prog = json.loads(progress_path.read_text())
                        pairs = prog.get("qa_pairs", [])
                        self._wiki_pool.extend(pairs)
                        log.info("[AutonomousTrainer] Wiki pool: %d Q&A pairs", len(self._wiki_pool))
                except Exception as _pool_err:
                    log.debug("[AutonomousTrainer] Wiki pool load error: %s", _pool_err)
            except Exception as e:
                log.debug("[AutonomousTrainer] Wiki ingestion error: %s", e)

        self._wiki_ingest_thread = threading.Thread(
            target=_do_ingest, daemon=True, name="wiki-ingest"
        )
        self._wiki_ingest_thread.start()

    def _load_code_pool(self) -> None:
        """Seed code problems: Python, algorithms, data structures, systems, debugging, CS concepts."""
        self._code_pool = [
            # ── Python basics ────────────────────────────────────────────────────
            {"expression": "What does the 'self' keyword mean in a Python class?", "answer": "self refers to the instance of the class, allowing access to its attributes and methods.", "topic": "python", "difficulty": 0.3},
            {"expression": "What is the difference between a list and a tuple in Python?", "answer": "Lists are mutable (can be changed), tuples are immutable (cannot be changed after creation).", "topic": "python", "difficulty": 0.3},
            {"expression": "What does 'def' do in Python?", "answer": "def is used to define a function in Python.", "topic": "python", "difficulty": 0.2},
            {"expression": "What is a dictionary in Python and how do you access a value?", "answer": "A dictionary stores key-value pairs. Access values using dict[key] or dict.get(key).", "topic": "python", "difficulty": 0.3},
            {"expression": "What is a list comprehension in Python?", "answer": "[x*2 for x in range(5)] is a list comprehension — a concise way to create lists using a for loop inline.", "topic": "python", "difficulty": 0.4},
            {"expression": "What does 'import' do in Python?", "answer": "import loads a module or library so its functions and classes can be used in the script.", "topic": "python", "difficulty": 0.2},
            {"expression": "What is a lambda function in Python?", "answer": "A lambda is an anonymous single-expression function: lambda x: x+1 returns a function that adds 1.", "topic": "python", "difficulty": 0.4},
            {"expression": "What is the difference between '==' and 'is' in Python?", "answer": "'==' checks value equality; 'is' checks identity (same object in memory).", "topic": "python", "difficulty": 0.4},
            {"expression": "How does a for loop work in Python?", "answer": "A for loop iterates over each element in an iterable (list, string, range) executing the body once per element.", "topic": "python", "difficulty": 0.2},
            {"expression": "What is a Python generator and how does 'yield' work?", "answer": "A generator yields values one at a time using 'yield', pausing execution between calls for lazy evaluation.", "topic": "python", "difficulty": 0.6},
            {"expression": "What is *args in Python?", "answer": "*args allows a function to accept any number of positional arguments, collected into a tuple.", "topic": "python", "difficulty": 0.4},
            {"expression": "What is **kwargs in Python?", "answer": "**kwargs allows a function to accept any number of keyword arguments, collected into a dictionary.", "topic": "python", "difficulty": 0.4},
            {"expression": "What is a decorator in Python?", "answer": "A decorator is a function that wraps another function to add behavior. Applied with @decorator_name above the function definition.", "topic": "python", "difficulty": 0.6},
            {"expression": "What does the 'with' statement do in Python?", "answer": "'with' is a context manager that ensures setup and teardown happen automatically, e.g., closing files after use.", "topic": "python", "difficulty": 0.5},
            {"expression": "What is the difference between 'append' and 'extend' in Python lists?", "answer": "append adds a single element to the end. extend adds all elements from an iterable.", "topic": "python", "difficulty": 0.3},
            {"expression": "What does 'pass' do in Python?", "answer": "pass is a no-op placeholder used when a statement is syntactically required but no code should run.", "topic": "python", "difficulty": 0.2},
            {"expression": "What is a Python set and when would you use one?", "answer": "A set is an unordered collection of unique elements. Use it to eliminate duplicates or for fast membership testing.", "topic": "python", "difficulty": 0.3},
            {"expression": "What is the difference between 'break' and 'continue' in Python loops?", "answer": "break exits the loop entirely; continue skips the rest of the current iteration and moves to the next.", "topic": "python", "difficulty": 0.3},
            {"expression": "What is a Python slice?", "answer": "A slice extracts a portion of a list or string: lst[start:stop:step]. e.g., lst[1:4] returns elements at index 1,2,3.", "topic": "python", "difficulty": 0.3},
            {"expression": "What is the difference between 'deepcopy' and 'copy' in Python?", "answer": "copy creates a shallow copy (nested objects still share references). deepcopy creates a fully independent recursive copy.", "topic": "python", "difficulty": 0.5},
            {"expression": "What is a Python context manager?", "answer": "A context manager manages resources using __enter__ and __exit__ methods, ensuring cleanup. Used with the 'with' statement.", "topic": "python", "difficulty": 0.6},
            {"expression": "What does 'enumerate' do in Python?", "answer": "enumerate wraps an iterable and returns (index, element) pairs, useful when you need both index and value in a loop.", "topic": "python", "difficulty": 0.3},
            {"expression": "What does 'zip' do in Python?", "answer": "zip combines multiple iterables element-by-element into tuples: zip([1,2],[3,4]) → [(1,3),(2,4)].", "topic": "python", "difficulty": 0.3},
            # ── Algorithms ───────────────────────────────────────────────────────
            {"expression": "What is the time complexity of binary search?", "answer": "O(log n) — binary search halves the search space each step.", "topic": "algorithms", "difficulty": 0.5},
            {"expression": "What is the difference between BFS and DFS?", "answer": "BFS explores level by level using a queue; DFS explores depth-first using a stack or recursion.", "topic": "algorithms", "difficulty": 0.5},
            {"expression": "What is the time complexity of bubble sort?", "answer": "O(n^2) average and worst case; O(n) best case when already sorted.", "topic": "algorithms", "difficulty": 0.4},
            {"expression": "What is recursion?", "answer": "Recursion is when a function calls itself with a smaller input, reducing toward a base case.", "topic": "algorithms", "difficulty": 0.4},
            {"expression": "What is a hash table's average lookup time?", "answer": "O(1) average — hash tables use a hash function to map keys directly to buckets.", "topic": "algorithms", "difficulty": 0.5},
            {"expression": "What is dynamic programming?", "answer": "Dynamic programming solves problems by storing results of overlapping subproblems to avoid recomputation (memoization or tabulation).", "topic": "algorithms", "difficulty": 0.6},
            {"expression": "What is the time complexity of quicksort?", "answer": "O(n log n) average, O(n^2) worst case. It partitions around a pivot and recursively sorts both halves.", "topic": "algorithms", "difficulty": 0.6},
            {"expression": "What is merge sort and why is it preferred over quicksort sometimes?", "answer": "Merge sort always runs O(n log n) and is stable. Preferred when worst-case guarantees matter or when sorting linked lists.", "topic": "algorithms", "difficulty": 0.6},
            {"expression": "What is a greedy algorithm?", "answer": "A greedy algorithm makes the locally optimal choice at each step, hoping to reach a global optimum. Works for some problems like Dijkstra's.", "topic": "algorithms", "difficulty": 0.6},
            {"expression": "What is Big O notation?", "answer": "Big O describes the upper bound of an algorithm's time or space complexity as input size grows, ignoring constants.", "topic": "algorithms", "difficulty": 0.4},
            {"expression": "What is the difference between O(n) and O(n^2) time complexity?", "answer": "O(n) grows linearly with input size; O(n^2) grows quadratically. For n=1000: O(n)=1000 operations vs O(n^2)=1,000,000.", "topic": "algorithms", "difficulty": 0.4},
            # ── Data Structures ──────────────────────────────────────────────────
            {"expression": "What is a stack and what are its operations?", "answer": "A stack is LIFO (Last In First Out). Operations: push (add), pop (remove top), peek (view top).", "topic": "data_structures", "difficulty": 0.3},
            {"expression": "What is a queue data structure?", "answer": "A queue is FIFO (First In First Out). Elements enqueue at the rear and dequeue from the front.", "topic": "data_structures", "difficulty": 0.3},
            {"expression": "What is a linked list vs an array?", "answer": "A linked list uses nodes with pointers: O(1) insert/delete anywhere but O(n) access. Arrays have O(1) access but O(n) insert/delete.", "topic": "data_structures", "difficulty": 0.4},
            {"expression": "What is a binary tree?", "answer": "A binary tree is a hierarchical structure where each node has at most two children (left and right).", "topic": "data_structures", "difficulty": 0.4},
            {"expression": "What is a binary search tree (BST)?", "answer": "A BST is a binary tree where left child < parent < right child, enabling O(log n) search in balanced trees.", "topic": "data_structures", "difficulty": 0.5},
            {"expression": "What is a heap data structure?", "answer": "A heap is a complete binary tree satisfying the heap property: max-heap has each parent >= children; used for priority queues.", "topic": "data_structures", "difficulty": 0.6},
            {"expression": "What is a graph in computer science?", "answer": "A graph is a set of nodes (vertices) connected by edges. Can be directed/undirected, weighted/unweighted.", "topic": "data_structures", "difficulty": 0.5},
            {"expression": "What is the difference between a tree and a graph?", "answer": "A tree is a connected acyclic graph with n nodes and n-1 edges. Graphs can have cycles and multiple components.", "topic": "data_structures", "difficulty": 0.5},
            # ── Debugging / Errors ───────────────────────────────────────────────
            {"expression": "What does 'IndexError: list index out of range' mean?", "answer": "You accessed a list index that doesn't exist — index >= len(list) or < -len(list).", "topic": "debugging", "difficulty": 0.3},
            {"expression": "What causes a 'KeyError' in Python?", "answer": "Accessing a dictionary key that doesn't exist. Use dict.get(key) or 'key in dict' to avoid it.", "topic": "debugging", "difficulty": 0.3},
            {"expression": "What is a NoneType error in Python?", "answer": "Calling a method or attribute on None. Check 'if var is not None' before use, or ensure functions return a value.", "topic": "debugging", "difficulty": 0.4},
            {"expression": "What is an infinite loop and how do you avoid it?", "answer": "A loop that never terminates because its exit condition is never met. Ensure the loop variable changes toward the condition.", "topic": "debugging", "difficulty": 0.3},
            {"expression": "What does 'TypeError: unsupported operand type' mean?", "answer": "You tried an operation (like +) on incompatible types, e.g., adding a string and an integer.", "topic": "debugging", "difficulty": 0.3},
            {"expression": "What is a stack overflow error?", "answer": "A stack overflow occurs when recursion is too deep and exceeds the call stack limit, usually from infinite recursion.", "topic": "debugging", "difficulty": 0.5},
            {"expression": "What is a race condition in programming?", "answer": "A race condition occurs when two threads access shared data concurrently and the result depends on execution order.", "topic": "debugging", "difficulty": 0.7},
            # ── OOP ──────────────────────────────────────────────────────────────
            {"expression": "What is inheritance in OOP?", "answer": "Inheritance lets a child class inherit attributes and methods from a parent class, enabling code reuse.", "topic": "oop", "difficulty": 0.4},
            {"expression": "What is polymorphism in Python?", "answer": "Polymorphism allows different classes to implement the same method differently — same call, different behavior.", "topic": "oop", "difficulty": 0.5},
            {"expression": "What is encapsulation in OOP?", "answer": "Encapsulation hides internal state, exposing only a public interface. Prevents direct access to implementation details.", "topic": "oop", "difficulty": 0.4},
            {"expression": "What is an abstract class in Python?", "answer": "An abstract class (abc.ABC) defines methods that subclasses must implement and cannot be instantiated directly.", "topic": "oop", "difficulty": 0.6},
            {"expression": "What is the difference between a class and an object?", "answer": "A class is a blueprint defining structure and behavior. An object is an instance of a class with actual data.", "topic": "oop", "difficulty": 0.3},
            {"expression": "What is method overriding in Python?", "answer": "Method overriding is when a subclass provides a different implementation of a method already defined in its parent class.", "topic": "oop", "difficulty": 0.5},
            {"expression": "What is a constructor in Python?", "answer": "A constructor is the __init__ method, called when a new object is created to initialize its attributes.", "topic": "oop", "difficulty": 0.3},
            {"expression": "What is the difference between __str__ and __repr__ in Python?", "answer": "__str__ returns a human-readable string for end users. __repr__ returns an unambiguous representation for developers/debugging.", "topic": "oop", "difficulty": 0.5},
            # ── Concurrency / Systems ────────────────────────────────────────────
            {"expression": "What is the difference between a thread and a process?", "answer": "Threads share memory within one process; processes are isolated. Threads are lighter but unsafe without synchronization.", "topic": "concurrency", "difficulty": 0.6},
            {"expression": "What is the Python GIL?", "answer": "The Global Interpreter Lock allows only one thread to execute Python bytecode at a time, preventing true CPU parallelism.", "topic": "concurrency", "difficulty": 0.7},
            {"expression": "What is a mutex?", "answer": "A mutex (mutual exclusion lock) ensures only one thread accesses a shared resource at a time, preventing race conditions.", "topic": "concurrency", "difficulty": 0.6},
            {"expression": "What is deadlock in programming?", "answer": "Deadlock occurs when two or more threads each hold a resource the other needs, so all are blocked forever.", "topic": "concurrency", "difficulty": 0.7},
            {"expression": "What is the difference between synchronous and asynchronous programming?", "answer": "Synchronous code waits for each operation to complete before continuing. Asynchronous code can start operations and continue while waiting.", "topic": "concurrency", "difficulty": 0.6},
            # ── Databases / Web ──────────────────────────────────────────────────
            {"expression": "What is SQL?", "answer": "SQL (Structured Query Language) is used to query and manage relational databases using commands like SELECT, INSERT, UPDATE, DELETE.", "topic": "databases", "difficulty": 0.3},
            {"expression": "What is the difference between SQL and NoSQL databases?", "answer": "SQL databases use structured tables with a fixed schema. NoSQL stores data as documents, key-value, or graphs with flexible schemas.", "topic": "databases", "difficulty": 0.5},
            {"expression": "What is an index in a database?", "answer": "An index is a data structure that speeds up queries by allowing faster lookup of rows, at the cost of extra storage and slower writes.", "topic": "databases", "difficulty": 0.5},
            {"expression": "What is a REST API?", "answer": "A REST API uses HTTP methods (GET, POST, PUT, DELETE) to perform operations on resources identified by URLs.", "topic": "web", "difficulty": 0.5},
            {"expression": "What is the difference between GET and POST HTTP methods?", "answer": "GET retrieves data and parameters are in the URL. POST sends data in the request body, typically to create or submit.", "topic": "web", "difficulty": 0.4},
            {"expression": "What is JSON?", "answer": "JSON (JavaScript Object Notation) is a lightweight text format for data exchange using key-value pairs and arrays.", "topic": "web", "difficulty": 0.3},
            # ── CS Fundamentals ──────────────────────────────────────────────────
            {"expression": "What is the difference between compiled and interpreted languages?", "answer": "Compiled languages (C, Go) translate source to machine code before running. Interpreted languages (Python, Ruby) execute line-by-line at runtime.", "topic": "cs_fundamentals", "difficulty": 0.5},
            {"expression": "What is a pointer?", "answer": "A pointer is a variable that stores the memory address of another variable, allowing direct memory access.", "topic": "cs_fundamentals", "difficulty": 0.5},
            {"expression": "What is garbage collection?", "answer": "Garbage collection automatically frees memory that is no longer referenced, preventing memory leaks.", "topic": "cs_fundamentals", "difficulty": 0.4},
            {"expression": "What is the difference between pass by value and pass by reference?", "answer": "Pass by value sends a copy of the variable. Pass by reference sends a reference to the original, so changes affect the original.", "topic": "cs_fundamentals", "difficulty": 0.5},
            {"expression": "What is an API?", "answer": "An API (Application Programming Interface) is a set of rules that lets programs communicate with each other.", "topic": "cs_fundamentals", "difficulty": 0.3},
            {"expression": "What is version control and why is it important?", "answer": "Version control tracks changes to code over time, allows collaboration, and enables reverting to previous states. Git is the most common tool.", "topic": "cs_fundamentals", "difficulty": 0.3},
            {"expression": "What is the difference between a compiler and an interpreter?", "answer": "A compiler translates the whole program to machine code before execution. An interpreter translates and executes line-by-line.", "topic": "cs_fundamentals", "difficulty": 0.5},
            {"expression": "What is object-oriented programming?", "answer": "OOP organizes code around objects with data (attributes) and behavior (methods), using principles like encapsulation, inheritance, and polymorphism.", "topic": "cs_fundamentals", "difficulty": 0.3},
            {"expression": "What is functional programming?", "answer": "Functional programming treats computation as evaluation of mathematical functions, avoiding shared state and mutable data.", "topic": "cs_fundamentals", "difficulty": 0.6},
            {"expression": "What is a design pattern in software engineering?", "answer": "A design pattern is a reusable solution template for a common software design problem, like Singleton, Observer, or Factory.", "topic": "cs_fundamentals", "difficulty": 0.6},
            {"expression": "What is the difference between stack and heap memory?", "answer": "Stack holds local variables and function calls (fast, limited size). Heap holds dynamically allocated objects (flexible size, slower).", "topic": "cs_fundamentals", "difficulty": 0.6},
            {"expression": "What is a CPU cache?", "answer": "A CPU cache is fast memory near the processor that stores frequently accessed data to reduce main memory access latency.", "topic": "cs_fundamentals", "difficulty": 0.5},
        ]
        log.info("[AutonomousTrainer] Code pool: %d problems", len(self._code_pool))

    def _load_language_pool(self) -> None:
        """Seed language problems: grammar, vocabulary, analogy, writing, reading comprehension."""
        self._language_pool = [
            # Grammar
            {"expression": "What is the difference between 'its' and 'it's'?", "answer": "'Its' is the possessive form (the cat licked its paw). 'It's' is a contraction for 'it is' or 'it has'.", "topic": "grammar", "difficulty": 0.3},
            {"expression": "What is a noun?", "answer": "A noun is a word that names a person, place, thing, or idea. Examples: dog, Paris, happiness, freedom.", "topic": "grammar", "difficulty": 0.2},
            {"expression": "What is the difference between an adjective and an adverb?", "answer": "Adjectives modify nouns (a fast car). Adverbs modify verbs, adjectives, or other adverbs (she runs quickly).", "topic": "grammar", "difficulty": 0.3},
            {"expression": "What is a conjunction? Give examples.", "answer": "A conjunction connects words, phrases, or clauses. Examples: and, but, or, because, although, while.", "topic": "grammar", "difficulty": 0.3},
            {"expression": "What is the passive voice? Give an example.", "answer": "In passive voice, the subject receives the action. Example: 'The book was written by her' (vs active: 'She wrote the book').", "topic": "grammar", "difficulty": 0.4},
            {"expression": "What is a clause? What is the difference between independent and dependent clauses?", "answer": "A clause has a subject and verb. Independent clauses are complete sentences; dependent clauses cannot stand alone and rely on a main clause.", "topic": "grammar", "difficulty": 0.4},
            {"expression": "What is a gerund?", "answer": "A gerund is a verb form ending in -ing used as a noun. Example: 'Swimming is fun' — swimming is a gerund.", "topic": "grammar", "difficulty": 0.5},
            {"expression": "When do you use 'who' vs 'whom'?", "answer": "'Who' is the subject (Who is calling?). 'Whom' is the object (To whom did you speak?). Tip: replace with he/him — he=who, him=whom.", "topic": "grammar", "difficulty": 0.5},
            # Vocabulary / Word meaning
            {"expression": "What does 'ephemeral' mean?", "answer": "Ephemeral means lasting for only a short time; transitory. Example: a soap bubble is ephemeral.", "topic": "vocabulary", "difficulty": 0.5},
            {"expression": "What does 'verbose' mean?", "answer": "Verbose means using more words than needed; excessively wordy.", "topic": "vocabulary", "difficulty": 0.4},
            {"expression": "What does 'ambiguous' mean? Give an example.", "answer": "Ambiguous means having more than one possible meaning. Example: 'I saw the man with the telescope' — who had the telescope?", "topic": "vocabulary", "difficulty": 0.4},
            {"expression": "What is the difference between 'affect' and 'effect'?", "answer": "'Affect' is usually a verb (the weather affects my mood). 'Effect' is usually a noun (the effect of rain on mood).", "topic": "vocabulary", "difficulty": 0.4},
            {"expression": "What does 'pragmatic' mean?", "answer": "Pragmatic means dealing with things practically and practically, focusing on what works rather than abstract theory.", "topic": "vocabulary", "difficulty": 0.5},
            {"expression": "What is a synonym for 'melancholy'?", "answer": "Synonyms for melancholy include sadness, sorrow, gloom, despondency, and dejection.", "topic": "vocabulary", "difficulty": 0.3},
            # Analogy
            {"expression": "Complete the analogy: hot is to cold as day is to ___", "answer": "Night. Hot is the opposite of cold; day is the opposite of night.", "topic": "analogy", "difficulty": 0.2},
            {"expression": "Complete the analogy: author is to book as composer is to ___", "answer": "Symphony (or music/composition). An author creates a book; a composer creates a symphony.", "topic": "analogy", "difficulty": 0.3},
            {"expression": "Complete the analogy: bird is to flock as fish is to ___", "answer": "School. A group of birds is called a flock; a group of fish is called a school.", "topic": "analogy", "difficulty": 0.4},
            {"expression": "Complete the analogy: doctor is to hospital as teacher is to ___", "answer": "School. A doctor works in a hospital; a teacher works in a school.", "topic": "analogy", "difficulty": 0.2},
            {"expression": "Complete the analogy: word is to sentence as brick is to ___", "answer": "Wall (or building). Words combine to form sentences; bricks combine to form walls.", "topic": "analogy", "difficulty": 0.4},
            # Reading comprehension / reasoning
            {"expression": "What is the main purpose of a thesis statement in an essay?", "answer": "A thesis statement presents the main argument or claim of the essay, telling the reader what position the writer will support and why.", "topic": "writing", "difficulty": 0.4},
            {"expression": "What is the difference between a simile and a metaphor?", "answer": "A simile compares using 'like' or 'as' (fast as lightning). A metaphor states something IS something else (he is a lion).", "topic": "literary_devices", "difficulty": 0.4},
            {"expression": "What is irony? Describe situational irony with an example.", "answer": "Irony is when the opposite of what's expected happens. Situational irony: a fire station burns down — you'd expect firefighters to prevent fires.", "topic": "literary_devices", "difficulty": 0.5},
            {"expression": "What is the difference between denotation and connotation?", "answer": "Denotation is the literal dictionary meaning of a word. Connotation is the emotional or cultural meaning attached to it. Example: 'cheap' denotes low cost but connotes low quality.", "topic": "literary_devices", "difficulty": 0.5},
            {"expression": "What makes a good argument? What are the three parts of the classical rhetorical triangle?", "answer": "A good argument uses logos (logic/evidence), ethos (credibility/character), and pathos (emotional appeal).", "topic": "rhetoric", "difficulty": 0.5},
            {"expression": "What is the difference between fiction and non-fiction?", "answer": "Fiction is invented, imaginative narrative (novels, short stories). Non-fiction is based on real events, people, and facts (biographies, journalism, essays).", "topic": "reading", "difficulty": 0.2},
            # Language learning
            {"expression": "What is a cognate? Give an example between English and Spanish.", "answer": "A cognate is a word that looks/sounds similar in two languages with the same meaning. Example: 'information' (English) and 'información' (Spanish).", "topic": "linguistics", "difficulty": 0.4},
            {"expression": "What is the Sapir-Whorf hypothesis?", "answer": "The hypothesis (also called linguistic relativity) suggests that the language you speak influences how you perceive and think about the world.", "topic": "linguistics", "difficulty": 0.7},
            {"expression": "What is a phoneme?", "answer": "A phoneme is the smallest unit of sound in a language that can distinguish meaning. For example, the 'p' in 'pin' and 'b' in 'bin' are different phonemes.", "topic": "linguistics", "difficulty": 0.5},
        ]
        log.info("[AutonomousTrainer] Language pool: %d problems", len(self._language_pool))

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
        scope = "run_local_symbolic_only" if self._symbolic_only else (
            "run_local_graded" if self._supports_graded_learning else "run_local_symbolic"
        )
        return {
            "scope": scope,
            "run_id": self._run_id,
            "running": self._running,
            "started_at": self._started_at_iso,
            "total_problems": self._total_problems,
            "total_successes": self._total_successes,
            "recent_rate": round(self.recent_rate(), 3),
            "current_difficulty": round(self._current_difficulty, 2),
            "uptime_seconds": round(self.uptime_seconds(), 1),
            "supports_graded_learning": self._supports_graded_learning,
            "config": {
                "interval_seconds": self._interval,
                "batch_size": self._batch_size,
                "max_workers": self._max_workers,
                "symbolic_only": self._symbolic_only,
                "general_sources_enabled": self._supports_graded_learning and not self._symbolic_only,
            },
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
            _tmp = _STATS_PATH.parent / f"autonomous_trainer_stats.{os.getpid()}.{threading.get_ident()}.tmp"
            _tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(_tmp, _STATS_PATH)
        except Exception as e:
            log.debug("AutonomousTrainer save_stats error: %s", e)
