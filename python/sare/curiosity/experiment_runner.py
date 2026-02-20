"""
ExperimentRunner — closes the curiosity loop.

Autonomously picks pending problems from CurriculumGenerator,
solves them with BeamSearch, passes successes through ReflectionEngine
→ CausalInduction → ConceptRegistry.

This is the heartbeat of SARE's self-learning:
  Generate → Attempt → Reflect → Induce → Learn → Repeat
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from sare.curiosity.curriculum_generator import CurriculumGenerator, GeneratedProblem

log = logging.getLogger(__name__)

# ── Types ──────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    problem_id: str
    solved: bool
    energy_before: float = 0.0
    energy_after: float  = 0.0
    rule_name: str       = ""        # Rule extracted (if any)
    rule_promoted: bool  = False     # Did CausalInduction promote it?
    elapsed_ms: float    = 0.0
    reasoning: str       = ""        # CausalInduction verdict


# ── ExperimentRunner ───────────────────────────────────────────

class ExperimentRunner:
    """
    Closes the Curiosity → Solve → Reflect loop.

    Usage (blocking batch):
        runner = ExperimentRunner(curriculum_gen, searcher, energy, ...)
        results = runner.run_batch(n=10)

    Usage (background daemon):
        runner.start_daemon(interval_seconds=30)
        runner.stop_daemon()
    """

    def __init__(
        self,
        curriculum_gen: "CurriculumGenerator",
        searcher,
        energy,
        reflection_engine=None,
        causal_induction=None,
        concept_registry=None,
        transforms=None,
        beam_width: int = 8,
        budget_seconds: float = 5.0,
    ):
        self.curriculum_gen    = curriculum_gen
        self.searcher          = searcher
        self.energy            = energy
        self.reflection_engine = reflection_engine
        self.causal_induction  = causal_induction
        self.concept_registry  = concept_registry
        self.transforms        = transforms or []
        self.beam_width        = beam_width
        self.budget_seconds    = budget_seconds

        self._history: List[ExperimentResult] = []
        self._daemon_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── Public: run one problem ────────────────────────────────

    def run_one(self, problem: "GeneratedProblem") -> ExperimentResult:
        t0 = time.time()
        result = ExperimentResult(problem_id=problem.id, solved=False)

        try:
            from sare.engine import EnergyEvaluator
            energy_fn = self.energy if self.energy else EnergyEvaluator()

            # 1. Evaluate initial energy
            e_before = energy_fn.compute(problem.graph).total
            result.energy_before = e_before

            # 2. Run search
            search_result = self.searcher.search(
                problem.graph,
                energy_fn,
                self.transforms,
                beam_width=self.beam_width,
                budget_seconds=self.budget_seconds,
            )
            e_after = search_result.energy.total
            result.energy_after = e_after
            delta = e_before - e_after
            result.solved = delta > 0.01

            log.info(
                "Experiment %s: energy %.2f → %.2f (Δ=%.2f) solved=%s",
                problem.id, e_before, e_after, delta, result.solved,
            )

            if result.solved:
                # Mark the problem solved and add result as new seed
                self.curriculum_gen.mark_solved(problem.id)
                self.curriculum_gen.add_seed(search_result.graph)

                # 3. Reflect: extract candidate rule
                if self.reflection_engine:
                    try:
                        rule = self.reflection_engine.reflect(
                            problem.graph, search_result.graph
                        )
                        if rule and rule.valid():
                            result.rule_name = rule.name

                            # 4. Causal Induction: test rule before accepting
                            if self.causal_induction and self.concept_registry:
                                induction = self.causal_induction.evaluate(
                                    rule, energy_fn
                                )
                                result.rule_promoted = induction.promoted
                                result.reasoning     = induction.reasoning

                                if induction.promoted:
                                    self.concept_registry.add_rule(rule)
                                    log.info(
                                        "Rule '%s' PROMOTED (score=%.2f)",
                                        rule.name, induction.evidence_score,
                                    )
                                else:
                                    log.info(
                                        "Rule '%s' rejected: %s",
                                        rule.name, induction.reasoning,
                                    )
                            elif self.concept_registry:
                                # No causal induction: add directly with prior confidence
                                self.concept_registry.add_rule(rule)
                                result.rule_promoted = True
                    except Exception as exc:
                        log.warning("Reflection failed for %s: %s", problem.id, exc)
            else:
                self.curriculum_gen.mark_stuck(problem.id)

        except Exception as exc:
            log.error("ExperimentRunner error for %s: %s", problem.id, exc)
        finally:
            result.elapsed_ms = (time.time() - t0) * 1000
            self._history.append(result)

        return result

    # ── Public: batch run ─────────────────────────────────────

    def run_batch(self, n: int = 5) -> List[ExperimentResult]:
        """Pick up to `n` pending problems and run them."""
        # Generate new problems if the queue is low
        pending = self.curriculum_gen.pending_problems()
        if len(pending) < n:
            self.curriculum_gen.generate_batch(size=n - len(pending))
            pending = self.curriculum_gen.pending_problems()

        results = []
        for problem in pending[:n]:
            results.append(self.run_one(problem))
        return results

    # ── Public: stats ─────────────────────────────────────────

    @property
    def history(self) -> List[ExperimentResult]:
        return list(self._history)

    def stats(self) -> dict:
        total   = len(self._history)
        solved  = sum(1 for r in self._history if r.solved)
        promoted = sum(1 for r in self._history if r.rule_promoted)
        avg_ms  = (
            sum(r.elapsed_ms for r in self._history) / total if total else 0.0
        )
        return {
            "total":         total,
            "solved":        solved,
            "solve_rate":    solved / total if total else 0.0,
            "rules_promoted": promoted,
            "avg_ms":        round(avg_ms, 1),
        }

    # ── Daemon mode ───────────────────────────────────────────

    def start_daemon(self, interval_seconds: float = 30.0, batch_size: int = 5):
        """Run experiments in a background thread."""
        if self._daemon_thread and self._daemon_thread.is_alive():
            return  # Already running
        self._stop_event.clear()

        def _loop():
            log.info("ExperimentRunner daemon started (interval=%.0fs)", interval_seconds)
            while not self._stop_event.is_set():
                try:
                    self.run_batch(n=batch_size)
                except Exception as exc:
                    log.error("Daemon batch error: %s", exc)
                self._stop_event.wait(interval_seconds)
            log.info("ExperimentRunner daemon stopped.")

        self._daemon_thread = threading.Thread(target=_loop, daemon=True, name="ExperimentRunner")
        self._daemon_thread.start()

    def stop_daemon(self):
        self._stop_event.set()
        if self._daemon_thread:
            self._daemon_thread.join(timeout=5)
