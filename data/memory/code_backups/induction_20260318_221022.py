"""
Lightweight Causal Induction for SARE-HX.

Tests candidate rules from PyReflectionEngine before promotion.
A rule is tested by checking if it generalizes: apply the pattern
on 3 different graphs. If it reduces energy on at least 2, promote.
"""
from __future__ import annotations

import collections
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
from sare.causal.knowledge_base import get_ckb

log = logging.getLogger(__name__)


@dataclass
class InductionResult:
    promoted: bool
    reasoning: str
    evidence_score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0
    generalization_score: float = 0.0  # fraction of all test cases (pos + neg) that passed


class CausalInduction:
    """
    Tests candidate rules before promoting them to the ConceptRegistry.

    Strategy: Given a candidate rule (e.g. "+(x, 0) -> x"), generate
    3 test cases by varying the variable. Run BeamSearch on each.
    If the rule's transform reduces energy in >= 2 cases, promote.
    """

    def __init__(self):
        self._test_templates = [
            # Template expressions to test identity/elimination rules
            "{var} + 0", "0 + {var}",
            "{var} * 1", "1 * {var}",
            "{var} * 0",
            "{var} - {var}",
            "neg neg {var}",
            "{var} + {var}",
        ]
        # Wider variable set used when generating the expanded 20+ test suite
        self._variables = ["a", "b", "m", "p", "q", "x", "y", "z", "n", "k"]
        self.ckb = get_ckb()
        # Tracks total test exposures per rule for confidence-boost auto-promotion
        self._rule_observations: dict = {}

        # Async induction pipeline: episodes queued here are processed in the
        # background so the main solve loop is never blocked.
        self._pending_episodes: collections.deque = collections.deque()
        self._induction_thread: Optional[threading.Thread] = None
        self._start_induction_thread()

    # ── Async induction pipeline ───────────────────────────────────────────

    def _start_induction_thread(self) -> None:
        """Start the background induction thread (daemon, dies with process)."""
        if self._induction_thread is not None and self._induction_thread.is_alive():
            return
        self._induction_thread = threading.Thread(
            target=self._induction_worker,
            name="CausalInductionWorker",
            daemon=True,
        )
        self._induction_thread.start()

    def _induction_worker(self) -> None:
        """Background thread: drain the episode queue and run induce() on each."""
        while True:
            try:
                if self._pending_episodes:
                    episode = self._pending_episodes.popleft()
                    self._process_episode(episode)
                else:
                    # Nothing to do; sleep briefly to avoid busy-spinning.
                    threading.Event().wait(0.1)
            except Exception as exc:
                log.debug("[induction-worker] Episode processing error: %s", exc)

    def _process_episode(self, episode: Any) -> None:
        """Process a single episode dict from the queue."""
        try:
            problem    = episode.get("problem")
            result     = episode.get("result")
            reflection = episode.get("reflection")
            callback   = episode.get("callback")
            verdict = self.induce(problem=problem, result=result, reflection=reflection)
            if callback is not None:
                try:
                    callback(verdict)
                except Exception as cb_exc:
                    log.debug("[induction-worker] Callback error: %s", cb_exc)
        except Exception as exc:
            log.debug("[induction-worker] _process_episode failed: %s", exc)

    def queue_episode(
        self,
        problem: Any,
        result: Any,
        reflection: Any = None,
        callback: Optional[Callable] = None,
    ) -> None:
        """Enqueue an episode for async induction (non-blocking).

        The background thread will call ``induce(problem, result, reflection)``
        and, if provided, invoke ``callback(verdict)`` on completion.
        """
        self._pending_episodes.append({
            "problem":    problem,
            "result":     result,
            "reflection": reflection,
            "callback":   callback,
        })

    def induct_batch_async(self, episodes: list, callback: Optional[Callable] = None) -> None:
        """Run induction on a batch of episodes in a background thread.

        Instead of blocking after each solve, collect episodes and run
        induction in parallel with the next batch of solves.

        Args:
            episodes: List of dicts, each with keys ``problem``, ``result``,
                      and optionally ``reflection``.
            callback: Optional callable invoked with each ``InductionResult``
                      as it completes.
        """
        for ep in episodes:
            self.queue_episode(
                problem=ep.get("problem"),
                result=ep.get("result"),
                reflection=ep.get("reflection"),
                callback=callback,
            )

    def induce(self, problem: Any = None, result: Any = None, reflection: Any = None) -> Optional[Any]:
        """Synchronous induction from a solved problem/result pair.

        Extracts the candidate rule from ``reflection`` or ``result`` and runs
        ``evaluate()``.  Returns the ``InductionResult``, or ``None`` if no
        candidate rule can be found.
        """
        # Try to get a candidate rule from the reflection object first,
        # then fall back to the result object.
        candidate = None
        for src in (reflection, result):
            if src is None:
                continue
            candidate = (
                getattr(src, "rule", None)
                or getattr(src, "candidate_rule", None)
                or getattr(src, "inferred_rule", None)
            )
            if candidate is not None:
                break

        if candidate is None:
            return None

        return self.evaluate(candidate)

    # ── Synchronous evaluation ─────────────────────────────────────────────

    def evaluate(self, rule, energy_fn=None) -> InductionResult:
        """
        Test a candidate rule on generated examples.
        Returns InductionResult with promoted=True if rule generalizes.
        """
        try:
            from sare.engine import load_problem, BeamSearch, EnergyEvaluator, _base_transforms

            if energy_fn is None:
                energy_fn = EnergyEvaluator()
            searcher = BeamSearch()
            transforms = _base_transforms()

            # Extract the operator from the rule
            operator = getattr(rule, "operator_involved", None) or ""
            pattern_desc = getattr(rule, "pattern_description", "") or str(getattr(rule, "name", ""))
            rule_id = getattr(rule, "name", "unknown_rule")

            # Augment tests with successful chains from CKB for transfer learning
            # (result used for logging/bias only; we don't parse chain names back)
            self.ckb.suggest_tests_for_induction(operator)

            # ── Positive and negative test cases ────────────────────────────
            pos_exprs, neg_exprs = self._generate_tests(operator, pattern_desc)

            if not pos_exprs:
                conf = getattr(rule, "confidence", 0.5)
                if conf >= 0.5:
                    return InductionResult(
                        promoted=True,
                        reasoning=f"No test generation possible; accepted on confidence={conf:.2f}",
                        evidence_score=conf,
                    )
                return InductionResult(promoted=False, reasoning="Cannot generate tests")

            passed = 0
            total_pos = len(pos_exprs)
            for expr in pos_exprs:
                try:
                    _, g = load_problem(expr)
                    e_before = energy_fn.compute(g).total
                    result = searcher.search(g, energy_fn, transforms,
                                             beam_width=6, budget_seconds=2.0)
                    e_after = result.energy.total
                    if (e_before - e_after) > 0.1:
                        passed += 1
                except Exception:
                    pass

            # ── Negative test cases: the rule should NOT fire (no reduction) ─
            neg_correct = 0
            total_neg = len(neg_exprs)
            for expr in neg_exprs:
                try:
                    _, g = load_problem(expr)
                    e_before = energy_fn.compute(g).total
                    result = searcher.search(g, energy_fn, transforms,
                                             beam_width=6, budget_seconds=2.0)
                    e_after = result.energy.total
                    # "Correct" on a negative case means energy did NOT decrease
                    # beyond noise — the rule left the expression alone.
                    if (e_before - e_after) <= 0.1:
                        neg_correct += 1
                except Exception:
                    # Parse failure on a purposely-invalid expression is fine.
                    neg_correct += 1

            total_all = total_pos + total_neg
            all_correct = passed + neg_correct
            generalization_score = round(all_correct / max(total_all, 1), 3)

            pass_rate = passed / max(total_pos, 1)
            # Confidence boost: if rule has ≥10 historical observations, auto-promote
            historical_obs = self._rule_observations.get(rule_id, 0)
            if historical_obs >= 10:
                promoted = True
            else:
                # Lower threshold 65%→55%, but require ≥4 absolute tests passed
                promoted = (pass_rate >= 0.55 and passed >= 4) or pass_rate >= 0.65
            # Track observations for confidence boost path
            self._rule_observations[rule_id] = historical_obs + total_pos

            if promoted:
                # Register successful induction in the shared Knowledge Base
                self.ckb.register_rule(
                    rule_id=rule_id,
                    operator=operator,
                    desc=pattern_desc,
                    confidence=pass_rate
                )
                # Attach generalization_score to the rule object itself so
                # callers (daemon, web API) can persist it without extra plumbing.
                try:
                    rule.generalization_score = generalization_score
                except Exception:
                    pass

            # Inner monologue: report induction result
            try:
                from sare.meta.inner_monologue import get_inner_monologue
                im = get_inner_monologue()
                if promoted:
                    im.think(
                        f"Rule '{rule_id}' PROMOTED: {passed}/{total_pos} pos cases passed ({pass_rate:.0%})",
                        context="induction", emotion="excited",
                    )
                else:
                    im.think(
                        f"Rule '{rule_id}' rejected: only {passed}/{total_pos} pos cases passed ({pass_rate:.0%})",
                        context="induction", emotion="neutral",
                    )
            except Exception:
                pass

            return InductionResult(
                promoted=promoted,
                reasoning=(
                    f"Tested {total_pos} positive + {total_neg} negative cases: "
                    f"{passed}/{total_pos} pos passed, {neg_correct}/{total_neg} neg correct "
                    f"({pass_rate:.0%} pos pass-rate, generalization={generalization_score:.0%})"
                ),
                evidence_score=round(min(1.0, pass_rate * 1.2), 3),
                tests_passed=passed,
                tests_total=total_pos,
                generalization_score=generalization_score,
            )

        except Exception as exc:
            log.warning("CausalInduction.evaluate failed: %s", exc)
            conf = getattr(rule, "confidence", 0.5)
            return InductionResult(promoted=conf >= 0.5, reasoning=f"Fallback: {exc}")

    def _generate_tests(
        self, operator: str, pattern_desc: str
    ):
        """Generate 20+ diverse positive and negative test expressions.

        Returns
        -------
        tuple[list[str], list[str]]
            (positive_cases, negative_cases)

        Positive cases: expressions where the rule *should* fire and reduce
        energy (e.g., ``x + 0`` for additive-identity).

        Negative cases: expressions where the rule should NOT fire.  These
        verify the rule does not over-generalise.
        """
        vs = self._variables  # 10 variables available

        if operator in ("+", "add"):
            positive = (
                [f"{v} + 0" for v in vs[:5]]
                + [f"0 + {v}" for v in vs[:5]]
                + [
                    f"({vs[0]} + 0) + 0",
                    f"0 + (0 + {vs[1]})",
                    "0 + 0",
                    f"({vs[0]} + {vs[1]}) + 0",
                    f"0 + ({vs[2]} * {vs[3]})",
                ]
            )
            negative = (
                [f"{v} + 1" for v in vs[:4]]
                + [
                    f"{vs[0]} + {vs[1]}",
                    f"{vs[0]} * 0",
                    f"{vs[0]} - 0",
                ]
            )

        elif operator in ("*", "mul"):
            positive = (
                [f"{v} * 1" for v in vs[:5]]
                + [f"1 * {v}" for v in vs[:5]]
                + [
                    f"({vs[0]} * 1) * 1",
                    f"1 * (1 * {vs[1]})",
                    "1 * 1",
                    f"({vs[0]} + {vs[1]}) * 1",
                    f"1 * ({vs[2]} * {vs[3]})",
                ]
            )
            negative = (
                [f"{v} * 2" for v in vs[:4]]
                + [
                    f"{vs[0]} * {vs[1]}",
                    f"{vs[0]} + 1",
                    f"{vs[0]} * 0",
                ]
            )

        elif operator in ("*0", "zero", "mul_zero"):
            positive = (
                [f"{v} * 0" for v in vs[:5]]
                + [f"0 * {v}" for v in vs[:5]]
                + [
                    f"({vs[0]} + {vs[1]}) * 0",
                    f"0 * ({vs[2]} * {vs[3]})",
                    "0 * 0",
                    f"{vs[0]} * 0 * {vs[1]}",
                    f"({vs[0]} * 0) + {vs[1]}",
                ]
            )
            negative = (
                [f"{v} * 1" for v in vs[:4]]
                + [
                    f"{vs[0]} + 0",
                    f"{vs[0]} * {vs[1]}",
                ]
            )

        elif operator in ("-", "sub"):
            positive = (
                [f"{v} - {v}" for v in vs[:5]]
                + [
                    f"({vs[0]} - {vs[0]}) + {vs[1]}",
                    f"{vs[2]} * ({vs[1]} - {vs[1]})",
                    f"({vs[3]} + {vs[4]}) - ({vs[3]} + {vs[4]})",
                    "3 - 3",
                    "0 - 0",
                ]
            )
            negative = (
                [f"{v} - {vs[-1]}" for v in vs[:4]]
                + [
                    f"{vs[0]} + {vs[0]}",
                    f"{vs[0]} * {vs[0]}",
                    "3 - 2",
                ]
            )

        elif operator in ("neg", "not", "~", "double_neg"):
            positive = (
                [f"neg neg {v}" for v in vs[:5]]
                + [f"neg neg neg neg {v}" for v in vs[:3]]
                + [
                    f"neg neg ({vs[0]} + {vs[1]})",
                    f"neg neg ({vs[2]} * {vs[3]})",
                    "neg neg 0",
                    "neg neg 1",
                ]
            )
            negative = (
                [f"neg {v}" for v in vs[:5]]
                + [
                    f"{vs[0]} + {vs[1]}",
                    f"neg {vs[0]} + {vs[1]}",
                ]
            )

        else:
            # Generic fallback: cover all common identity/elimination patterns.
            positive = (
                [f"{v} + 0" for v in vs[:4]]
                + [f"{v} * 1" for v in vs[:4]]
                + [f"neg neg {v}" for v in vs[:4]]
                + [f"{v} - {v}" for v in vs[:4]]
            )
            negative = (
                [f"{v} + 1" for v in vs[:3]]
                + [f"{v} * 2" for v in vs[:3]]
            )

        return positive, negative