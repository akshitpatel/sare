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
    a small suite of test cases by varying the variable. For each test
    case, compute energy before/after applying the rule and promote
    if the rule consistently reduces energy.
    """

    def __init__(self):
        self._test_templates = [
            # Template expressions to test identity/elimination rules
            "{var} + 0",
            "0 + {var}",
            "{var} * 1",
            "1 * {var}",
            "{var} * 0",
            "{var} - {var}",
            "neg neg {var}",
            "{var} + {var}",
        ]
        # Wider variable set used when generating the expanded test suite
        self._variables = ["a", "b", "m", "p", "q", "x", "y", "z", "n", "k"]

        self.ckb = get_ckb()
        # Tracks total test exposures per rule for confidence-boost auto-promotion
        self._rule_observations: dict = {}

        # Async induction pipeline: episodes queued here are processed in the
        # background so the main solve loop is never blocked.
        self._pending_episodes: collections.deque = collections.deque()
        self._induction_thread: Optional[threading.Thread] = None
        self._start_induction_thread()

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
                    threading.Event().wait(0.1)
            except Exception as exc:
                log.debug("[induction-worker] Episode processing error: %s", exc)

    def _process_episode(self, episode: Any) -> None:
        """Process a single episode dict from the queue."""
        try:
            problem = episode.get("problem")
            result = episode.get("result")
            reflection = episode.get("reflection")
            callback = episode.get("callback")
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
        """Enqueue an episode for async induction (non-blocking)."""
        self._pending_episodes.append(
            {
                "problem": problem,
                "result": result,
                "reflection": reflection,
                "callback": callback,
            }
        )

    def induct_batch_async(self, episodes: list, callback: Optional[Callable] = None) -> None:
        """Run induction on a batch of episodes in a background thread."""
        for ep in episodes:
            self.queue_episode(
                problem=ep.get("problem"),
                result=ep.get("result"),
                reflection=ep.get("reflection"),
                callback=callback,
            )

    def induce(self, problem: Any = None, result: Any = None, reflection: Any = None) -> Optional[Any]:
        """Synchronous induction from a solved problem/result pair."""
        # Try to get a candidate rule from the reflection object first,
        # then fall back to the result object.
        candidate = None
        for src in (reflection, result):
            if src is None:
                continue
            candidate = getattr(src, "rule", None) or getattr(src, "candidate_rule", None)
            if candidate is not None:
                break

        if candidate is None and problem is not None:
            candidate = getattr(problem, "rule", None) or getattr(problem, "candidate_rule", None)

        if candidate is None:
            return None

        return self.evaluate(candidate, energy_fn=None)

    def _get_rule_id(self, rule: Any) -> str:
        rid = getattr(rule, "rule_id", None) or getattr(rule, "id", None)
        if rid is not None:
            return str(rid)
        op = getattr(rule, "operator_involved", None)
        desc = getattr(rule, "pattern_description", None) or getattr(rule, "name", None)
        return f"{op or 'unknown'}::{desc or 'rule'}"

    def _extract_rule_callable(self, rule: Any) -> Optional[Callable[[Any], Any]]:
        # Support a few expected shapes:
        # - rule.apply(graph) -> graph
        # - rule.transform(graph) -> graph
        # - rule is already a callable(graph) -> graph
        if callable(rule):
            return rule
        for attr in ("apply", "transform", "run", "reduce"):
            fn = getattr(rule, attr, None)
            if callable(fn):
                return fn
        return None

    def _extract_rule_pattern(self, rule: Any) -> Optional[str]:
        for attr in ("pattern_description", "name", "operator_involved", "desc"):
            v = getattr(rule, attr, None)
            if v:
                return str(v)
        return None

    def _rule_energy_step(self, expr: Any, apply_fn: Callable[[Any], Any], energy_fn: Callable[[Any], float]) -> Tuple[float, float]:
        # energy_fn is expected to accept either expression string or graph
        before = energy_fn(expr)
        after_graph_or_expr = apply_fn(expr)
        # If apply_fn cannot apply, it may return None; treat as no change
        after = energy_fn(after_graph_or_expr if after_graph_or_expr is not None else expr)
        return before, after

    def evaluate(self, rule: Any, energy_fn=None) -> InductionResult:
        """
        Evaluate a candidate rule for promotion based on generalized energy reduction.

        APPROVED CHANGE:
          CausalInduction.evaluate(self, rule, energy_fn=None) -> InductionResult
        """
        try:
            rule_id = self._get_rule_id(rule)
        except Exception:
            rule_id = "unknown_rule"

        apply_fn = self._extract_rule_callable(rule)
        if apply_fn is None:
            return InductionResult(
                promoted=False,
                reasoning=f"[induction] No callable/apply method found for rule={rule_id}.",
                evidence_score=0.0,
                tests_passed=0,
                tests_total=0,
                generalization_score=0.0,
            )

        # Default energy function: if caller doesn't provide one, we cannot
        # objectively test energy reduction; so we conservatively fail.
        if energy_fn is None:
            return InductionResult(
                promoted=False,
                reasoning=(
                    f"[induction] energy_fn is None; cannot evaluate rule energetics for rule={rule_id}."
                ),
                evidence_score=0.0,
                tests_passed=0,
                tests_total=0,
                generalization_score=0.0,
            )

        # Create a positive test suite (rules should reduce energy)
        pos_exprs: List[str] = []
        for tmpl in self._test_templates:
            for var in self._variables:
                pos_exprs.append(tmpl.format(var=var))
        # Cap to keep induction lightweight
        # (use at least 8+ across operator types, but keep bounded)
        pos_exprs = pos_exprs[:12]

        # Create a negative suite: apply inverse/perturbations by changing var in templates
        # to make it likely the rule doesn't apply correctly.
        # We approximate by mixing templates and producing "wrong" contexts.
        neg_exprs: List[str] = []
        # For negative examples, swap identity constants in a way that's usually invalid:
        # - {var} + 0 => {var} + 1
        # - {var} * 1 => {var} * 2 (non-identity)
        # - {var} - {var} => {var} - (var+1) not expressible; use distinct vars
        # We'll generate with controlled distinct vars.
        for var1, var2 in zip(self._variables, reversed(self._variables)):
            if len(neg_exprs) >= 8:
                break
            neg_exprs.append(f"{var1} + 1")
            if len(neg_exprs) >= 8:
                break
            neg_exprs.append(f"{var1} * 2")
            if len(neg_exprs) >= 8:
                break
            neg_exprs.append(f"{var1} - {var2}")

        tests_passed = 0
        tests_total = 0
        pos_passed = 0
        neg_passed = 0

        # Energy reduction threshold: accept if energy_after is sufficiently lower
        # (tuned conservatively; small numeric noise allowed).
        delta_threshold = 0.1

        # Evidence: aggregate mean improvement on positives minus mean improvement on negatives
        pos_deltas: List[float] = []
        neg_deltas: List[float] = []

        for expr in pos_exprs:
            try:
                before, after = self._rule_energy_step(expr, apply_fn, energy_fn)
                delta = before - after
            except Exception:
                before, after = float("nan"), float("nan")
                delta = float("-inf")

            tests_total += 1
            if delta > delta_threshold:
                tests_passed += 1
                pos_passed += 1
                pos_deltas.append(delta)

        for expr in neg_exprs:
            try:
                before, after = self._rule_energy_step(expr, apply_fn, energy_fn)
                delta = before - after
            except Exception:
                before, after = float("nan"), float("nan")
                delta = float("-inf")

            tests_total += 1
            # For negative cases, we expect NOT to reduce energy; treat reduction as failure.
            # We count a "pass" if delta is small/negative (no meaningful energy drop).
            if delta <= delta_threshold:
                tests_passed += 1
                neg_passed += 1
                neg_deltas.append(delta)

        generalization_score = (tests_passed / tests_total) if tests_total > 0 else 0.0

        # Promotion logic (updated): pass_rate >= 55% AND passes >= 4, or pass_rate >= 65%.
        pass_rate = generalization_score
        passes_abs = tests_passed

        # Confidence boost auto-promotion if rule has enough historical observations
        hist = self._rule_observations.get(rule_id, {})
        obs_count = int(hist.get("count", 0))
        auto_promote_at = 10
        promoted = False

        reasoning_parts: List[str] = []
        reasoning_parts.append(f"[induction] rule_id={rule_id}")
        reasoning_parts.append(
            f"tests_passed={tests_passed}/{tests_total} (pass_rate={pass_rate:.3f}, pos_passed={pos_passed}, neg_passed={neg_passed})"
        )

        if pass_rate >= 0.55 and passes_abs >= 4:
            promoted = True
            reasoning_parts.append("promotion: pass_rate>=0.55 and passes_abs>=4")
        elif pass_rate >= 0.65:
            promoted = True
            reasoning_parts.append("promotion: pass_rate>=0.65")

        if not promoted and obs_count >= auto_promote_at:
            promoted = True
            reasoning_parts.append(
                f"promotion: historical observations >= {auto_promote_at} (obs_count={obs_count})"
            )

        # Evidence score: mean delta on positives (higher better) and mean delta on negatives (lower better)
        mean_pos_delta = (sum(pos_deltas) / len(pos_deltas)) if pos_deltas else 0.0
        mean_neg_delta = (sum(neg_deltas) / len(neg_deltas)) if neg_deltas else 0.0
        evidence_score = mean_pos_delta - mean_neg_delta

        # Update observation tracking
        self._rule_observations.setdefault(rule_id, {"count": 0, "total_pass_rate": 0.0, "last": None})
        self._rule_observations[rule_id]["count"] += 1
        self._rule_observations[rule_id]["total_pass_rate"] = (
            float(self._rule_observations[rule_id]["total_pass_rate"]) + float(pass_rate)
        )
        self._rule_observations[rule_id]["last"] = {
            "pass_rate": pass_rate,
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "promoted": promoted,
        }

        if promoted:
            try:
                op = getattr(rule, "operator_involved", None) or getattr(rule, "operator", None) or "unknown_op"
                desc = getattr(rule, "pattern_description", None) or getattr(rule, "name", None) or "induced_rule"
                conf = float(pass_rate)
                self.ckb.register_rule(rule_id=rule_id, operator=op, desc=desc, confidence=conf)
            except Exception as exc:
                log.debug("[induction] CKB register_rule failed: %s", exc)

        return InductionResult(
            promoted=promoted,
            reasoning="; ".join(reasoning_parts),
            evidence_score=float(evidence_score),
            tests_passed=int(tests_passed),
            tests_total=int(tests_total),
            generalization_score=float(generalization_score),
        )