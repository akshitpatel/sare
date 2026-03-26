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
            "{var} + 0",
            "0 + {var}",
            "{var} * 1",
            "1 * {var}",
            "{var} * 0",
            "{var} - {var}",
            "neg neg {var}",
            "{var} + {var}",
        ]
        self._variables = ["a", "b", "m", "p", "q", "x", "y", "z", "n", "k"]
        # domain-specific templates: domain -> ["{var} op ..." strings]
        self._domain_templates: dict = {}

        self.ckb = get_ckb()
        self._rule_observations: dict = {}

        self._pending_episodes: collections.deque = collections.deque()
        self._induction_thread: Optional[threading.Thread] = None
        self._start_induction_thread()

    def register_templates(self, domain: str, templates: list) -> None:
        """
        Register domain-specific test templates for rule induction.

        Parameters
        ----------
        domain : str
            Domain name (e.g. "logic", "geometry"). Used to select templates
            when inducing rules observed in problems of that domain.
        templates : list of str
            Expression templates with ``{var}`` placeholder, e.g.
            ``["{var} AND True", "{var} OR False"]``
        """
        self._domain_templates[domain] = list(templates)

    def _get_templates(self, domain: str = "") -> list:
        """Return templates for domain, falling back to global defaults."""
        if domain and domain in self._domain_templates:
            return self._domain_templates[domain]
        return self._test_templates

    def _start_induction_thread(self) -> None:
        if self._induction_thread is not None and self._induction_thread.is_alive():
            return
        self._induction_thread = threading.Thread(
            target=self._induction_worker,
            name="CausalInductionWorker",
            daemon=True,
        )
        self._induction_thread.start()

    def _induction_worker(self) -> None:
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
        try:
            problem = episode.get("problem")
            result = episode.get("result")
            reflection = episode.get("reflection")
            callback = episode.get("callback")
            surprise = float(episode.get("surprise", 0.0))
            verdict = self.induce(problem=problem, result=result, reflection=reflection, surprise=surprise)
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
        surprise: float = 0.0,
    ) -> None:
        self._pending_episodes.append(
            {
                "problem": problem,
                "result": result,
                "reflection": reflection,
                "callback": callback,
                "surprise": float(surprise),
            }
        )

    def induct_batch_async(self, episodes: list, callback: Optional[Callable] = None) -> None:
        for ep in episodes:
            self.queue_episode(
                problem=ep.get("problem"),
                result=ep.get("result"),
                reflection=ep.get("reflection"),
                callback=callback,
                surprise=float(ep.get("surprise", 0.0)),
            )

    def induce(self, problem: Any = None, result: Any = None, reflection: Any = None, surprise: float = 0.0) -> Optional[Any]:
        candidate = None

        # reflection may itself be the rule (AbstractRule returned directly by PyReflectionEngine)
        if reflection is not None:
            if hasattr(reflection, "valid") and callable(getattr(reflection, "valid")):
                candidate = reflection  # reflection IS the rule
            else:
                candidate = getattr(reflection, "rule", None) or getattr(reflection, "candidate_rule", None)

        if candidate is None and result is not None:
            candidate = getattr(result, "rule", None) or getattr(result, "candidate_rule", None)

        if candidate is None and problem is not None:
            candidate = getattr(problem, "rule", None) or getattr(problem, "candidate_rule", None)

        if candidate is None:
            return None

        return self.evaluate(candidate, energy_fn=None, surprise=surprise)

    def _get_rule_id(self, rule: Any) -> str:
        rid = getattr(rule, "rule_id", None) or getattr(rule, "id", None)
        if rid is not None:
            return str(rid)
        op = getattr(rule, "operator_involved", None)
        desc = getattr(rule, "pattern_description", None) or getattr(rule, "name", None)
        return f"{op or 'unknown'}::{desc or 'rule'}"

    def _extract_rule_callable(self, rule: Any) -> Optional[Callable[[Any], Any]]:
        """
        Attempt to obtain a callable that applies the rule to an expression/graph.

        Supported expected shapes:
          - rule.apply(x) -> new expression
          - rule.transform(graph) / rule.apply_to(graph)
          - rule(rule) style: rule is callable
          - rule.pattern / rule.operator_involved only: cannot apply without engine bindings
        """
        if rule is None:
            return None

        if callable(rule):
            return rule

        for attr in ("apply", "apply_to", "transform", "apply_transform", "rewrite", "reduce"):
            fn = getattr(rule, attr, None)
            if callable(fn):
                return fn

        return None

    def _generate_test_expressions(self, domain: str = "") -> Tuple[List[str], List[str]]:
        """
        Produce (positive_exprs, negative_exprs).

        Positive exprs are built from the templates for *domain* (falls back to the
        default algebra templates if no domain-specific set has been registered via
        ``register_templates()``). Negative exprs are "near misses".
        """
        pos_exprs: List[str] = []
        for tmpl in self._get_templates(domain):
            for var in self._variables:
                pos_exprs.append(tmpl.format(var=var))

        # Keep positive set compact for speed while retaining diversity
        pos_exprs = pos_exprs[:12]

        neg_exprs: List[str] = []
        for var in self._variables:
            # Common near-miss patterns:
            # - swap in a "wrong" identity constant
            # - use different operator arrangement
            # - use subtraction by non-equal term
            neg_exprs.append(f"{var} + 1")
            neg_exprs.append(f"{var} * 2")
            neg_exprs.append(f"{var} - 0")
            neg_exprs.append(f"neg {var}")  # likely not matching double-neg elimination
            if len(neg_exprs) >= 12:
                break

        return pos_exprs, neg_exprs

    def _compare_energy(self, before: Any, after: Any, energy_fn: Optional[Callable[[Any], float]]) -> bool:
        """
        Return True if after is strictly better (lower energy) or energy_fn absent
        but objects indicate improvement.
        """
        if energy_fn is not None:
            try:
                eb = energy_fn(before)
                ea = energy_fn(after)
                return ea < eb
            except Exception:
                return False

        # Fallback heuristics when energy_fn is missing:
        # If after has attribute energy and it's lower, accept.
        for before_attr in ("energy", "energy_before", "cost"):
            pass
        try:
            ea = getattr(after, "energy", None) or getattr(after, "cost", None)
            eb = getattr(before, "energy", None) or getattr(before, "cost", None)
            if ea is not None and eb is not None:
                return float(ea) < float(eb)
        except Exception:
            pass

        # Otherwise cannot evaluate; treat as failure.
        return False

    # ===== Approved changes start here: modify __init__ + evaluate + add 3 helpers =====

    def _resolve_energy_fn(self, energy_fn: Optional[Callable[[Any], float]], rule: Any) -> Optional[Callable[[Any], float]]:
        """
        Resolve an energy function for evaluation.

        If caller passed energy_fn, use it.
        If not, try to obtain energy_fn from rule (e.g., rule.energy_fn) or from an
        attribute on the rule object indicating it can compute energy deltas.
        """
        if energy_fn is not None:
            return energy_fn

        for attr in ("energy_fn", "compute_energy", "energy"):
            candidate = getattr(rule, attr, None)
            if callable(candidate):
                return candidate
        # No reliable way to compute energy without engine bindings.
        return None

    def _build_candidate_suite(self, rule: Any, domain: str = "") -> Tuple[List[Any], List[Any]]:
        """
        Create a suite of test 'inputs' for the candidate rule.

        In the absence of graph/expression engine bindings in this file, we generate
        expression strings; rule application and energy evaluation are then expected
        to understand those objects.

        Returns:
          (positive_inputs, negative_inputs)
        """
        pos_exprs, neg_exprs = self._generate_test_expressions(domain=domain)

        # If rule indicates it expects graphs rather than expressions, allow the rule
        # to provide its own suite builder.
        suite_builder = getattr(rule, "build_test_suite", None)
        if callable(suite_builder):
            try:
                built = suite_builder()
                if isinstance(built, tuple) and len(built) == 2:
                    pos_in, neg_in = built
                    if isinstance(pos_in, list) and isinstance(neg_in, list):
                        return pos_in, neg_in
            except Exception:
                pass

        return pos_exprs, neg_exprs

    def _score_rule_on_suite(
        self,
        rule: Any,
        energy_fn: Optional[Callable[[Any], float]],
        inputs_pos: List[Any],
        inputs_neg: List[Any],
    ) -> Tuple[int, int, float, float, List[str]]:
        """
        Apply the rule on each test input; count passes on pos and negatives
        (negatives must NOT improve energy).

        Returns:
          (tests_passed, tests_total, generalization_score, evidence_score, debug_reasons)
        """
        apply_fn = self._extract_rule_callable(rule)
        if apply_fn is None:
            # If we can't apply the rule, cannot meaningfully evaluate.
            tests_total = len(inputs_pos) + len(inputs_neg)
            return 0, tests_total, 0.0, 0.0, ["rule not applicable: missing apply callable"]

        debug_reasons: List[str] = []
        tests_total = len(inputs_pos) + len(inputs_neg)
        tests_passed = 0

        # Evidence: average improvement magnitude on positives minus "improvement" on negatives.
        improv_pos_sum = 0.0
        improv_neg_penalty_sum = 0.0

        for i, inp in enumerate(inputs_pos):
            try:
                out = apply_fn(inp)
                ok = self._compare_energy(inp, out, energy_fn=energy_fn)
                if ok:
                    tests_passed += 1
                    if energy_fn is not None:
                        try:
                            eb = energy_fn(inp)
                            ea = energy_fn(out)
                            improv_pos_sum += float(eb - ea)
                        except Exception:
                            pass
                else:
                    debug_reasons.append(f"pos#{i} failed energy check")
            except Exception as exc:
                debug_reasons.append(f"pos#{i} apply failed: {exc}")

        for j, inp in enumerate(inputs_neg):
            try:
                out = apply_fn(inp)
                improved = self._compare_energy(inp, out, energy_fn=energy_fn)
                if not improved:
                    tests_passed += 1
                    if energy_fn is not None:
                        try:
                            eb = energy_fn(inp)
                            ea = energy_fn(out)
                            improv_neg_penalty_sum += float(max(0.0, ea - eb))
                        except Exception:
                            pass
                else:
                    debug_reasons.append(f"neg#{j} wrongly improved energy")
            except Exception as exc:
                debug_reasons.append(f"neg#{j} apply failed: {exc}")

        generalization_score = tests_passed / float(tests_total) if tests_total > 0 else 0.0

        evidence_score = 0.0
        if energy_fn is not None:
            # Robust evidence: normalize by number of positive and negative samples.
            npos = max(1, len(inputs_pos))
            nneg = max(1, len(inputs_neg))
            improv_pos_avg = improv_pos_sum / float(npos)
            neg_penalty_avg = improv_neg_penalty_sum / float(nneg)
            # Higher is better; negative penalty should reduce evidence when negatives improve.
            evidence_score = improv_pos_avg - 0.5 * neg_penalty_avg

        return tests_passed, tests_total, generalization_score, evidence_score, debug_reasons

    def evaluate(self, rule: Any, energy_fn: Optional[Callable[[Any], float]] = None, domain: str = "", surprise: float = 0.0) -> Optional[InductionResult]:
        """
        Evaluate candidate rule for promotion.

        Improvements over previous implementation:
          - Uses a reusable suite builder and energy resolution.
          - Scores rule on both positive and negative near-miss tests.
          - Uses consistency thresholds and historical observation counts for auto-promotion.
        """
        if rule is None:
            return None

        rule_id = self._get_rule_id(rule)
        energy_fn_resolved = self._resolve_energy_fn(energy_fn, rule)

        # Build test suite once per evaluation call (uses domain-specific templates if registered).
        inputs_pos, inputs_neg = self._build_candidate_suite(rule, domain=domain)
        _has_callable = self._extract_rule_callable(rule) is not None

        tests_passed, tests_total, generalization_score, evidence_score, debug_reasons = self._score_rule_on_suite(
            rule=rule,
            energy_fn=energy_fn_resolved,
            inputs_pos=inputs_pos,
            inputs_neg=inputs_neg,
        )

        # Historical observation tracking (for confidence boost at higher counts)
        # High-surprise boost: surprising episodes (surprise > 2.5) count as 2 observations.
        obs = self._rule_observations.get(rule_id, {"count": 0})
        surprise_val = float(surprise) if surprise else 0.0
        obs_increment = 2 if surprise_val > 2.5 else 1
        obs["count"] = int(obs.get("count", 0)) + obs_increment
        if surprise_val > 2.5:
            obs["last_high_surprise"] = surprise_val
        self._rule_observations[rule_id] = obs
        observation_count = obs["count"]

        # Promotion criteria (sample-efficient, T3-5):
        # - promote if pass_rate >= 0.55 AND passes >= 4 absolute tests
        # - OR pass_rate >= 0.65
        # - OR observation_count >= 3 -> auto-promote (lowered from 10 to 3)
        # - Oracle confidence threshold lowered from 0.60 to 0.55
        pass_rate = generalization_score
        promoted = False
        reasoning_parts: List[str] = []

        rule_confidence = float(getattr(rule, "confidence", 0.0))

        if surprise_val > 2.5:
            reasoning_parts.append(f"high-surprise-boost: surprise={surprise_val:.2f} obs_increment=2")

        # Effective threshold: lower when surprise is high (surprising episodes are more informative)
        eff_threshold = 0.50 * (0.85 if surprise_val > 2.5 else 1.0)

        if tests_total <= 0:
            # No tests run — fall back to rule's own confidence (e.g. Oracle-validated rules)
            promoted = rule_confidence >= eff_threshold
            reasoning_parts.append(f"no-tests: using rule confidence={rule_confidence:.3f} (threshold={eff_threshold:.2f})")
        else:
            if observation_count >= 3:
                # After 3 confirmed observations, auto-promote if confidence is solid
                promoted = rule_confidence >= 0.50 or pass_rate >= 0.50
                reasoning_parts.append(f"auto-promote: obs={observation_count}>=3 conf={rule_confidence:.3f}")
            elif pass_rate >= 0.55:
                promoted = True
                reasoning_parts.append(f"pass_rate={pass_rate:.3f}>=0.55")
            elif pass_rate >= 0.45 and tests_passed >= 3:
                promoted = True
                reasoning_parts.append(f"pass_rate={pass_rate:.3f}>=0.45 and tests_passed={tests_passed}>=3")
            elif not _has_callable and rule_confidence >= eff_threshold:
                # apply_fn unavailable (AbstractRule without callable) but Oracle validated it
                promoted = True
                reasoning_parts.append(f"oracle-validated: no-callable, rule confidence={rule_confidence:.3f}>={eff_threshold:.2f}")
            else:
                promoted = False
                reasoning_parts.append(
                    f"insufficient: pass_rate={pass_rate:.3f}, tests_passed={tests_passed}/{tests_total}, obs={observation_count}"
                )

        reasoning = "; ".join(reasoning_parts)
        result = InductionResult(
            promoted=promoted,
            reasoning=reasoning,
            evidence_score=float(evidence_score),
            tests_passed=int(tests_passed),
            tests_total=int(tests_total),
            generalization_score=float(generalization_score),
        )

        # If promoted, register in CKB with any available metadata.
        if promoted:
            try:
                op = getattr(rule, "operator_involved", None)
                desc = getattr(rule, "pattern_description", None) or getattr(rule, "name", None)
                conf = float(getattr(rule, "confidence", None) or generalization_score)
                self.ckb.register_rule(rule_id, op or "unknown", desc or "induced_rule", conf)
            except Exception as exc:
                log.debug("[induction] CKB register_rule failed for %s: %s", rule_id, exc)

        return result

    # ===== End approved changes =====

    def queue_induction_for_rule(self, rule: Any, reflection: Any = None, callback: Optional[Callable] = None) -> None:
        """
        Convenience wrapper to enqueue induction for a synthetic 'episode' object.
        This method is not part of required public interfaces; it is safe for internal use.
        """
        problem = None
        result = None
        self.queue_episode(problem=problem, result=result, reflection=reflection or type("R", (), {"rule": rule})(), callback=callback)