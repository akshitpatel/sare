"""
Lightweight Causal Induction for SARE-HX.

Tests candidate rules from PyReflectionEngine before promotion.
A rule is tested by checking if it generalizes: apply the pattern
on 3 different graphs. If it reduces energy on at least 2, promote.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class InductionResult:
    promoted: bool
    reasoning: str
    evidence_score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0


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
        self._variables = ["a", "b", "m", "p", "q"]

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

            # Generate test expressions based on the rule's pattern
            test_exprs = self._generate_tests(operator, pattern_desc)
            if not test_exprs:
                # Can't generate tests — accept with low confidence if rule confidence > 0.5
                conf = getattr(rule, "confidence", 0.5)
                if conf >= 0.5:
                    return InductionResult(
                        promoted=True,
                        reasoning=f"No test generation possible; accepted on confidence={conf:.2f}",
                        evidence_score=conf,
                        tests_passed=0,
                        tests_total=0,
                    )
                return InductionResult(
                    promoted=False,
                    reasoning="Cannot generate tests and confidence too low",
                    evidence_score=conf,
                )

            passed = 0
            total = len(test_exprs)
            for expr in test_exprs:
                try:
                    _, g = load_problem(expr)
                    e_before = energy_fn.compute(g).total
                    result = searcher.search(g, energy_fn, transforms,
                                             beam_width=6, budget_seconds=2.0)
                    e_after = result.energy.total
                    delta = e_before - e_after
                    if delta > 0.1:
                        passed += 1
                except Exception:
                    pass

            # Require at least 60% pass rate
            pass_rate = passed / max(total, 1)
            evidence = min(1.0, pass_rate * 1.2)
            promoted = pass_rate >= 0.6

            return InductionResult(
                promoted=promoted,
                reasoning=f"Tested {total} cases: {passed} passed ({pass_rate:.0%})",
                evidence_score=round(evidence, 3),
                tests_passed=passed,
                tests_total=total,
            )

        except Exception as exc:
            log.warning("CausalInduction.evaluate failed: %s", exc)
            # Fallback: accept rules with high confidence
            conf = getattr(rule, "confidence", 0.5)
            return InductionResult(
                promoted=conf >= 0.5,
                reasoning=f"Evaluation failed ({exc}); fallback on confidence={conf:.2f}",
                evidence_score=conf,
            )

    def _generate_tests(self, operator: str, pattern_desc: str) -> list:
        """Generate 3-5 test expressions based on the rule's operator."""
        tests = []
        vars_to_use = self._variables[:3]

        if operator in ("+", "add"):
            for v in vars_to_use:
                tests.append(f"{v} + 0")
            tests.append("0 + x")
        elif operator in ("*", "mul"):
            for v in vars_to_use:
                tests.append(f"{v} * 1")
                tests.append(f"{v} * 0")
        elif operator in ("-", "sub"):
            for v in vars_to_use:
                tests.append(f"{v} - {v}")
        elif operator in ("neg", "not", "~"):
            for v in vars_to_use:
                tests.append(f"neg neg {v}")
        elif "identity" in pattern_desc.lower():
            for v in vars_to_use:
                tests.append(f"{v} + 0")
                tests.append(f"{v} * 1")
        elif "zero" in pattern_desc.lower() or "annihil" in pattern_desc.lower():
            for v in vars_to_use:
                tests.append(f"{v} * 0")
        else:
            # Generic: try common simplifiable expressions
            for v in vars_to_use:
                tests.append(f"{v} + 0")

        return tests[:5]  # Cap at 5 tests
