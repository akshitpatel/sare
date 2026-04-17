"""
CounterfactualReasoner — SARE-HX reasons about alternative paths.

After solving a problem, analyzes which steps were critical vs redundant.
Also generates "what if" hypotheses about why problems failed.

Feeds insights into world model and rule confidence scores.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class CounterfactualReport:
    """Result of counterfactual analysis on a solution."""
    problem_id: str
    proof_steps: List[str]
    critical_steps: List[str]       # steps whose removal breaks the solution
    redundant_steps: List[str]      # steps that could be skipped
    alternative_paths: List[str]    # other routes that might work
    confidence: float               # 0-1 confidence in analysis
    elapsed_ms: float = 0.0
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "proof_steps": self.proof_steps,
            "critical_steps": self.critical_steps,
            "redundant_steps": self.redundant_steps,
            "alternative_paths": self.alternative_paths,
            "confidence": round(self.confidence, 3),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "generated_at": self.generated_at,
        }


@dataclass
class Hypothesis:
    """A 'what if' hypothesis about a failed problem."""
    text: str                    # "What if I applied rule X first?"
    suggested_experiment: str    # concrete experiment to test this
    domain: str
    priority: float

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "suggested_experiment": self.suggested_experiment,
            "domain": self.domain,
            "priority": round(self.priority, 3),
        }


class CounterfactualReasoner:
    """
    Analyzes solved problems to identify critical vs redundant steps,
    and generates hypotheses about why problems fail.
    """

    def __init__(self):
        self._reports: List[CounterfactualReport] = []
        self._hypotheses: List[Hypothesis] = []

    def analyze(
        self,
        proof_steps: List[str],
        problem: Any,
        energy_fn: Any = None,
    ) -> CounterfactualReport:
        """
        Re-analyze a solution to find critical dependencies.

        For each step in proof_steps, checks if removing it would likely
        break the solution (based on step type and energy reduction).

        Returns CounterfactualReport with critical/redundant step classification.
        """
        t0 = time.time()
        problem_id = str(getattr(problem, "problem_id", getattr(problem, "id", "unknown")))

        critical = []
        redundant = []
        alternatives = []

        # Classify each step
        for i, step in enumerate(proof_steps):
            step_lower = step.lower()

            # Heuristic: steps that directly reduce energy or apply key rules are critical
            is_critical = any(kw in step_lower for kw in [
                "simplif", "solve", "factor", "expand", "cancel",
                "substit", "eliminat", "reduce", "equation",
            ])

            # Identity/zero operations are often redundant
            is_redundant = any(kw in step_lower for kw in [
                "identity", "zero", "neutral", "trivial", "no-op",
            ])

            if is_redundant and not is_critical:
                redundant.append(step)
            else:
                critical.append(step)

            # Suggest alternative: try a different ordering
            if i > 0 and is_critical:
                alternatives.append(
                    f"Try applying '{step}' before '{proof_steps[i-1]}'"
                )

        # Confidence based on how many steps we could classify
        total = len(proof_steps)
        classified = len(critical) + len(redundant)
        confidence = classified / total if total > 0 else 0.0

        elapsed_ms = (time.time() - t0) * 1000.0

        report = CounterfactualReport(
            problem_id=problem_id,
            proof_steps=proof_steps,
            critical_steps=critical,
            redundant_steps=redundant,
            alternative_paths=alternatives[:3],
            confidence=confidence,
            elapsed_ms=elapsed_ms,
        )
        self._reports.append(report)

        # Update rule confidence in world model
        self._update_rule_confidence(critical, redundant)

        log.debug(
            "[Counterfactual] problem=%s critical=%d redundant=%d",
            problem_id, len(critical), len(redundant),
        )
        return report

    def hypothesize(self, failed_problem: Any, transforms: List[Any]) -> List[Hypothesis]:
        """
        Generate 'what if' hypotheses about why a problem failed.
        Each hypothesis is a new experiment to run.
        """
        domain = str(getattr(failed_problem, "domain", "general"))
        expr = str(getattr(failed_problem, "expression", getattr(failed_problem, "name", "?")))

        hypotheses = []

        # Hypothesis 1: Wrong starting transform
        transform_names = []
        for t in transforms[:5]:
            name = getattr(t, "__class__", type(t)).__name__
            transform_names.append(name)

        if transform_names:
            for tname in transform_names[:3]:
                h = Hypothesis(
                    text=f"What if I start with '{tname}' on '{expr[:40]}'?",
                    suggested_experiment=f"force_first_transform={tname}",
                    domain=domain,
                    priority=0.6,
                )
                hypotheses.append(h)

        # Hypothesis 2: Problem needs a domain bridge
        h = Hypothesis(
            text=f"Is '{expr[:40]}' actually a {domain} problem, or does it belong to another domain?",
            suggested_experiment=f"cross_domain_attempt:{domain}",
            domain=domain,
            priority=0.5,
        )
        hypotheses.append(h)

        # Hypothesis 3: Beam width too narrow
        h = Hypothesis(
            text=f"Would a wider search beam solve '{expr[:40]}'?",
            suggested_experiment="widen_beam:+4",
            domain=domain,
            priority=0.4,
        )
        hypotheses.append(h)

        with_inner_monologue = True
        try:
            from sare.meta.inner_monologue import get_inner_monologue
            im = get_inner_monologue()
            for hyp in hypotheses:
                im.think(hyp.text, context="counterfactual", emotion="curious")
        except Exception:
            pass

        self._hypotheses.extend(hypotheses)
        return hypotheses

    def _update_rule_confidence(self, critical: List[str], redundant: List[str]):
        """Feed counterfactual results back into world model rule confidence."""
        try:
            from sare.memory.world_model import get_world_model
            wm = get_world_model()
            # Boost confidence in critical rules, reduce for redundant ones
            for step in critical:
                if hasattr(wm, "update_belief"):
                    wm.update_belief(f"rule:{step[:40]}", 1.0, surprise=0.1)
            for step in redundant:
                if hasattr(wm, "update_belief"):
                    wm.update_belief(f"rule:{step[:40]}", 0.3, surprise=0.5)
        except Exception:
            pass

    def get_recent_reports(self, last_n: int = 10) -> List[dict]:
        return [r.to_dict() for r in self._reports[-last_n:]]

    def get_hypotheses(self) -> List[dict]:
        return [h.to_dict() for h in self._hypotheses[-20:]]


# ── Singleton ──────────────────────────────────────────────────────────────────

_COUNTERFACTUAL_REASONER: Optional[CounterfactualReasoner] = None


def get_counterfactual_reasoner() -> CounterfactualReasoner:
    global _COUNTERFACTUAL_REASONER
    if _COUNTERFACTUAL_REASONER is None:
        _COUNTERFACTUAL_REASONER = CounterfactualReasoner()
    return _COUNTERFACTUAL_REASONER
