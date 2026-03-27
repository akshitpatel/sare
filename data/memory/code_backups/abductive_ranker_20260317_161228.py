"""
AbductiveRanker — TODO-E: Real Hypothesis Generation + Ranking
==============================================================
Replaces the 30-line C++ stub (hypothesis_ranker.cpp) with a full
abductive reasoning engine that:

  1. GENERATES candidate hypotheses from ConceptRegistry rules
     (inference to best explanation — "what rule would produce
      this observed outcome?")
  2. SCORES each by: P(evidence | hypothesis) × simplicity / prior
  3. RANKS by composite Occam-Bayesian score
  4. Returns a reasoning chain that explains its conclusion

This is what detectives do: "Given that the expression reduced
 by 4.2 energy using these transforms, what underlying rule
 best explains the pattern?"

Usage::

    ranker = AbductiveRanker(concept_registry)
    hypotheses = ranker.explain(
        observed_transforms=["additive_identity", "multiplicative_identity"],
        observed_delta=4.2,
        domain="arithmetic",
    )
    best = hypotheses[0]
    print(best.name, best.reasoning_chain)
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ── Hypothesis dataclass ──────────────────────────────────────────────────────

@dataclass
class AbductiveHypothesis:
    name:           str
    domain:         str
    confidence:     float           # rule confidence from registry / evidence
    complexity:     float           # number of interventions / rule size
    prediction_error: float         # |expected_delta - observed_delta|
    occam_score:    float           # prediction_error + λ * complexity (lower = better)
    evidence_strength: float        # P(evidence | hypothesis) in [0, 1]
    posterior:      float           # Bayesian posterior in [0, 1]
    reasoning_chain: List[str]      # step-by-step English explanation
    supporting_transforms: List[str]
    recommended_action: str         # "accept" | "verify" | "reject"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":              self.name,
            "domain":            self.domain,
            "confidence":        round(self.confidence, 3),
            "complexity":        round(self.complexity, 3),
            "prediction_error":  round(self.prediction_error, 4),
            "occam_score":       round(self.occam_score, 4),
            "evidence_strength": round(self.evidence_strength, 3),
            "posterior":         round(self.posterior, 3),
            "reasoning_chain":   self.reasoning_chain,
            "supporting_transforms": self.supporting_transforms,
            "recommended_action": self.recommended_action,
        }


# ── Prior knowledge about rule complexity and expected energy reduction ────────

_DOMAIN_PRIORS = {
    "arithmetic": {"expected_delta_per_rule": 1.5, "base_prior": 0.7},
    "logic":      {"expected_delta_per_rule": 1.2, "base_prior": 0.6},
    "algebra":    {"expected_delta_per_rule": 2.0, "base_prior": 0.5},
    "sets":       {"expected_delta_per_rule": 1.0, "base_prior": 0.5},
    "code":       {"expected_delta_per_rule": 1.8, "base_prior": 0.6},
    "general":    {"expected_delta_per_rule": 1.0, "base_prior": 0.4},
}

# Rules that typically co-occur (evidence of mutual causation)
_RULE_CLUSTERS = {
    "additive_identity":       {"multiplicative_identity", "constant_folding"},
    "double_negation":         {"double_negation_logic", "and_true", "or_false"},
    "de_morgan_and":           {"de_morgan_or", "double_negation_logic"},
    "distributive_mul_add":    {"factor_common", "commutativity_mul"},
    "exponent_product":        {"exponent_power", "exponent_quotient"},
}


# ── AbductiveRanker ───────────────────────────────────────────────────────────

class AbductiveRanker:
    """
    Abductive reasoning engine for SARE-HX.

    Given an observed outcome (transforms applied, energy delta, domain),
    generates the most probable causal explanation from the ConceptRegistry.

    Three-stage process:
      1. Generation — enumerate candidate rules from registry that match
      2. Scoring    — Bayesian posterior + Occam complexity penalty
      3. Ranking    — sort, annotate with reasoning chain + recommendation
    """

    def __init__(
        self,
        concept_registry=None,
        lambda_occam: float = 1.2,
        min_posterior_threshold: float = 0.05,
    ):
        """
        Parameters
        ----------
        concept_registry    : C++ ConceptRegistry object (optional).
        lambda_occam        : Occam penalty weight (higher = prefer simpler).
        min_posterior_threshold : Discard hypotheses below this posterior.
        """
        self.registry  = concept_registry
        self.lambda_   = lambda_occam
        self.min_post  = min_posterior_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def explain(
        self,
        observed_transforms: List[str],
        observed_delta: float,
        domain: str = "general",
        top_k: int = 5,
    ) -> List[AbductiveHypothesis]:
        """
        Generate and rank abductive hypotheses explaining the observed outcome.

        Parameters
        ----------
        observed_transforms : transforms SARE applied to reach the result.
        observed_delta      : total energy reduction achieved.
        domain              : problem domain string.
        top_k               : return at most this many hypotheses.

        Returns
        -------
        List of AbductiveHypothesis sorted best-first (lowest occam_score).
        """
        if not observed_transforms and observed_delta <= 0:
            return [self._no_change_hypothesis(domain)]

        candidates: List[AbductiveHypothesis] = []

        # Stage 1: Generate candidates from ConceptRegistry
        candidates += self._generate_from_registry(
            observed_transforms, observed_delta, domain
        )

        # Stage 2: Generate candidates from built-in prior knowledge
        candidates += self._generate_from_priors(
            observed_transforms, observed_delta, domain
        )

        # Deduplicate by name (keep higher-posterior copy)
        seen: Dict[str, AbductiveHypothesis] = {}
        for h in candidates:
            if h.name not in seen or h.posterior > seen[h.name].posterior:
                seen[h.name] = h
        candidates = list(seen.values())

        # Stage 3: Filter + rank
        candidates = [h for h in candidates if h.posterior >= self.min_post]
        candidates.sort(key=lambda h: h.occam_score)

        return candidates[:top_k] if candidates else [
            self._unknown_hypothesis(observed_transforms, observed_delta, domain)
        ]

    def explain_to_dict(
        self,
        observed_transforms: List[str],
        observed_delta: float,
        domain: str = "general",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Convenience wrapper returning a JSON-serializable report."""
        hyps = self.explain(observed_transforms, observed_delta, domain, top_k)
        return {
            "observed_transforms": observed_transforms,
            "observed_delta":      round(observed_delta, 4),
            "domain":              domain,
            "hypotheses":          [h.to_dict() for h in hyps],
            "best_explanation":    hyps[0].to_dict() if hyps else None,
            "confidence_level":    self._overall_confidence(hyps),
        }

    # ── Stage 1: Generate from ConceptRegistry ───────────────────────────────

    def _generate_from_registry(
        self,
        transforms: List[str],
        delta: float,
        domain: str,
    ) -> List[AbductiveHypothesis]:
        """Query live ConceptRegistry rules and score them against observations."""
        if not self.registry:
            return []
        hypotheses: List[AbductiveHypothesis] = []
        try:
            rules = self.registry.get_rules()
            for rule in rules:
                name      = getattr(rule, "name", "")
                rule_dom  = getattr(rule, "domain", domain)
                conf      = getattr(rule, "confidence", 0.5)
                obs_count = getattr(rule, "observations", 1)

                # Does this rule appear in observed transforms?
                rule_observed = any(
                    t == name or t.startswith(name) or name in t
                    for t in transforms
                )
                evidence = 0.85 if rule_observed else 0.15

                # Does domain match?
                if rule_dom and rule_dom != domain and domain != "general":
                    evidence *= 0.5

                # Complexity = inverse of rule conciseness (shorter name = simpler)
                complexity = _rule_complexity(name)
                prior      = _DOMAIN_PRIORS.get(rule_dom, _DOMAIN_PRIORS["general"])["base_prior"]

                # Expected delta for this rule
                expected_delta = prior * 1.5 * conf
                pred_err = abs(delta / max(len(transforms), 1) - expected_delta)

                occam    = pred_err + self.lambda_ * complexity
                posterior = (evidence * conf * (1 - pred_err / max(pred_err + 1, 1)))
                posterior = max(0.0, min(1.0, posterior))

                chain = self._reasoning_chain(
                    rule_name=name,
                    domain=rule_dom,
                    conf=conf,
                    obs_count=obs_count,
                    rule_observed=rule_observed,
                    evidence=evidence,
                    delta=delta,
                    expected_delta=expected_delta,
                    pred_err=pred_err,
                    posterior=posterior,
                )

                hypotheses.append(AbductiveHypothesis(
                    name=f"registry:{name}",
                    domain=rule_dom,
                    confidence=conf,
                    complexity=complexity,
                    prediction_error=pred_err,
                    occam_score=occam,
                    evidence_strength=evidence,
                    posterior=posterior,
                    reasoning_chain=chain,
                    supporting_transforms=[name] if rule_observed else [],
                    recommended_action=_recommend(posterior),
                ))
        except Exception as e:
            log.debug("Registry hypothesis generation failed: %s", e)
        return hypotheses

    # ── Stage 2: Generate from built-in priors ───────────────────────────────

    def _generate_from_priors(
        self,
        transforms: List[str],
        delta: float,
        domain: str,
    ) -> List[AbductiveHypothesis]:
        """
        Generate hypotheses from embedded domain knowledge — covers cases
        where the registry is empty or doesn't have relevant rules yet.
        """
        hypotheses: List[AbductiveHypothesis] = []
        prior_info = _DOMAIN_PRIORS.get(domain, _DOMAIN_PRIORS["general"])
        expected_delta_per = prior_info["expected_delta_per_rule"]
        base_prior = prior_info["base_prior"]

        for t in transforms:
            # Direct match
            evidence   = 0.9
            complexity = _rule_complexity(t)
            expected   = expected_delta_per
            pred_err   = abs(delta / max(len(transforms), 1) - expected)
            occam      = pred_err + self.lambda_ * complexity
            posterior  = max(0.0, min(1.0, base_prior * evidence * (1 - pred_err / max(pred_err + 2, 1))))

            # Check for co-occurrence bonus
            cooccurring = _RULE_CLUSTERS.get(t, set())
            support = [s for s in transforms if s in cooccurring and s != t]
            if support:
                posterior = min(1.0, posterior * 1.3)
                evidence  = min(1.0, evidence + 0.1)

            chain = [
                f"Transform '{t}' was applied during this solve.",
                f"Domain '{domain}': expected ~{expected:.2f} energy per rule step.",
                f"Observed delta per step: {delta / max(len(transforms),1):.2f}.",
                f"Prediction error: {pred_err:.3f}.",
                *([f"Co-occurring rules {support} strengthen this hypothesis."] if support else []),
                f"Posterior probability: {posterior:.2%}.",
            ]

            hypotheses.append(AbductiveHypothesis(
                name=f"prior:{t}",
                domain=domain,
                confidence=base_prior,
                complexity=complexity,
                prediction_error=pred_err,
                occam_score=occam,
                evidence_strength=evidence,
                posterior=posterior,
                reasoning_chain=chain,
                supporting_transforms=support + [t],
                recommended_action=_recommend(posterior),
            ))

        # Add a composite "multi-rule interaction" hypothesis if > 2 transforms
        if len(transforms) > 2:
            composite_posterior = min(0.95, base_prior * 0.8 + 0.1 * len(transforms))
            composite_complexity = sum(_rule_complexity(t) for t in transforms)
            pred_err = abs(delta - expected_delta_per * len(transforms))
            occam = pred_err + self.lambda_ * composite_complexity

            hypotheses.append(AbductiveHypothesis(
                name=f"composite:{domain}_chain",
                domain=domain,
                confidence=composite_posterior,
                complexity=composite_complexity,
                prediction_error=pred_err,
                occam_score=occam,
                evidence_strength=0.7,
                posterior=composite_posterior,
                reasoning_chain=[
                    f"Multiple transforms ({len(transforms)}) were chained together.",
                    f"This suggests a compound simplification pattern in '{domain}'.",
                    f"Composite delta {delta:.2f} vs expected {expected_delta_per * len(transforms):.2f}.",
                    f"Posterior: {composite_posterior:.2%} — this is an emergent behaviour.",
                ],
                supporting_transforms=transforms,
                recommended_action=_recommend(composite_posterior),
            ))

        return hypotheses

    # ── Fallback hypotheses ───────────────────────────────────────────────────

    def _no_change_hypothesis(self, domain: str) -> AbductiveHypothesis:
        return AbductiveHypothesis(
            name="no_simplification", domain=domain, confidence=1.0,
            complexity=0.0, prediction_error=0.0, occam_score=0.0,
            evidence_strength=1.0, posterior=1.0,
            reasoning_chain=[
                "No transforms were applied and energy did not change.",
                "The expression is already in its minimal form.",
                "No causal explanation is needed — this is a fixed point.",
            ],
            supporting_transforms=[],
            recommended_action="accept",
        )

    def _unknown_hypothesis(
        self, transforms: List[str], delta: float, domain: str
    ) -> AbductiveHypothesis:
        return AbductiveHypothesis(
            name="unknown_pattern", domain=domain, confidence=0.1,
            complexity=2.0, prediction_error=delta, occam_score=delta + 2.0,
            evidence_strength=0.1, posterior=0.1,
            reasoning_chain=[
                f"Transforms {transforms} were observed but no matching rule was found.",
                f"Energy changed by {delta:.3f} — this may be an undiscovered rule.",
                "Recommend: add more training episodes to discover this pattern.",
                "Action: ReflectionEngine should attempt rule extraction from this trace.",
            ],
            supporting_transforms=transforms,
            recommended_action="verify",
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _reasoning_chain(
        self, rule_name, domain, conf, obs_count, rule_observed,
        evidence, delta, expected_delta, pred_err, posterior,
    ) -> List[str]:
        chain = [
            f"Rule '{rule_name}' is a '{domain}' rule with confidence {conf:.2%}.",
            f"It has been verified {obs_count:,} times historically.",
        ]
        if rule_observed:
            chain.append(f"✓ This rule was directly observed in the solve trace.")
        else:
            chain.append(f"✗ This rule was NOT in the solve trace — inferred indirectly.")
        chain += [
            f"Evidence strength: {evidence:.2%}.",
            f"Expected energy reduction: {expected_delta:.2f}, observed: {delta:.2f}.",
            f"Prediction error: {pred_err:.3f}.",
            f"Bayesian posterior (P(rule | evidence)): {posterior:.2%}.",
            f"Recommendation: {_recommend(posterior)}.",
        ]
        return chain

    def _overall_confidence(self, hyps: List[AbductiveHypothesis]) -> str:
        if not hyps:
            return "none"
        best = hyps[0].posterior
        if best >= 0.8:
            return "high"
        if best >= 0.5:
            return "medium"
        if best >= 0.2:
            return "low"
        return "very_low"


# ── Utility functions ─────────────────────────────────────────────────────────

def _rule_complexity(name: str) -> float:
    """Simple proxy: longer/deeper rule names = more complex."""
    parts = name.replace(":", "_").split("_")
    return 0.5 + len(parts) * 0.2


def _recommend(posterior: float) -> str:
    if posterior >= 0.75:
        return "accept"
    if posterior >= 0.35:
        return "verify"
    return "reject"
