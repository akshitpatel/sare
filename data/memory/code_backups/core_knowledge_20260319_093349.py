"""
CoreKnowledgePrior — Spelke's 4 Core Systems as Transform Bias Weights
=======================================================================

Human infants are born with innate knowledge of:
  1. NUMBER   — counting, quantity, arithmetic intuition
  2. OBJECT   — object permanence, identity preservation
  3. CAUSALITY — causal relationships, Occam's razor (shortest derivation)
  4. AGENT    — goal-directed behaviour, energy-gradient reasoning

These are implemented as *bias weights* that re-order the transform list
during search — NOT hardcoded rules. The system can still discover any
transform; priors just influence trial order.

At higher developmental stages, learned experience overrides these priors
(epistemic curiosity takes over from innate bias).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ── Core system → transform name biases ─────────────────────────────────────
# Each entry: transform_name_fragment → (core_system, min_stage_level, bias)
#
# bias > 1.0 → prefer earlier in list
# bias < 1.0 → prefer later (normally not used for innate priors)

_CORE_BIASES: List[dict] = [
    # NUMBER system (active from INFANT)
    {"fragment": "AddZeroElim",        "system": "NUMBER",    "min_stage": 0, "bias": 2.5},
    {"fragment": "MulOneElim",         "system": "NUMBER",    "min_stage": 0, "bias": 2.5},
    {"fragment": "CombineLikeTerms",   "system": "NUMBER",    "min_stage": 0, "bias": 2.0},
    {"fragment": "ConstantFold",       "system": "NUMBER",    "min_stage": 0, "bias": 2.0},
    {"fragment": "add_zero",           "system": "NUMBER",    "min_stage": 0, "bias": 2.5},
    {"fragment": "mul_one",            "system": "NUMBER",    "min_stage": 0, "bias": 2.5},
    {"fragment": "combine_like",       "system": "NUMBER",    "min_stage": 0, "bias": 2.0},
    {"fragment": "const_fold",         "system": "NUMBER",    "min_stage": 0, "bias": 2.0},
    # OBJECT system (active from INFANT) — identity / annihilation transforms
    {"fragment": "Identity",           "system": "OBJECT",    "min_stage": 0, "bias": 2.3},
    {"fragment": "MulZeroElim",        "system": "OBJECT",    "min_stage": 0, "bias": 2.2},
    {"fragment": "SubSelf",            "system": "OBJECT",    "min_stage": 0, "bias": 2.2},
    {"fragment": "mul_zero",           "system": "OBJECT",    "min_stage": 0, "bias": 2.2},
    {"fragment": "sub_self",           "system": "OBJECT",    "min_stage": 0, "bias": 2.2},
    # CAUSALITY system (active from TODDLER = stage 1)
    # Shorter chains → prefer transforms that collapse depth quickly
    {"fragment": "Simplif",            "system": "CAUSALITY", "min_stage": 1, "bias": 1.8},
    {"fragment": "Elim",               "system": "CAUSALITY", "min_stage": 1, "bias": 1.6},
    {"fragment": "Cancel",             "system": "CAUSALITY", "min_stage": 1, "bias": 1.7},
    {"fragment": "cancel",             "system": "CAUSALITY", "min_stage": 1, "bias": 1.7},
    # AGENT system (active from CHILD = stage 2) — energy-gradient
    {"fragment": "Equation",           "system": "AGENT",     "min_stage": 2, "bias": 1.5},
    {"fragment": "equation",           "system": "AGENT",     "min_stage": 2, "bias": 1.5},
    {"fragment": "Distributive",       "system": "AGENT",     "min_stage": 2, "bias": 1.4},
    {"fragment": "distributive",       "system": "AGENT",     "min_stage": 2, "bias": 1.4},
]

# ── Pre-loaded innate facts injected into WorldModel at boot ─────────────────
_INNATE_FACTS: Dict[str, List[dict]] = {
    "identity_basics": [
        {"fact": "x + 0 = x", "confidence": 1.0, "core_system": "NUMBER"},
        {"fact": "x * 1 = x", "confidence": 1.0, "core_system": "NUMBER"},
        {"fact": "x * 0 = 0", "confidence": 1.0, "core_system": "OBJECT"},
        {"fact": "x - x = 0", "confidence": 1.0, "core_system": "OBJECT"},
    ],
    "causality": [
        {"fact": "shorter derivation preferred (Occam)", "confidence": 0.9, "core_system": "CAUSALITY"},
        {"fact": "each step reduces complexity", "confidence": 0.85, "core_system": "CAUSALITY"},
    ],
    "agency": [
        {"fact": "goal = minimize energy", "confidence": 0.95, "core_system": "AGENT"},
        {"fact": "transforms are goal-directed actions", "confidence": 0.9, "core_system": "AGENT"},
    ],
}


class CoreKnowledgePrior:
    """
    Spelke's 4 core knowledge systems encoded as transform bias weights.

    Does NOT hardcode rules — only influences search order.
    At higher stages (CHILD+), learned world-model predictions gradually
    replace these innate biases.
    """

    def get_transform_priors(self, stage_level: int) -> Dict[str, float]:
        """
        Return per-transform-name-fragment bias weights for the given stage level.

        stage_level: 0=INFANT, 1=TODDLER, 2=CHILD, 3=PRETEEN, ...
        """
        priors: Dict[str, float] = {}
        for entry in _CORE_BIASES:
            if stage_level >= entry["min_stage"]:
                priors[entry["fragment"]] = entry["bias"]
        return priors

    def get_innate_facts(self, domain: str) -> List[dict]:
        """Return innate facts for a domain (to pre-seed WorldModel at boot)."""
        return list(_INNATE_FACTS.get(domain, []))

    def get_all_innate_facts(self) -> Dict[str, List[dict]]:
        """Return all innate facts for all domains."""
        return {k: list(v) for k, v in _INNATE_FACTS.items()}

    def apply_priors_to_search(self, transforms: list, stage_level: int) -> list:
        """
        Re-order transform list so core-knowledge-preferred transforms come first.

        At INFANT/TODDLER (stages 0-1): priors dominate ordering.
        At CHILD+ (stage 2+): priors decay — learned world-model ordering
        already provided by _heuristic_reorder_transforms in ExperimentRunner;
        we only apply a soft boost here.

        Returns a new sorted list (does not mutate input).
        """
        if not transforms:
            return transforms

        priors = self.get_transform_priors(stage_level)
        if not priors:
            return transforms

        # Decay factor: at higher stages priors matter less
        # stage 0: factor=1.0, stage 2: factor=0.4, stage 5+: factor=0.05
        decay = max(0.05, 1.0 - stage_level * 0.19)

        def _score(t) -> float:
            name = ""
            if hasattr(t, "name"):
                try:
                    name = t.name() if callable(t.name) else str(t.name)
                except Exception:
                    name = str(t)
            elif hasattr(t, "__class__"):
                name = t.__class__.__name__
            score = 1.0
            for fragment, bias in priors.items():
                if fragment.lower() in name.lower():
                    score = max(score, 1.0 + (bias - 1.0) * decay)
                    break
            return score

        return sorted(transforms, key=_score, reverse=True)


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: Optional[CoreKnowledgePrior] = None


def get_core_knowledge() -> CoreKnowledgePrior:
    global _instance
    if _instance is None:
        _instance = CoreKnowledgePrior()
    return _instance
