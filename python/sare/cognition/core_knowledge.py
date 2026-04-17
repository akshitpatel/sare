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

DEVELOPMENTAL LOOP:
This module now supports Piagetian "assimilation/accommodation". 
When the PredictiveEngine observes low surprise (success), the corresponding 
core bias is reinforced. High surprise causes bias decay.
"""
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_MEMORY = Path(__file__).resolve().parents[4] / "data" / "memory"

# ── Core system → transform name biases ─────────────────────────────────────
# Each entry: transform_name_fragment → (core_system, min_stage_level, bias)

_DEFAULT_BIASES: List[dict] = [
    # NUMBER system (active from INFANT)
    {"fragment": "AddZeroElim",        "system": "NUMBER",    "min_stage": 0, "bias": 2.5},
    {"fragment": "MulOneElim",         "system": "NUMBER",    "min_stage": 0, "bias": 2.5},
    {"fragment": "CombineLikeTerms",   "system": "NUMBER",    "min_stage": 0, "bias": 2.0},
    {"fragment": "ConstantFold",       "system": "NUMBER",    "min_stage": 0, "bias": 2.0},
    {"fragment": "add_zero",           "system": "NUMBER",    "min_stage": 0, "bias": 2.5},
    {"fragment": "mul_one",            "system": "NUMBER",    "min_stage": 0, "bias": 2.5},
    {"fragment": "combine_like",       "system": "NUMBER",    "min_stage": 0, "bias": 2.0},
    {"fragment": "const_fold",         "system": "NUMBER",    "min_stage": 0, "bias": 2.0},
    # OBJECT system (active from INFANT)
    {"fragment": "Identity",           "system": "OBJECT",    "min_stage": 0, "bias": 2.3},
    {"fragment": "MulZeroElim",        "system": "OBJECT",    "min_stage": 0, "bias": 2.2},
    {"fragment": "SubSelf",            "system": "OBJECT",    "min_stage": 0, "bias": 2.2},
    {"fragment": "mul_zero",           "system": "OBJECT",    "min_stage": 0, "bias": 2.2},
    {"fragment": "sub_self",           "system": "OBJECT",    "min_stage": 0, "bias": 2.2},
    # CAUSALITY system (active from TODDLER = stage 1)
    {"fragment": "Simplif",            "system": "CAUSALITY", "min_stage": 1, "bias": 1.8},
    {"fragment": "Elim",               "system": "CAUSALITY", "min_stage": 1, "bias": 1.6},
    {"fragment": "Cancel",             "system": "CAUSALITY", "min_stage": 1, "bias": 1.7},
    {"fragment": "cancel",             "system": "CAUSALITY", "min_stage": 1, "bias": 1.7},
    # AGENT system (active from CHILD = stage 2)
    {"fragment": "Equation",           "system": "AGENT",     "min_stage": 2, "bias": 1.5},
    {"fragment": "equation",           "system": "AGENT",     "min_stage": 2, "bias": 1.5},
    {"fragment": "Distributive",       "system": "AGENT",     "min_stage": 2, "bias": 1.4},
    {"fragment": "distributive",       "system": "AGENT",     "min_stage": 2, "bias": 1.4},
]

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
    Supports adaptive bias adjustment based on prediction error (accommodation).
    """

    PERSIST_PATH = _MEMORY / "core_knowledge_state.json"

    def __init__(self):
        self._biases = [dict(b) for b in _DEFAULT_BIASES]
        self._load()

    def get_transform_priors(self, stage_level: int) -> Dict[str, float]:
        priors: Dict[str, float] = {}
        for entry in self._biases:
            if stage_level >= entry["min_stage"]:
                priors[entry["fragment"]] = entry["bias"]
        return priors

    def adjust_bias(self, transform_name: str, delta: float, min_val: float = 0.5, max_val: float = 5.0):
        """
        Adjust bias for a transform fragment based on experience.
        delta > 0 reinforces (assimilation), delta < 0 decays (accommodation).
        """
        updated = False
        for entry in self._biases:
            if entry["fragment"].lower() in transform_name.lower():
                old_bias = entry["bias"]
                entry["bias"] = max(min_val, min(max_val, entry["bias"] + delta))
                if entry["bias"] != old_bias:
                    updated = True
        
        if updated:
            self._save()

    def get_core_system_for_transform(self, transform_name: str) -> Optional[str]:
        """Identify which core system a transform belongs to."""
        for entry in self._biases:
            if entry["fragment"].lower() in transform_name.lower():
                return entry["system"]
        return None

    def get_innate_facts(self, domain: str) -> List[dict]:
        return list(_INNATE_FACTS.get(domain, []))

    def get_all_innate_facts(self) -> Dict[str, List[dict]]:
        return {k: list(v) for k, v in _INNATE_FACTS.items()}

    def apply_priors_to_search(self, transforms: list, stage_level: int) -> list:
        if not transforms:
            return transforms

        priors = self.get_transform_priors(stage_level)
        if not priors:
            return transforms

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

    def _save(self):
        try:
            _MEMORY.mkdir(parents=True, exist_ok=True)
            data = {"biases": self._biases}
            self.PERSIST_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.debug("[CoreKnowledge] Save error: %s", e)

    def _load(self):
        if not self.PERSIST_PATH.exists():
            return
        try:
            data = json.loads(self.PERSIST_PATH.read_text())
            self._biases = data.get("biases", self._biases)
        except Exception as e:
            log.debug("[CoreKnowledge] Load error: %s", e)


_instance: Optional[CoreKnowledgePrior] = None


def get_core_knowledge() -> CoreKnowledgePrior:
    global _instance
    if _instance is None:
        _instance = CoreKnowledgePrior()
    return _instance