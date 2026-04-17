"""
HypothesisEngine — Forward model for hypothesis-driven search.

Before BeamSearch runs, generates 3-5 hypotheses about what the simplified
form might look like. The beam gets a small bonus when candidates structurally
match a hypothesis — steering search toward known normal forms.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_DOMAIN_HINTS: Dict[str, List[tuple]] = {
    # (description, expected_node_types, expected_max_depth)
    "algebra":     [("simplified", ["VAR", "CONST", "ADD", "MUL"], 5),
                    ("single_var",  ["VAR"], 1)],
    "logic":       [("true_false",  ["BOOL_TRUE", "BOOL_FALSE", "TRUE", "FALSE"], 1),
                    ("cnf",         ["AND", "OR", "NOT", "VAR"], 4)],
    "calculus":    [("constant",    ["CONST", "NUMBER", "0"], 1),
                    ("power_form",  ["MUL", "VAR", "CONST", "POW"], 4)],
    "physics":     [("equation",    ["EQ", "MUL", "VAR", "CONST"], 4)],
    "arithmetic":  [("number",      ["CONST", "NUMBER"], 1)],
}


class Hypothesis:
    __slots__ = ("description", "expected_ops", "max_depth", "confidence")

    def __init__(self, description: str, expected_ops: List[str],
                 max_depth: int, confidence: float):
        self.description  = description
        self.expected_ops = set(expected_ops)
        self.max_depth    = max_depth
        self.confidence   = confidence

    def match_bonus(self, graph, bonus: float = 0.08) -> float:
        """Return energy-equivalent bonus [0, bonus] if graph matches hypothesis."""
        try:
            nodes = list(graph.nodes) if hasattr(graph, "nodes") else []
            if not nodes:
                return 0.0
            depth = len(nodes)
            if depth > self.max_depth * 2:
                return 0.0   # graph is too large to match
            actual_types = {getattr(n, "node_type", "") for n in nodes}
            if self.expected_ops:
                overlap = len(actual_types & self.expected_ops) / len(self.expected_ops)
            else:
                overlap = 0.5
            depth_ok = 1.0 if depth <= self.max_depth else max(0.0, 1.0 - (depth - self.max_depth) / 5)
            return bonus * self.confidence * (0.6 * overlap + 0.4 * depth_ok)
        except Exception:
            return 0.0


class HypothesisEngine:
    def __init__(self):
        self._hits  = 0
        self._total = 0
        self._wm    = None

    def _world_model(self):
        if self._wm is None:
            try:
                from sare.memory.world_model import get_world_model
                self._wm = get_world_model()
            except Exception:
                pass
        return self._wm

    def generate(self, graph, domain: str = "general",
                 recent_transforms: Optional[List[str]] = None) -> List[Hypothesis]:
        hyps: List[Hypothesis] = []
        depth = len(list(graph.nodes)) if hasattr(graph, "nodes") else 5

        # Domain normal forms
        for desc, ops, max_d in _DOMAIN_HINTS.get(domain, []):
            hyps.append(Hypothesis(desc, ops, max_d, 0.55))

        # World model causal prediction
        wm = self._world_model()
        if wm is not None:
            try:
                pred = wm.predict_transform(domain, None)
                if pred:
                    hyps.append(Hypothesis(f"wm:{pred}", [pred, "VAR", "CONST"],
                                           max(1, depth - 2), 0.65))
            except Exception:
                pass

        # Structural reduction heuristic
        if depth > 3:
            hyps.append(Hypothesis("reduce", [], max(1, depth // 2), 0.40))

        # Recent transform chain prediction
        if recent_transforms:
            last = recent_transforms[-1].lower()
            if any(k in last for k in ("zero", "identity", "elim", "fold")):
                hyps.append(Hypothesis("identity_chain", ["VAR", "CONST"],
                                       max(1, depth - 1), 0.50))

        return hyps[:5]

    def apply_bonuses(self, candidates: list, hyps: List[Hypothesis]) -> list:
        """
        Re-rank beam candidates by subtracting hypothesis match bonus from energy.
        candidates: list of objects with .energy and .graph attributes.
        Returns sorted list (lowest adjusted energy first).
        """
        if not hyps or not candidates:
            return candidates
        try:
            scored = []
            for c in candidates:
                bonus = sum(h.match_bonus(c.graph) for h in hyps)
                scored.append((c.energy - bonus, c))
            scored.sort(key=lambda x: x[0])
            return [c for _, c in scored]
        except Exception:
            return candidates

    def record(self, solved: bool) -> None:
        self._total += 1
        if solved:
            self._hits += 1

    def summary(self) -> dict:
        return {
            "total": self._total,
            "hits": self._hits,
            "hit_rate": round(self._hits / max(self._total, 1), 3),
        }


_instance: Optional[HypothesisEngine] = None

def get_hypothesis_engine() -> HypothesisEngine:
    global _instance
    if _instance is None:
        _instance = HypothesisEngine()
    return _instance
