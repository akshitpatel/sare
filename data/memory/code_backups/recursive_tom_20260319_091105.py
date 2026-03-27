"""
RecursiveToM — S28-3
Recursive Theory of Mind up to depth 3.

Depth 0: my beliefs about the world  (self_model)
Depth 1: my model of what agent X believes
Depth 2: my model of what X believes I believe
Depth 3: my model of what X believes Y believes Z believes

BeliefNode stores a probability-weighted claim.
ToMModel per (agent_id, depth) holds a set of BeliefNodes.

Key operations:
  update_model(agent_id, observation, depth)
    → update belief about what agent_id knows at this depth
  predict_action(agent_id, context, depth)
    → estimate agent_id's next move given their inferred beliefs
  resolve_disagreement(agent_id_a, agent_id_b, topic)
    → propose negotiation strategy using depth-2 model
"""
from __future__ import annotations

import time
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_MAX_DEPTH   = 3
_DECAY_RATE  = 0.05   # belief confidence decays per tick
_MIN_CONF    = 0.05


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class BeliefNode:
    claim:      str
    confidence: float
    domain:     str   = "general"
    depth:      int   = 0
    source:     str   = "observation"
    ts:         float = field(default_factory=time.time)

    def decay(self) -> None:
        self.confidence = max(_MIN_CONF, self.confidence * (1 - _DECAY_RATE))

    def to_dict(self) -> dict:
        return {
            "claim":      self.claim[:80],
            "confidence": round(self.confidence, 3),
            "domain":     self.domain,
            "depth":      self.depth,
            "source":     self.source,
        }


@dataclass
class ToMModel:
    """Theory-of-Mind model for one agent at one recursion depth."""
    agent_id: str
    depth:    int
    beliefs:  List[BeliefNode] = field(default_factory=list)
    updates:  int = 0
    last_updated: float = field(default_factory=time.time)

    _CAPACITY = 20

    def add(self, belief: BeliefNode) -> None:
        # merge if claim already exists
        for b in self.beliefs:
            if b.claim == belief.claim:
                b.confidence = min(1.0, (b.confidence + belief.confidence) / 1.5)
                b.ts = time.time()
                return
        self.beliefs.append(belief)
        if len(self.beliefs) > self._CAPACITY:
            self.beliefs.sort(key=lambda b: b.confidence)
            self.beliefs.pop(0)
        self.updates      += 1
        self.last_updated  = time.time()

    def tick_decay(self) -> None:
        for b in self.beliefs:
            b.decay()
        self.beliefs = [b for b in self.beliefs if b.confidence > _MIN_CONF]

    def top_beliefs(self, n: int = 5) -> List[BeliefNode]:
        return sorted(self.beliefs, key=lambda b: -b.confidence)[:n]

    def to_dict(self) -> dict:
        return {
            "agent_id":    self.agent_id,
            "depth":       self.depth,
            "n_beliefs":   len(self.beliefs),
            "updates":     self.updates,
            "top_beliefs": [b.to_dict() for b in self.top_beliefs(3)],
        }


# ── action prediction ─────────────────────────────────────────────────────────

_ACTION_TEMPLATES = {
    "agree":     "Agent {a} likely agrees: high confidence in shared domain {d}",
    "challenge": "Agent {a} likely challenges: conflicting belief on {d}",
    "abstain":   "Agent {a} likely abstains: low confidence on {d}",
    "teach":     "Agent {a} may teach: sole owner of high-confidence belief on {d}",
}


def _predict_from_beliefs(agent_id: str, beliefs: List[BeliefNode],
                          context: str) -> Tuple[str, float, str]:
    """Return (action, confidence, rationale)."""
    if not beliefs:
        return "abstain", 0.3, "no beliefs on topic"

    ctx_tokens = set(context.lower().split())
    relevant   = [b for b in beliefs if set(b.domain.split()) & ctx_tokens
                  or b.domain == "general"]
    if not relevant:
        relevant = beliefs[:3]

    avg_conf = sum(b.confidence for b in relevant) / len(relevant)

    if avg_conf > 0.7:
        action = "agree"
    elif avg_conf > 0.4:
        action = "challenge"
    elif avg_conf < 0.2:
        action = "abstain"
    else:
        action = "teach"

    domain   = relevant[0].domain
    rationale = _ACTION_TEMPLATES[action].format(a=agent_id, d=domain)
    return action, round(avg_conf, 3), rationale


# ── RecursiveToM ──────────────────────────────────────────────────────────────

class RecursiveToM:
    """
    Maintains a (agent_id, depth) → ToMModel map.
    Supports up to MAX_DEPTH=3 recursive belief nesting.
    """

    def __init__(self) -> None:
        self._models:   Dict[Tuple[str, int], ToMModel] = {}
        self._agent_society = None
        self._tick_count    = 0

        self._total_updates      = 0
        self._total_predictions  = 0
        self._total_resolutions  = 0

    # ── wiring ────────────────────────────────────────────────────────────────

    def wire(self, agent_society=None) -> None:
        self._agent_society = agent_society
        if agent_society:
            self._seed_from_society()

    def _seed_from_society(self) -> None:
        """Bootstrap depth-1 models from existing AgentSociety beliefs."""
        try:
            agents = getattr(self._agent_society, '_agents', {})
            for agent_id, agent in agents.items():
                beliefs = getattr(agent, 'beliefs', [])
                for b in beliefs[:5]:
                    claim = getattr(b, 'claim', str(b))
                    conf  = getattr(b, 'confidence', 0.5)
                    self.update_model(agent_id, claim, conf, "general", depth=1)
        except Exception as e:
            log.debug(f"RecursiveToM seed: {e}")

    # ── model access ──────────────────────────────────────────────────────────

    def _get_model(self, agent_id: str, depth: int) -> ToMModel:
        key = (agent_id, depth)
        if key not in self._models:
            self._models[key] = ToMModel(agent_id, depth)
        return self._models[key]

    # ── core operations ───────────────────────────────────────────────────────

    def update_model(self, agent_id: str, claim: str, confidence: float,
                     domain: str = "general", depth: int = 1) -> None:
        """Update what I believe agent_id believes at recursion depth."""
        depth = min(depth, _MAX_DEPTH)
        model = self._get_model(agent_id, depth)
        model.add(BeliefNode(claim, confidence, domain, depth, agent_id))
        self._total_updates += 1

        # Propagate upward: depth+1 model inherits dampened version
        if depth < _MAX_DEPTH:
            self.update_model(
                agent_id, f"[via {agent_id}] {claim}",
                confidence * 0.7, domain, depth + 1
            )

    def predict_action(self, agent_id: str, context: str,
                       depth: int = 1) -> dict:
        """Predict what agent_id will do in context, using their depth-d model."""
        depth  = min(depth, _MAX_DEPTH)
        model  = self._get_model(agent_id, depth)
        action, conf, rationale = _predict_from_beliefs(
            agent_id, model.beliefs, context
        )
        self._total_predictions += 1
        return {
            "agent_id":  agent_id,
            "depth":     depth,
            "action":    action,
            "confidence": conf,
            "rationale": rationale,
        }

    def resolve_disagreement(self, agent_a: str, agent_b: str,
                             topic: str) -> dict:
        """
        Use depth-2 (A's model of what B believes A believes) to propose
        a negotiation strategy.
        """
        # A's direct view of topic (depth 1)
        pred_a = self.predict_action(agent_a, topic, depth=1)
        # A's model of B's view (depth 1 from B's perspective)
        pred_b = self.predict_action(agent_b, topic, depth=1)
        # Depth-2: what A thinks B thinks A will do
        pred_a2 = self.predict_action(agent_a, topic, depth=2)

        strategy = "explore"  # default
        if pred_a["action"] == pred_b["action"]:
            strategy = "reinforce"  # both agree → strengthen shared belief
        elif pred_a2["confidence"] > 0.6:
            strategy = "signal"     # A knows B models A correctly → signal intent
        elif pred_b["action"] == "challenge":
            strategy = "concede_partial"
        else:
            strategy = "debate"

        self._total_resolutions += 1
        return {
            "agent_a":     agent_a,
            "agent_b":     agent_b,
            "topic":       topic[:60],
            "a_predicts":  pred_a["action"],
            "b_predicts":  pred_b["action"],
            "a_depth2":    pred_a2["action"],
            "strategy":    strategy,
            "confidence":  round((pred_a["confidence"] + pred_b["confidence"]) / 2, 3),
        }

    # ── tick ──────────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """Decay all belief confidences."""
        self._tick_count += 1
        for model in self._models.values():
            model.tick_decay()

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        all_agents = set(agent_id for (agent_id, _) in self._models)
        models_by_agent = {}
        for agent_id in all_agents:
            models_by_agent[agent_id] = [
                self._get_model(agent_id, d).to_dict()
                for d in range(1, _MAX_DEPTH + 1)
                if (agent_id, d) in self._models
            ]
        return {
            "total_agents":      len(all_agents),
            "total_models":      len(self._models),
            "max_depth":         _MAX_DEPTH,
            "total_updates":     self._total_updates,
            "total_predictions": self._total_predictions,
            "total_resolutions": self._total_resolutions,
            "tick_count":        self._tick_count,
            "agents":            models_by_agent,
        }
