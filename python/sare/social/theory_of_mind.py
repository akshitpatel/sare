"""
Theory of Mind (AGI Gap #5 — Social Reasoning)

Gives SARE-HX the ability to model OTHER agents:
  - What does Agent A *believe* is true?
  - What does Agent A *want*?
  - What action might Agent A *take*?
  - Does Agent B know that Agent A wants X?

This is essential for:
  - Multi-agent problem solving
  - Dialogue / negotiation reasoning
  - Understanding deception, cooperation, trust
  - Passing the "false belief test" (basic ToM benchmark)

Architecture:
  - Each agent is represented as a `BeliefGraph` (a SARE-HX graph
    where nodes are propositions and edges are belief/desire/intention links).
  - The `TheoryOfMindEngine` maintains a registry of known agents.
  - It can query "what would Agent X do given world state W?".
  - Uses the LLM bridge to predict agent behaviors in novel situations.

Reference: Premack & Woodruff (1978) Theory of Mind;
           Winogrande dataset for ToM evaluation.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

log = logging.getLogger(__name__)


# ── Mental States ─────────────────────────────────────────────────────────────

class Proposition:
    """A statement that can be believed, desired, or intended."""
    def __init__(self, content: str, truth_value: Optional[bool] = None, confidence: float = 1.0):
        self.content = content
        self.truth_value = truth_value  # None = unknown
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {"content": self.content, "truth": self.truth_value, "conf": self.confidence}


class MentalState:
    """Represents an agent's complete mental state: beliefs, desires, intentions."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.beliefs: List[Proposition] = []      # What the agent thinks is true
        self.desires: List[Proposition] = []       # What the agent wants
        self.intentions: List[Proposition] = []    # What the agent plans to do
        self.knowledge: Dict[str, Any] = {}        # Facts the agent explicitly knows

    def add_belief(self, content: str, truth: bool = True, confidence: float = 0.9):
        self.beliefs.append(Proposition(content, truth, confidence))

    def add_desire(self, content: str):
        self.desires.append(Proposition(content, None, 1.0))

    def add_intention(self, content: str, confidence: float = 0.8):
        self.intentions.append(Proposition(content, True, confidence))

    def believes(self, content: str) -> Optional[bool]:
        """Check if agent believes a proposition. Returns True/False/None."""
        content_lower = content.lower().strip()
        for b in self.beliefs:
            if content_lower in b.content.lower():
                return b.truth_value
        return None  # Unknown

    def wants(self, content: str) -> bool:
        content_lower = content.lower().strip()
        return any(content_lower in d.content.lower() for d in self.desires)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "beliefs": [b.to_dict() for b in self.beliefs],
            "desires": [d.to_dict() for d in self.desires],
            "intentions": [i.to_dict() for i in self.intentions],
        }


# ── Theory of Mind Engine ─────────────────────────────────────────────────────

class TheoryOfMindEngine:
    """
    Core ToM system. Maintains a registry of known agents and their mental states.
    Enables querying what an agent believes, wants, or would do in a given situation.
    """

    PERSIST_PATH = Path(__file__).resolve().parents[3] / "data" / "memory" / "theory_of_mind.json"

    def __init__(self):
        self._agents: Dict[str, MentalState] = {}
        # Create the "self" agent (SARE-HX's own mental state)
        self._self = MentalState("sare_hx")
        self._self.add_belief("I am a reasoning system", True, 1.0)
        self._self.add_desire("solve problems correctly")
        self._self.add_desire("learn from every experience")
        self._agents["sare_hx"] = self._self

    def register_agent(self, agent_id: str, description: str = "") -> MentalState:
        """Create a new agent model."""
        if agent_id not in self._agents:
            state = MentalState(agent_id)
            if description:
                state.knowledge["description"] = description
            self._agents[agent_id] = state
            log.info(f"ToM: Registered agent '{agent_id}'")
        return self._agents[agent_id]

    def get_agent(self, agent_id: str) -> Optional[MentalState]:
        return self._agents.get(agent_id)

    def false_belief_test(self, agent_id: str, reality: str, agent_belief: str) -> dict:
        """
        Classic 'Sally-Anne' false belief test.
        Checks if the system can reason that an agent's belief differs from reality.

        Returns: { agent_believes: ..., reality_is: ..., discrepancy: bool }
        """
        agent = self.get_agent(agent_id)
        if not agent:
            return {"error": f"Unknown agent '{agent_id}'"}

        agent_believes = agent.believes(agent_belief)
        discrepancy = agent_believes is not None and str(agent_believes) != str(True)
        return {
            "agent": agent_id,
            "agent_belief": agent_belief,
            "agent_believes_it": agent_believes,
            "reality": reality,
            "discrepancy_detected": discrepancy,
            "conclusion": (
                f"{agent_id} FALSELY believes '{agent_belief}' even though reality is '{reality}'"
                if discrepancy else
                f"{agent_id}'s belief aligns with reality: '{reality}'"
            )
        }

    def predict_action_llm(self, agent_id: str, situation: str) -> str:
        """
        Use the LLM bridge to predict what agent_id would do in a situation,
        given their known beliefs, desires, and intentions.
        """
        agent = self.get_agent(agent_id)
        if not agent:
            return f"Unknown agent '{agent_id}'"

        from sare.interface.llm_bridge import _call_llm
        beliefs_str = "; ".join(b.content for b in agent.beliefs[:5])
        desires_str = "; ".join(d.content for d in agent.desires[:3])

        prompt = (
            f"You are simulating the behavior of agent '{agent_id}'.\n"
            f"Their beliefs: {beliefs_str}\n"
            f"Their desires: {desires_str}\n"
            f"Situation: {situation}\n\n"
            f"Return ONLY a JSON object:\n"
            f'{{"predicted_action": "...", "reasoning": "one sentence", "confidence": 0.0-1.0}}'
        )
        try:
            import re, json as _json
            raw = _call_llm(prompt)
            raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
            return _json.loads(raw).get("predicted_action", raw)
        except Exception:
            return f"Agent '{agent_id}' would likely act to satisfy: {desires_str}"

    def reason_about_knowledge(self, observer_id: str, target_id: str, proposition: str) -> dict:
        """
        Higher-order ToM: Does observer know that target believes proposition?
        Example: Does Alice know that Bob thinks the ball is in the basket?
        """
        observer = self.get_agent(observer_id)
        target = self.get_agent(target_id)

        if not observer or not target:
            return {"error": "Unknown agent(s)"}

        target_believes = target.believes(proposition)
        observer_knows = observer.believes(f"{target_id} believes {proposition}")

        return {
            "observer": observer_id,
            "target": target_id,
            "proposition": proposition,
            f"{target_id}_believes": target_believes,
            f"{observer_id}_knows_that": observer_knows,
            "higher_order_tom_resolved": target_believes is not None,
        }

    def infer_beliefs_from_text(self, agent_id: str, text: str):
        """
        Use LLM to extract beliefs/desires/intentions from natural text
        describing an agent, and populate their mental state model.
        """
        agent = self.register_agent(agent_id)
        from sare.interface.llm_bridge import _call_llm
        import re, json as _json

        prompt = (
            f"Extract the mental state of '{agent_id}' from the following text.\n"
            f"Return ONLY JSON:\n"
            f'{{"beliefs": ["..."], "desires": ["..."], "intentions": ["..."]}}\n\n'
            f"Text: {text[:2000]}"
        )
        try:
            raw = _call_llm(prompt)
            raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
            data = _json.loads(raw)
            for b in data.get("beliefs", []):
                agent.add_belief(str(b))
            for d in data.get("desires", []):
                agent.add_desire(str(d))
            for i in data.get("intentions", []):
                agent.add_intention(str(i))
            log.info(f"ToM: Inferred {len(data.get('beliefs', []))} beliefs for '{agent_id}'")
        except Exception as e:
            log.warning(f"ToM belief inference failed for '{agent_id}': {e}")

    def save(self):
        self.PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {aid: ms.to_dict() for aid, ms in self._agents.items()}
        with open(self.PERSIST_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        log.info(f"TheoryOfMindEngine saved: {len(self._agents)} agents")

    def load(self):
        if not self.PERSIST_PATH.exists():
            return
        with open(self.PERSIST_PATH) as f:
            data = json.load(f)
        for agent_id, ms_data in data.items():
            ms = MentalState(agent_id)
            for b in ms_data.get("beliefs", []):
                ms.beliefs.append(Proposition(b["content"], b.get("truth"), b.get("conf", 0.9)))
            for d in ms_data.get("desires", []):
                ms.desires.append(Proposition(d["content"]))
            for i in ms_data.get("intentions", []):
                ms.intentions.append(Proposition(i["content"], True, i.get("conf", 0.8)))
            self._agents[agent_id] = ms
        log.info(f"TheoryOfMindEngine loaded: {len(self._agents)} agents")

    def summary(self) -> dict:
        return {
            "total_agents": len(self._agents),
            "agent_ids": list(self._agents.keys()),
            "sare_beliefs": len(self._self.beliefs),
            "sare_desires": len(self._self.desires),
        }
