"""
AgentSociety — Gap 3: Rich Multi-Agent Society

Instead of agents that merely race on problems (MultiAgentArena),
this implements a true multi-agent society where agents:

  1. Have distinct BELIEF STATES — each knows different things
  2. Have distinct GOALS       — each pursues different objectives
  3. COMMUNICATE               — share knowledge via messages
  4. DEBATE                    — argue conjectures, falsify each other
  5. SPECIALIZE                — some agents focus on specific domains
  6. TEACH                     — agents with high accuracy teach others

The society as a whole knows more than any individual agent.
This is how human science works: no single person knows everything,
but the collective knowledge grows through communication and debate.

Key mechanisms:
  - Belief propagation: accepted facts spread from agent to agent
  - Goal coordination: agents avoid redundant work, divide problem space
  - Emergent consensus: conjectures need N supporters to become facts
  - Specialization: agents drift toward domains where they perform best
  - Theory of mind: each agent models what others know and don't know
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


# ── Agent models ───────────────────────────────────────────────────────────────

@dataclass
class Belief:
    """A belief held by an agent — could be a symbolic rule, a fact, or a conjecture."""
    content: str
    domain: str
    confidence: float     # 0.0 → 1.0
    source: str           # which agent first proposed this
    supporters: Set[str] = field(default_factory=set)
    falsifiers: Set[str] = field(default_factory=set)
    status: str = "hypothesis"  # 'hypothesis', 'accepted', 'rejected'
    created_at: float = field(default_factory=time.time)

    @property
    def support_ratio(self) -> float:
        total = len(self.supporters) + len(self.falsifiers)
        return len(self.supporters) / max(total, 1)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "domain": self.domain,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "supporters": list(self.supporters),
            "falsifiers": list(self.falsifiers),
            "status": self.status,
            "support_ratio": round(self.support_ratio, 3),
        }


@dataclass
class AgentMessage:
    """A message from one agent to the society."""
    sender: str
    msg_type: str        # 'belief', 'conjecture', 'question', 'teach', 'falsify'
    content: str
    domain: str
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "type": self.msg_type,
            "content": self.content,
            "domain": self.domain,
            "confidence": round(self.confidence, 3),
        }


class SocietalAgent:
    """
    A member of the AgentSociety with:
      - Belief state (what it knows and believes)
      - Specialization (best domain)
      - Goals (what it's trying to learn)
      - Communication (messages in/out)
      - Theory of Mind (what it believes others know)
    """

    def __init__(self, agent_id: str, specialization: str = "general",
                 knowledge_set: Optional[List[str]] = None):
        self.agent_id = agent_id
        self.specialization = specialization
        self._beliefs: Dict[str, Belief] = {}
        self._goals: List[str] = []
        self._inbox: deque = deque(maxlen=50)
        self._outbox: deque = deque(maxlen=50)
        self._solve_history: List[dict] = []
        self._domain_accuracy: Dict[str, float] = {}
        self._known_agents: Set[str] = set()
        self._total_solved: int = 0
        self._total_tried: int = 0

        # Seed knowledge
        for fact in (knowledge_set or []):
            self.add_belief(fact, specialization, confidence=0.8, source="seed")

    def add_belief(self, content: str, domain: str, confidence: float,
                   source: str = "self"):
        key = f"{domain}:{content[:40]}"
        if key not in self._beliefs:
            b = Belief(content=content, domain=domain,
                       confidence=confidence, source=source)
            b.supporters.add(self.agent_id)
            self._beliefs[key] = b
        else:
            # Update confidence
            self._beliefs[key].confidence = max(
                self._beliefs[key].confidence, confidence)
            self._beliefs[key].supporters.add(self.agent_id)

    def receive(self, msg: AgentMessage):
        """Process an incoming message from another agent."""
        self._inbox.append(msg)
        self._known_agents.add(msg.sender)

        if msg.msg_type in ("belief", "teach"):
            # Accept high-confidence beliefs from others
            if msg.confidence > 0.6:
                self.add_belief(msg.content, msg.domain,
                                msg.confidence * 0.85, msg.sender)
        elif msg.msg_type == "conjecture":
            # Evaluate and reply
            self._evaluate_conjecture(msg)

    def _evaluate_conjecture(self, msg: AgentMessage):
        """Decide whether to support or falsify a conjecture."""
        key = f"{msg.domain}:{msg.content[:40]}"
        existing = self._beliefs.get(key)
        if existing:
            # If we already believe it → support
            if existing.confidence > 0.5:
                reply = AgentMessage(
                    sender=self.agent_id, msg_type="support",
                    content=msg.content, domain=msg.domain,
                    confidence=existing.confidence,
                )
            else:
                reply = AgentMessage(
                    sender=self.agent_id, msg_type="falsify",
                    content=msg.content, domain=msg.domain,
                    confidence=1.0 - existing.confidence,
                )
            self._outbox.append(reply)

    def propose_conjecture(self) -> Optional[AgentMessage]:
        """Propose a belief as a conjecture to the society."""
        candidates = [b for b in self._beliefs.values()
                      if b.status == "hypothesis" and b.confidence > 0.5
                      and b.source == self.agent_id]
        if not candidates:
            return None
        b = random.choice(candidates)
        return AgentMessage(
            sender=self.agent_id, msg_type="conjecture",
            content=b.content, domain=b.domain,
            confidence=b.confidence,
        )

    def teach(self, student_id: str) -> List[AgentMessage]:
        """Share highest-confidence beliefs with a student."""
        top = sorted(self._beliefs.values(),
                     key=lambda b: b.confidence, reverse=True)[:3]
        messages = []
        for b in top:
            if b.status == "accepted":
                messages.append(AgentMessage(
                    sender=self.agent_id, msg_type="teach",
                    content=b.content, domain=b.domain,
                    confidence=b.confidence,
                ))
        return messages

    def record_solve(self, domain: str, success: bool, delta: float):
        self._total_tried += 1
        if success:
            self._total_solved += 1
        # Update domain accuracy
        old = self._domain_accuracy.get(domain, 0.5)
        alpha = 0.1  # EMA
        self._domain_accuracy[domain] = old * (1 - alpha) + (1.0 if success else 0.0) * alpha

    @property
    def best_domain(self) -> str:
        if not self._domain_accuracy:
            return self.specialization
        return max(self._domain_accuracy, key=self._domain_accuracy.get)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "specialization": self.specialization,
            "beliefs": len(self._beliefs),
            "accepted_beliefs": sum(1 for b in self._beliefs.values()
                                    if b.status == "accepted"),
            "goals": self._goals[:3],
            "inbox_size": len(self._inbox),
            "known_agents": list(self._known_agents),
            "solve_rate": round(self._total_solved / max(self._total_tried, 1), 3),
            "best_domain": self.best_domain,
            "domain_accuracy": {d: round(v, 3) for d, v in
                                 self._domain_accuracy.items()},
        }


# ── Agent Society ──────────────────────────────────────────────────────────────

_DEFAULT_SOCIETY_CONFIG = [
    # (id, specialization, knowledge_seeds)
    ("arithmetician", "arithmetic",     ["x + 0 = x", "x * 1 = x", "0 * x = 0"]),
    ("logician",      "logic",          ["A AND TRUE = A", "NOT NOT A = A", "A OR FALSE = A"]),
    ("physicist",     "mechanics",      ["F = m * a", "E = m * c^2", "v = d/t"]),
    ("algebraist",    "algebra",        ["a(b+c) = ab + ac", "a^2 - b^2 = (a+b)(a-b)"]),
    ("geometer",      "geometry",       ["a^2 + b^2 = c^2", "Area = pi * r^2"]),
]

_CONSENSUS_THRESHOLD = 0.6    # fraction of agents that must agree for a fact to be accepted
_MIN_VOTES = 2                 # minimum votes before consensus is evaluated


class AgentSociety:
    """
    A rich multi-agent society where agents communicate, debate, and teach.

    Architecture:
      ┌─────────────────────────────────────────────────────────────┐
      │  Agent₁ (arithmetic)  ←→  Agent₂ (logic)  ←→  Agent₃ (physics)  │
      │         ↕                      ↕                     ↕           │
      │              Shared Belief Blackboard                            │
      │         ↕                      ↕                     ↕           │
      │  Agent₄ (algebra)    ←→  Agent₅ (geometry)                     │
      └─────────────────────────────────────────────────────────────┘

    Communication protocols:
      - broadcast: one agent sends a message to ALL others
      - teach: high-accuracy agent shares facts with lower-accuracy agents
      - debate: conjecture proposed → others support or falsify
      - consensus: when >60% support → belief promoted to "accepted fact"
    """

    def __init__(self, n_agents: int = 3):
        n = max(2, min(n_agents, len(_DEFAULT_SOCIETY_CONFIG)))
        self._agents: Dict[str, SocietalAgent] = {}
        for aid, spec, seeds in _DEFAULT_SOCIETY_CONFIG[:n]:
            self._agents[aid] = SocietalAgent(aid, spec, seeds)

        # Shared belief blackboard
        self._blackboard: Dict[str, Belief] = {}
        self._message_log: deque = deque(maxlen=300)
        self._consensus_log: List[dict] = []
        self._debate_rounds: int = 0
        self._total_messages: int = 0
        self._tick: int = 0

    # ── Communication ──────────────────────────────────────────────────────────

    def broadcast(self, msg: AgentMessage):
        """One agent sends a message to all others."""
        self._message_log.append(msg)
        self._total_messages += 1
        for aid, agent in self._agents.items():
            if aid != msg.sender:
                agent.receive(msg)

    def run_debate(self, conjecture: str, domain: str,
                   proposer_id: str) -> dict:
        """
        Run a full debate round: one agent proposes, others vote.
        Returns verdict + updated belief state.
        """
        self._debate_rounds += 1

        # Proposer broadcasts conjecture
        msg = AgentMessage(
            sender=proposer_id, msg_type="conjecture",
            content=conjecture, domain=domain, confidence=0.7,
        )
        self.broadcast(msg)

        # Collect votes from agents' outboxes
        supporters = set()
        falsifiers = set()
        for aid, agent in self._agents.items():
            while agent._outbox:
                reply = agent._outbox.popleft()
                if reply.content == conjecture:
                    if reply.msg_type == "support":
                        supporters.add(aid)
                    elif reply.msg_type == "falsify":
                        falsifiers.add(aid)

        # The proposer supports their own conjecture
        supporters.add(proposer_id)
        total = len(self._agents)
        support_ratio = len(supporters) / total

        # Consensus check
        if support_ratio >= _CONSENSUS_THRESHOLD and len(supporters) >= _MIN_VOTES:
            verdict = "accepted"
            # Promote to blackboard
            key = f"{domain}:{conjecture[:40]}"
            if key not in self._blackboard:
                self._blackboard[key] = Belief(
                    content=conjecture, domain=domain,
                    confidence=support_ratio, source=proposer_id,
                    supporters=supporters, falsifiers=falsifiers,
                    status="accepted",
                )
            else:
                self._blackboard[key].supporters.update(supporters)
                self._blackboard[key].status = "accepted"
            # Propagate to all agents
            for agent in self._agents.values():
                agent.add_belief(conjecture, domain,
                                 confidence=support_ratio, source="consensus")
        elif len(falsifiers) > len(supporters):
            verdict = "rejected"
            key = f"{domain}:{conjecture[:40]}"
            if key in self._blackboard:
                self._blackboard[key].status = "rejected"
        else:
            verdict = "undecided"

        result = {
            "conjecture": conjecture,
            "domain": domain,
            "proposer": proposer_id,
            "supporters": list(supporters),
            "falsifiers": list(falsifiers),
            "support_ratio": round(support_ratio, 3),
            "verdict": verdict,
        }
        self._consensus_log.append(result)
        log.debug(f"Society debate '{conjecture[:30]}': {verdict} "
                  f"({len(supporters)}/{total} support)")
        return result

    def teaching_round(self):
        """
        High-accuracy agents teach lower-accuracy agents.
        Top performer shares 3 facts with lowest performer.
        """
        if len(self._agents) < 2:
            return
        agents_by_rate = sorted(
            self._agents.values(),
            key=lambda a: a._total_solved / max(a._total_tried, 1),
            reverse=True,
        )
        teacher = agents_by_rate[0]
        student = agents_by_rate[-1]
        messages = teacher.teach(student.agent_id)
        for msg in messages:
            student.receive(msg)
            self._message_log.append(msg)
            self._total_messages += 1
        if messages:
            log.debug(f"Society: {teacher.agent_id} taught {student.agent_id} "
                      f"{len(messages)} facts")

    def deliberation_cycle(self, engine_fn=None) -> dict:
        """
        Run one full deliberation cycle:
          1. Each agent proposes a conjecture (from their beliefs)
          2. Debate runs for each proposed conjecture
          3. Teaching round: best agent teaches weakest
          4. Record results
        """
        self._tick += 1
        debates_run = 0
        accepted = 0

        for aid, agent in self._agents.items():
            msg = agent.propose_conjecture()
            if msg:
                result = self.run_debate(
                    msg.content, msg.domain, aid)
                debates_run += 1
                if result["verdict"] == "accepted":
                    accepted += 1

        self.teaching_round()

        return {
            "tick": self._tick,
            "debates_run": debates_run,
            "newly_accepted": accepted,
            "total_blackboard": len(self._blackboard),
            "total_messages": self._total_messages,
        }

    def feed_to_concept_graph(self, cg) -> int:
        """
        Push all accepted society beliefs into the ConceptGraph.
        Returns number of concepts enriched.
        """
        count = 0
        for belief in self._blackboard.values():
            if belief.status == "accepted":
                try:
                    cg.ground_example(
                        concept_name=belief.domain + "_society",
                        text=belief.content[:100],
                        operation="society_consensus",
                        inputs=[belief.source],
                        result=belief.content[:40],
                        domain=belief.domain,
                        symbolic=belief.content,
                    )
                    count += 1
                except Exception:
                    pass
        return count

    # ── Collective intelligence metrics ────────────────────────────────────────

    def collective_beliefs(self) -> List[Belief]:
        """All accepted facts from the shared blackboard."""
        return [b for b in self._blackboard.values() if b.status == "accepted"]

    def knowledge_coverage(self) -> Dict[str, int]:
        """How many accepted facts exist per domain."""
        coverage: Dict[str, int] = defaultdict(int)
        for b in self.collective_beliefs():
            coverage[b.domain] += 1
        return dict(coverage)

    def specialization_drift(self) -> Dict[str, str]:
        """Show how each agent's actual best domain may differ from seed."""
        return {
            aid: f"{a.specialization} → {a.best_domain}"
            for aid, a in self._agents.items()
            if a.best_domain != a.specialization
        }

    def summary(self) -> dict:
        return {
            "n_agents": len(self._agents),
            "tick": self._tick,
            "total_messages": self._total_messages,
            "debate_rounds": self._debate_rounds,
            "blackboard_size": len(self._blackboard),
            "accepted_beliefs": len(self.collective_beliefs()),
            "knowledge_coverage": self.knowledge_coverage(),
            "specialization_drift": self.specialization_drift(),
            "agents": [a.to_dict() for a in self._agents.values()],
            "recent_debates": self._consensus_log[-5:],
            "recent_messages": [m.to_dict() for m in
                                 list(self._message_log)[-5:]],
        }
