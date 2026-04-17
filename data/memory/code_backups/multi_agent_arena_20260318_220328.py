"""
MultiAgentArena — Upgrade 3: Multi-Agent Learning

Runs N independent solver agents on the same problem, then:
  1. Votes on the best solution (highest energy reduction)
  2. Shares discovered transforms across agents
  3. Debates conjectures: agents propose + falsify each other's rules
  4. Logs discoveries to shared memory

This accelerates learning enormously because:
  - Diversity: different beam widths / strategies explore different paths
  - Competition: agents race to find the best simplification
  - Collaboration: best transforms propagate to all agents
  - Debate: conjectures survive only if they cannot be falsified

Inspired by the "Society of Mind" (Minsky) and ensemble learning.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for one solver agent."""
    agent_id: str
    beam_width: int = 8
    budget_seconds: float = 2.0
    strategy: str = "beam_search"   # or "mcts"
    max_depth: int = 12

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "beam_width": self.beam_width,
            "budget_seconds": self.budget_seconds,
            "strategy": self.strategy,
        }


@dataclass
class AgentResult:
    """Result from one agent solving a problem."""
    agent_id: str
    expression: str
    result: str
    delta: float
    steps: int
    transforms_used: List[str]
    elapsed_ms: float
    success: bool

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "result": self.result,
            "delta": round(self.delta, 4),
            "steps": self.steps,
            "transforms_used": self.transforms_used,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "success": self.success,
        }


@dataclass
class DebateResult:
    """Result of agents debating a conjecture."""
    conjecture: str
    proposer: str
    supporters: List[str] = field(default_factory=list)
    falsifiers: List[str] = field(default_factory=list)
    verdict: str = "undecided"   # 'accepted', 'falsified', 'undecided'
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "conjecture": self.conjecture,
            "proposer": self.proposer,
            "supporters": self.supporters,
            "falsifiers": self.falsifiers,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
        }


class MultiAgentArena:
    """
    Runs multiple solver agents in parallel on problems.

    Architecture:
      ┌─────────────────────────────────────────────┐
      │  Problem → [Agent1 Agent2 Agent3 ... AgentN] │
      │              ↓      ↓      ↓         ↓       │
      │           Vote on best solution              │
      │           Share discovered transforms        │
      │           Debate conjectures                 │
      │           → Best result + shared knowledge   │
      └─────────────────────────────────────────────┘
    """

    # Default diverse agent fleet
    _DEFAULT_AGENTS = [
        AgentConfig("explorer",   beam_width=4,  budget_seconds=1.0, strategy="beam_search"),
        AgentConfig("thorough",   beam_width=16, budget_seconds=3.0, strategy="beam_search"),
        AgentConfig("standard",   beam_width=8,  budget_seconds=2.0, strategy="beam_search"),
        AgentConfig("deep",       beam_width=8,  budget_seconds=4.0, strategy="beam_search", max_depth=20),
        AgentConfig("mcts_light", beam_width=4,  budget_seconds=2.0, strategy="mcts"),
    ]

    def __init__(self, n_agents: int = 3):
        self._agents = self._DEFAULT_AGENTS[:max(2, min(n_agents, len(self._DEFAULT_AGENTS)))]
        self._shared_transforms: List[str] = []
        self._debate_log: List[DebateResult] = []
        self._solve_history: List[dict] = []
        self._wins: Dict[str, int] = {a.agent_id: 0 for a in self._agents}
        self._total_races: int = 0

    # ── Parallel solve ─────────────────────────────────────────────────────────

    def _run_agent(self, agent: AgentConfig, expression: str,
                   engine_fn) -> AgentResult:
        """Run one agent on an expression using the provided engine function."""
        t0 = time.time()
        try:
            result = engine_fn(
                expression,
                beam_width=agent.beam_width,
                budget_seconds=agent.budget_seconds,
                strategy=agent.strategy,
                max_depth=agent.max_depth,
            )
            elapsed = (time.time() - t0) * 1000
            return AgentResult(
                agent_id=agent.agent_id,
                expression=expression,
                result=result.get("result", expression),
                delta=result.get("delta", 0.0),
                steps=result.get("steps", 0),
                transforms_used=result.get("transforms_used", []),
                elapsed_ms=elapsed,
                success=result.get("success", False),
            )
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            return AgentResult(
                agent_id=agent.agent_id,
                expression=expression,
                result=expression,
                delta=0.0, steps=0,
                transforms_used=[],
                elapsed_ms=elapsed,
                success=False,
            )

    def race(self, expression: str, engine_fn,
             max_workers: int = 3) -> Tuple[AgentResult, List[AgentResult]]:
        """
        Race all agents on the same expression in parallel.
        Returns (winner, all_results) where winner has the best delta.
        """
        agents_to_run = self._agents[:max_workers]
        all_results: List[AgentResult] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._run_agent, agent, expression, engine_fn): agent
                for agent in agents_to_run
            }
            for fut in concurrent.futures.as_completed(futures, timeout=10.0):
                try:
                    r = fut.result()
                    all_results.append(r)
                except Exception:
                    pass

        if not all_results:
            return AgentResult("fallback", expression, expression,
                               0.0, 0, [], 0.0, False), []

        # Vote: best delta wins
        winner = max(all_results, key=lambda r: r.delta)
        self._wins[winner.agent_id] = self._wins.get(winner.agent_id, 0) + 1
        self._total_races += 1

        # Share transforms from winner to pool
        for t in winner.transforms_used:
            if t not in self._shared_transforms:
                self._shared_transforms.append(t)

        self._solve_history.append({
            "expression": expression,
            "winner": winner.agent_id,
            "winner_delta": winner.delta,
            "n_agents": len(all_results),
        })

        return winner, all_results

    # ── Conjecture debate ──────────────────────────────────────────────────────

    def debate_conjecture(self, conjecture: str, test_cases: List[dict],
                          engine_fn) -> DebateResult:
        """
        Agents debate whether a conjecture is valid.

        Each agent tests the conjecture on all test cases:
          - If transform improves energy → agent supports the conjecture
          - If transform makes things worse → agent falsifies it
        """
        proposer = self._agents[0].agent_id
        debate = DebateResult(conjecture=conjecture, proposer=proposer)

        for agent in self._agents:
            supported = 0
            falsified = 0
            for tc in test_cases[:5]:  # cap at 5 test cases for speed
                expr = tc.get("expression", "x + 0")
                try:
                    result = engine_fn(
                        expr,
                        beam_width=agent.beam_width,
                        budget_seconds=0.5,
                        strategy=agent.strategy,
                    )
                    if result.get("delta", 0) > 0.01:
                        supported += 1
                    elif result.get("delta", 0) < -0.01:
                        falsified += 1
                except Exception:
                    pass

            if supported > falsified:
                debate.supporters.append(agent.agent_id)
            elif falsified > supported:
                debate.falsifiers.append(agent.agent_id)

        # Verdict
        if len(debate.supporters) > len(debate.falsifiers):
            debate.verdict = "accepted"
            debate.confidence = len(debate.supporters) / max(len(self._agents), 1)
        elif len(debate.falsifiers) > len(debate.supporters):
            debate.verdict = "falsified"
            debate.confidence = len(debate.falsifiers) / max(len(self._agents), 1)
        else:
            debate.verdict = "undecided"
            debate.confidence = 0.5

        self._debate_log.append(debate)
        log.info(f"MultiAgent debate '{conjecture[:40]}': {debate.verdict} "
                 f"({len(debate.supporters)} support, {len(debate.falsifiers)} falsify)")
        return debate

    # ── Shared knowledge ───────────────────────────────────────────────────────

    def top_agents(self, n: int = 3) -> List[dict]:
        """Return the N most successful agents by win count."""
        ranked = sorted(self._wins.items(), key=lambda x: x[1], reverse=True)
        return [{"agent_id": aid, "wins": w,
                 "win_rate": round(w / max(self._total_races, 1), 3)}
                for aid, w in ranked[:n]]

    def accepted_conjectures(self) -> List[str]:
        """Return conjectures accepted by agent debate."""
        return [d.conjecture for d in self._debate_log if d.verdict == "accepted"]

    def summary(self) -> dict:
        return {
            "n_agents": len(self._agents),
            "total_races": self._total_races,
            "shared_transforms": len(self._shared_transforms),
            "debates_held": len(self._debate_log),
            "accepted_conjectures": len(self.accepted_conjectures()),
            "top_agents": self.top_agents(3),
            "agents": [a.to_dict() for a in self._agents],
            "recent_races": self._solve_history[-5:],
            "recent_debates": [d.to_dict() for d in self._debate_log[-3:]],
        }
