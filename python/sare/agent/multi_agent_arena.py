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
import math
import random
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

        # Track lightweight performance stats to inform agent selection
        self._agent_stats: Dict[str, Dict[str, float]] = {
            a.agent_id: {
                "wins": 0.0,
                "races": 0.0,
                "success_rate": 0.0,
                "mean_delta": 0.0,
                "mean_steps": 0.0,
                "mean_elapsed_ms": 0.0,
            }
            for a in self._agents
        }

        # Exploration for selection (epsilon-greedy / soft)
        self._selection_epsilon: float = 0.12

    # ── Parallel solve ─────────────────────────────────────────────────────────

    def _select_agents(self, max_workers: int, expression: str) -> List[AgentConfig]:
        """
        Select a subset of agents for this race.
        Closes the feedback loop by biasing toward historically high-performing agents,
        while keeping diversity via epsilon-greedy randomness.
        """
        max_workers = max(1, min(max_workers, len(self._agents)))
        if max_workers == len(self._agents):
            return list(self._agents)

        # With small probability, explore by selecting random subset
        if random.random() < self._selection_epsilon:
            agents_to_run = random.sample(self._agents, k=max_workers)
            return agents_to_run

        # Otherwise, exploit: rank agents by a composite score
        # Use log transforms for stability; ensure non-NaN.
        scored: List[Tuple[float, AgentConfig]] = []
        for a in self._agents:
            st = self._agent_stats.get(a.agent_id, {})
            races = float(st.get("races", 0.0))
            wins = float(st.get("wins", 0.0))
            success_rate = float(st.get("success_rate", 0.0))
            mean_delta = float(st.get("mean_delta", 0.0))
            mean_steps = float(st.get("mean_steps", 0.0))
            mean_elapsed_ms = float(st.get("mean_elapsed_ms", 0.0))

            # Base win-rate and success as proxies for quality, delta for magnitude
            win_rate = (wins / races) if races > 0 else 0.0

            # Prefer higher mean_delta; penalize elapsed time (efficiency)
            # Normalize elapsed penalty gently to avoid over-penalizing.
            elapsed_sec = mean_elapsed_ms / 1000.0 if mean_elapsed_ms > 0 else 0.0
            elapsed_penalty = math.log1p(elapsed_sec)  # grows slowly

            # Encourage exploration early on: uncertainty term decreases with races
            uncertainty = 1.0 / math.sqrt(1.0 + races)

            # Composite score:
            # - delta magnitude is primary
            # - success_rate and win_rate secondaries
            # - efficiency penalty third
            score = (
                1.8 * mean_delta
                + 1.2 * success_rate
                + 1.0 * win_rate
                - 0.12 * elapsed_penalty
                + 0.35 * uncertainty
            )

            if not math.isfinite(score):
                score = -1e9

            # Tiny deterministic tie-breaker from expression hash to reduce thrashing
            score += (hash(expression + a.agent_id) % 997) * 1e-12

            scored.append((score, a))

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [a for _, a in scored[:max_workers]]

        # Diversity safeguard: ensure at least 1 agent with different strategy if possible
        chosen_strategies = {c.strategy for c in chosen}
        if len(chosen_strategies) == 1:
            # Try to swap in an agent with a different strategy among top-ranked
            preferred_strategy = chosen[0].strategy
            alt_candidates = [a for _, a in scored if a.strategy != preferred_strategy]
            if alt_candidates:
                # Swap worst chosen with best alt
                worst_idx = min(range(len(chosen)), key=lambda i: scored[:max_workers][i][0])  # approximate
                # Find alt that maximizes score
                alt_candidates_scored = sorted(
                    [(next(s for s, ag in scored if ag.agent_id == a2.agent_id), a2) for a2 in alt_candidates],
                    key=lambda x: x[0],
                    reverse=True,
                )
                chosen[worst_idx] = alt_candidates_scored[0][1]

        return chosen

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
        except Exception:
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
        Race all selected agents on the same expression in parallel.
        Returns (winner, all_results) where winner has the best delta.
        """
        agents_to_run = self._select_agents(max_workers=max_workers, expression=expression)
        all_results: List[AgentResult] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents_to_run)) as pool:
            futures = [pool.submit(self._run_agent, agent, expression, engine_fn) for agent in agents_to_run]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                all_results.append(res)

        if not all_results:
            # Fallback: should be rare; return a deterministic "failure" result.
            dummy = AgentResult(
                agent_id="none",
                expression=expression,
                result=expression,
                delta=0.0,
                steps=0,
                transforms_used=[],
                elapsed_ms=0.0,
                success=False,
            )
            return dummy, []

        # Vote on best solution: highest delta, break ties by success then lower elapsed
        def sort_key(ar: AgentResult) -> Tuple[float, int, float]:
            return (ar.delta, 1 if ar.success else 0, -ar.elapsed_ms)

        all_results.sort(key=sort_key, reverse=True)
        winner = all_results[0]

        # Update win counts and stats (closes loop: success metrics influence future selection)
        self._total_races += 1
        for r in all_results:
            st = self._agent_stats.setdefault(r.agent_id, {
                "wins": 0.0, "races": 0.0, "success_rate": 0.0,
                "mean_delta": 0.0, "mean_steps": 0.0, "mean_elapsed_ms": 0.0
            })
            st["races"] = float(st.get("races", 0.0)) + 1.0

            # Exponential moving averages for robustness
            alpha = 0.25
            prev_success = float(st.get("success_rate", 0.0))
            st["success_rate"] = (1.0 - alpha) * prev_success + alpha * (1.0 if r.success else 0.0)

            prev_delta = float(st.get("mean_delta", 0.0))
            st["mean_delta"] = (1.0 - alpha) * prev_delta + alpha * float(r.delta)

            prev_steps = float(st.get("mean_steps", 0.0))
            st["mean_steps"] = (1.0 - alpha) * prev_steps + alpha * float(r.steps)

            prev_elapsed = float(st.get("mean_elapsed_ms", 0.0))
            st["mean_elapsed_ms"] = (1.0 - alpha) * prev_elapsed + alpha * float(r.elapsed_ms)

        self._wins[winner.agent_id] = self._wins.get(winner.agent_id, 0) + 1
        stw = self._agent_stats.setdefault(winner.agent_id, {})
        stw["wins"] = float(stw.get("wins", 0.0)) + 1.0

        # Keep shared transforms learning lightweight: union top transforms from winners
        # (No interface changes; only internal improvement loop.)
        try:
            top = max(1, min(2, len(all_results)))
            for r in all_results[:top]:
                for t in r.transforms_used[:10]:
                    if t not in self._shared_transforms:
                        self._shared_transforms.append(t)
        except Exception:
            pass

        # Log solve history for potential future tooling/debugging
        try:
            self._solve_history.append({
                "expression": expression,
                "winner": winner.agent_id,
                "winner_delta": winner.delta,
                "winner_success": winner.success,
                "agents": [r.to_dict() for r in all_results],
                "ts": time.time(),
            })
            # prevent unbounded growth
            if len(self._solve_history) > 5000:
                self._solve_history = self._solve_history[-2500:]
        except Exception:
            pass

        return winner, all_results

    def summary(self) -> dict:
        """Return agent stats for UI display."""
        recent = self._solve_history[-5:] if self._solve_history else []
        return {
            "n_agents":        len(self._agents),
            "total_races":     self._total_races,
            "agent_stats":     self._agent_stats,
            "wins":            self._wins,
            "shared_transforms": len(self._shared_transforms),
            "recent_races":    recent,
        }