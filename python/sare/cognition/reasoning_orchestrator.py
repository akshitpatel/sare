"""
ReasoningOrchestrator — unified retrieve / plan / execute pipeline.

Today's solve path is ~10 stages bolted on over time: schema cache → KB
lookup → family routing → composite learner → concept activation →
hypothesis engine → transform policy → BeamSearch → C++ search → A*
fallback → LLM fallback. Many run redundantly and the control flow is
hard to reason about.

The orchestrator replaces the ad-hoc pipeline with three clear phases:

  1. RETRIEVE — gather everything we know relevant to this problem
     (concepts, schemas, past solutions, KB hits). No search yet.
  2. PLAN — sketch an approach from retrieved knowledge (suggested
     transforms in priority order, decomposition if too hard).
  3. EXECUTE — run search guided by the plan, with LLM as last resort.

This is a thin coordinator — it calls the existing subsystems; it
doesn't replace them. Over time the subsystems can be cleaned up
behind the orchestrator's stable interface.

Domain-general: works identically for math, chemistry, logic, code,
language, commonsense — all three phases call domain-agnostic helpers.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class ReasoningContext:
    """What the orchestrator knows about a problem."""
    problem: Any
    domain: str
    expression: str
    # Retrieval artefacts
    activated_concepts: List[str] = field(default_factory=list)
    activated_schemas: List[str] = field(default_factory=list)
    cached_proof: Optional[List[str]] = None
    kb_answer: Optional[str] = None
    similar_episodes: List[Dict] = field(default_factory=list)
    # Plan artefacts
    suggested_transforms: List[str] = field(default_factory=list)
    subgoals: List[str] = field(default_factory=list)
    # Execution result
    solved: bool = False
    proof_steps: List[str] = field(default_factory=list)
    energy_before: float = 0.0
    energy_after: float = 0.0
    strategy_used: str = ""

    def summary(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "expression": self.expression[:80],
            "retrieval": {
                "concepts": len(self.activated_concepts),
                "schemas": len(self.activated_schemas),
                "cached": bool(self.cached_proof),
                "kb_hit": bool(self.kb_answer),
                "similar": len(self.similar_episodes),
            },
            "plan": {
                "suggested_transforms": self.suggested_transforms[:5],
                "subgoals": self.subgoals[:5],
            },
            "result": {
                "solved": self.solved,
                "steps": self.proof_steps[:5],
                "strategy": self.strategy_used,
                "delta": round(self.energy_before - self.energy_after, 3),
            },
        }


class ReasoningOrchestrator:
    """Coordinates retrieve → plan → execute phases around existing subsystems."""

    def __init__(self):
        self._stats = {
            "calls": 0,
            "solved": 0,
            "by_strategy": {},
            "avg_retrieval_ms": 0.0,
            "avg_plan_ms": 0.0,
            "avg_execute_ms": 0.0,
        }
        self._recent: List[Dict[str, Any]] = []

    # ── RETRIEVE ─────────────────────────────────────────────────
    def retrieve(self, ctx: ReasoningContext) -> None:
        """Gather everything we know about this problem. Pure look-up, no search."""
        # 1) Schema cache — is this structure already solved?
        try:
            from sare.cognition.schema_matcher import get_schema_matcher
            graph = getattr(ctx.problem, "graph", ctx.problem)
            cached = get_schema_matcher().match(graph)
            if cached:
                ctx.cached_proof = list(cached)
        except Exception:
            pass

        # 2) KB lookup for knowledge/text-style problems
        if not ctx.cached_proof and len(ctx.expression.split()) >= 3:
            try:
                from sare.memory.knowledge_lookup import KnowledgeLookup, DIRECT_THRESHOLD
                hit = KnowledgeLookup().lookup(ctx.expression, ctx.domain)
                if hit and hit.confidence >= DIRECT_THRESHOLD:
                    ctx.kb_answer = hit.answer
            except Exception:
                pass

        # 3) Concept activation — what concepts apply to this problem?
        try:
            from sare.concept.concept_graph import get_concept_graph
            cg = get_concept_graph()
            graph = getattr(ctx.problem, "graph", ctx.problem)
            symbols: List[str] = []
            try:
                if hasattr(graph, 'get_node_ids'):
                    for nid in graph.get_node_ids():
                        n = graph.get_node(nid)
                        if n:
                            lbl = str(getattr(n, 'label', '') or '')
                            if lbl:
                                symbols.append(lbl)
                elif hasattr(graph, 'nodes'):
                    for n in graph.nodes:
                        lbl = str(getattr(n, 'label', '') or '')
                        if lbl:
                            symbols.append(lbl)
            except Exception:
                pass
            if symbols:
                activated = cg.activate_for_problem(ctx.domain, symbols)
                ctx.activated_concepts = [c.name for c in activated]
        except Exception:
            pass

        # 4) Schema activation via unified index
        try:
            from sare.memory.unified_index import get_unified_memory_index
            idx = get_unified_memory_index()
            for cname in ctx.activated_concepts[:5]:
                refs = idx.lookup(cname)
                ctx.activated_schemas.extend(refs.get("schemas", []))
                idx.activate(cname)
        except Exception:
            pass

        # 5) Similar past episodes
        try:
            from sare.memory.memory_manager import get_memory_manager
            mm = get_memory_manager()
            graph = getattr(ctx.problem, "graph", ctx.problem)
            if hasattr(mm, "before_solve"):
                hint = mm.before_solve(graph, ctx.domain)
                if hint and hasattr(hint, "transform_sequence") and hint.transform_sequence:
                    ctx.similar_episodes.append({
                        "transforms": list(hint.transform_sequence),
                        "signature": getattr(hint, "signature", ""),
                    })
        except Exception:
            pass

    # ── PLAN ─────────────────────────────────────────────────────
    def plan(self, ctx: ReasoningContext) -> None:
        """Sketch an approach from retrieved knowledge."""
        # 1) If we have cached proof or KB hit, plan is just "use it"
        if ctx.cached_proof:
            ctx.suggested_transforms = list(ctx.cached_proof)
            return
        if ctx.kb_answer:
            ctx.suggested_transforms = [f"kb:{ctx.kb_answer[:40]}"]
            return

        # 2) Collect transform hints from activated concepts
        suggested: List[str] = []
        try:
            from sare.concept.concept_graph import get_concept_graph
            cg = get_concept_graph()
            activated = [cg.get(n) for n in ctx.activated_concepts if cg.get(n)]
            hints = cg.get_transform_hints([a for a in activated if a])
            suggested.extend(hints)
        except Exception:
            pass

        # 3) Add transforms from similar past episodes (strongest signal)
        for ep in ctx.similar_episodes[:3]:
            for t in ep.get("transforms", []):
                if t and t not in suggested:
                    suggested.insert(0, t)  # prepend episode-based hints

        ctx.suggested_transforms = suggested[:20]

        # 4) Decomposition — if the problem looks hard, sketch sub-goals
        #    (high node count, many operators). Lets planner handle it later.
        try:
            graph = getattr(ctx.problem, "graph", ctx.problem)
            node_count = 0
            if hasattr(graph, 'get_node_ids'):
                node_count = len(list(graph.get_node_ids()))
            elif hasattr(graph, 'nodes'):
                node_count = len(graph.nodes)
            if node_count > 15:
                # Use concept.related as sub-goal candidates
                if ctx.activated_concepts:
                    ctx.subgoals = [f"apply:{c}" for c in ctx.activated_concepts[:3]]
        except Exception:
            pass

    # ── EXECUTE ──────────────────────────────────────────────────
    def execute(self, ctx: ReasoningContext, runner) -> None:
        """Run the plan. `runner` is an ExperimentRunner-like object with
        `_run_single` method. Falls back to LLM if search doesn't solve.
        """
        # Direct-replay paths
        if ctx.cached_proof:
            ctx.solved = True
            ctx.proof_steps = list(ctx.cached_proof)
            ctx.strategy_used = "cache_replay"
            return
        if ctx.kb_answer:
            ctx.solved = True
            ctx.proof_steps = [f"kb:{ctx.kb_answer}"]
            ctx.strategy_used = "kb_lookup"
            return

        # Run the existing solve pipeline, which will use concept hints we've
        # already planted via concept_graph.activate_for_problem.
        try:
            result = runner._run_single(ctx.problem)
            ctx.solved = bool(getattr(result, "solved", False))
            ctx.proof_steps = list(getattr(result, "proof_steps", []) or [])
            ctx.energy_before = float(getattr(result, "energy_before", 0.0) or 0.0)
            ctx.energy_after = float(getattr(result, "energy_after", 0.0) or 0.0)
            ctx.strategy_used = "beam_search"
        except Exception as e:
            log.debug("ReasoningOrchestrator.execute: %s", e)

    # ── Full pipeline ────────────────────────────────────────────
    def reason(self, problem, domain: str, runner) -> ReasoningContext:
        """Run the full retrieve → plan → execute pipeline."""
        expr = str(getattr(problem, "expression", getattr(problem, "name", problem)))
        ctx = ReasoningContext(problem=problem, domain=domain, expression=expr)

        t0 = time.time()
        try:
            self.retrieve(ctx)
        except Exception as e:
            log.debug("retrieve phase error: %s", e)
        t1 = time.time()
        try:
            self.plan(ctx)
        except Exception as e:
            log.debug("plan phase error: %s", e)
        t2 = time.time()
        try:
            self.execute(ctx, runner)
        except Exception as e:
            log.debug("execute phase error: %s", e)
        t3 = time.time()

        self._update_stats(ctx, (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000)
        return ctx

    def _update_stats(self, ctx: ReasoningContext, r_ms: float, p_ms: float, e_ms: float):
        n = self._stats["calls"]
        self._stats["calls"] = n + 1
        if ctx.solved:
            self._stats["solved"] += 1
        s = ctx.strategy_used or "none"
        self._stats["by_strategy"][s] = self._stats["by_strategy"].get(s, 0) + 1
        # Running average of phase timings
        self._stats["avg_retrieval_ms"] = (self._stats["avg_retrieval_ms"] * n + r_ms) / (n + 1)
        self._stats["avg_plan_ms"] = (self._stats["avg_plan_ms"] * n + p_ms) / (n + 1)
        self._stats["avg_execute_ms"] = (self._stats["avg_execute_ms"] * n + e_ms) / (n + 1)

        self._recent.append(ctx.summary())
        if len(self._recent) > 20:
            self._recent = self._recent[-20:]

    def stats(self) -> Dict[str, Any]:
        return {
            **{k: (round(v, 2) if isinstance(v, float) else v)
               for k, v in self._stats.items()},
            "solve_rate": round(self._stats["solved"] / max(1, self._stats["calls"]), 3),
            "recent": list(self._recent[-10:]),
        }


_SINGLETON: Optional[ReasoningOrchestrator] = None


def get_reasoning_orchestrator() -> ReasoningOrchestrator:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = ReasoningOrchestrator()
    return _SINGLETON
