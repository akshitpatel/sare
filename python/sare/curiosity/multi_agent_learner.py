"""
multi_agent_learner.py — Parallel specialist agents that learn simultaneously.

Each agent specialises in one domain, runs its own solve→reflect→promote loop,
and broadcasts discoveries to a shared event feed visible in the dashboard.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ── Agent specialisation catalogue ───────────────────────────────────────────
AGENT_SPECS = [
    {"id": "α-Arith",  "domain": "arithmetic", "color": "#00e5ff", "icon": "∑",
     "seeds": ["x + 0", "x * 1", "x * 0", "2 + 3", "4 * 5", "x + x"]},
    {"id": "β-Logic",  "domain": "logic",       "color": "#ff2a6d", "icon": "⊢",
     "seeds": ["not not x", "neg neg x", "x and True", "x or False"]},
    {"id": "γ-Algebra","domain": "algebra",     "color": "#05d59e", "icon": "χ",
     "seeds": ["x * (y + 0)", "a * (b + 0)", "2 * (x + 0)", "x * (y + 1 - 1)"]},
    {"id": "δ-Calc",   "domain": "calculus",    "color": "#ffb800", "icon": "∂",
     "seeds": ["x^1", "x^0", "x^2 * x^3", "sin^2(x) + cos^2(x)"]},
    {"id": "ε-Code",   "domain": "code",        "color": "#a78bfa", "icon": "⟨/⟩",
     "seeds": [], "use_code_builder": True},
    {"id": "ζ-QA",     "domain": "qa",          "color": "#f97316", "icon": "?",
     "seeds": [], "use_qa_builder": True},
    {"id": "η-Plan",   "domain": "planning",    "color": "#4ade80", "icon": "▶",
     "seeds": [], "use_plan_builder": True},
]


# ── Event / Status data classes ───────────────────────────────────────────────
@dataclass
class AgentEvent:
    id: int
    agent_id: str
    domain: str
    color: str
    event_type: str   # start | solve | fail | promote | reflect | discover | error
    message: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "domain": self.domain,
            "color": self.color,
            "event_type": self.event_type,
            "message": self.message,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentStatus:
    agent_id: str
    domain: str
    color: str
    icon: str
    status: str = "idle"
    current_problem: str = ""
    solved: int = 0
    attempted: int = 0
    promoted: int = 0
    last_rule: str = ""
    solve_rate: float = 0.0
    started_at: float = field(default_factory=time.time)
    domain_problems_generated: int = 0

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "domain": self.domain,
            "color": self.color,
            "icon": self.icon,
            "status": self.status,
            "current_problem": self.current_problem,
            "solved": self.solved,
            "attempted": self.attempted,
            "promoted": self.promoted,
            "last_rule": self.last_rule,
            "solve_rate": round(self.solve_rate, 3),
            "uptime": round(time.time() - self.started_at),
            "domain_refills": self.domain_problems_generated,
        }


# ── Multi-Agent Learner ───────────────────────────────────────────────────────
class MultiAgentLearner:
    """Manages N parallel learning agents, each in its own daemon thread."""

    def __init__(self):
        self._agents: Dict[str, AgentStatus] = {}
        self._threads: List[threading.Thread] = []
        self._stop_events: Dict[str, threading.Event] = {}
        self._event_feed: deque = deque(maxlen=500)
        self._event_counter = 0
        self._feed_lock = threading.Lock()
        self._running = False
        # shared promotion registry (thread-safe writes via file lock in SeededConceptRegistry)
        self._registry = None

    # ── Event posting ─────────────────────────────────────────────────────────
    def _post(self, agent_id: str, domain: str, color: str,
              event_type: str, message: str) -> None:
        with self._feed_lock:
            self._event_counter += 1
            ev = AgentEvent(
                id=self._event_counter,
                agent_id=agent_id,
                domain=domain,
                color=color,
                event_type=event_type,
                message=message,
            )
            self._event_feed.append(ev)
        # mirror to main web log for grep-ability
        log.info("[Agent %s] %s: %s", agent_id, event_type, message)

    # ── Build a runner for one agent ─────────────────────────────────────────
    def _build_runner(self, spec: dict):
        from sare.engine import BeamSearch, EnergyEvaluator, get_transforms, load_problem
        from sare.curiosity.curriculum_generator import CurriculumGenerator
        from sare.curiosity.experiment_runner import ExperimentRunner

        curriculum = CurriculumGenerator()
        try:
            curriculum.load()
        except Exception:
            pass

        # Seed this agent's domain
        for expr in spec.get("seeds", []):
            try:
                g = load_problem(expr)
                curriculum.add_problem(g, domain=spec["domain"])
            except Exception:
                pass

        # Seed code domain via CodeGraphBuilder
        if spec.get("use_code_builder"):
            try:
                from sare.perception.graph_builders import CodeGraphBuilder
                cb = CodeGraphBuilder()
                code_seeds = [
                    "x if True else y", "a if False else b",
                    "not not x", "True and x", "x or False",
                    "x and x", "x or x", "not True", "not False",
                ]
                for expr in code_seeds:
                    try:
                        g = cb.build(expr)
                        curriculum.add_problem(g, domain="code")
                    except Exception:
                        pass
            except Exception as e:
                log.warning("Code builder seed failed: %s", e)

        # Seed QA domain via QAPipeline
        if spec.get("use_qa_builder"):
            try:
                from sare.knowledge.commonsense import CommonSenseBase
                from sare.agent.qa_pipeline import get_qa_pipeline
                kb = CommonSenseBase()
                kb.load()
                if kb.total_facts() == 0:
                    kb.seed()
                qa = get_qa_pipeline()
                problems = qa.generate_qa_problems(kb, n=15)
                for p in problems:
                    curriculum.add_problem(p)
            except Exception as e:
                log.warning("QA builder seed failed: %s", e)

        # Seed planning domain via TaskSchedulerDomain
        if spec.get("use_plan_builder"):
            try:
                from sare.agent.domains.task_scheduler_domain import get_task_domain
                td = get_task_domain()
                problems = td.generate_problems(n=8)
                for p in problems:
                    curriculum.add_problem(p)
            except Exception as e:
                log.warning("Plan builder seed failed: %s", e)

        # Build domain-specific transform set
        base_transforms = get_transforms(include_macros=True)
        domain = spec.get("domain", "general")

        if domain == "code":
            try:
                from sare.transforms.code_transforms import (
                    IfTrueElimTransform, IfFalseElimTransform,
                    NotTrueTransform, NotFalseTransform,
                    AndSelfTransform, OrSelfTransform, SelfAssignElimTransform,
                )
                base_transforms = [
                    IfTrueElimTransform(), IfFalseElimTransform(),
                    NotTrueTransform(), NotFalseTransform(),
                    AndSelfTransform(), OrSelfTransform(), SelfAssignElimTransform(),
                ] + base_transforms
            except Exception as e:
                log.warning("Code transforms unavailable: %s", e)

        elif domain == "qa":
            try:
                from sare.knowledge.commonsense import CommonSenseBase
                from sare.transforms.logic_transforms import (
                    FillUnknownTransform, ChainInferenceTransform,
                    DoubleNegRemoveTransform,
                )
                kb = CommonSenseBase()
                kb.load()
                if kb.total_facts() == 0:
                    kb.seed()
                base_transforms = [
                    FillUnknownTransform(kb),
                    ChainInferenceTransform(kb),
                    DoubleNegRemoveTransform(),
                ] + base_transforms
            except Exception as e:
                log.warning("Logic transforms unavailable: %s", e)

        elif domain == "planning":
            try:
                from sare.knowledge.commonsense import CommonSenseBase as _PlanCSB
                from sare.transforms.logic_transforms import (
                    FillUnknownTransform, ChainInferenceTransform,
                    ModusPonensTransform, ImpliesElimTransform,
                )
                _plan_kb = _PlanCSB(); _plan_kb.load()
                if _plan_kb.total_facts() == 0: _plan_kb.seed()
                base_transforms = [
                    FillUnknownTransform(_plan_kb),
                    ChainInferenceTransform(_plan_kb),
                    ModusPonensTransform(), ImpliesElimTransform(),
                ] + base_transforms
            except Exception as e:
                log.warning("Planning transforms unavailable: %s", e)

        runner = ExperimentRunner(
            curriculum_gen=curriculum,
            searcher=BeamSearch(),
            energy=EnergyEvaluator(),
            transforms=base_transforms,
            beam_width=6,
            budget_seconds=4.0,
        )

        # Wire optional modules
        try:
            from sare.reflection.py_reflection import get_reflection_engine
            runner.reflection_engine = get_reflection_engine()
        except Exception:
            pass

        try:
            from sare.causal.induction import CausalInduction
            runner.causal_induction = CausalInduction()
        except Exception:
            pass

        try:
            from sare.memory.concept_seed_loader import SeededConceptRegistry
            runner.concept_registry = SeededConceptRegistry()
        except Exception:
            pass

        return runner

    # ── Per-agent worker loop ─────────────────────────────────────────────────
    def _agent_loop(self, spec: dict, stop_event: threading.Event) -> None:
        agent_id = spec["id"]
        domain   = spec["domain"]
        color    = spec["color"]
        status   = self._agents[agent_id]

        # Build runner
        try:
            runner = self._build_runner(spec)
            self._post(agent_id, domain, color, "start",
                       f"Initialised — specialisation: {domain}")
            status.status = "ready"
            # Homeostasis: agent starting on a domain counts as exploration
            try:
                from sare.meta.homeostasis import get_homeostatic_system
                get_homeostatic_system().on_exploration()
            except Exception:
                pass
        except Exception as exc:
            self._post(agent_id, domain, color, "error", f"Init failed: {exc}")
            status.status = "error"
            return

        curriculum = runner.curriculum_gen
        _domain_refill_counter = 0
        _analogy_check_counter = 0

        # Self-model for domain competence tracking (shared across all results)
        _self_model = None
        try:
            from sare.meta.self_model import SelfModel
            _self_model = SelfModel()
            _self_model.load()
        except Exception:
            _self_model = None

        while not stop_event.is_set():
            try:
                # ── Domain-specific problem refill ────────────────────────────
                # Count how many pending problems are actually in this domain
                pending = [p for p in getattr(curriculum, 'generated_problems', [])
                           if p.status == 'pending' and p.domain == domain]
                if len(pending) == 0:
                    _domain_refill_counter += 1
                    added = self._refill_domain_problems(spec, curriculum, domain, color, agent_id)
                    if added == 0:
                        # No domain problems available — let curriculum generate math as fallback
                        # but flag it in the status
                        status.status = "seeking"
                    else:
                        status.domain_problems_generated = getattr(status, 'domain_problems_generated', 0) + added

                status.status = "solving"
                results = runner.run_batch(n=1)

                for r in results:
                    status.attempted += 1
                    status.current_problem = r.problem_id
                    # Track whether this was a real domain problem
                    is_domain_problem = (
                        domain in ("arithmetic", "algebra", "logic", "calculus") or
                        r.problem_id.startswith("qa_") or
                        r.problem_id.startswith("plan_") or
                        r.problem_id.startswith("code_") or
                        r.problem_id.startswith("ext_")
                    )

                    if r.solved:
                        status.solved += 1
                        proof_str = " → ".join(r.proof_steps[:3]) if r.proof_steps else (r.rule_name or "—")
                        tag = f"[{domain}]" if is_domain_problem else "[math↩]"
                        self._post(agent_id, domain, color, "solve",
                                   f"✓ {r.problem_id}  {tag}  [{proof_str[:50]}]  Δ={r.energy_before - r.energy_after:.1f}")
                    else:
                        self._post(agent_id, domain, color, "fail",
                                   f"✗ {r.problem_id}  no transform found")

                    # Update self-model with this result
                    if _self_model is not None:
                        try:
                            _self_model.observe(
                                domain=domain,
                                success=r.solved,
                                delta=r.energy_before - r.energy_after,
                                steps=len(r.proof_steps) if r.proof_steps else 0,
                                transforms_used=r.proof_steps or [],
                            )
                            # Save every 20 observations
                            if status.attempted % 20 == 0:
                                _self_model.save()
                        except Exception:
                            pass

                    if r.rule_promoted:
                        status.promoted += 1
                        status.last_rule = r.rule_name
                        self._post(agent_id, domain, color, "promote",
                                   f"★ Rule promoted: '{r.rule_name}'  ({r.reasoning[:40] if r.reasoning else '?'})")

                        try:
                            from sare.memory.world_model import get_world_model
                            get_world_model().log_activity(
                                "promote", domain,
                                f"Agent {agent_id} promoted: {r.rule_name}")
                        except Exception:
                            pass

                        # Homeostasis: on_analogy_found when non-math domain finds a rule
                        # (a rule from qa/code/planning domain is cross-domain by nature)
                        if domain not in ("arithmetic", "algebra", "logic", "calculus"):
                            try:
                                from sare.meta.homeostasis import get_homeostatic_system
                                get_homeostatic_system().on_analogy_found()
                            except Exception:
                                pass

                # Every 50 iterations: check for analogy transfer and fire homeostasis
                _analogy_check_counter += 1
                if _analogy_check_counter % 50 == 0:
                    try:
                        from sare.causal.analogy_transfer import AnalogyTransfer
                        from sare.meta.homeostasis import get_homeostatic_system
                        at = AnalogyTransfer()
                        transfers = at.transfer_from_domain(domain)
                        if transfers:
                            get_homeostatic_system().on_analogy_found()
                            log.info("[Agent %s] Cross-domain analogy found: %d transfers", agent_id, len(transfers))
                    except Exception:
                        pass

                status.solve_rate = status.solved / max(status.attempted, 1)
                status.status = "waiting"

            except Exception as exc:
                self._post(agent_id, domain, color, "error", f"Batch error: {exc}")
                log.debug("Agent %s batch error: %s", agent_id, exc)

            stop_event.wait(1.5)

    def _refill_domain_problems(self, spec: dict, curriculum, domain: str,
                                color: str, agent_id: str) -> int:
        """Re-inject domain-specific problems when the queue runs dry."""
        added = 0
        try:
            if domain == "qa":
                from sare.knowledge.commonsense import CommonSenseBase
                from sare.agent.qa_pipeline import get_qa_pipeline
                kb = CommonSenseBase()
                kb.load()
                if kb.total_facts() == 0:
                    kb.seed()
                qa = get_qa_pipeline()
                # Generate fresh questions (shuffle ensures variety)
                problems = qa.generate_qa_problems(kb, n=10)
                for p in problems:
                    curriculum.add_problem(p)
                    added += 1
                if added:
                    self._post(agent_id, domain, color, "discover",
                               f"💡 Refilled {added} QA problems from KB ({kb.total_facts()} facts)")

            elif domain == "code":
                from sare.perception.graph_builders import CodeGraphBuilder
                import random
                builder = CodeGraphBuilder()
                code_templates = [
                    "x if True else y", "a if False else b",
                    "not not x", "True and x", "x or False",
                    "x and x", "x or x", "not True", "not False",
                    "y if True else z", "b if False else c",
                    "not not not not x", "True and True and x",
                ]
                for expr in random.sample(code_templates, min(8, len(code_templates))):
                    try:
                        g = builder.build(expr)
                        curriculum.add_problem(g, domain="code")
                        added += 1
                    except Exception:
                        pass
                if added:
                    self._post(agent_id, domain, color, "discover",
                               f"💡 Refilled {added} code problems")

            elif domain == "planning":
                from sare.agent.domains.task_scheduler_domain import get_task_domain
                td = get_task_domain()
                problems = td.generate_problems(n=5)
                for p in problems:
                    curriculum.add_problem(p)
                    added += 1
                if added:
                    self._post(agent_id, domain, color, "discover",
                               f"💡 Refilled {added} planning problems")

        except Exception as e:
            log.warning("Domain refill failed for %s: %s", domain, e)
        return added

    # ── Public API ────────────────────────────────────────────────────────────
    def start(self, n_agents: int = 3) -> dict:
        if self._running:
            return {"status": "already_running", "agents": len(self._agents)}

        self._running = True
        specs = AGENT_SPECS[:max(1, min(n_agents, len(AGENT_SPECS)))]

        for spec in specs:
            aid = spec["id"]
            stop_ev = threading.Event()
            self._stop_events[aid] = stop_ev
            self._agents[aid] = AgentStatus(
                agent_id=aid,
                domain=spec["domain"],
                color=spec["color"],
                icon=spec["icon"],
                status="starting",
            )
            t = threading.Thread(
                target=self._agent_loop,
                args=(spec, stop_ev),
                name=f"sare-agent-{aid}",
                daemon=True,
            )
            self._threads.append(t)
            t.start()

        return {"status": "started", "agents": len(specs),
                "ids": [s["id"] for s in specs]}

    def stop(self) -> dict:
        for ev in self._stop_events.values():
            ev.set()
        self._running = False
        self._threads.clear()
        self._stop_events.clear()
        self._agents.clear()
        return {"status": "stopped"}

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "total_solved": sum(a.solved for a in self._agents.values()),
            "total_promoted": sum(a.promoted for a in self._agents.values()),
            "agents": [a.to_dict() for a in self._agents.values()],
        }

    def get_feed(self, since_id: int = 0, limit: int = 80) -> List[dict]:
        with self._feed_lock:
            snapshot = list(self._event_feed)
        filtered = [e for e in snapshot if e.id > since_id][-limit:]
        return [e.to_dict() for e in filtered]

    def get_feed_since(self, since_id: int) -> List[AgentEvent]:
        with self._feed_lock:
            snapshot = list(self._event_feed)
        return [e for e in snapshot if e.id > since_id]


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: Optional[MultiAgentLearner] = None
_instance_lock = threading.Lock()


def get_multi_agent_learner() -> MultiAgentLearner:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MultiAgentLearner()
    return _instance
