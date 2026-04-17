"""
GoalPlanner — Hierarchical goal decomposition for SARE-HX.

Biological intelligence plans hierarchically:
  Goal: build a house
    → design house  → draft blueprints → choose materials
    → get materials → buy materials    → transport
    → construct     → lay foundation   → build walls → add roof

SARE previously only had flat goals (solve expression X).
This adds multi-level goal trees with dependency tracking.

GoalNode structure:
  - description: what to achieve
  - subgoals: list of GoalNode (children)
  - preconditions: what must be true before this goal
  - status: pending | in_progress | completed | failed
  - action: optional callable to execute this goal
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

log = logging.getLogger(__name__)


class GoalStatus(str, Enum):
    PENDING    = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED  = "completed"
    FAILED     = "failed"
    BLOCKED    = "blocked"


class GoalType(str, Enum):
    LEARN_CONCEPT    = "learn_concept"
    MASTER_DOMAIN    = "master_domain"
    SOLVE_EXPRESSION = "solve_expression"
    RUN_EXPERIMENT   = "run_experiment"
    ABSTRACT_RULE    = "abstract_rule"
    VERIFY_RULE      = "verify_rule"
    GENERALIZE       = "generalize"
    BUILD_PLAN       = "build_plan"


@dataclass
class GoalNode:
    """A single node in the hierarchical goal tree."""
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal_type: GoalType = GoalType.SOLVE_EXPRESSION
    description: str = ""
    domain: str = "general"
    priority: float = 0.5
    status: GoalStatus = GoalStatus.PENDING
    subgoals: List["GoalNode"] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)  # goal_ids that must complete first
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    @property
    def is_leaf(self) -> bool:
        return len(self.subgoals) == 0

    @property
    def is_complete(self) -> bool:
        return self.status == GoalStatus.COMPLETED

    @property
    def depth(self) -> int:
        if not self.subgoals:
            return 0
        return 1 + max(sg.depth for sg in self.subgoals)

    @property
    def completion_ratio(self) -> float:
        if not self.subgoals:
            return 1.0 if self.is_complete else 0.0
        done = sum(1 for sg in self.subgoals if sg.is_complete)
        return done / len(self.subgoals)

    def complete(self):
        self.status = GoalStatus.COMPLETED
        self.completed_at = time.time()

    def fail(self):
        self.status = GoalStatus.FAILED

    def to_dict(self, depth: int = 0) -> dict:
        return {
            "id": self.goal_id,
            "type": self.goal_type,
            "description": self.description,
            "domain": self.domain,
            "priority": self.priority,
            "status": self.status,
            "is_leaf": self.is_leaf,
            "depth": self.depth,
            "completion": round(self.completion_ratio, 2),
            "subgoals": [sg.to_dict(depth + 1) for sg in self.subgoals] if depth < 3 else [],
        }


# ── Built-in decomposition templates ──────────────────────────────────────────

def _decompose_learn_concept(concept: str, domain: str) -> List[GoalNode]:
    """Learning a concept requires: observe → abstract → verify."""
    observe = GoalNode(
        goal_type=GoalType.RUN_EXPERIMENT,
        description=f"Run {concept} experiments to collect observations",
        domain=domain, priority=0.9,
        metadata={"concept": concept, "n_experiments": 5},
    )
    abstract = GoalNode(
        goal_type=GoalType.ABSTRACT_RULE,
        description=f"Abstract symbolic rule from {concept} observations",
        domain=domain, priority=0.8,
        preconditions=[observe.goal_id],
        metadata={"concept": concept},
    )
    verify = GoalNode(
        goal_type=GoalType.VERIFY_RULE,
        description=f"Verify {concept} rule on held-out problems",
        domain=domain, priority=0.7,
        preconditions=[abstract.goal_id],
        metadata={"concept": concept},
    )
    generalize = GoalNode(
        goal_type=GoalType.GENERALIZE,
        description=f"Generalize {concept} rule to other domains",
        domain=domain, priority=0.6,
        preconditions=[verify.goal_id],
        metadata={"concept": concept},
    )
    return [observe, abstract, verify, generalize]


def _decompose_master_domain(domain: str) -> List[GoalNode]:
    """Mastering a domain requires learning its core concepts."""
    _DOMAIN_CONCEPTS = {
        "arithmetic": ["identity_addition", "identity_multiplication", "annihilation", "addition"],
        "logic":      ["negation", "double_negation", "conjunction"],
        "calculus":   ["differentiation", "integration"],
        "algebra":    ["addition", "multiplication", "distribution"],
    }
    concepts = _DOMAIN_CONCEPTS.get(domain, ["addition"])
    subgoals = []
    for concept in concepts:
        learn_node = GoalNode(
            goal_type=GoalType.LEARN_CONCEPT,
            description=f"Learn concept: {concept}",
            domain=domain, priority=0.8,
            metadata={"concept": concept},
        )
        learn_node.subgoals = _decompose_learn_concept(concept, domain)
        subgoals.append(learn_node)
    return subgoals


# ── GoalPlanner ────────────────────────────────────────────────────────────────

class GoalPlanner:
    """
    Hierarchical goal planner for SARE-HX.

    Converts high-level goals into executable subgoal trees,
    tracks progress, and selects the next actionable leaf goal.
    """

    def __init__(self):
        self._roots: Dict[str, GoalNode] = {}
        self._all_nodes: Dict[str, GoalNode] = {}
        self._completed: List[str] = []

    # ── Build plans ────────────────────────────────────────────────────────────

    def plan_learn_concept(self, concept: str, domain: str = "arithmetic") -> GoalNode:
        """Create a full hierarchical plan to learn a concept from scratch."""
        root = GoalNode(
            goal_type=GoalType.LEARN_CONCEPT,
            description=f"Learn concept '{concept}' in {domain}",
            domain=domain, priority=0.9,
            metadata={"concept": concept},
        )
        root.subgoals = _decompose_learn_concept(concept, domain)
        self._register(root)
        log.debug(f"GoalPlanner: plan for '{concept}' in {domain} — {len(root.subgoals)} subgoals")
        return root

    def plan_master_domain(self, domain: str) -> GoalNode:
        """Create a hierarchical plan to master all core concepts in a domain."""
        root = GoalNode(
            goal_type=GoalType.MASTER_DOMAIN,
            description=f"Master domain: {domain}",
            domain=domain, priority=0.95,
            metadata={"domain": domain},
        )
        root.subgoals = _decompose_master_domain(domain)
        self._register(root)
        log.debug(f"GoalPlanner: master plan for {domain} — {len(root.subgoals)} concept goals")
        return root

    def plan_solve(self, expression: str, domain: str = "arithmetic") -> GoalNode:
        """Create a plan for solving a specific expression."""
        root = GoalNode(
            goal_type=GoalType.SOLVE_EXPRESSION,
            description=f"Solve: {expression}",
            domain=domain, priority=0.7,
            metadata={"expression": expression},
        )
        # Subgoals: identify operators → choose transforms → apply → verify
        identify = GoalNode(
            goal_type=GoalType.ABSTRACT_RULE,
            description=f"Identify applicable rules for: {expression}",
            domain=domain, priority=0.8,
        )
        apply = GoalNode(
            goal_type=GoalType.SOLVE_EXPRESSION,
            description=f"Apply transforms to simplify: {expression}",
            domain=domain, priority=0.7,
            preconditions=[identify.goal_id],
        )
        verify = GoalNode(
            goal_type=GoalType.VERIFY_RULE,
            description=f"Verify simplified form is correct",
            domain=domain, priority=0.6,
            preconditions=[apply.goal_id],
        )
        root.subgoals = [identify, apply, verify]
        self._register(root)
        return root

    # ── Progress tracking ──────────────────────────────────────────────────────

    def next_actionable(self, domain: Optional[str] = None) -> Optional[GoalNode]:
        """
        Return the highest-priority leaf goal that is ready to execute
        (pending + all preconditions met).
        """
        candidates = []
        for root in self._roots.values():
            candidates.extend(self._collect_ready_leaves(root))
        if domain:
            candidates = [g for g in candidates if g.domain == domain] or candidates
        if not candidates:
            return None
        return max(candidates, key=lambda g: g.priority)

    def _collect_ready_leaves(self, node: GoalNode) -> List[GoalNode]:
        """Recursively collect all leaf nodes that are ready to execute."""
        if node.status != GoalStatus.PENDING:
            return []
        # Check preconditions
        for pre_id in node.preconditions:
            pre = self._all_nodes.get(pre_id)
            if pre and not pre.is_complete:
                return []  # blocked
        if node.is_leaf:
            return [node]
        ready = []
        for sg in node.subgoals:
            ready.extend(self._collect_ready_leaves(sg))
        return ready

    def mark_complete(self, goal_id: str):
        """Mark a goal as completed and propagate upward."""
        node = self._all_nodes.get(goal_id)
        if node:
            node.complete()
            self._completed.append(goal_id)
            # Check if parent is now complete
            self._propagate_completion(goal_id)

    def _propagate_completion(self, completed_id: str):
        """If all subgoals of a parent are done, complete the parent."""
        for node in self._all_nodes.values():
            if any(sg.goal_id == completed_id for sg in node.subgoals):
                if all(sg.is_complete for sg in node.subgoals):
                    node.complete()

    def _register(self, node: GoalNode):
        """Register a goal tree into the lookup dicts."""
        self._roots[node.goal_id] = node
        self._register_recursive(node)

    def _register_recursive(self, node: GoalNode):
        self._all_nodes[node.goal_id] = node
        for sg in node.subgoals:
            self._register_recursive(sg)

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_plan(self, goal_id: str) -> Optional[GoalNode]:
        return self._all_nodes.get(goal_id)

    def next_actionable_for_plan(self, plan_id: str) -> Optional[GoalNode]:
        """Return the next ready leaf within a specific plan (ignores other plans)."""
        root = self._roots.get(plan_id)
        if not root:
            return None
        candidates = self._collect_ready_leaves(root)
        return max(candidates, key=lambda g: g.priority) if candidates else None

    def active_plans(self) -> List[GoalNode]:
        return [r for r in self._roots.values()
                if r.status not in (GoalStatus.COMPLETED, GoalStatus.FAILED)]

    def summary(self) -> dict:
        active = self.active_plans()
        return {
            "total_plans": len(self._roots),
            "active_plans": len(active),
            "completed_plans": len(self._completed),
            "total_nodes": len(self._all_nodes),
            "next_actionable": (self.next_actionable().description
                                if self.next_actionable() else None),
            "plans": [p.to_dict() for p in active[:5]],
        }
