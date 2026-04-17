"""
TaskSchedulerDomain — generates planning/action knowledge problems as graphs.
Problems use the FillUnknown pattern: action -[achieves/causes]-> unknown(?)
FillUnknownTransform fills in the ? using the commonsense knowledge base.
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import Dict, List, Optional, Set

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine / Graph import — graceful fallback
# ---------------------------------------------------------------------------
try:
    from sare.engine import Graph
except ImportError:
    try:
        from sare.sare_bindings import Graph
    except ImportError:
        log.warning("TaskSchedulerDomain: no Graph implementation found; using None placeholder.")
        Graph = None  # type: ignore

# ---------------------------------------------------------------------------
# GeneratedProblem import
# ---------------------------------------------------------------------------
try:
    from sare.curiosity.curriculum_generator import GeneratedProblem
except ImportError:
    log.warning("TaskSchedulerDomain: could not import GeneratedProblem.")
    GeneratedProblem = None  # type: ignore


# ---------------------------------------------------------------------------
# Helper: build a fresh Graph safely
# ---------------------------------------------------------------------------
def _new_graph() -> Optional[object]:
    if Graph is None:
        return None
    try:
        return Graph()
    except Exception as exc:
        log.error("TaskSchedulerDomain: failed to instantiate Graph: %s", exc)
        return None


# ---------------------------------------------------------------------------
# TaskSchedulerDomain
# ---------------------------------------------------------------------------

class TaskSchedulerDomain:
    """
    Generates partially-ordered planning problems as SARE graphs.

    Each plan is a sequence of step nodes connected by 'precedes' edges,
    with the final step connected to a goal node via an 'achieves' edge.
    One 'precedes' edge is randomly removed to create an incomplete plan —
    SARE's task is to restore the missing ordering constraint.
    """

    # Planning knowledge: action -> (relation, outcome)
    # These become FillUnknown problems: concept(action) -[relation]-> unknown(?)
    _PLANNING_FACTS: List[tuple] = [
        ("boil_water",     "Causes",    "hot_water"),
        ("exercise",       "Causes",    "fitness"),
        ("study",          "Causes",    "knowledge"),
        ("eat",            "Causes",    "energy"),
        ("sleep",          "Causes",    "rest"),
        ("practice",       "Causes",    "skill"),
        ("planning",       "Causes",    "organisation"),
        ("testing",        "Causes",    "quality"),
        ("debugging",      "Causes",    "fix"),
        ("compile",        "Causes",    "binary"),
        ("deploy",         "Causes",    "release"),
        ("merge",          "Causes",    "integration"),
        ("review",         "Causes",    "improvement"),
        ("backup",         "Causes",    "safety"),
        ("monitor",        "Causes",    "awareness"),
        ("train_model",    "Causes",    "prediction"),
        ("gather_data",    "RequiredFor", "analysis"),
        ("write_tests",    "RequiredFor", "deploy"),
        ("warm_up",        "RequiredFor", "exercise"),
        ("make_list",      "RequiredFor", "shopping"),
        ("make_coffee",    "UsedFor",   "caffeine"),
        ("cook_dinner",    "UsedFor",   "nourishment"),
        ("fix_bug",        "UsedFor",   "reliability"),
        ("learn_skill",    "UsedFor",   "career"),
        ("go_shopping",    "UsedFor",   "supplies"),
    ]

    def __init__(self):
        self._facts = list(self._PLANNING_FACTS)
        # Also seed these facts into CommonSenseBase for FillUnknownTransform
        self._seed_commonsense()

    def _seed_commonsense(self):
        """Add planning facts to CommonsenseBase so FillUnknownTransform can resolve them."""
        try:
            from sare.knowledge.commonsense import CommonSenseBase
            kb = CommonSenseBase()
            kb.load()
            added = 0
            for action, relation, outcome in self._facts:
                existing = kb._forward.get(action, [])
                if (relation, outcome) not in existing:
                    kb._add(action, relation, outcome)
                    added += 1
            if added:
                kb.save()
                log.info("TaskSchedulerDomain: seeded %d planning facts into CommonsenseBase", added)
        except Exception as exc:
            log.debug("TaskSchedulerDomain._seed_commonsense: %s", exc)

    def build_qa_graph(self, action: str, relation: str) -> Optional[object]:
        """
        Build a FillUnknown-style graph: concept(action) -[has_relation]-> relation -[object]-> unknown(?)
        The FillUnknownTransform fills in the ? using CommonsenseBase.
        """
        g = _new_graph()
        if g is None:
            return None
        try:
            concept_id = g.add_node("concept", action)
            rel_id = g.add_node("relation", relation)
            unknown_id = g.add_node("unknown", "?")
            g.add_edge(concept_id, rel_id, "has_relation")
            g.add_edge(rel_id, unknown_id, "object")
        except Exception as exc:
            log.error("TaskSchedulerDomain.build_qa_graph failed: %s", exc)
        return g

    # ------------------------------------------------------------------
    # generate_problems
    # ------------------------------------------------------------------
    def generate_problems(self, n: int = 5) -> List[object]:
        """
        Generate up to `n` planning FillUnknown problems with domain="planning".
        Each problem: action -[relation]-> ? (answered by FillUnknownTransform).
        """
        if GeneratedProblem is None:
            log.error("TaskSchedulerDomain.generate_problems: GeneratedProblem not available.")
            return []

        problems = []
        facts = self._facts
        if not facts:
            return []

        indices = list(range(len(facts)))
        random.shuffle(indices)

        for i in range(n):
            action, relation, _ = facts[indices[i % len(indices)]]
            try:
                graph = self.build_qa_graph(action, relation)
                if graph is None:
                    continue
                prob = GeneratedProblem(
                    id=f"plan_{uuid.uuid4().hex[:8]}",
                    graph=graph,
                    origin=f"planning:{action}_{relation}",
                    status="pending",
                    domain="planning",
                )
                problems.append(prob)
            except Exception as exc:
                log.warning("TaskSchedulerDomain: failed to build problem for '%s': %s", action, exc)

        log.info("TaskSchedulerDomain: generated %d planning problems", len(problems))
        return problems

    # ------------------------------------------------------------------
    # plan_energy
    # ------------------------------------------------------------------
    def plan_energy(self, graph) -> float:
        """
        Compute planning energy for a graph.

        Energy = (number of disconnected steps) * 3.0

        A step is 'disconnected' if it has no incoming 'precedes' edge AND
        it is not the first step (i.e. not reachable via any path from the
        initial step).  We use a simple reachability check from any step
        that has no incoming precedes edge (candidate start steps).

        Lower energy = better-connected (closer to a complete plan).
        """
        if graph is None:
            return 0.0

        try:
            nodes = graph.nodes
            edges = graph.edges

            # Collect step node ids
            step_ids: Set[int] = set()
            for node in nodes:
                if getattr(node, "type", "") == "step":
                    step_ids.add(node.id)

            if not step_ids:
                return 0.0

            # Build adjacency for 'precedes' edges among steps
            precedes_out: Dict[int, Set[int]] = {sid: set() for sid in step_ids}
            precedes_in: Dict[int, Set[int]] = {sid: set() for sid in step_ids}

            for edge in edges:
                src = getattr(edge, "source", None)
                tgt = getattr(edge, "target", None)
                rel = getattr(edge, "relationship_type", "")
                if rel == "precedes" and src in step_ids and tgt in step_ids:
                    precedes_out[src].add(tgt)
                    precedes_in[tgt].add(src)

            # Find candidate start steps (no incoming precedes edges)
            starts: Set[int] = {sid for sid in step_ids if not precedes_in[sid]}

            if not starts:
                # Cycle detected or all steps have incoming — treat as full disconnect
                return float(len(step_ids)) * 3.0

            # BFS/DFS reachability from all starts
            reachable: Set[int] = set()
            frontier = list(starts)
            while frontier:
                current = frontier.pop()
                if current in reachable:
                    continue
                reachable.add(current)
                for nxt in precedes_out.get(current, set()):
                    if nxt not in reachable:
                        frontier.append(nxt)

            disconnected = step_ids - reachable
            return float(len(disconnected)) * 3.0

        except Exception as exc:
            log.error("TaskSchedulerDomain.plan_energy: %s", exc)
            return 0.0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_SINGLETON: Optional[TaskSchedulerDomain] = None


def get_task_domain() -> TaskSchedulerDomain:
    """Return the module-level singleton TaskSchedulerDomain."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = TaskSchedulerDomain()
    return _SINGLETON
