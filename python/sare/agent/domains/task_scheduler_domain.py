"""
TaskSchedulerDomain — generates planning/action knowledge problems as graphs.
Problems use the FillUnknown pattern: action -[achieves/causes]-> unknown(?)
FillUnknownTransform fills in the ? using the commonsense knowledge base.
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import Dict, List, Optional, Set, Tuple

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
# Helper: check reachability of all step nodes via precedes edges
# ---------------------------------------------------------------------------
def _all_steps_reachable(graph) -> bool:
    """
    Return True if all step-type nodes in the graph are reachable from at
    least one start step (a step with no incoming 'precedes' edge) via
    'precedes' edges.

    A graph with no step nodes is considered trivially reachable (True).
    A graph where a cycle causes every step to have an incoming edge is
    considered unreachable (False), matching the behaviour in plan_energy.
    """
    if graph is None:
        return False

    try:
        nodes = graph.nodes
        edges = graph.edges

        # Collect step node ids
        step_ids: Set[int] = set()
        for node in nodes:
            if getattr(node, "type", "") == "step":
                step_ids.add(node.id)

        if not step_ids:
            return True

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
            # Cycle detected or all steps have incoming — treat as unreachable
            return False

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

        return reachable >= step_ids

    except Exception as exc:
        log.error("_all_steps_reachable: %s", exc)
        # On error, be conservative and accept the graph to avoid silent data loss
        return True


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
    _PLANNING_FACTS: List[Tuple[str, str, str]] = [
        ("boil_water", "Causes", "hot_water"),
        ("exercise", "Causes", "fitness"),
        ("study", "Causes", "knowledge"),
        ("eat", "Causes", "energy"),
        ("sleep", "Causes", "rest"),
        ("practice", "Causes", "skill"),
        ("planning", "Causes", "organisation"),
        ("testing", "Causes", "quality"),
        ("debugging", "Causes", "fix"),
        ("compile", "Causes", "binary"),
        ("deploy", "Causes", "release"),
        ("merge", "Causes", "integration"),
        ("review", "Causes", "improvement"),
        ("backup", "Causes", "safety"),
        ("monitor", "Causes", "awareness"),
        ("train_model", "Causes", "prediction"),
        ("gather_data", "RequiredFor", "analysis"),
        ("write_tests", "RequiredFor", "deploy"),
        ("warm_up", "RequiredFor", "exercise"),
        ("make_list", "RequiredFor", "shopping"),
        ("shop", "RequiredFor", "cook"),
        ("cook", "Causes", "meal"),
        ("charge_battery", "RequiredFor", "use_laptop"),
        ("install_dependencies", "RequiredFor", "compile"),
        ("design", "RequiredFor", "implement"),
        ("implement", "RequiredFor", "test"),
        ("test", "RequiredFor", "release"),
        ("clean", "Causes", "tidiness"),
        ("save_work", "RequiredFor", "shutdown"),
        ("authenticate", "RequiredFor", "access"),
        ("seed", "RequiredFor", "grow_plant"),
        ("water_plant", "Causes", "growth"),
        ("read_docs", "RequiredFor", "configure"),
        ("configure", "RequiredFor", "deploy"),
    ]

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def _safe_add_node(self, graph, node_type: str, label: str, **attrs):
        if graph is None:
            return None
        node_id = attrs.pop("id", str(uuid.uuid4()))
        try:
            if hasattr(graph, "add_node"):
                return graph.add_node(node_id, node_type, label, **attrs)
        except TypeError:
            try:
                return graph.add_node(node_id=node_id, type=node_type, label=label, **attrs)
            except Exception:
                pass
        except Exception:
            pass
        try:
            if hasattr(graph, "create_node"):
                return graph.create_node(node_id, node_type, label, **attrs)
        except TypeError:
            try:
                return graph.create_node(node_id=node_id, type=node_type, label=label, **attrs)
            except Exception:
                pass
        except Exception:
            pass
        return None

    def _safe_add_edge(self, graph, source, target, relationship_type: str, **attrs):
        if graph is None:
            return None
        src = getattr(source, "id", source)
        tgt = getattr(target, "id", target)
        try:
            if hasattr(graph, "add_edge"):
                return graph.add_edge(src, tgt, relationship_type, **attrs)
        except TypeError:
            try:
                return graph.add_edge(source=src, target=tgt, relationship_type=relationship_type, **attrs)
            except Exception:
                pass
        except Exception:
            pass
        try:
            if hasattr(graph, "create_edge"):
                return graph.create_edge(src, tgt, relationship_type, **attrs)
        except TypeError:
            try:
                return graph.create_edge(source=src, target=tgt, relationship_type=relationship_type, **attrs)
            except Exception:
                pass
        except Exception:
            pass
        return None

    def _extract_node_label(self, node) -> str:
        for attr in ("label", "name", "value", "text"):
            value = getattr(node, attr, None)
            if isinstance(value, str) and value:
                return value
        return str(getattr(node, "id", ""))

    def _step_nodes(self, graph) -> List[object]:
        if graph is None:
            return []
        try:
            return [n for n in graph.nodes if getattr(n, "type", "") == "step"]
        except Exception:
            return []

    def _find_goal_nodes(self, graph) -> List[object]:
        if graph is None:
            return []
        try:
            return [n for n in graph.nodes if getattr(n, "type", "") == "goal"]
        except Exception:
            return []

    def _remove_random_precedes_edge(self, graph) -> bool:
        if graph is None:
            return False
        try:
            precedes_edges = [e for e in graph.edges if getattr(e, "relationship_type", "") == "precedes"]
            if not precedes_edges:
                return False
            edge = self._rng.choice(precedes_edges)
            if hasattr(graph, "remove_edge"):
                edge_id = getattr(edge, "id", None)
                if edge_id is not None:
                    graph.remove_edge(edge_id)
                    return True
            return False
        except Exception as exc:
            log.error("TaskSchedulerDomain: failed to remove edge: %s", exc)
            return False

    def generate_problem(self, difficulty: int = 3):
        if GeneratedProblem is None:
            return None
        graph = _new_graph()
        if graph is None:
            return None

        difficulty = max(2, int(difficulty))
        facts = list(self._PLANNING_FACTS)
        self._rng.shuffle(facts)
        chosen = facts[:difficulty]

        step_nodes = []
        for idx, (action, _relation, _outcome) in enumerate(chosen):
            step = self._safe_add_node(graph, "step", action, index=idx)
            step_nodes.append(step)

        for i in range(len(step_nodes) - 1):
            self._safe_add_edge(graph, step_nodes[i], step_nodes[i + 1], "precedes")

        goal_label = chosen[-1][2] if chosen else "goal"
        goal = self._safe_add_node(graph, "goal", goal_label)
        if step_nodes and goal is not None:
            self._safe_add_edge(graph, step_nodes[-1], goal, "achieves")

        self._remove_random_precedes_edge(graph)

        try:
            return GeneratedProblem(
                graph=graph,
                domain="task_scheduler",
                answer=None,
                metadata={"difficulty": difficulty},
            )
        except TypeError:
            try:
                return GeneratedProblem(graph, "task_scheduler", None, {"difficulty": difficulty})
            except Exception:
                return None
        except Exception:
            return None

    def generate_problems(self, n: int = 8, difficulty: int = 3):
        """Generate n problems. Alias used by multi_agent_learner."""
        problems = []
        for _ in range(n):
            p = self.generate_problem(difficulty=difficulty)
            if p is not None:
                problems.append(p)
        return problems

    def plan_energy(self, graph) -> float:
        """
        Score a candidate task plan. Lower is better.

        Existing structural behavior:
        - Penalize missing/degenerate plans.
        - Penalize lack of step connectivity / reachability.
        - Penalize missing goal achievement links.

        Approved extension:
        - Add semantic_penalty for commonsense temporal/causal ordering violations
          implied by _PLANNING_FACTS:
            * (A, "RequiredFor", B) means A should precede B if both appear as steps.
            * (A, "Causes", X) means if a goal node labeled X exists and A is a step,
              A should be able to reach that goal through the plan's ordering.
        """
        if graph is None:
            return 1e6

        try:
            nodes = list(getattr(graph, "nodes", []))
            edges = list(getattr(graph, "edges", []))
        except Exception:
            return 1e6

        energy = 0.0

        try:
            step_nodes = [n for n in nodes if getattr(n, "type", "") == "step"]
            goal_nodes = [n for n in nodes if getattr(n, "type", "") == "goal"]
        except Exception:
            return 1e6

        if not step_nodes:
            return 1e6

        step_ids: Set[object] = set()
        step_by_id: Dict[object, object] = {}
        step_label_to_ids: Dict[str, Set[object]] = {}
        goal_ids: Set[object] = set()
        goal_label_to_ids: Dict[str, Set[object]] = {}

        for n in step_nodes:
            nid = getattr(n, "id", None)
            if nid is None:
                continue
            step_ids.add(nid)
            step_by_id[nid] = n
            label = self._extract_node_label(n)
            step_label_to_ids.setdefault(label, set()).add(nid)

        for n in goal_nodes:
            nid = getattr(n, "id", None)
            if nid is None:
                continue
            goal_ids.add(nid)
            label = self._extract_node_label(n)
            goal_label_to_ids.setdefault(label, set()).add(nid)

        precedes_out: Dict[object, Set[object]] = {sid: set() for sid in step_ids}
        precedes_in: Dict[object, Set[object]] = {sid: set() for sid in step_ids}
        achieves_out: Dict[object, Set[object]] = {sid: set() for sid in step_ids}

        for e in edges:
            src = getattr(e, "source", None)
            tgt = getattr(e, "target", None)
            rel = getattr(e, "relationship_type", "")
            if rel == "precedes" and src in step_ids and tgt in step_ids:
                precedes_out[src].add(tgt)
                precedes_in[tgt].add(src)
            elif rel == "achieves" and src in step_ids and tgt in goal_ids:
                achieves_out[src].add(tgt)

        # Structural energy behavior
        num_precedes = sum(len(v) for v in precedes_out.values())
        if num_precedes == 0 and len(step_ids) > 1:
            energy += 50.0

        if not _all_steps_reachable(graph):
            energy += 100.0

        if goal_nodes:
            has_any_goal_link = any(achieves_out[sid] for sid in step_ids)
            if not has_any_goal_link:
                energy += 25.0

        starts = [sid for sid in step_ids if not precedes_in.get(sid)]
        if not starts and step_ids:
            energy += 100.0

        terminals = [sid for sid in step_ids if not precedes_out.get(sid)]
        if not terminals and step_ids:
            energy += 50.0

        # Semantic penalty extension
        semantic_penalty = 0.0

        # Precompute transitive reachability among steps
        reachable_steps: Dict[object, Set[object]] = {}
        for sid in step_ids:
            seen: Set[object] = set()
            stack = list(precedes_out.get(sid, set()))
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                for nxt in precedes_out.get(cur, set()):
                    if nxt not in seen:
                        stack.append(nxt)
            reachable_steps[sid] = seen

        # RequiredFor: A must precede B when both are present
        for lhs, relation, rhs in self._PLANNING_FACTS:
            if relation != "RequiredFor":
                continue
            lhs_ids = step_label_to_ids.get(lhs, set())
            rhs_ids = step_label_to_ids.get(rhs, set())
            if not lhs_ids or not rhs_ids:
                continue

            for a_id in lhs_ids:
                for b_id in rhs_ids:
                    if a_id == b_id:
                        continue
                    if b_id not in reachable_steps.get(a_id, set()):
                        semantic_penalty += 10.0

        # Causes: if goal X exists and step A causes X, A should lead to a terminal
        # step that achieves goal X, or directly achieve X itself.
        achievers_for_goal: Dict[object, Set[object]] = {}
        for sid, gids in achieves_out.items():
            for gid in gids:
                achievers_for_goal.setdefault(gid, set()).add(sid)

        for action, relation, outcome in self._PLANNING_FACTS:
            if relation != "Causes":
                continue
            action_ids = step_label_to_ids.get(action, set())
            matching_goal_ids = goal_label_to_ids.get(outcome, set())
            if not action_ids or not matching_goal_ids:
                continue

            for a_id in action_ids:
                satisfied = False
                direct_goals = achieves_out.get(a_id, set())
                if direct_goals & matching_goal_ids:
                    satisfied = True
                else:
                    for gid in matching_goal_ids:
                        for achiever_sid in achievers_for_goal.get(gid, set()):
                            if achiever_sid == a_id or achiever_sid in reachable_steps.get(a_id, set()):
                                satisfied = True
                                break
                        if satisfied:
                            break
                if not satisfied:
                    semantic_penalty += 5.0

        energy += semantic_penalty
        return float(energy)


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