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
        # Original causal and ordering relations
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
        
        # Temporal constraints
        ("boil_water",     "before",    "make_tea"),
        ("warm_up",        "before",    "exercise"),
        ("gather_data",    "before",    "analyze_data"),
        ("compile",        "before",    "test_binary"),
        ("write_documentation", "before", "release"),
        ("plan_sprint",    "before",    "execute_sprint"),
        ("design_system",  "before",    "implement_system"),
        ("backup_database", "before",   "migrate_database"),
        ("exercise",       "after",     "cool_down"),
        ("eat",            "after",     "digest"),
        ("deploy",         "after",     "monitor_performance"),
        ("merge_code",     "after",     "run_tests"),
        ("study",          "during",    "take_notes"),
        ("commute",        "during",    "listen_podcast"),
        ("cook",           "during",    "prepare_ingredients"),
        ("present",        "during",    "answer_questions"),
        ("exercise",       "during",    "stay_hydrated"),
        
        # Resource dependencies
        ("compile",        "requires",  "source_code"),
        ("test",           "requires",  "test_cases"),
        ("deploy",         "requires",  "server_capacity"),
        ("train_model",    "requires",  "training_data"),
        ("bake_cake",      "requires",  "oven"),
        ("video_call",     "requires",  "internet_connection"),
        ("drive_car",      "requires",  "fuel"),
        ("print_document", "requires",  "printer_ink"),
        ("run_simulation", "requires",  "computational_power"),
        ("host_event",     "requires",  "venue"),
        ("compile",        "consumes",  "cpu_cycles"),
        ("drive_car",      "consumes",  "fuel"),
        ("bake_cake",      "consumes",  "electricity"),
        ("print_document", "consumes",  "paper"),
        ("video_call",     "consumes",  "bandwidth"),
        ("train_model",    "consumes",  "gpu_memory"),
        ("run_simulation", "consumes",  "time"),
        ("host_event",     "consumes",  "budget"),
        ("clean_house",    "consumes",  "cleaning_supplies"),
        ("cook_meal",      "consumes",  "ingredients"),
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
        graph = _new_graph()
        if graph is None:
            return None

        try:
            # Create nodes
            action_node = graph.add_node(type="concept", name=action)
            relation_node = graph.add_node(type="relation", name=relation)
            unknown_node = graph.add_node(type="unknown", name="?")

            # Create edges
            graph.add_edge(action_node.id, relation_node.id, "has_relation")
            graph.add_edge(relation_node.id, unknown_node.id, "object")

            return graph
        except Exception as exc:
            log.error("TaskSchedulerDomain.build_qa_graph: %s", exc)
            return None

    def build_plan_graph(self, actions: List[str], goal: str, missing_edge_idx: Optional[int] = None) -> Optional[object]:
        """
        Build a planning graph with steps, temporal constraints, and resource dependencies.
        
        Args:
            actions: List of action names in the plan
            goal: Goal outcome name
            missing_edge_idx: If provided, remove the edge at this index to create the problem
        
        Returns:
            Graph with steps, constraints, and a goal node
        """
        graph = _new_graph()
        if graph is None:
            return None

        try:
            # Create step nodes
            step_nodes = []
            for action in actions:
                node = graph.add_node(type="step", name=action)
                step_nodes.append(node.id)

            # Create goal node
            goal_node = graph.add_node(type="goal", name=goal)

            # Collect all edges to potentially remove one
            edges_to_add = []
            
            # 1. Add "precedes" edges between consecutive steps (original behavior)
            for i in range(len(step_nodes) - 1):
                edges_to_add.append((step_nodes[i], step_nodes[i+1], "precedes"))
            
            # 2. Add "achieves" edge from last step to goal (original behavior)
            edges_to_add.append((step_nodes[-1], goal_node.id, "achieves"))
            
            # 3. Add temporal and resource constraint edges based on facts
            for idx, action in enumerate(actions):
                # Find all facts for this action
                for fact_action, relation, outcome in self._facts:
                    if fact_action == action:
                        # Only add constraints that make sense in this plan context
                        if relation in ("before", "after", "during"):
                            # Temporal constraints: connect to other steps if outcome matches
                            for other_idx, other_action in enumerate(actions):
                                if other_action == outcome and other_idx != idx:
                                    # Add temporal edge between steps
                                    edges_to_add.append((step_nodes[idx], step_nodes[other_idx], relation))
                                    break
                        elif relation in ("requires", "consumes"):
                            # Resource dependencies: create resource nodes
                            resource_node = graph.add_node(type="resource", name=outcome)
                            edges_to_add.append((step_nodes[idx], resource_node.id, relation))
                        # Note: "Causes", "RequiredFor", "UsedFor" edges are omitted from plan graphs
                        # as they're handled by separate QA graphs
            
            # Remove one edge if requested (to create the problem)
            if missing_edge_idx is not None and 0 <= missing_edge_idx < len(edges_to_add):
                edges_to_add.pop(missing_edge_idx)
            
            # Add all remaining edges to the graph
            for src, tgt, rel in edges_to_add:
                graph.add_edge(src, tgt, rel)

            return graph
        except Exception as exc:
            log.error("TaskSchedulerDomain.build_plan_graph: %s", exc)
            return None

    def generate_plan_problems(self, num_problems: int = 5) -> List[GeneratedProblem]:
        """Generate simple plan problems with one missing ordering constraint."""
        if GeneratedProblem is None:
            log.warning("TaskSchedulerDomain: GeneratedProblem not available, returning empty list.")
            return []

        problems = []
        actions_pool = [
            "boil_water", "make_tea", "drink_tea",
            "warm_up", "exercise", "cool_down",
            "gather_data", "analyze_data", "present_results",
            "compile", "test", "deploy",
            "plan_sprint", "execute_sprint", "review_sprint",
        ]

        for _ in range(num_problems):
            try:
                # Select 3-5 random actions
                num_actions = random.randint(3, 5)
                selected_actions = random.sample(actions_pool, min(num_actions, len(actions_pool)))
                
                # Goal is outcome of last action
                goal = f"completed_{selected_actions[-1]}"
                
                # Randomly select which "precedes" edge to remove
                missing_edge_idx = random.randint(0, len(selected_actions) - 2)
                
                graph = self.build_plan_graph(selected_actions, goal, missing_edge_idx)
                if graph is None:
                    continue

                prob = GeneratedProblem(
                    id=f"plan_{uuid.uuid4().hex[:8]}",
                    graph=graph,
                    origin=f"simple_planning:{'_'.join(selected_actions)}",
                    status="pending",
                    domain="planning",
                )
                problems.append(prob)
            except Exception as exc:
                log.warning("TaskSchedulerDomain.generate_plan_problems: failed to build problem: %s", exc)

        log.info("TaskSchedulerDomain: generated %d simple plan problems", len(problems))
        return problems

    def generate_multi_step_problems(self, num_problems: int = 3) -> List[GeneratedProblem]:
        """Generate multi-step planning problems with temporal and resource constraints."""
        if GeneratedProblem is None:
            log.warning("TaskSchedulerDomain: GeneratedProblem not available, returning empty list.")
            return []

        problems = []
        
        # Define more complex action sequences with natural constraints
        complex_plans = [
            (["gather_data", "analyze_data", "write_report", "present_results"], "project_completed"),
            (["plan_sprint", "execute_sprint", "review_sprint", "retrospective"], "sprint_completed"),
            (["design_system", "implement_system", "test_system", "deploy_system"], "system_released"),
            (["backup_database", "migrate_database", "verify_migration", "restore_backup"], "migration_completed"),
            (["write_tests", "run_tests", "fix_failures", "regression_test"], "quality_assured"),
        ]

        for _ in range(num_problems):
            try:
                # Select a random complex plan
                actions, base_goal = random.choice(complex_plans)
                
                # Randomly decide which type of constraint to remove
                constraint_type = random.choice(["precedes", "temporal", "resource"])
                
                if constraint_type == "precedes":
                    # Remove a precedes edge (original behavior)
                    missing_edge_idx = random.randint(0, len(actions) - 2)
                elif constraint_type == "temporal":
                    # Remove a temporal constraint edge (will be added during graph building)
                    # We'll mark this by using a special index beyond the precedes edges
                    missing_edge_idx = len(actions) - 1 + random.randint(0, len(actions) - 1)
                else:  # resource
                    # Remove a resource dependency edge
                    missing_edge_idx = len(actions) - 1 + len(actions) + random.randint(0, len(actions) - 1)
                
                graph = self.build_plan_graph(actions, base_goal, missing_edge_idx)
                if graph is None:
                    continue

                prob = GeneratedProblem(
                    id=f"multi_plan_{uuid.uuid4().hex[:8]}",
                    graph=graph,
                    origin=f"multi_step_planning:{'_'.join(actions)}",
                    status="pending",
                    domain="planning",
                )
                problems.append(prob)
            except Exception as exc:
                log.warning("TaskSchedulerDomain.generate_multi_step_problems: failed to build problem: %s", exc)

        log.info("TaskSchedulerDomain: generated %d multi-step planning problems", len(problems))
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