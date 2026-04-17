"""
HierarchicalSolver — when a problem is too hard for direct solve,
decompose it into sub-problems, solve each, then compose.

Domain-general decomposition strategies:
  * Graph-based: split a complex expression at its root operator.
    (x+y)*(a+b) → solve (x+y) and (a+b) separately.
  * Text-based: split a compound question at "and", "also", "but".
  * Fallback: if the problem simplifies partially, continue recursively.

Triggered when:
  * Initial graph has > HARD_NODE_THRESHOLD nodes.
  * Direct solve failed AND problem is structurally decomposable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_HARD_NODE_THRESHOLD = 15
_MAX_DEPTH = 3


@dataclass
class DecomposedSolve:
    """Result of a hierarchical solve attempt."""
    solved: bool = False
    subgoal_count: int = 0
    subgoals_solved: int = 0
    proof_steps: List[str] = field(default_factory=list)
    depth: int = 0
    strategy: str = "direct"
    note: str = ""


class HierarchicalSolver:
    """Lightweight hierarchical wrapper around an ExperimentRunner."""

    def __init__(self):
        self._stats = {
            "attempts": 0,
            "solved": 0,
            "decomposed": 0,
            "decomposed_solved": 0,
            "avg_subgoals": 0.0,
        }

    # ── Decomposition helpers ────────────────────────────────

    @staticmethod
    def _count_nodes(graph) -> int:
        try:
            if hasattr(graph, "get_node_ids"):
                return len(list(graph.get_node_ids()))
            if hasattr(graph, "nodes"):
                return len(graph.nodes)
        except Exception:
            pass
        return 0

    @staticmethod
    def _split_graph_at_root(graph) -> List[Any]:
        """Return the operand subgraphs of the graph's root operator.
        Empty list if the graph isn't splittable."""
        try:
            # Find root: node with no incoming edges
            if not hasattr(graph, "get_node_ids"):
                return []
            roots = []
            for nid in graph.get_node_ids():
                incoming = 0
                try:
                    incoming = len(list(graph.incoming(nid))) if hasattr(graph, "incoming") else 0
                except Exception:
                    pass
                if incoming == 0:
                    roots.append(nid)
            if not roots:
                return []
            root_id = roots[0]
            root = graph.get_node(root_id)
            if not root or root.type != "operator":
                return []
            # Operands of root
            subs = []
            try:
                for e in graph.outgoing(root_id):
                    subs.append(e.target)
            except Exception:
                pass
            if len(subs) < 2:
                return []
            # Build each operand as a standalone subgraph via clone + prune
            subgraphs = []
            for sub_id in subs:
                try:
                    g = graph.clone()
                    # Remove the root (and its other edges) to isolate the subtree
                    to_remove = [root_id]
                    # Collect sibling subtrees to remove
                    for other in subs:
                        if other != sub_id:
                            to_remove.append(other)
                    for n in to_remove:
                        if hasattr(g, "remove_node"):
                            try:
                                g.remove_node(n)
                            except Exception:
                                pass
                    subgraphs.append(g)
                except Exception:
                    pass
            return subgraphs
        except Exception:
            return []

    @staticmethod
    def _split_text(expression: str) -> List[str]:
        """Split a text question into sub-questions on common connectives."""
        import re as _re
        if not expression or len(expression) < 20:
            return []
        parts = _re.split(r"\s+(?:and|also|but|moreover|additionally)\s+", expression)
        parts = [p.strip() for p in parts if p.strip()]
        return parts if len(parts) >= 2 else []

    # ── Public solve ─────────────────────────────────────────

    def solve(self, runner, problem, depth: int = 0) -> DecomposedSolve:
        """Attempt hierarchical solve. `runner` must have `_run_single(problem)`."""
        self._stats["attempts"] += 1
        dec = DecomposedSolve(depth=depth)

        graph = getattr(problem, "graph", problem)
        expr = str(getattr(problem, "expression", getattr(problem, "name", "")))
        node_count = self._count_nodes(graph)

        # Phase 1: Try direct solve first (cheapest path)
        direct_failed = False
        if node_count <= _HARD_NODE_THRESHOLD or depth >= _MAX_DEPTH:
            try:
                res = runner._run_single(problem)
                if getattr(res, "solved", False):
                    dec.solved = True
                    dec.proof_steps = list(getattr(res, "proof_steps", []) or [])
                    dec.strategy = "direct"
                    self._stats["solved"] += 1
                    return dec
                direct_failed = True
            except Exception as e:
                log.debug("HierarchicalSolver.direct failed: %s", e)
                direct_failed = True

        # Phase 2: Decompose if hard or direct failed
        if depth >= _MAX_DEPTH:
            dec.note = f"max depth ({_MAX_DEPTH}) reached"
            return dec

        subgraphs = self._split_graph_at_root(graph)
        text_parts = self._split_text(expr) if not subgraphs else []

        if not subgraphs and not text_parts:
            dec.note = "not decomposable; direct solve " + ("failed" if direct_failed else "not attempted")
            return dec

        self._stats["decomposed"] += 1
        subgoal_results: List[DecomposedSolve] = []

        # Decompose by graph
        for sg in subgraphs:
            try:
                sub_problem = type(problem)(
                    id=f"{getattr(problem, 'id', 'sub')}_sg{len(subgoal_results)}",
                    graph=sg,
                    origin="decomposed",
                ) if hasattr(problem, "__class__") and type(problem).__name__ == "GeneratedProblem" else problem
                # GeneratedProblem's __post_init__ sets domain/expression from graph
                sub_res = self.solve(runner, sub_problem, depth=depth + 1)
                subgoal_results.append(sub_res)
            except Exception as e:
                log.debug("HierarchicalSolver.subgraph error: %s", e)

        # Or decompose by text (for QA / commonsense)
        if not subgraphs:
            for part in text_parts[:3]:
                try:
                    from sare.engine import load_problem as _lp
                    loaded = _lp(part)
                    sub_g = loaded[1] if isinstance(loaded, tuple) else loaded
                    if sub_g is None:
                        continue
                    try:
                        from sare.curiosity.curriculum_generator import GeneratedProblem
                        sub_problem = GeneratedProblem(id=f"text_sg{len(subgoal_results)}",
                                                        graph=sub_g, origin="decomposed_text",
                                                        domain=getattr(problem, "domain", ""),
                                                        expression=part)
                    except Exception:
                        sub_problem = sub_g
                    sub_res = self.solve(runner, sub_problem, depth=depth + 1)
                    subgoal_results.append(sub_res)
                except Exception as e:
                    log.debug("HierarchicalSolver.text error: %s", e)

        dec.subgoal_count = len(subgoal_results)
        dec.subgoals_solved = sum(1 for r in subgoal_results if r.solved)

        # Compose: treat as solved only if all subgoals solved
        if subgoal_results and dec.subgoals_solved == dec.subgoal_count:
            dec.solved = True
            for r in subgoal_results:
                dec.proof_steps.extend(r.proof_steps)
            dec.strategy = "decomposed"
            self._stats["solved"] += 1
            self._stats["decomposed_solved"] += 1
        else:
            dec.note = f"{dec.subgoals_solved}/{dec.subgoal_count} subgoals solved"

        # Update running avg
        n = self._stats["decomposed"]
        if n > 0:
            self._stats["avg_subgoals"] = (
                (self._stats["avg_subgoals"] * (n - 1) + dec.subgoal_count) / n
            )
        return dec

    def stats(self) -> Dict[str, Any]:
        s = dict(self._stats)
        s["solve_rate"] = round(self._stats["solved"] / max(1, self._stats["attempts"]), 3)
        s["decomposition_rate"] = round(
            self._stats["decomposed"] / max(1, self._stats["attempts"]), 3
        )
        s["decomposed_solve_rate"] = round(
            self._stats["decomposed_solved"] / max(1, self._stats["decomposed"]), 3
        )
        return s


_SINGLETON: Optional[HierarchicalSolver] = None


def get_hierarchical_solver() -> HierarchicalSolver:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = HierarchicalSolver()
    return _SINGLETON
