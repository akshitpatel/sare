"""
AStarSearch — Best-first search with learned heuristic.

g(n) = total energy reduced so far (actual cost)
h(n) = heuristic_model.predict_graph(n) (predicted remaining reduction)
f(n) = g(n) + h(n) — priority in the open set heap

Same result interface as BeamSearch.search() for drop-in compatibility.
"""
from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass(order=False)
class AStarNode:
    f_score: float
    g_score: float        # energy reduced so far
    depth: int
    graph: Any
    energy: float         # current graph energy
    transforms_path: List[str] = field(default_factory=list)
    parent_hash: Optional[int] = None

    def __lt__(self, other):
        return self.f_score < other.f_score


def _graph_hash(graph) -> int:
    try:
        nodes = list(graph.nodes) if hasattr(graph, 'nodes') else []
        sig = tuple(sorted(
            (getattr(n, 'node_type', ''), str(getattr(n, 'value', '')))
            for n in nodes
        ))
        return hash(sig)
    except Exception:
        return id(graph)


class AStarSearch:
    """
    Best-first search using A* with a learned heuristic.
    Drop-in replacement for BeamSearch for use in ExperimentRunner.
    """

    def __init__(self, max_open_set: int = 512):
        self.max_open_set = max_open_set

    def search(self, graph, energy_evaluator, transforms: list,
               beam_width: int = 8,
               max_depth: int = 50,
               budget_seconds: float = 10.0,
               heuristic_fn: Optional[Callable] = None,
               attention_scorer=None,
               on_step: Optional[Callable] = None,
               **kwargs):
        """
        Run A* search.

        Returns dict compatible with BeamSearch.search() result:
          {result_graph, initial_energy, final_energy, delta, steps,
           transforms_used, success, elapsed_ms}
        """
        t0 = time.time()
        deadline = t0 + budget_seconds

        initial_energy = energy_evaluator.evaluate(graph) if hasattr(energy_evaluator, 'evaluate') else (
            energy_evaluator.compute(graph).total if hasattr(energy_evaluator, 'compute') else 0.0
        )
        start_node = AStarNode(
            f_score=0.0,
            g_score=0.0,
            depth=0,
            graph=graph,
            energy=initial_energy,
        )

        open_heap: List[AStarNode] = [start_node]
        closed_hashes = set()
        best_node = start_node
        steps = 0

        while open_heap and time.time() < deadline:
            node = heapq.heappop(open_heap)
            node_hash = _graph_hash(node.graph)

            if node_hash in closed_hashes:
                continue
            closed_hashes.add(node_hash)

            # Track best seen
            if node.g_score > best_node.g_score:
                best_node = node

            if node.depth >= max_depth:
                continue

            # Expand: apply all transforms
            for transform in transforms:
                if time.time() >= deadline:
                    break
                try:
                    # Support both match/apply and direct apply interfaces
                    if hasattr(transform, 'match') and hasattr(transform, 'apply'):
                        matches = transform.match(node.graph)
                        for ctx in (matches[:3] if matches else []):
                            try:
                                new_g, _ = transform.apply(node.graph, ctx)
                                rg_hash = _graph_hash(new_g)
                                if rg_hash in closed_hashes:
                                    continue
                                rg_energy = energy_evaluator.compute(new_g).total if hasattr(energy_evaluator, 'compute') else float(energy_evaluator.evaluate(new_g))
                                g_new = initial_energy - rg_energy
                                h_new = heuristic_fn(new_g) if heuristic_fn else 0.0
                                f_new = -(g_new + h_new)
                                child = AStarNode(
                                    f_score=f_new,
                                    g_score=g_new,
                                    depth=node.depth + 1,
                                    graph=new_g,
                                    energy=rg_energy,
                                    transforms_path=node.transforms_path + [type(transform).__name__],
                                )
                                heapq.heappush(open_heap, child)
                                steps += 1
                            except Exception:
                                pass
                    else:
                        result_graphs = transform.apply(node.graph)
                        if not result_graphs:
                            continue
                        for rg in (result_graphs if isinstance(result_graphs, list) else [result_graphs]):
                            rg_hash = _graph_hash(rg)
                            if rg_hash in closed_hashes:
                                continue
                            try:
                                rg_energy = energy_evaluator.compute(rg).total if hasattr(energy_evaluator, 'compute') else float(energy_evaluator.evaluate(rg))
                            except Exception:
                                continue
                            g_new = initial_energy - rg_energy
                            h_new = heuristic_fn(rg) if heuristic_fn else 0.0
                            f_new = -(g_new + h_new)
                            child = AStarNode(
                                f_score=f_new,
                                g_score=g_new,
                                depth=node.depth + 1,
                                graph=rg,
                                energy=rg_energy,
                                transforms_path=node.transforms_path + [type(transform).__name__],
                            )
                            heapq.heappush(open_heap, child)
                            steps += 1
                except Exception:
                    pass

            # Prune open set to prevent unbounded growth
            if len(open_heap) > self.max_open_set:
                open_heap = heapq.nsmallest(self.max_open_set // 2, open_heap)
                heapq.heapify(open_heap)

            if on_step:
                try:
                    on_step({"depth": node.depth, "energy": node.energy, "steps": steps})
                except Exception:
                    pass

        elapsed_ms = (time.time() - t0) * 1000
        delta = best_node.g_score
        success = delta > 0.05

        # Build a SearchResult-compatible dict
        return {
            "result_graph": best_node.graph,
            "initial_energy": initial_energy,
            "final_energy": best_node.energy,
            "delta": delta,
            "steps": steps,
            "transforms_used": best_node.transforms_path,
            "success": success,
            "elapsed_ms": elapsed_ms,
            "search_type": "astar",
        }
