"""
Attention Mechanism (Pillar 2 — Human Brain Architecture)

Prevents combinatorial explosion on large graphs by maintaining a "Working Memory
Workspace" — a 3-to-5 node sub-graph that the solver focuses on at any given time.

Architecture:
    1. `AttentionSelector` scores nodes by a combination of:
       - Structural complexity (high-degree nodes)
       - Uncertainty (from EnergyEvaluator)
       - Recency (nodes touched in the most recent solve steps)
    2. The top-K nodes + their immediate neighbors form the `WorkingMemoryWindow`.
    3. Transforms are applied ONLY within that window.
    4. The modified sub-graph is re-integrated into the parent graph.

This mirrors human "spotlight attention" — we don't perceive the whole visual
field at once; we focus on whichever part is most energetically uncertain.
"""

import logging
from typing import List, Set, Tuple

from sare.engine import Graph, Node

log = logging.getLogger(__name__)

_WINDOW_SIZE = 5  # Max nodes in the attention window at once


class AttentionSelector:
    """Scores every node in the graph and selects the top-K most attention-worthy."""

    @staticmethod
    def score(node: Node, graph: Graph) -> float:
        """
        Returns an attention score for a node.
        Higher = more worthy of focus.
        """
        score = 0.0

        # 1. Uncertainty is the primary driver
        score += float(getattr(node, "uncertainty", 0.0)) * 3.0

        # 2. Structural complexity: prefer operator nodes over leaf nodes
        node_type = str(getattr(node, "type", "") or "")
        if node_type == "operator":
            score += 1.5
        elif node_type == "variable":
            score += 1.0

        # 3. Reward nodes with more outgoing edges (higher arity = more complexity)
        #    Also keep this robust to different Graph implementations.
        try:
            out_deg = len(graph.outgoing(node.id))
        except Exception as e:
            log.debug("[AttentionSelector] outgoing() failed for node %s: %s", node.id, e)
            out_deg = 0
        score += float(out_deg) * 0.5

        # 4. Penalize leaf constants — they rarely need transformation
        if node_type == "constant":
            score -= 0.5

        # 5. Light recency modulation if available in node attributes.
        #    (Does not assume WorkingMemory/recency is always present.)
        try:
            rec = float(getattr(node, "recency", 0.0))
        except Exception as e:
            log.debug("[AttentionSelector] recency read failed for node %s: %s", node.id, e)
            rec = 0.0
        if rec:
            score += max(0.0, rec) * 0.25

        return score

    @staticmethod
    def select_window(graph: Graph, window_size: int = _WINDOW_SIZE) -> List[Node]:
        """
        Selects the top-K nodes by attention score, then expands the window
        to include immediate neighbors so the sub-graph is contiguous.
        """
        nodes = list(getattr(graph, "nodes", []) or [])
        if not nodes or window_size <= 0:
            return []

        # Score all nodes
        scored: List[Tuple[float, Node]] = [(AttentionSelector.score(n, graph), n) for n in nodes]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Seed set: anchors are the top nodes, not a hard-coded 3.
        # Ensure we can still reach window_size through neighbor expansion.
        seed_k = min(max(2, int(window_size // 2)), len(scored))
        anchors: List[Node] = [n for _, n in scored[:seed_k]]

        window_ids: Set[int] = {n.id for n in anchors}

        # Expand to immediate neighbors until we reach window_size.
        # Use a bounded BFS-like growth: alternate across anchors fairly by iterating
        # through anchors repeatedly as needed.
        anchors_queue: List[Node] = list(anchors)
        idx = 0
        # Safety cap on expansion attempts to prevent pathological graph edge iteration.
        max_rounds = max(1, window_size * 3)

        rounds = 0
        while len(window_ids) < min(window_size, len(nodes)) and rounds < max_rounds and anchors_queue:
            anchor = anchors_queue[idx % len(anchors_queue)]
            idx += 1
            rounds += 1

            try:
                outgoing_edges = graph.outgoing(anchor.id)
            except Exception:
                outgoing_edges = []

            for edge in outgoing_edges:
                try:
                    target_id = edge.target
                except Exception:
                    continue

                if target_id in window_ids:
                    continue

                neighbor = None
                try:
                    neighbor = graph.get_node(target_id)
                except Exception:
                    neighbor = None

                if neighbor is None:
                    continue

                window_ids.add(neighbor.id)
                if len(window_ids) >= min(window_size, len(nodes)):
                    break

            if not getattr(graph, "nodes", None):
                break

        # If the window is still undersized (e.g., missing edges), fill with next best nodes.
        target_n = min(window_size, len(nodes))
        if len(window_ids) < target_n:
            for _, n in scored:
                if n.id not in window_ids:
                    window_ids.add(n.id)
                    if len(window_ids) >= target_n:
                        break

        window_nodes = [n for n in nodes if n.id in window_ids]
        log.debug(f"AttentionWindow: {len(window_nodes)} nodes selected from {len(nodes)} total.")
        return window_nodes


class WorkingMemoryWorkspace:
    """
    Extracts a sub-graph, applies transforms within it,
    and grafts the result back into the parent graph.
    """

    def __init__(self, parent_graph: Graph, window_nodes: List[Node]):
        self.parent = parent_graph
        self.window_ids = {n.id for n in window_nodes}

    @staticmethod
    def _node_to_dict(n) -> dict:
        if isinstance(n, dict):
            return n
        return {
            "id": n.id,
            "type": getattr(n, "type", ""),
            "label": getattr(n, "label", ""),
            "attributes": getattr(n, "attributes", {}),
        }

    @staticmethod
    def _edge_to_dict(e) -> dict:
        if isinstance(e, dict):
            return e
        return {
            "id": getattr(e, "id", 0),
            "source": e.source,
            "target": e.target,
            # Graph stores it as relationship_type internally; from_dict expects 'type'
            "type": getattr(e, "relationship_type", getattr(e, "type", getattr(e, "label", ""))),
        }

    def extract_subgraph(self) -> Graph:
        """Create a new Graph containing only the windowed nodes and their edges."""
        sub_dict = {
            "nodes": [
                WorkingMemoryWorkspace._node_to_dict(n)
                for n in self.parent.nodes
                if n.id in self.window_ids
            ],
            "edges": [
                WorkingMemoryWorkspace._edge_to_dict(e)
                for e in self.parent.edges
                if e.source in self.window_ids and e.target in self.window_ids
            ],
        }
        return Graph.from_dict(sub_dict)

    def graft(self, mutated_subgraph: Graph) -> Graph:
        """Replace the windowed nodes in the parent graph with the mutated sub-graph."""
        parent_dict = {
            "nodes": [
                WorkingMemoryWorkspace._node_to_dict(n)
                for n in self.parent.nodes
                if n.id not in self.window_ids
            ]
            + [WorkingMemoryWorkspace._node_to_dict(n) for n in mutated_subgraph.nodes],
            "edges": [
                WorkingMemoryWorkspace._edge_to_dict(e)
                for e in self.parent.edges
                if e.source not in self.window_ids or e.target not in self.window_ids
            ]
            + [WorkingMemoryWorkspace._edge_to_dict(e) for e in mutated_subgraph.edges],
        }
        return Graph.from_dict(parent_dict)