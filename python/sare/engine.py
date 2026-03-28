"""
SARE-HX Pure-Python Engine

A Python-native implementation of the core SARE engine.
This provides immediate usability without requiring C++ compilation.

The engine mirrors the C++ core API:
- Graph: node/edge management
- Energy: compositional energy evaluation
- Transforms: pattern-matched graph rewriting
- Search: beam search and MCTS
"""

from __future__ import annotations
import time
import math
import random
import json
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# ═══════════════════════════════════════════════════════════════
#  Graph
# ═══════════════════════════════════════════════════════════════

@dataclass
class Node:
    id: int
    type: str
    label: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    uncertainty: float = 0.0

    def __repr__(self):
        return f"Node({self.id}, {self.type!r}, {self.label!r})"


@dataclass
class Edge:
    id: int
    source: int
    target: int
    relationship_type: str

    def __repr__(self):
        return f"Edge({self.source}→{self.target}, {self.relationship_type!r})"


class Graph:
    """Typed directed graph with attributes."""

    def __init__(self):
        self._nodes: Dict[int, Node] = {}
        self._edges: Dict[int, Edge] = {}
        self._next_node_id = 1
        self._next_edge_id = 1
        self._outgoing: Dict[int, List[int]] = {}  # node_id → [edge_ids]
        self._incoming: Dict[int, List[int]] = {}

    def add_node(self, type: str, label: str = "",
                 attributes: Dict[str, str] = None) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        self._nodes[nid] = Node(nid, type, label, attributes or {})
        self._outgoing[nid] = []
        self._incoming[nid] = []
        return nid

    def add_edge(self, source: int, target: int, rel_type: str) -> int:
        eid = self._next_edge_id
        self._next_edge_id += 1
        self._edges[eid] = Edge(eid, source, target, rel_type)
        self._outgoing.setdefault(source, []).append(eid)
        self._incoming.setdefault(target, []).append(eid)
        return eid

    def remove_node(self, node_id: int):
        # Remove connected edges first
        edges_to_remove = []
        for eid, e in self._edges.items():
            if e.source == node_id or e.target == node_id:
                edges_to_remove.append(eid)
        for eid in edges_to_remove:
            self.remove_edge(eid)
        self._nodes.pop(node_id, None)
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)

    def remove_edge(self, edge_id: int):
        edge = self._edges.pop(edge_id, None)
        if edge:
            if edge_id in self._outgoing.get(edge.source, []):
                self._outgoing[edge.source].remove(edge_id)
            if edge_id in self._incoming.get(edge.target, []):
                self._incoming[edge.target].remove(edge_id)

    def get_node(self, node_id: int) -> Optional[Node]:
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: int) -> Optional[Edge]:
        return self._edges.get(edge_id)

    @property
    def nodes(self) -> List[Node]:
        return list(self._nodes.values())

    @property
    def edges(self) -> List[Edge]:
        return list(self._edges.values())

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def outgoing(self, node_id: int) -> List[Edge]:
        return [self._edges[eid] for eid in self._outgoing.get(node_id, [])
                if eid in self._edges]

    def incoming(self, node_id: int) -> List[Edge]:
        return [self._edges[eid] for eid in self._incoming.get(node_id, [])
                if eid in self._edges]

    def degree(self, node_id: int) -> int:
        return len(self._outgoing.get(node_id, [])) + len(self._incoming.get(node_id, []))

    def clone(self) -> Graph:
        g = Graph()
        g._next_node_id = self._next_node_id
        g._next_edge_id = self._next_edge_id
        g._nodes = {k: Node(v.id, v.type, v.label, dict(v.attributes), v.uncertainty)
                     for k, v in self._nodes.items()}
        g._edges = {k: Edge(v.id, v.source, v.target, v.relationship_type)
                     for k, v in self._edges.items()}
        g._outgoing = {k: list(v) for k, v in self._outgoing.items()}
        g._incoming = {k: list(v) for k, v in self._incoming.items()}
        return g

    def to_dict(self) -> dict:
        return {
            "nodes": [{"id": n.id, "type": n.type, "label": n.label,
                        "attributes": n.attributes}
                       for n in self._nodes.values()],
            "edges": [{"id": e.id, "source": e.source, "target": e.target,
                        "type": e.relationship_type}
                       for e in self._edges.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Graph:
        g = cls()
        max_node_id = 0
        for n in data.get("nodes", []):
            nid = int(n.get("id", g._next_node_id))
            g._nodes[nid] = Node(
                nid,
                n.get("type", "unknown"),
                n.get("label", ""),
                dict(n.get("attributes", {})),
                float(n.get("uncertainty", 0.0)),
            )
            g._outgoing.setdefault(nid, [])
            g._incoming.setdefault(nid, [])
            max_node_id = max(max_node_id, nid)
        g._next_node_id = max_node_id + 1 if g._nodes else 1

        max_edge_id = 0
        for e in data.get("edges", []):
            source = int(e.get("source", -1))
            target = int(e.get("target", -1))
            if source not in g._nodes or target not in g._nodes:
                continue
            eid = int(e.get("id", g._next_edge_id))
            g._edges[eid] = Edge(eid, source, target, e.get("type", "unknown"))
            g._outgoing.setdefault(source, []).append(eid)
            g._incoming.setdefault(target, []).append(eid)
            max_edge_id = max(max_edge_id, eid)
        g._next_edge_id = max_edge_id + 1 if g._edges else 1
        return g

    def pretty_print(self) -> str:
        lines = []
        lines.append(f"Graph: {self.node_count} nodes, {self.edge_count} edges")
        for n in self._nodes.values():
            label = f' "{n.label}"' if n.label else ""
            lines.append(f"  [{n.id}] {n.type}{label}")
        for e in self._edges.values():
            lines.append(f"  {e.source} ──{e.relationship_type}──▶ {e.target}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Energy System
# ═══════════════════════════════════════════════════════════════

@dataclass
class EnergyBreakdown:
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(self.components.values())

    def __repr__(self):
        parts = ", ".join(f"{k}={v:.3f}" for k, v in self.components.items())
        return f"Energy({self.total:.3f}: {parts})"


class EnergyEvaluator:
    """Compositional energy function: E_total = Σ w_i · E_i"""

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 domain: Optional[str] = None):
        self.weights = weights or {
            "syntax": 1.0,
            "complexity": 0.5,
            "redundancy": 0.8,
            "uncertainty": 0.2,
        }
        self.domain = domain  # optional domain hint for domain-aware scoring

    def compute(self, graph: Graph) -> EnergyBreakdown:
        breakdown = EnergyBreakdown()

        # Syntax energy: penalize error nodes, empty types, and unresolved unknowns
        syntax = 0.0
        for n in graph.nodes:
            if n.type == "error" or not n.type:
                syntax += 5.0
            elif n.type == "unknown" or n.label == "?":
                # Unknown/unresolved nodes in QA graphs — high penalty, drives inference
                syntax += 8.0
        breakdown.components["syntax"] = self.weights.get("syntax", 1.0) * syntax

        # Complexity: node count + edge density
        complexity = graph.node_count * 1.0 + graph.edge_count * 0.5
        breakdown.components["complexity"] = self.weights.get("complexity", 0.5) * complexity

        # Redundancy: detect duplicate patterns
        redundancy = 0.0
        labels = [n.label for n in graph.nodes if n.label]
        seen = set()
        for label in labels:
            if label in seen:
                redundancy += 2.0
            seen.add(label)
        # Detect x+0, x*1 patterns
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "add"):
                for e in graph.outgoing(n.id):
                    child = graph.get_node(e.target)
                    if child and child.type == "constant" and child.label == "0":
                        redundancy += 3.0
            if n.type == "operator" and n.label in ("*", "mul"):
                for e in graph.outgoing(n.id):
                    child = graph.get_node(e.target)
                    if child and child.type == "constant" and child.label == "1":
                        redundancy += 3.0
        breakdown.components["redundancy"] = self.weights.get("redundancy", 0.8) * redundancy

        # Uncertainty
        uncertainty = sum(n.uncertainty for n in graph.nodes)
        breakdown.components["uncertainty"] = self.weights.get("uncertainty", 0.2) * uncertainty

        # Domain-aware: calculus transforms are rewrites, penalize unexpanded ops
        if self.domain == "calculus" or any(
            n.type == "operator" and n.label in ("d/dx", "derivative", "diff", "integral")
            for n in graph.nodes
        ):
            calc_penalty = 0.0
            for n in graph.nodes:
                if n.type == "operator" and n.label in ("d/dx", "derivative", "diff", "integral"):
                    is_integ = n.label == "integral"
                    ch = graph.outgoing(n.id)
                    if not ch:
                        calc_penalty += 5.0
                        continue
                    inner = graph.get_node(ch[0].target)
                    if inner is None:
                        calc_penalty += 5.0
                    elif inner.type == "variable":
                        # integral(x) → x²/2 (5 nodes, energy ~5.1); needs penalty > 5.1
                        calc_penalty += 8.0 if is_integ else 1.5
                    elif inner.type == "constant":
                        # integral(c) → c*x (3 nodes, energy ~2); derivative(c) → 0 (1 node)
                        calc_penalty += 5.0 if is_integ else 1.5
                    elif inner.type == "operator" and inner.label in ("^", "**"):
                        # integral(x^n) → x^(n+1)/(n+1) (~5 nodes); derivative(x^n) → n*x^(n-1)
                        calc_penalty += 10.0 if is_integ else 5.0
                    elif inner.type == "operator" and inner.label in ("+", "add"):
                        # integral(f+g) → integral(f)+integral(g) then recurse
                        calc_penalty += 14.0 if is_integ else 8.0
                    elif inner.type == "operator" and inner.label in ("*", "mul"):
                        calc_penalty += 12.0 if is_integ else 8.0
                    elif inner.type == "operator" and inner.label in ("sin", "cos"):
                        sch = graph.outgoing(inner.id)
                        if sch:
                            sin_inner = graph.get_node(sch[0].target)
                            if sin_inner and sin_inner.type == "variable":
                                calc_penalty += 3.0   # deriv(sin/cos(x)): direct rule
                            else:
                                calc_penalty += 15.0  # deriv(sin/cos(compound)): chain rule needed
                        else:
                            calc_penalty += 5.0
                    else:
                        calc_penalty += 10.0 if is_integ else 8.0
            breakdown.components["redundancy"] = (
                breakdown.components.get("redundancy", 0.0) + calc_penalty
            )

        # Domain-aware: distribution - penalize factored forms a*(b+c) in favor of distributed a*b + a*c
        if self.domain == "distribution":
            distribution_penalty = 0.0
            for n in graph.nodes:
                if n.type == "operator" and n.label in ("*", "mul"):
                    # Check if one child is an addition operator (factored form)
                    for e in graph.outgoing(n.id):
                        child = graph.get_node(e.target)
                        if child and child.type == "operator" and child.label in ("+", "add"):
                            distribution_penalty += 5.0  # factored form is "expensive" in distribution domain
                            break
            breakdown.components["complexity"] = (
                breakdown.components.get("complexity", 0.0) + distribution_penalty
            )

        # Domain-aware: factoring - penalize distributed forms a*b + a*c in favor of factored a*(b+c)
        if self.domain == "factoring":
            factoring_penalty = 0.0
            for n in graph.nodes:
                if n.type == "operator" and n.label in ("+", "add"):
                    # Check if both children are multiplication operators with common factor
                    children = graph.outgoing(n.id)
                    if len(children) == 2:
                        c1 = graph.get_node(children[0].target)
                        c2 = graph.get_node(children[1].target)
                        if (c1 and c2 and c1.type == "operator" and c1.label in ("*", "mul") and
                            c2.type == "operator" and c2.label in ("*", "mul")):
                            # Check if they share a common factor
                            c1_children = graph.outgoing(c1.id)
                            c2_children = graph.outgoing(c2.id)
                            if len(c1_children) == 2 and len(c2_children) == 2:
                                c1_labels = [graph.get_node(e.target).label for e in c1_children]
                                c2_labels = [graph.get_node(e.target).label for e in c2_children]
                                # If they share any label, they have a common factor
                                if any(l in c2_labels for l in c1_labels):
                                    factoring_penalty += 5.0  # distributed form is "expensive" in factoring domain
            breakdown.components["complexity"] = (
                breakdown.components.get("complexity", 0.0) + factoring_penalty
            )

        return breakdown


# ═══════════════════════════════════════════════════════════════
#  Graph Utilities
# ═══════════════════════════════════════════════════════════════

def _clone_subtree(graph: "Graph", root_id: int) -> Tuple[int, Dict[int, int]]:
    """
    Deep-copy a subtree in `graph`, returning (new_root_id, old_id→new_id map).
    Uses BFS; does not modify `graph`.
    """
    id_map: Dict[int, int] = {}
    queue = [root_id]
    while queue:
        nid = queue.pop(0)
        if nid in id_map:
            continue
        n = graph.get_node(nid)
        if n is None:
            continue
        new_id = graph.add_node(n.type, n.label)
        if n.attributes:
            graph.get_node(new_id).attributes = dict(n.attributes)
        id_map[nid] = new_id
        for e in graph.outgoing(nid):
            queue.append(e.target)
    # Add edges for the copy
    for old_nid, new_nid in id_map.items():
        for e in graph.outgoing(old_nid):
            new_target = id_map.get(e.target)
            if new_target is not None:
                graph.add_edge(new_nid, new_target, e.relationship_type)
    return id_map[root_id], id_map


# ═══════════════════════════════════════════════════════════════
#  Transforms
# ═══════════════════════════════════════════════════════════════

class Transform:
    """Base class for graph transformations."""

    def name(self) -> str:
        return self.__class__.__name__

    def match(self, graph: Graph) -> List[dict]:
        """Return list of match contexts (dicts with matched node IDs)."""
        return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        """Apply transform. Returns (new_graph, delta_energy_estimate)."""
        return graph.clone(), 0.0


class AddZeroElimination(Transform):
    """x + 0 → x"""

    def name(self) -> str:
        return "add_zero_elim"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "add"):
                children = graph.outgoing(n.id)
                for e in children:
                    child = graph.get_node(e.target)
                    if child and child.type == "constant" and child.label == "0":
                        # Find the other operand
                        other = None
                        for e2 in children:
                            if e2.id != e.id:
                                other = graph.get_node(e2.target)
                        if other:
                            matches.append({
                                "op": n.id,
                                "zero": child.id,
                                "other": other.id,
                            })
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]
        zero_id = context["zero"]
        other_id = context["other"]

        # Redirect edges coming into the operator to point to the other operand
        for e in g.edges:
            if e.target == op_id:
                g.remove_edge(e.id)
                g.add_edge(e.source, other_id, e.relationship_type)

        g.remove_node(op_id)
        g.remove_node(zero_id)
        return g, -3.0


class MulOneElimination(Transform):
    """x * 1 → x"""

    def name(self) -> str:
        return "mul_one_elim"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("*", "mul"):
                children = graph.outgoing(n.id)
                for e in children:
                    child = graph.get_node(e.target)
                    if child and child.type == "constant" and child.label == "1":
                        other = None
                        for e2 in children:
                            if e2.id != e.id:
                                other = graph.get_node(e2.target)
                        if other:
                            matches.append({
                                "op": n.id,
                                "one": child.id,
                                "other": other.id,
                            })
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]
        one_id = context["one"]
        other_id = context["other"]

        for e in g.edges:
            if e.target == op_id:
                g.remove_edge(e.id)
                g.add_edge(e.source, other_id, e.relationship_type)

        g.remove_node(op_id)
        g.remove_node(one_id)
        return g, -3.0


class ConstantFolding(Transform):
    """(const op const) → result"""

    def name(self) -> str:
        return "const_fold"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "-", "*", "/", "add", "sub", "mul", "div"):
                children = graph.outgoing(n.id)
                if len(children) == 2:
                    c1 = graph.get_node(children[0].target)
                    c2 = graph.get_node(children[1].target)
                    if (c1 and c2 and
                        c1.type == "constant" and c2.type == "constant" and
                        c1.label.replace(".", "").replace("-", "").isdigit() and
                        c2.label.replace(".", "").replace("-", "").isdigit()):
                        matches.append({
                            "op": n.id,
                            "left": c1.id, "left_val": float(c1.label),
                            "right": c2.id, "right_val": float(c2.label),
                            "operator": n.label,
                        })
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]
        left_val = context["left_val"]
        right_val = context["right_val"]
        operator = context["operator"]

        ops = {"+": lambda a, b: a + b, "add": lambda a, b: a + b,
               "-": lambda a, b: a - b, "sub": lambda a, b: a - b,
               "*": lambda a, b: a * b, "mul": lambda a, b: a * b,
               "/": lambda a, b: a / b if b != 0 else float("inf"),
               "div": lambda a, b: a / b if b != 0 else float("inf")}

        result = ops.get(operator, lambda a, b: 0)(left_val, right_val)
        import math as _math
        if _math.isinf(result) or _math.isnan(result):
            result_str = str(result)
        else:
            result_str = str(int(result)) if result == int(result) else f"{result:.6g}"

        # Replace operator node with result constant
        op_node = g.get_node(op_id)
        if op_node:
            op_node.type = "constant"
            op_node.label = result_str
            # Remove children
            for e in list(g.outgoing(op_id)):
                target = e.target
                g.remove_edge(e.id)
                # Remove child if no other parents
                if target != op_id and len(g.incoming(target)) == 0:
                    g.remove_node(target)

        return g, -4.0


class DoubleNegation(Transform):
    """--x → x"""

    def name(self) -> str:
        return "double_neg"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        _neg_labels = ("neg", "-", "NOT", "not", "~")
        for n in graph.nodes:
            if n.type == "operator" and n.label in _neg_labels and len(graph.outgoing(n.id)) == 1:
                child_edge = graph.outgoing(n.id)[0]
                child = graph.get_node(child_edge.target)
                if child and child.type == "operator" and child.label in _neg_labels and len(graph.outgoing(child.id)) == 1:
                    inner_edge = graph.outgoing(child.id)[0]
                    matches.append({
                        "outer_neg": n.id,
                        "inner_neg": child.id,
                        "inner_target": inner_edge.target,
                    })
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        outer = context["outer_neg"]
        inner = context["inner_neg"]
        target = context["inner_target"]

        for e in list(g.edges):
            if e.target == outer:
                g.remove_edge(e.id)
                g.add_edge(e.source, target, e.relationship_type)

        # Remove inner→target edge before removing inner
        for e in list(g.outgoing(inner)):
            g.remove_edge(e.id)
        g.remove_node(outer)
        g.remove_node(inner)
        return g, -2.0


class MulZeroElimination(Transform):
    """x * 0 → 0"""

    def name(self) -> str:
        return "mul_zero_elim"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("*", "mul"):
                children = graph.outgoing(n.id)
                for e in children:
                    child = graph.get_node(e.target)
                    if child and child.type == "constant" and child.label == "0":
                        matches.append({"op": n.id, "zero": child.id})
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]
        op_node = g.get_node(op_id)

        # Remove all children
        for e in list(g.outgoing(op_id)):
            target = e.target
            g.remove_edge(e.id)
            if target != op_id and len(g.incoming(target)) == 0:
                g.remove_node(target)

        # Convert operator to constant 0
        if op_node:
            op_node.type = "constant"
            op_node.label = "0"

        return g, -3.0


class MacroTransform(Transform):
    """Composite transform built from a sequence of primitive transforms."""

    def __init__(self, macro_name: str, steps: List[Transform]):
        self._macro_name = macro_name
        self._steps = list(steps)

    def name(self) -> str:
        return self._macro_name

    def match(self, graph: Graph) -> List[dict]:
        if not self._steps:
            return []

        first = self._steps[0]
        contexts = first.match(graph)
        if not contexts:
            return []

        # Filter contexts where the full chain is applicable.
        ok: List[dict] = []
        for ctx in contexts[:12]:
            working = graph.clone()
            new_g, _ = first.apply(working, ctx)
            working = new_g

            valid = True
            for step in self._steps[1:]:
                step_matches = step.match(working)
                if not step_matches:
                    valid = False
                    break
                working, _ = step.apply(working, step_matches[0])

            if valid:
                ok.append(ctx)

        return ok

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        if not self._steps:
            return graph.clone(), 0.0

        # Atomic: if any step fails, return a no-op.
        working = graph.clone()
        total_delta_est = 0.0

        g1, d1 = self._steps[0].apply(working, context)
        working = g1
        total_delta_est += d1

        for step in self._steps[1:]:
            matches = step.match(working)
            if not matches:
                return graph.clone(), 0.0
            working, d = step.apply(working, matches[0])
            total_delta_est += d

        return working, total_delta_est


class DistributiveExpansion(Transform):
    """a * (b + c) → a*b + a*c  [only when CombineLikeTerms can follow]"""

    def name(self) -> str:
        return "distributive_expand"

    def _raw_matches(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("*", "mul"):
                children = graph.outgoing(n.id)
                if len(children) != 2:
                    continue
                c1 = graph.get_node(children[0].target)
                c2 = graph.get_node(children[1].target)
                for a_node, plus_node in [(c1, c2), (c2, c1)]:
                    if (a_node and plus_node and
                            plus_node.type == "operator" and
                            plus_node.label in ("+", "add")):
                        plus_children = graph.outgoing(plus_node.id)
                        if len(plus_children) == 2:
                            b_node = graph.get_node(plus_children[0].target)
                            c_node = graph.get_node(plus_children[1].target)
                            if b_node and c_node:
                                matches.append({
                                    "mul_op": n.id,
                                    "a": a_node.id,
                                    "plus_op": plus_node.id,
                                    "b": b_node.id,
                                    "c": c_node.id,
                                })
                                break
        return matches

    def match(self, graph: Graph) -> List[dict]:
        # Return all raw matches - distribution should apply regardless of like terms
        return self._raw_matches(graph)

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        mul_op_id = context["mul_op"]
        a_id = context["a"]
        plus_op_id = context["plus_op"]
        b_id = context["b"]
        c_id = context["c"]

        a_node = g.get_node(a_id)
        b_node = g.get_node(b_id)
        c_node = g.get_node(c_id)

        # Build a*b
        mul1_id = g.add_node("operator", "*")
        a1_id = g.add_node(a_node.type, a_node.label)
        b1_id = g.add_node(b_node.type, b_node.label)
        g.add_edge(mul1_id, a1_id, "left_operand")
        g.add_edge(mul1_id, b1_id, "right_operand")

        # Build a*c
        mul2_id = g.add_node("operator", "*")
        a2_id = g.add_node(a_node.type, a_node.label)
        c1_id = g.add_node(c_node.type, c_node.label)
        g.add_edge(mul2_id, a2_id, "left_operand")
        g.add_edge(mul2_id, c1_id, "right_operand")

        # Build a*b + a*c
        add_id = g.add_node("operator", "+")
        g.add_edge(add_id, mul1_id, "left_operand")
        g.add_edge(add_id, mul2_id, "right_operand")

        # Redirect parents of mul_op to new add node
        for e in list(g.edges):
            if e.target == mul_op_id:
                g.remove_edge(e.id)
                g.add_edge(e.source, add_id, e.relationship_type)

        g.remove_node(mul_op_id)
        g.remove_node(a_id)
        g.remove_node(plus_op_id)
        g.remove_node(b_id)
        g.remove_node(c_id)
        return g, -2.0


class AlgebraicFactoring(Transform):
    """a*x + b*x → (a+b)*x"""

    def name(self) -> str:
        return "algebraic_factor"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "add"):
                children = graph.outgoing(n.id)
                if len(children) != 2:
                    continue
                left = graph.get_node(children[0].target)
                right = graph.get_node(children[1].target)
                if not (left and right):
                    continue
                if not (left.type == "operator" and left.label in ("*", "mul")):
                    continue
                if not (right.type == "operator" and right.label in ("*", "mul")):
                    continue
                lc = graph.outgoing(left.id)
                rc = graph.outgoing(right.id)
                if len(lc) != 2 or len(rc) != 2:
                    continue
                l0 = graph.get_node(lc[0].target)
                l1 = graph.get_node(lc[1].target)
                r0 = graph.get_node(rc[0].target)
                r1 = graph.get_node(rc[1].target)
                if not (l0 and l1 and r0 and r1):
                    continue
                # Find common factor
                for (la, lb), (ra, rb) in [
                    ((l0, l1), (r0, r1)),
                    ((l0, l1), (r1, r0)),
                    ((l1, l0), (r0, r1)),
                    ((l1, l0), (r1, r0)),
                ]:
                    if la.label == ra.label and la.type == ra.type:
                        matches.append({
                            "add_op": n.id,
                            "left_mul": left.id,
                            "right_mul": right.id,
                            "common": la.id,
                            "right_common": ra.id,
                            "left_other": lb.id,
                            "right_other": rb.id,
                            "common_label": la.label,
                            "common_type": la.type,
                        })
                        break
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        add_op_id = context["add_op"]
        left_mul_id = context["left_mul"]
        right_mul_id = context["right_mul"]
        common_id = context["common"]
        right_common_id = context.get("right_common")
        left_other_id = context["left_other"]
        right_other_id = context["right_other"]

        # Build (left_other + right_other) — reuse original nodes
        inner_add_id = g.add_node("operator", "+")
        g.add_edge(inner_add_id, left_other_id, "left_operand")
        g.add_edge(inner_add_id, right_other_id, "right_operand")

        # Build common * (...) — reuse original common node
        result_mul_id = g.add_node("operator", "*")
        g.add_edge(result_mul_id, common_id, "left_operand")
        g.add_edge(result_mul_id, inner_add_id, "right_operand")

        # Redirect parents of add_op
        for e in list(g.edges):
            if e.target == add_op_id:
                g.remove_edge(e.id)
                g.add_edge(e.source, result_mul_id, e.relationship_type)

        # Disconnect mul operators from their children before removing them
        for e in list(g.outgoing(left_mul_id)):
            g.remove_edge(e.id)
        for e in list(g.outgoing(right_mul_id)):
            g.remove_edge(e.id)

        g.remove_node(add_op_id)
        g.remove_node(left_mul_id)
        g.remove_node(right_mul_id)
        # Remove the duplicate common factor from right_mul (now orphaned)
        if right_common_id is not None:
            try:
                if not g.incoming(right_common_id):
                    g.remove_node(right_common_id)
            except Exception:
                pass
        return g, -2.0


class PerfectSquareTrinomial(Transform):
    """x^2 + 2*b*x + b^2 → (x + b)^2"""

    def name(self) -> str:
        return "perfect_square_trinomial"

    def _collect_addends(self, graph: "Graph", node_id: int) -> List[int]:
        """Recursively collect leaf addend IDs from a chain of + nodes."""
        node = graph.get_node(node_id)
        if node is None:
            return []
        if node.type == "operator" and node.label == "+":
            result: List[int] = []
            for e in graph.outgoing(node_id):
                result.extend(self._collect_addends(graph, e.target))
            return result
        return [node_id]

    def _term_info(self, graph: "Graph", node_id: int):
        """Return (coeff, var_name|None, degree) or None for unrecognised shapes."""
        node = graph.get_node(node_id)
        if node is None:
            return None
        if node.type == "constant":
            try:
                return (float(node.label), None, 0)
            except (ValueError, TypeError):
                return None
        if node.type == "variable":
            return (1.0, node.label, 1)
        if node.type != "operator":
            return None
        ch = graph.outgoing(node_id)
        if node.label in ("^", "**", "pow") and len(ch) == 2:
            base = graph.get_node(ch[0].target)
            exp = graph.get_node(ch[1].target)
            if base and base.type == "variable" and exp and exp.type == "constant":
                try:
                    return (1.0, base.label, int(float(exp.label)))
                except (ValueError, TypeError):
                    pass
        if node.label == "*" and len(ch) == 2:
            left = graph.get_node(ch[0].target)
            right = graph.get_node(ch[1].target)
            if left and left.type == "constant" and right and right.type == "variable":
                try:
                    return (float(left.label), right.label, 1)
                except (ValueError, TypeError):
                    pass
            if right and right.type == "constant" and left and left.type == "variable":
                try:
                    return (float(right.label), left.label, 1)
                except (ValueError, TypeError):
                    pass
            # c * (var ^ n)
            if left and left.type == "constant" and right and right.type == "operator":
                inner = self._term_info(graph, ch[1].target)
                if inner and inner[1] is not None and inner[2] >= 2:
                    try:
                        return (float(left.label) * inner[0], inner[1], inner[2])
                    except (ValueError, TypeError):
                        pass
        return None

    def match(self, graph: "Graph") -> List[dict]:
        matches: List[dict] = []
        seen: set = set()
        for n in graph.nodes:
            if n.type != "operator" or n.label != "+":
                continue
            if n.id in seen:
                continue
            addends = self._collect_addends(graph, n.id)
            if len(addends) != 3:
                continue
            terms = []
            ok = True
            for aid in addends:
                info = self._term_info(graph, aid)
                if info is None:
                    ok = False
                    break
                terms.append(info)
            if not ok:
                continue
            vars_used = {t[1] for t in terms if t[1] is not None}
            if len(vars_used) != 1:
                continue
            var_name = next(iter(vars_used))
            by_deg: dict = {}
            for coeff, _, deg in terms:
                by_deg.setdefault(deg, []).append(coeff)
            if set(by_deg.keys()) != {0, 1, 2}:
                continue
            if any(len(v) != 1 for v in by_deg.values()):
                continue
            a2, a1, a0 = by_deg[2][0], by_deg[1][0], by_deg[0][0]
            if abs(a2 - 1.0) > 1e-9 or a1 <= 0:
                continue
            b = a1 / 2.0
            if abs(a0 - b * b) > 1e-9:
                continue
            seen.add(n.id)
            matches.append({"root": n.id, "var": var_name, "b": b})
        return matches

    def apply(self, graph: "Graph", ctx: dict) -> Tuple["Graph", float]:
        g = graph.clone()
        root_id: int = ctx["root"]
        var_name: str = ctx["var"]
        b: float = ctx["b"]
        b_str = str(int(b)) if b == int(b) else str(b)

        def _remove_subtree(nid: int) -> None:
            for e in list(g.outgoing(nid)):
                child = e.target
                g.remove_edge(e.id)
                _remove_subtree(child)
                if nid != root_id and len(g.incoming(child)) == 0:
                    try:
                        g.remove_node(child)
                    except Exception:
                        pass

        _remove_subtree(root_id)

        root_node = g.get_node(root_id)
        if root_node:
            root_node.type = "operator"
            root_node.label = "^"

        inner_id = g.add_node("operator", "+")
        var_id = g.add_node("variable", var_name)
        const_id = g.add_node("constant", b_str)
        exp_id = g.add_node("constant", "2")

        g.add_edge(root_id, inner_id, "left_operand")
        g.add_edge(root_id, exp_id, "right_operand")
        g.add_edge(inner_id, var_id, "left_operand")
        g.add_edge(inner_id, const_id, "right_operand")

        return g, -4.0


class SetUnionIdentity(Transform):
    """X ∪ ∅ → X"""

    def name(self) -> str:
        return "set_union_identity"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("union", "∪"):
                children = graph.outgoing(n.id)
                if len(children) != 2:
                    continue
                c1 = graph.get_node(children[0].target)
                c2 = graph.get_node(children[1].target)
                for empty, other in [(c1, c2), (c2, c1)]:
                    if (empty and other and
                            empty.type == "constant" and
                            empty.label in ("∅", "empty")):
                        matches.append({
                            "op": n.id,
                            "empty": empty.id,
                            "other": other.id,
                        })
                        break
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]
        empty_id = context["empty"]
        other_id = context["other"]

        for e in list(g.edges):
            if e.target == op_id:
                g.remove_edge(e.id)
                g.add_edge(e.source, other_id, e.relationship_type)

        g.remove_node(op_id)
        g.remove_node(empty_id)
        return g, -2.0


class SetIntersectionIdentity(Transform):
    """X ∩ U → X"""

    def name(self) -> str:
        return "set_intersect_identity"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("intersect", "∩"):
                children = graph.outgoing(n.id)
                if len(children) != 2:
                    continue
                c1 = graph.get_node(children[0].target)
                c2 = graph.get_node(children[1].target)
                for univ, other in [(c1, c2), (c2, c1)]:
                    if (univ and other and
                            univ.type == "constant" and
                            univ.label in ("U", "universal")):
                        matches.append({
                            "op": n.id,
                            "univ": univ.id,
                            "other": other.id,
                        })
                        break
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]
        univ_id = context["univ"]
        other_id = context["other"]

        for e in list(g.edges):
            if e.target == op_id:
                g.remove_edge(e.id)
                g.add_edge(e.source, other_id, e.relationship_type)

        g.remove_node(op_id)
        g.remove_node(univ_id)
        return g, -2.0


class EquationSolver(Transform):
    """Generic = simplification: equal constants → true"""

    def name(self) -> str:
        return "equation_solve"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "=":
                children = graph.outgoing(n.id)
                if len(children) != 2:
                    continue
                lhs = graph.get_node(children[0].target)
                rhs = graph.get_node(children[1].target)
                if not (lhs and rhs):
                    continue
                # Case: both sides are constants with same value
                if (lhs.type == "constant" and rhs.type == "constant" and
                        lhs.label == rhs.label):
                    matches.append({
                        "op": n.id,
                        "lhs": lhs.id,
                        "rhs": rhs.id,
                        "action": "both_equal",
                    })
                # Case: lhs is variable, rhs is constant (already solved)
                elif lhs.type == "variable" and rhs.type == "constant":
                    matches.append({
                        "op": n.id,
                        "lhs": lhs.id,
                        "rhs": rhs.id,
                        "action": "already_solved",
                    })
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]
        action = context["action"]

        if action == "both_equal":
            lhs_id = context["lhs"]
            rhs_id = context["rhs"]
            op_node = g.get_node(op_id)
            if op_node:
                for e in list(g.outgoing(op_id)):
                    target = e.target
                    g.remove_edge(e.id)
                    if len(g.incoming(target)) == 0:
                        g.remove_node(target)
                op_node.type = "constant"
                op_node.label = "true"
        # already_solved: no structural change needed
        return g, -1.0


class CommutativityCanonicalize(Transform):
    """a + b → b + a when a > b alphabetically (canonical ordering)"""

    def name(self) -> str:
        return "commutativity_canonicalize"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "*", "add", "mul"):
                children = graph.outgoing(n.id)
                if len(children) != 2:
                    continue
                left = graph.get_node(children[0].target)
                right = graph.get_node(children[1].target)
                if (left and right and
                        left.type in ("variable", "constant") and
                        right.type in ("variable", "constant") and
                        left.label > right.label):
                    matches.append({
                        "op": n.id,
                        "left_edge": children[0].id,
                        "right_edge": children[1].id,
                        "left": left.id,
                        "right": right.id,
                    })
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]
        left_id = context["left"]
        right_id = context["right"]

        # Remove existing child edges and re-add with swapped targets
        left_edge_id = context["left_edge"]
        right_edge_id = context["right_edge"]
        g.remove_edge(left_edge_id)
        g.remove_edge(right_edge_id)
        g.add_edge(op_id, right_id, "left_operand")
        g.add_edge(op_id, left_id, "right_operand")
        return g, -0.1


class LinearEquationSolver(Transform):
    """x + c = d → x = d-c; x - c = d → x = d+c"""

    def name(self) -> str:
        return "linear_equation_solve"

    def _is_numeric(self, label: str) -> bool:
        try:
            float(label)
            return True
        except ValueError:
            return False

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type != "operator" or n.label != "=":
                continue
            children = graph.outgoing(n.id)
            if len(children) != 2:
                continue
            lhs = graph.get_node(children[0].target)
            rhs = graph.get_node(children[1].target)
            if not (lhs and rhs):
                continue
            if not (rhs.type == "constant" and self._is_numeric(rhs.label)):
                continue
            if lhs.type != "operator" or lhs.label not in ("+", "-"):
                continue
            lhs_children = graph.outgoing(lhs.id)
            if len(lhs_children) != 2:
                continue
            lc0 = graph.get_node(lhs_children[0].target)
            lc1 = graph.get_node(lhs_children[1].target)
            if not (lc0 and lc1):
                continue
            for var_node, const_node in [(lc0, lc1), (lc1, lc0)]:
                if (var_node.type == "variable" and
                        const_node.type == "constant" and
                        self._is_numeric(const_node.label)):
                    matches.append({
                        "eq_op": n.id,
                        "lhs_op": lhs.id,
                        "lhs_op_label": lhs.label,
                        "var": var_node.id,
                        "var_label": var_node.label,
                        "const": const_node.id,
                        "const_val": float(const_node.label),
                        "rhs": rhs.id,
                        "rhs_val": float(rhs.label),
                    })
                    break
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        eq_op_id = context["eq_op"]
        lhs_op_id = context["lhs_op"]
        op_label = context["lhs_op_label"]
        var_label = context["var_label"]
        var_id = context["var"]
        const_id = context["const"]
        rhs_id = context["rhs"]
        const_val = context["const_val"]
        rhs_val = context["rhs_val"]

        # Compute result: x + c = d → x = d - c; x - c = d → x = d + c
        if op_label == "+":
            result = rhs_val - const_val
        else:
            result = rhs_val + const_val
        result_str = str(int(result)) if result == int(result) else f"{result:.6g}"

        # Detach eq_op from old subtrees
        for e in list(g.outgoing(eq_op_id)):
            g.remove_edge(e.id)
        # Remove lhs subtree edges and nodes
        for e in list(g.outgoing(lhs_op_id)):
            g.remove_edge(e.id)
        for nid in (lhs_op_id, var_id, const_id, rhs_id):
            try:
                g.remove_node(nid)
            except Exception:
                pass

        # Build new: eq_op → (new_var, result_const)
        new_var_id = g.add_node("variable", var_label)
        new_const_id = g.add_node("constant", result_str)
        g.add_edge(eq_op_id, new_var_id, "left_operand")
        g.add_edge(eq_op_id, new_const_id, "right_operand")
        return g, -4.0


class MultiplyEquationSolver(Transform):
    """c*x = d → x = d/c"""

    def name(self) -> str:
        return "multiply_equation_solve"

    def _is_numeric(self, label: str) -> bool:
        try:
            float(label)
            return True
        except ValueError:
            return False

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type != "operator" or n.label != "=":
                continue
            children = graph.outgoing(n.id)
            if len(children) != 2:
                continue
            lhs = graph.get_node(children[0].target)
            rhs = graph.get_node(children[1].target)
            if not (lhs and rhs):
                continue
            if not (rhs.type == "constant" and self._is_numeric(rhs.label)):
                continue
            if lhs.type != "operator" or lhs.label not in ("*", "mul"):
                continue
            lhs_children = graph.outgoing(lhs.id)
            if len(lhs_children) != 2:
                continue
            lc0 = graph.get_node(lhs_children[0].target)
            lc1 = graph.get_node(lhs_children[1].target)
            if not (lc0 and lc1):
                continue
            for coeff_node, var_node in [(lc0, lc1), (lc1, lc0)]:
                if (coeff_node.type == "constant" and
                        self._is_numeric(coeff_node.label) and
                        float(coeff_node.label) != 0 and
                        var_node.type == "variable"):
                    matches.append({
                        "eq_op": n.id,
                        "lhs_mul": lhs.id,
                        "coeff_id": coeff_node.id,
                        "var_id": var_node.id,
                        "rhs_id": rhs.id,
                        "coeff_val": float(coeff_node.label),
                        "var_label": var_node.label,
                        "rhs_val": float(rhs.label),
                    })
                    break
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        eq_op_id = context["eq_op"]
        lhs_mul_id = context["lhs_mul"]
        coeff_val = context["coeff_val"]
        var_label = context["var_label"]
        rhs_val = context["rhs_val"]

        result = rhs_val / coeff_val
        result_str = str(int(result)) if result == int(result) else f"{result:.6g}"

        for e in list(g.outgoing(eq_op_id)):
            g.remove_edge(e.id)
        for e in list(g.outgoing(lhs_mul_id)):
            g.remove_edge(e.id)
        for nid in (lhs_mul_id, context.get("coeff_id"), context.get("var_id"), context.get("rhs_id")):
            if nid is not None:
                try:
                    g.remove_node(nid)
                except Exception:
                    pass

        new_var_id = g.add_node("variable", var_label)
        new_const_id = g.add_node("constant", result_str)
        g.add_edge(eq_op_id, new_var_id, "left_operand")
        g.add_edge(eq_op_id, new_const_id, "right_operand")
        return g, -4.0


class EquationSubtractConst(Transform):
    """expr + c = d  →  expr = d-c   (generalises LinearEquationSolver to non-linear LHS)
    Also handles: expr - c = d → expr = d+c"""

    def name(self) -> str:
        return "equation_subtract_const"

    def _is_numeric(self, label: str) -> bool:
        try:
            float(label)
            return True
        except ValueError:
            return False

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type != "operator" or n.label != "=":
                continue
            children = graph.outgoing(n.id)
            if len(children) != 2:
                continue
            lhs = graph.get_node(children[0].target)
            rhs = graph.get_node(children[1].target)
            if not (lhs and rhs):
                continue
            # rhs must be a constant
            if rhs.type != "constant" or not self._is_numeric(rhs.label):
                continue
            # lhs must be an addition or subtraction whose ONE child is a constant
            if lhs.type != "operator" or lhs.label not in ("+", "-"):
                continue
            lhs_ch = graph.outgoing(lhs.id)
            if len(lhs_ch) != 2:
                continue
            lc0 = graph.get_node(lhs_ch[0].target)
            lc1 = graph.get_node(lhs_ch[1].target)
            if not (lc0 and lc1):
                continue
            # Identify which child is the addend constant vs the sub-expression
            for expr_node, addend_node in [(lc0, lc1), (lc1, lc0)]:
                if (addend_node.type == "constant" and
                        self._is_numeric(addend_node.label) and
                        expr_node.type != "constant"):   # sub-expression side
                    matches.append({
                        "eq_op": n.id,
                        "lhs_op": lhs.id,
                        "lhs_label": lhs.label,
                        "expr_node": expr_node.id,
                        "addend_id": addend_node.id,
                        "addend_val": float(addend_node.label),
                        "rhs_id": rhs.id,
                        "rhs_val": float(rhs.label),
                    })
                    break
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        import math
        g = graph.clone()
        eq_op_id   = context["eq_op"]
        lhs_op_id  = context["lhs_op"]
        lhs_label  = context["lhs_label"]
        expr_id    = context["expr_node"]
        addend_id  = context["addend_id"]
        addend_val = context["addend_val"]
        rhs_id     = context["rhs_id"]
        rhs_val    = context["rhs_val"]

        # new RHS constant: expr+c=d → expr=d-c; expr-c=d → expr=d+c
        new_rhs = rhs_val - addend_val if lhs_label == "+" else rhs_val + addend_val
        new_rhs_str = str(int(new_rhs)) if new_rhs == int(new_rhs) else f"{new_rhs:.6g}"

        # detach eq_op
        for e in list(g.outgoing(eq_op_id)):
            g.remove_edge(e.id)
        # detach lhs_op, remove addend and old rhs
        for e in list(g.outgoing(lhs_op_id)):
            g.remove_edge(e.id)
        for nid in (lhs_op_id, addend_id, rhs_id):
            try:
                g.remove_node(nid)
            except Exception:
                pass

        # re-attach: eq_op → (expr_node, new_rhs_const)
        new_rhs_id = g.add_node("constant", new_rhs_str)
        g.add_edge(eq_op_id, expr_id, "left_operand")
        g.add_edge(eq_op_id, new_rhs_id, "right_operand")
        return g, -5.0


class ReflexiveEquality(Transform):
    """x = x  →  True  — reflexive equality tautology."""

    def name(self) -> str:
        return "reflexive_equality"

    def match(self, graph: "Graph") -> list:
        for n in graph.nodes:
            if n.type == "operator" and n.label == "=":
                kids = []
                for e in graph.outgoing(n.id):
                    c = graph.get_node(e.target)
                    if c is not None:
                        kids.append(c)
                if len(kids) == 2 and kids[0].label == kids[1].label and kids[0].type == kids[1].type:
                    return [{"eq_id": n.id, "lhs_id": kids[0].id, "rhs_id": kids[1].id}]
        return []

    def apply(self, graph: "Graph", context: dict):
        g = graph.clone()
        # Replace x = x with a single "True" constant node and rewire
        eq_id  = context["eq_id"]
        lhs_id = context["lhs_id"]
        rhs_id = context["rhs_id"]
        # Remove lhs, rhs, eq; add True node
        for nid in (lhs_id, rhs_id, eq_id):
            try:
                g.remove_node(nid)
            except Exception:
                pass
        g.add_node("constant", "True")
        return g, -4.0


class QuadraticSolver(Transform):
    """x^n = c  →  x = c^(1/n)  when c >= 0 and n is a positive integer.
    E.g. x^2 = 4 → x = 2"""

    def name(self) -> str:
        return "quadratic_solve"

    def _is_numeric(self, label: str) -> bool:
        try:
            float(label)
            return True
        except ValueError:
            return False

    def match(self, graph: Graph) -> List[dict]:
        import math
        matches = []
        for n in graph.nodes:
            if n.type != "operator" or n.label != "=":
                continue
            children = graph.outgoing(n.id)
            if len(children) != 2:
                continue
            lhs = graph.get_node(children[0].target)
            rhs = graph.get_node(children[1].target)
            if not (lhs and rhs):
                continue
            # rhs: non-negative constant
            if rhs.type != "constant" or not self._is_numeric(rhs.label):
                continue
            rhs_val = float(rhs.label)
            if rhs_val < 0:
                continue
            # lhs: x^n  (operator=^ with variable and positive-int constant)
            if lhs.type != "operator" or lhs.label not in ("^", "**", "pow"):
                continue
            lhs_ch = graph.outgoing(lhs.id)
            if len(lhs_ch) != 2:
                continue
            base = graph.get_node(lhs_ch[0].target)
            exp  = graph.get_node(lhs_ch[1].target)
            if not (base and exp):
                continue
            if base.type != "variable":
                continue
            if exp.type != "constant" or not self._is_numeric(exp.label):
                continue
            n_exp = float(exp.label)
            if n_exp <= 0 or n_exp != int(n_exp):
                continue
            matches.append({
                "eq_op":    n.id,
                "lhs_pow":  lhs.id,
                "base_id":  base.id,
                "base_label": base.label,
                "exp_id":   exp.id,
                "n_exp":    n_exp,
                "rhs_id":   rhs.id,
                "rhs_val":  rhs_val,
            })
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        import math
        g = graph.clone()
        eq_op_id   = context["eq_op"]
        lhs_pow_id = context["lhs_pow"]
        base_id    = context["base_id"]
        base_label = context["base_label"]
        exp_id     = context["exp_id"]
        n_exp      = context["n_exp"]
        rhs_id     = context["rhs_id"]
        rhs_val    = context["rhs_val"]

        root = rhs_val ** (1.0 / n_exp)
        # Round to integer if close enough
        root_int = round(root)
        if abs(root - root_int) < 1e-9:
            root_str = str(root_int)
        else:
            root_str = f"{root:.6g}"

        # detach eq_op
        for e in list(g.outgoing(eq_op_id)):
            g.remove_edge(e.id)
        # detach lhs_pow subtree
        for e in list(g.outgoing(lhs_pow_id)):
            g.remove_edge(e.id)
        for nid in (lhs_pow_id, base_id, exp_id, rhs_id):
            try:
                g.remove_node(nid)
            except Exception:
                pass

        # eq_op → (new_var = root)
        new_var_id = g.add_node("variable", base_label)
        new_rhs_id = g.add_node("constant", root_str)
        g.add_edge(eq_op_id, new_var_id, "left_operand")
        g.add_edge(eq_op_id, new_rhs_id, "right_operand")
        return g, -4.0


class CombineLikeTerms(Transform):
    """x+x → 2*x; a*x + b*x → (a+b)*x"""

    def name(self) -> str:
        return "combine_like_terms"

    def _raw_matches(self, graph: Graph) -> List[dict]:
        """Expose match results for lookahead use by DistributiveExpansion."""
        return self.match(graph)

    def _is_numeric(self, label: str) -> bool:
        try:
            float(label)
            return True
        except ValueError:
            return False

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type != "operator" or n.label not in ("+", "add"):
                continue
            children = graph.outgoing(n.id)
            if len(children) != 2:
                continue
            left = graph.get_node(children[0].target)
            right = graph.get_node(children[1].target)
            if not (left and right):
                continue

            # Case A: x + x (bare variables)
            if (left.type == "variable" and right.type == "variable" and
                    left.label == right.label):
                matches.append({
                    "add_op": n.id,
                    "left": left.id,
                    "right": right.id,
                    "left_mul": None,
                    "right_mul": None,
                    "case": "A",
                    "var_label": left.label,
                    "left_coeff": None,
                    "right_coeff": None,
                })
                continue

            # Case C: a*x + x or x + a*x (one coeff explicit, one implicit=1)
            _case_c_found = False
            for mul_side, bare_side in [(left, right), (right, left)]:
                if (mul_side.type == "operator" and mul_side.label in ("*", "mul") and
                        bare_side.type == "variable"):
                    mc = graph.outgoing(mul_side.id)
                    if len(mc) != 2:
                        continue
                    m0 = graph.get_node(mc[0].target)
                    m1 = graph.get_node(mc[1].target)
                    if not (m0 and m1):
                        continue
                    for coeff_n, var_n in [(m0, m1), (m1, m0)]:
                        if (coeff_n.type == "constant" and self._is_numeric(coeff_n.label) and
                                var_n.type == "variable" and var_n.label == bare_side.label):
                            matches.append({
                                "add_op": n.id,
                                "left": left.id,
                                "right": right.id,
                                "left_mul": mul_side.id,
                                "right_mul": None,
                                "case": "C",
                                "var_label": var_n.label,
                                "left_coeff": float(coeff_n.label),
                                "right_coeff": 1.0,
                                "left_is_mul": mul_side is left,
                            })
                            _case_c_found = True
                            break
                    if _case_c_found:
                        break

            # Case B: a*x + b*x
            if not (left.type == "operator" and left.label in ("*", "mul")):
                continue
            if not (right.type == "operator" and right.label in ("*", "mul")):
                continue
            lc = graph.outgoing(left.id)
            rc = graph.outgoing(right.id)
            if len(lc) != 2 or len(rc) != 2:
                continue
            l0 = graph.get_node(lc[0].target)
            l1 = graph.get_node(lc[1].target)
            r0 = graph.get_node(rc[0].target)
            r1 = graph.get_node(rc[1].target)
            if not (l0 and l1 and r0 and r1):
                continue
            for (la, lb), (ra, rb) in [
                ((l0, l1), (r0, r1)),
                ((l0, l1), (r1, r0)),
                ((l1, l0), (r0, r1)),
                ((l1, l0), (r1, r0)),
            ]:
                if (la.type == "constant" and self._is_numeric(la.label) and
                        lb.type == "variable" and
                        ra.type == "constant" and self._is_numeric(ra.label) and
                        rb.type == "variable" and lb.label == rb.label):
                    matches.append({
                        "add_op": n.id,
                        "left": left.id,
                        "right": right.id,
                        "left_mul": left.id,
                        "right_mul": right.id,
                        "case": "B",
                        "var_label": lb.label,
                        "left_coeff": float(la.label),
                        "right_coeff": float(ra.label),
                    })
                    break
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        add_op_id = context["add_op"]
        case = context["case"]
        var_label = context["var_label"]

        if case == "A":
            left_id = context["left"]
            right_id = context["right"]
            # Replace add_op with * 2 x
            for e in list(g.outgoing(add_op_id)):
                g.remove_edge(e.id)
            add_node = g.get_node(add_op_id)
            if add_node:
                add_node.label = "*"
            const2_id = g.add_node("constant", "2")
            var_copy_id = g.add_node("variable", var_label)
            g.add_edge(add_op_id, const2_id, "left_operand")
            g.add_edge(add_op_id, var_copy_id, "right_operand")
            if len(g.incoming(left_id)) == 0:
                g.remove_node(left_id)
            if len(g.incoming(right_id)) == 0:
                g.remove_node(right_id)
            return g, -1.0

        else:  # case B or C
            left_mul_id = context.get("left_mul")
            right_mul_id = context.get("right_mul")
            left_coeff = context["left_coeff"]
            right_coeff = context["right_coeff"]
            total = left_coeff + right_coeff
            total_str = str(int(total)) if total == int(total) else f"{total:.6g}"

            for e in list(g.outgoing(add_op_id)):
                g.remove_edge(e.id)
            add_node = g.get_node(add_op_id)
            if add_node:
                add_node.label = "*"
            coeff_id = g.add_node("constant", total_str)
            var_copy_id = g.add_node("variable", var_label)
            g.add_edge(add_op_id, coeff_id, "left_operand")
            g.add_edge(add_op_id, var_copy_id, "right_operand")
            for mid in (left_mul_id, right_mul_id):
                if mid is not None:
                    try:
                        if len(g.incoming(mid)) == 0:
                            # Collect children before removing edges
                            child_ids = [e.target for e in list(g.outgoing(mid))]
                            for e in list(g.outgoing(mid)):
                                g.remove_edge(e.id)
                            g.remove_node(mid)
                            # Remove orphaned children (the old coeff + var nodes)
                            for cid in child_ids:
                                try:
                                    if len(g.incoming(cid)) == 0:
                                        g.remove_node(cid)
                                except Exception:
                                    pass
                    except Exception:
                        pass
            # Also remove orphaned bare var node (Case C right side)
            if case == "C":
                bare_id = context["right"] if context.get("left_is_mul") else context["left"]
                try:
                    if len(g.incoming(bare_id)) == 0:
                        g.remove_node(bare_id)
                except Exception:
                    pass
            return g, -3.0


class SubtractSelfElimination(Transform):
    """x - x → 0"""

    def name(self) -> str:
        return "self_subtraction"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "-":
                children = graph.outgoing(n.id)
                if len(children) != 2:
                    continue
                left = graph.get_node(children[0].target)
                right = graph.get_node(children[1].target)
                if (left and right and
                        left.type == "variable" and right.type == "variable" and
                        left.label == right.label):
                    matches.append({
                        "op": n.id,
                        "left": left.id,
                        "right": right.id,
                    })
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]
        left_id = context["left"]
        right_id = context["right"]

        for e in list(g.outgoing(op_id)):
            g.remove_edge(e.id)
        op_node = g.get_node(op_id)
        if op_node:
            op_node.type = "constant"
            op_node.label = "0"
        if len(g.incoming(left_id)) == 0:
            g.remove_node(left_id)
        if len(g.incoming(right_id)) == 0:
            g.remove_node(right_id)
        return g, -3.0


class PowerZeroElimination(Transform):
    """x ^ 0 → 1"""

    def name(self) -> str:
        return "power_zero_elim"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("^", "**", "pow"):
                children = graph.outgoing(n.id)
                if len(children) != 2:
                    continue
                left = graph.get_node(children[0].target)
                right = graph.get_node(children[1].target)
                if right and right.type == "constant" and right.label == "0":
                    matches.append({
                        "op": n.id,
                        "left": left.id if left else None,
                        "right": right.id,
                    })
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]

        for e in list(g.outgoing(op_id)):
            target = e.target
            g.remove_edge(e.id)
            if len(g.incoming(target)) == 0:
                g.remove_node(target)

        op_node = g.get_node(op_id)
        if op_node:
            op_node.type = "constant"
            op_node.label = "1"
        return g, -2.0


class PowerOneElimination(Transform):
    """x ^ 1 → x"""

    def name(self) -> str:
        return "power_one_elim"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("^", "**", "pow"):
                children = graph.outgoing(n.id)
                if len(children) != 2:
                    continue
                left = graph.get_node(children[0].target)
                right = graph.get_node(children[1].target)
                if (right and right.type == "constant" and right.label == "1" and left):
                    matches.append({
                        "op": n.id,
                        "left": left.id,
                        "right": right.id,
                    })
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        op_id = context["op"]
        left_id = context["left"]
        right_id = context["right"]

        for e in list(g.edges):
            if e.target == op_id:
                g.remove_edge(e.id)
                g.add_edge(e.source, left_id, e.relationship_type)

        for e in list(g.outgoing(op_id)):
            g.remove_edge(e.id)
        g.remove_node(op_id)
        if len(g.incoming(right_id)) == 0:
            g.remove_node(right_id)
        return g, -2.0


class AdditiveCancellation(Transform):
    """(x + c) - c → x  and  (x - c) + c → x"""

    def name(self) -> str:
        return "additive_cancellation"

    def _is_numeric(self, label: str) -> bool:
        try:
            float(label)
            return True
        except ValueError:
            return False

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for n in graph.nodes:
            if n.type != "operator" or n.label not in ("-", "+"):
                continue
            outer_edges = graph.outgoing(n.id)
            if len(outer_edges) != 2:
                continue
            left = graph.get_node(outer_edges[0].target)
            right = graph.get_node(outer_edges[1].target)
            if not (left and right):
                continue
            outer_op = n.label  # + or -

            # right must be a constant
            if right.type != "constant" or not self._is_numeric(right.label):
                continue
            outer_const_val = float(right.label)

            # left must be a + or - operator
            if left.type != "operator" or left.label not in ("+", "-"):
                continue
            inner_edges = graph.outgoing(left.id)
            if len(inner_edges) != 2:
                continue
            inner_left = graph.get_node(inner_edges[0].target)
            inner_right = graph.get_node(inner_edges[1].target)
            if not (inner_left and inner_right):
                continue
            inner_op = left.label

            # inner_right must be a constant matching outer_const (cancellation)
            if inner_right.type != "constant" or not self._is_numeric(inner_right.label):
                continue
            inner_const_val = float(inner_right.label)

            # (x + c) - c → x: inner_op=+ outer_op=- same constant
            # (x - c) + c → x: inner_op=- outer_op=+ same constant
            if abs(inner_const_val - outer_const_val) < 1e-9:
                if (inner_op == "+" and outer_op == "-") or (inner_op == "-" and outer_op == "+"):
                    matches.append({
                        "outer_op": n.id,
                        "inner_op": left.id,
                        "keep_node": inner_left.id,
                        "remove_inner_const": inner_right.id,
                        "remove_outer_const": right.id,
                    })
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        outer_op_id = context["outer_op"]
        inner_op_id = context["inner_op"]
        keep_id = context["keep_node"]
        inner_const_id = context["remove_inner_const"]
        outer_const_id = context["remove_outer_const"]

        # Find what points to outer_op and rewire to keep_node
        incoming = list(g.incoming(outer_op_id))
        for e in incoming:
            # rewire parent → keep_node
            parent_id = e.source
            rel = e.relationship_type
            g.remove_edge(e.id)
            g.add_edge(parent_id, keep_id, rel)

        # Remove all edges from outer_op and inner_op
        for e in list(g.outgoing(outer_op_id)):
            g.remove_edge(e.id)
        for e in list(g.outgoing(inner_op_id)):
            g.remove_edge(e.id)

        # Remove orphaned nodes
        for nid in (outer_op_id, inner_op_id, inner_const_id, outer_const_id):
            try:
                if len(g.incoming(nid)) == 0:
                    g.remove_node(nid)
            except Exception:
                pass

        return g, -4.0


class DivisionSelfElimination(Transform):
    """x / x → 1"""
    def name(self): return "division_self_elim"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("/", "div"):
                children = graph.outgoing(n.id)
                if len(children) == 2:
                    l = graph.get_node(children[0].target)
                    r = graph.get_node(children[1].target)
                    if l and r and l.type == r.type and l.label == r.label and l.type in ("variable", "constant"):
                        matches.append({"op": n.id, "left": l.id, "right": r.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.outgoing(ctx["op"])): g.remove_edge(e.id)
        op = g.get_node(ctx["op"])
        if op: op.type = "constant"; op.label = "1"
        for nid in (ctx["left"], ctx["right"]):
            try:
                if len(g.incoming(nid)) == 0: g.remove_node(nid)
            except: pass
        return g, -3.0


class BooleanAndTrue(Transform):
    """x and true → x"""
    def name(self): return "bool_and_true"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("and", "AND", "∧"):
                for e in graph.outgoing(n.id):
                    c = graph.get_node(e.target)
                    if c and c.type == "constant" and c.label.lower() in ("true", "1"):
                        others = [graph.get_node(e2.target) for e2 in graph.outgoing(n.id) if e2.id != e.id]
                        if others and others[0]:
                            matches.append({"op": n.id, "true_node": c.id, "other": others[0].id})
                            break
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.edges):
            if e.target == ctx["op"]:
                g.remove_edge(e.id); g.add_edge(e.source, ctx["other"], e.relationship_type)
        g.remove_node(ctx["op"]); g.remove_node(ctx["true_node"])
        return g, -3.0


class BooleanAndFalse(Transform):
    """x and false → false"""
    def name(self): return "bool_and_false"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("and", "AND", "∧"):
                for e in graph.outgoing(n.id):
                    c = graph.get_node(e.target)
                    if c and c.type == "constant" and c.label.lower() in ("false", "0"):
                        others = [graph.get_node(e2.target) for e2 in graph.outgoing(n.id) if e2.id != e.id]
                        matches.append({"op": n.id, "false_node": c.id,
                                        "others": [o.id for o in others if o]})
                        break
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        new_false = g.add_node("constant", "false")
        for e in list(g.edges):
            if e.target == ctx["op"]:
                g.remove_edge(e.id)
                g.add_edge(e.source, new_false, e.relationship_type)
        for cid in ctx.get("others", []):
            try:
                if len(g.incoming(cid)) == 0: g.remove_node(cid)
            except Exception: pass
        try: g.remove_node(ctx["op"])
        except Exception: pass
        try:
            if len(g.incoming(ctx["false_node"])) == 0: g.remove_node(ctx["false_node"])
        except Exception: pass
        return g, -3.0


class BooleanOrFalse(Transform):
    """x or false → x"""
    def name(self): return "bool_or_false"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("or", "OR", "∨"):
                for e in graph.outgoing(n.id):
                    c = graph.get_node(e.target)
                    if c and c.type == "constant" and c.label.lower() in ("false", "0"):
                        others = [graph.get_node(e2.target) for e2 in graph.outgoing(n.id) if e2.id != e.id]
                        if others and others[0]:
                            matches.append({"op": n.id, "false_node": c.id, "other": others[0].id})
                            break
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.edges):
            if e.target == ctx["op"]:
                g.remove_edge(e.id); g.add_edge(e.source, ctx["other"], e.relationship_type)
        g.remove_node(ctx["op"]); g.remove_node(ctx["false_node"])
        return g, -3.0


class BooleanOrTrue(Transform):
    """x or true → true"""
    def name(self): return "bool_or_true"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("or", "OR", "∨"):
                for e in graph.outgoing(n.id):
                    c = graph.get_node(e.target)
                    if c and c.type == "constant" and c.label.lower() in ("true", "1"):
                        others = [graph.get_node(e2.target) for e2 in graph.outgoing(n.id) if e2.id != e.id]
                        matches.append({"op": n.id, "true_node": c.id,
                                        "others": [o.id for o in others if o]})
                        break
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        new_true = g.add_node("constant", "true")
        for e in list(g.edges):
            if e.target == ctx["op"]:
                g.remove_edge(e.id)
                g.add_edge(e.source, new_true, e.relationship_type)
        for cid in ctx.get("others", []):
            try:
                if len(g.incoming(cid)) == 0: g.remove_node(cid)
            except Exception: pass
        try: g.remove_node(ctx["op"])
        except Exception: pass
        try:
            if len(g.incoming(ctx["true_node"])) == 0: g.remove_node(ctx["true_node"])
        except Exception: pass
        return g, -3.0


class BooleanIdempotent(Transform):
    """x and x → x; x or x → x"""
    def name(self): return "bool_idempotent"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("and", "or", "AND", "OR"):
                ch = graph.outgoing(n.id)
                if len(ch) == 2:
                    l = graph.get_node(ch[0].target); r = graph.get_node(ch[1].target)
                    if l and r and l.type == r.type and l.label == r.label:
                        matches.append({"op": n.id, "keep": l.id, "dup": r.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.edges):
            if e.target == ctx["op"]:
                g.remove_edge(e.id); g.add_edge(e.source, ctx["keep"], e.relationship_type)
        g.remove_node(ctx["op"])
        if len(g.incoming(ctx["dup"])) == 0: g.remove_node(ctx["dup"])
        return g, -2.0


class MultiplicativeInverseElim(Transform):
    """x / 1 → x"""
    def name(self): return "div_one_elim"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("/", "div"):
                ch = graph.outgoing(n.id)
                if len(ch) == 2:
                    divisor = graph.get_node(ch[1].target)
                    dividend = graph.get_node(ch[0].target)
                    if divisor and divisor.type == "constant" and divisor.label == "1" and dividend:
                        matches.append({"op": n.id, "one": divisor.id, "other": dividend.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.edges):
            if e.target == ctx["op"]:
                g.remove_edge(e.id); g.add_edge(e.source, ctx["other"], e.relationship_type)
        g.remove_node(ctx["op"]); g.remove_node(ctx["one"])
        return g, -2.0


class AdditionAssociativity(Transform):
    """(a + b) + c → a + (b + c) when b,c are constants (enables folding)"""
    def name(self): return "add_assoc_fold"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "add"):
                ch = graph.outgoing(n.id)
                if len(ch) != 2: continue
                left = graph.get_node(ch[0].target); right = graph.get_node(ch[1].target)
                if not (left and right): continue
                if left.type == "operator" and left.label in ("+", "add"):
                    lch = graph.outgoing(left.id)
                    if len(lch) == 2:
                        ll = graph.get_node(lch[0].target); lr = graph.get_node(lch[1].target)
                        if lr and right and lr.type == "constant" and right.type == "constant":
                            try:
                                float(lr.label); float(right.label)
                                matches.append({"outer": n.id, "inner": left.id, "a": ll.id if ll else None,
                                                "b": lr.id, "c": right.id})
                            except: pass
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        b = g.get_node(ctx["b"]); c = g.get_node(ctx["c"])
        if b and c:
            try:
                result = float(b.label) + float(c.label)
                rs = str(int(result)) if result == int(result) else f"{result:.6g}"
                for e in list(g.outgoing(ctx["outer"])): g.remove_edge(e.id)
                for e in list(g.outgoing(ctx["inner"])): g.remove_edge(e.id)
                a_id = ctx["a"]
                new_const = g.add_node("constant", rs)
                g.get_node(ctx["outer"]).label = "+"
                g.add_edge(ctx["outer"], a_id, "left_operand")
                g.add_edge(ctx["outer"], new_const, "right_operand")
                g.remove_node(ctx["inner"])
                if len(g.incoming(ctx["b"])) == 0: g.remove_node(ctx["b"])
                if len(g.incoming(ctx["c"])) == 0: g.remove_node(ctx["c"])
            except: return g, 0.0
        return g, -2.0


class AbsoluteValueZero(Transform):
    """abs(0) → 0"""
    def name(self): return "abs_zero"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("abs", "||"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    c = graph.get_node(ch[0].target)
                    if c and c.type == "constant" and c.label == "0":
                        matches.append({"op": n.id, "child": c.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        op = g.get_node(ctx["op"])
        for e in list(g.outgoing(ctx["op"])): g.remove_edge(e.id)
        if op: op.type = "constant"; op.label = "0"
        if len(g.incoming(ctx["child"])) == 0: g.remove_node(ctx["child"])
        return g, -2.0


class MultiplyNegativeOne(Transform):
    """x * neg 1 → neg x (or -1 * x → -x)"""
    def name(self): return "mul_neg_one"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("*", "mul"):
                ch = graph.outgoing(n.id)
                if len(ch) == 2:
                    c1 = graph.get_node(ch[0].target)
                    c2 = graph.get_node(ch[1].target)
                    for neg, other in [(c1,c2),(c2,c1)]:
                        if neg and neg.type == "constant" and neg.label == "-1" and other:
                            matches.append({"op": n.id, "neg1": neg.id, "other": other.id})
                            break
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        op = g.get_node(ctx["op"])
        if op: op.label = "neg"; op.type = "operator"
        for e in list(g.outgoing(ctx["op"])): g.remove_edge(e.id)
        g.add_edge(ctx["op"], ctx["other"], "operand")
        if len(g.incoming(ctx["neg1"])) == 0: g.remove_node(ctx["neg1"])
        return g, -1.0


class ZeroDivision(Transform):
    """0 / x → 0 (where x != 0)"""
    def name(self): return "zero_div"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("/", "div"):
                ch = graph.outgoing(n.id)
                if len(ch) == 2:
                    num = graph.get_node(ch[0].target)
                    den = graph.get_node(ch[1].target)
                    if num and num.type == "constant" and num.label == "0" and den:
                        if not (den.type == "constant" and den.label == "0"):
                            matches.append({"op": n.id, "zero": num.id, "den": den.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.outgoing(ctx["op"])): g.remove_edge(e.id)
        op = g.get_node(ctx["op"])
        if op: op.type = "constant"; op.label = "0"
        for nid in (ctx["zero"], ctx["den"]):
            try:
                if len(g.incoming(nid)) == 0: g.remove_node(nid)
            except: pass
        return g, -3.0


class PowerProduct(Transform):
    """x^a * x^b → x^(a+b) when a,b are constants"""
    def name(self): return "power_product"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("*", "mul"):
                ch = graph.outgoing(n.id)
                if len(ch) != 2: continue
                l = graph.get_node(ch[0].target); r = graph.get_node(ch[1].target)
                if not (l and r): continue
                if not (l.type == "operator" and l.label in ("^","**","pow")): continue
                if not (r.type == "operator" and r.label in ("^","**","pow")): continue
                lch = graph.outgoing(l.id); rch = graph.outgoing(r.id)
                if len(lch) != 2 or len(rch) != 2: continue
                lb = graph.get_node(lch[0].target); le = graph.get_node(lch[1].target)
                rb = graph.get_node(rch[0].target); re_n = graph.get_node(rch[1].target)
                if not (lb and le and rb and re_n): continue
                if lb.label == rb.label and lb.type == rb.type:
                    try:
                        ea = float(le.label); eb = float(re_n.label)
                        matches.append({"mul": n.id, "lpow": l.id, "rpow": r.id,
                                        "base": lb.label, "base_type": lb.type,
                                        "exp_sum": ea + eb})
                    except: pass
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        es = ctx["exp_sum"]
        es_str = str(int(es)) if es == int(es) else f"{es:.6g}"
        for e in list(g.outgoing(ctx["mul"])): g.remove_edge(e.id)
        for e in list(g.outgoing(ctx["lpow"])): g.remove_edge(e.id)
        for e in list(g.outgoing(ctx["rpow"])): g.remove_edge(e.id)
        op = g.get_node(ctx["mul"])
        if op: op.label = "^"; op.type = "operator"
        base_id = g.add_node(ctx["base_type"], ctx["base"])
        exp_id = g.add_node("constant", es_str)
        g.add_edge(ctx["mul"], base_id, "base")
        g.add_edge(ctx["mul"], exp_id, "exponent")
        for nid in (ctx["lpow"], ctx["rpow"]):
            try: g.remove_node(nid)
            except: pass
        return g, -3.0


class TrigZero(Transform):
    def name(self): return "trig_zero"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("sin", "tan"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "constant" and child.label == "0":
                        matches.append({"op": n.id, "child": child.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.outgoing(ctx["op"])):
            g.remove_edge(e.id)
        op = g.get_node(ctx["op"])
        if op:
            op.type = "constant"
            op.label = "0"
        if len(g.incoming(ctx["child"])) == 0:
            g.remove_node(ctx["child"])
        return g, -2.0


class LogOne(Transform):
    def name(self): return "log_one"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "log":
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "constant" and child.label == "1":
                        matches.append({"op": n.id, "child": child.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.outgoing(ctx["op"])):
            g.remove_edge(e.id)
        op = g.get_node(ctx["op"])
        if op:
            op.type = "constant"
            op.label = "0"
        if len(g.incoming(ctx["child"])) == 0:
            g.remove_node(ctx["child"])
        return g, -2.0


class CosZero(Transform):
    """cos(0) → 1"""
    def name(self): return "cos_zero"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "cos":
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "constant" and child.label == "0":
                        matches.append({"op": n.id, "child": child.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.outgoing(ctx["op"])):
            g.remove_edge(e.id)
        op = g.get_node(ctx["op"])
        if op:
            op.type = "constant"
            op.label = "1"
        if len(g.incoming(ctx["child"])) == 0:
            g.remove_node(ctx["child"])
        return g, -2.0


class DerivativeConstant(Transform):
    """d/dx(c) → 0 (derivative of constant is zero)"""
    def name(self): return "deriv_const_zero"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("d/dx", "derivative", "diff"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "constant":
                        matches.append({"op": n.id, "child": child.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.outgoing(ctx["op"])):
            g.remove_edge(e.id)
        op = g.get_node(ctx["op"])
        if op:
            op.type = "constant"
            op.label = "0"
        if len(g.incoming(ctx["child"])) == 0:
            g.remove_node(ctx["child"])
        return g, -2.5


class DerivativePower(Transform):
    """d/dx(x^n) → n*x^(n-1) where n is a known integer constant"""
    def name(self): return "deriv_power_rule"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("d/dx", "derivative", "diff"):
                ch = graph.outgoing(n.id)
                if len(ch) != 1:
                    continue
                inner = graph.get_node(ch[0].target)
                if not (inner and inner.type == "operator" and inner.label in ("^", "**", "pow")):
                    continue
                pch = graph.outgoing(inner.id)
                if len(pch) != 2:
                    continue
                base = graph.get_node(pch[0].target)
                exp = graph.get_node(pch[1].target)
                if base and exp and exp.type == "constant" and base.type == "variable":
                    try:
                        n_val = int(float(exp.label))
                        if n_val >= 0:
                            matches.append({"deriv": n.id, "pow": inner.id,
                                            "base": base.id, "exp": exp.id,
                                            "n": n_val})
                    except (ValueError, TypeError):
                        pass
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        n = ctx["n"]
        base_node = g.get_node(ctx["base"])
        if n == 0:
            # d/dx(x^0) = d/dx(1) = 0
            for e in list(g.outgoing(ctx["pow"])):
                g.remove_edge(e.id)
            for nid in (ctx["base"], ctx["exp"]):
                if len(g.incoming(nid)) == 0:
                    try:
                        g.remove_node(nid)
                    except Exception:
                        pass
            for e in list(g.outgoing(ctx["deriv"])):
                g.remove_edge(e.id)
            if len(g.incoming(ctx["pow"])) == 0:
                try:
                    g.remove_node(ctx["pow"])
                except Exception:
                    pass
            deriv_node = g.get_node(ctx["deriv"])
            if deriv_node:
                deriv_node.type = "constant"
                deriv_node.label = "0"
            return g, -3.0
        if n == 1:
            for e in list(g.edges):
                if e.target == ctx["deriv"]:
                    g.remove_edge(e.id)
                    g.add_edge(e.source, ctx["base"], e.relationship_type)
            try:
                g.remove_node(ctx["deriv"])
                g.remove_node(ctx["pow"])
                if len(g.incoming(ctx["exp"])) == 0:
                    g.remove_node(ctx["exp"])
            except Exception:
                pass
            return g, -3.0

        new_exp = n - 1
        coeff_id = g.add_node("constant", str(n))
        pow_node = g.get_node(ctx["pow"])
        if pow_node:
            pow_node.label = "^"
        exp_node = g.get_node(ctx["exp"])
        if exp_node:
            exp_node.label = str(new_exp)
        deriv_node = g.get_node(ctx["deriv"])
        if deriv_node:
            deriv_node.type = "operator"
            deriv_node.label = "*"
        for e in list(g.outgoing(ctx["deriv"])):
            g.remove_edge(e.id)
        g.add_edge(ctx["deriv"], coeff_id, "left_operand")
        g.add_edge(ctx["deriv"], ctx["pow"], "right_operand")
        return g, -2.0


class DerivativeLinear(Transform):
    """d/dx(x) → 1"""
    def name(self): return "deriv_linear"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("d/dx", "derivative", "diff"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "variable":
                        matches.append({"op": n.id, "child": child.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.outgoing(ctx["op"])):
            g.remove_edge(e.id)
        op = g.get_node(ctx["op"])
        if op:
            op.type = "constant"
            op.label = "1"
        if len(g.incoming(ctx["child"])) == 0:
            g.remove_node(ctx["child"])
        return g, -2.0


class SumRuleDerivative(Transform):
    """derivative(f + g) → derivative(f) + derivative(g)"""
    def name(self): return "deriv_sum_rule"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("d/dx", "derivative", "diff"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "operator" and child.label in ("+", "add"):
                        gch = graph.outgoing(child.id)
                        if len(gch) == 2:
                            f = graph.get_node(gch[0].target)
                            g_node = graph.get_node(gch[1].target)
                            if f and g_node:
                                matches.append({"op": n.id, "plus": child.id,
                                                "f": f.id, "g": g_node.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        # Build derivative(f) + derivative(g) — reuse f and g nodes
        df_id = g.add_node("operator", "derivative")
        g.add_edge(df_id, ctx["f"], "operand")
        dg_id = g.add_node("operator", "derivative")
        g.add_edge(dg_id, ctx["g"], "operand")
        add_id = g.add_node("operator", "+")
        g.add_edge(add_id, df_id, "left_operand")
        g.add_edge(add_id, dg_id, "right_operand")
        for e in list(g.edges):
            if e.target == ctx["op"]:
                g.remove_edge(e.id)
                g.add_edge(e.source, add_id, e.relationship_type)
        # Remove old op and plus (children f, g are reused)
        for e in list(g.outgoing(ctx["op"])): g.remove_edge(e.id)
        for e in list(g.outgoing(ctx["plus"])): g.remove_edge(e.id)
        g.remove_node(ctx["op"])
        g.remove_node(ctx["plus"])
        return g, -1.5


class ProductRuleDerivative(Transform):
    """derivative(f * g) → f * derivative(g) + g * derivative(f)"""
    def name(self): return "deriv_product_rule"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("d/dx", "derivative", "diff"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "operator" and child.label in ("*", "mul"):
                        gch = graph.outgoing(child.id)
                        if len(gch) == 2:
                            f = graph.get_node(gch[0].target)
                            g_node = graph.get_node(gch[1].target)
                            if f and g_node:
                                matches.append({"op": n.id, "mul": child.id,
                                                "f": f.id, "g": g_node.id,
                                                "f_type": f.type, "f_label": f.label,
                                                "g_type": g_node.type, "g_label": g_node.label})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        # Build f * derivative(g) + g * derivative(f)
        # Copies needed since each appears in two places
        f_copy = g.add_node(ctx["f_type"], ctx["f_label"])
        g_copy = g.add_node(ctx["g_type"], ctx["g_label"])
        dg_id = g.add_node("operator", "derivative")
        g.add_edge(dg_id, ctx["g"], "operand")
        df_id = g.add_node("operator", "derivative")
        g.add_edge(df_id, ctx["f"], "operand")
        mul1_id = g.add_node("operator", "*")
        g.add_edge(mul1_id, f_copy, "left_operand")
        g.add_edge(mul1_id, dg_id, "right_operand")
        mul2_id = g.add_node("operator", "*")
        g.add_edge(mul2_id, g_copy, "left_operand")
        g.add_edge(mul2_id, df_id, "right_operand")
        add_id = g.add_node("operator", "+")
        g.add_edge(add_id, mul1_id, "left_operand")
        g.add_edge(add_id, mul2_id, "right_operand")
        for e in list(g.edges):
            if e.target == ctx["op"]:
                g.remove_edge(e.id)
                g.add_edge(e.source, add_id, e.relationship_type)
        for e in list(g.outgoing(ctx["op"])): g.remove_edge(e.id)
        for e in list(g.outgoing(ctx["mul"])): g.remove_edge(e.id)
        g.remove_node(ctx["op"])
        g.remove_node(ctx["mul"])
        return g, -1.0


class SinDerivative(Transform):
    """derivative(sin(u)) → cos(u)  when u is a simple variable"""
    def name(self): return "deriv_sin"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("d/dx", "derivative", "diff"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "operator" and child.label in ("sin",):
                        sch = graph.outgoing(child.id)
                        if len(sch) == 1:
                            u = graph.get_node(sch[0].target)
                            if u and u.type == "variable":
                                matches.append({"op": n.id, "sin": child.id, "u": u.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        sin_node = g.get_node(ctx["sin"])
        if sin_node:
            sin_node.label = "cos"
        op = g.get_node(ctx["op"])
        if op:
            for e in list(g.incoming(ctx["op"])):
                g.add_edge(e.source, ctx["sin"], e.relationship_type)
                g.remove_edge(e.id)
            g.remove_node(ctx["op"])
        return g, -2.0


class CosDerivative(Transform):
    """derivative(cos(u)) → neg(sin(u))  when u is a simple variable"""
    def name(self): return "deriv_cos"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("d/dx", "derivative", "diff"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "operator" and child.label in ("cos",):
                        sch = graph.outgoing(child.id)
                        if len(sch) == 1:
                            u = graph.get_node(sch[0].target)
                            if u and u.type == "variable":
                                matches.append({"op": n.id, "cos": child.id, "u": u.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        cos_node = g.get_node(ctx["cos"])
        if cos_node:
            cos_node.label = "sin"
        neg_id = g.add_node("operator", "neg")
        for e in list(g.incoming(ctx["op"])):
            g.add_edge(e.source, neg_id, e.relationship_type)
            g.remove_edge(e.id)
        g.add_edge(neg_id, ctx["cos"], "operand")
        g.remove_node(ctx["op"])
        return g, -2.0


class ChainRuleSin(Transform):
    """derivative(sin(u)) → cos(u) * derivative(u)  for non-trivial u"""
    def name(self): return "chain_rule_sin"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("d/dx", "derivative", "diff"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "operator" and child.label == "sin":
                        sch = graph.outgoing(child.id)
                        if len(sch) == 1:
                            u = graph.get_node(sch[0].target)
                            if u and u.type != "variable" and u.type != "constant":
                                matches.append({"op": n.id, "sin": child.id, "u": u.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        # Deep-copy u subtree for derivative; original u goes to cos
        u_copy_id, _ = _clone_subtree(g, ctx["u"])
        # Build cos(u_original) * derivative(u_copy)
        cos_id = g.add_node("operator", "cos")
        g.add_edge(cos_id, ctx["u"], "operand")
        du_id = g.add_node("operator", "derivative")
        g.add_edge(du_id, u_copy_id, "operand")
        mul_id = g.add_node("operator", "*")
        g.add_edge(mul_id, cos_id, "left_operand")
        g.add_edge(mul_id, du_id, "right_operand")
        for e in list(g.edges):
            if e.target == ctx["op"]:
                g.remove_edge(e.id)
                g.add_edge(e.source, mul_id, e.relationship_type)
        for e in list(g.outgoing(ctx["op"])): g.remove_edge(e.id)
        g.remove_node(ctx["op"])
        for e in list(g.outgoing(ctx["sin"])): g.remove_edge(e.id)
        g.remove_node(ctx["sin"])
        return g, -1.5


class ChainRuleCos(Transform):
    """derivative(cos(u)) → neg(sin(u)) * derivative(u)  for non-trivial u"""
    def name(self): return "chain_rule_cos"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("d/dx", "derivative", "diff"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "operator" and child.label == "cos":
                        sch = graph.outgoing(child.id)
                        if len(sch) == 1:
                            u = graph.get_node(sch[0].target)
                            if u and u.type != "variable" and u.type != "constant":
                                matches.append({"op": n.id, "cos": child.id, "u": u.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        # Deep-copy u subtree for derivative; original u goes to sin
        u_copy_id, _ = _clone_subtree(g, ctx["u"])
        # Build neg(sin(u_original)) * derivative(u_copy)
        sin_id = g.add_node("operator", "sin")
        g.add_edge(sin_id, ctx["u"], "operand")
        neg_id = g.add_node("operator", "neg")
        g.add_edge(neg_id, sin_id, "operand")
        du_id = g.add_node("operator", "derivative")
        g.add_edge(du_id, u_copy_id, "operand")
        mul_id = g.add_node("operator", "*")
        g.add_edge(mul_id, neg_id, "left_operand")
        g.add_edge(mul_id, du_id, "right_operand")
        for e in list(g.edges):
            if e.target == ctx["op"]:
                g.remove_edge(e.id)
                g.add_edge(e.source, mul_id, e.relationship_type)
        for e in list(g.outgoing(ctx["op"])): g.remove_edge(e.id)
        g.remove_node(ctx["op"])
        for e in list(g.outgoing(ctx["cos"])): g.remove_edge(e.id)
        g.remove_node(ctx["cos"])
        return g, -1.5


class ExpDerivative(Transform):
    """derivative(exp(x)) → exp(x)  when x is a simple variable"""
    def name(self): return "deriv_exp"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("d/dx", "derivative", "diff"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "operator" and child.label in ("exp", "e^"):
                        sch = graph.outgoing(child.id)
                        if len(sch) == 1:
                            u = graph.get_node(sch[0].target)
                            if u and u.type == "variable":
                                matches.append({"op": n.id, "exp": child.id, "u": u.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.incoming(ctx["op"])):
            g.add_edge(e.source, ctx["exp"], e.relationship_type)
            g.remove_edge(e.id)
        g.remove_node(ctx["op"])
        return g, -2.0


class LnDerivative(Transform):
    """derivative(ln(x)) → 1 / x  when x is a simple variable"""
    def name(self): return "deriv_ln"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("d/dx", "derivative", "diff"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "operator" and child.label in ("ln", "log"):
                        sch = graph.outgoing(child.id)
                        if len(sch) == 1:
                            u = graph.get_node(sch[0].target)
                            if u and u.type == "variable":
                                matches.append({"op": n.id, "ln": child.id,
                                                "u": u.id, "u_label": u.label})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        one_id = g.add_node("constant", "1")
        x_copy = g.add_node("variable", ctx["u_label"])
        div_id = g.add_node("operator", "/")
        g.add_edge(div_id, one_id, "left_operand")
        g.add_edge(div_id, x_copy, "right_operand")
        for e in list(g.incoming(ctx["op"])):
            g.add_edge(e.source, div_id, e.relationship_type)
            g.remove_edge(e.id)
        for e in list(g.outgoing(ctx["op"])): g.remove_edge(e.id)
        for e in list(g.outgoing(ctx["ln"])): g.remove_edge(e.id)
        g.remove_node(ctx["op"])
        g.remove_node(ctx["ln"])
        if not g.incoming(ctx["u"]): g.remove_node(ctx["u"])
        return g, -2.0


class ChainRuleExp(Transform):
    """derivative(exp(u)) → exp(u) * derivative(u)  for non-trivial u"""
    def name(self): return "chain_rule_exp"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("d/dx", "derivative", "diff"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "operator" and child.label in ("exp", "e^"):
                        sch = graph.outgoing(child.id)
                        if len(sch) == 1:
                            u = graph.get_node(sch[0].target)
                            if u and u.type != "variable" and u.type != "constant":
                                matches.append({"op": n.id, "exp": child.id, "u": u.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        u_copy_id, _ = _clone_subtree(g, ctx["u"])
        exp_id = g.add_node("operator", "exp")
        g.add_edge(exp_id, ctx["u"], "operand")
        du_id = g.add_node("operator", "derivative")
        g.add_edge(du_id, u_copy_id, "operand")
        mul_id = g.add_node("operator", "*")
        g.add_edge(mul_id, exp_id, "left_operand")
        g.add_edge(mul_id, du_id, "right_operand")
        for e in list(g.edges):
            if e.target == ctx["op"]:
                g.remove_edge(e.id)
                g.add_edge(e.source, mul_id, e.relationship_type)
        for e in list(g.outgoing(ctx["op"])): g.remove_edge(e.id)
        g.remove_node(ctx["op"])
        for e in list(g.outgoing(ctx["exp"])): g.remove_edge(e.id)
        g.remove_node(ctx["exp"])
        return g, -1.5


class QuotientRule(Transform):
    """derivative(f / g) → (g * derivative(f) - f * derivative(g)) / g^2"""
    def name(self): return "deriv_quotient_rule"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("d/dx", "derivative", "diff"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.type == "operator" and child.label in ("/", "div"):
                        dch = graph.outgoing(child.id)
                        if len(dch) == 2:
                            f = graph.get_node(dch[0].target)
                            g_node = graph.get_node(dch[1].target)
                            if f and g_node:
                                matches.append({"op": n.id, "div": child.id,
                                                "f": f.id, "g": g_node.id,
                                                "f_type": f.type, "f_label": f.label,
                                                "g_type": g_node.type, "g_label": g_node.label})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        # Build (g * derivative(f) - f * derivative(g)) / g^2
        f_copy = g.add_node(ctx["f_type"], ctx["f_label"])
        g_copy1 = g.add_node(ctx["g_type"], ctx["g_label"])
        g_copy2 = g.add_node(ctx["g_type"], ctx["g_label"])
        df_id = g.add_node("operator", "derivative")
        g.add_edge(df_id, ctx["f"], "operand")
        dg_id = g.add_node("operator", "derivative")
        g.add_edge(dg_id, ctx["g"], "operand")
        mul1_id = g.add_node("operator", "*")
        g.add_edge(mul1_id, g_copy1, "left_operand")
        g.add_edge(mul1_id, df_id, "right_operand")
        mul2_id = g.add_node("operator", "*")
        g.add_edge(mul2_id, f_copy, "left_operand")
        g.add_edge(mul2_id, dg_id, "right_operand")
        numer_id = g.add_node("operator", "-")
        g.add_edge(numer_id, mul1_id, "left_operand")
        g.add_edge(numer_id, mul2_id, "right_operand")
        two_id = g.add_node("constant", "2")
        denom_id = g.add_node("operator", "^")
        g.add_edge(denom_id, g_copy2, "left_operand")
        g.add_edge(denom_id, two_id, "right_operand")
        result_id = g.add_node("operator", "/")
        g.add_edge(result_id, numer_id, "left_operand")
        g.add_edge(result_id, denom_id, "right_operand")
        for e in list(g.edges):
            if e.target == ctx["op"]:
                g.remove_edge(e.id)
                g.add_edge(e.source, result_id, e.relationship_type)
        for e in list(g.outgoing(ctx["op"])): g.remove_edge(e.id)
        for e in list(g.outgoing(ctx["div"])): g.remove_edge(e.id)
        g.remove_node(ctx["op"])
        g.remove_node(ctx["div"])
        return g, -1.0


class IntegralConstant(Transform):
    """integral(c) → c * x  (c is a numeric constant)"""
    def name(self): return "integ_constant"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "integral":
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    inner = graph.get_node(ch[0].target)
                    if inner and inner.type == "constant":
                        try:
                            float(inner.label)
                            matches.append({"integ": n.id, "c": inner.id, "c_label": inner.label})
                        except ValueError:
                            pass
        return matches

    def apply(self, graph, ctx):
        g = graph.clone()
        # Detach integral→c edge first so c is free to be re-parented
        for e in list(g.outgoing(ctx["integ"])):
            g.remove_edge(e.id)
        x_id = g.add_node("variable", "x")
        mul_id = g.add_node("operator", "*")
        g.add_edge(mul_id, ctx["c"], "left_operand")   # reuse original c node
        g.add_edge(mul_id, x_id, "right_operand")
        for e in list(g.edges):
            if e.target == ctx["integ"]:
                g.remove_edge(e.id)
                g.add_edge(e.source, mul_id, e.relationship_type)
        g.remove_node(ctx["integ"])
        return g, -3.0


class IntegralLinear(Transform):
    """integral(x) → x^2 / 2"""
    def name(self): return "integ_linear"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "integral":
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    inner = graph.get_node(ch[0].target)
                    if inner and inner.type == "variable":
                        matches.append({"integ": n.id, "var": inner.id, "var_label": inner.label})
        return matches

    def apply(self, graph, ctx):
        g = graph.clone()
        # Detach integral→var so var can be re-parented
        for e in list(g.outgoing(ctx["integ"])):
            g.remove_edge(e.id)
        two_id = g.add_node("constant", "2")
        pow_id = g.add_node("operator", "^")
        g.add_edge(pow_id, ctx["var"], "left_operand")  # reuse original var
        g.add_edge(pow_id, two_id, "right_operand")
        two2_id = g.add_node("constant", "2")
        div_id = g.add_node("operator", "/")
        g.add_edge(div_id, pow_id, "left_operand")
        g.add_edge(div_id, two2_id, "right_operand")
        for e in list(g.edges):
            if e.target == ctx["integ"]:
                g.remove_edge(e.id)
                g.add_edge(e.source, div_id, e.relationship_type)
        g.remove_node(ctx["integ"])
        return g, -3.0


class IntegralPower(Transform):
    """integral(x^n) → x^(n+1) / (n+1)  for integer n ≥ 2"""
    def name(self): return "integ_power_rule"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "integral":
                ch = graph.outgoing(n.id)
                if len(ch) != 1:
                    continue
                inner = graph.get_node(ch[0].target)
                if not (inner and inner.type == "operator" and inner.label in ("^", "**")):
                    continue
                pch = graph.outgoing(inner.id)
                if len(pch) != 2:
                    continue
                base = graph.get_node(pch[0].target)
                exp = graph.get_node(pch[1].target)
                if not (base and exp):
                    continue
                if base.type != "variable":
                    continue
                try:
                    n_val = int(exp.label)
                    if n_val >= 2:
                        matches.append({
                            "integ": n.id, "pow": inner.id,
                            "var": base.id, "var_label": base.label,
                            "exp_id": exp.id, "n": n_val,
                        })
                except (ValueError, AttributeError):
                    pass
        return matches

    def apply(self, graph, ctx):
        g = graph.clone()
        n1 = ctx["n"] + 1
        # Detach integral→pow so the subtree is free to be re-parented
        for e in list(g.outgoing(ctx["integ"])):
            g.remove_edge(e.id)
        # Update the exponent in-place on the clone (n → n+1)
        exp_node = g.get_node(ctx["exp_id"])
        if exp_node:
            exp_node.label = str(n1)
        # Create denominator and division node, reuse original pow subtree
        denom_id = g.add_node("constant", str(n1))
        div_id = g.add_node("operator", "/")
        g.add_edge(div_id, ctx["pow"], "left_operand")  # reuse original ^ subtree
        g.add_edge(div_id, denom_id, "right_operand")
        for e in list(g.edges):
            if e.target == ctx["integ"]:
                g.remove_edge(e.id)
                g.add_edge(e.source, div_id, e.relationship_type)
        g.remove_node(ctx["integ"])
        return g, -3.0


class IntegralSumRule(Transform):
    """integral(f + g) → integral(f) + integral(g)"""
    def name(self): return "integ_sum_rule"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "integral":
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    inner = graph.get_node(ch[0].target)
                    if inner and inner.type == "operator" and inner.label in ("+", "add"):
                        ach = graph.outgoing(inner.id)
                        if len(ach) == 2:
                            f = graph.get_node(ach[0].target)
                            g_node = graph.get_node(ach[1].target)
                            if f and g_node:
                                matches.append({
                                    "integ": n.id, "add": inner.id,
                                    "f": f.id, "g": g_node.id,
                                })
        return matches

    def apply(self, graph, ctx):
        g = graph.clone()
        if_id = g.add_node("operator", "integral")
        g.add_edge(if_id, ctx["f"], "operand")
        ig_id = g.add_node("operator", "integral")
        g.add_edge(ig_id, ctx["g"], "operand")
        add_id = g.add_node("operator", "+")
        g.add_edge(add_id, if_id, "left_operand")
        g.add_edge(add_id, ig_id, "right_operand")
        for e in list(g.edges):
            if e.target == ctx["integ"]:
                g.remove_edge(e.id)
                g.add_edge(e.source, add_id, e.relationship_type)
        for e in list(g.outgoing(ctx["integ"])): g.remove_edge(e.id)
        for e in list(g.outgoing(ctx["add"])): g.remove_edge(e.id)
        g.remove_node(ctx["integ"])
        g.remove_node(ctx["add"])
        return g, -1.5


class SqrtSquare(Transform):
    def name(self): return "sqrt_square"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "sqrt":
                ch = graph.outgoing(n.id)
                if len(ch) != 1:
                    continue
                inner = graph.get_node(ch[0].target)
                if not (inner and inner.type == "operator" and inner.label in ("^", "**", "pow")):
                    continue
                pch = graph.outgoing(inner.id)
                if len(pch) != 2:
                    continue
                base = graph.get_node(pch[0].target)
                exp = graph.get_node(pch[1].target)
                if base and exp and exp.type == "constant" and exp.label == "2":
                    matches.append({"sqrt": n.id, "pow": inner.id, "base": base.id, "exp": exp.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        sqrt_node = g.get_node(ctx["sqrt"])
        base_node = g.get_node(ctx["base"])
        if sqrt_node and base_node:
            sqrt_node.type = base_node.type
            sqrt_node.label = base_node.label
            sqrt_node.attributes = dict(base_node.attributes)
        for e in list(g.outgoing(ctx["sqrt"])):
            g.remove_edge(e.id)
        for nid in (ctx["pow"], ctx["exp"], ctx["base"]):
            try:
                if len(g.incoming(nid)) == 0:
                    g.remove_node(nid)
            except: pass
        return g, -3.0


# ─────────────────────────────────────────────────────────────
#  PROBABILITY DOMAIN
# ─────────────────────────────────────────────────────────────

class ProbabilityEmptySet(Transform):
    """P(empty) → 0"""
    def name(self): return "prob_empty_zero"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("P", "p", "prob"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.label in ("empty", "Empty", "EMPTY", "∅", "{}"):
                        matches.append({"op": n.id, "child": child.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.outgoing(ctx["op"])): g.remove_edge(e.id)
        op = g.get_node(ctx["op"])
        if op: op.type = "constant"; op.label = "0"
        if len(g.incoming(ctx["child"])) == 0: g.remove_node(ctx["child"])
        return g, -2.0


class ProbabilityComplement(Transform):
    """P(A) + P(not A) → 1"""
    def name(self): return "prob_complement_sum"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "add"):
                ch = graph.outgoing(n.id)
                if len(ch) != 2: continue
                l = graph.get_node(ch[0].target)
                r = graph.get_node(ch[1].target)
                if not (l and r): continue
                for pa, pnota in [(l, r), (r, l)]:
                    if not (pa.type == "operator" and pa.label in ("P", "p", "prob")): continue
                    if not (pnota.type == "operator" and pnota.label in ("P", "p", "prob")): continue
                    pa_ch = graph.outgoing(pa.id)
                    pnota_ch = graph.outgoing(pnota.id)
                    if len(pa_ch) != 1 or len(pnota_ch) != 1: continue
                    a_node = graph.get_node(pa_ch[0].target)
                    nota_node = graph.get_node(pnota_ch[0].target)
                    if not (a_node and nota_node): continue
                    if nota_node.type == "operator" and nota_node.label in ("not", "¬"):
                        nota_ch = graph.outgoing(nota_node.id)
                        if len(nota_ch) == 1:
                            inner = graph.get_node(nota_ch[0].target)
                            if inner and inner.label == a_node.label:
                                matches.append({"add": n.id, "pa": pa.id, "pnota": pnota.id})
                                break
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.edges):
            if e.target == ctx["add"]:
                one_id = g.add_node("constant", "1")
                g.add_edge(e.source, one_id, e.relationship_type)
                g.remove_edge(e.id)
        g.remove_node(ctx["add"]); g.remove_node(ctx["pa"]); g.remove_node(ctx["pnota"])
        return g, -4.0


class ProbabilityOne(Transform):
    """P(universal) → 1; P(Ω) → 1"""
    def name(self): return "prob_universal_one"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("P", "p", "prob"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.label in ("Ω", "omega", "Omega", "OMEGA",
                                                 "universal", "S", "U"):
                        matches.append({"op": n.id, "child": child.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.outgoing(ctx["op"])): g.remove_edge(e.id)
        op = g.get_node(ctx["op"])
        if op: op.type = "constant"; op.label = "1"
        if len(g.incoming(ctx["child"])) == 0: g.remove_node(ctx["child"])
        return g, -2.0


# ─────────────────────────────────────────────────────────────
#  GEOMETRY DOMAIN
# ─────────────────────────────────────────────────────────────

class AngleSumTriangle(Transform):
    """angle_sum(triangle) → 180"""
    def name(self): return "angle_sum_180"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("angle_sum", "sum_angles"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    child = graph.get_node(ch[0].target)
                    if child and child.label in ("triangle", "tri"):
                        matches.append({"op": n.id, "child": child.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        for e in list(g.outgoing(ctx["op"])): g.remove_edge(e.id)
        op = g.get_node(ctx["op"])
        if op: op.type = "constant"; op.label = "180"
        if len(g.incoming(ctx["child"])) == 0: g.remove_node(ctx["child"])
        return g, -2.0


class PythagoreanSimplify(Transform):
    """sqrt(a^2 + b^2) stays, but a^2+b^2 can evaluate when a,b are constants"""
    def name(self): return "pythagorean_eval"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("+", "add"):
                ch = graph.outgoing(n.id)
                if len(ch) != 2: continue
                l = graph.get_node(ch[0].target)
                r = graph.get_node(ch[1].target)
                if not (l and r): continue
                for pow1, pow2 in [(l, r), (r, l)]:
                    if pow1.type == "operator" and pow1.label in ("^", "**"):
                        p1ch = graph.outgoing(pow1.id)
                        if len(p1ch) == 2:
                            b1 = graph.get_node(p1ch[0].target)
                            e1 = graph.get_node(p1ch[1].target)
                            if b1 and e1 and b1.type == "constant" and e1.label == "2":
                                if pow2.type == "operator" and pow2.label in ("^", "**"):
                                    p2ch = graph.outgoing(pow2.id)
                                    if len(p2ch) == 2:
                                        b2 = graph.get_node(p2ch[0].target)
                                        e2 = graph.get_node(p2ch[1].target)
                                        if b2 and e2 and b2.type == "constant" and e2.label == "2":
                                            try:
                                                val = float(b1.label)**2 + float(b2.label)**2
                                                matches.append({"add": n.id, "pow1": pow1.id, "pow2": pow2.id,
                                                                "b1": b1.id, "b2": b2.id, "e1": e1.id, "e2": e2.id,
                                                                "val": val})
                                            except (ValueError, TypeError): pass
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        val = ctx["val"]
        vs = str(int(val)) if val == int(val) else f"{val:.4g}"
        for e in list(g.edges):
            if e.target == ctx["add"]:
                new_id = g.add_node("constant", vs)
                g.add_edge(e.source, new_id, e.relationship_type)
                g.remove_edge(e.id)
        for nid in [ctx["add"], ctx["pow1"], ctx["pow2"], ctx["b1"], ctx["b2"], ctx["e1"], ctx["e2"]]:
            try:
                if len(g.incoming(nid)) == 0: g.remove_node(nid)
            except Exception: pass
        return g, -4.0


# ─────────────────────────────────────────────────────────────
#  EXTENDED PROPOSITIONAL LOGIC
# ─────────────────────────────────────────────────────────────

class ImplicationExpansion(Transform):
    """p → q  ≡  (not p) or q"""
    def name(self): return "implies_expansion"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("→", "implies", "==>", "->"):
                ch = graph.outgoing(n.id)
                if len(ch) == 2:
                    p = graph.get_node(ch[0].target)
                    q = graph.get_node(ch[1].target)
                    if p and q:
                        matches.append({"implies": n.id, "p": p.id, "q": q.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        not_p_id = g.add_node("operator", "not")
        p_copy = g.get_node(ctx["p"])
        p_new_id = g.add_node(p_copy.type if p_copy else "variable", p_copy.label if p_copy else "p")
        g.add_edge(not_p_id, p_new_id, "operand")
        or_id = g.add_node("operator", "or")
        g.add_edge(or_id, not_p_id, "left_operand")
        g.add_edge(or_id, ctx["q"], "right_operand")
        for e in list(g.edges):
            if e.target == ctx["implies"]:
                g.remove_edge(e.id)
                g.add_edge(e.source, or_id, e.relationship_type)
        g.remove_node(ctx["implies"])
        if len(g.incoming(ctx["p"])) == 0: g.remove_node(ctx["p"])
        return g, -1.0


class DeMorganAnd(Transform):
    """not(A and B) → (not A) or (not B)"""
    def name(self): return "demorgan_and"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("not", "¬", "!"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    inner = graph.get_node(ch[0].target)
                    if inner and inner.type == "operator" and inner.label in ("and", "AND", "∧"):
                        and_ch = graph.outgoing(inner.id)
                        if len(and_ch) == 2:
                            a = graph.get_node(and_ch[0].target)
                            b = graph.get_node(and_ch[1].target)
                            if a and b:
                                matches.append({"not": n.id, "and": inner.id, "a": a.id, "b": b.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        not_a = g.add_node("operator", "not")
        a_copy = g.get_node(ctx["a"])
        a_new = g.add_node(a_copy.type if a_copy else "variable", a_copy.label if a_copy else "a")
        g.add_edge(not_a, a_new, "operand")
        not_b = g.add_node("operator", "not")
        b_copy = g.get_node(ctx["b"])
        b_new = g.add_node(b_copy.type if b_copy else "variable", b_copy.label if b_copy else "b")
        g.add_edge(not_b, b_new, "operand")
        or_id = g.add_node("operator", "or")
        g.add_edge(or_id, not_a, "left_operand")
        g.add_edge(or_id, not_b, "right_operand")
        for e in list(g.edges):
            if e.target == ctx["not"]:
                g.remove_edge(e.id); g.add_edge(e.source, or_id, e.relationship_type)
        g.remove_node(ctx["not"]); g.remove_node(ctx["and"])
        return g, -1.5


class DeMorganOr(Transform):
    """not(A or B) → (not A) and (not B)"""
    def name(self): return "demorgan_or"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("not", "¬", "!"):
                ch = graph.outgoing(n.id)
                if len(ch) == 1:
                    inner = graph.get_node(ch[0].target)
                    if inner and inner.type == "operator" and inner.label in ("or", "OR", "∨"):
                        or_ch = graph.outgoing(inner.id)
                        if len(or_ch) == 2:
                            a = graph.get_node(or_ch[0].target)
                            b = graph.get_node(or_ch[1].target)
                            if a and b:
                                matches.append({"not": n.id, "or": inner.id, "a": a.id, "b": b.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        not_a = g.add_node("operator", "not")
        a_copy = g.get_node(ctx["a"])
        a_new = g.add_node(a_copy.type if a_copy else "variable", a_copy.label if a_copy else "a")
        g.add_edge(not_a, a_new, "operand")
        not_b = g.add_node("operator", "not")
        b_copy = g.get_node(ctx["b"])
        b_new = g.add_node(b_copy.type if b_copy else "variable", b_copy.label if b_copy else "b")
        g.add_edge(not_b, b_new, "operand")
        and_id = g.add_node("operator", "and")
        g.add_edge(and_id, not_a, "left_operand")
        g.add_edge(and_id, not_b, "right_operand")
        for e in list(g.edges):
            if e.target == ctx["not"]:
                g.remove_edge(e.id); g.add_edge(e.source, and_id, e.relationship_type)
        g.remove_node(ctx["not"]); g.remove_node(ctx["or"])
        return g, -1.5


class ContrapositiveLaw(Transform):
    """(p → q) ≡ (not q → not p) — simplifies if not is already present"""
    def name(self): return "contrapositive"
    def match(self, graph):
        matches = []
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("→", "implies", "==>", "->"):
                ch = graph.outgoing(n.id)
                if len(ch) == 2:
                    p = graph.get_node(ch[0].target)
                    q = graph.get_node(ch[1].target)
                    if p and q:
                        if (p.type == "operator" and p.label in ("not", "¬") and
                                q.type == "operator" and q.label in ("not", "¬")):
                            p_ch = graph.outgoing(p.id)
                            q_ch = graph.outgoing(q.id)
                            if len(p_ch) == 1 and len(q_ch) == 1:
                                pp = graph.get_node(p_ch[0].target)
                                qq = graph.get_node(q_ch[0].target)
                                if pp and qq:
                                    matches.append({"impl": n.id, "notp": p.id, "notq": q.id,
                                                    "pp": pp.id, "qq": qq.id})
        return matches
    def apply(self, graph, ctx):
        g = graph.clone()
        impl = g.get_node(ctx["impl"])
        if impl: impl.label = "→"
        for e in list(g.outgoing(ctx["impl"])): g.remove_edge(e.id)
        g.add_edge(ctx["impl"], ctx["notq"], "left_operand")
        g.add_edge(ctx["impl"], ctx["notp"], "right_operand")
        return g, -1.0


def _base_transforms() -> List[Transform]:
    transforms = [
        AddZeroElimination(),
        MulOneElimination(),
        ConstantFolding(),
        DoubleNegation(),
        MulZeroElimination(),
        DistributiveExpansion(),
        AlgebraicFactoring(),
        PerfectSquareTrinomial(),
        SetUnionIdentity(),
        SetIntersectionIdentity(),
        EquationSolver(),
        CommutativityCanonicalize(),
        LinearEquationSolver(),
        MultiplyEquationSolver(),
        EquationSubtractConst(),
        QuadraticSolver(),
        CombineLikeTerms(),
        SubtractSelfElimination(),
        PowerZeroElimination(),
        PowerOneElimination(),
        AdditiveCancellation(),
        DivisionSelfElimination(),
        BooleanAndTrue(),
        BooleanAndFalse(),
        BooleanOrFalse(),
        BooleanOrTrue(),
        BooleanIdempotent(),
        MultiplicativeInverseElim(),
        AdditionAssociativity(),
        AbsoluteValueZero(),
        MultiplyNegativeOne(),
        ZeroDivision(),
        PowerProduct(),
        TrigZero(),
        CosZero(),
        LogOne(),
        SqrtSquare(),
        DerivativeConstant(),
        DerivativeLinear(),
        DerivativePower(),
        SumRuleDerivative(),
        ProductRuleDerivative(),
        SinDerivative(),
        CosDerivative(),
        ChainRuleSin(),
        ChainRuleCos(),
        ExpDerivative(),
        LnDerivative(),
        ChainRuleExp(),
        QuotientRule(),
        # Integration
        IntegralConstant(),
        IntegralLinear(),
        IntegralPower(),
        IntegralSumRule(),
        # Probability
        ProbabilityEmptySet(),
        ProbabilityOne(),
        ProbabilityComplement(),
        # Geometry
        AngleSumTriangle(),
        PythagoreanSimplify(),
        # Extended propositional logic
        ImplicationExpansion(),
        DeMorganAnd(),
        DeMorganOr(),
        ContrapositiveLaw(),
    ]

    # Physics transforms (Phase 2 domain expansion)
    try:
        from sare.transforms.physics_transforms import (
            NewtonsSecondLaw, OhmsLaw, KinematicVelocity, KinematicDisplacement,
            EnergyKinetic, EnergyPotential, PVnRT,
            KineticEnergy, GravitationalForce, Momentum, WorkEnergyTransfer,
            HookesLaw, OhmsLawDiv, NewtonAccel
        )
        transforms += [
            NewtonsSecondLaw(), OhmsLaw(), KinematicVelocity(),
            KinematicDisplacement(), EnergyKinetic(), EnergyPotential(), PVnRT(),
            KineticEnergy(), GravitationalForce(), Momentum(), WorkEnergyTransfer(),
            HookesLaw(), OhmsLawDiv(), NewtonAccel()
        ]
    except Exception:
        pass

    # Chemistry transforms (Phase 2 domain expansion)
    try:
        from sare.transforms.chemistry_transforms import (
            IdealGasLaw, StoichiometryCoefficients, AvogadroConversion,
            ConservationOfMass, ChemicalReactionStoichiometry, MassBalance,
            MolarMassEquation, PHDefinition, GibbsFreeEnergy
        )
        transforms += [
            IdealGasLaw(), StoichiometryCoefficients(), AvogadroConversion(),
            ConservationOfMass(), ChemicalReactionStoichiometry(), MassBalance(),
            MolarMassEquation(), PHDefinition(), GibbsFreeEnergy()
        ]
    except Exception:
        pass

    return transforms


def _macro_transforms(base: List[Transform]) -> List[Transform]:
    try:
        from sare.meta.macro_registry import list_macros
    except Exception:
        return []

    by_name = {t.name(): t for t in base}
    macros = []
    for spec in list_macros():
        if not spec.enabled:
            continue
        steps: List[Transform] = []
        missing = False
        for step_name in spec.steps:
            t = by_name.get(step_name)
            if not t:
                missing = True
                break
            steps.append(t)
        if missing or len(steps) < 2:
            continue
        macros.append(MacroTransform(spec.name, steps))
    return macros


def get_transforms(include_macros: bool = True,
                   concept_registry=None,
                   include_synthesized: bool = True) -> List[Transform]:
    """
    Build the full transform list for search.

    Args:
        include_macros:      Include mined macro-transforms.
        concept_registry:    Optional ConceptRegistry (C++ or Python).
                             If provided, learned rules are injected as
                             ConceptRule transforms (TODO-06).
        include_synthesized: If True, include promoted synthesized transforms
                             from TransformSynthesizer (Transfer Learning).
    """
    base = _base_transforms()

    # ── Inject learned concept transforms (TODO-06) ──
    concept_rules: List[Transform] = []
    if concept_registry is not None:
        try:
            from sare.memory.concept_rule import concept_transforms_from_registry  # type: ignore
            concept_rules = concept_transforms_from_registry(concept_registry)
        except Exception as _e:
            pass  # graceful degradation

    # ── Inject promoted synthesized transforms (Transfer Learning) ──
    synth_transforms: List[Transform] = []
    if include_synthesized:
        try:
            from sare.transfer.synthesizer import TransformSynthesizer
            _synth = TransformSynthesizer()
            synth_transforms = _synth.get_live_transforms()
        except Exception as _e:
            pass  # graceful degradation

    # Equation-solving transforms must come BEFORE learned concept rules to prevent
    # identity rules (additive_identity etc.) from incorrectly simplifying equations
    _eq_solvers = [
        ReflexiveEquality(),
        LinearEquationSolver(),
        MultiplyEquationSolver(),
        EquationSubtractConst(),
    ]

    if not include_macros:
        return _eq_solvers + concept_rules + synth_transforms + base
    return _eq_solvers + concept_rules + synth_transforms + _macro_transforms(base) + base


def reload_transforms(include_macros: bool = True,
                      concept_registry=None) -> List[Transform]:
    global ALL_TRANSFORMS
    ALL_TRANSFORMS = get_transforms(include_macros=include_macros,
                                    concept_registry=concept_registry)
    return ALL_TRANSFORMS


# Default transforms (macros + primitives)
ALL_TRANSFORMS = get_transforms(include_macros=True)


# ═══════════════════════════════════════════════════════════════
#  Search
# ═══════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    graph: Graph
    energy: EnergyBreakdown
    transforms_applied: List[str]
    steps_taken: int = 0
    expansions: int = 0
    elapsed_seconds: float = 0.0
    energy_trajectory: List[float] = field(default_factory=list)


class BeamSearch:
    """Deterministic energy-minimizing beam search with attention guidance."""

    def search(self, graph: Graph, energy: EnergyEvaluator,
               transforms: List[Transform],
               beam_width: int = 8, max_depth: int = 50,
               budget_seconds: float = 30.0,
               kappa: float = 0.1,
               heuristic_fn: Optional[Callable[[Graph], float]] = None,
               attention_guided: bool = True,
               on_step: Optional[Callable] = None,
               attention_scorer=None,
               **kwargs) -> SearchResult:

        def score(g: Graph, e: EnergyBreakdown) -> float:
            h = heuristic_fn(g) if heuristic_fn else 0.0
            return e.total - (kappa * h)

        start_time = time.time()
        initial_energy = energy.compute(graph)

        # Beam: list of (graph, energy, trace, score)
        initial_score = score(graph, initial_energy)
        beam = [(graph, initial_energy, [], initial_score)]
        best = (graph, initial_energy, [], initial_score)
        trajectory = [initial_energy.total]
        expansions = 0

        # P1-C: per-search expanded graph hash dedup set
        _expanded_hashes: set = set()

        # P1-B: early-exit tracking
        _best_energy_history: list = []

        # Attention: focus transforms on high-priority nodes
        _attention_available = False
        if attention_guided:
            try:
                from sare.memory.attention import AttentionSelector
                _attention_available = True
            except Exception:
                pass

        for depth in range(max_depth):
            if time.time() - start_time > budget_seconds:
                break

            candidates = []
            for g, e, trace, _ in beam:
                # P1-C: skip expansion if this graph was already expanded
                try:
                    _g_hash = hash(tuple(sorted(
                        (getattr(n, 'node_type', getattr(n, 'type', '')),
                         str(getattr(n, 'value', '') or ''))
                        for n in g.nodes
                    )))
                    if _g_hash in _expanded_hashes:
                        continue
                    _expanded_hashes.add(_g_hash)
                except Exception:
                    pass

                # Attention-guided: score nodes, prioritize transforms
                # that match high-attention nodes
                attention_boost = {}
                if _attention_available and g.node_count > 3:
                    try:
                        scored_nodes = [
                            (AttentionSelector.score(n, g), n)
                            for n in g.nodes
                        ]
                        scored_nodes.sort(key=lambda x: x[0], reverse=True)
                        # Top nodes get attention boost
                        for rank, (ascore, node) in enumerate(scored_nodes[:3]):
                            attention_boost[node.id] = max(0, 1.0 - rank * 0.3)
                    except Exception:
                        pass

                for transform in transforms:
                    matches = transform.match(g)
                    for ctx in matches[:3]:
                        new_g, delta_est = transform.apply(g, ctx)
                        new_e = energy.compute(new_g)
                        new_score = score(new_g, new_e)

                        # Attention bonus: lower score (= better) if transform
                        # operates on high-attention nodes
                        if attention_boost and isinstance(ctx, dict):
                            for v in ctx.values():
                                if isinstance(v, int) and v in attention_boost:
                                    new_score -= attention_boost[v] * 0.5
                                    break

                        candidates.append(
                            (new_g, new_e, trace + [transform.name()], new_score)
                        )
                        if on_step is not None:
                            try:
                                on_step({
                                    "step": expansions,
                                    "transform": transform.name(),
                                    "energy_before": e.total,
                                    "energy_after": new_e.total,
                                    "delta": e.total - new_e.total,
                                })
                            except Exception:
                                pass
                        expansions += 1

            if not candidates:
                break

            # Keep top-k by score, where lower is better.
            candidates.sort(key=lambda x: x[3])

            # P1-A: AttentionBeamScorer rerank (optional)
            if attention_scorer is not None:
                try:
                    _attn_pairs = [(c[0], c[1].total) for c in candidates[:beam_width * 2]]
                    _attn_paths = [c[2] for c in candidates[:beam_width * 2]]
                    _reranked = attention_scorer.rerank(_attn_pairs, beam_width, _attn_paths)
                    # Rebuild beam from reranked BeamState objects
                    _beam_new = []
                    for _bs in _reranked:
                        # Match back to full candidate tuple
                        for c in candidates:
                            if c[0] is _bs.graph:
                                _beam_new.append(c)
                                break
                        if len(_beam_new) >= beam_width:
                            break
                    if _beam_new:
                        beam = _beam_new
                    else:
                        beam = candidates[:beam_width]
                except Exception:
                    beam = candidates[:beam_width]
            else:
                beam = candidates[:beam_width]

            if beam[0][3] < best[3]:
                best = beam[0]

            trajectory.append(best[1].total)

            # P1-B: Early-exit if no improvement for 3 consecutive depths
            _best_energy_history.append(best[1].total)
            if len(_best_energy_history) >= 3:
                _last3 = _best_energy_history[-3:]
                if all(abs(_last3[i] - _last3[i+1]) < 0.001 for i in range(2)):
                    break

        elapsed = time.time() - start_time
        return SearchResult(
            graph=best[0],
            energy=best[1],
            transforms_applied=best[2],
            steps_taken=len(best[2]),
            expansions=expansions,
            elapsed_seconds=elapsed,
            energy_trajectory=trajectory,
        )


_HEURISTIC_CACHE: Dict[str, Optional[Callable[[Graph], float]]] = {}
_HEURISTIC_LOCK = threading.Lock()


def load_heuristic_scorer(model_path: Optional[str] = None) -> Optional[Callable[[Graph], float]]:
    if torch is None:
        return None

    from sare.heuristics.heuristic_model import HeuristicModel

    resolved = Path(model_path) if model_path else (Path(__file__).resolve().parents[2] / "models" / "heuristic_v1.pt")
    key = str(resolved)
    with _HEURISTIC_LOCK:
        if key in _HEURISTIC_CACHE:
            return _HEURISTIC_CACHE[key]

    if not resolved.exists():
        with _HEURISTIC_LOCK:
            _HEURISTIC_CACHE[key] = None
        return None

    from sare.heuristics.graph_embedding import _DEVICE as _H_DEVICE
    model = HeuristicModel()
    state = torch.load(resolved, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(_H_DEVICE)
    model.eval()

    def scorer(graph: Graph) -> float:
        nodes = graph.nodes
        if not nodes:
            return 0.0

        id_to_idx = {n.id: idx for idx, n in enumerate(nodes)}
        type_indices = torch.tensor(
            [model.embedding.encoder.get_type_idx(n.type or "unknown") for n in nodes],
            dtype=torch.long,
        )
        adjacency = []
        for edge in graph.edges:
            src = id_to_idx.get(edge.source)
            tgt = id_to_idx.get(edge.target)
            if src is not None and tgt is not None:
                adjacency.append((src, tgt))
        return float(model.predict(type_indices, adjacency))

    with _HEURISTIC_LOCK:
        _HEURISTIC_CACHE[key] = scorer
    return scorer


class MCTSSearch:
    """Monte Carlo Tree Search with UCB1."""

    def search(self, graph: Graph, energy: EnergyEvaluator,
               transforms: List[Transform],
               iterations: int = 100,
               budget_seconds: float = 10.0,
               exploration: float = 1.414) -> SearchResult:

        start_time = time.time()
        initial_energy = energy.compute(graph)

        best_graph = graph
        best_energy = initial_energy
        best_trace = []
        trajectory = [initial_energy.total]
        expansions = 0

        # Simple MCTS: iterate and random-walk
        for _ in range(iterations):
            if time.time() - start_time > budget_seconds:
                break

            # Random rollout from current best
            g = best_graph.clone()
            trace = list(best_trace)
            rollout_depth = random.randint(1, 5)

            for _ in range(rollout_depth):
                all_matches = []
                for t in transforms:
                    for ctx in t.match(g):
                        all_matches.append((t, ctx))

                if not all_matches:
                    break

                t, ctx = random.choice(all_matches)
                g, _ = t.apply(g, ctx)
                trace.append(t.name())
                expansions += 1

            e = energy.compute(g)
            if e.total < best_energy.total:
                best_graph = g
                best_energy = e
                best_trace = trace
                trajectory.append(e.total)

        elapsed = time.time() - start_time
        return SearchResult(
            graph=best_graph,
            energy=best_energy,
            transforms_applied=best_trace,
            steps_taken=len(best_trace),
            expansions=expansions,
            elapsed_seconds=elapsed,
            energy_trajectory=trajectory,
        )


# ═══════════════════════════════════════════════════════════════
#  Problem Builders
# ═══════════════════════════════════════════════════════════════

_UNICODE_SUPERSCRIPTS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
}

def _normalize_informal_math(expr: str) -> str:
    """
    Normalize informal/unicode math notation to strict infix before tokenizing.

    Handles:
      - Unicode superscripts: x² → x^2, x³ → x^3
      - Unicode minus/multiplication: − → -, × → *
      - Implied multiplication: 2x → 2*x, 3(x+1) → 3*(x+1), (a+b)(a-b) → (a+b)*(a-b)
      - Trailing 'd' in derivatives: "dx^2" stays as-is (derivative notation handled by parser)
    """
    import re

    # 1. Unicode operators
    expr = expr.replace("−", "-").replace("–", "-")
    expr = expr.replace("×", "*").replace("·", "*")
    expr = expr.replace("÷", "/")

    # 2. Unicode superscripts — e.g. x² → x^2, x²³ → x^(23)
    # Collect consecutive superscript digits into an exponent
    result = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c in _UNICODE_SUPERSCRIPTS:
            # Collect full superscript run
            exp_digits = []
            while i < len(expr) and expr[i] in _UNICODE_SUPERSCRIPTS:
                exp_digits.append(_UNICODE_SUPERSCRIPTS[expr[i]])
                i += 1
            result.append("^")
            result.append("".join(exp_digits))
        else:
            result.append(c)
            i += 1
    expr = "".join(result)

    # 3. Implied multiplication — insert '*' between:
    #    number followed by variable/letter:  2x → 2*x, 3y → 3*y
    expr = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr)
    #    ) followed by (  or letter/number:   (x+1)(x-1) → (x+1)*(x-1), (x+1)y → (x+1)*y
    expr = re.sub(r'\)([a-zA-Z0-9(])', r')*\1', expr)
    #    letter followed by (  (function calls are already "name(": skip those handled by tokenizer)
    #    e.g. "a(b+c)" where a is a single-letter non-function variable
    #    We can't reliably distinguish x(y) from sin(y) here, so leave letter-paren alone.

    return expr


def build_expression_graph(expr_str: str) -> Graph:
    """
    Parse simple mathematical expressions into SARE graphs.
    Supports: x, y, z (variables), numbers, +, -, *, /
    """
    import re as _re
    # Normalize d/dx(...) → derivative(...) before tokenizing
    expr_str = _re.sub(r'\bd/d[a-z]\s*\(', 'derivative(', expr_str.strip())
    # Strip leading unary '+': '+x = x' → 'x = x' (unary plus is identity)
    expr_str = _re.sub(r'^\s*\+\s*', '', expr_str)
    expr_str = _normalize_informal_math(expr_str)
    g = Graph()

    # Simple recursive descent parser
    tokens = _tokenize(expr_str)
    if not tokens:
        g.add_node("error", "empty_expression")
        return g

    _, root_id = _parse_expression(g, tokens, 0)
    root = g.get_node(root_id)
    if root and root.type == "error":
        import logging as _logging
        _logging.getLogger(__name__).warning("[engine] Parse error: %s in '%s'", root.label, expr_str)
    return g


def _tokenize(s: str) -> List[str]:
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c == "*" and i + 1 < len(s) and s[i + 1] == "*":
            tokens.append("**")
            i += 2
        elif c in "+-*/()=^,∪∩":
            tokens.append(c)
            i += 1
        elif c == "∅":
            tokens.append("∅")
            i += 1
        elif c.isdigit() or (c == '.' and i + 1 < len(s) and s[i+1].isdigit()):
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                j += 1
            tokens.append(s[i:j])
            i = j
        elif c.isalpha() or c == '_':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            i += 1
    return tokens


def _parse_expression(g: Graph, tokens: List[str], pos: int) -> Tuple[int, int]:
    """Returns (new_pos, node_id). Handles = at lowest precedence."""
    pos, left_id = _parse_set(g, tokens, pos)

    if pos < len(tokens) and tokens[pos] == "=":
        pos += 1  # consume '='
        pos, right_id = _parse_set(g, tokens, pos)
        eq_id = g.add_node("operator", "=")
        g.add_edge(eq_id, left_id, "left_operand")
        g.add_edge(eq_id, right_id, "right_operand")
        left_id = eq_id

    return pos, left_id


def _parse_boolean(g: Graph, tokens: List[str], pos: int) -> Tuple[int, int]:
    """Returns (new_pos, node_id) for and/or operators."""
    pos, left_id = _parse_additive(g, tokens, pos)

    while pos < len(tokens) and tokens[pos].lower() in ("and", "or", "∧", "∨",
                                                          "implies", "→", "==>"): 
        op = tokens[pos].lower()
        if op in ("and", "∧"): op_canonical = "and"
        elif op in ("implies", "→", "==>"): op_canonical = "→"
        else: op_canonical = "or"
        pos += 1
        pos, right_id = _parse_additive(g, tokens, pos)
        op_id = g.add_node("operator", op_canonical)
        g.add_edge(op_id, left_id, "left_operand")
        g.add_edge(op_id, right_id, "right_operand")
        left_id = op_id

    return pos, left_id


def _parse_set(g: Graph, tokens: List[str], pos: int) -> Tuple[int, int]:
    """Returns (new_pos, node_id) for union and intersect operators."""
    pos, left_id = _parse_boolean(g, tokens, pos)

    while pos < len(tokens) and tokens[pos] in ("∪", "∩", "union", "intersect"):
        op = tokens[pos]
        if op in ("∪", "union"): op_canonical = "union"
        elif op in ("∩", "intersect"): op_canonical = "intersect"
        else: op_canonical = op
        pos += 1
        pos, right_id = _parse_boolean(g, tokens, pos)
        op_id = g.add_node("operator", op_canonical)
        g.add_edge(op_id, left_id, "left_operand")
        g.add_edge(op_id, right_id, "right_operand")
        left_id = op_id

    return pos, left_id


def _parse_additive(g: Graph, tokens: List[str], pos: int) -> Tuple[int, int]:
    """Returns (new_pos, node_id) for + and - operators."""
    pos, left_id = _parse_term(g, tokens, pos)

    while pos < len(tokens) and tokens[pos] in ("+", "-"):
        op = tokens[pos]
        pos += 1
        pos, right_id = _parse_term(g, tokens, pos)
        op_id = g.add_node("operator", op)
        g.add_edge(op_id, left_id, "left_operand")
        g.add_edge(op_id, right_id, "right_operand")
        left_id = op_id

    return pos, left_id


def _parse_term(g: Graph, tokens: List[str], pos: int) -> Tuple[int, int]:
    pos, left_id = _parse_power(g, tokens, pos)

    while pos < len(tokens) and tokens[pos] in ("*", "/"):
        op = tokens[pos]
        pos += 1
        pos, right_id = _parse_power(g, tokens, pos)
        op_id = g.add_node("operator", op)
        g.add_edge(op_id, left_id, "left_operand")
        g.add_edge(op_id, right_id, "right_operand")
        left_id = op_id

    return pos, left_id


def _parse_power(g: Graph, tokens: List[str], pos: int) -> Tuple[int, int]:
    pos, left_id = _parse_unary(g, tokens, pos)
    if pos < len(tokens) and tokens[pos] in ("^", "**"):
        op = tokens[pos]
        pos += 1
        pos, right_id = _parse_power(g, tokens, pos)
        op_id = g.add_node("operator", op)
        g.add_edge(op_id, left_id, "left_operand")
        g.add_edge(op_id, right_id, "right_operand")
        left_id = op_id
    return pos, left_id


def _parse_unary(g: Graph, tokens: List[str], pos: int) -> Tuple[int, int]:
    """Handle unary - / ~ / NOT / NEG / not / neg operators."""
    if pos < len(tokens) and tokens[pos].upper() in ("-", "~", "NOT", "NEG"):
        tok = tokens[pos]
        label = "neg" if tok.upper() in ("-", "NEG") else "not"
        pos += 1
        pos, child_id = _parse_unary(g, tokens, pos)
        neg_id = g.add_node("operator", label)
        g.add_edge(neg_id, child_id, "operand")
        return pos, neg_id
    return _parse_atom(g, tokens, pos)


def _parse_atom(g: Graph, tokens: List[str], pos: int) -> Tuple[int, int]:
    if pos >= len(tokens):
        return pos, g.add_node("error", "unexpected_end")

    token = tokens[pos]

    if token == "(":
        pos += 1  # skip (
        pos, node_id = _parse_expression(g, tokens, pos)
        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1  # skip )
        return pos, node_id

    if token[0].isdigit() or (token[0] == '.' and len(token) > 1):
        node_id = g.add_node("constant", token)
        return pos + 1, node_id

    _FUNC_NAMES = {"sin", "cos", "tan", "log", "sqrt", "abs", "exp", "ln",
                   "derivative", "diff", "integral",
                   "p", "prob", "angle_sum", "sum_angles"}
    if (token[0].isalpha() or token[0] == '_') and pos + 1 < len(tokens) and tokens[pos + 1] == "(" and token.lower() in _FUNC_NAMES:
        fn = token.lower()
        pos += 2
        pos, arg_id = _parse_expression(g, tokens, pos)
        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1
        fn_id = g.add_node("operator", fn)
        g.add_edge(fn_id, arg_id, "operand")
        return pos, fn_id

    if token.lower() in ("true", "false"):
        node_id = g.add_node("constant", token.lower())
        return pos + 1, node_id

    if token == "∅":
        node_id = g.add_node("constant", "∅")
        return pos + 1, node_id

    if token == "U":
        node_id = g.add_node("constant", "U")
        return pos + 1, node_id

    if token[0].isalpha() or token[0] == '_':
        node_id = g.add_node("variable", token)
        return pos + 1, node_id

    return pos + 1, g.add_node("error", f"unexpected: {token}")


# ═══════════════════════════════════════════════════════════════
#  Example Problems
# ═══════════════════════════════════════════════════════════════

EXAMPLE_PROBLEMS = {
    "x+0": "x + 0",
    "x*1": "x * 1",
    "const_fold": "3 + 4",
    "complex_simplify": "(x + 0) * 1 + (3 + 4)",
    "nested": "((x + 0) * 1) + ((y * 1) + 0)",
    "mul_zero": "x * 0 + y",
    "redundant": "(2 + 3) * (4 + 5) + 0",
    "double_neg": "-(-x) + 0",
}


def graph_to_expr(graph: Graph) -> str:
    """
    Reconstruct a human-readable expression string from a Graph.

    Finds the root (node with no incoming edges), then renders the tree
    recursively.  Returns e.g. "4", "x", "x = 2", "(a + b) * c".
    """
    if not graph.nodes:
        return "?"

    # Find root: node that is never the *target* of any edge
    targeted = {e.target for e in graph.edges}
    roots = [n.id for n in graph.nodes if n.id not in targeted]
    if not roots:
        # Fallback: pick operator with most outgoing edges
        roots = [max(graph.nodes, key=lambda n: len(graph.outgoing(n.id))).id]
    root_id = roots[0]

    def _render(nid: int, depth: int = 0) -> str:
        if depth > 40:
            return "..."
        node = graph.get_node(nid)
        if node is None:
            return "?"

        if node.type in ("constant", "variable", "identifier"):
            return node.label or "?"

        if node.type == "not":
            children = graph.outgoing(nid)
            if children:
                inner = _render(children[0].target, depth + 1)
                return f"not {inner}"
            return "not ?"

        if node.type == "function":
            fn_name = node.label or "f"
            out_edges = graph.outgoing(nid)
            # argument edges
            args = [e for e in out_edges if e.relationship_type in ("argument", "operand", "left_operand")]
            if not args:
                args = out_edges
            arg_strs = [_render(e.target, depth + 1) for e in args]
            return f"{fn_name}({', '.join(arg_strs)})"

        if node.type == "operator":
            op = node.label or "?"
            out_edges = graph.outgoing(nid)
            edge_map = {e.relationship_type: e.target for e in out_edges}

            # Unary operators
            if op in ("not", "neg", "~"):
                child_id = (edge_map.get("operand") or edge_map.get("left_operand")
                            or (out_edges[0].target if out_edges else None))
                if child_id:
                    return f"-{_render(child_id, depth + 1)}"
                return f"{op}?"

            left_id = edge_map.get("left_operand")
            right_id = edge_map.get("right_operand")
            base_id = edge_map.get("base")
            exp_id = edge_map.get("exponent")

            # Power / exponent
            if op in ("^", "**", "pow") or (base_id and exp_id):
                b = _render(base_id or left_id, depth + 1) if (base_id or left_id) else "?"
                e = _render(exp_id or right_id, depth + 1) if (exp_id or right_id) else "?"
                # If base is compound, wrap in parens
                if base_id and graph.get_node(base_id) and graph.get_node(base_id).type == "operator":
                    b = f"({b})"
                return f"{b}^{e}"

            if left_id is None and right_id is None:
                # Try positional
                if len(out_edges) >= 2:
                    left_id = out_edges[0].target
                    right_id = out_edges[1].target
                elif len(out_edges) == 1:
                    return f"{op}({_render(out_edges[0].target, depth + 1)})"
                else:
                    return op

            left_str = _render(left_id, depth + 1) if left_id else "?"
            right_str = _render(right_id, depth + 1) if right_id else "?"

            # Wrap sub-expression in parens if needed to preserve precedence
            _prec = {"=": 0, "+": 1, "-": 1, "*": 2, "/": 2, "^": 3, "**": 3}
            my_prec = _prec.get(op, 2)
            if left_id:
                left_node = graph.get_node(left_id)
                if left_node and left_node.type == "operator":
                    lp = _prec.get(left_node.label, 2)
                    if lp < my_prec:
                        left_str = f"({left_str})"
            if right_id:
                right_node = graph.get_node(right_id)
                if right_node and right_node.type == "operator":
                    rp = _prec.get(right_node.label, 2)
                    if rp <= my_prec and op in ("*", "/", "-"):
                        right_str = f"({right_str})"

            # Clean up float formatting
            if op == "=":
                return f"{left_str} = {right_str}"
            return f"{left_str} {op} {right_str}"

        # Fallback
        return node.label or node.type or "?"

    result = _render(root_id)
    # Post-process: clean up floats like "2.0" → "2"
    import re as _re2
    result = _re2.sub(r'\b(\d+)\.0\b', r'\1', result)
    return result


def graph_to_answer(original_expr: str, graph: Graph) -> str:
    """
    Return a concise human-readable answer, e.g.:
      "2 + 2"  →  "= 4"
      "x^2 + 5 = 9"  →  "x = 2"
      "(x + 0) * 1"  →  "= x"
    """
    expr_str = graph_to_expr(graph)
    # If the result contains "=" it's an equation answer
    if "=" in expr_str:
        return expr_str
    # Otherwise format as "= <result>"
    orig_stripped = original_expr.strip()
    if orig_stripped == expr_str:
        return f"= {expr_str} (already simplified)"
    return f"= {expr_str}"


def load_problem(name_or_expr: str) -> Tuple[str, Graph]:
    """Load a named example or parse an expression."""
    if name_or_expr in EXAMPLE_PROBLEMS:
        expr = EXAMPLE_PROBLEMS[name_or_expr]
        return expr, build_expression_graph(expr)
    else:
        return name_or_expr, build_expression_graph(name_or_expr)


def load_problem_from_file(path: str) -> Tuple[str, Graph]:
    """Load a problem from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    if "expression" in data:
        expr = data["expression"]
        return expr, build_expression_graph(expr)
    elif "graph" in data:
        g = Graph.from_dict(data["graph"])
        return data.get("name", path), g
    else:
        raise ValueError(f"Invalid problem file: must have 'expression' or 'graph' key")
