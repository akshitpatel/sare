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
        for n in data.get("nodes", []):
            nid = g.add_node(n["type"], n.get("label", ""),
                              n.get("attributes", {}))
        for e in data.get("edges", []):
            g.add_edge(e["source"], e["target"], e["type"])
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

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "syntax": 1.0,
            "complexity": 0.5,
            "redundancy": 0.8,
            "uncertainty": 0.2,
        }

    def compute(self, graph: Graph) -> EnergyBreakdown:
        breakdown = EnergyBreakdown()

        # Syntax energy: penalize error nodes, empty types
        syntax = 0.0
        for n in graph.nodes:
            if n.type == "error" or not n.type:
                syntax += 5.0
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

        return breakdown


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
        for n in graph.nodes:
            if n.type == "operator" and n.label in ("neg", "-") and len(graph.outgoing(n.id)) == 1:
                child_edge = graph.outgoing(n.id)[0]
                child = graph.get_node(child_edge.target)
                if child and child.type == "operator" and child.label in ("neg", "-") and len(graph.outgoing(child.id)) == 1:
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

        for e in g.edges:
            if e.target == outer:
                g.remove_edge(e.id)
                g.add_edge(e.source, target, e.relationship_type)

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


def _base_transforms() -> List[Transform]:
    return [
        AddZeroElimination(),
        MulOneElimination(),
        ConstantFolding(),
        DoubleNegation(),
        MulZeroElimination(),
    ]


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
                   concept_registry=None) -> List[Transform]:
    """
    Build the full transform list for search.

    Args:
        include_macros:    Include mined macro-transforms.
        concept_registry:  Optional ConceptRegistry (C++ or Python).
                           If provided, learned rules are injected as
                           ConceptRule transforms (TODO-06).
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

    if not include_macros:
        return concept_rules + base
    return concept_rules + _macro_transforms(base) + base


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
    """Deterministic energy-minimizing beam search."""

    def search(self, graph: Graph, energy: EnergyEvaluator,
               transforms: List[Transform],
               beam_width: int = 8, max_depth: int = 50,
               budget_seconds: float = 30.0,
               kappa: float = 0.1,
               heuristic_fn: Optional[Callable[[Graph], float]] = None) -> SearchResult:

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

        for depth in range(max_depth):
            if time.time() - start_time > budget_seconds:
                break

            candidates = []
            for g, e, trace, _ in beam:
                for transform in transforms:
                    matches = transform.match(g)
                    for ctx in matches[:3]:  # limit matches per transform
                        new_g, delta_est = transform.apply(g, ctx)
                        new_e = energy.compute(new_g)
                        new_score = score(new_g, new_e)
                        candidates.append(
                            (new_g, new_e, trace + [transform.name()], new_score)
                        )
                        expansions += 1

            if not candidates:
                break

            # Keep top-k by score, where lower is better.
            candidates.sort(key=lambda x: x[3])
            beam = candidates[:beam_width]

            if beam[0][3] < best[3]:
                best = beam[0]

            trajectory.append(best[1].total)

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


def load_heuristic_scorer(model_path: Optional[str] = None) -> Optional[Callable[[Graph], float]]:
    if torch is None:
        return None

    from sare.heuristics.heuristic_model import HeuristicModel

    resolved = Path(model_path) if model_path else (Path(__file__).resolve().parents[2] / "models" / "heuristic_v1.pt")
    key = str(resolved)
    if key in _HEURISTIC_CACHE:
        return _HEURISTIC_CACHE[key]

    if not resolved.exists():
        _HEURISTIC_CACHE[key] = None
        return None

    model = HeuristicModel()
    state = torch.load(resolved, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
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

def build_expression_graph(expr_str: str) -> Graph:
    """
    Parse simple mathematical expressions into SARE graphs.
    Supports: x, y, z (variables), numbers, +, -, *, /
    """
    expr_str = expr_str.strip()
    g = Graph()

    # Simple recursive descent parser
    tokens = _tokenize(expr_str)
    if not tokens:
        g.add_node("error", "empty_expression")
        return g

    _, root_id = _parse_expression(g, tokens, 0)
    return g


def _tokenize(s: str) -> List[str]:
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "+-*/()":
            tokens.append(c)
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
    """Returns (new_pos, node_id)"""
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
    pos, left_id = _parse_atom(g, tokens, pos)

    while pos < len(tokens) and tokens[pos] in ("*", "/"):
        op = tokens[pos]
        pos += 1
        pos, right_id = _parse_atom(g, tokens, pos)
        op_id = g.add_node("operator", op)
        g.add_edge(op_id, left_id, "left_operand")
        g.add_edge(op_id, right_id, "right_operand")
        left_id = op_id

    return pos, left_id


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

    if token == "-":
        pos += 1
        pos, child_id = _parse_atom(g, tokens, pos)
        neg_id = g.add_node("operator", "neg")
        g.add_edge(neg_id, child_id, "operand")
        return pos, neg_id

    if token[0].isdigit() or (token[0] == '.' and len(token) > 1):
        node_id = g.add_node("constant", token)
        return pos + 1, node_id

    if token[0].isalpha():
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
