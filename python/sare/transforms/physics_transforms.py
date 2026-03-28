"""
Physics-domain Transform subclasses for SARE-HX BeamSearch.

These transforms recognise classical physics equation patterns on the
expression graph and return a small negative energy delta so BeamSearch
can prefer states where a known physical law has been identified.

All transforms follow the standard SARE Transform contract:
  ``match(graph)  -> List[dict]``             — context dicts, empty = no match
  ``apply(graph, context) -> (Graph, float)`` — (new_graph, delta_energy)
                                                delta is negative (energy reduction)
"""

from __future__ import annotations

from typing import List, Tuple, Optional

from sare.engine import Transform, Graph, Node


# ──────────────────────────────────────────────────────────────────────────────
#  Private helpers  (mirrors the pattern in logic_transforms.py)
# ──────────────────────────────────────────────────────────────────────────────

def _children(graph: Graph, node_id: int,
              rel: Optional[str] = None) -> List[Node]:
    """Return child nodes reachable from *node_id* via outgoing edges."""
    out: List[Node] = []
    for edge in graph.outgoing(node_id):
        if rel is not None and edge.relationship_type != rel:
            continue
        child = graph.get_node(edge.target)
        if child is not None:
            out.append(child)
    return out


def _label(node: Node) -> str:
    return (node.label or "").strip()


def _is_op(node: Node, *ops: str) -> bool:
    return node.type == "operator" and _label(node) in ops


def _is_var(node: Node, *names: str) -> bool:
    return node.type == "variable" and _label(node) in names


def _is_const(node: Node) -> bool:
    return node.type == "constant"


def _eq_nodes(graph: Graph) -> List[Node]:
    """Return all '=' operator nodes in the graph."""
    return [n for n in graph.nodes
            if n.type == "operator" and _label(n) in ("=", "eq", "==")]


def _collect_mul_labels(graph: Graph, node: Node, depth: int = 1) -> set:
    """Collect labels from *node* and its immediate * sub-tree up to *depth* levels."""
    labels: set = {_label(node)}
    if _is_op(node, "*", "mul"):
        for k in _children(graph, node.id):
            labels.add(_label(k))
            if depth > 1 and _is_op(k, "*", "mul"):
                for kk in _children(graph, k.id):
                    labels.add(_label(kk))
    return labels


# ──────────────────────────────────────────────────────────────────────────────
#  1. Newton's Second Law  F = m * a
# ──────────────────────────────────────────────────────────────────────────────

class NewtonsSecondLaw(Transform):
    """
    Recognise  F = m * a  (Newton's second law).

    Match: '=' node whose left child is variable 'F' and right child is a
    '*' operator with children 'm' and 'a' (in either order).
    """

    def name(self) -> str:
        return "physics_newton_f_ma"

    def match(self, graph: Graph) -> List[dict]:
        try:
            matches: List[dict] = []
            for eq in _eq_nodes(graph):
                kids = _children(graph, eq.id)
                if len(kids) < 2:
                    continue
                lhs, rhs = kids[0], kids[1]
                if not _is_var(lhs, "F"):
                    continue
                if not _is_op(rhs, "*", "mul"):
                    continue
                rhs_kids = _children(graph, rhs.id)
                labels = {_label(k) for k in rhs_kids}
                if {"m", "a"}.issubset(labels):
                    matches.append({
                        "eq_id": eq.id,
                        "lhs_id": lhs.id,
                        "rhs_id": rhs.id,
                        "law": "newton_f_ma",
                    })
            return matches
        except Exception:
            return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        try:
            g = graph.clone()
            # Tag the equation node as recognised
            eq = g.get_node(context["eq_id"])
            if eq is not None and hasattr(eq, "attributes"):
                eq.attributes["physics_law"] = "newton_f_ma"
            return g, -1.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  2. Ohm's Law  V = I * R
# ──────────────────────────────────────────────────────────────────────────────

class OhmsLaw(Transform):
    """
    Recognise  V = I * R  (Ohm's law).

    Match: '=' node with LHS variable 'V' and RHS '*' node whose children
    include 'I' and 'R' (order-independent).
    """

    def name(self) -> str:
        return "physics_ohm_vir"

    def match(self, graph: Graph) -> List[dict]:
        try:
            matches: List[dict] = []
            for eq in _eq_nodes(graph):
                kids = _children(graph, eq.id)
                if len(kids) < 2:
                    continue
                lhs, rhs = kids[0], kids[1]
                if not _is_var(lhs, "V"):
                    continue
                if not _is_op(rhs, "*", "mul"):
                    continue
                labels = {_label(k) for k in _children(graph, rhs.id)}
                if {"I", "R"}.issubset(labels):
                    matches.append({
                        "eq_id": eq.id,
                        "lhs_id": lhs.id,
                        "rhs_id": rhs.id,
                        "law": "ohm_vir",
                    })
            return matches
        except Exception:
            return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        try:
            g = graph.clone()
            eq = g.get_node(context["eq_id"])
            if eq is not None and hasattr(eq, "attributes"):
                eq.attributes["physics_law"] = "ohm_vir"
            return g, -2.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  3. Kinematic Velocity  v = v0 + a * t
# ──────────────────────────────────────────────────────────────────────────────

class KinematicVelocity(Transform):
    """
    Recognise  v = v0 + a * t  (constant-acceleration kinematics).

    Match: '=' with LHS 'v' and RHS '+' node that contains:
      - a child that is variable 'v0' (or constant 0)
      - a child that is '*' node containing 'a' and 't'

    If v0 is the constant 0, simplify the graph to  v = a * t  (remove + 0).
    """

    def name(self) -> str:
        return "physics_kinematic_v"

    def match(self, graph: Graph) -> List[dict]:
        try:
            matches: List[dict] = []
            for eq in _eq_nodes(graph):
                kids = _children(graph, eq.id)
                if len(kids) < 2:
                    continue
                lhs, rhs = kids[0], kids[1]
                if not _is_var(lhs, "v"):
                    continue
                if not _is_op(rhs, "+", "add"):
                    continue
                plus_kids = _children(graph, rhs.id)
                v0_node = None
                mul_node = None
                for k in plus_kids:
                    if _is_op(k, "*", "mul"):
                        mul_node = k
                    elif k.type in ("variable", "constant"):
                        v0_node = k
                if mul_node is None:
                    continue
                mul_labels = {_label(c) for c in _children(graph, mul_node.id)}
                if not {"a", "t"}.issubset(mul_labels):
                    continue
                v0_is_zero = (
                    v0_node is not None
                    and v0_node.type == "constant"
                    and _label(v0_node) in ("0", "0.0")
                )
                matches.append({
                    "eq_id": eq.id,
                    "lhs_id": lhs.id,
                    "rhs_plus_id": rhs.id,
                    "v0_id": v0_node.id if v0_node else None,
                    "mul_id": mul_node.id,
                    "v0_is_zero": v0_is_zero,
                    "law": "kinematic_v",
                })
            return matches
        except Exception:
            return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        try:
            g = graph.clone()
            if context.get("v0_is_zero") and context.get("v0_id") is not None:
                # Simplify: remove the + 0 term; wire the mul directly to '='
                eq = g.get_node(context["eq_id"])
                mul_id = context["mul_id"]
                plus_id = context["rhs_plus_id"]
                v0_id = context["v0_id"]
                # Redirect edges from plus_id to mul_id in the parent (eq)
                for edge in g.edges:
                    if edge.source == context["eq_id"] and edge.target == plus_id:
                        g.remove_edge(edge.id)
                        g.add_edge(context["eq_id"], mul_id, edge.relationship_type)
                        break
                g.remove_node(plus_id)
                g.remove_node(v0_id)
                return g, -2.5
            else:
                eq = g.get_node(context["eq_id"])
                if eq is not None and hasattr(eq, "attributes"):
                    eq.attributes["physics_law"] = "kinematic_v"
                return g, -2.5
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  4. Kinematic Displacement  d = v0*t + 0.5*a*t²
# ──────────────────────────────────────────────────────────────────────────────

class KinematicDisplacement(Transform):
    """
    Recognise  d = v0*t + (1/2)*a*t²  (kinematic displacement).

    Match: '=' with LHS variable 'd' and RHS '+' node that has exactly two
    '*' (multiplication) children — a loose structural heuristic that fits
    the canonical form.
    """

    def name(self) -> str:
        return "physics_kinematic_d"

    def match(self, graph: Graph) -> List[dict]:
        try:
            matches: List[dict] = []
            for eq in _eq_nodes(graph):
                kids = _children(graph, eq.id)
                if len(kids) < 2:
                    continue
                lhs, rhs = kids[0], kids[1]
                if not _is_var(lhs, "d"):
                    continue
                if not _is_op(rhs, "+", "add"):
                    continue
                plus_kids = _children(graph, rhs.id)
                mul_kids = [k for k in plus_kids if _is_op(k, "*", "mul")]
                if len(mul_kids) >= 2:
                    matches.append({
                        "eq_id": eq.id,
                        "lhs_id": lhs.id,
                        "rhs_plus_id": rhs.id,
                        "law": "kinematic_d",
                    })
            return matches
        except Exception:
            return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        try:
            g = graph.clone()
            eq = g.get_node(context["eq_id"])
            if eq is not None and hasattr(eq, "attributes"):
                eq.attributes["physics_law"] = "kinematic_d"
            return g, -2.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  5. Kinetic Energy  KE = 0.5 * m * v²
# ──────────────────────────────────────────────────────────────────────────────

class EnergyKinetic(Transform):
    """
    Recognise  KE = (1/2) * m * v^2  (kinetic energy).

    Match: '=' with LHS variable 'KE' and RHS a '*' chain that mentions 'm'
    and 'v' (with an optional '^' for v²).
    If m and v are numeric constants fold the expression.
    """

    def name(self) -> str:
        return "physics_energy_ke"

    def match(self, graph: Graph) -> List[dict]:
        try:
            matches: List[dict] = []
            for eq in _eq_nodes(graph):
                kids = _children(graph, eq.id)
                if len(kids) < 2:
                    continue
                lhs, rhs = kids[0], kids[1]
                if not _is_var(lhs, "KE"):
                    continue
                if not _is_op(rhs, "*", "mul"):
                    continue
                labels = _collect_mul_labels(graph, rhs, depth=2)
                has_m = "m" in labels
                has_v = "v" in labels
                if has_m and has_v:
                    matches.append({
                        "eq_id": eq.id,
                        "lhs_id": lhs.id,
                        "rhs_id": rhs.id,
                        "law": "energy_ke",
                    })
            return matches
        except Exception:
            return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        try:
            g = graph.clone()
            eq = g.get_node(context["eq_id"])
            if eq is not None and hasattr(eq, "attributes"):
                eq.attributes["physics_law"] = "energy_ke"
            return g, -3.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  6. Potential Energy  PE = m * g * h
# ──────────────────────────────────────────────────────────────────────────────

class EnergyPotential(Transform):
    """
    Recognise  PE = m * g * h  (gravitational potential energy).

    Match: '=' with LHS variable 'PE' and RHS '*' chain containing
    'm', 'g' (or constant 9.8/9.81), and 'h'.
    """

    _G_VALUES = {"g", "9.8", "9.81", "G"}

    def name(self) -> str:
        return "physics_energy_pe"

    def match(self, graph: Graph) -> List[dict]:
        try:
            matches: List[dict] = []
            for eq in _eq_nodes(graph):
                kids = _children(graph, eq.id)
                if len(kids) < 2:
                    continue
                lhs, rhs = kids[0], kids[1]
                if not _is_var(lhs, "PE"):
                    continue
                if not _is_op(rhs, "*", "mul"):
                    continue
                labels = _collect_mul_labels(graph, rhs, depth=2)
                has_m = "m" in labels
                has_g = bool(labels & self._G_VALUES)
                has_h = "h" in labels
                if has_m and has_g and has_h:
                    matches.append({
                        "eq_id": eq.id,
                        "lhs_id": lhs.id,
                        "rhs_id": rhs.id,
                        "law": "energy_pe",
                    })
            return matches
        except Exception:
            return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        try:
            g = graph.clone()
            eq = g.get_node(context["eq_id"])
            if eq is not None and hasattr(eq, "attributes"):
                eq.attributes["physics_law"] = "energy_pe"
            return g, -2.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  7. Ideal Gas Law  P * V = n * R * T
# ──────────────────────────────────────────────────────────────────────────────

class PVnRT(Transform):
    """
    Recognise  P * V = n * R * T  (ideal gas law).

    Match: '=' with LHS '*' node containing 'P' and 'V', and RHS '*' chain
    containing 'n', 'R' (or 'r'), and 'T'.
    """

    def name(self) -> str:
        return "physics_ideal_gas"

    def match(self, graph: Graph) -> List[dict]:
        try:
            matches: List[dict] = []
            for eq in _eq_nodes(graph):
                kids = _children(graph, eq.id)
                if len(kids) < 2:
                    continue
                lhs, rhs = kids[0], kids[1]
                # LHS must be P*V
                lhs_labels = _collect_mul_labels(graph, lhs, depth=2)
                if not ({"P", "V"}.issubset(lhs_labels) and _is_op(lhs, "*", "mul")):
                    continue
                # RHS must contain n, R (or r), T
                rhs_labels = _collect_mul_labels(graph, rhs, depth=2)
                has_n = "n" in rhs_labels
                has_R = bool(rhs_labels & {"R", "r"})
                has_T = "T" in rhs_labels
                if has_n and has_R and has_T:
                    matches.append({
                        "eq_id": eq.id,
                        "lhs_id": lhs.id,
                        "rhs_id": rhs.id,
                        "law": "ideal_gas_pvnrt",
                    })
            return matches
        except Exception:
            return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        try:
            g = graph.clone()
            eq = g.get_node(context["eq_id"])
            if eq is not None and hasattr(eq, "attributes"):
                eq.attributes["physics_law"] = "ideal_gas_pvnrt"
            return g, -2.0
        except Exception:
            return graph.clone(), 0.0


def _eq_operands(graph: Graph, eq_id: int):
    """Return (lhs_node, rhs_node) for an '=' operator node, or (None, None)."""
    kids = _children(graph, eq_id)
    if len(kids) >= 2:
        return kids[0], kids[1]
    return None, None


class KineticEnergy(Transform):
    """E = 0.5 * m * v^2 — kinetic energy recognition."""

    def name(self) -> str:
        return "physics_kinetic_energy"

    def match(self, graph: Graph) -> List[dict]:
        for n in graph.nodes:
            if n.type == "operator" and n.label == "=":
                lhs, rhs = _eq_operands(graph, n.id)
                if lhs and rhs and lhs.type == "variable" and lhs.label in ("E", "KE", "E_k"):
                    return [{"eq_id": n.id}]
        return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        return graph.clone(), -2.0


class GravitationalForce(Transform):
    """F_g = G * m1 * m2 / r^2 — gravitational force recognition."""

    def name(self) -> str:
        return "physics_gravitational"

    def match(self, graph: Graph) -> List[dict]:
        for n in graph.nodes:
            if n.type == "variable" and n.label in ("G", "F_g", "F_gravity"):
                return [{"node_id": n.id}]
        return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        return graph.clone(), -2.0


class Momentum(Transform):
    """p = m * v — momentum recognition."""

    def name(self) -> str:
        return "physics_momentum"

    def match(self, graph: Graph) -> List[dict]:
        for n in graph.nodes:
            if n.type == "operator" and n.label == "=":
                lhs, rhs = _eq_operands(graph, n.id)
                if lhs and rhs and lhs.type == "variable" and lhs.label in ("p", "momentum"):
                    return [{"eq_id": n.id}]
        return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        return graph.clone(), -2.0


class WorkEnergyTransfer(Transform):
    """W = F * d — work recognition."""

    def name(self) -> str:
        return "physics_work"

    def match(self, graph: Graph) -> List[dict]:
        for n in graph.nodes:
            if n.type == "operator" and n.label == "=":
                lhs, rhs = _eq_operands(graph, n.id)
                if lhs and rhs and lhs.type == "variable" and lhs.label in ("W", "Work", "w"):
                    return [{"eq_id": n.id}]
        return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        return graph.clone(), -2.0
