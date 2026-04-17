"""
Chemistry domain transforms for SARE-HX.

These transforms operate on graphs representing chemical equations,
stoichiometric expressions, and physical chemistry formulas.

All transforms follow the SARE Transform contract:
  ``match(graph) → List[dict]``     — return context dicts with matched node IDs
  ``apply(graph, context) → (Graph, float)`` — return (new_graph, delta_energy)

Positive delta = energy reduction = improvement.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from sare.engine import Graph, Node, Transform


# ──────────────────────────────────────────────────────────────────────────────
#  Private helpers
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


def _eq_operands(graph: Graph, eq_id: int):
    """Return (lhs_node, rhs_node) for an '=' operator node, or (None, None)."""
    edges = graph.outgoing(eq_id)
    edge_map = {e.relationship_type: e.target for e in edges}
    lhs = graph.get_node(edge_map.get("left_operand"))
    rhs = graph.get_node(edge_map.get("right_operand"))
    return lhs, rhs


def _redirect_parents(g: Graph, old_id: int, new_id: int) -> None:
    """Redirect all incoming edges of *old_id* to point to *new_id* instead."""
    to_remove: List[int] = []
    to_add: List[Tuple[int, int, str]] = []
    for edge in g.edges:
        if edge.target == old_id:
            to_remove.append(edge.id)
            to_add.append((edge.source, new_id, edge.relationship_type))
    for eid in to_remove:
        g.remove_edge(eid)
    for src, tgt, rel in to_add:
        g.add_edge(src, tgt, rel)


# ──────────────────────────────────────────────────────────────────────────────
#  Transforms
# ──────────────────────────────────────────────────────────────────────────────

class IdealGasLaw(Transform):
    """P * V = n * R * T — recognize ideal gas expressions."""

    def name(self) -> str:
        return "chemistry_ideal_gas_pv_nrt"

    def match(self, graph: Graph) -> List[dict]:
        for n in graph.nodes:
            if n.type == "operator" and n.label == "=":
                lhs, rhs = _eq_operands(graph, n.id)
                if lhs and rhs and lhs.type == "operator" and rhs.type == "operator":
                    return [{"eq_id": n.id}]
        return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        node = g.get_node(context["eq_id"])
        if node is not None:
            node.value = "recognized:ideal_gas"
        return g, -1.5


class ChemicalReactionStoichiometry(Transform):
    """Recognize stoichiometric reaction patterns: coeff * formula + formula."""

    def name(self) -> str:
        return "chemistry_stoich_reaction"

    def match(self, graph: Graph) -> List[dict]:
        results = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "+":
                edges = graph.outgoing(n.id)
                edge_map = {e.relationship_type: e.target for e in edges}
                left = graph.get_node(edge_map.get("left_operand"))
                right = graph.get_node(edge_map.get("right_operand"))
                if left and right:
                    has_coeff = (
                        (left.type == "operator" and left.label == "*") or
                        (right.type == "operator" and right.label == "*")
                    )
                    has_vars = (
                        left.type in ("variable", "constant") or
                        right.type in ("variable", "constant")
                    )
                    if has_coeff or has_vars:
                        results.append({"plus_id": n.id})
        return results

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        plus_id = context["plus_id"]
        plus_node = g.get_node(plus_id)
        if plus_node is None:
            return g, 0.0
        # Remove direct children (reactant subtrees) to reduce complexity
        for edge in list(g.outgoing(plus_id)):
            child = g.get_node(edge.target)
            if child is not None:
                try:
                    g.remove_node(child.id)
                except Exception:
                    pass
        # Replace + with a constant "reaction" node
        plus_node.type = "constant"
        plus_node.label = "reaction"
        return g, -3.0


class MassBalance(Transform):
    """Recognize mass conservation equations: variable = variable."""

    def name(self) -> str:
        return "chemistry_mass_balance"

    def match(self, graph: Graph) -> List[dict]:
        for n in graph.nodes:
            if n.type == "operator" and n.label == "=":
                lhs, rhs = _eq_operands(graph, n.id)
                if lhs and rhs and lhs.type == "variable" and rhs.type == "variable":
                    return [{"eq_id": n.id, "lhs_id": lhs.id, "rhs_id": rhs.id}]
        return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        lhs_id = context.get("lhs_id")
        rhs_id = context.get("rhs_id")
        eq_id = context["eq_id"]
        # Remove lhs/rhs variable nodes, replace = with True constant
        for nid in (lhs_id, rhs_id):
            if nid is not None:
                try:
                    g.remove_node(nid)
                except Exception:
                    pass
        eq_node = g.get_node(eq_id)
        if eq_node is not None:
            eq_node.type = "constant"
            eq_node.label = "True"
        return g, -2.0


class StoichiometryCoefficients(Transform):
    """1 * formula → formula — remove unit stoichiometric coefficients."""

    def name(self) -> str:
        return "chemistry_stoich_coeff_one"

    def match(self, graph: Graph) -> List[dict]:
        results = []
        for n in graph.nodes:
            if n.type == "operator" and n.label == "*":
                edges = graph.outgoing(n.id)
                edge_map = {e.relationship_type: e.target for e in edges}
                left = graph.get_node(edge_map.get("left_operand"))
                if left and left.type == "constant":
                    try:
                        if float(left.label) == 1.0:
                            results.append({
                                "mul_id": n.id,
                                "coeff_id": left.id,
                                "formula_id": edge_map.get("right_operand"),
                            })
                    except (ValueError, TypeError):
                        pass
        return results

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        mul_id = context["mul_id"]
        formula_id = context.get("formula_id")
        if formula_id is None:
            return g, 0.0
        _redirect_parents(g, mul_id, formula_id)
        g.remove_node(context["coeff_id"])
        g.remove_node(mul_id)
        return g, -2.0


class AvogadroConversion(Transform):
    """n * N_A — recognize Avogadro number usage."""

    _AVOGADRO_LABELS = frozenset(("N_A", "6.022e23", "6.02e23", "avogadro"))

    def name(self) -> str:
        return "chemistry_avogadro"

    def match(self, graph: Graph) -> List[dict]:
        for n in graph.nodes:
            if n.type in ("variable", "constant") and n.label in self._AVOGADRO_LABELS:
                for e in graph.incoming(n.id):
                    parent = graph.get_node(e.source)
                    if parent and parent.type == "operator" and parent.label == "*":
                        return [{"na_id": n.id, "mul_id": e.source}]
        return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        mul_id = context["mul_id"]
        na_id = context["na_id"]
        # Redirect parents of mul to a new "molecule_count" variable, removing N_A
        new_node_id = g.add_node("variable", "molecule_count")
        _redirect_parents(g, mul_id, new_node_id)
        try:
            g.remove_node(na_id)
        except Exception:
            pass
        try:
            g.remove_node(mul_id)
        except Exception:
            pass
        return g, -3.0


class ConservationOfMass(Transform):
    """Recognize balanced equations: A + B = C + D pattern."""

    def name(self) -> str:
        return "chemistry_conservation_mass"

    def match(self, graph: Graph) -> List[dict]:
        for n in graph.nodes:
            if n.type == "operator" and n.label == "=":
                lhs, rhs = _eq_operands(graph, n.id)
                if (
                    lhs and rhs
                    and lhs.type == "operator" and lhs.label == "+"
                    and rhs.type == "operator" and rhs.label == "+"
                ):
                    return [{"eq_id": n.id}]
        return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        node = g.get_node(context["eq_id"])
        if node is not None:
            node.value = "recognized:conservation_mass"
        return g, -1.5


class MolarMassEquation(Transform):
    """n = m / M — moles = mass / molar_mass recognition."""

    def name(self) -> str:
        return "chemistry_molar_mass"

    def match(self, graph: Graph) -> List[dict]:
        for n in graph.nodes:
            if n.type == "operator" and n.label == "=":
                lhs, rhs = _eq_operands(graph, n.id)
                if lhs and rhs and lhs.type == "variable" and lhs.label == "n":
                    if rhs.type == "operator" and rhs.label == "/":
                        return [{"eq_id": n.id, "lhs_id": lhs.id, "rhs_id": rhs.id}]
        return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        node = g.get_node(context.get("lhs_id") or context["eq_id"])
        if node is not None:
            node.value = "recognized:molar_mass"
        return g, -2.0


class PHDefinition(Transform):
    """pH = -log[H+] — pH definition recognition."""

    def name(self) -> str:
        return "chemistry_ph_definition"

    def match(self, graph: Graph) -> List[dict]:
        for n in graph.nodes:
            if n.type == "operator" and n.label == "=":
                lhs, rhs = _eq_operands(graph, n.id)
                if lhs and rhs and lhs.type == "variable" and lhs.label == "pH":
                    # rhs should be neg/negation operator or log
                    if rhs.type == "operator" and rhs.label in ("neg", "-", "log", "negate"):
                        return [{"eq_id": n.id, "lhs_id": lhs.id, "rhs_id": rhs.id}]
                    # or rhs is a variable with "log" in it
                    if rhs.type == "variable" and "log" in rhs.label.lower():
                        return [{"eq_id": n.id, "lhs_id": lhs.id, "rhs_id": rhs.id}]
        return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        node = g.get_node(context.get("lhs_id") or context["eq_id"])
        if node is not None:
            node.value = "recognized:ph_definition"
        return g, -2.0


class GibbsFreeEnergy(Transform):
    """delta_G = delta_H - T * delta_S — Gibbs free energy recognition."""

    def name(self) -> str:
        return "chemistry_gibbs"

    def match(self, graph: Graph) -> List[dict]:
        for n in graph.nodes:
            if n.type == "operator" and n.label == "=":
                lhs, rhs = _eq_operands(graph, n.id)
                if lhs and rhs and lhs.type == "variable" and lhs.label in ("delta_G", "G", "dG"):
                    if rhs.type == "operator" and rhs.label == "-":
                        return [{"eq_id": n.id, "lhs_id": lhs.id, "rhs_id": rhs.id}]
        return []

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        node = g.get_node(context.get("lhs_id") or context["eq_id"])
        if node is not None:
            node.value = "recognized:gibbs"
        return g, -2.0
