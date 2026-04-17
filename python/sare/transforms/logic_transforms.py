"""
Logic and commonsense inference transforms for SARE-HX.

These transforms operate on graphs produced by ``SentenceGraphBuilder``
(commonsense triples) and by direct logic graph construction.

All transforms follow the SARE Transform contract:
  ``match(graph) → List[dict]``     — return context dicts with matched node IDs
  ``apply(graph, context) → (Graph, float)`` — return (new_graph, delta_energy)

Positive delta = energy reduction = improvement.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Any

from sare.engine import Transform, Graph, Node


# ──────────────────────────────────────────────────────────────────────────────
#  Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_children(
    graph: Graph,
    node_id: int,
    rel_type: Optional[str] = None,
) -> List[Node]:
    """Return child nodes reachable from *node_id* via outgoing edges."""
    children: List[Node] = []
    for edge in graph.outgoing(node_id):
        if rel_type is not None and edge.relationship_type != rel_type:
            continue
        child = graph.get_node(edge.target)
        if child is not None:
            children.append(child)
    return children


def _get_parents(
    graph: Graph,
    node_id: int,
    rel_type: Optional[str] = None,
) -> List[Node]:
    """Return parent nodes that have edges pointing TO *node_id*."""
    parents: List[Node] = []
    for edge in graph.incoming(node_id):
        if rel_type is not None and edge.relationship_type != rel_type:
            continue
        parent = graph.get_node(edge.source)
        if parent is not None:
            parents.append(parent)
    return parents


def _edge_between(graph: Graph, src_id: int, tgt_id: int) -> bool:
    """Return True if any outgoing edge from *src_id* points to *tgt_id*."""
    return any(e.target == tgt_id for e in graph.outgoing(src_id))


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


def _remove_subtree(g: Graph, root_id: int) -> None:
    """Remove *root_id* and all nodes reachable from it (DFS)."""
    to_remove: List[int] = []
    stack = [root_id]
    visited: set = set()
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        to_remove.append(nid)
        for edge in g.outgoing(nid):
            stack.append(edge.target)
    for nid in to_remove:
        if g.get_node(nid) is not None:
            g.remove_node(nid)


# ──────────────────────────────────────────────────────────────────────────────
#  1. FillUnknownTransform
# ──────────────────────────────────────────────────────────────────────────────

class FillUnknownTransform(Transform):
    """
    Resolve ``?`` nodes in commonsense question graphs using a KB.

    Expected graph pattern (from SentenceGraphBuilder.build_question):

        concept_node(subject) ──[has_relation]──▶ relation_node(rel)
                                                          │[object]
                                                          ▼
                                                    unknown_node(?)

    The transform queries ``commonsense_kb.query(subject)`` and looks for a
    fact whose relation matches the relation_node label.  On success it
    replaces the ``?`` node with a concrete concept node.

    Parameters
    ----------
    commonsense_kb : object with ``query(concept: str) → List[dict]`` where
                     each dict has keys ``subject``, ``relation``, ``object``.
                     Pass ``None`` to create a no-op transform.
    """

    def __init__(self, commonsense_kb: Any = None) -> None:
        self._kb = commonsense_kb

    def name(self) -> str:
        return "fill_unknown"

    def match(self, graph: Graph) -> List[dict]:
        if self._kb is None:
            return []
        if not hasattr(self._kb, 'query'):
            return []
        matches = []
        for node in graph.nodes:
            if node.type != "unknown" or node.label != "?":
                continue
            try:
                # The unknown should be the object target of a relation_node.
                rel_parents = _get_parents(graph, node.id, "object")
                if not rel_parents:
                    continue
                rel_node = rel_parents[0]

                # The relation_node should be pointed to by a concept_node.
                concept_parents = _get_parents(graph, rel_node.id, "has_relation")
                if not concept_parents:
                    continue
                subj_node = concept_parents[0]

                matches.append({
                    "unknown_id": node.id,
                    "relation": rel_node.label,
                    "subject": subj_node.label,
                    "rel_node_id": rel_node.id,
                    "subj_node_id": subj_node.id,
                })
            except Exception:
                continue
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        if self._kb is None:
            return graph.clone(), 0.0
        g = graph.clone()
        try:
            unknown_id = context["unknown_id"]
            subject = context["subject"]
            relation = context["relation"]

            facts = self._kb.query(subject)
            answer = None
            for fact in facts:
                if fact.get("relation") == relation:
                    answer = fact.get("object")
                    break

            if answer is None:
                return graph.clone(), 0.0

            # Replace the unknown node with the resolved concept
            new_id = g.add_node("concept", answer)
            _redirect_parents(g, unknown_id, new_id)
            g.remove_node(unknown_id)
            return g, 10.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  2. ChainInferenceTransform
# ──────────────────────────────────────────────────────────────────────────────

class ChainInferenceTransform(Transform):
    """
    Extend inference chains: if A→rel1→B and B→rel2→C, and A is not yet
    connected to C, add a shortcut edge A──[rel_chain(rel1+rel2)]──▶C.

    Parameters
    ----------
    commonsense_kb : optional; not needed for pure graph chaining but accepted
                     for interface consistency.
    """

    def __init__(self, commonsense_kb: Any = None) -> None:
        self._kb = commonsense_kb

    def name(self) -> str:
        return "chain_inference"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        # Find node B that sits in the middle of a chain A→B→C
        for b_node in graph.nodes:
            try:
                incoming = graph.incoming(b_node.id)
                outgoing = graph.outgoing(b_node.id)
                if not incoming or not outgoing:
                    continue
                for in_edge in incoming:
                    a_node = graph.get_node(in_edge.source)
                    if a_node is None:
                        continue
                    for out_edge in outgoing:
                        c_node = graph.get_node(out_edge.target)
                        if c_node is None or c_node.id == a_node.id:
                            continue
                        # Only add shortcut if A→C does not already exist
                        if not _edge_between(graph, a_node.id, c_node.id):
                            chain_label = f"{in_edge.relationship_type}+{out_edge.relationship_type}"
                            matches.append({
                                "a_id": a_node.id,
                                "b_id": b_node.id,
                                "c_id": c_node.id,
                                "chain_label": chain_label,
                            })
            except Exception:
                continue
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        try:
            a_id = context["a_id"]
            c_id = context["c_id"]
            chain_label = context["chain_label"]

            # Guard: don't add duplicate edges
            if not _edge_between(g, a_id, c_id):
                g.add_edge(a_id, c_id, chain_label)
            return g, 3.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  3. ModusPonensTransform
# ──────────────────────────────────────────────────────────────────────────────

class ModusPonensTransform(Transform):
    """
    Modus Ponens: given A and (A → B), derive B.

    Graph encoding expected:
      - A concrete node (type != "unknown", type != "implies").
      - An "implies" node with edges:
          implies_node ──[antecedent]──▶ A_copy (same label as A)
          implies_node ──[consequent]──▶ B_node

    When both A and the antecedent match, B_node is added as a "derived" node
    with an incoming "derived_by_mp" edge from A.
    """

    def name(self) -> str:
        return "modus_ponens"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        # Collect all concrete (non-implies, non-unknown) node labels
        concrete_labels = {
            n.label
            for n in graph.nodes
            if n.type not in ("implies", "unknown") and n.label
        }

        for node in graph.nodes:
            if node.type != "implies" and node.label not in ("implies", "→", "=>"):
                continue
            try:
                ant_nodes = _get_children(graph, node.id, "antecedent")
                con_nodes = _get_children(graph, node.id, "consequent")
                if not ant_nodes or not con_nodes:
                    continue
                ant = ant_nodes[0]
                con = con_nodes[0]
                # Fire if the antecedent label is among known facts
                if ant.label in concrete_labels:
                    # Only fire if B is not already a concrete node
                    if con.label not in concrete_labels:
                        # Find the matching concrete node ID
                        a_id = next(
                            (n.id for n in graph.nodes if n.label == ant.label
                             and n.type not in ("implies", "unknown")),
                            ant.id,
                        )
                        matches.append({
                            "implies_id": node.id,
                            "a_id": a_id,
                            "antecedent_label": ant.label,
                            "consequent_id": con.id,
                            "consequent_label": con.label,
                        })
            except Exception:
                continue
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        try:
            a_id = context["a_id"]
            con_label = context["consequent_label"]

            # Add derived conclusion
            derived_id = g.add_node("concept", con_label, {"derived": "modus_ponens"})
            g.add_edge(a_id, derived_id, "derived_by_mp")
            return g, 5.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  4. DoubleNegRemoveTransform
# ──────────────────────────────────────────────────────────────────────────────

class DoubleNegRemoveTransform(Transform):
    """
    ``not (not X)`` → ``X``

    Match: "not" node whose single child is also a "not" node.
    Apply: replace the outer "not" with the grandchild (X).
    """

    def name(self) -> str:
        return "double_neg_remove"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for node in graph.nodes:
            if node.type != "operator" or node.label not in ("not", "neg"):
                continue
            try:
                children = _get_children(graph, node.id)
                if len(children) != 1:
                    continue
                inner = children[0]
                if inner.type == "operator" and inner.label in ("not", "neg"):
                    grandchildren = _get_children(graph, inner.id)
                    if grandchildren:
                        matches.append({
                            "outer_not_id": node.id,
                            "inner_not_id": inner.id,
                            "grandchild_id": grandchildren[0].id,
                        })
            except Exception:
                continue
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        try:
            outer_id = context["outer_not_id"]
            inner_id = context["inner_not_id"]
            gc_id = context["grandchild_id"]

            # Redirect parents of outer_not → grandchild
            _redirect_parents(g, outer_id, gc_id)
            g.remove_node(outer_id)
            g.remove_node(inner_id)
            return g, 4.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  5. ImpliesElimTransform
# ──────────────────────────────────────────────────────────────────────────────

class ImpliesElimTransform(Transform):
    """
    Materialise implications when both sides are concrete.

    Match: "implies" node (label "implies" / "→" / "=>") whose antecedent
    and consequent children are both concrete (not unknown/implies).
    Apply: add the consequent as a "derived_fact" node connected by
    "derived_by_implies" from the antecedent.

    This is weaker than ModusPonens — it simply materialises the consequent
    as a new fact node without requiring the antecedent to already be asserted.
    """

    def name(self) -> str:
        return "implies_elim"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for node in graph.nodes:
            is_implies = (
                node.label in ("implies", "→", "=>")
                or node.type == "implies"
            )
            if not is_implies:
                continue
            try:
                ant_nodes = _get_children(graph, node.id, "antecedent")
                con_nodes = _get_children(graph, node.id, "consequent")
                if not ant_nodes or not con_nodes:
                    continue
                ant = ant_nodes[0]
                con = con_nodes[0]
                # Both sides must be concrete
                if (
                    ant.type not in ("unknown", "implies")
                    and con.type not in ("unknown", "implies")
                    and ant.label
                    and con.label
                ):
                    # Avoid firing if the fact is already derived
                    already_derived = any(
                        n.label == con.label and n.type == "derived_fact"
                        for n in graph.nodes
                    )
                    if not already_derived:
                        matches.append({
                            "implies_id": node.id,
                            "ant_id": ant.id,
                            "ant_label": ant.label,
                            "con_id": con.id,
                            "con_label": con.label,
                        })
            except Exception:
                continue
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        try:
            ant_id = context["ant_id"]
            con_label = context["con_label"]

            derived_id = g.add_node(
                "derived_fact",
                con_label,
                {"derived": "implies_elim"},
            )
            g.add_edge(ant_id, derived_id, "derived_by_implies")
            return g, 3.0
        except Exception:
            return graph.clone(), 0.0


class TaxonomyChainTransform(Transform):
    """
    Reduces transitive IsA chains: A IsA B, B IsA C → A IsA C.

    Applies to graphs produced by SentenceGraphBuilder with relation='IsA'.
    Allows the solver to 'answer' commonsense taxonomy questions by collapsing
    intermediate hops in an IsA hierarchy.
    """

    def name(self) -> str:
        return "taxonomy_chain"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        try:
            isa_nodes = [n for n in graph.nodes if getattr(n, "label", "") == "IsA"]
            if len(isa_nodes) < 2:
                return []
            # Find A -[IsA]-> B -[IsA]-> C patterns via node adjacency
            for i, n1 in enumerate(isa_nodes):
                for n2 in isa_nodes[i + 1:]:
                    # Check if n2 is reachable from n1 via any edge
                    if any(e.target == n2.id for e in graph.outgoing(n1.id)):
                        matches.append({
                            "hop1_id": n1.id,
                            "hop2_id": n2.id,
                        })
        except Exception:
            pass
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        try:
            hop2_id = context["hop2_id"]
            # Remove one intermediate IsA hop to flatten the chain
            node = g.get_node(hop2_id)
            if node is not None:
                g.remove_node(hop2_id)
            return g, 2.0
        except Exception:
            return graph.clone(), 0.0
