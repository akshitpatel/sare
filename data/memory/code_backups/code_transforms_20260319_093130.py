"""
Code-domain graph transforms for SARE-HX.

Each transform targets a specific code-simplification pattern in graphs
produced by ``CodeGraphBuilder``.  All transforms are safe: ``apply()``
always returns a valid graph (cloned from the input on failure).

Node conventions (from CodeGraphBuilder):
  type  : "if" | "operator" | "value" | "var" | "assign" | "return" | "unknown" | "constant"
  label : "True" | "False" | "not" | "and" | "or" | "=" | "+" | "*" | …
  edges : relationship_type in {"condition", "then", "else",
                                  "left", "right", "lhs", "rhs",
                                  "value", "operand"}
"""

from __future__ import annotations

from typing import List, Tuple, Optional

from sare.engine import Transform, Graph, Node


# ──────────────────────────────────────────────────────────────────────────────
#  Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_children(
    graph: Graph,
    node_id: int,
    rel_type: Optional[str] = None,
) -> List[Node]:
    """
    Return child nodes reachable from *node_id* via outgoing edges.

    Parameters
    ----------
    graph    : Graph to query.
    node_id  : Source node.
    rel_type : If given, only edges with this relationship_type are followed.
    """
    children: List[Node] = []
    for edge in graph.outgoing(node_id):
        if rel_type is not None and edge.relationship_type != rel_type:
            continue
        child = graph.get_node(edge.target)
        if child is not None:
            children.append(child)
    return children


def _first_child(
    graph: Graph,
    node_id: int,
    rel_type: Optional[str] = None,
) -> Optional[Node]:
    """Convenience wrapper — return the first child or None."""
    children = _get_children(graph, node_id, rel_type)
    return children[0] if children else None


def _redirect_parents(g: Graph, old_id: int, new_id: int) -> None:
    """Redirect all edges pointing TO *old_id* so they point to *new_id*."""
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


def _replace_subtree_with_node(
    g: Graph,
    root_id: int,
    new_type: str,
    new_label: str,
) -> int:
    """
    Replace the entire subtree rooted at *root_id* with a single new node.

    Returns the ID of the replacement node.
    """
    new_id = g.add_node(new_type, new_label)
    _redirect_parents(g, root_id, new_id)
    _remove_subtree(g, root_id)
    return new_id


def _remove_subtree(g: Graph, node_id: int) -> None:
    """Remove *node_id* and all nodes reachable from it (DFS)."""
    # Collect all nodes in this subtree first (avoid modifying during traversal)
    to_remove: List[int] = []
    stack = [node_id]
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
#  1. IfTrueElimTransform
# ──────────────────────────────────────────────────────────────────────────────

class IfTrueElimTransform(Transform):
    """
    ``(x if True else y)`` → ``x``

    Match: if-node whose condition edge leads to a value node labelled "True".
    Apply: redirect parents of the if-node to the then-branch; remove the rest.
    """

    def name(self) -> str:
        return "if_true_elim"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for node in graph.nodes:
            if node.type != "if":
                continue
            try:
                cond = _first_child(graph, node.id, "condition")
                if cond and cond.type == "value" and cond.label == "True":
                    then_node = _first_child(graph, node.id, "then")
                    else_node = _first_child(graph, node.id, "else")
                    if then_node:
                        matches.append({
                            "if_id": node.id,
                            "cond_id": cond.id,
                            "then_id": then_node.id,
                            "else_id": else_node.id if else_node else None,
                        })
            except Exception:
                continue
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        try:
            if_id = context["if_id"]
            then_id = context["then_id"]
            else_id = context.get("else_id")

            _redirect_parents(g, if_id, then_id)

            # Remove if-node itself (edges already redirected)
            g.remove_node(if_id)

            # Remove else subtree if present
            if else_id and g.get_node(else_id) is not None:
                _remove_subtree(g, else_id)

            return g, 8.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  2. IfFalseElimTransform
# ──────────────────────────────────────────────────────────────────────────────

class IfFalseElimTransform(Transform):
    """
    ``(x if False else y)`` → ``y``

    Match: if-node whose condition leads to value node labelled "False".
    Apply: keep the else-branch, discard the then-branch.
    """

    def name(self) -> str:
        return "if_false_elim"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for node in graph.nodes:
            if node.type != "if":
                continue
            try:
                cond = _first_child(graph, node.id, "condition")
                if cond and cond.type == "value" and cond.label == "False":
                    then_node = _first_child(graph, node.id, "then")
                    else_node = _first_child(graph, node.id, "else")
                    if else_node:
                        matches.append({
                            "if_id": node.id,
                            "cond_id": cond.id,
                            "then_id": then_node.id if then_node else None,
                            "else_id": else_node.id,
                        })
            except Exception:
                continue
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        try:
            if_id = context["if_id"]
            else_id = context["else_id"]
            then_id = context.get("then_id")

            _redirect_parents(g, if_id, else_id)
            g.remove_node(if_id)

            if then_id and g.get_node(then_id) is not None:
                _remove_subtree(g, then_id)

            return g, 8.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  3. NotTrueTransform
# ──────────────────────────────────────────────────────────────────────────────

class NotTrueTransform(Transform):
    """
    ``not True`` → ``False``

    Match: operator node labelled "not"/"neg" with a child labelled "True".
    """

    def name(self) -> str:
        return "not_true"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for node in graph.nodes:
            if node.type != "operator" or node.label not in ("not", "neg"):
                continue
            try:
                child = _first_child(graph, node.id)
                if child and child.label == "True":
                    matches.append({"op_id": node.id, "child_id": child.id})
            except Exception:
                continue
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        try:
            op_id = context["op_id"]
            _replace_subtree_with_node(g, op_id, "value", "False")
            return g, 5.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  4. NotFalseTransform
# ──────────────────────────────────────────────────────────────────────────────

class NotFalseTransform(Transform):
    """
    ``not False`` → ``True``
    """

    def name(self) -> str:
        return "not_false"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for node in graph.nodes:
            if node.type != "operator" or node.label not in ("not", "neg"):
                continue
            try:
                child = _first_child(graph, node.id)
                if child and child.label == "False":
                    matches.append({"op_id": node.id, "child_id": child.id})
            except Exception:
                continue
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        try:
            op_id = context["op_id"]
            _replace_subtree_with_node(g, op_id, "value", "True")
            return g, 5.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  5. AndSelfTransform
# ──────────────────────────────────────────────────────────────────────────────

class AndSelfTransform(Transform):
    """
    ``x and x`` → ``x``

    Match: "and" operator node whose both children carry the same label.
    """

    def name(self) -> str:
        return "and_self"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for node in graph.nodes:
            if node.type != "operator" or node.label != "and":
                continue
            try:
                children = _get_children(graph, node.id)
                if len(children) == 2 and children[0].label == children[1].label:
                    matches.append({
                        "op_id": node.id,
                        "keep_id": children[0].id,
                        "drop_id": children[1].id,
                    })
            except Exception:
                continue
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        try:
            op_id = context["op_id"]
            keep_id = context["keep_id"]
            drop_id = context["drop_id"]

            _redirect_parents(g, op_id, keep_id)
            g.remove_node(op_id)
            if g.get_node(drop_id) is not None:
                g.remove_node(drop_id)
            return g, 4.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  6. OrSelfTransform
# ──────────────────────────────────────────────────────────────────────────────

class OrSelfTransform(Transform):
    """
    ``x or x`` → ``x``
    """

    def name(self) -> str:
        return "or_self"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for node in graph.nodes:
            if node.type != "operator" or node.label != "or":
                continue
            try:
                children = _get_children(graph, node.id)
                if len(children) == 2 and children[0].label == children[1].label:
                    matches.append({
                        "op_id": node.id,
                        "keep_id": children[0].id,
                        "drop_id": children[1].id,
                    })
            except Exception:
                continue
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        try:
            op_id = context["op_id"]
            keep_id = context["keep_id"]
            drop_id = context["drop_id"]

            _redirect_parents(g, op_id, keep_id)
            g.remove_node(op_id)
            if g.get_node(drop_id) is not None:
                g.remove_node(drop_id)
            return g, 4.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  7. SelfAssignElimTransform
# ──────────────────────────────────────────────────────────────────────────────

class SelfAssignElimTransform(Transform):
    """
    ``x = x`` → ``x``

    Match: "assign" or "=" node whose lhs and rhs edges point to nodes with
    identical labels.  Apply: remove the assign node, replace with the variable.
    """

    def name(self) -> str:
        return "self_assign_elim"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for node in graph.nodes:
            if node.type not in ("assign",) and node.label not in ("=", "assign"):
                continue
            try:
                lhs_nodes = _get_children(graph, node.id, "lhs")
                rhs_nodes = _get_children(graph, node.id, "rhs")
                if not lhs_nodes or not rhs_nodes:
                    continue
                lhs = lhs_nodes[0]
                rhs = rhs_nodes[0]
                if lhs.label == rhs.label:
                    matches.append({
                        "assign_id": node.id,
                        "var_id": lhs.id,
                        "dup_id": rhs.id,
                    })
            except Exception:
                continue
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        try:
            assign_id = context["assign_id"]
            var_id = context["var_id"]
            dup_id = context["dup_id"]

            _redirect_parents(g, assign_id, var_id)
            g.remove_node(assign_id)
            # Remove the duplicate var node if it is separate
            if dup_id != var_id and g.get_node(dup_id) is not None:
                g.remove_node(dup_id)
            return g, 6.0
        except Exception:
            return graph.clone(), 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  8. ReturnConstantFoldTransform
# ──────────────────────────────────────────────────────────────────────────────

class ReturnConstantFoldTransform(Transform):
    """
    ``return <const> + <const>``  or  ``return <const> * <const>``  →
    ``return <folded_value>``

    Match: "return" node whose value child is a "+" or "*" operator with
    two constant children.
    """

    def name(self) -> str:
        return "return_const_fold"

    def match(self, graph: Graph) -> List[dict]:
        matches = []
        for node in graph.nodes:
            if node.type != "return":
                continue
            try:
                # value child of the return node
                val_children = _get_children(graph, node.id, "value")
                if not val_children:
                    # Also try unlabelled outgoing edge
                    val_children = _get_children(graph, node.id)
                if not val_children:
                    continue
                op_node = val_children[0]
                if op_node.type != "operator" or op_node.label not in ("+", "*"):
                    continue
                # The operator must have exactly two constant children
                op_children = _get_children(graph, op_node.id)
                if len(op_children) != 2:
                    continue
                a, b = op_children
                if a.type not in ("constant", "value") or b.type not in ("constant", "value"):
                    continue
                try:
                    float(a.label)
                    float(b.label)
                except ValueError:
                    continue
                matches.append({
                    "return_id": node.id,
                    "op_id": op_node.id,
                    "a_id": a.id,
                    "b_id": b.id,
                    "operator": op_node.label,
                    "a_val": a.label,
                    "b_val": b.label,
                })
            except Exception:
                continue
        return matches

    def apply(self, graph: Graph, context: dict) -> Tuple[Graph, float]:
        g = graph.clone()
        try:
            op_id = context["op_id"]
            operator = context["operator"]
            a_val = float(context["a_val"])
            b_val = float(context["b_val"])

            result = a_val + b_val if operator == "+" else a_val * b_val

            # Format the result cleanly (drop trailing .0 for integers)
            label = str(int(result)) if result == int(result) else str(result)

            _replace_subtree_with_node(g, op_id, "constant", label)
            return g, 3.0
        except Exception:
            return graph.clone(), 0.0
